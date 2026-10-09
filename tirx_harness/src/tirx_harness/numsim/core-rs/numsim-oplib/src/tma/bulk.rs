//! Pure operand/addressing rules of non-tensor bulk copies and shared TMA
//! helpers: multicast CTA masks, `.report` sampling, `st.bulk` and
//! `cp.async.bulk` byte counts, byte-masked store runs, `.ignore_oob`
//! windows, and cache-hint alignment checks.
//!
//! Legacy sources: `engine-rs/src/runtime/io.rs`
//! (`validate_tma_multicast_mask`), `engine-rs/src/runtime/memory_ops.rs`
//! (`copy_report_matches[_runs]`, `st_bulk_byte_count`, `bulk_byte_len`,
//! `raw_bulk_copy_layout`, `masked_destination_runs`,
//! `raw_bulk_copy_ignore_oob_lane_window`), and
//! `engine-rs/src/runtime/instructions/async_copy.rs`
//! (`tma_multicast_mask`, `validate_global_cache_hint_address`,
//! `validate_bulk_cache_hint_range`).

use crate::types::{OpError, OpResult};

/// Validate a `.multicast::cluster` CTA mask against the cluster size.
pub fn validate_tma_multicast_mask(mask: u64, ctas_per_cluster: usize) -> OpResult<()> {
    if mask == 0 {
        return Err(OpError::message(
            "copy_async multicast mask must select at least one CTA",
        ));
    }
    // Operand widths belong to the canonical PTX schema. The shared physical
    // routing path accepts the widest supported cluster mask.
    if mask > u32::MAX as u64 {
        return Err(OpError::message(format!(
            "copy_async multicast mask 0x{mask:x} exceeds the 32-bit PTX operand"
        )));
    }
    if ctas_per_cluster < u64::BITS as usize && (mask >> ctas_per_cluster) != 0 {
        return Err(OpError::message(format!(
            "copy_async multicast mask 0x{mask:x} names a CTA outside cluster size {ctas_per_cluster}"
        )));
    }
    Ok(())
}

/// Destination CTA ranks of a TMA load: the mask's CTAs when multicasting,
/// else just the issuing CTA.
pub fn multicast_target_ctas(
    multicast: bool,
    cta_mask: u64,
    ctas_per_cluster: usize,
    issuing_cta_in_cluster: usize,
) -> OpResult<Vec<usize>> {
    if !multicast {
        return Ok(vec![issuing_cta_in_cluster]);
    }
    validate_tma_multicast_mask(cta_mask, ctas_per_cluster)?;
    Ok((0..ctas_per_cluster)
        .filter(|target| cta_mask & (1_u64 << target) != 0)
        .collect())
}

/// Raw per-lane multicast operand: negative masks are rejected.
pub fn tma_multicast_mask(mask: Option<i64>) -> OpResult<u64> {
    mask.map_or(Ok(0), |mask| {
        u64::try_from(mask).map_err(|_| OpError::message("negative TMA multicast CTA mask"))
    })
}

/// PTX 9.4 permits one implementation-chosen sample per aligned 16-byte
/// chunk. NumSim chooses its first element (low nibble for FP4). `0xff`
/// instead inspects every byte; zero is the explicitly disabled mechanism.
pub fn copy_report_matches(pattern: u32, bytes: &[u8]) -> OpResult<bool> {
    copy_report_matches_runs(pattern, std::iter::once((0, bytes)))
}

/// Tensor copies can touch discontiguous parts of a source chunk. Choose the
/// lowest-addressed copied element of each chunk, independently of the tensor
/// walk, shared swizzle, padding, and OOB fill. Never inspect extra source bytes.
pub fn copy_report_matches_runs<'a>(
    pattern: u32,
    runs: impl IntoIterator<Item = (usize, &'a [u8])>,
) -> OpResult<bool> {
    let width = match pattern {
        0 => return Ok(false),
        0xff => return Ok(runs.into_iter().any(|(_, bytes)| bytes.contains(&0xff))),
        0x8 | 0x80 => 1,
        0x8000 => 2,
        0x80000000 => 4,
        _ => return Err(OpError::message("invalid bulk copy report pattern")),
    };
    let mut samples = std::collections::BTreeMap::new();
    for (offset, bytes) in runs {
        if !offset.is_multiple_of(width) || !bytes.len().is_multiple_of(width) {
            return Err(OpError::message(
                "bulk report sample is not a whole aligned element",
            ));
        }
        for (index, element) in bytes.chunks_exact(width).enumerate() {
            let address = offset
                .checked_add(index * width)
                .ok_or_else(|| OpError::message("bulk report source address overflow"))?;
            let mut bits = [0u8; 4];
            bits[..width].copy_from_slice(element);
            if pattern == 0x8 {
                bits[0] &= 0xf;
            }
            let sample = (address, u32::from_le_bytes(bits));
            let previous = samples.entry(address / 16).or_insert(sample);
            if address < previous.0 {
                *previous = sample;
            }
        }
    }
    Ok(samples.values().any(|&(_, value)| value == pattern))
}

/// `st.bulk` size operand: a multiple of 8 up to 16 MiB.
pub fn st_bulk_byte_count(value: i64, lane: usize) -> OpResult<usize> {
    let byte_len = usize::try_from(value)
        .map_err(|_| OpError::message(format!("st.bulk byte count is negative on lane {lane}")))?;
    if byte_len % 8 != 0 || byte_len > 16_777_216 {
        return Err(OpError::message(format!(
            "st.bulk byte count {byte_len} must be a multiple of 8 with maximum 16777216 on lane {lane}"
        )));
    }
    Ok(byte_len)
}

/// `cp.async.bulk` size operand of one lane: a positive multiple of 16.
/// `operation` prefixes the message (legacy `DiagnosticLabel`).
pub fn bulk_byte_len(value: i64, lane: usize, operation: &str) -> OpResult<usize> {
    let byte_len = usize::try_from(value).map_err(|_| {
        OpError::message(format!("{operation} byte count is negative on lane {lane}"))
    })?;
    if byte_len == 0 || byte_len % 16 != 0 {
        return Err(OpError::message(format!(
            "{operation} byte count {byte_len} must be a positive multiple of 16 on lane {lane}"
        )));
    }
    Ok(byte_len)
}

/// Both ends of a bulk copy must be 16-byte aligned (physical addresses).
pub fn validate_bulk_alignment(
    source_address: usize,
    destination_address: usize,
    lane: usize,
    operation: &str,
) -> OpResult<()> {
    if !source_address.is_multiple_of(16) || !destination_address.is_multiple_of(16) {
        return Err(OpError::message(format!(
            "{operation} requires 16-byte aligned source and destination addresses on lane {lane}"
        )));
    }
    Ok(())
}

/// Byte runs `(offset, len)` written by `cp.async.bulk...cp_mask`: the 16-bit
/// mask repeats every 16 bytes; adjacent selected bytes merge, so a full mask
/// yields one span.
pub fn masked_destination_runs(
    byte_mask: i64,
    lane: usize,
    destination_offset: usize,
    byte_len: usize,
) -> OpResult<Vec<(usize, usize)>> {
    let byte_mask = u16::try_from(byte_mask).map_err(|_| {
        OpError::message(format!(
            "cp.async.bulk.s2g byte mask is outside 16 bits on lane {lane}"
        ))
    })?;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for index in 0..byte_len {
        if byte_mask & (1_u16 << (index % 16)) == 0 {
            continue;
        }
        let start = destination_offset + index;
        match runs.last_mut() {
            Some((run_start, run_len)) if *run_start + *run_len == start => *run_len += 1,
            _ => runs.push((start, 1)),
        }
    }
    Ok(runs)
}

const IGNORE_OOB_LABEL: &str = "cp.async.bulk.g2s.cta.ignore_oob";

/// In-bounds window of a `.ignore_oob` global-to-shared copy: the whole
/// destination window is written; only `valid_start..valid_start+valid_len`
/// is read (the ignored bytes keep their prior contents).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IgnoreOobWindow {
    pub byte_len: usize,
    pub valid_start: usize,
    pub valid_len: usize,
}

/// Window of a `cp.async.bulk ... .ignore_oob` copy of `num_bytes`: bytes
/// `[left, len - right)` are read, the ignored edges keep prior contents. No numerics;
/// errors on negative/over-15 ignore counts, bad length or misaligned addresses.
pub fn ignore_oob_window(
    num_bytes: i64,
    ignore_bytes_left: i64,
    ignore_bytes_right: i64,
    source_address: usize,
    destination_address: usize,
    lane: usize,
) -> OpResult<IgnoreOobWindow> {
    let byte_len = bulk_byte_len(num_bytes, lane, IGNORE_OOB_LABEL)?;
    let ignore_left = usize::try_from(ignore_bytes_left).map_err(|_| {
        OpError::message(format!(
            "{IGNORE_OOB_LABEL} left byte count is negative on lane {lane}"
        ))
    })?;
    let ignore_right = usize::try_from(ignore_bytes_right).map_err(|_| {
        OpError::message(format!(
            "{IGNORE_OOB_LABEL} right byte count is negative on lane {lane}"
        ))
    })?;
    if ignore_left > 15 || ignore_right > 15 {
        return Err(OpError::message(format!(
            "{IGNORE_OOB_LABEL} ignored byte counts must be in 0..=15 on lane {lane}"
        )));
    }
    validate_bulk_alignment(source_address, destination_address, lane, IGNORE_OOB_LABEL)?;
    let valid_start = ignore_left.min(byte_len);
    let valid_end = byte_len.saturating_sub(ignore_right).max(valid_start);
    Ok(IgnoreOobWindow {
        byte_len,
        valid_start,
        valid_len: valid_end - valid_start,
    })
}

/// Address alignment of `prefetch`/`applypriority` style cache hints.
pub fn validate_cache_hint_alignment(
    physical_byte_offset: usize,
    alignment: usize,
    lane: usize,
    label: &str,
) -> OpResult<()> {
    if !physical_byte_offset.is_multiple_of(alignment) {
        return Err(OpError::message(format!(
            "{label} requires a {alignment}-byte aligned global address on lane {lane}, got byte offset {physical_byte_offset}"
        )));
    }
    Ok(())
}

/// Size operand of bulk cache hints: a multiple of 16 (the window itself is
/// not required to be valid memory).
pub fn validate_bulk_cache_hint_size(byte_len: u32, lane: usize, label: &str) -> OpResult<usize> {
    let byte_len = usize::try_from(byte_len)
        .map_err(|_| OpError::message(format!("{label} size exceeds usize on lane {lane}")))?;
    if byte_len % 16 != 0 {
        return Err(OpError::message(format!(
            "{label} size {byte_len} must be a multiple of 16 on lane {lane}"
        )));
    }
    Ok(byte_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multicast_masks_select_cluster_ctas() {
        assert_eq!(multicast_target_ctas(true, 0b11, 2, 0).unwrap(), vec![0, 1]);
        assert_eq!(multicast_target_ctas(false, 0, 2, 1).unwrap(), vec![1]);
        assert!(multicast_target_ctas(true, 0, 2, 0).is_err());
        assert!(validate_tma_multicast_mask(0b100, 2)
            .unwrap_err()
            .to_string()
            .contains("outside cluster size 2"));
        assert!(validate_tma_multicast_mask(1 << 32, 64).is_err());
        assert!(tma_multicast_mask(Some(-1)).is_err());
        assert_eq!(tma_multicast_mask(None).unwrap(), 0);
    }

    #[test]
    fn report_samples_lowest_element_of_each_16_byte_chunk() {
        let mut bytes = vec![0_u8; 32];
        bytes[16] = 0x80;
        assert!(copy_report_matches(0x80, &bytes).unwrap());
        bytes[16] = 0;
        bytes[17] = 0x80;
        assert!(!copy_report_matches(0x80, &bytes).unwrap());
        assert!(copy_report_matches(0xff, &[0, 0xff]).unwrap());
        bytes[0] = 0xf8;
        assert!(copy_report_matches(0x8, &bytes).unwrap());
        assert!(!copy_report_matches(0, &bytes).unwrap());
        assert!(copy_report_matches(0x7, &bytes).is_err());
        assert!(copy_report_matches_runs(0x8000, [(1_usize, &[0_u8, 0][..])]).is_err());
    }

    #[test]
    fn bulk_sizes_masks_and_windows() {
        assert_eq!(st_bulk_byte_count(16, 0).unwrap(), 16);
        assert!(st_bulk_byte_count(12, 0).is_err());
        assert!(st_bulk_byte_count(16_777_224, 0).is_err());
        assert!(bulk_byte_len(0, 3, "cp.async.bulk")
            .unwrap_err()
            .to_string()
            .contains("must be a positive multiple of 16 on lane 3"));
        assert_eq!(
            masked_destination_runs(0xffff, 0, 32, 32).unwrap(),
            vec![(32, 32)]
        );
        assert_eq!(
            masked_destination_runs(0b1001, 0, 0, 20).unwrap(),
            vec![(0, 1), (3, 1), (16, 1), (19, 1)]
        );
        assert!(masked_destination_runs(1 << 16, 0, 0, 16).is_err());
        let window = ignore_oob_window(32, 3, 15, 16, 0, 0).unwrap();
        assert_eq!(
            window,
            IgnoreOobWindow {
                byte_len: 32,
                valid_start: 3,
                valid_len: 14
            }
        );
        assert!(ignore_oob_window(32, 16, 0, 0, 0, 0).is_err());
        assert!(ignore_oob_window(32, 0, 0, 8, 0, 0).is_err());
    }

    #[test]
    fn address_cache_hints_enforce_alignment_and_bulk_size_only() {
        // async_copy.rs cache_hint_tests (adapted to the pure checks).
        validate_cache_hint_alignment(0, 128, 0, "bulk applypriority").unwrap();
        assert!(
            validate_cache_hint_alignment(16, 128, 0, "bulk applypriority")
                .unwrap_err()
                .to_string()
                .contains("128-byte aligned")
        );
        assert!(validate_bulk_cache_hint_size(12, 0, "bulk prefetch")
            .unwrap_err()
            .to_string()
            .contains("multiple of 16"));
        validate_bulk_cache_hint_size(128, 0, "bulk prefetch").unwrap();
        validate_bulk_cache_hint_size(0, 0, "bulk prefetch").unwrap();
    }
}
