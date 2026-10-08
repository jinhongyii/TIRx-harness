//! Translation cache for tiled TMA plans (perf, W2-21).
//!
//! A tile-mode box that lies wholly inside the tensor touches no OOB
//! element, and its global addresses are affine in the coordinates:
//! element `e` of the box is at `global_address + sum_i (c_i + e_i) *
//! stride_i` (`stride_0` = element bytes). Its plan is therefore the plan of
//! the same map at coordinates `0` (also interior: `box_i <= dim_i`) with
//! every global span shifted by `global_address + sum_i c_i * stride_i`;
//! the shared side (swizzle included) does not depend on the coordinates at
//! all. Plans are cached per (map, direction, shared offset), planned at
//! coordinates 0 with the map's own address (so every address check sees the
//! real map), and shifted on use. Everything else (OOB boxes, gather/scatter,
//! im2col, interleaved and sub-byte maps) is planned directly.

use super::super::{OpResult, TensorMapDesc, TmaPlan, TmaPlanDir};
use crate::arena::ByteSpan;
use crate::program::TmaMode;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Entries kept per thread before the cache is cleared.
const CAPACITY: usize = 4096;

type Key = (TensorMapDesc, TmaPlanDir, u64);

thread_local! {
    static PLANS: RefCell<HashMap<Key, Rc<TmaPlan>>> = RefCell::new(HashMap::new());
}

/// Byte offset of `coords` from the tensor base, when the box at `coords`
/// is interior and the map qualifies; `None` otherwise.
fn interior_shift(map: &TensorMapDesc, mode: TmaMode, coords: &[i64], im2col_offsets: &[i64]) -> Option<u64> {
    let rank = usize::from(map.rank);
    if mode != TmaMode::Tile
        || !im2col_offsets.is_empty()
        || map.im2col.is_some()
        || map.interleave != 0
        || coords.len() != rank
        || !(1..=5).contains(&rank)
    {
        return None;
    }
    let elem = map.elem?;
    let bits = elem.bits();
    if bits < 8 || bits % 8 != 0 {
        return None;
    }
    let mut shift: u64 = 0;
    for (i, &c) in coords.iter().enumerate() {
        let c = u64::try_from(c).ok()?;
        let extent = u64::from(map.box_dim[i]);
        if extent == 0 || c.checked_add(extent)? > map.global_dim[i] {
            return None;
        }
        let stride = if i == 0 { u64::from(bits / 8) } else { map.global_stride[i - 1] };
        shift = shift.checked_add(c.checked_mul(stride)?)?;
    }
    Some(shift)
}

pub(super) fn plan(
    map: &TensorMapDesc,
    dir: TmaPlanDir,
    mode: TmaMode,
    coords: &[i64],
    im2col_offsets: &[i64],
    smem_offset: u64,
) -> OpResult<TmaPlan> {
    let Some(shift) = interior_shift(map, mode, coords, im2col_offsets) else {
        return super::plan_uncached(map, dir, mode, coords, im2col_offsets, smem_offset);
    };
    let delta = shift;
    let key: Key = (map.clone(), dir, smem_offset);
    let cached = PLANS.with(|plans| plans.borrow().get(&key).cloned());
    let reference = match cached {
        Some(reference) => reference,
        None => {
            let zeros = vec![0_i64; coords.len()];
            let reference = Rc::new(super::plan_uncached(&key.0, dir, mode, &zeros, im2col_offsets, smem_offset)?);
            if !reference.global_bits.is_empty() || !reference.smem_oob_fill.is_empty() {
                return super::plan_uncached(map, dir, mode, coords, im2col_offsets, smem_offset);
            }
            PLANS.with(|plans| {
                let mut plans = plans.borrow_mut();
                if plans.len() >= CAPACITY {
                    plans.clear();
                }
                plans.insert(key, Rc::clone(&reference));
            });
            reference
        }
    };
    let mut global = Vec::with_capacity(reference.global.len());
    for span in &reference.global {
        let Some(start) = span.start.checked_add(delta) else {
            return super::plan_uncached(map, dir, mode, coords, im2col_offsets, smem_offset);
        };
        global.push(ByteSpan::new(start, span.len));
    }
    Ok(TmaPlan {
        global,
        smem: reference.smem.clone(),
        smem_oob_fill: Vec::new(),
        bytes: reference.bytes,
        fill: reference.fill,
        fill_pattern: reference.fill_pattern.clone(),
        tf32_round: reference.tf32_round,
        global_bits: Vec::new(),
    })
}
