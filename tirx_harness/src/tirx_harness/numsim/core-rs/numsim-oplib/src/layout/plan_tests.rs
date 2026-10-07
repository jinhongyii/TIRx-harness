//! Tests for `layout::plan` (no legacy test module covered these paths; the
//! cases encode the legacy behavior directly).

use super::*;
use crate::layout::element::ElementRef;

const WARP: TileWarp = TileWarp {
    active_mask: WarpMask::FULL,
    warp_id_in_cta: 0,
    warps_per_cta: 4,
};

fn row_major(columns: i64, itemsize: i64) -> impl Fn(&[i64], usize) -> OpResult<ElementRef> {
    move |c: &[i64], _lane: usize| Ok(ElementRef::byte(i128::from((c[0] * columns + c[1]) * itemsize)))
}

#[test]
fn async_copy_plan_assigns_warp_scope_owners() {
    let source = row_major(8, 4);
    let destination = |c: &[i64], _lane: usize| {
        Ok(ElementRef::byte(i128::from((c[1] * 4 + c[0]) * 4)))
    };
    let plan = mapped_copy_plan(WARP, &source, &destination, &[4, 8], TileScope::Warp, 4).unwrap();
    assert_eq!(plan.elements.len(), 32);
    assert_eq!(plan.destination_rank, None);
    let e = plan.elements[9];
    assert_eq!((e.source_lane, e.source_index, e.destination_index), (9, 9, 4 + 1));
    let remote = |_: &[i64], _: usize| Ok(ElementRef::in_bounds(ElementLocation::ByteOffset(0), Some(1)));
    assert!(mapped_copy_plan(WARP, &remote, &destination, &[1, 1], TileScope::Warp, 4)
        .unwrap_err()
        .to_string()
        .contains("cannot select a remote CTA"));
    let bad = |_: &[i64], _: usize| Err(OpError::message("boom"));
    assert_eq!(
        mapped_copy_plan(WARP, &bad, &destination, &[1, 1], TileScope::Warp, 4)
            .unwrap_err()
            .to_string(),
        "typed copy source map: boom"
    );
}

#[test]
fn sync_copy_plan_pairs_owner_lanes() {
    // Source: replicated shared tile. Destination: one owner lane per element
    // (lane = linear % 32), lane-private register slot 0.
    let source = row_major(8, 2);
    let destination = |c: &[i64], lane: usize| {
        let linear = (c[0] * 8 + c[1]) as usize;
        Ok(if linear % 32 == lane {
            ElementRef::byte(i128::from((linear / 32) as i64 * 2))
        } else {
            ElementRef::unowned(None)
        })
    };
    let plan =
        mapped_sync_copy_plan(WARP, &source, &destination, &[8, 8], TileScope::Warp, 2, true).unwrap();
    assert_eq!(plan.elements.len(), 64);
    for (linear, e) in plan.elements.iter().enumerate() {
        assert_eq!(e.source_lane, linear % 32);
        assert_eq!(e.destination_lane, linear % 32);
        assert_eq!(e.source_index, linear as i64);
        assert_eq!(e.destination_index, (linear / 32) as i64);
    }
    // Fully replicated on both sides: lane-private destinations copy on every
    // lane, shared destinations only on the scope owner.
    let both = mapped_sync_copy_plan(WARP, &source, &source, &[1, 2], TileScope::Warp, 2, true).unwrap();
    assert_eq!(both.elements.len(), 64);
    let both = mapped_sync_copy_plan(WARP, &source, &source, &[1, 2], TileScope::Warp, 2, false).unwrap();
    assert_eq!(both.elements.len(), 2);
    let nobody = |_: &[i64], _: usize| Ok(ElementRef::unowned(None));
    assert!(mapped_sync_copy_plan(WARP, &source, &nobody, &[1, 1], TileScope::Warp, 2, true)
        .unwrap_err()
        .to_string()
        .contains("has no source or destination owner"));
    assert!(
        mapped_sync_copy_plan(WARP, &source, &nobody, &[1, 1], TileScope::Cta, 2, true)
            .unwrap()
            .elements
            .is_empty()
    );
}

#[test]
fn gemm_maps_check_targets_and_unique_register_owners() {
    let tmem = |c: &[i64], _lane: usize| {
        Ok(ElementRef::in_bounds(
            ElementLocation::Tmem {
                mapped_lane: c[0],
                tcol_element: c[1] % 2,
                allocated_addr: 0,
                bit_offset: 0,
            },
            Some(1),
        ))
    };
    assert!(map_gemm_element(&tmem, &[0, 0], 0, Some(0), "scale")
        .unwrap_err()
        .to_string()
        .contains("selected CTA 1, but the instruction targets CTA 0"));
    let matrix = map_gemm_matrix(&tmem, 2, 3, true, 5, None, 4, "a").unwrap();
    assert_eq!(matrix.len(), 6);
    assert_eq!(matrix[1].target_cta, Some(1));
    assert_eq!(
        matrix[1].location,
        ElementLocation::Tmem {
            mapped_lane: 1,
            tcol_element: 0,
            allocated_addr: 0,
            bit_offset: 0
        }
    );
    // Scale selector: columns repeat every 2 cells -> redirected to selector.
    let scales = map_gemm_scale_matrix(&tmem, 1, 4, 0, None, 0, Some(10), 2, "s").unwrap();
    let tcols = scales
        .iter()
        .map(|e| match e.location {
            ElementLocation::Tmem { tcol_element, .. } => tcol_element,
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    assert_eq!(tcols, vec![10, 12, 10, 12]);

    let owner = |c: &[i64], lane: usize| {
        Ok(if (c[0] * 2 + c[1]) as usize == lane {
            ElementRef::byte(0)
        } else {
            ElementRef::out_of_bounds(None)
        })
    };
    let registers = map_register_gemm_matrix(&owner, WarpMask::FULL, 2, 2, false, "d").unwrap();
    assert_eq!(
        registers.iter().map(|e| e.execution_lane).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    let everyone = |_: &[i64], _: usize| Ok(ElementRef::byte(0));
    assert!(map_register_gemm_matrix(&everyone, WarpMask::FULL, 1, 1, false, "d")
        .unwrap_err()
        .to_string()
        .contains("more than one owning lane at (0, 0)"));
    assert!(map_register_gemm_matrix(&owner, WarpMask(1), 1, 2, false, "d")
        .unwrap_err()
        .to_string()
        .contains("no owning lane at (0, 1)"));
}

#[test]
fn tcgen_elements_follow_owner_masks_and_bounds() {
    struct Half;
    impl ElementMap for Half {
        fn owners(&self, logical: &[i64]) -> OpResult<WarpMask> {
            Ok(WarpMask(1 << (logical[0] as u32 % 32) | 1 << 31))
        }
        fn map(&self, logical: &[i64], lane: usize) -> OpResult<ElementRef> {
            Ok(if lane == 31 {
                ElementRef::out_of_bounds(None)
            } else {
                ElementRef::byte(i128::from(logical[0] * 4))
            })
        }
    }
    let elements = mapped_tcgen_elements(WarpMask(0xffff), &Half, &[40], "ld").unwrap();
    // Lanes 16..31 inactive; lane 31 out of bounds.
    assert_eq!(elements.len(), 16 + 8);
    assert!(elements.iter().all(|e| e.execution_lane < 16));
}

fn m64_elements(base_lane: i64, base_tcol: i64, warp: usize) -> (Vec<MappedElement>, Vec<MappedElement>) {
    let mut source = Vec::new();
    let mut destination = Vec::new();
    for lane in 0..WARP_SIZE {
        for slot in 0..32_usize {
            let tlane = base_lane + (warp % 4 * 32 + (slot % 4) / 2 * 8 + lane / 4) as i64;
            let tcol = base_tcol + (slot / 4 * 8 + lane % 4 * 2 + slot % 2) as i64;
            source.push(MappedElement {
                execution_lane: lane,
                target_cta: None,
                location: ElementLocation::Tmem {
                    mapped_lane: tlane,
                    tcol_element: tcol,
                    allocated_addr: 7,
                    bit_offset: 0,
                },
            });
            destination.push(MappedElement {
                execution_lane: lane,
                target_cta: None,
                location: ElementLocation::ByteOffset((slot * 4) as i128),
            });
        }
    }
    (source, destination)
}

#[test]
fn fast_m64_recognizer_accepts_only_the_canonical_slice() {
    let (source, destination) = m64_elements(64, 16, 1);
    assert_eq!(
        fast_tmem_f32_m64_load(WarpMask::FULL, 1, &source, &destination),
        Some(FastTmemF32M64Load {
            base_lane: 64,
            base_tcol: 16,
            allocated_addr: 7
        })
    );
    assert_eq!(fast_tmem_f32_m64_load(WarpMask(1), 1, &source, &destination), None);
    // Wrong warp -> lane offsets disagree.
    assert_eq!(
        fast_tmem_f32_m64_load(WarpMask::FULL, 0, &source, &destination).map(|b| b.base_lane),
        Some(96)
    );
    let mut swapped = destination.clone();
    swapped.swap(0, 1);
    assert_eq!(fast_tmem_f32_m64_load(WarpMask::FULL, 1, &source, &swapped), None);
}

#[test]
fn canonical_32x32b_rows_offset_tlanes_by_warp_quarter() {
    let rows = canonical_32x32b_tmem_rows(WarpMask(0b11), 6, 0, 0, 8, 3, 64, "ld").unwrap();
    assert_eq!(rows, vec![(0, 0, None, 64, 8, 3, 64), (0, 1, None, 65, 8, 3, 64)]);
    assert!(canonical_32x32b_tmem_rows(WarpMask(1), 1, 0, i64::MAX, 0, 0, 4, "ld")
        .unwrap_err()
        .to_string()
        .contains("ld TLane overflow"));
}
