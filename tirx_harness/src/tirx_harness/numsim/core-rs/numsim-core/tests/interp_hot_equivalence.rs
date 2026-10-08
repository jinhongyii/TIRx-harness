//! W13 hot-path rewrites are behaviour-preserving:
//!
//! * `read_special` computes all lanes at once and writes them in one masked
//!   pass: every per-lane special register is checked against the PTX
//!   definition for several block shapes (2-D/3-D, partial last warp), under
//!   a divergent mask (inactive lanes keep their value), and into a narrow
//!   destination (truncated to its width).
//! * `WarpState::spin_hash` is memoized: it always equals the from-scratch
//!   hash, across register writes, mask changes and repeated calls.
//! * Register files come from one zeroed allocation: every slot reads 0.

use numsim_core::dtype::{Dtype, Ty};
use numsim_core::interp::WarpState;
use numsim_core::observe::{CtaId, NoopObserver, WarpId};
use numsim_core::program::*;
use numsim_core::sched::{self, RunStatus};
use numsim_core::testutil::scenarios::{inputs, u32_buf};
use numsim_core::testutil::ProgramBuilder;
use numsim_core::value::WarpMask;

const PER_LANE: [SpecialReg; 10] = [
    SpecialReg::LaneId,
    SpecialReg::ThreadInCta,
    SpecialReg::Tid(Axis::X),
    SpecialReg::Tid(Axis::Y),
    SpecialReg::Tid(Axis::Z),
    SpecialReg::LaneMaskEq,
    SpecialReg::LaneMaskLt,
    SpecialReg::LaneMaskLe,
    SpecialReg::LaneMaskGt,
    SpecialReg::LaneMaskGe,
];

/// PTX value of a per-lane special register for linear thread `t`.
fn reference(s: SpecialReg, block: [u32; 3], t: u64) -> u64 {
    let (bx, by) = (block[0] as u64, block[1] as u64);
    let lane = t % 32;
    let mask = |f: fn(u64, u64) -> bool| (0..32u64).filter(|&i| f(i, lane)).fold(0u64, |m, i| m | 1 << i);
    match s {
        SpecialReg::LaneId => lane,
        SpecialReg::ThreadInCta => t,
        SpecialReg::Tid(Axis::X) => t % bx,
        SpecialReg::Tid(Axis::Y) => (t / bx) % by,
        SpecialReg::Tid(Axis::Z) => t / (bx * by),
        SpecialReg::LaneMaskEq => mask(|i, l| i == l),
        SpecialReg::LaneMaskLt => mask(|i, l| i < l),
        SpecialReg::LaneMaskLe => mask(|i, l| i <= l),
        SpecialReg::LaneMaskGt => mask(|i, l| i > l),
        SpecialReg::LaneMaskGe => mask(|i, l| i >= l),
        _ => unreachable!(),
    }
}

/// Every thread reads each register of `PER_LANE` (plus `%ntid.x`,
/// `%warpid`, `%activemask`) three ways: full mask into a u32, odd lanes only
/// (even lanes keep a sentinel), and into a u8 destination.
fn run_shape(block: [u32; 3]) {
    let threads = block[0] * block[1] * block[2];
    let mut regs = PER_LANE.to_vec();
    regs.extend([SpecialReg::NTid(Axis::X), SpecialReg::WarpInCta, SpecialReg::ActiveMask]);
    let nreg = regs.len() as u32;
    let mut b = ProgramBuilder::new("special_equiv", threads);
    b.block(block[0], block[1], block[2]);
    let out = b.global("out", Dtype::U32);
    let t = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let odd = b.reg(Ty::PRED);
    let bit = b.reg(Ty::U32);
    let v = b.reg(Ty::U32);
    let v8 = b.reg(Ty::U8);
    let w = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    b.read_special(t, SpecialReg::ThreadInCta);
    b.lane_id(lane);
    let k1 = b.k_u32(1);
    b.binary(BinOp::And, Ty::U32, bit, lane, k1);
    b.compare(CmpOp::Eq, Ty::U32, odd, bit, k1);
    let k3n = b.k_u32(3 * nreg);
    b.mul(Ty::U32, idx, t, k3n);
    let sentinel = b.k_u32(0xdead_beef);
    for (k, &s) in regs.iter().enumerate() {
        let k = k as u32;
        b.read_special(v, s);
        let o = b.k_u32(3 * k);
        b.add_u32(w, idx, o);
        b.st_u32(out, w, v);
        b.mov(v, sentinel);
        b.if_(odd);
        b.read_special(v, s);
        b.end_if();
        let o = b.k_u32(3 * k + 1);
        b.add_u32(w, idx, o);
        b.st_u32(out, w, v);
        b.read_special(v8, s);
        b.cast(Ty::U8, Ty::U32, v, v8);
        let o = b.k_u32(3 * k + 2);
        b.add_u32(w, idx, o);
        b.st_u32(out, w, v);
    }
    b.exit();
    let n = (threads * 3 * nreg) as usize;
    let o = sched::run_with_config(
        &b.build_module(),
        &inputs(vec![("out", u32_buf(vec![0; n]))]),
        &mut NoopObserver,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(o.status, RunStatus::Completed, "{block:?}: {:?}", o.status);
    let got: Vec<u32> = o.outputs.buffers["out"].0.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let warps = threads.div_ceil(32);
    for t in 0..threads as u64 {
        let warp = t / 32;
        let live = if warp + 1 == warps as u64 && threads % 32 != 0 { (1u64 << (threads % 32)) - 1 } else { 0xffff_ffff };
        let odd_lanes = live & 0xaaaa_aaaa;
        for (k, &s) in regs.iter().enumerate() {
            let (full, half) = match s {
                SpecialReg::NTid(_) => (block[0] as u64, block[0] as u64),
                SpecialReg::WarpInCta => (warp, warp),
                SpecialReg::ActiveMask => (live, odd_lanes),
                _ => (reference(s, block, t), reference(s, block, t)),
            };
            let base = (t as usize) * 3 * regs.len() + 3 * k;
            let ctx = format!("block {block:?} thread {t} {s:?}");
            assert_eq!(got[base] as u64, full & 0xffff_ffff, "{ctx} full mask");
            let want_half = if t % 2 == 1 { half & 0xffff_ffff } else { 0xdead_beef };
            assert_eq!(got[base + 1] as u64, want_half, "{ctx} odd lanes");
            assert_eq!(got[base + 2] as u64, full & 0xff, "{ctx} u8 destination");
        }
    }
}

#[test]
fn read_special_matches_ptx_definition() {
    for block in [[64, 4, 1], [33, 3, 2], [8, 8, 4], [1, 1, 1], [96, 1, 1], [5, 7, 3]] {
        run_shape(block);
    }
}

#[test]
fn spin_hash_memo_is_exact() {
    let mut w = WarpState::new(WarpId(0), CtaId(0), 0, 40, WarpMask::ALL);
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for step in 0..2000 {
        match next() % 6 {
            // Unchanged state: memo hit.
            0 | 1 => {}
            2 => {
                let s = (next() % 40) as u32;
                let l = (next() % 32) as usize;
                w.regs.get_mut(s)[l] = next();
            }
            // Write the same value back: still a hit.
            3 => {
                let s = (next() % 40) as u32;
                let l = (next() % 32) as usize;
                let v = w.regs.get(s)[l];
                w.regs.get_mut(s)[l] = v;
            }
            4 => w.active = WarpMask(next() as u32),
            _ => {
                // Flip one bit and flip it back over two calls.
                let s = (next() % 40) as u32;
                w.regs.get_mut(s)[3] ^= 1;
                assert_eq!(w.spin_hash(), w.spin_hash_uncached(), "step {step} flipped");
                w.regs.get_mut(s)[3] ^= 1;
            }
        }
        assert_eq!(w.spin_hash(), w.spin_hash_uncached(), "step {step}");
    }
}

#[test]
fn register_file_starts_zeroed() {
    for n in [0usize, 1, 7, 4096] {
        let w = WarpState::new(WarpId(1), CtaId(0), 1, n, WarpMask::ALL);
        assert_eq!(w.regs.regs.len(), n);
        assert!(w.regs.regs.iter().all(|s| s.iter().all(|&v| v == 0)));
    }
}
