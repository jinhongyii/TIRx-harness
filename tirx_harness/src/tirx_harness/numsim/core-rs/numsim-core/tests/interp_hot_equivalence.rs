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
        let live = if warp + 1 == warps as u64 && !threads.is_multiple_of(32) { (1u64 << (threads % 32)) - 1 } else { 0xffff_ffff };
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
                w.reg_mut(s)[l] = next();
            }
            // Write the same value back: still a hit.
            3 => {
                let s = (next() % 40) as u32;
                let l = (next() % 32) as usize;
                let v = w.regs.get(s)[l];
                w.reg_mut(s)[l] = v;
            }
            4 => w.active = WarpMask(next() as u32),
            _ => {
                // Flip one bit and flip it back over two calls.
                let s = (next() % 40) as u32;
                w.reg_mut(s)[3] ^= 1;
                assert_eq!(w.spin_hash(), w.spin_hash_uncached(), "step {step} flipped");
                w.reg_mut(s)[3] ^= 1;
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

/// `tcgen05.ld.32x32b.x64` with a distinct value per (thread, column),
/// loaded back twice (into fresh and into overwritten registers) through the
/// register-major fast path (chosen by data shape, with or without an
/// observer): the registers equal what was stored and the run does not
/// depend on whether anyone observes it. (Observer-stream identity with the
/// per-lane path is checked by the before/after digest of W13's report.)
#[test]
fn tcgen_ld_fast_path_matches_per_lane_path() {
    use numsim_core::observe::RecordingObserver;
    let mut b = ProgramBuilder::new("tcgen_ld_equiv", 128);
    let out = b.global("out", Dtype::U32);
    let slot = b.shared("taddr", Dtype::U32, 1);
    let tid = b.reg(Ty::U32);
    let w = b.reg(Ty::U32);
    let p = b.reg(Ty::PRED);
    let sa = b.reg(Ty::U32);
    let t = b.reg(Ty::U32);
    let t2 = b.reg(Ty::U32);
    let lanebits = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    let src: Vec<Reg> = (0..64).map(|_| b.reg(Ty::U32)).collect();
    let v: Vec<Reg> = (0..64).map(|_| b.reg(Ty::U32)).collect();
    let v2: Vec<Reg> = (0..64).map(|_| b.reg(Ty::U32)).collect();
    b.thread_rank(tid);
    b.warp_id(w);
    let k0 = b.k_u32(0);
    let k32 = b.k_u32(32);
    let k64 = b.k_u32(64);
    let k16 = b.k_u32(16);
    let k128 = b.k_u32(128);
    for (j, &r) in src.iter().enumerate() {
        let kj = b.k_u32(j as u32 * 7919 + 1);
        b.mul(Ty::U32, r, tid, k64);
        b.add_u32(r, r, kj);
    }
    let sentinel = b.k_u32(0xdead_beef);
    for &r in &v2 {
        b.mov(r, sentinel);
    }
    b.smem_addr(sa, slot, k0);
    b.compare(CmpOp::Eq, Ty::U32, p, w, k0);
    b.if_(p);
    b.push(Instr::TcgenAlloc { dst: sa.into(), ncols: k64, cta_group: 1, exclusive: false });
    b.end_if();
    b.bar_sync(0);
    b.ld_u32(t, slot, k0);
    b.mul(Ty::U32, lanebits, w, k32);
    b.binary(BinOp::Shl, Ty::U32, lanebits, lanebits, k16);
    b.add_u32(t2, t, lanebits);
    b.push(Instr::TcgenSt(Box::new(TcgenStArgs {
        srcs: src.iter().map(|&r| r.into()).collect(),
        taddr: t2.into(),
        row: k0,
        col: k0,
        shape: TcShape::S32x32b,
        num: 64,
        unpack: false,
    })));
    b.push(Instr::TcgenWait { st: true });
    let ld = |b: &mut ProgramBuilder, dsts: &[Reg]| {
        b.push(Instr::TcgenLd(Box::new(TcgenLdArgs {
            dsts: dsts.to_vec(),
            taddr: t2.into(),
            row: k0,
            col: k0,
            shape: TcShape::S32x32b,
            num: 64,
            pack: false,
            red: None,
            red_abs: false,
            red_nan: false,
            spcompress: false,
        })));
        b.push(Instr::TcgenWait { st: false });
    };
    ld(&mut b, &v);
    ld(&mut b, &v2);
    for (j, (&a, &c)) in v.iter().zip(&v2).enumerate() {
        let kj = b.k_u32(j as u32);
        let kj2 = b.k_u32(64 + j as u32);
        b.mul(Ty::U32, idx, tid, k128);
        b.add_u32(idx, idx, kj);
        b.st_u32(out, idx, a);
        b.mul(Ty::U32, idx, tid, k128);
        b.add_u32(idx, idx, kj2);
        b.st_u32(out, idx, c);
    }
    b.bar_sync(0);
    b.if_(p);
    b.push(Instr::TcgenDealloc { taddr: t.into(), ncols: k64, cta_group: 1, exclusive: false });
    b.push(Instr::TcgenRelinquish { cta_group: 1 });
    b.end_if();
    b.exit();
    let m = b.build_module();
    let inp = inputs(vec![("out", u32_buf(vec![0; 128 * 128]))]);
    let fast = sched::run_with_config(&m, &inp, &mut NoopObserver, &Default::default()).unwrap();
    let slow = sched::run_with_config(&m, &inp, &mut RecordingObserver::new(), &Default::default()).unwrap();
    assert_eq!(fast.status, RunStatus::Completed, "{:?}", fast.status);
    assert_eq!(fast.status, slow.status);
    assert_eq!(fast.outputs, slow.outputs);
    let got: Vec<u32> = fast.outputs.buffers["out"].0.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for tid in 0..128u32 {
        for j in 0..64u32 {
            let want = tid * 64 + j * 7919 + 1;
            assert_eq!(got[(tid * 128 + j) as usize], want, "thread {tid} column {j}");
            assert_eq!(got[(tid * 128 + 64 + j) as usize], want, "thread {tid} column {j} second load");
        }
    }
}

/// Register-array indexing with a warp-uniform index (whole-slot fast path)
/// and a per-lane index agree with the per-element definition, for 32- and
/// 64-bit elements; a uniform out-of-bounds index fails at the first active
/// lane.
#[test]
fn reg_indexed_uniform_and_per_lane() {
    for (ty, dt) in [(Ty::U32, Dtype::U32), (Ty::U64, Dtype::U64)] {
        let mut b = ProgramBuilder::new("reg_indexed", 64);
        let out = b.global("out", dt);
        let lane = b.reg(Ty::U32);
        let tid = b.reg(Ty::U32);
        let q = b.reg(Ty::U32);
        let wide = b.reg(ty);
        let v = b.reg(ty);
        let idx = b.reg(Ty::U32);
        let arr: Vec<Reg> = (0..4).map(|_| b.reg(ty)).collect();
        let base = arr[0];
        b.lane_id(lane);
        b.thread_rank(tid);
        let k3 = b.k_u32(3);
        let k8 = b.k_u32(8);
        b.cast(Ty::U32, ty, wide, tid);
        for (j, &r) in arr.iter().enumerate() {
            let kj = b.k_u32(1000 * (j as u32 + 1));
            b.add_u32(q, tid, kj);
            b.cast(Ty::U32, ty, r, q);
        }
        // Uniform store (element 2 := tid), then uniform and per-lane loads.
        let k2 = b.k_u32(2);
        b.push(Instr::StoreRegIndexed { base, len: 4, idx: k2, value: wide.into() });
        b.mov(idx, k3);
        b.push(Instr::LoadRegIndexed { dst: v, base, len: 4, idx: idx.into() });
        b.mul(Ty::U32, q, tid, k8);
        b.st(ty, out, q, v);
        b.binary(BinOp::And, Ty::U32, idx, lane, k3);
        b.push(Instr::LoadRegIndexed { dst: v, base, len: 4, idx: idx.into() });
        let k1 = b.k_u32(1);
        b.add_u32(q, q, k1);
        b.st(ty, out, q, v);
        // Per-lane store (element lane&3 := tid + 7), read back uniformly.
        let k7 = b.k_u32(7);
        b.add_u32(q, tid, k7);
        b.cast(Ty::U32, ty, wide, q);
        b.push(Instr::StoreRegIndexed { base, len: 4, idx: idx.into(), value: wide.into() });
        for j in 0..4u32 {
            let kj = b.k_u32(j);
            b.push(Instr::LoadRegIndexed { dst: v, base, len: 4, idx: kj });
            b.mul(Ty::U32, q, tid, k8);
            let o = b.k_u32(2 + j);
            b.add_u32(q, q, o);
            b.st(ty, out, q, v);
        }
        b.exit();
        let n = 64 * 8;
        let bytes = dt.mem_bytes() as usize;
        let o = sched::run_with_config(
            &b.build_module(),
            &inputs(vec![("out", numsim_core::sched::ArgValue::Buffer { bytes: vec![0; n * bytes], valid: None })]),
            &mut NoopObserver,
            &Default::default(),
        )
        .unwrap();
        assert_eq!(o.status, RunStatus::Completed, "{ty:?}: {:?}", o.status);
        let raw = &o.outputs.buffers["out"].0;
        let got = |i: usize| -> u64 {
            let mut w = [0u8; 8];
            w[..bytes].copy_from_slice(&raw[i * bytes..(i + 1) * bytes]);
            u64::from_le_bytes(w)
        };
        for t in 0..64u64 {
            let l = t % 32;
            let init = |j: u64| t + 1000 * (j + 1);
            let after_store = |j: u64| if j == 2 { t } else { init(j) };
            assert_eq!(got((t * 8) as usize), after_store(3), "{ty:?} t{t} uniform load");
            assert_eq!(got((t * 8 + 1) as usize), after_store(l & 3), "{ty:?} t{t} per-lane load");
            for j in 0..4u64 {
                let want = if j == (l & 3) { t + 7 } else { after_store(j) };
                assert_eq!(got((t * 8 + 2 + j) as usize), want, "{ty:?} t{t} element {j}");
            }
        }
    }
    // Uniform out-of-bounds index: error names the first active lane.
    let mut b = ProgramBuilder::new("reg_indexed_oob", 32);
    let v = b.reg(Ty::U32);
    let arr: Vec<Reg> = (0..4).map(|_| b.reg(Ty::U32)).collect();
    let k9 = b.k_u32(9);
    b.push(Instr::LoadRegIndexed { dst: v, base: arr[0], len: 4, idx: k9 });
    b.exit();
    let o = sched::run_with_config(&b.build_module(), &Default::default(), &mut NoopObserver, &Default::default()).unwrap();
    match o.status {
        RunStatus::Error(e) => assert_eq!(e.lanes, WarpMask::lane(0), "{e:?}"),
        other => panic!("expected an out-of-bounds error, got {other:?}"),
    }
}

/// Vector (`u32x4`, 16-byte) stores take the store fast path (W13): values,
/// element order and the untouched bytes of inactive lanes match the
/// definition, through shared and global memory.
#[test]
fn vector_store_fast_path() {
    let v4 = Ty::vector(Dtype::U32, 4);
    let mut b = ProgramBuilder::new("vec_store", 64);
    let inp = b.global("inp", Dtype::U32);
    let out = b.global("out", Dtype::U32);
    let sh = b.shared("sh", Dtype::U32, 64 * 4);
    let tid = b.reg(Ty::U32);
    let lane = b.reg(Ty::U32);
    let bit = b.reg(Ty::U32);
    let odd = b.reg(Ty::PRED);
    let i4 = b.reg(Ty::U32);
    let j4 = b.reg(Ty::U32);
    let v = b.reg(v4);
    let w = b.reg(v4);
    b.thread_rank(tid);
    b.lane_id(lane);
    let k1 = b.k_u32(1);
    let k4 = b.k_u32(4);
    let k63 = b.k_u32(63);
    b.binary(BinOp::And, Ty::U32, bit, lane, k1);
    b.compare(CmpOp::Eq, Ty::U32, odd, bit, k1);
    b.mul(Ty::U32, i4, tid, k4);
    b.ld(v4, v, inp, i4);
    // Reverse thread order through shared memory.
    b.binary(BinOp::Sub, Ty::U32, j4, k63, tid);
    b.mul(Ty::U32, j4, j4, k4);
    b.st(v4, sh, j4, v);
    b.bar_sync(0);
    b.ld(v4, w, sh, i4);
    b.st(v4, out, i4, w);
    // Odd lanes overwrite their slot with their own input (even lanes keep
    // the reversed value).
    b.if_(odd);
    b.st(v4, out, i4, v);
    b.end_if();
    b.exit();
    let input: Vec<u32> = (0..256u32).map(|x| x.wrapping_mul(2654435761)).collect();
    let o = sched::run_with_config(
        &b.build_module(),
        &inputs(vec![("inp", u32_buf(input.clone())), ("out", u32_buf(vec![0; 256]))]),
        &mut NoopObserver,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(o.status, RunStatus::Completed, "{:?}", o.status);
    let got: Vec<u32> = o.outputs.buffers["out"].0.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for t in 0..64usize {
        let src = if t % 2 == 1 { t } else { 63 - t };
        for e in 0..4 {
            assert_eq!(got[t * 4 + e], input[src * 4 + e], "thread {t} element {e}");
        }
    }
}

/// Register files of at least `interp::DEFER_REGS_BYTES` are allocated at a
/// warp's first step. (1) A launch that stops on a trap in its first warp
/// leaves the other warps never stepped (never allocated): the run still
/// returns the same status as with an eagerly allocated file. (2) A
/// completed run reads registers placed past the deferral threshold.
#[test]
fn deferred_register_files() {
    fn trap_program(pad: u32) -> numsim_core::Module {
        let mut b = ProgramBuilder::new("deferred_trap", 64);
        b.grid(3, 1, 1);
        for _ in 0..pad {
            b.reg(Ty::U32);
        }
        let f = b.konst(Ty::PRED, 0);
        b.push(Instr::Assert { cond: f, msg: None });
        b.exit();
        b.build_module()
    }
    let slots = numsim_core::interp::DEFER_REGS_BYTES / 256;
    let big = sched::run_with_config(&trap_program(slots as u32 + 8), &Default::default(), &mut NoopObserver, &Default::default()).unwrap();
    let small = sched::run_with_config(&trap_program(0), &Default::default(), &mut NoopObserver, &Default::default()).unwrap();
    match (&big.status, &small.status) {
        (RunStatus::Error(a), RunStatus::Error(b)) => {
            assert_eq!((a.kind.clone(), a.warp, a.pc, a.lanes, &a.message), (b.kind.clone(), b.warp, b.pc, b.lanes, &b.message));
        }
        other => panic!("expected the trap in both runs, got {other:?}"),
    }
    assert_eq!(big.stats, small.stats);

    // Completed run: every thread writes and reads a register beyond the
    // threshold (slot > DEFER_REGS_BYTES / 256).
    let mut b = ProgramBuilder::new("deferred_regs", 64);
    b.grid(2, 1, 1);
    let out = b.global("out", Dtype::U32);
    let tid = b.reg(Ty::U32);
    let cta = b.reg(Ty::U32);
    let idx = b.reg(Ty::U32);
    for _ in 0..slots + 8 {
        b.reg(Ty::U32);
    }
    let far = b.reg(Ty::U32);
    b.thread_rank(tid);
    b.read_special(cta, SpecialReg::CtaLinear);
    let k3 = b.k_u32(3);
    b.mul(Ty::U32, far, tid, k3);
    let k64 = b.k_u32(64);
    b.mul(Ty::U32, idx, cta, k64);
    b.add_u32(idx, idx, tid);
    b.add_u32(far, far, cta);
    b.st_u32(out, idx, far);
    b.exit();
    let o = sched::run_with_config(&b.build_module(), &inputs(vec![("out", u32_buf(vec![0; 128]))]), &mut NoopObserver, &Default::default()).unwrap();
    assert_eq!(o.status, RunStatus::Completed, "{:?}", o.status);
    let got: Vec<u32> = o.outputs.buffers["out"].0.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    for c in 0..2u32 {
        for t in 0..64u32 {
            assert_eq!(got[(c * 64 + t) as usize], t * 3 + c);
        }
    }
}
