//! `ProgramBuilder`: write `Program`s by hand in Rust tests (no Python/TVM).
//!
//! ```
//! use numsim_core::testutil::ProgramBuilder;
//! use numsim_core::{Dtype, Ty};
//! let mut b = ProgramBuilder::new("vadd", 64);
//! let x = b.global("x", Dtype::F32);
//! let y = b.global("y", Dtype::F32);
//! let i = b.reg(Ty::U32);
//! let v = b.reg(Ty::F32);
//! b.thread_rank(i);
//! b.ld_f32(v, x, i);
//! b.add_f32(v, v, v);
//! b.st_f32(y, i, v);
//! let p = b.build();
//! assert!(p.validate().is_ok());
//! ```

use crate::arena::Space;
use crate::dtype::{Dtype, Ty};
use crate::program::*;
use crate::site::{SiteId, SiteInfo};
use crate::sync::async_group::Domain;

enum Open {
    If { if_pc: Pc, else_pc: Option<Pc> },
    Loop { begin_pc: Pc, if_pc: Option<Pc> },
}

/// Incremental `Program` builder; control-flow helpers patch jump targets.
pub struct ProgramBuilder {
    p: Program,
    open: Vec<Open>,
    site: SiteId,
}

impl ProgramBuilder {
    /// One CTA of `threads` threads.
    pub fn new(name: &str, threads: u32) -> ProgramBuilder {
        ProgramBuilder { p: Program::empty(name, threads), open: Vec::new(), site: SiteId::NONE }
    }

    pub fn grid(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.p.topology.grid = [DimExpr::Const(x as i64), DimExpr::Const(y as i64), DimExpr::Const(z as i64)];
        self
    }

    pub fn cluster(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.p.topology.cluster = [x, y, z];
        self
    }

    pub fn block(&mut self, x: u32, y: u32, z: u32) -> &mut Self {
        self.p.topology.block = [x, y, z];
        self
    }

    /// Start a new source site for subsequently pushed instructions.
    pub fn site(&mut self, op: &str, line: u32) -> SiteId {
        let id = SiteId(self.p.sites.len() as u32);
        self.p.sites.push(SiteInfo {
            op_name: op.to_string(),
            spans: vec![crate::site::Span { file: None, line, col: 0, end_line: line, end_col: 0 }],
            ..SiteInfo::default()
        });
        self.site = id;
        id
    }

    /// Clear the current site (pure ALU code).
    pub fn no_site(&mut self) {
        self.site = SiteId::NONE;
    }

    /// A fresh register.
    pub fn reg(&mut self, ty: Ty) -> Reg {
        let r = Reg(self.p.regs.len() as u32);
        self.p.regs.push(RegDecl { ty, name: None, uniform: false });
        r
    }

    /// An interned constant.
    pub fn konst(&mut self, ty: Ty, bits: u128) -> Operand {
        let k = Const { ty, bits };
        let id = match self.p.consts.iter().position(|c| *c == k) {
            Some(i) => ConstId(i as u32),
            None => {
                self.p.consts.push(k);
                ConstId(self.p.consts.len() as u32 - 1)
            }
        };
        Operand::Const(id)
    }
    pub fn k_u32(&mut self, v: u32) -> Operand {
        self.konst(Ty::U32, v as u128)
    }
    pub fn k_i32(&mut self, v: i32) -> Operand {
        self.konst(Ty::S32, v as u32 as u128)
    }
    pub fn k_f32(&mut self, v: f32) -> Operand {
        self.konst(Ty::F32, v.to_bits() as u128)
    }

    pub fn string(&mut self, s: &str) -> StrId {
        self.p.strings.push(s.to_string());
        StrId(self.p.strings.len() as u32 - 1)
    }

    /// Intern a generic op.
    pub fn op(&mut self, name: &str, mods: &[&str]) -> OpId {
        let key = OpKey { name: name.to_string(), mods: mods.iter().map(|m| m.to_string()).collect() };
        match self.p.ops.iter().position(|o| *o == key) {
            Some(i) => OpId(i as u32),
            None => {
                self.p.ops.push(key);
                OpId(self.p.ops.len() as u32 - 1)
            }
        }
    }

    fn buffer(&mut self, decl: BufferDecl) -> Buf {
        self.p.buffers.push(decl);
        Buf(self.p.buffers.len() as u32 - 1)
    }

    fn param(&mut self, name: &str, kind: ParamKind, dtype: Option<Ty>, buf: Option<Buf>) -> ParamId {
        self.p.host_abi.push(ParamSlot {
            name: name.to_string(),
            local_name: name.to_string(),
            aliases: Vec::new(),
            kind,
            dtype,
            shape: Vec::new(),
            tensor_map: None,
            implicit_base: None,
            buf,
        });
        ParamId(self.p.host_abi.len() as u32 - 1)
    }

    /// A global buffer parameter (size from the bound host argument).
    pub fn global(&mut self, name: &str, dtype: Dtype) -> Buf {
        let pid = ParamId(self.p.host_abi.len() as u32);
        let buf = self.buffer(BufferDecl {
            name: name.to_string(),
            space: Space::Global,
            dtype: dtype.into(),
            shape: Vec::new(),
            strides: Vec::new(),
            param_slot: Some(pid),
            base: 0,
            byte_len: None,
            align: 16,
            view_of: None,
            sync_words: false,
        });
        self.param(name, ParamKind::Buffer, Some(dtype.into()), Some(buf));
        buf
    }

    /// A scalar parameter; read it with [`Instr::ReadParam`].
    pub fn scalar_param(&mut self, name: &str, dtype: Dtype) -> ParamId {
        self.param(name, ParamKind::Scalar, Some(dtype.into()), None)
    }

    /// A shared buffer of `elems` elements at the next 16-byte aligned
    /// offset of the shared window (grows `static_smem_bytes`).
    pub fn shared(&mut self, name: &str, dtype: Dtype, elems: u64) -> Buf {
        let bytes = elems * dtype.mem_bytes() as u64;
        let base = (self.p.topology.static_smem_bytes as u64).div_ceil(16) * 16;
        self.p.topology.static_smem_bytes = (base + bytes) as u32;
        self.buffer(BufferDecl {
            name: name.to_string(),
            space: Space::Shared,
            dtype: dtype.into(),
            shape: vec![DimExpr::Const(elems as i64)],
            strides: Vec::new(),
            param_slot: None,
            base,
            byte_len: Some(DimExpr::Const(bytes as i64)),
            align: 16,
            view_of: None,
            sync_words: false,
        })
    }

    pub fn declare_sync_words(&mut self, b: Buf) {
        self.p.buffers[b.0 as usize].sync_words = true;
    }

    /// Append an instruction at the current site.
    pub fn push(&mut self, ins: Instr) -> Pc {
        self.p.code.push(ins);
        self.p.code_sites.push(self.site);
        Pc(self.p.code.len() as u32 - 1)
    }

    pub fn next_pc(&self) -> Pc {
        Pc(self.p.code.len() as u32)
    }

    // ----- registers / ALU -----

    pub fn read_special(&mut self, dst: Reg, sreg: SpecialReg) {
        self.push(Instr::ReadSpecial { dst, sreg });
    }
    pub fn lane_id(&mut self, dst: Reg) {
        self.read_special(dst, SpecialReg::LaneId);
    }
    pub fn warp_id(&mut self, dst: Reg) {
        self.read_special(dst, SpecialReg::WarpInCta);
    }
    pub fn thread_rank(&mut self, dst: Reg) {
        self.read_special(dst, SpecialReg::ThreadInCta);
    }
    pub fn mov(&mut self, dst: Reg, src: impl Into<Operand>) {
        self.push(Instr::Mov { dst, src: src.into() });
    }
    pub fn binary(&mut self, op: BinOp, ty: Ty, dst: Reg, a: impl Into<Operand>, b: impl Into<Operand>) {
        self.push(Instr::Binary { op, ty, dst, a: a.into(), b: b.into() });
    }
    pub fn add(&mut self, ty: Ty, dst: Reg, a: impl Into<Operand>, b: impl Into<Operand>) {
        self.binary(BinOp::Add, ty, dst, a, b);
    }
    pub fn mul(&mut self, ty: Ty, dst: Reg, a: impl Into<Operand>, b: impl Into<Operand>) {
        self.binary(BinOp::Mul, ty, dst, a, b);
    }
    pub fn add_f32(&mut self, dst: Reg, a: impl Into<Operand>, b: impl Into<Operand>) {
        self.add(Ty::F32, dst, a, b);
    }
    pub fn mul_f32(&mut self, dst: Reg, a: impl Into<Operand>, b: impl Into<Operand>) {
        self.mul(Ty::F32, dst, a, b);
    }
    pub fn add_u32(&mut self, dst: Reg, a: impl Into<Operand>, b: impl Into<Operand>) {
        self.add(Ty::U32, dst, a, b);
    }
    pub fn fma(&mut self, ty: Ty, dst: Reg, a: impl Into<Operand>, b: impl Into<Operand>, c: impl Into<Operand>) {
        self.push(Instr::Ternary { op: TerOp::Fma, ty, dst, a: a.into(), b: b.into(), c: c.into() });
    }
    pub fn compare(&mut self, op: CmpOp, ty: Ty, dst: Reg, a: impl Into<Operand>, b: impl Into<Operand>) {
        self.push(Instr::Compare { op, ty, dst, a: a.into(), b: b.into() });
    }
    pub fn cast(&mut self, from: Ty, to: Ty, dst: Reg, src: impl Into<Operand>) {
        self.push(Instr::Cast { from, to, dst, src: src.into(), rnd: Rounding::Default, sat: false });
    }
    /// Generic PTX op.
    pub fn ptx(&mut self, name: &str, mods: &[&str], dsts: &[Reg], srcs: &[Operand]) {
        let op = self.op(name, mods);
        self.push(Instr::Ptx { op, dsts: dsts.to_vec(), srcs: srcs.to_vec(), pred: None, keep_dst: true });
    }

    // ----- memory -----

    /// `dst = buf[idx]` (element index), weak.
    pub fn ld(&mut self, ty: Ty, dst: Reg, buf: Buf, idx: impl Into<Operand>) {
        self.push(Instr::Load {
            ty,
            dst,
            buf,
            offset: idx.into(),
            sem: Sem::Weak,
            scope: Scope::default(),
            mods: MemMods::default(),
        });
    }
    /// `buf[idx] = value`, weak.
    pub fn st(&mut self, ty: Ty, buf: Buf, idx: impl Into<Operand>, value: impl Into<Operand>) {
        self.push(Instr::Store {
            ty,
            buf,
            offset: idx.into(),
            value: value.into(),
            sem: Sem::Weak,
            scope: Scope::default(),
            mods: MemMods::default(),
        });
    }
    pub fn ld_f32(&mut self, dst: Reg, buf: Buf, idx: impl Into<Operand>) {
        self.ld(Ty::F32, dst, buf, idx);
    }
    pub fn st_f32(&mut self, buf: Buf, idx: impl Into<Operand>, value: impl Into<Operand>) {
        self.st(Ty::F32, buf, idx, value);
    }
    pub fn ld_u32(&mut self, dst: Reg, buf: Buf, idx: impl Into<Operand>) {
        self.ld(Ty::U32, dst, buf, idx);
    }
    pub fn st_u32(&mut self, buf: Buf, idx: impl Into<Operand>, value: impl Into<Operand>) {
        self.st(Ty::U32, buf, idx, value);
    }
    /// `dst` (u64 generic) = `&buf[idx]`.
    pub fn addr_of(&mut self, dst: Reg, buf: Buf, idx: impl Into<Operand>) {
        self.push(Instr::AddrOf { dst, buf, offset: idx.into() });
    }
    /// `dst` (u32 shared::cta) = `cvta.to.shared(&buf[idx])`; allocates a temp.
    pub fn smem_addr(&mut self, dst: Reg, buf: Buf, idx: impl Into<Operand>) {
        let g = self.reg(Ty::U64);
        self.addr_of(g, buf, idx);
        self.push(Instr::Cvta { dst, src: g.into(), space: AddrSpace::Shared, to_generic: false });
    }
    /// `red.add.relaxed.gpu` through a generic address register.
    pub fn red_add(&mut self, ty: Ty, addr: Reg, value: impl Into<Operand>) {
        self.push(Instr::Atom {
            op: AtomOp::Add,
            ty,
            dst: None,
            addr: addr.into(),
            space: AddrSpace::Generic,
            value: value.into(),
            cmp: None,
            sem: Sem::Relaxed,
            scope: Scope::Gpu,
            ftz: false,
        });
    }

    // ----- control flow -----

    pub fn if_(&mut self, cond: impl Into<Operand>) {
        let pc = self.push(Instr::If { cond: cond.into(), else_pc: Pc(0), end_pc: Pc(0), elect: false });
        self.open.push(Open::If { if_pc: pc, else_pc: None });
    }
    pub fn else_(&mut self) {
        let pc = self.push(Instr::Else { end_pc: Pc(0) });
        match self.open.last_mut() {
            Some(Open::If { else_pc, .. }) if else_pc.is_none() => *else_pc = Some(pc),
            _ => panic!("else_ without matching if_"),
        }
    }
    pub fn end_if(&mut self) {
        let end = self.push(Instr::EndIf);
        let Some(Open::If { if_pc, else_pc }) = self.open.pop() else { panic!("end_if without if_") };
        if let Instr::If { else_pc: e, end_pc, .. } = &mut self.p.code[if_pc.0 as usize] {
            *e = else_pc.unwrap_or(end);
            *end_pc = end;
        }
        if let Some(epc) = else_pc {
            if let Instr::Else { end_pc } = &mut self.p.code[epc.0 as usize] {
                *end_pc = end;
            }
        }
    }
    /// Begin a loop. Then emit the condition computation, [`Self::loop_if`],
    /// the body and [`Self::loop_end`]. `LoopEnd` jumps back to the first
    /// instruction after `LoopBegin` (the condition computation).
    pub fn loop_begin(&mut self) {
        let pc = self.push(Instr::LoopBegin { end_pc: Pc(0) });
        self.open.push(Open::Loop { begin_pc: pc, if_pc: None });
    }
    pub fn loop_if(&mut self, cond: impl Into<Operand>) {
        let pc = self.push(Instr::LoopIf { cond: cond.into(), end_pc: Pc(0) });
        match self.open.last_mut() {
            Some(Open::Loop { if_pc, .. }) if if_pc.is_none() => *if_pc = Some(pc),
            _ => panic!("loop_if outside loop_begin"),
        }
    }
    pub fn break_(&mut self) {
        self.push(Instr::Break);
    }
    pub fn continue_(&mut self) {
        self.push(Instr::Continue);
    }
    pub fn loop_end(&mut self) {
        let Some(Open::Loop { begin_pc, if_pc }) = self.open.pop() else { panic!("loop_end without loop") };
        let if_pc = if_pc.expect("loop without loop_if");
        let end = self.push(Instr::LoopEnd { head_pc: Pc(begin_pc.0 + 1) });
        if let Instr::LoopBegin { end_pc } = &mut self.p.code[begin_pc.0 as usize] {
            *end_pc = end;
        }
        if let Instr::LoopIf { end_pc, .. } = &mut self.p.code[if_pc.0 as usize] {
            *end_pc = end;
        }
    }
    pub fn exit(&mut self) {
        self.push(Instr::Exit);
    }

    // ----- sync -----

    pub fn bar_sync(&mut self, id: u32) {
        let id = self.k_u32(id);
        self.push(Instr::Barrier { kind: BarKind::Sync, id, count: None, aligned: true });
    }
    /// mbarrier.init through a shared::cta address register.
    pub fn mbar_init(&mut self, mbar: Reg, count: u32) {
        let count = self.k_u32(count);
        self.push(Instr::MbarInit { mbar: mbar.into(), space: AddrSpace::Shared, count, layout_v1: false });
    }
    pub fn mbar_arrive(&mut self, mbar: Reg, state: Option<Reg>) {
        self.push(Instr::MbarArrive(MbarArriveArgs {
            mbar: mbar.into(),
            space: AddrSpace::Shared,
            count: None,
            expect_tx: None,
            drop: false,
            no_complete: false,
            sem: Sem::Release,
            scope: Scope::Cta,
            multicast: None,
            state,
        }));
    }
    pub fn mbar_wait_parity(&mut self, mbar: Reg, parity: impl Into<Operand>) {
        self.push(Instr::MbarWait {
            mbar: mbar.into(),
            space: AddrSpace::Shared,
            phase: PhaseArg::Parity(parity.into()),
            sem: Sem::Acquire,
            scope: Scope::Cta,
        });
    }
    pub fn cp_async_commit_wait_all(&mut self) {
        self.push(Instr::AsyncCommit { domain: Domain::CpAsync });
        self.push(Instr::AsyncWait { domain: Domain::CpAsync, n: 0, read: false });
    }
    pub fn fence(&mut self, kind: FenceKind, sem: Sem, scope: Scope) {
        self.push(Instr::Fence { kind, sem, scope });
    }

    /// Finish; panics if the program does not validate.
    pub fn build(self) -> Program {
        assert!(self.open.is_empty(), "unclosed control-flow frame");
        if let Err(e) = self.p.validate() {
            panic!("ProgramBuilder produced an invalid program: {e}\n{}", self.p);
        }
        self.p
    }

    pub fn build_module(self) -> Module {
        Module::new(vec![self.build()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Program {
        let mut b = ProgramBuilder::new("t", 64);
        b.site("tirx.ptx.ld", 3);
        let x = b.global("x", Dtype::F32);
        let s = b.shared("s", Dtype::F32, 64);
        let bar = b.shared("bar", Dtype::U64, 1);
        let r0 = b.reg(Ty::U32);
        let r1 = b.reg(Ty::F32);
        let p = b.reg(Ty::PRED);
        let m = b.reg(Ty::U32);
        let wide = b.reg(Ty::vector(Dtype::U32, 4));
        b.thread_rank(r0);
        b.ld_f32(r1, x, r0);
        let k32 = b.k_u32(32);
        b.compare(CmpOp::Lt, Ty::U32, p, r0, k32);
        b.if_(p);
        b.st_f32(s, r0, r1);
        b.else_();
        let two = b.k_f32(2.0);
        b.mul_f32(r1, r1, two);
        b.end_if();
        b.loop_begin();
        b.loop_if(p);
        let zero = b.konst(Ty::PRED, 0);
        b.mov(p, zero);
        b.loop_end();
        let z = b.k_u32(0);
        b.smem_addr(m, bar, z);
        b.mbar_init(m, 1);
        b.bar_sync(0);
        b.ptx("tirx.ptx.mov_pack_b32x4", &[], &[wide], &[r0.into(), r0.into(), r0.into(), r0.into()]);
        b.exit();
        b.build()
    }

    #[test]
    fn builder_patches_targets() {
        let p = sample();
        let if_pc = p.code.iter().position(|i| matches!(i, Instr::If { .. })).unwrap();
        match &p.code[if_pc] {
            Instr::If { else_pc, end_pc, .. } => {
                assert!(matches!(p.code[else_pc.0 as usize], Instr::Else { .. }));
                assert!(matches!(p.code[end_pc.0 as usize], Instr::EndIf));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(p.topology.static_smem_bytes, 264);
        let slots = p.reg_slot_offsets();
        assert_eq!(*slots.last().unwrap(), 4 + 2 + 1); // 4 scalars + u32x4 + smem temp
        let text = p.to_string();
        assert!(text.contains("Load f32 r1 <- b0[r0]"), "{text}");
    }

    #[test]
    fn serde_roundtrip() {
        let m = Module::new(vec![sample()]);
        assert_eq!(Module::from_bytes(&m.to_bytes()).unwrap(), m);
        assert_eq!(Module::from_json(&m.to_json()).unwrap(), m);
    }
}
