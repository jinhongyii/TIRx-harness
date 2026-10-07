//! Codegen backend (W7): a *printer* from `Program` to Rust source in which
//! every instruction is a call to the same `interp::handlers` function the
//! interpreter uses, with register numbers, dtypes, spaces and other fields
//! as constants. No semantics may appear in generated code (plan 2.3).
//!
//! # Pipeline
//!
//! * [`emit_rust`] / [`emit_module`] print one `lib.rs` (see [`emit`] for
//!   the shape): per kernel a resumable step function exported as
//!   `{STEP_SYMBOL_PREFIX}{kernel_index}` with the [`WarpStepFn`] type, the
//!   kernel's constant pool and op table as `static`s, and a load-time
//!   check against the runtime `Program`.
//! * [`build`] / [`build_module`] compile it into a cdylib (one `rustc`
//!   invocation) against a `numsim-core` rlib built from this crate's
//!   sources (one cached `cargo build` per source/toolchain/profile), load
//!   it and return [`LoadedBackend`] (`Backend::Codegen`).
//!
//! # ABI
//!
//! The cdylib statically contains its *own* copy of `numsim-core` and std;
//! `ExecCtx` and everything reachable from it cross the boundary by
//! reference with the Rust ABI. That is sound only for identical layouts,
//! which requires the same rustc and the same `numsim-core` sources; the
//! loader verifies [`abi_fingerprint`] (sizes, alignments and field
//! offsets of every boundary type) and refuses a mismatch. Consequences of
//! the second std copy:
//! * A panic cannot unwind across the boundary (the host's runtime aborts
//!   on a foreign Rust exception), so generated step functions catch
//!   panics and return `ExecErrorKind::Internal` ([`rt::guard`]). A panic
//!   raised by *host* code called from generated code (an `Observer`
//!   method, a resolved `PtxFn`) still aborts: observers must not panic.
//! * Heap blocks change owners across the boundary (`Vec` growth, error
//!   strings), so host and generated code must share the system allocator
//!   (no custom `#[global_allocator]` in the host).
//! * Loaded libraries are never unloaded: `WarpStepFn` pointers are plain
//!   `fn`s with no lifetime.

pub mod emit;
pub mod lit;
mod compile;

pub use compile::{backend_for, build, build_module, load, rustc_version, BuildError, BuildOptions, BuildStats, LoadedBackend, OptLevel};
pub use emit::{emit_module, EMIT_VERSION};

use crate::program::{Module, Program};

/// Exported symbol prefix of per-kernel step functions.
pub const STEP_SYMBOL_PREFIX: &str = "numsim_step_kernel_";
/// Exported symbol prefix of per-kernel `fn(&Program) -> bool` load checks.
pub const CHECK_SYMBOL_PREFIX: &str = "numsim_check_kernel_";

/// Print the generated source for a single-kernel module made of `program`.
pub fn emit_rust(program: &Program) -> String {
    let module = Module::new(vec![program.clone()]);
    emit_module(&module, &compile::sha256(&module.to_bytes()))
}

/// Runtime support called by generated code (beyond the interpreter's own
/// `interp::{begin_instr, end_instr, fall_off_end}`).
pub mod rt {
    use crate::interp::{ExecCtx, ExecError, ExecErrorKind, StepResult, WarpStepFn};
    use std::any::Any;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Run `body` and turn a panic into `ExecErrorKind::Internal` at the
    /// current pc (a panic must not unwind into the host; see module docs).
    #[inline(always)]
    pub fn guard(ctx: &mut ExecCtx<'_>, quantum: u32, body: WarpStepFn) -> StepResult {
        match catch_unwind(AssertUnwindSafe(|| body(&mut *ctx, quantum))) {
            Ok(r) => r,
            Err(payload) => StepResult::Error(panic_error(ctx, &*payload)),
        }
    }

    /// The error a caught panic becomes (also used by the differential
    /// tests to compare an interpreter panic with a codegen result).
    pub fn panic_error(ctx: &ExecCtx<'_>, payload: &(dyn Any + Send)) -> ExecError {
        ctx.error(ExecErrorKind::Internal, format!("panic: {}", panic_message(payload)))
    }

    pub fn panic_message(payload: &(dyn Any + Send)) -> String {
        if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        }
    }
}

/// Number of words in [`abi_fingerprint`].
pub const ABI_WORDS: usize = 96;

/// Layout facts of every type crossing the host/cdylib boundary; host and
/// library must agree word for word.
pub fn abi_fingerprint() -> [u64; ABI_WORDS] {
    use crate::arena::{Arena, View};
    use crate::interp::*;
    use crate::observe::{Access, SyncEvent};
    use crate::program::{Instr, LaunchShape, Operand};
    use crate::sched::{InboxMsg, RunConfig};
    use crate::sync::{ResourceId, SyncTable};
    use crate::value::RegFile;
    use std::mem::{align_of, offset_of, size_of};

    macro_rules! sa {
        ($($t:ty),*) => { [$(size_of::<$t>(), align_of::<$t>()),*] };
    }
    let words: Vec<usize> = [
        // Version of this list.
        vec![1, EMIT_VERSION as usize],
        sa!(
            ExecCtx<'static>, WarpState, Program, Instr, Operand, StepResult, ExecError, ExecErrorKind, Flow,
            Result<Flow, ExecError>, Arena, View, SyncTable, ResourceId, RunConfig, LaunchShape, CtaCtx, Loaded,
            LaunchCounters, BufBinding, MaskFrame, FrameKind, PollState, WarpStatus, InboxMsg, Vec<InboxMsg>,
            RegFile, SyncEvent, Access<'static>, String
        )
        .to_vec(),
        vec![
            offset_of!(ExecCtx<'static>, program),
            offset_of!(ExecCtx<'static>, loaded),
            offset_of!(ExecCtx<'static>, launch),
            offset_of!(ExecCtx<'static>, config),
            offset_of!(ExecCtx<'static>, warp),
            offset_of!(ExecCtx<'static>, cta),
            offset_of!(ExecCtx<'static>, buffers),
            offset_of!(ExecCtx<'static>, arena),
            offset_of!(ExecCtx<'static>, sync),
            offset_of!(ExecCtx<'static>, outbox),
            offset_of!(ExecCtx<'static>, observer),
            offset_of!(ExecCtx<'static>, observing),
            offset_of!(ExecCtx<'static>, counters),
            offset_of!(WarpState, pc),
            offset_of!(WarpState, regs),
            offset_of!(WarpState, active),
            offset_of!(WarpState, live),
            offset_of!(WarpState, frames),
            offset_of!(WarpState, status),
            offset_of!(WarpState, resume),
            offset_of!(WarpState, poll),
            offset_of!(WarpState, epoch),
            offset_of!(WarpState, steps),
            offset_of!(Program, code),
            offset_of!(Program, consts),
            offset_of!(Program, code_sites),
            offset_of!(Program, ops),
            offset_of!(LaunchCounters, instrs),
            offset_of!(Loaded, slots),
            offset_of!(Loaded, ops),
        ],
    ]
    .concat();
    assert!(words.len() <= ABI_WORDS, "ABI_WORDS too small: {}", words.len());
    let mut out = [0u64; ABI_WORDS];
    for (o, w) in out.iter_mut().zip(words) {
        *o = w as u64;
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::ProgramBuilder;
    use crate::{Dtype, Ty};

    #[test]
    fn emits_handler_calls_with_constant_operands() {
        let mut b = ProgramBuilder::new("vadd", 64);
        let x = b.global("x", Dtype::F32);
        let y = b.global("y", Dtype::F32);
        let i = b.reg(Ty::U32);
        let v = b.reg(Ty::F32);
        b.thread_rank(i);
        b.ld_f32(v, x, i);
        b.add_f32(v, v, v);
        b.st_f32(y, i, v);
        let src = emit_rust(&b.build());
        assert!(src.contains("h::read_special(ctx, Reg(0u32), SpecialReg::ThreadInCta)"), "{src}");
        assert!(src.contains("h::load(ctx, Ty { elem: Dtype::F32, lanes: 1u8 }, Reg(1u32), Buf(0u32), Operand::Reg(Reg(0u32)), Sem::Weak, Scope::Gpu, MemMods { cache: CacheOp::Default, evict: Evict::Normal, l2_prefetch: 0u16, policy: None, nc: false, uniform: false })"), "{src}");
        assert!(src.contains("pub extern \"Rust\" fn numsim_step_kernel_0("), "{src}");
        assert!(!src.contains("Instr::Load"), "no decode of simple instructions");
    }

    #[test]
    fn abi_fingerprint_is_stable_in_process() {
        assert_eq!(abi_fingerprint(), abi_fingerprint());
    }
}
