# Tool issues found by the ablation runs

Problems in our tools (NumSim, Synccheck, Racecheck, the kcoral remote tools) and in
the GPU-side tools the agents fell back on, as surfaced by the ablation agents.
Each entry names the run, the time (UTC), and where the evidence lives. Paths under
`kda_flow_runs/<run>/worktree/` are relative to that run's worktree.

Status legend: **open**, **fixed** (with commit), **worked around**.

## 1. Divergent warp in front of `.aligned` tcgen05 ops hangs the GPU; Synccheck / Racecheck pass it (false negative) — open

- Run: `abl3-kda-full-r4`, 2026-11-18 22:55–23:40; verified independently 2026-11-19
  02:00–03:00 (scripts and logs in the session scratchpad `v11exp/`).
- Kernel: `kda_flow_runs/abl3-kda-full-r4/worktree/candidates/kda/forward/scratch/v11-aqmat/solution.py`.
  v11 makes MAT thread 0 issue AQ's MMA:
  `with txl.if_(mt == 0), txl.then_(): wait(B_FAC, ...); tcgen05.fence__after_thread_sync(); issue_aq()`
  (`mt = tid - 256`: only lane 0 of MAT warp 0, warp 8). Lanes 1–31 go straight to
  `wait(B_AQ, ...)`. After that wait the same warp runs, with no reconvergence,
  `tcgen05.ld.sync.aligned.32x32b.x32.b32` / `tcgen05.wait::ld.sync.aligned` (lines 675–676,
  and again in the `else` branch), inside further lane-divergent branches (`hsel = lane >= 16`),
  then `bar.sync NB_MAT, 128` (line 763).
- Local Synccheck and Racecheck: **clean** on v11. NumSim did not stall.
- Evidence (my verification):
  1. Placement test, warp_sync() inserted at different points; each cell run once on
     H=2/256, H=96/512, H=96/8192, with and without the watchdog build (6 configs), and
     4 more repeats on `2:256;8:256;32:512` without watchdog:

     | variant | hangs |
     |---|---|
     | v11 as written | 6/6 configs, 4/4 repeats |
     | warp_sync right after the single-lane AQ block (agent's fix) | 0/6, 0/4 |
     | warp_sync right after `wait(B_AQ)`, before the tcgen05.ld region | 0/6, 0/4 |
     | warp_sync only right before `bar.sync NB_MAT` (after the tcgen05.ld region) | 6/6 |

     So the warp must be converged **before the `tcgen05.ld.sync.aligned` region**;
     divergence during the B_FAC / B_AQ spin waits themselves is harmless, and
     reconverging after the region is too late.
  2. `compute-sanitizer --tool synccheck` on the watchdog build, H=2/256: 16 errors
     "Barrier error detected. Divergent thread(s) in warp", threads 272–287 of block 0
     (lanes 16–31 of MAT warp 0), PC 0xa7e0. In the captured NVRTC cubin that PC is in
     the `wait(B_W)` loop right after `BAR.SYNC 0x2, 0x80` (NB_MAT) at 0xa6c0. The fixed
     build: 0 errors. On the plain build at H=96/512, the sanitizer run did not hang and
     reported 0 errors (it changes scheduling), so the sanitizer catches this only
     intermittently.
  3. The hang is nondeterministic: one earlier plain run of v11 at H=2/256 completed.
- Not established: the exact hardware mechanism (why a non-converged
  `tcgen05.ld.sync.aligned` leads to a hang rather than wrong data), and why the agent's
  per-warp trace showed all 16 warps past the final `cta_sync`. PTX defines executing a
  `.aligned` instruction from a non-converged warp as undefined behavior, which is
  consistent with everything above.
- Checker gap: NumSim executes per thread and does not track warp convergence. Suggested
  check: keep an active mask per warp and flag any `.aligned` instruction
  (tcgen05.ld/st/alloc/dealloc/wait, wgmma, `bar.sync` / `barrier.sync.aligned`, ...)
  executed while the warp is not converged, i.e. after a lane-dependent branch with no
  reconvergence point (`__syncwarp`, a full-warp barrier) in between.

- Related (performance, not a checker bug): `abl3-kda-full-r4`, 2026-11-19 01:22–01:28.
  NCU source view showed each `tcgen05.mma` issued from inside `if tid == 0` wrapped in
  a waterfall loop by the compiler (~55–65 cycles per MMA, 16 MMAs ≈ 860 ns on the
  critical path). Issuing from a converged warp as the KB GEMM does (elect-style)
  made the UTCHMMAs issue back to back: 10–15% on every shape, 1.678x → 1.979x.
  A lint for "single-thread-guarded tcgen05/.aligned ops" would catch both problems.

## 2. NumSim has no classifier for `prim.exp2`, so checkers return INCOMPLETE — open

- Run: `abl3-kda-nokb-r1`, 2026-11-18. Report and reproducer:
  `kda_flow_runs/abl3-kda-nokb-r1/worktree/bugs/synccheck-prim-exp2-incomplete/`.
- Any kernel calling `txl.exp2(...)` gets verdict `incomplete` from both Synccheck
  and Racecheck: `op#48:Call(prim.exp2(float32)->float32): unregistered raw call (no
  NumSim classifier registration)`. The same kernel with `txl.exp` is `clean`.
- The agent's workaround was `txl.ptx.ex2.approx`. A pure math op should not block
  certification.

## 3. NumSim cannot build under the hmz Landlock fence (EXDEV on rename) — worked around

- Runs: batch 1 `abl-kda-full-r1`, `abl-kda-full-r2`, `abl-kda-nokb-r1`, `abl-kda-nokb-r2`
  (2026-11-17/18). Reports: `bugs/numsim-checkers-blocked/`, `bugs/numsim-exdev-rename/`,
  `bugs/numsim-exdev-sandbox/` in those worktrees.
- Host kernel 5.15 only has Landlock ABI 1, under which every cross-directory rename
  returns EXDEV. `tirx_harness/numsim/transpiler/compile_cache.py:_write_cached`
  publishes with `os.replace(dir, dir)`, and cargo/rustc also write `.rmeta` by rename,
  so the Rust build fails (`Invalid cross-device link (os error 18)`).
- Only `abl-kda-nokb-r2` got NumSim working, with its own LD_PRELOAD copy+unlink shim.
  The others fell back to GPU compute-sanitizer.
- Remote fallback also failed: the kcoral server has no Rust toolchain
  (`cannot resolve NumSim Rust tool 'cargo'`).
- Workaround for batch 3: flow `ablation/flows/ralph_open` (no fence).
- Possible fix: copy-then-unlink fallback on EXDEV in `compile_cache.py`, plus a
  prebuilt NumSim native library so cargo is not needed at run time.

## 4. Racecheck: recurring "D_N lane overlap" ERROR the agent judges benign — needs triage

- Run: `abl3-kda-full-r3`, 2026-11-18 22:08–22:57, on every split/fused variant (s8, f2,
  f6, s10).
- Racecheck reports one ERROR between the Gram MMA (`mma_n(1)`) and the `elem` norm load
  on the D_N tile; Synccheck is clean, NumSim matches the fp64 reference, and the GPU
  output is correct. The agent calls it "the known benign D_N lane-overlap pattern".
  It also reports "uninitialized-lane" review items.
- To triage: either a Racecheck false positive (the lanes the MMA writes and the lanes
  the load reads don't actually overlap) or a real but harmless race. A recurring ERROR
  the agent learns to ignore weakens the tool either way.

## 5. compute-sanitizer racecheck flags correct mbarrier / named-barrier hand-offs — external, worked around

- Runs: `abl-kda-full-r1` (`bugs/mbarrier-racecheck-false-positive/`) and
  `abl-kda-nokb-r1` (`bugs/sanitizer-racecheck-mbarrier/`), 2026-11-17.
- Producer stores → `mbarrier.arrive` (release) → consumer `try_wait.parity.acquire` →
  loads is reported as a RAW hazard (thousands of reports), although outputs are
  correct. Named-barrier hand-offs (`bar.arrive` / `bar.sync`) are flagged the same way.
- This is NVIDIA's tool, not ours, but it matters because it is the fallback when
  NumSim is unavailable (issue 3): device racecheck cannot certify warp-specialized
  kernels.

## 6. NCU profiled a different binary (NVRTC mismatch) — fixed in 901708b

- Runs: batch 1 `abl-kda-notools-r1` (diagnosed), `abl-kda-full-r1` (hit it).
- Under NCU, the injected LD_LIBRARY_PATH made TIRx load the system NVRTC 13.5 instead
  of the pip NVRTC 13.2 the benchmark uses, so NCU showed different SASS and spills.
- Fixed by LD_PRELOADing the worker's pip NVRTC in `kcoral_tool_worker.ncu`.

## 7. kcoral remote tools — open

- **A hung kernel blocks a shared worker.** There is no server-side timeout on a
  request: a kernel that deadlocks holds its worker until the client timeout, about
  5 min (`abl3-kda-full-r3`, 2026-11-18 20:10). Agents wrote their own wrappers with a
  shorter server-side timeout, and device-side watchdog builds, to avoid this.
- **Outages are not visible to agents.** During the 2026-11-18 19:45–21:30 pool outage
  (`status: unavailable`, capacity 0), all four batch-3 runs read the stalled
  requests as their own kernel hanging (nokb-r1 v95, nokb-r2 v107, full-r3 split
  kernel, full-r4 v3) and spent time debugging kernels that were fine. The client
  should fail fast with a clear "pool unavailable" message instead of queueing silently.
- **Remote runs return only stdout**, and large outputs are truncated
  (`abl-kda-nokb-r1`: a 16.8 MB racecheck log was cut; `abl3-kda-full-r3` noted
  "remote runs return only stdout" as something it had to rediscover).

## 8. NumSim usability — open (minor)

- NumSim needs a static grid; kernels with a dynamic grid had to be copied into a
  static-grid variant before simulation (`abl3-kda-full-r3`, 2026-11-18 21:07, which
  saved this as a memory note because it "cost time to rediscover").
