# TIRx Harness Ablation Study — Plan

Status: kda done (batch 1 and batch 3, see "Results so far" at the end); Phase 3 and the mla batch not started. Last updated 2026-11-19.

## Question

How much of the full TIRx harness's result comes from (a) the authoring
language, (b) our tools, and (c) the knowledge base?

## Fixed conditions (identical in every cell)

| Item | Setting |
| --- | --- |
| Agent | Opus 5.5, max effort, Humanize ralph loop |
| Hardware / scoring | GB200 (sm_100a) via the kcoral pool behind `http://127.0.0.1:9000` |
| Budget | 1 day of wall-clock per run |
| Repetitions | 2 runs per cell |
| Tasks | `kda_forward` (12-case multi-shape), `mla_dsv4` (multi-shape) — may switch to single-shape variants depending on benchmark latency (see "Benchmark latency") |
| Concurrency | At most 12 runs at the same time (kcoral capacity) |

## Dimension definitions

- **Tools (T)** — only the tools we wrote: `tirx-debug-kernel`
  (Synccheck, Racecheck, NumSim) and `tirx-profile-kernel` (IKET,
  dump-source), plus their `python -m tirx_harness` / kcoral IKET entry points.
  NCU, Compute Sanitizer and `ncu-report-skill` stay available in every cell.
- **Knowledge base (KB)** — `tirx-wiki` (canonical kernels, optimization
  playbook, perf checklist, manuals, blogs, fetched repos) and `KernelWiki`.
  KB-off gives no language manual; the agent may read our source directly,
  but everything in the installed `tirx_kernels` package other than
  `tirx_lite` is banned (it contains canonical kernels).
- **Language (L)** — TIRx-lite (current contract, `check_ir` on);
  CuTeDSL; CUDA (inline PTX if necessary); Triton. Non-TIRx languages have a
  loose contract: "use <language> (CUDA: with inline PTX if necessary)".

## Experiment matrix

| # | Language | Our tools | KB | kda_forward | mla_dsv4 |
| --- | --- | --- | --- | --- | --- |
| 1 | TIRx-lite | yes | yes | x2 | x2 |
| 2 | TIRx-lite | no | yes | x2 | x2 |
| 3 | TIRx-lite | yes | no | x2 | x2 |
| 5 | CuTeDSL | n/a | yes | x2 | x2 |
| 6 | CUDA (+ inline PTX) | n/a | yes | x2 | x2 |
| 7 | Triton | n/a | yes | x2 | x2 |
| 4 | TIRx-lite | no | no | deferred | deferred |

- 6 configurations x 2 tasks x 2 runs = **24 runs**.
- Row 4 (bare TIRx) is deferred: decide after the 24 runs finish, if
  resources remain.
- Rows 5-7 reuse the existing `kernel_authoring: task` path (as in
  `gdn_prefill_native`): KernelWiki + IKET + ncu-report-skill. Our
  TIRx-only tools do not apply to them.
- Comparisons: 1 vs 2 = tools; 1 vs 3 = KB; 1 vs 5/6/7 = language
  (full harness vs each alternative stack).

## Scheduling (max 12 concurrent)

| Batch | Contents | Runs | Duration |
| --- | --- | --- | --- |
| 1 | Run #1 of every cell (6 configs x 2 tasks) | 12 | 1 day |
| 2 | Run #2 of every cell | 12 | 1 day |

Each batch is a complete replicate, so batch 1 alone already yields every
curve. Monitor kcoral queueing during batch 1: the pool has ~8-12 GPUs, so
12 agents benchmarking at once can queue (see "Benchmark latency").

## Metric

One plot per task: x = cumulative tokens (all Humanize agents combined),
y = best-so-far passing speedup. Thin lines per run, bold line per
configuration mean. Data points come from frontier commits; re-score every
frontier commit post hoc on the same pool so all numbers share one
measurement setup.

## Steps

### Phase 0 — Infrastructure

1. `evolution/cli.py`: `TOOLSET_NAME = "kda_flow"` is hard-coded. Add
   `--tools on|off` and `--kb on|off`; record them in `manifest.json`.
   Language is selected through the task YAML.
2. Toolsets: keep `kda_flow` (full); add `no_tools` (drop
   `tirx-debug-kernel`, `tirx-profile-kernel`) and `no_kb` (drop `tirx-wiki`,
   `KernelWiki`, skip reference fetching). (`bare` later, for row 4.)
3. Tools-off guard (`preparation/guards.py`): block `python -m tirx_harness`
   (NumSim / Synccheck / Racecheck / dump) and the IKET entry points, so the
   agent cannot rediscover them from the venv.
4. KB-off bans: ban the installed `tirx_kernels` package except `tirx_lite`,
   and the fetched references.
5. `evolution/prompts/render.py`: `kernel_search_references` and
   `kernel_research_debugging_contract` name the skills directly; branch on
   T/KB so the prompt never mentions removed skills.
6. Language task variants: `kda_forward_{cutedsl,cuda,triton}.yaml`,
   `mla_dsv4_{cutedsl,cuda,triton}.yaml` with `kernel_authoring: task`.
   Remove "no Triton, CUTE DSL, raw CUDA" from their specs
   (`kda_forward.yaml:64`, `mla_dsv4.yaml:52`), keeping the PyTorch /
   baseline-library bans.
7. Same-language leak bans:
   - kda: the correctness reference is FLA's Triton `chunk_kda` — Triton cell
     must not import `fla` in production and must not read its source.
     FlashKDA (CUTLASS) source banned for CUDA/CuTeDSL.
   - mla: trtllm / FlashInfer sparse MLA sources banned.
8. Token logging: keep complete transcripts for every Humanize agent; write a
   parser emitting (timestamp, cumulative tokens) and join with frontier
   commits.

### Phase 1 — Smoke test

All 6 configurations on `kda_forward` for ~1 hour each. Check:

- generated `PROMPT.md` files differ only where intended;
- tools-off cells cannot invoke our tools;
- KB-off cells cannot read banned `tirx_kernels` / references;
- language-variant cells produce kernels in their language;
- transcripts and token logs are complete.

### Phase 2 — Main runs

Batch 1, then batch 2 (12 runs each, 1 day each).

### Phase 3 — Analysis

Re-score frontier commits, build token-vs-speedup curves, decide on row 4.

## Phase 0 — what was implemented

Branch `ablation-study` (3 commits on top of `bf7b0e0`). `evolve init` checks
out the committed HEAD, so every harness change must be committed there.
`ablation/` itself stays untracked so the plan never reaches a run worktree.

| Piece | Where |
| --- | --- |
| `--toolset` flag; `our_tools` / `knowledge_base` recorded in manifest | `evolution/cli.py` |
| Toolset switches | `evolution/preparation/declare.py` |
| Tools-off: delete `tirx_harness` + `bin/tirx-harness` from the run venv, ban `tools/**` | `environment.py`, `toolsets/ablation_no_tools.yaml` |
| KB-off: delete everything in installed `tirx_kernels` except `tirx_lite`; drop tirx-wiki/KernelWiki; ban `docs/**` | `environment.py`, `toolsets/ablation_no_kb.yaml` |
| `site-packages/...` bans for any installed package | `environment.py` |
| Prompts mention only the tools/references a run has | `evolution/prompts/render.py` |
| Language cells | `evolution/tasks/{kda_forward,mla_dsv4}_{triton,cuda,cutedsl}.yaml` (`kernel_authoring: task`, toolset `ablation_no_tools`) |
| Launcher (init / tmux + `hmz exec -f ralph_loop -a agent=claude/claude-opus-5-5:max`) | `ablation/launch.py`, `ablation/configs.py` |
| Token-vs-speedup collector + plot | `ablation/collect.py` |

Leaks found and fixed while building it (apply to every cell):

- `mla_dsv4.yaml` had `banned_paths: []` while `tirx_kernels/mla/mla_dsv4_multishape.py`
  is this workload's own TIRx solution. Now banned (plus its bench config and test).
- The hmz sandbox can read the whole filesystem. The home directory (other
  checkouts such as `KDA-internal`, `kda-for-kda-release`, `experiments/`, and other
  runs' transcripts under `~/.hmz/epics`) is now banned; only the run directory and
  the venv's base interpreter stay allowed.

Same-language near-solution bans (only in that language's cell):

| Cell | Banned |
| --- | --- |
| kda Triton | FLA `fla/ops/kda`, `fla/layers/kda.py` (wiki checkout + venv) |
| kda CUDA, CuTeDSL | FlashKDA checkout (CUTLASS C++) |
| mla CUDA | flashinfer `sparse_mla_sm120` DSV4 CUDA headers |
| mla CuTeDSL | flashinfer `cute_dsl/attention/dsa`, `wrappers/batch_hca.py` (DSV4 HCA) |

Known limits: the banned-path hook inspects tool arguments, so a script that builds
a banned path at runtime is not stopped (pre-existing harness behavior). With KB
off, the tool skills still reference `tirx-wiki` and banned canonical kernels as
examples (e.g. IKET's FA4 annotation example).

Transcripts: hmz keeps each session under
`~/.hmz/epics/<encoded worktree>/<epic>/sessions/claude/projects/`.

## Phase 1 — smoke results (kda_forward, ~1 h each, 2026-11-16 21:18)

| Cell | Tokens (M) | Output (M) | Best frontier speedup | Our tools used | KB used | Language in candidates |
| --- | --- | --- | --- | --- | --- | --- |
| full | 48.8 | 0.32 | 0.820 | IKET (tools skill not yet invoked) | tirx-wiki | TIRx-lite |
| notools | 38.8 | 0.34 | 0.628 | none (never attempted) | — | TIRx-lite |
| nokb | 34.0 | 0.32 | — | NumSim 17, Synccheck 7, Racecheck 7, dump 4 | none | TIRx-lite |
| cutedsl | 30.7 | 0.28 | 0.291 | IKET | — | CuTeDSL |
| cuda | 25.7 | 0.39 | — | NCU | KernelWiki | CUDA via NVRTC |
| triton | 14.6 | 0.26 | — | NCU | KernelWiki | Triton |

Speedups are the agents' own frontier records; final numbers will be re-scored.
Plot: `ablation/smoke-curve.png`.

Checks:
- Generated prompts differ only in the tools/KB sections, language rule and bans.
- Tools-off venvs have no `tirx_harness`; KB-off venvs keep only `tirx_lite`.
- Remote scoring works from every ablated venv (kda 24/24, mla 46/46 on baseline).
- MLA: `tirx_kernels/mla` removed in every cell; CuTeDSL/CUDA same-language sources removed.
- No candidate used FLA/FlashKDA for production (`fla` imports are the harness's own
  baseline copies or the agent's baseline timing scripts).
- Hook false positives seen: a lone `~` (CUDA bitwise NOT) — fixed; `cd sub && ../..`
  resolved against the worktree root — pre-existing harness behavior, agents recover.
  Agents also try to write Claude auto-memory under `~/.claude/projects` via Bash, which
  the home ban blocks (the Write tool still works; uniform across cells).

Issues fixed after the smoke runs:
- hmz budgets are graceful by default: the ralph round in progress (a whole session)
  finishes after the deadline, so all six overran 1 h. Launcher now passes
  `budget.graceful=false`, so every run stops at exactly its duration.
- The collector reads agent-named speedup fields (`speedup_geomean`, ...).

Token rate: ~15-50M tokens/h per run (mostly cache reads), ~0.3M output/h.

## Benchmark latency

Measured 2026-11-16 on the idle GB200 pool (`127.0.0.1:9000`), scoring the
baseline as the candidate (`kcoral_remote.py <task> baseline`). Wall-clock
includes upload, worker setup, correctness gates and timing; it excludes
candidate JIT compilation, so real candidates are slower.

| Task (adapter key) | Rows | Wall-clock |
| --- | --- | --- |
| `kda/forward` (multi-shape) | 12 timed + 12 correctness-only probes | 349 s |
| `kda/forward_b1_t8192_h96` (single) | 1 timed + 9 probes | 32 s |
| `mla_dsv4` (multi-shape) | 46 timed | 120 s |
| `mla_dsv4_prefill_b2_..._hnd_varlen` (single) | 1 timed | 14 s |

Multi-shape is ~11x (kda) and ~9x (mla) slower per benchmark call.

Pool load estimate for 12 concurrent runs on ~8 GPUs, assuming each agent
benchmarks every ~10 min (kda) / ~5 min (mla): 6 x 349/600 + 6 x 120/300
= ~6 GPUs busy from official scoring alone, before agent-side debug and
profiling requests. The pool will be close to saturation, so queueing will
add to the measured latency.

## Changes after Phase 1 review (2026-11-17)

- Language cells ban exactly what the TIRx cells ban; the same-language
  near-solution bans (FlashKDA, FLA kda, flashinfer sm120 sparse MLA / DSV4 HCA)
  are removed. The spec still forbids calling FlashKDA/FLA (kda) or FlashInfer
  (mla) for production computation.
- The whole-home ban is replaced by `evolve init --ban DIR` with
  `configs.HOME_BANS` = ~/KDA-internal, ~/kda-for-kda-release, ~/experiments,
  ~/.hmz. The hook also allows this run's own ~/.claude/projects/<key> and
  ~/.hmz/epics/<key> (Claude Code's saved tool outputs and memory).
- launch.py sets TMPDIR to <worktree>/.local/tmp.
- Replaying all 623 smoke tool calls: 3 blocked (all the pre-existing
  `cd sub && ../../.venv/bin/python` relative-path false positive), down from 9.

## Phase 1 re-smoke (smoke2, 2026-11-17 00:42–01:42 UTC, kda, PT1H)

| cell | tokens | best frontier geomean | hook blocks |
|---|---|---|---|
| full | 37.8M | — (v1–v3 in scratch, no frontier commit) | 0 |
| notools | 35.1M | 0.678 | 0 |
| nokb | 30.1M | 0.349 | 1 (relative-path FP) |
| cutedsl | 34.3M | 0.235 | 1 (relative-path FP) |
| cuda | 25.7M | 1.261 (official re-score 1.273, 24/24) | 0 |
| triton | 18.4M | — (v0–v4 in scratch) | 0 |

All runs stopped at exactly 60 min (graceful=false works). collect.py fixed to
take a candidate's geomean and ignore its per-case speedups.

## Batch 1 (started 2026-11-17 04:08 UTC, P1D)

Per user: batch 1 = all 6 kda cells x 2 replicates (abl-kda-<cell>-r1/r2);
batch 2 = all 6 mla cells x 2 replicates. Prompt kept as is (no forced use of
our tools); collect.py reports per-run tool use to show whether tools were used.

## Batch 3 (2026-11-18/19 UTC, P1D, kda full and no-KB, no hmz fence)

The hmz Landlock fence (host kernel 5.15 = Landlock ABI 1) turns every
cross-directory rename into EXDEV, so in batch 1 the NumSim build failed for
full and no-KB (see `TOOL_ISSUES.md` #3). Batch 3 reran those two cells with
`launch.py start --flow open` (`flows/ralph_open`, no fence). Isolation was
per-run cache env vars plus a prompt rule on temp files; hourly transcript
checks found no cheating. `abl3-kda-full-r1/r2` were stopped at 12.9 h by
decision (unlucky start) and rerun as r3/r4; their partial curves are kept.

## Results so far (kda_forward)

Best passing geomean speedup vs FlashKDA at the end of each 1-day run. Curve
data: `results/kda/<run>.json` (copies of each run's `curve.json`); plots:
`python ablation/plot.py` -> `abl-kda-{all,best,mean}.png`.

| Config | Runs | Final r1 / r2 | Mean |
| --- | --- | --- | --- |
| TIRx full | batch 3 r3 / r4 | 2.788 / 2.946 | 2.87 |
| TIRx no-KB | batch 3 r1 / r2 | 2.792 / 2.945 | 2.87 |
| CUDA | batch 1 | 2.470 / 2.940 | 2.70 |
| TIRx no-tools | batch 1 | 2.541 / 2.611 | 2.58 |
| CuTeDSL | batch 1 | 2.505 / 2.198 | 2.35 |
| Triton | batch 1 | 1.759 / 1.239 | 1.50 |
| (TIRx full, batch 1, fence) | batch 1 | 2.980 / 2.376 | 2.68 |
| (TIRx no-KB, batch 1, fence) | batch 1 | 2.498 / 2.177 | 2.34 |

Two runs per cell and a 0.3-0.5x spread between replicates in several cells:
differences between the top cells are not yet significant. Phase 3
(re-scoring frontiers, new inputs) is still to do.
