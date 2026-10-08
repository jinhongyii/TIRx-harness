# compute-sanitizer

NVIDIA **compute-sanitizer** runs on the real GPU. Use its `memcheck`,
`racecheck`, `initcheck`, and `synccheck` tools to detect hardware-visible
memory access, initialization, data race, and synchronization failures.

## When to use

- Invalid, misaligned, or out-of-bounds memory accesses; need `memcheck`
- Reads from uninitialized device memory; need `initcheck`
- Wrong results that look like data races; need hardware `racecheck`
- Invalid synchronization usage or a device hang; need `synccheck`

## Relation to pre-GPU analysis

| Concern | Prefer first | Device follow-up if needed |
|---------|--------------|------|
| Barrier protocol / deadlock | [synccheck](synccheck.md) (fast, pre-GPU) | `compute-sanitizer --tool synccheck` |
| Data race on device memory | [racecheck](racecheck.md) (global/SMEM/TMEM model) | `compute-sanitizer --tool racecheck` |
| Invalid or out-of-bounds memory access | — | `compute-sanitizer --tool memcheck` |
| Uninitialized device memory access | — | `compute-sanitizer --tool initcheck` |

Pre-GPU checkers reason about TIRx IR and a modeled barrier/memory system.
Compute Sanitizer observes the launched CUDA binary. Start synchronization and
race investigations with the fast model; use sanitizer when the pre-GPU
checkers do not catch the issue.

## Typical commands

Use the resolved launch command with an explicit working directory and
environment. User-provided values are authoritative; when they are omitted,
`$tirx-debug-kernel` first discovers the current task's existing test or
benchmark launch rather than inventing one:

```bash
# Invalid, misaligned, or out-of-bounds memory access
compute-sanitizer --tool memcheck <launch-command> [args...]

# Uninitialized device memory access
compute-sanitizer --tool initcheck <launch-command> [args...]

# Barrier / deadlock on device
compute-sanitizer --tool synccheck <launch-command> [args...]

# Race detection on device
compute-sanitizer --tool racecheck <launch-command> [args...]
```

Useful extras (see `compute-sanitizer --help`):

- `--launch-timeout <sec>` for suspected hangs
- `--target-processes all` when the workload spawns helpers
- Filter kernels if the process launches many unrelated kernels

## Complements

- [synccheck](synccheck.md) — CPU-side barrier check; sanitizer is the GPU-side follow-up for hangs
- [racecheck](racecheck.md) — CPU-side global/SMEM/TMEM data-race check; sanitizer observes the launched GPU binary
