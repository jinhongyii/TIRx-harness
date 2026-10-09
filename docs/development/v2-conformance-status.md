---
orphan: true
---

# Conformance status

Every canonical corpus case (`tests/numsim/corpus/canonical_cases.py`) runs in
the numsim, racecheck and synccheck modes and is compared with its snapshot in
`tirx_harness/tests/conformance/snapshots/`, the NumSim oracle (policy in that
directory's README). Matrix at 79f04eb, 2026-10-08:

| mode | match | fail |
| --- | --- | --- |
| numsim | 101 | 0 |
| racecheck | 101 | 0 |
| synccheck | 101 | 0 |

Regenerate from `tirx_harness/` with:

```bash
$PY ../scripts/numsim-v2/status.py --run -n 16
```
