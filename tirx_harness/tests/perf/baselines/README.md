# Relative performance baselines

Absolute second limits hold only on the machine where someone measured them.
For the NumSim redesign, performance checks compare against a **baseline
recorded on the same class of host** and allow a relative tolerance:

```
elapsed <= baseline_seconds * (1 + tolerance)
```

## Host classes

A host class names hosts whose timings are directly comparable: the same CPU
model and the same logical CPU count. `perf_baseline.host_class()` derives it
as `<cpu-model-slug>-<nproc>c`, for example
`amd-epyc-7763-64-core-processor-256c`. Set `NUMSIM_PERF_HOST_CLASS` to choose a
different file, for example when a CI runner pool reports varying model
strings.

Each class has one file, `baselines/<host-class>.json`:

```json
{
  "host_class": "amd-epyc-7763-64-core-processor-256c",
  "cpu_model": "AMD EPYC 7763 64-Core Processor",
  "nproc": 256,
  "recorded": "2026-10-07",
  "commit": "c3eac62",
  "preflight": "load average / idle observed right before the run",
  "pytest": "-n 16 --dist=worksteal -m performance",
  "tolerance": 0.10,
  "baselines": {
    "<metric name>": {"seconds": 12.3, "samples": [12.1, 12.3], "workload": "..."}
  }
}
```

A metric can override the file-wide `tolerance` with its own `tolerance`.
`seconds` is the maximum of the recorded samples.

## Using a baseline in a test

```python
from tests.perf.perf_baseline import assert_within_baseline

@pytest.mark.performance
def test_something_fast_enough():
    ...  # prepare and build outside the timed region
    started = perf_counter()
    run()
    assert_within_baseline("numsim.mega_moe.max_config", perf_counter() - started)
```

On a host without a baseline file, or when the file has no entry for the
metric, the check is skipped with a message naming the missing file.

## Recording or refreshing a baseline

Follow the preflight in `tests/CLAUDE.md`: record `nproc`, the load average and
instantaneous CPU idle, and do not measure while the host is busy. Run the
performance tests at the concurrency the test documents, take the maximum of
at least one clean run (more samples are better), and write the file with the
date, commit, and preflight readings. A baseline may only go up with a
written justification, exactly like the absolute thresholds today.

## Status

The existing absolute-threshold tests
(`tests/numsim/corpus/test_canonical_kernels.py::test_maximum_mega_moe_numsim_completes_within_performance_budget`
and
`tests/analysis_tools/racecheck/corpus/test_canonical_kernels_racecheck.py::test_real_mega_moe_racecheck_completes_within_performance_budget`)
stay unchanged until the legacy engine is deleted. Their metrics are recorded
here so the v2 engine can be held to the same workloads relatively.
