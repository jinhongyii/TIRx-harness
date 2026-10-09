"""Schedule-independent conformance snapshots for the canonical kernel corpus.

One JSON file per (case, mode) under ``snapshots/<case>/<mode>.json``. A
snapshot records only facts every correct NumSim implementation must
reproduce bit-for-bit:

* ``numsim``: per output buffer, dtype, shape, and the sha256 of its bytes;
  whether the outputs match the case's independent reference; the verdict;
  and the diagnostic groups (kind, status, source anchors, byte footprint).
* ``racecheck`` / ``synccheck``: per launch phase, the verdict and the
  normalized finding set. Each finding group is keyed by category (finding,
  advisory, incomplete, sync finding, ...), kind, status, memory space and
  sorted source anchors; byte overlaps of all members of a group are merged
  into one interval list.

Everything that depends on the schedule, the worker count, or the engine's
internal identifiers is dropped: message text, timings, stats, poll and
transition counts, occurrence counts, warp ids, per-warp sequence numbers,
loop iteration ordinals, internal op ids (they are resolved to source spans
first) and anonymous buffer names.

An engine failure (exception before any payload is produced) is recorded as
``{"error": "<ExceptionType>"}`` so a later change in behavior shows up as a
diff rather than as a silent skip.

The snapshots are the NumSim oracle (``tirx_harness.numsim``): ``pytest
tests/conformance --update-snapshots`` regenerates them, and every commit that
changes one cites a behaviour-delta row (``scripts/numsim-v2/
check_snapshot_deltas.py``) or carries a ``Snapshot-Regen: schema <reason>``
trailer (see README.md).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 4  # 2: no window alloc ids; 3: register reads -> presence; 4: tmem_lifetime_review -> kind + anchors
MODES = ("numsim", "racecheck", "synccheck")
# ``NUMSIM_SNAPSHOT_ROOT`` points the suite at another snapshot tree.
SNAPSHOT_ROOT = Path(os.environ.get("NUMSIM_SNAPSHOT_ROOT") or Path(__file__).resolve().parent / "snapshots")

# Payload lists that carry diagnostics, by mode. ``sync`` is Racecheck's
# embedded synchronization-protocol sub-report.
_DIAGNOSTIC_LISTS = ("findings", "advisories", "incomplete")


# --------------------------------------------------------------------------
# Snapshot files


def snapshot_path(case_name: str, mode: str, root: Path = SNAPSHOT_ROOT) -> Path:
    if mode not in MODES:
        raise ValueError(f"unknown conformance mode {mode!r}")
    return root / case_name / f"{mode}.json"


def dumps(snapshot: Mapping[str, Any]) -> str:
    return json.dumps(snapshot, indent=1, sort_keys=True) + "\n"


def load_snapshot(case_name: str, mode: str, root: Path = SNAPSHOT_ROOT) -> dict[str, Any] | None:
    path = snapshot_path(case_name, mode, root)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_snapshot(
    case_name: str, mode: str, snapshot: Mapping[str, Any], root: Path = SNAPSHOT_ROOT
) -> Path:
    path = snapshot_path(case_name, mode, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(dumps(snapshot))
    tmp.replace(path)
    return path


def diff_snapshots(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> str:
    import difflib

    return "".join(
        difflib.unified_diff(
            dumps(expected).splitlines(keepends=True),
            dumps(actual).splitlines(keepends=True),
            fromfile="snapshot",
            tofile="actual",
            n=2,
        )
    )


# --------------------------------------------------------------------------
# Source anchors


def _source_name(name: str) -> str:
    """Strip host-specific prefixes from a source file name."""

    for marker in ("site-packages/", "dist-packages/"):
        if marker in name:
            return name.rsplit(marker, 1)[1]
    for marker in ("/tirx_harness/tests/", "/tirx_harness/src/"):
        if marker in name:
            return marker.strip("/").split("/", 1)[1] + "/" + name.rsplit(marker, 1)[1]
    return os.path.basename(name)


def format_span(span: Any) -> str | None:
    """Render a serialized SourceSpan / SequentialSourceSpan dict as one anchor."""

    if not isinstance(span, Mapping):
        return None
    if span.get("kind") == "sequential":
        parts = [format_span(child) for child in span.get("spans", ())]
        parts = [part for part in parts if part]
        return " > ".join(parts) if parts else None
    if span.get("kind") == "span" or {"source_name", "line"} <= set(span):
        return (
            f"{_source_name(str(span['source_name']))}:{span['line']}:{span.get('column', 0)}"
            f"-{span.get('end_line', span['line'])}:{span.get('end_column', 0)}"
        )
    return None


class SourceResolver:
    """Maps ``(kernel_index, site)`` pairs (a record's ``source_op_id``) to the
    site's source anchor. A site without a span adds no anchor."""

    def __init__(self, module: Any):
        self._sites: dict[tuple[int, int], str] = {}
        for kernel in getattr(getattr(module, "spec", None), "kernels", ()) or ():
            for site in range(len(kernel.sites)):
                anchor = format_span(kernel.source_span(site))
                if anchor is not None:
                    self._sites[(kernel.index, site)] = anchor

    def resolve(self, kernel_index: Any, source_op_id: Any) -> str | None:
        if not isinstance(kernel_index, int) or isinstance(kernel_index, bool):
            return None
        if not isinstance(source_op_id, int) or isinstance(source_op_id, bool):
            return None
        return self._sites.get((kernel_index, source_op_id))


def collect_anchors(value: Any, resolver: SourceResolver | None, *, kernel_index: int | None = None) -> set[str]:
    """All source anchors referenced anywhere inside one diagnostic record."""

    anchors: set[str] = set()
    if isinstance(value, Mapping):
        kernel_index = value.get("kernel_index", kernel_index)
        if "source_op_id" in value and resolver is not None:
            anchor = resolver.resolve(kernel_index, value["source_op_id"])
            if anchor is not None:
                anchors.add(anchor)
        for key in ("source_span", "span"):
            anchor = format_span(value.get(key))
            if anchor is not None:
                anchors.add(anchor)
        for key, child in value.items():
            if key in {"source_span", "span"}:
                continue
            anchors |= collect_anchors(child, resolver, kernel_index=kernel_index)
    elif isinstance(value, (list, tuple)):
        for child in value:
            anchors |= collect_anchors(child, resolver, kernel_index=kernel_index)
    return anchors


# --------------------------------------------------------------------------
# Byte footprints


def _intervals(value: Any) -> Iterable[tuple[str, int, int]]:
    """Yield ``(region, start, end)`` byte intervals referenced by a record.

    ``overlaps`` lists (racecheck) and a top-level ``byte_offset/byte_len``
    pair (NumSim diagnostics) are understood. The region is the memory space
    plus allocation id when present.
    """

    if not isinstance(value, Mapping):
        return
    space = record_space(value)
    overlaps = value.get("overlaps")
    if isinstance(value.get("overlap"), Mapping):
        overlaps = [value["overlap"]]
    if isinstance(overlaps, list):
        for item in overlaps:
            if isinstance(item, Mapping) and "byte_offset" in item:
                start = int(item["byte_offset"])
                end = int(item.get("byte_end", start + int(item.get("byte_len", 0))))
                alloc = item.get("allocation_id", value.get("allocation_id"))
                yield _region(space, alloc), start, end
    elif "byte_offset" in value and "byte_len" in value:
        start = int(value["byte_offset"])
        alloc = value.get("allocation", value.get("allocation_id"))
        yield _region(space, alloc), start, start + int(value["byte_len"])


# Canonical space spellings in snapshots (numsim-core says ``reg``).
_SPACE_ALIASES = {"reg": "register"}


def record_space(value: Mapping[str, Any]) -> str | None:
    """Memory space of a record, falling back to its witness accesses."""

    if value.get("space") is not None:
        return _SPACE_ALIASES.get(str(value["space"]), str(value["space"]))
    spaces = {
        _SPACE_ALIASES.get(str(side["space"]), str(side["space"]))
        for side in (value.get("prior"), value.get("current"))
        if isinstance(side, Mapping) and side.get("space") is not None
    }
    return "+".join(sorted(spaces)) if spaces else None


# Spaces with one window per CTA: the allocation id is an engine-internal
# numbering, so only the space is kept. Global allocations keep their id
# (host parameter order).
_WINDOW_SPACES = frozenset({"shared", "tmem", "local", "register", "param"})


def _region(space: Any, alloc: Any) -> str:
    region = str(space) if space is not None else "?"
    if alloc is None or region.split("+")[0] in _WINDOW_SPACES:
        return region
    return f"{region}#{alloc}"


def merge_intervals(items: Iterable[tuple[int, int]]) -> str:
    """Merge byte intervals and render them compactly as ``"a-b,c-d"`` (half-open)."""

    merged: list[list[int]] = []
    for start, end in sorted(items):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return ",".join(f"{start}-{end}" for start, end in merged)


# --------------------------------------------------------------------------
# Diagnostic normalization


# Fields that carry a semantic classification and are stable across
# schedules. Everything else is dropped.
# One TMEM lane spans 512 columns of 4 bytes in the engine's byte addressing.
TMEM_ROW_BYTES = 512 * 4

_SEMANTIC_FIELDS = ("access_pair", "ordering_domain", "ordering_failure", "reason", "cause")


def normalize_records(
    records: Iterable[tuple[str, Mapping[str, Any]]],
    resolver: SourceResolver | None,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for category, record in records:
        if not isinstance(record, Mapping):
            record = {"kind": str(record)}
        default_status = {"advisories": "review", "incomplete": "incomplete"}.get(category, "error")
        key_fields: dict[str, Any] = {
            "category": category,
            "kind": str(record.get("kind", category)),
            "status": str(record.get("status", default_status)),
        }
        space = record_space(record)
        if space is not None:
            key_fields["space"] = space
        for field in _SEMANTIC_FIELDS:
            value = record.get(field)
            if isinstance(value, (str, int, bool)) and not isinstance(value, float):
                key_fields[field] = value
        key_fields["anchors"] = sorted(collect_anchors(record, resolver))
        # Projection rule (README): which dynamic instance of a static
        # (load site, store site) pair witnesses a TMEM lifetime conflict is
        # schedule-dependent, so `tmem_lifetime_review` compares kind and
        # anchors only (W5, CONTRACT_REQUESTS).
        tmem_review = key_fields["kind"] == "tmem_lifetime_review"
        register_read = key_fields["kind"] == "uninitialized_read" and key_fields.get("space") == "register"
        if register_read:
            # Projection rule (README, coordinator ruling on V2C-20): register
            # byte numbering and the reading site are engine details; compare
            # only that such reads exist, per (category, kind, status).
            key_fields["anchors"] = []
        key = json.dumps(key_fields, sort_keys=True)
        group = groups.setdefault(key, {**key_fields, "_bytes": {}})
        if register_read or tmem_review:
            continue
        columns = record.get("tmem_columns")
        # numsim-core states TMEM footprints as exact column ranges
        # (`tmem_columns = [[lo, hi], ...]` in 4-byte columns; a single
        # `[lo, hi]` is accepted too); its byte spans are taddr-encoded.
        if isinstance(columns, (list, tuple)) and len(columns) == 2 and all(isinstance(c, int) for c in columns):
            columns = [columns]
        explicit_columns = isinstance(columns, (list, tuple)) and bool(columns) and all(
            isinstance(r, (list, tuple)) and len(r) == 2
            and all(isinstance(c, int) and not isinstance(c, bool) for c in r)
            for r in columns
        )
        if explicit_columns:
            for lo, hi in columns:
                group["_bytes"].setdefault("tmem-columns", []).append((lo * 4, hi * 4))
        for region, start, end in _intervals(record):
            if region.startswith("tmem") and explicit_columns:
                continue
            if region.startswith("tmem"):
                # TMEM offsets are lane * row + column. Which warp (lane
                # quadrant) witnesses a conflict depends on the schedule, so
                # keep only the column footprint.
                region = region.replace("tmem", "tmem-columns", 1)
                row = start - start % TMEM_ROW_BYTES
                start, end = start - row, end - row
            group["_bytes"].setdefault(region, []).append((start, end))
    result = []
    for key in sorted(groups):
        group = groups[key]
        footprint = {region: merge_intervals(items) for region, items in sorted(group.pop("_bytes").items())}
        if footprint:
            group["bytes"] = footprint
        result.append(group)
    return result


def _payload_records(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any]]] = []
    for field in _DIAGNOSTIC_LISTS:
        for item in payload.get(field) or ():
            records.append((field, item))
    sync = payload.get("sync")
    if isinstance(sync, Mapping):
        for field in _DIAGNOSTIC_LISTS:
            for item in sync.get(field) or ():
                records.append((f"sync.{field}", item))
    error = payload.get("execution_error")
    if isinstance(error, Mapping):
        records.append(("execution_error", error))
    elif error is not None:
        records.append(("execution_error", {"kind": str(error)}))
    return records


def normalize_analysis_phase(payload: Mapping[str, Any], resolver: SourceResolver | None) -> dict[str, Any]:
    return {
        "verdict": str(payload.get("verdict")),
        "diagnostics": normalize_records(_payload_records(payload), resolver),
    }


def hash_array(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
    }


def normalize_numsim(
    outputs: Mapping[str, Any],
    diagnostics: Iterable[Mapping[str, Any]],
    *,
    reference_ok: bool | None,
    resolver: SourceResolver | None,
) -> dict[str, Any]:
    diagnostics = list(diagnostics)
    verdict = "review" if any(item.get("status") == "review" for item in diagnostics) else "clean"
    return {
        "verdict": verdict,
        "reference_ok": reference_ok,
        "outputs": {name: hash_array(outputs[name]) for name in sorted(outputs)},
        "diagnostics": normalize_records((("diagnostics", item) for item in diagnostics), resolver),
    }


def _synccheck_limits(numsim: Any, max_diagnostic_bytes: int) -> Any:
    # Same budget as the synccheck corpus gate
    # (tests/analysis_tools/synccheck/corpus/test_tirx_kernels_synccheck.py).
    return numsim.ResourceLimits(
        max_schedules=16,
        max_backtrack_nodes=100_000,
        max_events_per_run=4_000_000,
        max_total_events=8_000_000,
        max_loop_steps=4_000_000,
        max_wall_time_ms=180_000,
        max_diagnostic_bytes=max_diagnostic_bytes,
    )


def _run_numsim_case(numsim: Any, entry: Any) -> dict[str, Any]:
    """Mirror ``numsim.run_case`` but keep the raw outputs for hashing.

    The reference is computed before the run; ``Engine.run`` never mutates
    its inputs (v2 binder rule), so the arguments it sees are unchanged."""

    case = entry.prepare()
    engine = numsim.Engine(max_workers=entry.engine_max_workers)
    module = numsim.transpile(case.kernel)
    expected = copy.deepcopy(case.reference())
    result = engine.run(
        module,
        case.args,
        subset=case.subset,
        assumptions=case.assumptions,
        outputs=case.outputs,
    )
    reference_ok = bool(numsim.compare(result, expected, tolerances=case.comparisons).ok)
    return normalize_numsim(
        result.outputs,
        result.diagnostics,
        reference_ok=reference_ok,
        resolver=SourceResolver(module),
    )


def _run_analysis_case(numsim: Any, entry: Any, mode: str) -> dict[str, Any]:
    # Synccheck limits mirror the former corpus gate budget.
    case = entry.prepare()
    engine = numsim.Engine(max_workers=entry.engine_max_workers)
    module = numsim.transpile(case.kernel)
    resolver = SourceResolver(module)
    phases = []
    for phase_index in range(len(module.spec.kernels)):
        if mode == "racecheck":
            result = engine.run_racecheck_phase(
                module,
                case.args,
                phase_index=phase_index,
                subset=case.subset,
            )
        else:
            result = engine.run_synccheck_phase(
                module,
                case.args,
                phase_index=phase_index,
                subset=case.subset,
                coverage_bounds=numsim.CoverageBounds(
                    max_warp_preemptions=0,
                    max_completion_schedule_deviations=0,
                ),
                resource_limits=_synccheck_limits(numsim, entry.synccheck_max_diagnostic_bytes),
            )
        phases.append(normalize_analysis_phase(result.to_dict(), resolver))
    return {"phases": phases}


def collect_snapshot(
    entry: Any,
    mode: str,
    numsim: Any = None,
    *,
    reraise: tuple[type[BaseException], ...] = (),
) -> dict[str, Any]:
    """Run one canonical case in one mode and return its normalized snapshot.

    ``numsim`` defaults to ``tirx_harness.numsim``. Exceptions are recorded
    as ``{"error": type}`` except those in ``reraise``.
    """

    if numsim is None:
        from tirx_harness import numsim
    header = {"schema": SCHEMA_VERSION, "case": entry.name, "mode": mode}
    try:
        if mode == "numsim":
            body = _run_numsim_case(numsim, entry)
        elif mode in ("racecheck", "synccheck"):
            body = _run_analysis_case(numsim, entry, mode)
        else:
            raise ValueError(f"unknown conformance mode {mode!r}")
    except reraise:
        raise
    except Exception as error:  # noqa: BLE001 - recorded as part of the oracle
        body = {"error": type(error).__name__}
    return {**header, **body}


__all__ = [
    "MODES",
    "SCHEMA_VERSION",
    "SNAPSHOT_ROOT",
    "SourceResolver",
    "collect_snapshot",
    "diff_snapshots",
    "dumps",
    "load_snapshot",
    "normalize_analysis_phase",
    "normalize_numsim",
    "write_snapshot",
]
