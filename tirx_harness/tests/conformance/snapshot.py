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

An implementation failure (exception before any payload is produced) is
recorded as ``{"error": "<ExceptionType>"}`` so a later change in behavior
shows up as a diff rather than as a silent skip.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2  # 2: no allocation id for per-CTA window spaces
MODES = ("numsim", "racecheck", "synccheck")
SNAPSHOT_ROOT = Path(__file__).resolve().parent / "snapshots"
IMPL_ENV = "NUMSIM_IMPL"
IMPLS = ("legacy", "v2")

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


def delta_snapshot_path(case_name: str, mode: str, root: Path = SNAPSHOT_ROOT) -> Path:
    return root / case_name / f"{mode}.delta.json"


def load_expected(case_name: str, mode: str, impl_name: str, root: Path = SNAPSHOT_ROOT) -> dict[str, Any] | None:
    """The oracle for ``impl_name``.

    Legacy always compares with ``<mode>.json`` (legacy output). A new
    implementation compares with ``<mode>.delta.json`` when present: the
    legacy snapshot corrected by a behaviour-delta row (its ``delta`` field
    names the row and is not part of the comparison)."""

    if impl_name != "legacy":
        path = delta_snapshot_path(case_name, mode, root)
        if path.exists():
            data = json.loads(path.read_text())
            data.pop("delta", None)
            return data
    return load_snapshot(case_name, mode, root)


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
    """Maps legacy ``(kernel_index, source_op_id)`` pairs to source anchors."""

    def __init__(self, module: Any):
        self._sites: dict[tuple[int, int], str] = {}
        spec = getattr(module, "spec", None)
        for kernel_index, kernel in enumerate(getattr(spec, "kernels", ()) or ()):
            for entry in getattr(kernel, "source_map", ()) or ():
                span = getattr(entry, "span", None)
                anchor = format_span(span.to_dict()) if span is not None else None
                if anchor is None:
                    # No span: fall back to the bounded IR text, which is
                    # still independent of internal op numbering.
                    anchor = f"<{entry.kind}> {entry.text}"
                self._sites[(kernel_index, entry.op_id)] = anchor

    def resolve(self, kernel_index: Any, source_op_id: Any) -> str | None:
        if not isinstance(kernel_index, int) or isinstance(kernel_index, bool):
            return None
        if not isinstance(source_op_id, int) or isinstance(source_op_id, bool):
            return None
        return self._sites.get((kernel_index, source_op_id), f"<unmapped op {source_op_id}>")


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


# numsim-core space names that legacy spelled differently.
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
# numbering (legacy counts per space, numsim-core counts globally), so only
# the space is kept. Global allocations keep their id (host parameter order
# in both engines).
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
        key = json.dumps(key_fields, sort_keys=True)
        group = groups.setdefault(key, {**key_fields, "_bytes": {}})
        columns = record.get("tmem_columns")
        # numsim-core states TMEM footprints as exact column ranges
        # (`tmem_columns = [[lo, hi], ...]` in 4-byte columns; an older single
        # `[lo, hi]` is accepted too); its byte spans are taddr-encoded and
        # not comparable to legacy's lane*2048+col*4.
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


def _regroup(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        key_fields = {k: v for k, v in group.items() if k != "bytes"}
        key = json.dumps(key_fields, sort_keys=True)
        target = merged.setdefault(key, {**key_fields, "_bytes": {}})
        for region, text in (group.get("bytes") or {}).items():
            for part in filter(None, text.split(",")):
                start, end = part.split("-")
                target["_bytes"].setdefault(region, []).append((int(start), int(end)))
    out = []
    for key in sorted(merged):
        group = merged[key]
        footprint = {r: merge_intervals(items) for r, items in sorted(group.pop("_bytes").items())}
        if footprint:
            group["bytes"] = footprint
        out.append(group)
    return out


def relax_unanchored(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Drop source anchors the legacy oracle could not record.

    Legacy NumSim/Racecheck emitted some diagnostics (notably
    ``uninitialized_read`` without a source op) with no source evidence, so
    their snapshot groups have ``anchors == []``. A new implementation that
    attaches the real source site is not wrong. For every
    (category, kind, status, space) whose expected groups are all
    unanchored, the actual groups lose their anchors and are re-merged;
    everything else (kinds, statuses, byte footprints) is still compared.
    Used only for non-legacy implementations.
    """

    def unanchored_keys(groups: list[dict[str, Any]]) -> set[tuple]:
        keys: dict[tuple, bool] = {}
        for g in groups:
            key = (g.get("category"), g.get("kind"), g.get("status"), g.get("space"))
            keys[key] = keys.get(key, True) and not g.get("anchors")
        return {k for k, unanchored in keys.items() if unanchored}

    def relax(exp_groups: list[dict[str, Any]], act_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keys = unanchored_keys(exp_groups)
        changed = [
            {**g, "anchors": []} if (g.get("category"), g.get("kind"), g.get("status"), g.get("space")) in keys else g
            for g in act_groups
        ]
        return _regroup(changed)

    actual = copy.deepcopy(actual)
    if "diagnostics" in expected and "diagnostics" in actual:
        actual["diagnostics"] = relax(expected["diagnostics"], actual["diagnostics"])
    for exp_phase, act_phase in zip(expected.get("phases") or (), actual.get("phases") or ()):
        act_phase["diagnostics"] = relax(exp_phase["diagnostics"], act_phase["diagnostics"])
    return actual


# --------------------------------------------------------------------------
# Implementations


@dataclass(frozen=True)
class Implementation:
    name: str
    numsim: Any  # module exposing ``transpile``, ``Engine``, ``CoverageBounds``...


class ImplementationUnavailable(RuntimeError):
    pass


def selected_impl_name() -> str:
    value = os.environ.get(IMPL_ENV, "legacy").strip() or "legacy"
    if value not in IMPLS:
        raise ValueError(f"{IMPL_ENV} must be one of {IMPLS}, got {value!r}")
    return value


def load_implementation(name: str | None = None) -> Implementation:
    """Return the NumSim implementation selected by ``NUMSIM_IMPL``.

    ``v2`` must expose the legacy public surface (``transpile``, ``Engine``
    with ``run``/``run_racecheck_phase``/``run_synccheck_phase``,
    ``CoverageBounds``, ``ResourceLimits``) per the redesign plan; payload
    diagnostics may reference source either through legacy
    ``(kernel_index, source_op_id)`` pairs resolvable via
    ``module.spec.kernels[i].source_map`` or through embedded serialized
    ``source_span`` dicts.
    """

    name = name or selected_impl_name()
    if name == "legacy":
        from tirx_harness import numsim

        return Implementation("legacy", numsim)
    try:
        from tirx_harness.numsim import v2  # type: ignore[attr-defined]
    except ImportError as error:
        raise ImplementationUnavailable(
            f"NUMSIM_IMPL=v2 but tirx_harness.numsim.v2 is not importable: {error}"
        ) from error
    missing = [name for name in _REQUIRED_SURFACE if not hasattr(v2, name)]
    if missing:
        raise ImplementationUnavailable(
            f"tirx_harness.numsim.v2 does not expose the conformance surface yet: missing {missing}"
        )
    return Implementation("v2", v2)


_REQUIRED_SURFACE = ("transpile", "Engine", "compare", "CoverageBounds", "ResourceLimits")


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
    """Mirror ``numsim.run_case`` but keep the raw outputs for hashing."""

    case = entry.prepare()
    engine = numsim.Engine(max_workers=entry.engine_max_workers)
    module = numsim.transpile(case.kernel)
    if hasattr(engine, "_prepare_execution"):
        from dataclasses import replace

        execution = engine._prepare_execution(
            module, case.args, outputs=case.outputs, assumptions=case.assumptions
        )
        execution = replace(execution, bindings=execution.bindings.freeze())
        try:
            expected = copy.deepcopy(case.reference())
        finally:
            execution.bindings.restore_host_buffers()
        result = engine._execute_prepared(module, execution, subset=case.subset)
    else:
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
    # Transpile/run options mirror the corpus gates:
    # tests/analysis_tools/{racecheck,synccheck}/corpus/.
    case = entry.prepare()
    engine = numsim.Engine(max_workers=entry.engine_max_workers)
    if mode == "racecheck":
        module = numsim.transpile(case.kernel, _analysis_capable=True, _analysis_checker="racecheck")
    else:
        module = numsim.transpile(
            case.kernel, _analysis_capable=True, _default_generated_opt_level=3
        )
    resolver = SourceResolver(module)
    phases = []
    for phase_index in range(len(module.spec.kernels)):
        if mode == "racecheck":
            result = engine.run_racecheck_phase(
                module,
                case.args,
                phase_index=phase_index,
                subset=case.subset,
                advance_prefix=True,
            )
        else:
            result = engine.run_synccheck_phase(
                module,
                case.args,
                phase_index=phase_index,
                subset=case.subset,
                advance_prefix=True,
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
    impl: Implementation,
    *,
    reraise: tuple[type[BaseException], ...] = (),
) -> dict[str, Any]:
    """Run one canonical case in one mode and return its normalized snapshot.

    Exceptions are recorded as ``{"error": type}`` except those in
    ``reraise`` (v2 re-raises ``NotImplementedError`` so unfinished engine
    bodies skip instead of failing).
    """

    header = {"schema": SCHEMA_VERSION, "case": entry.name, "mode": mode}
    try:
        if mode == "numsim":
            body = _run_numsim_case(impl.numsim, entry)
        elif mode in ("racecheck", "synccheck"):
            body = _run_analysis_case(impl.numsim, entry, mode)
        else:
            raise ValueError(f"unknown conformance mode {mode!r}")
    except reraise:
        raise
    except Exception as error:  # noqa: BLE001 - recorded as part of the oracle
        body = {"error": type(error).__name__}
    return {**header, **body}


__all__ = [
    "IMPL_ENV",
    "MODES",
    "SNAPSHOT_ROOT",
    "ImplementationUnavailable",
    "SourceResolver",
    "collect_snapshot",
    "diff_snapshots",
    "load_implementation",
    "load_expected",
    "load_snapshot",
    "normalize_analysis_phase",
    "normalize_numsim",
    "relax_unanchored",
    "selected_impl_name",
    "write_snapshot",
]
