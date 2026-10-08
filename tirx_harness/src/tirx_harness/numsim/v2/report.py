"""Payloads, findings and the one renderer for NumSim v2 (schema_version 5).

The engine returns ``report::Report`` JSON plus a run status; this module
turns them into the public payload dict (legacy-compatible field names:
``verdict``, ``findings``, ``advisories``, ``incomplete``,
``execution_error``) and the report objects
(``NumSimReport``/``RaceReport``/``SyncCheckReport``) with
``.verdict/.findings/.to_dict/.print/.require_clean``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 5
VERDICTS = ("clean", "review", "incomplete", "error")
_VERDICT_RANK = {name: rank for rank, name in enumerate(VERDICTS)}

# Engine kind name -> public (legacy-stable) kind name.
_KIND_NAMES = {"uninit_read": "uninitialized_read"}
# Kinds reported as advisories (review, never a proof of a defect) when a
# payload is built from raw ``report::Finding``s (the checkers' own
# ``serialize`` decides placement otherwise). The legacy ``alias_stale_read``
# is not listed: Racecheck no longer emits it (W5 decides whether it returns).
ADVISORY_KINDS = frozenset({"uninitialized_read", "undeclared_protocol_word", "cross_cta_async_order"})


def worst(verdicts: Iterable[str]) -> str:
    result = "clean"
    for verdict in verdicts:
        if _VERDICT_RANK[verdict] > _VERDICT_RANK[result]:
            result = verdict
    return result


def kind_name(raw: Any) -> str:
    """``FindingKind`` serde value -> public kind string."""

    if isinstance(raw, Mapping):  # FindingKind::Other(String) -> {"other": "..."}
        raw = next(iter(raw.values()), "other")
    name = str(raw)
    return _KIND_NAMES.get(name, name)


def status_name(raw: Any) -> str:
    return str(raw).lower()


# --------------------------------------------------------------------------
# Findings


@dataclass(frozen=True)
class Finding:
    id: str
    status: str
    kind: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "kind": self.kind,
            "message": self.message,
            "details": self.details,
        }


def _finding_id(checker: str, record: Mapping[str, Any]) -> str:
    identity = {
        key: record.get(key)
        for key in ("kind", "status", "sources", "overlaps", "space", "reason")
        if key in record
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    return f"{checker}:{record.get('kind')}:{digest[:16]}"


def findings_of(checker: str, payload: Mapping[str, Any]) -> list[Finding]:
    """All diagnostics of one payload as :class:`Finding` objects, sorted."""

    out: dict[str, Finding] = {}
    groups = [
        ("findings", None),
        ("advisories", "review"),
        ("incomplete", "incomplete"),
        ("diagnostics", None),
    ]
    for key, default_status in groups:
        for record in payload.get(key) or ():
            status = str(record.get("status") or default_status or "error")
            finding = Finding(
                id=_finding_id(checker, record),
                status=status,
                kind=str(record.get("kind", key)),
                message=str(record.get("message", "")),
                details=dict(record),
            )
            out.setdefault(finding.id, finding)
    error = payload.get("execution_error")
    if isinstance(error, Mapping) and not any(f.status == "error" for f in out.values()):
        finding = Finding(_finding_id(checker, error), "error", str(error.get("kind", "execution_error")),
                          str(error.get("message", "")), dict(error))
        out.setdefault(finding.id, finding)
    order = {"error": 0, "incomplete": 1, "review": 2}
    return sorted(out.values(), key=lambda f: (order.get(f.status, 3), f.kind, f.id))


# --------------------------------------------------------------------------
# Engine output -> payload


SpanOf = Callable[[int | None], "dict[str, Any] | None"]


def _overlap(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    spans = [e["bytes"] for e in evidence if isinstance(e.get("bytes"), Mapping)]
    if not spans:
        return []
    if len(spans) >= 2:
        start = max(int(spans[0]["start"]), int(spans[1]["start"]))
        end = min(int(spans[0]["start"]) + int(spans[0]["len"]), int(spans[1]["start"]) + int(spans[1]["len"]))
        if end > start:
            return [{"byte_offset": start, "byte_len": end - start, "byte_end": end}]
    start, length = int(spans[0]["start"]), int(spans[0]["len"])
    return [{"byte_offset": start, "byte_len": length, "byte_end": start + length}]


def record_from_core(raw: Mapping[str, Any], span_of: SpanOf) -> dict[str, Any]:
    """One ``report::Finding`` -> one payload record with embedded source spans."""

    kind = kind_name(raw.get("kind"))
    evidence = []
    for item in raw.get("evidence") or ():
        site = item.get("site")
        entry = {
            "role": item.get("role"),
            "site": site,
            "source_span": span_of(site),
            "buffer": item.get("buffer"),
            "actor": item.get("actor"),
            "detail": item.get("detail"),
        }
        if isinstance(item.get("bytes"), Mapping):
            entry["bytes"] = dict(item["bytes"])
        evidence.append(entry)
    record: dict[str, Any] = {
        "kind": kind,
        "status": status_name(raw.get("status", "error")),
        "message": raw.get("message", ""),
        "sources": [{"site": site, "source_span": span_of(site)} for site in raw.get("sites") or ()],
        "evidence": evidence,
    }
    if (space := raw.get("space")) is not None:
        record["space"] = space
    overlaps = _overlap(evidence)
    if overlaps:
        record["overlaps"] = overlaps
    return record


def diagnostic_from_core(raw: Mapping[str, Any], span_of: SpanOf) -> dict[str, Any]:
    """A runtime diagnostic (status/leftover) with its site resolved."""

    record = dict(raw)
    record["kind"] = kind_name(record.get("kind"))
    site = record.get("site")
    if isinstance(site, int):
        record["source_span"] = span_of(site)
    return record


def phase_payload(
    *,
    checker: str,
    phase_index: int,
    phase_name: str,
    records: Iterable[Mapping[str, Any]],
    status: Mapping[str, Any],
    diagnostics: Iterable[Mapping[str, Any]],
    coverage: Iterable[Sequence[Any]] = (),
    stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public schema-5 payload of one checker phase."""

    findings, advisories, incomplete = [], [], []
    for record in records:
        if record["status"] == "incomplete":
            incomplete.append(dict(record))
        elif record["kind"] in ADVISORY_KINDS:
            advisories.append({**record, "status": "review"})
        else:
            findings.append(dict(record))
    execution_error = None
    for diagnostic in diagnostics:
        if diagnostic.get("status") == "incomplete":
            incomplete.append(dict(diagnostic))
        elif execution_error is None:
            execution_error = dict(diagnostic)
    verdicts = [r["status"] for r in findings] + ["review"] * bool(advisories)
    verdicts += ["incomplete"] * bool(incomplete) + ["error"] * (execution_error is not None)
    verdict = worst(v if v in _VERDICT_RANK else "error" for v in verdicts)
    return {
        "schema_version": SCHEMA_VERSION,
        "checker": checker,
        "engine": "numsim-core",
        "phase": {"index": phase_index, "name": phase_name},
        "verdict": verdict,
        "findings": findings,
        "advisories": advisories,
        "incomplete": incomplete,
        "execution_error": execution_error,
        "status": dict(status),
        "coverage": {str(name): value for name, value in coverage},
        "stats": dict(stats or {}),
    }


def attach_sources(value: Any, span_of_kernel: Callable[[int, int | None], Any], kernel: int) -> Any:
    """Add ``source_span`` next to every ``site`` (and ``sources`` next to
    every ``sites`` list) of a checker payload, in place."""

    if isinstance(value, dict):
        kernel = value.get("kernel_index", value.get("kernel", kernel))
        kernel = kernel if isinstance(kernel, int) and not isinstance(kernel, bool) else 0
        site = value.get("site")
        if isinstance(site, int) and not isinstance(site, bool) and "source_span" not in value:
            value["source_span"] = span_of_kernel(kernel, site)
        sites = value.get("sites")
        if isinstance(sites, list) and "sources" not in value:
            value["sources"] = [
                {"site": s, "source_span": span_of_kernel(kernel, s)} for s in sites if isinstance(s, int)
            ]
        for key, child in list(value.items()):
            if key not in {"source_span", "sources"}:
                attach_sources(child, span_of_kernel, kernel)
    elif isinstance(value, list):
        for child in value:
            attach_sources(child, span_of_kernel, kernel)
    return value


def checker_phase_payload(
    base: Mapping[str, Any],
    *,
    checker: str,
    phase_index: int,
    phase_name: str,
    status: Mapping[str, Any],
    diagnostics: Iterable[Mapping[str, Any]],
    span_of_kernel: Callable[[int, int | None], Any],
) -> dict[str, Any]:
    """Complete a checker's own per-launch payload (``racecheck::serialize``
    / ``synccheck::serialize``) with phase facts, source spans and the
    runtime status of that launch."""

    payload = json.loads(json.dumps(base))
    for key in ("findings", "advisories", "incomplete"):
        payload.setdefault(key, [])
    payload.setdefault("execution_error", None)
    attach_sources(payload, span_of_kernel, phase_index)
    for item in payload.pop("review", None) or ():
        # Synccheck's own review slot (exit lints, `sync_exit_lint`).
        payload["advisories"].append({**item, "status": "review"})
    if checker == "synccheck":
        # Decision (dev-loop.md): Synccheck `review` items (exit lints such
        # as a dangling bar.arrive) are advisories, as in the legacy payload
        # where `findings` holds only errors. The verdict stays `review`.
        reviews = [r for r in payload["findings"] if r.get("status") == "review"]
        payload["findings"] = [r for r in payload["findings"] if r.get("status") != "review"]
        payload["advisories"].extend(reviews)
    runtime_error = None
    truncated = False
    for diagnostic in diagnostics:
        if diagnostic.get("status") == "incomplete":
            payload["incomplete"].append(dict(diagnostic))
            # Only the run's own stop (status-derived) truncates the event
            # log; other incompletes (subset execution, W2-14 stream cycle)
            # keep the checker's verdict and make it at least `incomplete`.
            truncated = truncated or diagnostic.get("source") == "run_status"
        elif diagnostic.get("status") == "review":
            payload["advisories"].append(dict(diagnostic))
        elif runtime_error is None:
            runtime_error = dict(diagnostic)
    if runtime_error is not None or truncated:
        # The launch did not run to completion, so the checker saw a
        # truncated event log: the engine's own error is the execution
        # error, and a checker verdict derived from the truncated log (e.g.
        # Synccheck "executor deadlock") is kept only as a note.
        notes = []
        if payload["execution_error"] is not None:
            notes.append(payload["execution_error"])
        # The checker's own "launch did not run to completion" incomplete
        # repeats the engine's stop, which is already reported.
        notes += [i for i in payload["incomplete"] if i.get("reason") == "truncated_launch"]
        payload["incomplete"] = [i for i in payload["incomplete"] if i.get("reason") != "truncated_launch"]
        if notes:
            payload["checker_on_truncated_log"] = notes
        payload["execution_error"] = runtime_error
    # On a truncated log the checker's overall verdict is void; recompute it
    # from what remains (engine stop + findings proven before the stop).
    verdicts = ["clean" if "checker_on_truncated_log" in payload else str(payload.get("verdict") or "clean")]
    verdicts += [r.get("status", "error") for r in payload["findings"]]
    verdicts += ["review"] * bool(payload["advisories"]) + ["incomplete"] * bool(payload["incomplete"])
    verdicts += ["error"] * (payload["execution_error"] is not None)
    payload.update(
        schema_version=SCHEMA_VERSION,
        checker=checker,
        engine="numsim-core",
        phase={"index": phase_index, "name": phase_name, "kernel_index": phase_index},
        analysis_scope=payload.get("analysis_scope") or {"kind": "full_launch"},
        verdict=worst(v if v in _VERDICT_RANK else "error" for v in verdicts),
        status=dict(status),
    )
    return payload


# --------------------------------------------------------------------------
# Results and reports


@dataclass(frozen=True)
class AnalysisResult:
    """One checker phase (legacy ``NativeAnalysisResult`` surface)."""

    checker: str
    payload: dict[str, Any]

    @property
    def verdict(self) -> str:
        return str(self.payload["verdict"])

    @property
    def findings(self) -> list[dict[str, Any]]:
        return list(self.payload.get("findings", ()))

    @property
    def advisories(self) -> list[dict[str, Any]]:
        return list(self.payload.get("advisories", ()))

    @property
    def incomplete(self) -> list[dict[str, Any]]:
        return list(self.payload.get("incomplete", ()))

    @property
    def coverage(self) -> dict[str, Any]:
        return dict(self.payload.get("coverage", {}))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload))


@dataclass
class NumSimResult:
    outputs: dict[str, Any]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return worst(
            d.get("status", "error") if d.get("status") in _VERDICT_RANK else "error"
            for d in self.diagnostics
        )

    def assert_close(self, expected: dict[str, Any], tolerances: dict[str, Any] | None = None) -> None:
        compare(self, expected, tolerances=tolerances).require_ok()


class _Report:
    checker_name = "report"

    def payloads(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @property
    def findings(self) -> list[Finding]:
        out: list[Finding] = []
        for payload in self.payloads():
            out.extend(findings_of(self.checker_name, payload))
        return out

    @property
    def verdict(self) -> str:
        return worst(p["verdict"] for p in self.payloads())

    def to_dict(self) -> dict[str, Any]:
        findings = self.findings
        return {
            "schema_version": SCHEMA_VERSION,
            "checker": self.checker_name,
            "engine": "numsim-core",
            "verdict": self.verdict,
            "findings": [f.to_dict() for f in findings],
            "summary": f"{self.checker_name} {self.verdict.upper()} ({len(findings)} findings)",
            "phases": self.payloads(),
        }

    def format(self) -> str:
        return render(self)

    def print(self) -> None:
        print(self.format())

    def require_clean(self) -> None:
        if self.verdict != "clean":
            raise AssertionError(self.format())


class RaceReport(_Report):
    checker_name = "racecheck"

    def __init__(self, phases: Sequence[AnalysisResult]):
        self.phases = list(phases)

    def payloads(self) -> list[dict[str, Any]]:
        return [phase.payload for phase in self.phases]


class SyncCheckReport(RaceReport):
    checker_name = "synccheck"


class NumSimReport(_Report):
    checker_name = "numsim"

    def __init__(self, ok: bool, mismatches: Sequence[Any] = (), diagnostics: Sequence[dict[str, Any]] = ()):
        self.ok = bool(ok)
        self.mismatches = list(mismatches)
        self.diagnostics = list(diagnostics)

    def payloads(self) -> list[dict[str, Any]]:
        verdict = "error" if not self.ok else worst(
            d.get("status", "error") if d.get("status") in _VERDICT_RANK else "error"
            for d in self.diagnostics
        )
        return [{
            "schema_version": SCHEMA_VERSION,
            "checker": "numsim",
            "verdict": verdict,
            "diagnostics": self.diagnostics,
            "mismatches": [m.render() if hasattr(m, "render") else str(m) for m in self.mismatches],
        }]

    def require_ok(self) -> None:
        if not self.ok:
            first = self.mismatches[0] if self.mismatches else None
            detail = "<missing>" if first is None else (first.render() if hasattr(first, "render") else str(first))
            raise AssertionError(f"NumSim comparison failed; first mismatch: {detail}")


def compare(result: Any, expected: dict[str, Any], *, tolerances: dict[str, Any] | None = None) -> NumSimReport:
    """Compare outputs with a reference.

    The tolerance/region/encoding logic is shared with the legacy package
    (``numsim.api.compare``) until step 5 of the migration moves it here.
    """

    from tirx_harness.numsim.api import compare as legacy_compare

    legacy = legacy_compare(result, expected, tolerances=tolerances)
    return NumSimReport(legacy.ok, legacy.mismatches, list(getattr(result, "diagnostics", ())))


# --------------------------------------------------------------------------
# The one renderer


def _location(record: Mapping[str, Any]) -> str:
    spans = [s.get("source_span") for s in record.get("sources") or () if s.get("source_span")]
    if not spans and record.get("source_span"):
        spans = [record["source_span"]]
    parts = []
    for span in spans:
        leaves = span["spans"] if span.get("kind") == "sequential" else [span]
        leaf = leaves[-1]
        parts.append(f"{leaf['source_name']}:{leaf['line']}")
    return ", ".join(dict.fromkeys(parts))


def render(report: _Report) -> str:
    findings = report.findings
    lines = [f"{report.checker_name} {report.verdict.upper()} - {len(findings)} finding(s)"]
    for finding in findings:
        where = _location(finding.details)
        lines.append(f"  [{finding.status.upper()}] {finding.kind}: {finding.message}".rstrip())
        if where:
            lines.append(f"    at {where}")
        for overlap in finding.details.get("overlaps") or ():
            lines.append(f"    bytes [{overlap['byte_offset']}, {overlap['byte_end']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# JSON schema


def payload_json_schema() -> dict[str, Any]:
    """JSON schema (draft 2020-12) of one schema-5 checker phase payload."""

    span = {
        "type": ["object", "null"],
        "properties": {
            "kind": {"enum": ["span", "sequential"]},
            "source_name": {"type": "string"},
            "line": {"type": "integer"},
            "column": {"type": "integer"},
            "end_line": {"type": "integer"},
            "end_column": {"type": "integer"},
            "spans": {"type": "array"},
        },
        "required": ["kind"],
    }
    record = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "status": {"enum": ["error", "review", "incomplete"]},
            "message": {"type": "string"},
            "space": {"type": "string"},
            "sources": {"type": "array", "items": {
                "type": "object",
                "properties": {"site": {"type": ["integer", "null"]}, "source_span": span},
            }},
            "overlaps": {"type": "array", "items": {
                "type": "object",
                "properties": {k: {"type": "integer"} for k in ("byte_offset", "byte_len", "byte_end")},
                "required": ["byte_offset", "byte_len", "byte_end"],
            }},
            "evidence": {"type": "array"},
        },
        "required": ["kind", "status"],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://tirxharness.mlc.ai/schemas/numsim-v2-phase-payload-5.json",
        "title": "NumSim v2 checker phase payload",
        "type": "object",
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "checker": {"enum": ["racecheck", "synccheck"]},
            "engine": {"type": "string"},
            "phase": {"type": "object", "properties": {"index": {"type": "integer"}, "name": {"type": "string"}}},
            "verdict": {"enum": list(VERDICTS)},
            "findings": {"type": "array", "items": record},
            "advisories": {"type": "array", "items": record},
            "incomplete": {"type": "array", "items": {"type": "object"}},
            "execution_error": {"type": ["object", "null"]},
            "status": {"type": "object"},
            "coverage": {"type": "object"},
            "stats": {"type": "object"},
        },
        "required": ["schema_version", "checker", "verdict", "findings", "advisories", "incomplete", "execution_error"],
    }


__all__ = [
    "ADVISORY_KINDS",
    "SCHEMA_VERSION",
    "AnalysisResult",
    "Finding",
    "NumSimReport",
    "NumSimResult",
    "RaceReport",
    "SyncCheckReport",
    "attach_sources",
    "checker_phase_payload",
    "compare",
    "findings_of",
    "payload_json_schema",
    "phase_payload",
    "record_from_core",
    "render",
]
