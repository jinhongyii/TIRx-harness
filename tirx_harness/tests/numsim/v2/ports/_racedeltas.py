"""Shared assertions for the racecheck-delta ports (``test_racedeltas_*``).

Racecheck delta **B7** (``docs/development/racecheck-behaviour-deltas.md``):
a qualifier-less ``mbarrier.arrive.shared::cluster`` (any arrive form,
including ``.expect_tx`` and ``.multicast::cluster``) defaults to
``.release.cta``. When the barrier lives in a peer CTA, the ``.cta`` arrive
does not include the peer waiter, so the arrive -> wait pair is a
``scope_mismatch`` error (legacy had no scope model and was clean).
"""

from __future__ import annotations


def span_text(span: dict, source: str | None = None) -> str:
    """The source lines a ``source_span`` covers (``<str>`` spans need ``source``)."""

    name = span["source_name"]
    if name == "<str>":
        assert source is not None, span
        text = source
    else:
        with open(name, encoding="utf-8") as handle:
            text = handle.read()
    return "\n".join(text.split("\n")[span["line"] - 1 : span["end_line"]])


def _site_text(details: dict, site: int, source: str | None) -> str:
    spans = [entry["source_span"] for entry in details["sources"] if entry["site"] == site]
    assert len(spans) == 1, details
    return span_text(spans[0], source)


def assert_b7_scope_mismatch(report, *, acquire_warps, count: int, source: str | None = None) -> None:
    """Racecheck delta B7: verdict ``error`` whose findings are exactly ``count``
    ``scope_mismatch`` errors, each pairing CTA 0's qualifier-less remote
    ``mbarrier.arrive`` (release ``.cta``, warp 0) with a ``.cta`` acquire
    (``mbarrier_wait`` / ``mbarrier.test_wait``) by a waiter in a peer CTA
    (``acquire_warps``). No other finding (in particular no follow-on
    ``data_race``) is present."""

    assert report.verdict == "error", report.format()
    assert {(f.status, f.kind) for f in report.findings} == {("error", "scope_mismatch")}, report.format()
    assert len(report.findings) == count, report.format()
    acquire_sites = set()
    for finding in report.findings:
        details = finding.details
        assert details["ordering_failure"] == "scope_mismatch", details
        assert (details["release_scope"], details["acquire_scope"]) == ("cta", "cta"), details
        assert details["release_warp_id"] == 0, details
        assert details["acquire_warp_id"] in set(acquire_warps), details
        assert details["operation"]["source"]["op_name"].startswith("tirx.ptx.mbarrier_arrive"), details
        release = _site_text(details, details["release_site"], source)
        assert "mbarrier.arrive" in release, release
        assert "shared__cluster" in release or "shared::cluster" in release, release
        assert ".release" not in release and "relaxed" not in release, release
        acquire = _site_text(details, details["acquire_site"], source)
        assert "mbarrier_wait" in acquire or "mbarrier.test_wait" in acquire, acquire
        acquire_sites.add(details["acquire_site"])
    assert len(acquire_sites) == count, report.format()
