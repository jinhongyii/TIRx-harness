"""Classify every legacy Python test function for the NumSim redesign's test retirement.

Usage (from the repository root or anywhere; stdlib only, plus numpy for the
optional cvt-golden check)::

    python scripts/numsim-v2/classify_tests.py \
        [--tests tirx_harness/tests] [--csv OUT.csv] [--summary OUT.md] [--check-goldens]

Categories (``docs/development/test-migration.md``):

* **A** semantic kernel-level test: runs a kernel and checks outputs, verdict or
  finding kinds. Target ``conformance`` (corpus / wiki kernels, already
  snapshotted) or ``v2-kernel-case`` (small kernel → a conformance micro-case
  or a ``tests/numsim/v2`` kernel test run through the v2 API).
* **B** checker-semantics unit test with a small kernel → a Rust scenario test
  built from contract events (``numsim-core/tests/{racecheck,synccheck}_*.rs``).
  ``status`` says ``covered`` / ``ported`` / ``v2_kernel_test`` /
  ``needs_kernel`` (from the reviewed maps
  ``coverage/{other_b,racecheck,synccheck}.tsv``) or ``unreviewed``.
* **C** pins the implementation: generated Rust text, ``v2::`` paths, poll /
  transition counts, raw payload field shapes, legacy-private APIs, legacy
  Python internals (transpiler, registry, ABI) → delete with the legacy code.
* **D** numerical op golden, GPU-paired (``tests/numsim/microtests``):
  ``D-recorded`` (cvt goldens, replayed bit-exactly in ``numsim-oplib``) or
  ``D-live`` (paired against a live GPU run; record goldens before retiring).
* **E** tile-form kernel that TVM's own ``TilePrimitiveDispatch`` rejects
  (lowering-inventory.md §D.2) → delete. Read from
  ``coverage/tile_dispatch_rejections.txt`` (``tile_rejections.py``).
* **F** infrastructure / CLI / packaging / ``dump_kernel``, unrelated to
  NumSim semantics → keep.
* **N** already a new-layer test (``tests/conformance``, ``tests/numsim/v2``) → keep.

Precedence: reviewed coverage maps > ``OVERRIDES`` below > E list > path rules
> AST signals. Every row carries the rule that decided it (``rule`` column),
so a disputed row can be traced and overridden.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COVERAGE = HERE / "coverage"

# Manual decisions for rows the signals get wrong. Key: test id prefix
# (``file`` or ``file::function``), value: (category, target/status, reason).
OVERRIDES: dict[str, tuple[str, str, str]] = {
    "tests/numsim/microtests/test_gpu_harness.py": ("C", "delete", "tests the GPU pairing harness itself"),
    "tests/numsim/abi/": ("C", "delete", "legacy engine ABI / public engine surface"),
    "tests/analysis_tools/test_report.py": ("C", "delete", "legacy report payload shape"),
    "tests/analysis_tools/synccheck/test_report_readability.py": ("C", "delete", "report message text"),
    "tests/analysis_tools/shared/test_checker_defaults.py": ("C", "delete", "legacy checker option defaults"),
    "tests/analysis_tools/shared/test_codegen_dependency_paths.py": ("C", "delete", "legacy codegen crate paths"),
    "tests/analysis_tools/shared/test_instruction_profile.py": ("C", "delete", "legacy instruction profile counters"),
    "tests/numsim/registry/test_engine_support_matrix.py": ("C", "delete", "legacy engine support matrix"),
    "tests/numsim/registry/test_op_registry_contract.py": ("C", "delete", "legacy op registry tables"),
    "tests/numsim/registry/test_call_registry.py": ("C", "delete", "legacy call registry tables"),
    "tests/numsim/registry/test_tile_registry_contract.py": ("C", "delete", "legacy tile registry tables"),
    "tests/numsim/registry/test_ptx_dialect_decoder.py": ("C", "delete", "legacy PTX dialect decoder"),
    "tests/numsim/integration/test_api.py": ("A", "v2-api", "public API contract (compare / run_case / Engine); v2 keeps the legacy signatures, rerun under NUMSIM_IMPL=v2"),
    "tests/numsim/integration/test_build_workspace.py": ("C", "delete", "legacy Rust build workspace"),
    "tests/numsim/integration/test_compile_cache.py": ("C", "delete", "legacy compile cache"),
    "tests/numsim/integration/test_source_map.py": ("C", "delete", "legacy source-map serialization"),
    "tests/numsim/integration/test_native_frontend.py": ("C", "delete", "legacy frontend-rs"),
    "tests/numsim/integration/test_profile_build.py": ("C", "delete", "legacy profile build"),
    "tests/numsim/integration/test_packaging.py": ("C", "delete", "legacy engine wheel contents"),
    "tests/numsim/integration/test_numpy_backend.py": ("C", "delete", "legacy numpy backend switch"),
    "tests/numsim/integration/test_dependency_isolation.py": ("C", "delete", "legacy import layering"),
    "tests/numsim/integration/test_tirx_kernels.py": ("F", "keep", "tirx-kernels package loader used by the corpus"),
    # Phase-3 review of the v2 public-API run: the only internals-pin failure
    # with no semantic assert left once the pin is stripped.
    "tests/numsim/runtime/test_non_tensor_bulk_forms.py::test_bulk_g2s_cluster_dynamic_predicate_transpiles": (
        "C", "delete", "only asserts the legacy resolved PTX op-name set"),
    # W1 public-API triage (001a09f): out of scope for v2, deleted.
    "tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_cta_group2_supports_float16_payloads": (
        "E", "delete", "W1 001a09f: replicated TMEM view (contract item 29 fail-closed)"),
    "tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_expands_tlane_replicas": (
        "E", "delete", "W1 001a09f: replicated TMEM view (contract item 29 fail-closed)"),
    "tests/numsim/integration/test_tcgen_transfer_artifact.py::test_tcgen_cp_supports_rank3_multi_instruction_layout": (
        "E", "delete", "W1 001a09f: replicated TMEM view (contract item 29 fail-closed)"),
    "tests/numsim/runtime/test_tile_general_semantics.py::test_mxfp4_uses_ue8m0_scales_over_32_element_vectors": (
        "E", "delete", "W1 001a09f: replicated TMEM view (contract item 29 fail-closed)"),
    "tests/numsim/runtime/test_dense_mma_forms.py::test_legacy_m16n8k32_int8_reuses_dense_form_and_engine": (
        "E", "delete", "W1 001a09f: ptx_legacy surface, out of scope"),
    "tests/numsim/runtime/test_matrix_memory_domain_oracle.py::test_legacy_ldmatrix_x1_domain_matches_independent_fragment_mapping": (
        "E", "delete", "W1 001a09f: ptx_legacy surface, out of scope"),
    "tests/test_dump_kernel.py": ("F", "keep", "dump_kernel tool"),
    "tests/test_packaging.py": ("F", "keep", "packaging"),
    "tests/test_skills.py": ("F", "keep", "skills CLI"),
}

_S = "tests/analysis_tools/synccheck/"
REVIEWED_F: dict[str, tuple[str, str, str]] = {
    _S + "runtime/": ("A", "v2-kernel-case", "per-PTX-op runtime support under synccheck"),
    _S + "test_native_gate_intrinsics.py": ("A", "v2-kernel-case", "math intrinsic support"),
    _S + "test_reported_tool_regressions.py": ("A", "v2-kernel-case", "instruction support / predication"),
    _S + "test_native_frontend_source_anchor.py": ("C", "delete", "legacy frontend unsupported-form anchors"),
    _S + "test_sync_site_manifest.py": ("C", "delete", "legacy frontend source map / manifest"),
    _S + "test_native_polling_control.py": ("C", "delete", "legacy fixed-trace eligibility"),
    _S + "test_native_lost_wake_regression": ("C", "delete", "legacy multi-threaded executor lost-wake race"),
    _S + "test_native_shared_mbar_pointer.py": ("C", "delete", "rustc build regression of generated code"),
    _S + "test_native_synccheck_artifact.py": ("C", "delete", "legacy executor plumbing (opt level, subset runner, worker counts)"),
    "tests/analysis_tools/racecheck/test_native_racecheck_artifact.py": ("C", "delete", "legacy worker parallelism / payload stability"),
}

KERNEL_RUN = {
    "Engine", "transpile", "run_checked", "assert_rejected", "analyze", "racecheck", "synccheck",
    "require_clean", "run_paired_primfunc", "run_racecheck_phase", "run_synccheck_phase", "checker",
    "internal_synccheck", "run_case", "run_numsim", "_run_racecheck", "_run_synccheck", "compare",
    "run_three_way", "run_canonical_case", "run_wiki_case", "run_kernel", "execute", "run_three_way_case",
    "require_ok", "require_numsim_gpu",
}
RUNNERS: set[str] = set(KERNEL_RUN)
CHECKER_RUN = {
    "racecheck", "synccheck", "run_racecheck_phase", "run_synccheck_phase", "internal_synccheck",
    "_run_racecheck", "_run_synccheck", "checker",
}
SEMANTIC_ASSERT = {
    "assert_array_equal", "assert_allclose", "assert_array_almost_equal", "array_equal", "allclose",
    "require_clean", "assert_rejected", "raises", "isnan", "reference",
}
SEMANTIC_WORDS = re.compile(r"\b(verdict|findings?|kind|status|incomplete|advisor(y|ies)|data_race|deadlock|clean)\b")
# Identifiers that only exist to inspect the legacy implementation.
C_NAMES = {
    "emit_rust_module", "rust_source", "_emitted_heads", "artifact_template", "decode_ptx_call", "to_manifest",
    "load_cached_generated_artifact", "render_rust", "emit_rust", "generated_source", "support_matrix",
    "abi_version", "ENGINE_ABI_VERSION", "instruction_profile", "native_calls", "post_order_nodes",
    "resolved_kernel", "evaluated_kernel", "_lines_with", "native_loop_iteration_budget",
    "ExecutionSubset", "_shared_backing_sizes", "_resolved_ptx_raw_tcgen_calls",
}
# Identifiers whose presence makes a test C even when it also runs a kernel:
# the run only feeds an inspection of legacy internals.
STRONG_C = {"ExecutionSubset", "_shared_backing_sizes", "_resolved_ptx_raw_tcgen_calls", "native_calls", "post_order_nodes", "resolved_kernel", "evaluated_kernel", "_lines_with", "decode_ptx_call", "_emitted_heads"}
C_STRINGS = re.compile(
    r"v2::|\bfn [a-z_]+\(|let mut |WarpValue|\bpoll(s|_count|_order)?\b|transition(s|_count)\b|task_count|cargo|rustc|\.rs\b"
)
# Test-function names that describe the legacy implementation (Rust codegen
# slicing / splitting, compile caches, manifests, worker plumbing, profiling).
C_FUNC_NAME = re.compile(
    r"codegen_slice|native_codegen|_splits?_|^test_split|_split$|splits_|cache(?!_hint)|manifest|compile|rust|lazy|"
    r"import|worker_configuration|loop_policy|instruction_profile|emission|precompile|engine_phase"
)
CHECKER_TOPIC = re.compile(
    r"race|sync|hb|happens|publi|order|visib|wait|fence|mbarrier|release|acquire|proxy|deadlock|hang|lifetime|reuse|overlap|scope"
)
CORPUS_PATH = re.compile(r"/(corpus|wiki)/|canonical")
CORPUS_IMPORT = re.compile(r"canonical_cases|corpus\.kernels|_tirx_kernels|wiki|tirx_kernels")


@dataclass
class Features:
    calls: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    strings: list[str] = field(default_factory=list)
    assert_text: list[str] = field(default_factory=list)

    def merge(self, other: Features) -> None:
        self.calls |= other.calls
        self.names |= other.names
        self.strings += other.strings
        self.assert_text += other.assert_text


def features(node: ast.AST) -> Features:
    out = Features()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name:
                out.calls.add(name)
        elif isinstance(n, ast.Name):
            out.names.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.names.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.strings.append(n.value)
        if isinstance(n, ast.Assert):
            out.assert_text.append(ast.unparse(n))
    return out


def param_count(fn: ast.FunctionDef) -> str:
    total = 1
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Call) and getattr(dec.func, "attr", "") == "parametrize" and len(dec.args) >= 2:
            values = dec.args[1]
            if isinstance(values, (ast.List, ast.Tuple)):
                total *= max(1, len(values.elts))
            else:
                return "?"
    return str(total)


@dataclass
class Row:
    test_id: str
    directory: str
    params: str
    category: str = ""
    target: str = ""
    rule: str = ""
    reason: str = ""
    rust_tests: str = ""
    signals: str = ""
    surface: str = ""
    v2_status: str = ""


def directory_of(rel: str) -> str:
    parts = rel.split("/")
    # tests/<a>/<b>/file.py -> a/b ; tests/file.py -> (top)
    if len(parts) <= 2:
        return "(top)"
    return "/".join(parts[1:3]) if len(parts) > 3 else parts[1]


def load_tsv(path: Path) -> dict[str, tuple[str, str, str]]:
    out: dict[str, tuple[str, str, str]] = {}
    if not path.exists():
        return out
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            out[row["legacy_test"].strip()] = (row["status"].strip(), row.get("rust_tests", "").strip(), row.get("note", "").strip())
    return out


def load_v2_status(path: Path) -> dict[str, str]:
    """``NUMSIM_IMPL=v2`` outcome of the public-API A tests (per function)."""
    if not path.exists():
        return {}
    with path.open() as handle:
        return {row["legacy_test"]: row["v2_status"] for row in csv.DictReader(handle, delimiter="\t")}


def load_tile_rejections(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                test, _, note = line.partition("\t")
                out[test.strip()] = note.strip()
    return out


def classify(row: Row, f: Features, module_src: str, rel: str, maps, tile) -> None:
    calls, names = f.calls, f.names | f.calls
    runs = bool(calls & RUNNERS) or "run" in calls and ("numsim" in module_src)
    checker = bool(calls & CHECKER_RUN)
    texts = " ".join(f.assert_text)
    semantic = bool(calls & SEMANTIC_ASSERT) or bool(SEMANTIC_WORDS.search(texts)) or "np.testing" in texts
    c_hits = sorted(names & C_NAMES)
    c_str = sorted({m.group(0) for s in f.strings + f.assert_text for m in [C_STRINGS.search(s)] if m})
    row.signals = ";".join(
        x for x in [
            "runs" if runs else "", "checker" if checker else "", "semantic" if semantic else "",
            ("c:" + ",".join(c_hits + c_str)[:80]) if (c_hits or c_str) else "",
        ] if x
    )

    # 1. Reviewed coverage maps (B rows and their non-B rejections).
    for kind, table in maps.items():
        if row.test_id in table:
            status, rust, note = table[row.test_id]
            if status == "not_b:F" and "synccheck" in note:
                # A synccheck verdict asserted from a racecheck file: B, but
                # outside the racecheck map's scope.
                row.category, row.target, row.rule = "B", "unreviewed", f"{kind}.tsv"
                row.reason = note
                return
            if status == "not_b:F":
                # The maps only say "not B"; legacy-executor and frontend rows
                # they file under F are refined by REVIEWED_F.
                for prefix in sorted(REVIEWED_F, key=len, reverse=True):
                    if row.test_id.startswith(prefix):
                        row.category, row.target, row.reason = REVIEWED_F[prefix]
                        row.rule = f"{kind}.tsv+refine"
                        return
            if status.startswith("not_b:"):
                row.category, row.target = status.split(":", 1)[1], ("conformance" if status.endswith("A") else "")
                row.rule, row.reason = f"{kind}.tsv", note
                if row.category == "C":
                    row.target = "delete"
                elif row.category == "E":
                    row.target = "delete"
                elif row.category == "F":
                    row.target = "keep"
                elif row.category == "A" and not CORPUS_PATH.search(rel):
                    row.target = "v2-kernel-case"
            else:
                row.category, row.target, row.rule, row.reason, row.rust_tests = "B", status, f"{kind}.tsv", note, rust
            return
    # 2. Manual overrides.
    for prefix in sorted(OVERRIDES, key=len, reverse=True):
        if row.test_id.startswith(prefix):
            row.category, row.target, row.reason = OVERRIDES[prefix]
            row.rule = "override"
            return
    # 3. TVM-dispatch-rejected tile forms.
    if row.test_id in tile:
        ratio = tile[row.test_id].split()[0]
        done, _, total = ratio.partition("/")
        if done == total:
            row.category, row.target, row.rule, row.reason = "E", "delete", "tile_dispatch_rejections", tile[row.test_id]
            return
        # Only some parametrizations hit a rejected tile form: classify the
        # function normally; retiring just those params is a manual edit.
        row.signals = f"partial-E:{ratio}"
    # 4. Path rules.
    if rel.startswith(("tests/conformance/", "tests/numsim/v2/")):
        row.category, row.target, row.rule, row.reason = "N", "keep", "path", "new-layer test"
        return
    if rel.startswith("tests/numsim/microtests/"):
        recorded = bool(re.search(r"_goldens\b", module_src))
        row.category = "D"
        row.target = "D-recorded" if recorded else "D-live"
        row.rule = "path"
        row.reason = (
            "GPU-recorded cvt goldens; replayed in numsim-oplib src/cvt/goldens"
            if recorded else "paired against a live GPU run; record outputs as goldens before retiring"
        )
        return
    if rel.count("/") == 1:
        row.category, row.target, row.rule, row.reason = "F", "keep", "path", "top-level infrastructure test"
        return
    corpus = bool(CORPUS_PATH.search(rel)) or bool(CORPUS_IMPORT.search(module_src)) and runs
    if CORPUS_PATH.search(rel) and not runs:
        row.category, row.target, row.rule = "F", "keep", "path:corpus-fixture"
        row.reason = "corpus fixture / independent reference self-test (conformance relies on these oracles)"
        return
    # 5. Signals.
    name = row.test_id.rsplit("::", 1)[1]
    if C_FUNC_NAME.search(name) and not corpus:
        row.category, row.target, row.rule = "C", "delete", "name:impl"
        row.reason = "test name describes the legacy implementation (" + C_FUNC_NAME.search(name).group(0) + ")"
        return
    if c_hits or c_str:
        if runs and semantic and (corpus or not set(c_hits) & STRONG_C):
            row.category, row.rule = ("B" if checker and "analysis_tools" in rel else "A"), "signals:mixed"
            row.target = "unreviewed" if row.category == "B" else ("conformance" if corpus else "v2-kernel-case")
            row.reason = "semantic asserts plus implementation pins (" + ",".join(c_hits + c_str)[:60] + "); port the semantic part only"
        else:
            row.category, row.target, row.rule = "C", "delete", "signals:impl"
            row.reason = "pins implementation: " + ",".join(c_hits + c_str)[:80]
        return
    if runs and semantic:
        if corpus:
            row.category, row.target, row.rule, row.reason = "A", "conformance", "signals", "corpus/wiki kernel verdict or output"
        elif checker and (CHECKER_TOPIC.search(name) or "analysis_tools" in rel):
            row.category, row.target, row.rule, row.reason = "B", "unreviewed", "signals", "checker verdict on a small kernel"
        elif checker:
            row.category, row.target, row.rule = "A", "v2-kernel-case", "signals:three-mode"
            row.reason = "op test that also runs the checkers for a clean verdict (keep as a three-mode v2 kernel case)"
        else:
            row.category, row.target, row.rule, row.reason = "A", "v2-kernel-case", "signals", "small kernel output / error"
        return
    if runs:
        row.category, row.target, row.rule = "A", ("conformance" if corpus else "v2-kernel-case"), "signals:weak"
        row.reason = "runs a kernel without recognisable semantic asserts (smoke)"
        return
    if re.search(r"from tirx_harness\.numsim(\.(?!v2)|\s+import)", module_src):
        row.category, row.target, row.rule, row.reason = "C", "delete", "signals:internals", "legacy NumSim Python internals, no kernel run"
        return
    row.category, row.target, row.rule, row.reason = "F", "keep", "fallback", "no kernel, no legacy internals"


def discover_runners(tests_root: Path) -> set[str]:
    """Names of non-test helpers (support modules, harnesses, module-local
    ``_run_*``) that transitively run a kernel; they count as kernel runs."""
    bodies: dict[str, Features] = {}
    for path in tests_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("test_"):
                feats = bodies.setdefault(node.name, Features())
                feats.merge(features(node))
    runners = set(KERNEL_RUN)
    changed = True
    while changed:
        changed = False
        for name, feats in bodies.items():
            if name not in runners and feats.calls & runners:
                runners.add(name)
                changed = True
    return runners


PUBLIC_MODULES = {"tirx_harness.numsim", "tirx_harness.numsim.api", "tirx_harness.numsim.errors", "tirx_harness.numsim.v2"}


def module_surface(tree: ast.Module) -> str:
    """``public`` when the module reaches NumSim only through the public
    facade (which v2 mirrors), else ``internal:<first private module>``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tirx_harness.numsim"):
            if node.module not in PUBLIC_MODULES and not node.module.startswith("tirx_harness.numsim.v2"):
                return "internal:" + node.module.removeprefix("tirx_harness.numsim.")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tirx_harness.numsim.") and alias.name not in PUBLIC_MODULES:
                    return "internal:" + alias.name.removeprefix("tirx_harness.numsim.")
    return "public"


def walk(tests_root: Path, maps, tile) -> list[Row]:
    rows: list[Row] = []
    base = tests_root.parent
    for path in sorted(tests_root.rglob("test_*.py")):
        rel = path.relative_to(base).as_posix()
        src = path.read_text()
        tree = ast.parse(src)
        surface = module_surface(tree)
        helpers = {
            n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("test_")
        }

        def emit(fn: ast.FunctionDef, prefix: str) -> None:
            feats = features(fn)
            # One level of module-local helpers (``_run_*`` wrappers etc.).
            # Module-local fixtures requested by argument name.
            used = list(feats.calls) + [a.arg for a in fn.args.args]
            for name in used:
                if name in helpers:
                    feats.merge(features(helpers[name]))
            row = Row(test_id=f"{rel}::{prefix}{fn.name}", directory=directory_of(rel), params=param_count(fn))
            classify(row, feats, src, rel, maps, tile)
            ratio = tile.get(row.test_id, "").split(" ", 1)[0]
            if ratio and ratio.split("/")[0] != ratio.split("/")[-1]:
                row.reason += f" [partial E: {ratio} kernels hit a TVM-rejected tile form; drop those params by hand]"
            row.surface = surface
            rows.append(row)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                emit(node, "")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test_"):
                        emit(sub, node.name + "::")
    return rows


def check_goldens(tests_root: Path) -> str:
    """Every GPU-recorded cvt golden form has a generated numsim-oplib test."""
    import importlib.util

    cases = tests_root / "numsim/microtests/cases"
    gen = REPO / "tirx_harness/src/tirx_harness/numsim/core-rs/numsim-oplib/src/cvt/goldens"
    forms = 0
    for name, attr in [("ptx_cvt_scalar_goldens", "SCALAR_CVT_GOLDENS"), ("ptx_cvt_fp8_goldens", "GOLDENS"), ("ptx_cvt_narrow_goldens", "GOLDENS")]:
        spec = importlib.util.spec_from_file_location(name, cases / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        forms += len(getattr(module, attr))
    rust = len(re.findall(r"^fn ", (gen / "tests.rs").read_text(), re.M))
    return f"cvt goldens: {forms} recorded forms in Python, {rust} generated numsim-oplib tests" + (
        "" if forms == rust else "  ** MISMATCH **"
    )


def summary(rows: list[Row]) -> str:
    cats = ["A", "B", "C", "D", "E", "F", "N"]
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r.directory][r.category] += 1
    lines = ["| directory | " + " | ".join(cats) + " | total |", "| --- |" + " ---: |" * (len(cats) + 1)]
    total = collections.Counter()
    for d in sorted(by):
        total.update(by[d])
        lines.append(f"| {d} | " + " | ".join(str(by[d][c] or "") for c in cats) + f" | {sum(by[d].values())} |")
    lines.append("| **total** | " + " | ".join(f"**{total[c]}**" for c in cats) + f" | **{sum(total.values())}** |")
    surf = collections.Counter(
        (r.target, "public" if r.surface == "public" else "internal") for r in rows if r.category == "A"
    )
    lines += ["", "| A target | public facade only | imports legacy internals |", "| --- | ---: | ---: |"]
    for t in sorted({t for t, _ in surf}):
        lines.append(f"| {t} | {surf[(t, 'public')]} | {surf[(t, 'internal')]} |")
    sub = collections.Counter((r.category, r.target) for r in rows)
    lines += ["", "| category | target / status | tests |", "| --- | --- | ---: |"]
    for (c, t), n in sorted(sub.items()):
        lines.append(f"| {c} | {t} | {n} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tests", type=Path, default=REPO / "tirx_harness/tests")
    parser.add_argument("--csv", type=Path, default=COVERAGE / "test_classification.csv")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--check-goldens", action="store_true")
    args = parser.parse_args()
    # ``other_b`` (reviewed B rows outside the two checker directories) first.
    maps = {k: load_tsv(COVERAGE / f"{k}.tsv") for k in ("other_b", "racecheck", "synccheck")}
    tile = load_tile_rejections(COVERAGE / "tile_dispatch_rejections.txt")
    RUNNERS.update(discover_runners(args.tests))
    rows = walk(args.tests, maps, tile)
    v2 = load_v2_status(COVERAGE / "v2_public_status.tsv")
    for r in rows:
        r.v2_status = v2.get(r.test_id, "")
    seen = {r.test_id for r in rows}
    for kind, table in maps.items():
        for stale in sorted(set(table) - seen):
            print(f"warning: {kind}.tsv row not found in the test tree: {stale}", file=sys.stderr)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_id", "directory", "params", "category", "target", "rule", "reason", "rust_tests", "surface", "v2_status", "signals"])
        for r in rows:
            writer.writerow([r.test_id, r.directory, r.params, r.category, r.target, r.rule, r.reason, r.rust_tests, r.surface, r.v2_status, r.signals])
    text = summary(rows)
    if args.check_goldens:
        text += "\n\n" + check_goldens(args.tests)
    print(text)
    if args.summary:
        args.summary.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
