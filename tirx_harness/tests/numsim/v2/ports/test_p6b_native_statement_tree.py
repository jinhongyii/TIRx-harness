"""v2 port of ``tests/numsim/integration/test_native_statement_tree.py``."""

from __future__ import annotations

import pytest
from tvm import tirx
from tvm_ffi import structural_map
from tvm_ffi.dataclasses import py_class

from tests.numsim.v2.checkers._runnable import requires_v2_engine
from tirx_harness.numsim import v2
from tirx_harness.numsim.errors import UnsupportedTIRxError

pytestmark = requires_v2_engine


def test_frontend_rejects_unknown_native_nodes_in_a_known_parent():
    """Port of ``tests/numsim/integration/test_native_statement_tree.py::test_frontend_rejects_unknown_native_nodes_in_a_known_parent``.

    ``analyze`` -> ``v2.transpile``. Legacy raised ``NumSimBuildError`` ("node
    kind without a native binding"); v2 fails closed with
    ``UnsupportedTIRxError`` whose ``unsupported`` entry names the unknown
    statement type. The reflected type key differs from the legacy test's
    (``...FutureStatementV2``) so both can register in one process."""

    @py_class("numsim.testing.FutureStatementV2")
    class FutureStatement(tirx.Stmt):
        body: tirx.Stmt

    known = tirx.PrimFunc([], tirx.SeqStmt([tirx.Evaluate(0), tirx.Evaluate(1)]))
    # Construct the extension through reflected fields: the current TVM
    # PrimFunc constructor's purity visitor cannot visit future node types.
    extended = structural_map(
        known,
        (
            tirx.Evaluate,
            lambda stmt: (
                FutureStatement(span=None, body=stmt) if int(stmt.value.value) == 1 else stmt
            ),
        ),
    )
    with pytest.raises(UnsupportedTIRxError) as excinfo:
        v2.transpile(extended)
    assert any("FutureStatementV2" in entry for entry in excinfo.value.unsupported), excinfo.value.unsupported
    assert not v2.transpile(known).document["kernels"][0]["unsupported"]
