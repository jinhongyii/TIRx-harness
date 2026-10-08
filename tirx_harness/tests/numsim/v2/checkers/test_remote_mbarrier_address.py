"""Local-form mbarrier arrive / expect_tx on a ``mapa``-mapped remote address.

Replaces two ``gap_unportable`` rows of
``tests/analysis_tools/synccheck/test_native_synccheck_artifact.py``. The
check is decided at address resolution (``mapa`` + local-form arrive), not by
a contract event. sync-semantics.md (mbarrier.arrive, "A local form naming a
remote CTA is rejected with ``MbarrierLocalArriveRemoteAddress``") keeps the
legacy kind ``mbarrier_local_arrive_remote_address``. The v2 interpreter
rejects the access at address resolution as the runtime diagnostic
``bad_address`` ("shared::cta address ... names CTA rank 0, not the executing
CTA"); no delta row records that rename, so both names are accepted.
"""

from __future__ import annotations

import numpy as np

from tests.numsim.support.kernels import mapped_remote_mbarrier_pointer
from tests.numsim.support.remote_mbarrier import mapped_remote_mbarrier_pointer_expect_tx
from tirx_harness.numsim import v2

from ._runnable import assert_error_kind, coverage_bounds, requires_v2_engine, resource_limits

pytestmark = requires_v2_engine

_KIND = {"mbarrier_local_arrive_remote_address", "bad_address"}


def _sync(kernel):
    return v2.synccheck(
        kernel,
        {"output": np.zeros(2, dtype=np.int32)},
        coverage_bounds=coverage_bounds(),
        resource_limits=resource_limits(),
    )


def test_synccheck_rejects_local_arrive_on_mapped_remote_address():
    """Replaces ``tests/analysis_tools/synccheck/test_native_synccheck_artifact.py::test_public_native_synccheck_rejects_local_arrive_on_mapped_remote_address``.

    CTA 1 arrives with the local (``.shared``) form on an address ``mapa``-ed to
    CTA 0: error ``mbarrier_local_arrive_remote_address``.
    """

    assert_error_kind(_sync(mapped_remote_mbarrier_pointer), _KIND)


def test_synccheck_rejects_local_expect_tx_on_mapped_remote_address():
    """Replaces ``tests/analysis_tools/synccheck/test_native_synccheck_artifact.py::test_public_native_synccheck_rejects_local_expect_tx_on_mapped_remote_address``.

    Same as above for the local-form ``mbarrier.arrive.expect_tx``.
    """

    assert_error_kind(_sync(mapped_remote_mbarrier_pointer_expect_tx), _KIND)
