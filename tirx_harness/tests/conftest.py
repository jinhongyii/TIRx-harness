from __future__ import annotations

import os

# Many engine processes share the host under `-n 16`; worker-thread
# confinement would stack them on the cores that were idle at launch time.
os.environ["NUMSIM_WORKER_AFFINITY"] = "off"


def pytest_addoption(parser) -> None:
    # Declared at the tests root so every invocation (full run or a targeted
    # one) accepts it. Consumed by tests/conformance/test_conformance.py.
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="rewrite tests/conformance/snapshots from NumSim instead of comparing",
    )
