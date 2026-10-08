from __future__ import annotations


def pytest_addoption(parser) -> None:
    # Declared at the tests root so every invocation (full run or a targeted
    # one) accepts it. Consumed by tests/conformance/test_conformance.py.
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="rewrite tests/conformance/snapshots from NumSim instead of comparing",
    )
