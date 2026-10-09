import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: corpus-wide lowering sweep (minutes)")


@pytest.fixture
def lower_source():
    """Parse TVMScript and lower it; returns the Program."""
    import tvm
    from tvm.script import tirx as T
    from tvm.tirx.layout import Axis

    from tirx_harness.numsim.v2.lowering import lower

    def run(source: str, strict: bool = True):
        return lower(tvm.script.from_source(source, {"T": T, "Axis": Axis}), strict=strict)

    return run
