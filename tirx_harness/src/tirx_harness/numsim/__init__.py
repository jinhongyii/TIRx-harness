"""NumSim: TIRx -> ``Program`` bytecode executed by the Rust engine (``core-rs``).

The public names are the v2 implementation (``tirx_harness.numsim.v2``).
"""

from .cases import (
    ComparisonRegion,
    ComparisonSpec,
    ExecutionAssumptions,
    Im2col,
    NumSimCase,
    TensorMap,
)
from .errors import (
    NumSimBuildError,
    NumSimError,
    NumSimExecutionError,
    UnmodeledTIRxFormError,
    UnsupportedTIRxError,
)
from .v2 import *  # noqa: F403
from .v2 import __all__ as _v2_all

__all__ = sorted(
    {
        *_v2_all,
        "ComparisonRegion",
        "ComparisonSpec",
        "ExecutionAssumptions",
        "Im2col",
        "NumSimBuildError",
        "NumSimCase",
        "NumSimError",
        "NumSimExecutionError",
        "TensorMap",
        "UnmodeledTIRxFormError",
        "UnsupportedTIRxError",
    }
)
