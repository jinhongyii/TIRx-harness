"""TIRx -> ``Program`` lowering for NumSim v2.

``lower(func)`` returns a provisional :class:`~.program_builder.Program` that
mirrors the ``numsim-core`` contract (see ``# CONTRACT:`` markers) and
serializes to serde-compatible JSON with ``Program.to_json()``.
"""

from .ir_walk import LoweringUnsupported, lower
from .program_builder import Program

__all__ = ["LoweringUnsupported", "Program", "lower"]
