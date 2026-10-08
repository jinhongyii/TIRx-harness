"""TIRx -> ``numsim_core::Program`` lowering for NumSim v2.

``lower(func)`` returns a :class:`~.program_builder.Program`; ``lower_module``
wraps one or more kernels in a :class:`~.program_builder.Module` whose
``to_json()`` deserializes with ``numsim_core::program::Module::from_json``.
"""

from .ir_walk import LoweringUnsupported, lower, lower_module
from .program_builder import Module, Program

__all__ = ["LoweringUnsupported", "Module", "Program", "lower", "lower_module"]
