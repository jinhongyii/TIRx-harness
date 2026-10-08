"""NumSim v2: TIRx -> ``Program`` bytecode, executed by ``core-rs``.

See ``docs/development/numsim-redesign.md``. The public surface (``api``)
mirrors the legacy ``tirx_harness.numsim`` names; it is not wired into the
legacy entry points. Conformance runs select it with ``NUMSIM_IMPL=v2``.
"""

from .api import *  # noqa: F403
from .api import __all__
