"""NumSim v2: TIRx -> ``Program`` bytecode, executed by ``core-rs``.

See ``docs/development/numsim-redesign.md``. The public surface (``api``) is
what ``tirx_harness.numsim`` re-exports; v2 is the only implementation.
"""

from .api import *  # noqa: F403
from .api import __all__
