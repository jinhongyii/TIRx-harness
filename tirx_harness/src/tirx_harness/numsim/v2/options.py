"""The one place NumSim v2 reads environment variables.

| variable | meaning | default |
| --- | --- | --- |
| ``NUMSIM_CACHE_DIR`` | cache root; v2 modules live under ``<root>/v2-modules`` | ``~/.cache/tirx-harness/numsim`` |
| ``NUMSIM_V2_SEED`` | scheduler seed | ``0`` |
| ``NUMSIM_V2_NO_CACHE`` | ``1`` disables the module cache | unset |
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Options:
    cache_root: Path
    seed: int
    use_cache: bool

    @property
    def module_cache_dir(self) -> Path:
        return self.cache_root / "v2-modules"


def options() -> Options:
    """Read the environment (cheap; call per operation so tests can monkeypatch)."""

    root = os.environ.get("NUMSIM_CACHE_DIR")
    cache_root = Path(root) if root else Path.home() / ".cache" / "tirx-harness" / "numsim"
    seed = int(os.environ.get("NUMSIM_V2_SEED", "0") or 0)
    use_cache = os.environ.get("NUMSIM_V2_NO_CACHE", "") not in {"1", "true", "yes"}
    return Options(cache_root=cache_root, seed=seed, use_cache=use_cache)


__all__ = ["Options", "options"]
