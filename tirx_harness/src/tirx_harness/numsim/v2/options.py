"""The one place NumSim v2 reads environment variables.

| variable | meaning | default |
| --- | --- | --- |
| ``NUMSIM_CACHE_DIR`` | cache root; v2 modules live under ``<root>/v2-modules`` | ``~/.cache/tirx-harness/numsim`` |
| ``NUMSIM_SEED`` | scheduler seed | ``0`` |
| ``NUMSIM_NO_CACHE`` | ``1`` disables the module cache | unset |
| ``NUMSIM_PIN_WORKERS`` | ``1`` pins the engine's worker threads (``Engine(pin_workers=...)``; dedicated hosts only, never set in CI) | unset (off) |
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    """``NUMSIM_<name>``; ``NUMSIM_V2_<name>`` is accepted for one release."""
    return os.environ.get(f"NUMSIM_{name}", os.environ.get(f"NUMSIM_V2_{name}", default))


@dataclass(frozen=True)
class Options:
    cache_root: Path
    seed: int
    use_cache: bool
    pin_workers: bool = False

    @property
    def module_cache_dir(self) -> Path:
        return self.cache_root / "v2-modules"


def options() -> Options:
    """Read the environment (cheap; call per operation so tests can monkeypatch)."""

    root = os.environ.get("NUMSIM_CACHE_DIR")
    cache_root = Path(root) if root else Path.home() / ".cache" / "tirx-harness" / "numsim"
    seed = int(_env("SEED", "0") or 0)
    use_cache = _env("NO_CACHE", "") not in {"1", "true", "yes"}
    pin_workers = _env("PIN_WORKERS", "") in {"1", "true", "yes"}
    return Options(cache_root=cache_root, seed=seed, use_cache=use_cache, pin_workers=pin_workers)


__all__ = ["Options", "options"]
