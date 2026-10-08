"""Installable tooling for TIRx kernel development."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvm.tir import PrimFunc

__version__ = version("tirx-harness")

__all__ = ["racecheck", "synccheck"]


def synccheck(kernel: PrimFunc, inputs: dict | None = None, **options):
    """Run synchronization analysis for one concrete TIRx invocation."""
    from .numsim.v2 import synccheck as _synccheck

    return _synccheck(kernel, inputs, **options)


def racecheck(kernel: PrimFunc, inputs: dict | None = None):
    """Run data-race analysis for one concrete TIRx invocation."""
    from .numsim.v2 import racecheck as _racecheck

    return _racecheck(kernel, inputs)
