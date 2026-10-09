"""Host reference for PTX ``add.rz.ftz.f32x2`` (shared by the triage tile copies)."""

from __future__ import annotations

import numpy as np


def add_rz_ftz_f32(a, b) -> np.ndarray:
    """binary32 ``a + b`` rounded toward zero, subnormal results flushed (operands are normal here)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    exact = a.astype(np.float64) + b.astype(np.float64)  # exact for two f32 operands of this range
    nearest = exact.astype(np.float32)
    overshoot = np.abs(nearest.astype(np.float64)) > np.abs(exact)
    result = np.where(overshoot, np.nextafter(nearest, np.float32(0)), nearest).astype(np.float32)
    tiny = np.abs(result) < np.finfo(np.float32).tiny
    return np.where(tiny, np.copysign(np.float32(0), result), result).astype(np.float32)
