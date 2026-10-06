

from __future__ import annotations

import os
import numpy as np


def _select_backend():
    if os.environ.get("FORCE_NUMPY", "").lower() in {"1", "true", "yes"}:
        return np

    try:
        import cupy as cp  # type: ignore

        # Importing CuPy can succeed even when no usable CUDA device/runtime exists.
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp
        except Exception:
            pass
    except Exception:
        pass

    return np


xp = _select_backend()
BACKEND_NAME = "cupy" if xp.__name__ == "cupy" else "numpy"


def _to_numpy(a):
    if xp.__name__ == "cupy":
        return xp.asnumpy(a)
    return np.asarray(a)
