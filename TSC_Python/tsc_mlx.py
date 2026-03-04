from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import mlx.core as mx
from tqdm.auto import trange

Array = mx.array

# -------------
# Utilities
# -------------

def _as_mx(X: Union[np.ndarray, Array], dtype=mx.float32) -> Array:
    """Ensure X is an MLX array."""
    if isinstance(X, mx.array):
        if X.ndim != 2:
            raise ValueError("X must be 2D with shape (D, N)")
        return X.astype(dtype)
    X = np.asarray(X)
    if X.ndim !=2:
        raise ValueError("X must be 2D with shape (D, N)")
    return mx.array(X, dtype=dtype)

def _normalize_columns(X: Array, eps: float=1e-12) -> Array:
    """Column-wise ℓ2 normalization for (D, N) matrices."""
    norms = mx.linalg.norm(X, axis=0) + eps
    return X/norms

