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

# ---------------------------
# Sparse Affinity Construction
# ----------------------------

def tsc_affinity(
    X: Union[np.ndarray, Array],
    *,
    max_nz: int = 15,
    max_chunksize: int = 1024,
    dtype=mx.float32,
    eps: float = 1e-12,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    use_gpu: bool = True,
):
    """
    Build a sparse TSC affinity matrix A (CSR) without forming dense NxN.

    Parameters
    ----------
    X : (D, N)
        Data matrix; columns are points (HSI pixels/features).
    max_nz : int
        Number of neighbors (q) to keep per point (directed). Total nnz ~ N*q.
    max_chunksize : int
        Number of query columns processed per chunk.
    dtype : mx dtype
        float32 recommended for speed/memory.
    symmetrize : bool
        If True, returns A + A.T (still sparse).
    zero_diagonal : bool
        If True, removes any diagonal entries.
    use_gpu : bool
        If True, uses mx.gpu stream for matmul.

    Returns
    -------
    A : scipy.sparse.csr_matrix (N, N)
    """
    try:
        import scipy.sparse as sp
    except Exception as e:
        raise ImportError(
            "tsc_affinity_sparse requires SciPy (scipy.sparse). "
            "Install with `pip install scipy`."
        ) from e
    
    X_mx = _as_mx(X, dtype=dtype)
    D, N = X_mx.shape

    if max_nz < 1:
        raise ValueError(f"max_nz must be >= 1, got {max_nz}.")
    q = int(min(max_nz, N))
    chunksize = int(min(max_chunksize, N))

    Y = _normalize_columns(X_mx, eps=eps)

    # Preallocate arrays for sparse affinity matrix construction
    nnz_est = N * q
    rows = np.empty(nnz_est, dtype=np.int32)
    cols = np.empty(nnz_est, dtype=np.int32)
    data = np.empty(nnz_est, dtype=np.float32 if dtype == mx.float32 else np.float64)

    stream = mx.gpu if use_gpu else mx.cpu

    write_ptr = 0
    n_chunks = (N + chunksize - 1) // chunksize

    for chunk_idx in range(n_chunks):
        j0 = chunk_idx * chunksize
        j1 = min(N, (chunk_idx + 1) * chunksize)
        m = j1-j0

        Y_chunk = Y[:, j0:j1]
        C_mx = mx.abs(mx.matmul(Y.T, Y_chunk, stream=stream))
        C = np.array(C_mx, copy=False)

        diag_rows = np.arange(j0, j1, dtype=np.int64)
        diag_cols = np.arange(0, m , dtype=dtype)
        C[diag_rows, diag_cols] = 0.0

        # Row Indices for each column's top-q similarities
        inds = np.argpartition(-C, q - 1, axis=0)[:q, :]
        sims = np.take_along_axis(C, inds, axis=0)

        order = np.argsort(-sims, axis=0)
        inds = np.take_along_axis(inds, order, axis=0)
        sims = np.take_along_axis(sims, order, axis=0)