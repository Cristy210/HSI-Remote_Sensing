from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import mlx.core as mx
from tqdm.auto import trange

Array = mx.array

def rand_affinespace(X: Array) -> Tuple[Array, Array]:
    """
    Affine approximation of the columns of X.

    Parameters
    ----------
    X : (D, N) mx.array
        Data matrix.

    Returns
    -------
    b_hat : (D,) mx.array
        Affine offset (cluster mean).
    U_hat : (D, k) mx.array
        Orthonormal basis for the linear part of the affine subspace.
    """
    
    M, N = X.shape
    k = mx.minimum(M, N).item()
    # Mean of columns
    b_hat = mx.mean(X, axis=1)
    # Centered data
    Xc = X - b_hat[:, None]

    U, S, Vt = mx.linalg.svd(Xc, stream=mx.cpu)
    return U[:, :k]@Vt, b_hat

class KAS:
    """
    K-Affine Subspaces (KAS) Clusering

    Parameters
    ----------
    n_clusters: int
        Number of clusters.
    affinespace_dims = int or sequence of int
        Dimension(s) of each affine space. If an int is provided, all affine spaces
        will have the same dimension.
    max_iter = int, default=100
        Maximum number of iterations.
    n_int = int, default=10
        Number of random initializations. The best run (Highest cost) is kept
    verbose = int, default=0
        Verbosity level. 0=Silent, 1=per-run messsages, 2=per-iter bar.
    random_state = int, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters:int,
        affinespace_dims:Union[int, Iterable[int]],
        max_iter:int=100,
        n_init:int=10,
        verbose:int=0,
        random_state:Optional[int]=None,
    ) -> None:
        
        self.n_clusters = int(n_clusters)
        self.affinespace_dims = affinespace_dims
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.verbose = int(verbose)
        self.random_state = random_state

        # attributes filled by fit()
        self.labels_:Optional[np.ndarray] = None
        self.affinespaces_:Optional[List[Array]] = None
        self.offsets_:Optional[List[Array]] = None
        self.cost_:Optional[float] = None

        def _check_affinespace_dims(self) -> List[int]:
        
            if isinstance(self.affinespace_dims, int):
                d = [int(self.affinespace_dims)] * self.n_clusters
                return d

            d = [int(di) for di in self.affinespace_dims]
            if len(d) != self.n_clusters:
                raise ValueError(
                    "Length of affinespace_dims must match n_clusters "
                    f"({len(d)} != {self.n_clusters})."
                )
            return d
        
    @staticmethod
    def _to_mlx_column_major(X) -> Tuple[Array, int, int]:
        """
        Convert input X to MLX array:

        Accepts:
            - Numpy Array of shape(n_samples, n_features)
            - MLX Array of shape (n_features, n_samples)
        """

        if isinstance(X, mx.array):
            D, N = X.shape
            return X, D, N
        
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim != 2:
            raise ValueError("Input data X must be 2D (n_samples, n_features).")
        
        n_samples, n_features = X_np.shape
        X_mlx = mx.array(X_np.T, dtype=mx.float64)
        return X_mlx, n_features, n_samples
    
    

