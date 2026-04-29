from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import mlx.core as mx
from tqdm.auto import trange

Array = mx.array

class KMeans:
    """
    K-Means Clustering using Apple MLX

    Parameters
    ----------
    n_clusters: int
        Number of clusters
    max_iter: int, default = 100
        Maximum number of iterations
    n_init: int, default = 10
        Number of random initializations
    tol: float, default=1e-4
        Convergence tolerance based on centroid shift
    use_gpu: bool, default=True
        Whether to use MLX GPU stream for distance computations
    verbose:int, default=0
        Verbosity Level. 0=silent, 1=summary, 2=progress bar.
    random_state: int or None
        Random Seed
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter:int = 100,
        n_init:int = 10,
        tol:float = 1e-4,
        use_gpu:bool = True,
        verbose:int = 0,
        random_state:Optional[int] = None,
    ) -> None:
        
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.use_gpu = bool(use_gpu)
        self.verbose = int(verbose)
        self.random_state = random_state

        # attributes filled by fit()
        self.labels_:Optional[np.ndarray] = None
        self.centers_:Optional[Array] = None
        self.inertia_:Optional[float] = None
        self.n_iter_:Optional[int] = None
    
    @staticmethod
    def _to_mlx_column_major(X) -> Tuple[Array, int, int]:
        """
        Convert input X to MLX array:

        Accepts:
            - Numpy Array of shape(n_features, n_samples)
            - MLX Array of shape (n_features, n_samples)
        """

        if isinstance(X, mx.array):
            D, N = X.shape
            return X, D, N
        
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim != 2:
            raise ValueError("Input data X must be 2D (n_features, n_samples).")
        
        
        X_mlx = mx.array(X_np.T, dtype=mx.float64)
        return X_mlx