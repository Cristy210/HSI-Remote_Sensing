from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import mlx.core as mx
from tqdm.auto import trange

Array = mx.array

def polar(X: Array) -> Array:
    """
    Compute the (orthogonal) polar factor of X using SVD.

    Parameters
    ----------
    X : (D, N) mx.array
        Data matrix.

    Returns
    -------
    Q : (K, N) mx.array
        Polar factor with orthonormal columns (up to numerical precision).
    """
    M, N = X.shape
    k = mx.minimum(M, N).item()
    U, S, Vt = mx.linalg.svd(X, stream=mx.cpu)
    return U[:, :k] @ Vt


class KSS:
    """
    K-Subspaces (KSS) Clusering

    Parameters
    ----------
    n_clusters: int
        Number of clusters.
    subspaces_dims = int or sequence of int
        Dimension(s) of each subspace. If an int is provided, all subspaces
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
        subspaces_dims:Union[int, Iterable[int]],
        max_iter:int=100,
        n_init:int=10,
        verbose:int=0,
        random_state:Optional[int]=None,
    ) -> None:
        
        self.n_clusters = int(n_clusters)
        self.subspaces_dims = subspaces_dims
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.verbose = int(verbose)
        self.random_state = random_state

        # attributes filled by fit()
        self.labels_:Optional[np.ndarray] = None
        self.subspaces_:Optional[List[Array]] = None
        self.cost_:Optional[float] = None
        
    def _check_susbpace_dims(self) -> List[int]:
        if isinstance(self.subspaces_dims, int):
            d = [int(self.subspaces_dims)] * self.n_clusters
            return d
        else:
            d = [int(di) for di in self.subspaces_dims]
            if len(d) != self.n_clusters:
                raise ValueError(
                    "Length of subspaces_dims must match n_clusters"
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
    
    @staticmethod
    def _cost(U: Sequence[Array], X:Array, labels: np.ndarray) -> float:
        """
        Compute cost: sum over i of ||U_k^T x_i||^2 for assigned cluster k.
        """

        K = len(U)
        scores = mx.stack(
            [mx.sum(mx.matmul(Uk.T, X, stream=mx.gpu) ** 2, axis=0) for Uk in U],
            axis=0,
        )
        scores_np = np.array(scores)
        c0 = labels
        return float(scores_np[c0, np.arange(scores_np.shape[1])].sum())
    
    def _kss_single(
        self,
        X: Array,
        d: Sequence[int],
        seed: Optional[int] = None,
    ) -> Tuple[List[Array], np.ndarray, float]:
        """
        Single run of KSS clustering.

        Parameters
        ----------
        X : (D, N) mx.array
            Data matrix with N points in R^D (columns are points).
        d : sequence of int, length K
            Subspace dimensions for each cluster.
        seed : int, optional
            Random seed for initialization.
        
        Returns
        -------
        U : list of (D, d_k) mx.array
            Learned subspace bases.
        c: (N,) np.ndarray of ints in 0...K-1
            Cluster labels.
        cost: float
            Cost of each run.
        """
        if seed is not None:
            mx.random.seed(seed)
        
        K = len(d)
        D, N = X.shape

        # Initialize subspaces
        U: List[Array] = [
            polar(mx.random.normal(shape=(D, dk)))
            for dk in d
        ]

        # Initial cluster assignment
        scores = mx.stack(
            [mx.sum(mx.matmul(Uk.T, X, stream=mx.gpu) ** 2, axis=0) for Uk in U],
            axis=0,
        )
        labels = np.argmax(np.array(scores), axis=0).astype(np.int32)
        labels_prev = labels.copy()

        # Iterations
        if self.verbose >= 2:
            iter_range = trange(self.max_iter, desc="KSS", leave=False)
        else:
            iter_range = range(self.max_iter)
        
        for t in iter_range:
            # Update subspaces
            for k in range(K):
                ilist = np.nonzero(labels == k)[0]

                if ilist.size == 0:
                    # Empty cluster: reinitialize its subspace
                    U[k] = polar(mx.random.normal(shape=(D, d[k])))
                    continue
                idx = mx.array(ilist)
                X_k = mx.take(X, idx, axis=1)
                A = mx.matmul(X_k, X_k.T, stream=mx.gpu)
                w, V = mx.linalg.eigh(A, stream=mx.cpu)
                U[k] = V[:, -d[k]:]

            # Update clusters
            scores = mx.stack(
                [mx.sum(mx.matmul(Uk.T, X, stream=mx.gpu) ** 2, axis=0) for Uk in U],
                axis=0,
            )
            labels = np.argmax(np.array(scores), axis=0).astype(np.int32)

            # Break if clusters did not change, update otherwise
            if np.array_equal(labels, labels_prev):
                if self.verbose >= 2:
                    print(f"KSS terminated early at iteration {t + 1}")
                break
            labels_prev = labels.copy()

        # Compute final cost
        cost = self._cost(U, X, labels)

        return U, labels, cost

    # Public API
    def fit(self, X) -> "KSS":
        """
        Compute K-Subspaces clustering.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or MLX array (D, N)
        Returns
        -------
        self : KSS
            Fitted estimator.
        """

        X_mx, D, N = self._to_mlx_column_major(X)
        d = self._check_susbpace_dims()

        if self.verbose >= 1:
            print(
                f"Running KSS with n_clusters={self.n_clusters}"
                f", subspaces_dims={d}, max_iter={self.max_iter}, n_init={self.n_init}"
            )

        best_cost = -np.inf
        best_labels = None
        best_U: Optional[List[Array]] = None

        for run in range(self.n_init):
            if self.verbose >= 1:
                print(f" KSS run {run + 1}/{self.n_init}")
            
            seed = None if self.random_state is None else self.random_state + run
            U_run, labels_run, cost_run = self._kss_single(X_mx, d, seed=seed)

            if self.verbose >= 1:
                print(f"  Run cost: {cost_run:.4e}")
            
            if cost_run > best_cost:
                best_cost = cost_run
                best_labels = labels_run.copy()
                best_U = [Uk for Uk in U_run]
        
        assert best_U is not None and best_labels is not None
        self.subspaces_ = best_U
        self.labels_ = best_labels.astype(np.int32) + 1
        self.cost_ = best_cost

        return self
    
    def fit_predict(self, X, y=None) -> np.ndarray:
        """
        Fit KSS and return labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or MLX array (D, N)

        Returns
        -------
        labels: (n_samples, )
            Labels in {1, ..., n_clusters}.
        """
        self.fit(X, y=y)
        return self.labels_
    
    def predict(self, X) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or MLX array (D, N)

        Returns
        -------
        labels: (n_samples, )
            Labels in {1, ..., n_clusters}.
        """

        if self.subspaces_ is None:
            raise ValueError("KSS instance is not fitted yet. Call 'fit' first.")
        
        X_mx, D, N = self._to_mlx_column_major(X)
        K = len(self.subspaces_)
        U = self.subspaces_

        scores = mx.stack(
            [
                mx.sum(mx.matmul(Uk.T, X_mx, stream=mx.gpu) ** 2, axis=0)
                for Uk in U
            ],
            axis=0,
        )
        labels = np.argmax(np.array(scores), axis=0).astype(np.int32) + 1

        return labels
