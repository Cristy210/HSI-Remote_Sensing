from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import mlx.core as mx
from tqdm.auto import trange

Array = mx.array

def affine_approx(X: Array, k: int) -> Tuple[Array, Array]:
    """
    Generate a d-dimensional affine space. 

    Parameters
    ----------
    X : (D, N) mx.array
        Data matrix.
    k : int
        Dimension of the affine space.

    Returns
    -------
    U_hat : (D, k) mx.array
        Orthonormal basis for the linear part of the affine space.
    b_hat : (D,) mx.array
        Affine offset (cluster mean).
    """
    
    # Mean of columns
    b_hat = mx.mean(X, axis=1)
    # Centered data
    Xc = X - b_hat[:, None]

    U, S, Vt = mx.linalg.svd(Xc, stream=mx.cpu)
    return U[:, :k], b_hat

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
            - Numpy Array of shape(n_features, n_samples)
            - MLX Array of shape (n_features, n_samples)
        """

        if isinstance(X, mx.array):
            X_mx = X if X.dtype == mx.float32 else mx.astype(X, mx.float32)
            if X_mx.shape[0] >= X_mx.shape[1]:
                X_mx = X_mx.T
            return X_mx

        if isinstance(X, np.ndarray):
            X_np = X.astype(np.float32, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        if X_np.ndim != 2:
            raise ValueError("Input data X must be 2D.")

        n_rows, n_cols = X_np.shape
        # transpose only if rows >= cols (typical for (N, D))
        X_mx = mx.array(X_np.T if n_rows >= n_cols else X_np, dtype=mx.float32)
        return X_mx
    
    @staticmethod
    def _scores(U:Sequence[Array], b: Sequence[Array], X:Array) -> Array:
        """
        Returns squared residual distances
        """
        scores = mx.stack([
            mx.sum((X - b_k[:, None])**2, axis=0) -
            mx.sum(mx.matmul(U_k.T, X - b_k[:, None], stream=mx.gpu)**2, axis=0)
            for U_k, b_k in zip(U, b)], axis=0)
        return np.array(scores)
    
    @staticmethod
    def _cost(U: Sequence[Array], b: Sequence[Array], X:Array, labels: np.ndarray) -> float:
        """
        Compute cost: sum over i of ||x_i - b_k||^2 - (Uk*U_k^T (x_i - b_k))||^2 
        for assigned cluster k.
        """
        scores = KAS._scores(U, b, X)

        # c0 = labels
        return float(scores[labels, np.arange(scores.shape[1])].sum())
    
    def _kas_single(
        self,
        X: Array,
        d: Sequence[int],
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Array], List[Array], float]:
        """
        Single run of KAS clustering.

        Parameters
        ----------
        X : (D, N) mx.array
            Data matrix with N points in R^D (columns are points).
        d : sequence of int, length K
            affine space dimensions for each cluster.
        seed : int, optional
            Random seed for initialization.

        Returns
        -------
        U : list of (D, d_k) mx.array
            Learned affine space bases.
        b : list of (D,) mx.array
            Learned offsets.
        c: (N,) np.ndarray of ints in 0...K-1
            Cluster labels.
        cost: float
            Cost of each run.
        """
        if seed is not None:
            mx.random.seed(seed)
        
        K = len(d)
        D, N = X.shape

        # Intialize offsets and bases
        U: List[Array] = []
        b : List[Array] = []
        for dk in d:
            Uk, bk = affine_approx(mx.random.normal(shape=(D, dk)), dk)
            U.append(Uk)
            b.append(bk)

        # Initial cluster assignment
        labels = np.argmin(KAS._scores(U, b, X), axis=0).astype(np.int32)
        labels_prev = labels.copy()

        # Iterations
        if self.verbose >= 2:
            iter_range = trange(self.max_iter, desc="KAS", leave=False)
        else:
            iter_range = range(self.max_iter)
        
        for t in iter_range:
            # Update offsets and bases
            for k in range(K):
                ilist = np.nonzero(labels == k)[0]

                if ilist.size == 0:
                    # Empty cluster, reinitialize
                    U[k], b[k] = affine_approx(mx.random.normal(shape=(D, d[k])), d[k])
                    continue
                idx = mx.array(ilist)
                Xk = mx.take(X, idx, axis=1)
                U[k], b[k] = affine_approx(Xk, d[k])
            
            # Update clusters
            labels = np.argmin(KAS._scores(U, b, X), axis=0).astype(np.int32)

            # Break if clusters did not change, update otherwise
            if np.array_equal(labels, labels_prev):
                if self.verbose >= 2:
                    print(f"KAS terminated early at iteration {t + 1}")
                break
            labels_prev = labels.copy()
        
        # Compute final cost
        cost = self._cost(U, b, X, labels)
        return U, b, labels, cost
    
    # Public API
    def fit(self, X) -> "KAS":
        """
        Compute KAS clustering.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) or MLX array (D, N)
            Input data.
        Returns
        -------
        self : KAS
            Fitted estimator.
        """

        X_mx = self._to_mlx_column_major(X)
        d = self._check_affinespace_dims()

        if self.verbose >= 1:
            print(
                f"Running KAS with n_clusters={self.n_clusters}"
                f", affine space dims={d}, max_iter={self.max_iter}, n_init={self.n_init}"
            )
        
        best_cost = np.inf
        best_labels = None
        best_U: Optional[List[Array]] = None
        best_b : Optional[List[Array]] = None

        for run in range(self.n_init):
            if self.verbose >= 1:
                print(f" KAS run {run + 1}/{self.n_init}")
            
            seed = None if self.random_state is None else self.random_state + run
            U_run, b_run, labels_run, cost_run = self._kas_single(X_mx, d, seed=seed)

            if self.verbose >= 1:
                print(f"  Run cost: {cost_run:.4e}")

            if cost_run < best_cost:
                best_cost = cost_run
                best_labels = labels_run.copy()
                best_U = [Uk for Uk in U_run]
                best_b = [bk for bk in b_run]
        
        assert best_U is not None and best_labels is not None and best_b is not None
        self.affinespaces_ = best_U
        self.offsets_ = best_b
        self.labels_ = best_labels.astype(np.int32) + 1
        self.cost_ = best_cost

        return self
    
    def fit_predict(self, X, y=None) -> np.ndarray:
        """
        Fit KAS and return labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or MLX array (D, N)

        Returns
        -------
        labels: (n_samples, )
            Labels in {1, ..., n_clusters}.
        """
        self.fit(X)
        return self.labels_
    
    def predict(self, X) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples) or MLX array (D, N)

        Returns
        -------
        labels: (n_samples, )
            Labels in {1, ..., n_clusters}.
        """

        if self.affinespaces_ is None:
            raise ValueError("KAS instance is not fitted yet. Call 'fit' first.")
        
        X_mx = self._to_mlx_column_major(X)
        U = self.affinespaces_
        b = self.offsets_

        labels = np.argmin(KAS._scores(U, b, X_mx), axis=0).astype(np.int32) + 1
        return labels






    

