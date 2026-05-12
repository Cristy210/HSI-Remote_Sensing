from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from tqdm.auto import trange


# -----------------
# Helper functions
# -----------------

@dataclass
class TSCResult:
    """
    Result object returned by TSC pipeline.

    Attributes
    ----------
    affinity : sp.csr_matrix
        Sparse symmetric affinity matrix of shape (N, N).

    embedding : np.ndarray
        Spectral embedding of shape (N, K).

    eigenvalues : np.ndarray
        Leading eigenvalues of normalized affinity.

    labels : np.ndarray
        Cluster assignments in {1, ..., K}.

    kmeans_model : object
        Fitted sklearn KMeans object.

    n_clusters : int
        Number of clusters used (estimated or user-provided).
    """
    affinity: sp.csr_matrix
    embedding: np.ndarray
    eigenvalues: np.ndarray
    labels: np.ndarray
    kmeans_model: object
    n_clusters: int


def _row_normalize(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-normalize a NumPy matrix.
    """
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, eps)


class TSC:
    """
    Thresholded Subspace Clustering (TSC)

    Parameters
    ----------
    n_clusters : int or None, default=None
        Number of clusters. If None, estimate using eigengap heuristic.
    max_nz : int, default=15
        Number of retained neighbors per point in the affinity graph.
    max_chunksize : int, default=1024
        Number of query columns processed per chunk during affinity construction.
    n_eig : int or None, default=None
        Number of leading eigenvectors/eigenvalues to compute. If None, a sensible
        value is chosen automatically.
    n_init : int, default=20
        Number of k-means initializations.
    max_iter : int, default=100
        Maximum number of k-means iterations.
    symmetrize : bool, default=True
        If True, use A + A.T after directed neighbor selection.
    normalize_embedding : bool, default=True
        If True, row-normalize the spectral embedding before k-means.
    verbose : int, default=0
        Verbosity level. 0=silent, 1=summary messages, 2=progress during affinity.
    random_state : int or None, default=None
        Random seed for reproducibility.
    dtype : numpy dtype, default=np.float64
        Floating-point dtype used for the NumPy pipeline.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        max_nz: int = 15,
        max_chunksize: int = 1024,
        n_eig: Optional[int] = None,
        n_init: int = 20,
        max_iter: int = 100,
        symmetrize: bool = True,
        normalize_embedding: bool = True,
        verbose: int = 0,
        random_state: Optional[int] = None,
        dtype=np.float64,
    ) -> None:
        self.n_clusters = None if n_clusters is None else int(n_clusters)
        self.max_nz = int(max_nz)
        self.max_chunksize = int(max_chunksize)
        self.n_eig = None if n_eig is None else int(n_eig)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.symmetrize = bool(symmetrize)
        self.normalize_embedding = bool(normalize_embedding)
        self.verbose = int(verbose)
        self.random_state = random_state
        self.dtype = dtype

        # attributes filled by fit()
        self.labels_: Optional[np.ndarray] = None
        self.affinity_: Optional[sp.csr_matrix] = None
        self.embedding_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.kmeans_ = None
        self.n_clusters_: Optional[int] = None
        self.result_: Optional[TSCResult] = None

    def _to_numpy_column_major(self, X) -> Tuple[np.ndarray, int, int]:
        """
        Convert input X to NumPy array of shape (D, N).

        Accepts
        -------
        - NumPy array of shape (n_features, n_samples)
        - array-like of shape (n_features, n_samples)
        """
        X_np = np.asarray(X, dtype=self.dtype)
        if X_np.ndim != 2:
            raise ValueError("Input data X must be 2D with shape (n_features, n_samples).")

        if self.verbose >= 2:
            print(f"[TSC] Converting input to NumPy array with dtype={X_np.dtype}, shape={X_np.shape}")

        D, N = X_np.shape
        return X_np, D, N

    @staticmethod
    def _normalize_columns(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Normalize columns of X to unit l2 norm.
        """
        norms = np.linalg.norm(X, axis=0) + eps
        return X / norms

    @staticmethod
    def _estimate_k_eigengap(
        eigenvalues: np.ndarray,
        kmax: Optional[int] = None,
        min_k: int = 2,
    ) -> int:
        """
        Estimate number of clusters from descending eigenvalues.
        """
        vals = np.asarray(eigenvalues).ravel()
        if vals.size < 2:
            raise ValueError("Need at least two eigenvalues for eigengap estimation.")

        upper = vals.size - 1 if kmax is None else min(int(kmax), vals.size - 1)
        lower = max(1, int(min_k) - 1)
        if lower > upper:
            lower = 1

        gaps = vals[:-1] - vals[1:]
        idx = np.argmax(gaps[lower - 1:upper]) + (lower - 1)
        return int(idx + 1)

    def _build_affinity(self, X: np.ndarray) -> sp.csr_matrix:
        """
        Build sparse TSC affinity matrix.
        """
        _, N = X.shape

        if self.max_nz < 1:
            raise ValueError(f"max_nz must be >= 1, got {self.max_nz}.")

        q = min(self.max_nz, max(N - 1, 1))
        chunksize = min(self.max_chunksize, N)
        n_chunks = (N + chunksize - 1) // chunksize

        Y = self._normalize_columns(X)

        nnz_est = N * q
        rows = np.empty(nnz_est, dtype=np.int32)
        cols = np.empty(nnz_est, dtype=np.int32)
        data = np.empty(nnz_est, dtype=self.dtype)

        write_ptr = 0

        if self.verbose >= 1:
            print(
                f"[TSC] Building affinity matrix | "
                f"N={N}, q={q}, chunksize={chunksize}, n_chunks={n_chunks}, backend=CPU"
            )

        if self.verbose >= 2:
            chunk_iter = trange(n_chunks, desc="TSC affinity", leave=False)
        else:
            chunk_iter = range(n_chunks)

        for chunk_idx in chunk_iter:
            j0 = chunk_idx * chunksize
            j1 = min(N, (chunk_idx + 1) * chunksize)
            m = j1 - j0

            Y_chunk = Y[:, j0:j1]
            C = np.abs(Y.T @ Y_chunk)

            # zero self-similarities for current chunk columns
            local_cols = np.arange(m, dtype=np.int64)
            global_rows = np.arange(j0, j1, dtype=np.int64)
            C[global_rows, local_cols] = 0.0

            inds = np.argpartition(-C, q - 1, axis=0)[:q, :]
            sims = np.take_along_axis(C, inds, axis=0)

            order = np.argsort(-sims, axis=0)
            inds = np.take_along_axis(inds, order, axis=0)
            sims = np.take_along_axis(sims, order, axis=0)

            block_nnz = q * m
            rows[write_ptr:write_ptr + block_nnz] = inds.reshape(-1, order="F")
            cols[write_ptr:write_ptr + block_nnz] = np.repeat(
                np.arange(j0, j1, dtype=np.int32), q
            )
            data[write_ptr:write_ptr + block_nnz] = sims.reshape(-1, order="F")
            write_ptr += block_nnz

        A = sp.csr_matrix(
            (data[:write_ptr], (rows[:write_ptr], cols[:write_ptr])),
            shape=(N, N),
        )

        A.setdiag(0)
        A.eliminate_zeros()

        if self.symmetrize:
            A = A + A.T
            A.setdiag(0)
            A.eliminate_zeros()

        if self.verbose >= 1:
            print(f"[TSC] Affinity construction finished | nnz={A.nnz}")

        return A.tocsr()

    def _spectral_embedding(self, A: sp.csr_matrix):
        """
        Compute spectral embedding from sparse affinity matrix.
        """
        from scipy.sparse.linalg import eigsh

        N = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Affinity matrix must be square, got {A.shape}.")

        degrees = np.asarray(A.sum(axis=1)).ravel()
        inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
        D_inv_sqrt = sp.diags(inv_sqrt_deg)
        L = D_inv_sqrt @ A @ D_inv_sqrt

        if self.n_clusters is None:
            nev = self.n_eig
            if nev is None:
                nev = min(max(10, 2), N - 1)
        else:
            nev = self.n_eig if self.n_eig is not None else max(self.n_clusters, 2)
            nev = min(nev, N - 1)

        if self.verbose >= 1:
            mode = "eigengap" if self.n_clusters is None else f"fixed K={self.n_clusters}"
            print(f"[TSC] Computing spectral embedding | nev={nev}, mode={mode}")

        if N == 1:
            vals = np.array([1.0], dtype=self.dtype)
            vecs = np.ones((1, 1), dtype=self.dtype)
        elif nev >= N - 1:
            if self.verbose >= 2:
                print("[TSC] Using dense eigendecomposition")
            dense = L.toarray()
            vals, vecs = np.linalg.eigh(dense)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
        else:
            if self.verbose >= 2:
                print("[TSC] Using sparse eigsh")
            vals, vecs = eigsh(L, k=nev, which="LA")
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

        if self.n_clusters is None:
            if self.verbose >= 1:
                print("[TSC] Estimating number of clusters using eigengap heuristic")
            K = self._estimate_k_eigengap(vals)
            if self.verbose >= 1:
                print(f"[TSC] Selected n_clusters={K}")
        else:
            K = self.n_clusters

        embedding = np.asarray(vecs[:, :K], dtype=self.dtype)

        if self.normalize_embedding:
            embedding = _row_normalize(embedding)

        return embedding, np.asarray(vals), int(K)

    def _cluster_embedding(self, embedding: np.ndarray):
        """
        Run k-means on spectral embedding.
        """
        try:
            from sklearn.cluster import KMeans
        except Exception as e:
            raise ImportError(
                "scikit-learn is required for k-means. "
                "Install it with: pip install scikit-learn"
            ) from e

        if self.verbose >= 1:
            print(
                f"[TSC] Running k-means on embedding with shape={embedding.shape}, "
                f"n_clusters={self.n_clusters_}, n_init={self.n_init}, max_iter={self.max_iter}"
            )

        km = KMeans(
            n_clusters=self.n_clusters_,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        labels = km.fit_predict(embedding).astype(np.int32)
        return labels, km

    # Public API
    def fit(self, X) -> "TSC":
        """
        Compute Thresholded Subspace Clustering.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)

        Returns
        -------
        self : TSC
            Fitted estimator.
        """
        X_np, _, N = self._to_numpy_column_major(X)

        if self.n_clusters is not None and not (1 <= self.n_clusters <= N):
            raise ValueError(
                f"n_clusters must satisfy 1 <= n_clusters <= N, got {self.n_clusters}."
            )

        if self.verbose >= 1:
            print(
                f"Running TSC with n_clusters={self.n_clusters}, "
                f"max_nz={self.max_nz}, max_chunksize={self.max_chunksize}, "
                f"n_init={self.n_init}, max_iter={self.max_iter}"
            )

        A = self._build_affinity(X_np)
        embedding, eigenvalues, K_used = self._spectral_embedding(A)

        self.n_clusters_ = K_used
        labels, km = self._cluster_embedding(embedding)
        labels = labels + 1

        self.affinity_ = A
        self.embedding_ = embedding
        self.eigenvalues_ = eigenvalues
        self.kmeans_ = km
        self.labels_ = labels

        self.result_ = TSCResult(
            affinity=A,
            embedding=embedding,
            eigenvalues=eigenvalues,
            labels=labels,
            kmeans_model=km,
            n_clusters=K_used,
        )

        if self.verbose >= 1:
            print(f"Finished TSC with n_clusters_={self.n_clusters_}")

        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        """
        Fit TSC and return labels.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Labels in {1, ..., n_clusters}.
        """
        self.fit(X)
        return self.labels_