function [A, embedding, eigenvalues, c, K_used] = tsc(X, K, varargin)
%TSC Thresholded Subspace Clustering (TSC)
%
%   [A, embedding, eigenvalues, c, K_used] = tsc(X, K, varargin)
%
%   Inputs
%   ------
%   X : D x N numeric matrix
%       Data matrix whose columns are data points.
%
%   K : positive integer or []
%       Number of clusters. If empty, estimate K using eigengap heuristic.
%
%   Name-Value Parameters
%   ---------------------
%   'MaxNZ'              : number of retained neighbors per point (default: 30)
%   'ChunkSize'          : number of query columns processed at once during
%                          affinity construction (default: 1024)
%   'NInit'              : number of k-means initializations (default: 100)
%   'MaxIter'            : maximum k-means iterations (default: 100)
%   'Symmetrize'         : whether to use A + A' (default: true)
%   'NormalizeEmbedding' : whether to row-normalize embedding (default: true)
%   'NEig'               : number of leading eigenvectors/eigenvalues to compute
%                          (default: [] -> automatic choice)
%   'Verbose'            : print progress messages (default: false)
%
%   Outputs
%   -------
%   A           : N x N sparse affinity matrix
%   embedding   : N x K_used spectral embedding
%   eigenvalues : leading eigenvalues of normalized affinity matrix
%   c           : 1 x N cluster labels in {1, ..., K_used}
%   K_used      : number of clusters used (estimated or user-provided)

    p = inputParser;
    p.FunctionName = "tsc";

    addParameter(p, 'MaxNZ', 30, @(z) isscalar(z) && z >= 1);
    addParameter(p, 'ChunkSize', 1024, @(z) isscalar(z) && z >= 1);
    addParameter(p, 'NInit', 100, @(z) isscalar(z) && z >= 1);
    addParameter(p, 'MaxIter', 100, @(z) isscalar(z) && z >= 1);
    addParameter(p, 'Symmetrize', true, @(z) islogical(z) || z == 0 || z == 1);
    addParameter(p, 'NormalizeEmbedding', true, @(z) islogical(z) || z == 0 || z == 1);
    addParameter(p, 'NEig', [], @(z) isempty(z) || (isscalar(z) && z >= 1));
    addParameter(p, 'Verbose', false, @(z) islogical(z) || z == 0 || z == 1);

    parse(p, varargin{:});

    max_nz = p.Results.MaxNZ;
    chunk_size = p.Results.ChunkSize;
    n_init = p.Results.NInit;
    max_iter = p.Results.MaxIter;
    symmetrize = logical(p.Results.Symmetrize);
    normalize_embedding = logical(p.Results.NormalizeEmbedding);
    n_eig = p.Results.NEig;
    verbose = logical(p.Results.Verbose);

    if ~isnumeric(X) || ~ismatrix(X)
        error('Input X must be a numeric 2D matrix of size D x N.');
    end

    [~, N] = size(X);

    if ~isempty(K)
        if ~(isscalar(K) && K >= 1 && K <= N)
            error('K must satisfy 1 <= K <= N, or be empty [].');
        end
        K = double(K);
    end

    if verbose
        fprintf('Running TSC with K=%s, max_nz=%d, chunk_size=%d, n_init=%d, max_iter=%d\n', ...
            mat2str(K), max_nz, chunk_size, n_init, max_iter);
    end

    % -------------------------
    % Step 1: Build affinity
    % -------------------------
    A = build_affinity_chunked(X, max_nz, chunk_size, symmetrize, verbose);

    % -------------------------
    % Step 2: Spectral embedding
    % -------------------------
    [embedding, eigenvalues, K_used] = spectral_embedding( ...
        A, K, n_eig, normalize_embedding, verbose);

    % -------------------------
    % Step 3: k-means on embedding
    % -------------------------
    if verbose
        fprintf('Running k-means on embedding of size %d x %d with K=%d\n', ...
            size(embedding,1), size(embedding,2), K_used);
    end

    c = kmeans(embedding, K_used, ...
        'Replicates', n_init, ...
        'MaxIter', max_iter);

    c = c(:).';

    if verbose
        fprintf('Finished TSC with K_used=%d\n', K_used);
    end
end

% -------------------------------------------------------------------------
% Helper: Build sparse TSC affinity matrix using chunked similarity blocks
% -------------------------------------------------------------------------
function A = build_affinity_chunked(X, max_nz, chunk_size, symmetrize, verbose)
    [~, N] = size(X);

    if max_nz < 1
        error('MaxNZ must be >= 1.');
    end

    q = min(max_nz, max(N - 1, 1));
    chunk_size = min(chunk_size, N);
    n_chunks = ceil(N / chunk_size);

    % Normalize columns once
    Y = normalize_columns(X);

    % Preallocate sparse triplets
    nnz_est = N * q;
    rows = zeros(nnz_est, 1);
    cols = zeros(nnz_est, 1);
    vals = zeros(nnz_est, 1);

    write_ptr = 1;

    if verbose
        fprintf('Building chunked affinity matrix | N=%d, q=%d, chunk_size=%d, n_chunks=%d\n', ...
            N, q, chunk_size, n_chunks);
    end

    for chunk_idx = 1:n_chunks
        j0 = (chunk_idx - 1) * chunk_size + 1;
        j1 = min(N, chunk_idx * chunk_size);
        m = j1 - j0 + 1;

        Ychunk = Y(:, j0:j1);

        % Similarity block: N x m
        C = abs(Y' * Ychunk);

        % Zero self-similarities for this block
        local_cols = 1:m;
        global_rows = j0:j1;
        linear_idx = sub2ind([N, m], global_rows(:), local_cols(:));
        C(linear_idx) = 0;

        % For each query column in the chunk, keep top-q neighbors
        for t = 1:m
            sims_col = C(:, t);
            [top_vals, top_idx] = maxk(sims_col, q);

            block_start = write_ptr + (t - 1) * q;
            block_end = block_start + q - 1;

            rows(block_start:block_end) = top_idx;
            cols(block_start:block_end) = j0 + t - 1;
            vals(block_start:block_end) = top_vals;
        end

        write_ptr = write_ptr + q * m;

        if verbose
            fprintf('  Processed chunk %d / %d (columns %d:%d)\n', ...
                chunk_idx, n_chunks, j0, j1);
        end
    end

    used = 1:(write_ptr - 1);
    A = sparse(rows(used), cols(used), vals(used), N, N);

    % Zero diagonal
    A = A - spdiags(diag(A), 0, N, N);
    A = sparse(A);

    if symmetrize
        A = A + A.';
        A = A - spdiags(diag(A), 0, N, N);
        A = sparse(A);
    end

    if verbose
        fprintf('Affinity construction finished | nnz(A)=%d\n', nnz(A));
    end
end

% -------------------------------------------------------------------------
% Helper: Spectral embedding from sparse affinity
% -------------------------------------------------------------------------
function [embedding, eigenvalues, K_used] = spectral_embedding(A, K, n_eig, normalize_embedding, verbose)
    N = size(A, 1);

    if size(A,1) ~= size(A,2)
        error('Affinity matrix must be square.');
    end

    degrees = full(sum(A, 2));
    inv_sqrt_deg = 1 ./ sqrt(max(degrees, 1e-12));
    D_inv_sqrt = spdiags(inv_sqrt_deg, 0, N, N);

    L = D_inv_sqrt * A * D_inv_sqrt;

    if isempty(K)
        if isempty(n_eig)
            nev = min(10, N);
        else
            nev = min(n_eig, N);
        end
        mode_str = 'eigengap';
    else
        if isempty(n_eig)
            nev = min(max(K, 2), N);
        else
            nev = min(n_eig, N);
        end
        mode_str = sprintf('fixed K=%d', K);
    end

    if verbose
        fprintf('Computing spectral embedding | nev=%d, mode=%s\n', nev, mode_str);
    end

    if N == 1
        eigenvalues = 1;
        vecs = 1;
    else
        if nev == N
            [V, Dmat] = eig(full(L));
            vals = diag(Dmat);
            [eigenvalues, order] = sort(vals, 'descend');
            vecs = V(:, order);
        else
            opts.isreal = true;
            opts.issym = true;
            try
                [vecs, Dmat] = eigs(L, nev, 'largestreal', opts);
                vals = diag(Dmat);
                [eigenvalues, order] = sort(vals, 'descend');
                vecs = vecs(:, order);
            catch
                [V, Dmat] = eig(full(L));
                vals = diag(Dmat);
                [eigenvalues, order] = sort(vals, 'descend');
                vecs = V(:, order);
                eigenvalues = eigenvalues(1:nev);
                vecs = vecs(:, 1:nev);
            end
        end
    end

    if isempty(K)
        if verbose
            fprintf('Estimating K using eigengap heuristic\n');
        end
        K_used = estimate_k_eigengap(eigenvalues);
        if verbose
            fprintf('Selected K = %d\n', K_used);
        end
    else
        K_used = K;
    end

    embedding = vecs(:, 1:K_used);

    if normalize_embedding
        embedding = row_normalize(embedding);
    end
end

% -------------------------------------------------------------------------
% Helper: Normalize columns of X
% -------------------------------------------------------------------------
function Y = normalize_columns(X)
    norms = vecnorm(X, 2, 1);
    norms = max(norms, 1e-12);
    Y = X ./ norms;
end

% -------------------------------------------------------------------------
% Helper: Row-normalize matrix
% -------------------------------------------------------------------------
function Z = row_normalize(Z)
    norms = sqrt(sum(Z.^2, 2));
    norms = max(norms, 1e-12);
    Z = Z ./ norms;
end

% -------------------------------------------------------------------------
% Helper: Eigengap heuristic
% -------------------------------------------------------------------------
function K_est = estimate_k_eigengap(eigenvalues)
    vals = eigenvalues(:);

    if numel(vals) < 2
        error('Need at least two eigenvalues for eigengap estimation.');
    end

    gaps = vals(1:end-1) - vals(2:end);

    if numel(gaps) >= 2
        search_idx = 2:numel(gaps);
        [~, rel_idx] = max(gaps(search_idx));
        idx = search_idx(rel_idx);
    else
        [~, idx] = max(gaps);
    end

    K_est = idx;
end