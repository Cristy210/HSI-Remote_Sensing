function [U, c, total_cost] = kss(X, K, r, varargin)

    p = inputParser;
    p.FunctionName = "kss";
    
    addParameter(p, 'U0', [], @(z) isnumeric(z) || isempty(z));
    addParameter(p, 'MaxIter', 100, @(z) isscalar(z) && z > 0);
    addParameter(p, 'Verbose', false, @(z) islogical(z) || z == 0 || z == 1);
    addParameter(p, 'NInit', 1, @(z) isscalar(z) && z >= 1);

    parse(p, varargin{:});
    U0 = p.Results.U0;
    maxIter = p.Results.MaxIter;
    verbose = logical(p.Results.Verbose);
    NInit   = p.Results.NInit;

    [D, N] = size(X);

    best_cost = -Inf;
    U_best    = [];
    c_best    = [];

    for init=1:NInit
        % Initialize Subspace basis

        if isempty(U0)
            U = rand_subspaces(D, r, K);
        else
            U = U0;
        end
        
        % Sanity Check
        
        if ~isequal(size(U), [D, r, K])
            error('U0 must be of size D x r x K (here D=%d, r=%d, K=%d).', D, r, K);
        end
        
        % Initial Labels (Will be overwritten after first iteration)
        
        c=ones(1, N);
        total_cost = NaN;
        
        % ----- Main KSS Loop -----
        
        for it=1:maxIter
            c_prev = c;
        
            % Scores --- K x N matrix
            scores = zeros(K, N);
        
            for k=1:K
                Uk = U(:, :, k);
                proj = Uk'*X;
                scores(k, :)=vecnorm(proj, 2, 1).^2;
            end
        
            % Cluster assignment --- Based on the projections (Maximum score)
            [~, c] = max(scores, [], 1);
        
            % Cost from the projections
            linear_idx  = sub2ind([K, N], c, 1:N);
            bestScores  = scores(linear_idx);
            total_cost  = sum(bestScores);
        
            % Update Step
            for k=1:K
                idx = find(c == k);
        
                if isempty(idx)
                    if verbose
                        fprintf('Iter %d: cluster %d empty, reinitializing.\n', it, k);
                    end
                    A = randn(D, r);
                    [Q, ~] = qr(A, 0);
                    U(:, :, k) = Q;
                else
                    Xk = X(:, idx);
                    U(:, :, k) = polar_subspace(Xk, r);
                end
            end
        
            % Convergence Check
            if all(c == c_prev)
                if verbose
                    fprintf('Converged at iteration %d.\n', it);
                end
                break;
            end
        end
        % Keep best initialization
        if total_cost > best_cost
            best_cost = total_cost;
            U_best = U;
            c_best = c;
        end
    end

    U = U_best;
    c= c_best;
    total_cost = best_cost;
end

% ---------- helper: polar_subspace ----------
function Uk = polar_subspace(Xk, r)
    A = Xk * Xk';
    [V, Dmat] = eig(A);
    eigvals = diag(Dmat);
    [~, idx] = sort(eigvals, 'descend');
    Uk = V(:, idx(1:r));
end

% ---------- helper: rand_subspaces ----------
function U0 = rand_subspaces(D, r, K)
    U0 = zeros(D, r, K);
    for k=1:K
        A = randn(D, r);
        [Q, ~] = qr(A, 0);
        U0(:, :, k) = Q;
    end
end
