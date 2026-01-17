"""
KSS Module. Clusters `N` data points into `K` clusters via K-Subspaces(KSS) Algorithm.
"""

module KSS

using ArnoldiMethod: partialschur
using LinearAlgebra: mul!, norm, svd!, Diagonal, Symmetric, I, normalize
using Random: AbstractRNG, default_rng, randn!
using Logging: @info, @warn
using Compat
using ProgressLogging: @withprogress, @logprogress

export KSSResult, fit, fit_predict, predict
@compat public randsubspace


# Utility functions
"""
    randsubspace([rng=default_rng()], [T=Float64], D, d)

Generate a random `d`-dimensional subspace of `ℝᴰ`
and return a basis matrix with element type `T<:AbstractFloat`.

See also [`randsubspace!`](@ref)
"""
randsubspace(rng::AbstractRNG, ::Type{T}, D::Integer, d::Integer) where {T<:AbstractFloat} =
    randsubspace!(rng, Array{T}(undef, D, d))
randsubspace(::Type{T}, D::Integer, d::Integer) where {T<:AbstractFloat} =
    randsubspace(default_rng(), T, D, d)
randsubspace(rng::AbstractRNG, D::Integer, d::Integer) = randsubspace(rng, Float64, D, d)
randsubspace(D::Integer, d::Integer) = randsubspace(default_rng(), Float64, D, d)

"""
    randsubspace!([rng=default_rng()], U::AbstractMatrix)

Set the `D×d` matrix `U` to be the basis matrix of
a randomly generated `d`-dimensional subspace of `ℝᴰ`.

See also [`randsubspace`](@ref)
"""
function randsubspace!(rng::AbstractRNG, U::AbstractMatrix)
    # Check arguments
    eltype(U) <: AbstractFloat ||
        throw(ArgumentError("Basis matrix `U` must have real (floating point) elements."))
    size(U, 2) <= size(U, 1) || throw(
        ArgumentError(
            "Subspace dimension `d` cannot be greater than the ambient dimension `D`.",
        ),
    )

    # Generate random subspace
    randn!(rng, U)
    P, _, Q = svd!(U)
    return mul!(U, P, Q')
end
randsubspace!(U::AbstractMatrix) = randsubspace!(default_rng(), U)

mutable struct KSSResult{
    TU<:AbstractVector{<:AbstractMatrix{<:AbstractFloat}},
    Tc<:AbstractVector{<:Integer},
    T<:Real,
}
    U::TU
    c::Tc
    iterations::Int
    totalcost::T
    counts::Vector{Int}
    converged::Bool
end

# Main function

"""
    kss(X::AbstractMatrix{<:Real}, d::AbstractVector{<:Integer};
        maxiters = 100,
        rng = default_rng(),
        Uinit = [randsubspace(rng, size(X, 1), di) for di in d])

Cluster the `N` data points in the `D×N` data matrix `X`
into `K` clusters via the **K**-**s**ub**s**paces (KSS) algorithm
with corresponding subspace dimensions `d[1],...,d[K]`.
Output is a [`KSSResult`](@ref) containing the resulting
cluster assignments `c[1],...,c[N]`,
subspace basis matrices `U[1],...,U[K]`,
and metadata about the algorithm run.

KSS seeks to cluster data points by their subspace
by minimizing the following total cost
```math
\\sum_{i=1}^N \\| X[:, i] - U[c[i]] U[c[i]]' X[:, i] \\|_2^2
```
with respect to the cluster assignments `c[1],...,c[N]`
and subspace basis matrices `U[1],...,U[K]`.

# Keyword arguments
- `maxiters::Integer = 100`: maximum number of iterations
- `rng::AbstractRNG = default_rng()`: random number generator
    (used when reinitializing the subspace for an empty cluster)
- `Uinit::AbstractVector{<:AbstractMatrix{<:AbstractFloat}}
    = [randsubspace(rng, size(X, 1), di) for di in d]`:
    vector of `K` initial subspace basis matrices to use
    (each `Uinit[k]` should be `D×d[k]`)

See also [`KSSResult`](@ref).
"""
function kss(
    X::AbstractMatrix{<:Real},
    d::AbstractVector{<:Integer};
    maxiters::Integer = 100,
    rng::AbstractRNG = default_rng(),
    Uinit::AbstractVector{<:AbstractMatrix{<:AbstractFloat}} = [
        randsubspace(rng, size(X, 1), di) for di in d
    ],
)
    # Require one-based indexing
    Base.require_one_based_indexing(X, d, Uinit)
    for Uk in Uinit
        Base.require_one_based_indexing(Uk)
    end

    # Extract sizes and check that they agree
    K = (only ∘ unique)([length(d), length(Uinit)])
    D = (only ∘ unique)([size(X, 1); size.(Uinit, 1)])

    # Check subspace dimensions
    for k in 1:K
        d[k] == size(Uinit[k], 2) || throw(
            ArgumentError(
                "Basis matrix initialization `Uinit[$k]` must have `d[$k]=$(d[k])` columns.",
            ),
        )
        0 <= d[k] <= D || throw(
            DimensionMismatch(
                "Subspace dimension `d[$k]=$(d[k])` must be between `0` and `D=$D`.",
            ),
        )
    end

    # Check maxiters
    maxiters >= 0 || throw(
        ArgumentError(
            "Maximum number of iterations must be nonnegative. Got `maxiters=$maxiters`.",
        ),
    )

    # Initialize model parameters
    U = deepcopy(Uinit)
    c = kss_assign_clusters(U, X)

    # Main loop
    cprev = copy(c)
    iterations, converged = 0, false
    @withprogress while iterations < maxiters && !converged
        iterations += 1

        # Update subspaces
        for k in 1:K
            inds = findall(==(k), c)
            if !isempty(inds)
                U[k] = kss_estimate_subspace(view(X, :, inds), d[k])
            else
                @warn "Empty cluster detected at iteration $iterations - reinitializing the subspace. Consider reducing the number of clusters."
                U[k] = randsubspace(rng, D, d[k])
            end
        end

        # Update cluster assignments
        kss_assign_clusters!(c, U, X)

        # Check for convergence
        if cprev == c
            @info "Converged after $iterations $(iterations == 1 ? "iteration" : "iterations")."
            converged = true
        end
        copyto!(cprev, c)

        # Log progress
        if iterations % (maxiters ÷ 100) == 0
            @logprogress iterations / maxiters
        end
    end

    # Compute final counts and costs
    counts = [count(==(k), c) for k in 1:K]
    costs = [sum(abs2, xi) - sum(abs2, U[c[i]]' * xi) for (i, xi) in pairs(eachcol(X))]

    return KSSResult(U, c, iterations, sum(costs), counts, converged)
end

# Subroutines

"""
    kss_assign_clusters(U, X)

Assign the `N` data points in `X` to the `K` subspaces in `U`
and return a vector of the assignments.

See also [`kss_assign_clusters!`](@ref), [`kss`](@ref).
"""
kss_assign_clusters(U, X) = kss_assign_clusters!(similar(Vector{Int}, (axes(X, 2),)), U, X)

"""
    kss_assign_clusters!(c, U, X)

Assign the `N` data points in `X` to the `K` subspaces in `U`,
update the vector of assignments `c`,
and return this vector of assignments.

See also [`kss_assign_clusters`](@ref), [`kss`](@ref).
"""
function kss_assign_clusters!(c, U, X)
    for (i, xi) in pairs(eachcol(X))
        c[i] = argmax(sum(abs2, U[k]' * xi) for k in eachindex(U))
    end
    return c
end

"""
    kss_estimate_subspace(Xk, dk)

Return `dk`-dimensional subspace that best fits the data points in `Xk`.

See also [`kss`](@ref).
"""
function kss_estimate_subspace(Xk, dk)
    decomp, history = partialschur(Xk * Xk'; nev = dk, which = :LR)
    history.converged ||
        @warn "Iterative algorithm for subspace update did not converge - results may be inaccurate."
    return decomp.Q
end


# =========================
# Public API
# =========================

"""
    Fit KSS and return labels

`fit_predict` runs `kss(X, d; kwargs...)` and returns only labels
"""

function fit_predict(X::AbstractMatrix{<:Real}, d::AbstractVector{<:Integer}; kwargs...)
    res = kss(X, d; kwargs...)
    return res.c
end

predict(res::KSSResult, X::AbstractMatrix{<:Real}) = predict(res.U, X)

"""
    Predict labels by assigning each column of `X` to the closest subspace in `U`
    Returns an `N` length vector of cluster labels in `1:K`. 
"""
function predict(U::AbstractVector{<:AbstractMatrix{<:AbstractFloat}}, X::AbstractMatrix{<:Real})
    
    isempty(U) && throw(ArgumentError(
        "Subspace basis list `U` is empty. Call `fit` first."
    ))
    
    Base.require_one_based_indexing(X, U)
    for Uk in U
        Base.require_one_based_indexing(Uk)
    end
    return kss_assign_clusters(U, X)
end

"""
    fit(X, d; kwargs...) -> KSSResult

Run **K-Subspaces (KSS)** clustering on the data matrix `X` and return a
[`KSSResult`](@ref) containing the learned subspaces and cluster assignments.

# Arguments
- `X::AbstractMatrix{<:Real}`: Data matrix of size `D × N`,
  where each column is a data point (e.g. a pixel spectrum).
- `d::AbstractVector{<:Integer}`: Vector of subspace dimensions,
  one per cluster.

# Keyword Arguments
All keyword arguments are forwarded to [`kss`](@ref), including:
- `maxiters`: maximum number of KSS iterations
- `rng`: random number generator
- `Uinit`: initial subspace bases

# Returns
- `res::KSSResult`: clustering result containing
  - `res.U`: learned subspace bases
  - `res.c`: cluster labels (`1:K`)
  - `res.totalcost`, `res.iterations`, etc.
"""

function fit(X::AbstractMatrix{<:Real}, d::AbstractVector{<:Integer}; kwargs...)
    return kss(X, d; kwargs...)
end

function fit!(res::KSSResult, X::AbstractMatrix{<:Real}, d::AbstractVector{<:Integer}; kwargs...)
    newres = kss(X, d; kwargs...)
    res.U = newres.U
    res.c = newres.c
    res.iterations = newres.iterations
    res.totalcost = newres.totalcost
    res.counts = newres.counts
    res.converged = newres.converged
    return res
end

end