"""
KAS Module. Clusters `N` data points into `K` clusters via K-Affine spaces(KAS) Algorithm.
"""

module KAS

using ArnoldiMethod: partialschur
using LinearAlgebra: mul!, norm, svd!, Diagonal, Symmetric, I, normalize
using Random: AbstractRNG, default_rng, randn!
using Logging: @info, @warn
using Compat
using ProgressLogging: @withprogress, @logprogress
using Statistics: mean

export KASResult, fit, fit_predict, predict, fit!
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

"""
    @withprogressif cond [name=""] [parentid=uuid4()] ex

Conditional version of `@withprogress` that only sets up a progress bar if `cond` is `true`,
in which case it passes the remaining arguments to `@withprogress`. Otherwise, it executes
`ex` directly.

See also [`@logprogressif`](@ref).
"""
macro withprogressif(cond, exprs...)
    ex_withlogging = :(@withprogress $(exprs...))  # version with logging
    ex_withoutlogging = exprs[end]                 # version without logging
    return quote
        if $(esc(cond))
            $(esc(ex_withlogging))
        else
            $(esc(ex_withoutlogging))
        end
    end
end

"""
    @logprogressif cond [name] progress [key1=val1 [key2=val2 ...]]

Conditional version of `@logprogress` that only logs progress if `cond` is `true`,
in which case it passes the remaining arguments to `@logprogress`. Otherwise, it
does nothing.

See also [`@withprogressif`](@ref).
"""
macro logprogressif(cond, exprs...)
    logexpr = :(@logprogress $(exprs...))
    return quote
        if $(esc(cond))
            $(esc(logexpr))
        end
    end
end

mutable struct KASResult{
    TUb<:Union{AbstractFloat,Complex{<:AbstractFloat}},
    TU<:AbstractVector{<:AbstractMatrix{TUb}},
    Tb<:AbstractVector{<:AbstractVector{TUb}},
    Tc<:AbstractVector{<:Integer},
    T<:Real,
}
    U::TU
    b::Tb
    c::Tc
    iterations::Int
    totalcost::T
    counts::Vector{Int}
    converged::Bool
end

# Main function

"""
    kas(X::AbstractMatrix{<:Number}, d::AbstractVector{<:Integer};
        maxiters = 100,
        rng = default_rng(),
        init = [(randsubspace(rng, float(eltype(X)), size(X, 1), di), zeros(float(eltype(X)), size(X, 1))) for di in d],
        showprogress = false)

Cluster the `N` data points in the `D×N` data matrix `X`
into `K` clusters via the **K**-**a**ffine-**s**paces (KAS) algorithm
with corresponding affine space dimensions `d[1],...,d[K]`.
Output is a [`KASResult`](@ref) containing the resulting
cluster assignments `c[1],...,c[N]`,
affine space basis matrices `U[1],...,U[K]`,
bias vectors `b[1],...,b[K]`,
and metadata about the algorithm run.

KAS seeks to cluster data points by their affine space
by minimizing the following total cost
```math
\\sum_{i=1}^N \\| X[:, i] - (U[c[i]] U[c[i]]' (X[:, i] - b[c[i]]) + b[c[i]]) \\|_2^2
```
with respect to the cluster assignments `c[1],...,c[N]`,
affine space basis matrices `U[1],...,U[K]`,
and bias vectors `b[1],...,b[K]`.

# Keyword arguments
- `maxiters::Integer = 100`: maximum number of iterations
- `rng::AbstractRNG = default_rng()`: random number generator
    (used when reinitializing the affine space for an empty cluster)
- `init::AbstractVector{<:Tuple{<:AbstractMatrix{TUb},<:AbstractVector{TUb}}}
    = [(randsubspace(rng, float(eltype(X)), size(X, 1), di), zeros(float(eltype(X)), size(X, 1))) for di in d]`:
    vector of `K` initial pair of affine space basis matrices containing `U[1],...,U[K]`
    and bias vectors containing `b[1],...,b[K]`
    where `TUb` is a floating point type.
- `showprogress::Bool = false`: whether to log progress during the algorithm run

See also [`KASResult`](@ref).
"""
function kas(
    X::AbstractMatrix{<:Number},
    d::AbstractVector{<:Integer};
    maxiters::Integer = 100,
    rng::AbstractRNG = default_rng(),
    init::AbstractVector{<:Tuple{<:AbstractMatrix{TUb},<:AbstractVector{TUb}}} = [
        (
            randsubspace(rng, float(eltype(X)), size(X, 1), di),
            zeros(float(eltype(X)), size(X, 1)),
        ) for di in d
    ],
    showprogress::Bool = true,
) where {TUb<:Union{AbstractFloat,Complex{<:AbstractFloat}}}
    # Unpack the initial affine space basis matrices and bias vectors
    Uinit = first.(init)
    binit = last.(init)

    # Require one-based indexing
    Base.require_one_based_indexing(X, d, Uinit, binit)
    for Uk in Uinit
        Base.require_one_based_indexing(Uk)
    end
    for bk in binit
        Base.require_one_based_indexing(bk)
    end

    # Extract sizes and check that they agree
    K = (only ∘ unique)([length(d), length(Uinit), length(binit)])
    D = (only ∘ unique)([size(X, 1); size.(Uinit, 1); length.(binit)])

    # Check affine space dimensions
    for k in 1:K
        d[k] == size(Uinit[k], 2) || throw(
            ArgumentError(
                "Basis matrix initialization `Uinit[$k]` must have `d[$k]=$(d[k])` columns.",
            ),
        )
        0 <= d[k] <= D || throw(
            DimensionMismatch(
                "Affine space dimension `d[$k]=$(d[k])` must be between `0` and `D=$D`.",
            ),
        )
        length(binit[k]) == D || throw(
            ArgumentError(
                "Bias vector initialization `binit[$k]` must be of length `D=$D`.",
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
    b = deepcopy(binit)
    c = kas_assign_clusters(U, b, X)

    # Main loop
    changed = trues(K)
    iterations, converged = 0, false
    log_every = max(1, maxiters ÷ 100)
    @withprogressif showprogress while iterations < maxiters && !converged
        iterations += 1

        # Update affine space basis matrices and bias vectors
        for k in 1:K
            changed[k] || continue

            inds = findall(==(k), c)
            if !isempty(inds)
                U[k], b[k] = kas_estimate_affinespace(view(X, :, inds), d[k])
            else
                @warn "Empty cluster detected at iteration $iterations - reinitializing the affine space. Consider reducing the number of clusters."
                randsubspace!(rng, U[k])
                fill!(b[k], zero(eltype(b[k])))
            end
        end

        # Update cluster assignments
        kas_assign_clusters!(c, U, b, X, changed)

        # Check for convergence
        if !any(changed)
            @info "Converged after $iterations $(iterations == 1 ? "iteration" : "iterations")."
            converged = true
        end

        # Log progress
        if iterations % log_every == 0
            @logprogressif showprogress iterations / maxiters
        end
    end

    # Compute final counts and costs
    counts = [count(==(k), c) for k in 1:K]
    costs = [
        sum(abs2, (xi - b[c[i]])) - sum(abs2, U[c[i]]' * (xi - b[c[i]])) for
        (i, xi) in pairs(eachcol(X))
    ]

    return KASResult(U, b, c, iterations, sum(costs), counts, converged)
end

# Subroutines

"""
    kas_assign_clusters(U, b, X)

Assign the `N` data points in `X` to the `K` affine spaces in `(U,b)`
and return a vector of the assignments.

See also [`kas_assign_clusters!`](@ref), [`kas`](@ref).
"""
kas_assign_clusters(U, b, X) =
    kas_assign_clusters!(similar(Vector{Int}, (axes(X, 2),)), U, b, X)

"""
    kas_assign_clusters!(c, U, b, X)

Assign the `N` data points in `X` to the `K` affine spaces in `(U,b)`,
update the vector of assignments `c`,
and return this vector of assignments.

See also [`kas_assign_clusters`](@ref), [`kas`](@ref).
"""
function kas_assign_clusters!(c, U, b, X)
    for (i, xi) in pairs(eachcol(X))
        c[i] = argmin(
            sum(abs2, (xi - b[k])) - sum(abs2, U[k]' * (xi - b[k])) for k in eachindex(U)
        )
    end
    return c
end

"""
    kas_assign_clusters!(c, U, b, X, changed)

Assign each data point in `X` to an affine space in `(U, b)`, and update `c`.
Set `changed[k]` to true when cluster `k` gains or loses a member.

Return the updated vector assignment `c`.

See also [`kas_assign_clusters`](@ref), [`kas`](@ref).
"""
function kas_assign_clusters!(c, U, b, X, changed)
    fill!(changed, false)

    for (i, xi) in pairs(eachcol(X))
        old_assignment = c[i]
        new_assignment = argmin(
            sum(abs2, (xi - b[k])) - sum(abs2, U[k]' * (xi - b[k])) for k in eachindex(U)
        )

        if old_assignment != new_assignment
            changed[old_assignment] = true
            changed[new_assignment] = true
            c[i] = new_assignment
        end
    end

    return c
end

"""
    kas_estimate_affinespace(Xk, dk)

Return `dk`-dimensional affine space that best fits the data points in `Xk`.

See also [`kas`](@ref).
"""
function kas_estimate_affinespace(Xk, dk)
    bhat = mean(eachcol(Xk))
    Uhat = svd!(Xk .- bhat; full = true).U[:, 1:dk]
    return Uhat, bhat
end

# =========================
# Public API
# =========================
"""
    fit(X, d; kwargs...) -> KASResult

Run K-Affine Spaces (KAS) clustering on the `D × N` data matrix `X`,
where each column is a data point.

`d` contains the affine space dimension for each cluster.

All keyword arguments are forwarded to [`kas`](@ref), including
`maxiters`, `rng`, `init`, and `showprogress`.

The returned [`KASResult`](@ref) contains:
- `U`: learned affine space bases
- `b`: learned bias vectors
- `c`: cluster assignments
- `iterations`, `totalcost`, `counts`, and `converged`: run metadata
"""
function fit(
    X::AbstractMatrix{<:Number},
    d::AbstractVector{<:Integer};
    kwargs...,
)
    return kas(X, d; kwargs...)
end

"""
    fit_predict(X, d; kwargs...) -> Vector{Int}

Run KAS clustering and return only the cluster assignments.

All keyword arguments are forwarded to [`kas`](@ref).
"""
function fit_predict(
    X::AbstractMatrix{<:Number},
    d::AbstractVector{<:Integer};
    kwargs...,
)
    return fit(X, d; kwargs...).c
end

"""
    predict(res::KASResult, X)
    predict(U, b, X)

Assign each column of the `D × N` data matrix `X` to its closest
affine space using the basis matrices `U` and bias vectors `b`.

Return a vector of cluster labels in `1:K`.
The learned parameters are not modified.
"""
predict(res::KASResult, X::AbstractMatrix{<:Number}) =
    predict(res.U, res.b, X)

function predict(
    U::AbstractVector{<:AbstractMatrix},
    b::AbstractVector{<:AbstractVector},
    X::AbstractMatrix{<:Number},
)
    isempty(U) && throw(
        ArgumentError("Affine space basis list `U` is empty. Call `fit` first."),
    )

    length(U) == length(b) || throw(
        DimensionMismatch(
            "Expected one bias vector per basis matrix; got $(length(U)) bases and $(length(b)) biases.",
        ),
    )

    Base.require_one_based_indexing(X, U, b)

    D = size(X, 1)
    for k in eachindex(U, b)
        Base.require_one_based_indexing(U[k], b[k])

        size(U[k], 1) == D || throw(
            DimensionMismatch("Basis matrix `U[$k]` must have `D=$D` rows."),
        )

        length(b[k]) == D || throw(
            DimensionMismatch("Bias vector `b[$k]` must have length `D=$D`."),
        )
    end

    return kas_assign_clusters(U, b, X)
end

"""
    fit!(res::KASResult, X, d; kwargs...) -> KASResult

Run KAS clustering and replace the fields of `res` with the new result.
Return the same result object.

All keyword arguments are forwarded to [`kas`](@ref).
The existing parameters are used as initialization only when explicitly
provided through `init`.

The new result fields must be compatible with the concrete field types
of `res`.
"""
function fit!(
    res::KASResult,
    X::AbstractMatrix{<:Number},
    d::AbstractVector{<:Integer};
    kwargs...,
)
    newres = fit(X, d; kwargs...)

    res.U = newres.U
    res.b = newres.b
    res.c = newres.c
    res.iterations = newres.iterations
    res.totalcost = newres.totalcost
    res.counts = newres.counts
    res.converged = newres.converged

    return res
end

end