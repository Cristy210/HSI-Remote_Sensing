# Julia K-Subspaces (KSS)

This page documents the Julia implementation of the **K-Subspaces (KSS)** algorithm.

Source code lives in `KSS_Julia/kss.jl`

- **Module:** `KSS`
- **Exports:** `fit`, `fit_predict`, `predict`, `KSSResult`

## Quickstart

### Fit + get labels


```julia
# From the repository root:
include("KSS_Julia/kss.jl")
using .KSS

# X is D×N (each column is a point)
# d is a length-K vector of subspace dimensions
labels = fit_predict(X, d; maxiters=100)
```

## Example usage

The following Pluto notebook shows an example of running the KSS algorithm on the Pavia dataset.

👉 [KSS Implementation - Julia (HTML)](KSS_Julia/KSS_Pavia_Julia.html)