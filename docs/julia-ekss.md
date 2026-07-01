# Julia Ensembke K-Subspaces (EKSS)

This page documents the Julia implementation of the **Ensemble K-Subspaces (EKSS)** algorithm.

Source code lives in `EKSS_Julia/ekss.jl`

- **Module:** `EKSS`
- **Exports:** `fit`, `fit_predict`, `EKSSResult`

## Quickstart

### Fit + get labels

```julia
# From the repository root:
include("EKSS_Julia/ekss.jl")
using .EKSS

# X is D×N (each column is a point)
# d is the subspace dimension
# K is the number of clusters
labels = fit(X, d, K; maxiters=100)
```

## Example usage

The following Pluto notebook shows an example of running the EKSS algorithm on the Pavia dataset.

👉 [EKSS Implementation - Julia (HTML)](EKSS_Julia/EKSS_Pavia.html)