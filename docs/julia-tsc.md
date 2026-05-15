# Julia Thresholded Subspace Clustering (TSC)

This page documents the Julia implementation of the **Thresholded Subspace Clustering (TSC)** algorithm.

Source code lives in `TSC_Julia/tsc.jl`

- **Module:** `TSC`
- **Exports:** `fit`, `fit_predict`, `predict`, `KSSResult`

## Quickstart

### Fit + get labels

```julia
# From the repository root:
include("TSC_Julia/tsc.jl")
using .TSC

# X is D×N (each column is a point)
# K is the number of clusters
labels = fit_predict(X, K)
```

## Example usage

The following Pluto notebook shows an example of running the TSC algorithm on the Pavia dataset.

👉 [TSC Implementation - Julia (HTML)](TSC_Julia/TSC_Pavia_Julia.html)