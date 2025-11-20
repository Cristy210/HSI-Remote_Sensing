# MATLAB K-Subspaces (KSS)

The MATLAB implementation of the K-Subspaces algorithm is provided in the
`KSS_MATLAB` folder.

## Files

- `kss.m` – core KSS algorithm (D × N data matrix, K clusters, r-dimensional subspaces)
- `kss_rs.m` – example script for running KSS on the Pavia dataset

## Example usage

```matlab
addpath("KSS_MATLAB");

load("MAT Files/Pavia.mat");
load("GT Files/Pavia_gt.mat");

% X: D x N data matrix constructed from the cube
[U, c, total_cost] = kss(X, K, r);

% See kss_rs.m for a complete example on Pavia
```

## MATLAB Live Script Demo

You can view the full interactive clustering demo here:


👉 [KSS Live Script (HTML)](KSS_MATLAB/KSS_HSI_Live.html)

<!-- <iframe 
    src="../KSS_MATLAB/KSS_HSI_Live.html"
    style="width: 100%; height: 1200px; border: none; overflow: auto;"
></iframe> -->