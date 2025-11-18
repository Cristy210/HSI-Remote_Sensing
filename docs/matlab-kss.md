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