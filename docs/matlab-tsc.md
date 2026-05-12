# MATLAB Thresholding-based Subspace Clustering (TSC)

The MATLAB implementation of the TSC algorithm is provided in the
`TSC_MATLAB` folder.

## Files

- `tsc.m` – core TSC algorithm (D × N data matrix, K clusters)
- `tsc_rs.m` – example script for running TSC on the Pavia dataset

## Example usage

```matlab
addpath("TSC_MATLAB");

% Load the Pavia dataset and its corresponding ground truth
load("MAT Files/Pavia.mat");
load("GT Files/Pavia_gt.mat");

% X: D x N data matrix constructed from the cube
[A, E, EigVals, c, K_used] = tsc(X, K);

% See tsc_rs.m for a complete example on Pavia
```

## MATLAB Live Script Demo

You can view the full interactive clustering demo here:


👉 [TSC Live Script (HTML)](TSC_MATLAB/TSC_Pavia_Live.html)