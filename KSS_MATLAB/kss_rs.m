clear; clc;

load("MAT Files/Pavia.mat");
load("GT Files/Pavia_gt.mat");

data_cube = pavia;
gt_data = pavia_gt;

[H, W, B] = size(data_cube);

% Build foreground mask (exclude label 0 = background)

mask_fg = gt_data ~= 0;
mask_vec = mask_fg(:);

% Get unique non-zero labels and set K accordingly

gt_labels = unique(gt_data);
gt_labels(gt_labels == 0) = [];
K = numel(gt_labels);

% reshape -> (H*W) x B, then transpose to D x N where D = B
X_full = reshape(data_cube, [], B);
X_full = X_full.'; 

X = X_full(:, mask_vec);

% Define subspace dimensions and number of initializations

r=2;
nInit=10;

% Run KSS on the filtered out pixels
[U, c, totalcost] = kss(X, K, r, 'NInit', nInit, 'Verbose', true);

fprintf('Best total cost over %d inits: %.6e\n', nInit, totalcost);

% Map cluster labels back to image grid (H x W)
c_image_vec = zeros(H*W, 1);
c_image_vec(mask_vec) = c;            % fill only foreground pixels
c_image = reshape(c_image_vec, H, W);

% Custom colormap
colors = lines(K);
cmap = [0 0 0; colors];

% Visualize clustering result
figure;
imagesc(c_image);
axis image off;
title(sprintf('KSS clustering on Pavia (K = %d, r = %d)', K, r));
colormap(cmap);
colorbar;