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

% Run TSC on the filtered out pixels
[A, E, EigVals, c, K_used] = tsc(X, K, 'Verbose', true);

% Map cluster labels back to image grid (H x W)
c_image_vec = zeros(H*W, 1);
c_image_vec(mask_vec) = c;            % fill only foreground pixels
c_image = reshape(c_image_vec, H, W);


gt_image = gt_data;
%% 


% Visualize clustering result
figure('Color', 'w', 'Position', [100 100 900 800]);

t = tiledlayout(2,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');

% -------------------------
% Shared colormap
% -------------------------
colors = lines(K);
cmap = [0 0 0; colors];
colormap(cmap);

% -------------------------
% LEFT: Ground Truth
% -------------------------
ax1 = nexttile(1);
imagesc(gt_data);
axis image off;
title('Ground Truth', 'FontWeight','bold');

clim([-0.5 K+0.5]);

% -------------------------
% RIGHT: Clustering
% -------------------------
ax2 = nexttile(2);
imagesc(c_image);
axis image off;
title(sprintf('TSC Clustering (K = %d)', K_used), 'FontWeight','bold');

clim([-0.5 K+0.5]);

% -----------
% Colorbars
% -----------
cb1 = colorbar(ax1, 'southoutside');
cb1.Ticks = 0:K;

cb2 = colorbar(ax2, 'southoutside');
cb2.Ticks = 0:K;