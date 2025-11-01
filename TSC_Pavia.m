pavia_gt = load("GT Files/Pavia.mat");
hsi_cube = load("MAT Files/Pavia.mat");
gt_fieldnames = fieldnames(pavia_gt);
cube_fieldnames = fieldnames(hsi_cube);


% disp(pavia_gt.pavia_gt)
gt_data = pavia_gt.pavia_gt;
cube = hsi_cube.pavia;
[nrows, ncols, bands] = size(cube);

gt_labels = unique(pavia_gt.pavia_gt);
% disp(gt_labels)

% Fetch indices of the background pixels

[bgidx_row, bgidx_col] = find(pavia_gt.pavia_gt == 0);
% disp(bgidx_col)

%Define mask to remove background pixels, i.e., pixels labeled zero

mask = true(size(gt_data));

mask(gt_data == 0) = false;

data_mat = reshape(cube, [], bands);
labeled_cube = data_mat(mask(:), :);


% Display Ground Truth data

K = numel(gt_labels);
cmap = lines(K);
cmap = [0 0 0; cmap];

imagesc(double(gt_data));
colormap(cmap);
