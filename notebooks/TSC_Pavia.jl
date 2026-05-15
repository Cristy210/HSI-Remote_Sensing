### A Pluto.jl notebook ###
# v0.20.16

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ eb33cfc1-58c4-4f34-a70c-a07928e02039
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ 58baac76-e492-4c7d-8d84-a5a571c81c7d
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 0e14a739-a926-45b4-8df1-96090d257bfd
begin
    include(joinpath(@__DIR__, "..", "TSC_Julia", "tsc.jl"))
    using .TSC
end

# ╔═╡ 0bdfe2ea-7c33-11f0-3423-e9a930a7eff5
html"""<style>
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 96%;
    margin-left: 0%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 4ab3db82-595e-4cba-99e3-651b069ba7d9
md"""
### This Notebook demonstrates the implementation of Thresholded Subspace Clustering (TSC) Clustering Algorithm on Pavia Dataset
"""

# ╔═╡ 9fa11256-e680-4735-993c-84fa584bc65f


# ╔═╡ bd10a546-ef6f-44c5-a840-97fc07d1da12
md"""
#### Project Setup

> We start by activating the project environment and importing the required Julia packages.
"""

# ╔═╡ f265ae93-bd07-4fcc-8adb-b6aa6d135f07
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ a23d06a4-e0de-458e-afe5-d7989342dc37
filepath = abspath(joinpath(@__DIR__, "..", "MAT Files", "$Location.mat"))

# ╔═╡ 92620cae-f1bd-4864-a795-d2a8c8dbf16e
gt_filepath = abspath(joinpath(@__DIR__, "..", "GT Files", "$Location.mat"))

# ╔═╡ c8376c69-f1e6-459f-aedd-766e2d84cb3a
CACHEDIR = abspath(joinpath(@__DIR__, "..", "cache_files", "Aerial Datasets"))

# ╔═╡ ba3ecda0-aa6b-46c3-80f4-eb8ab9a8a4bd
md"""
#### Defined cachet function to cache runs
"""

# ╔═╡ 683658a8-7fb4-4cf2-9af5-dea630aaf5d5
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ d6236062-97f9-45a5-96fc-54865dc7adfa
vars = matread(filepath)

# ╔═╡ 63d1a002-ee29-4e6c-99b5-33f0b035501b
vars_gt = matread(gt_filepath)

# ╔═╡ d51cba8e-2131-4480-b585-471ca84c529d
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 5c796783-68fc-4a7e-b4e2-e483a9296f3a
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 670896f7-f26e-4d8f-8973-b0f857c39f3c
md"""
### Hyperspectral Image Cube - $Location
"""

# ╔═╡ 7e32f893-16f4-468c-bf6c-fd4f7cfa34f1
data = vars[data_key]

# ╔═╡ 8858eaba-a317-4635-9f05-f28755672cea
md"""
### Ground Truth Data
"""

# ╔═╡ b90b3cff-d287-4738-a108-20b676359b22
gt_data = vars_gt[gt_key]

# ╔═╡ b6cd76d8-1b07-48bd-af93-a3cd991756ae
md"""
### Ground Truth Labels
"""

# ╔═╡ 1f97e07d-c586-4c60-91d6-ba44f39f7886
gt_labels = sort(unique(gt_data))

# ╔═╡ 3f891d4f-1631-4455-9cb5-7bef91d5fb68
bg_indices = findall(gt_data .== 0)

# ╔═╡ 77ec0819-7c35-46ca-ab4e-2f66d719a8a5
md"""
### Define mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ eef28252-7264-4970-90b2-60737c610e29
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ fa18556c-63cb-4f1c-a261-d3b98964c5d0
md"""
##### Number of clusters equivalent to the number of unique labels from the ground truth data
"""

# ╔═╡ 87e5df0a-862f-4e95-afea-427b85771b47
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 84a49020-fd8b-4ab6-b096-bf87b09712f3
md"""
#### Slider to choose the band of the image
"""

# ╔═╡ 5db6119b-537c-4d05-81e5-49b120f5f34d
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 420cb46e-66dc-4b09-bb3d-ca02b5c5322b
with_theme() do
	fig = Figure(; size=(750, 600))
	labels = length(unique(gt_data))
	colors = Makie.Colors.distinguishable_colors(n_clusters+1)
	ax = Axis(fig[1, 1], aspect=DataAspect(), title ="Image, Band - $band", yreversed=true)
	ax1 = Axis(fig[1, 2], aspect=DataAspect(), title ="Masked Image", yreversed=true)
	image!(ax, permutedims(data[:, :, band]))
	hm = heatmap!(ax1, permutedims(gt_data); colormap=Makie.Categorical(colors))
	fig
end

# ╔═╡ ced71964-16de-44bd-b34a-98054a0ea185


# ╔═╡ 6de6a857-cf65-4c59-841d-6d4c9353e617


# ╔═╡ 2763c03a-8470-41ce-a991-5aa8762ca971
md"""
> ###### Thresholded Subspace Clustering (TSC) Algorithm treats data points as nodes in the graph which are then clustered using techniques from spectral graph theory [Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.] 
>
> - Three important matrices that make up the TSC algorithm are the Adjacency Matrix, Degree Matrix, and Laplacian Matrix 
>
> - The adjacency matrix ``\mathbf{A}`` defines the similarity between any two nodes in the data set. To compute the adjacency matrix, we first compute a matrix of (transformed) pairwise cosine similarities ``\mathbf{C}`` where each vertex ``\mathbf{V_i} \in \mathbb{R}^L`` represents a data point.
> -  A thresholded version ``\mathbf{Z}`` is then created from ``\mathbf{C}`` by keeping only the ``q`` largest values in each column and zeroing out the rest. This thresholded matrix is then symmetrized to obtain the adjacency matrix ``\mathbf{A}``.
> - The degree matrix ``\mathbf{D}`` represents the sum of the weights of all edges connected to a node.
> - Finally, the Laplacian matrix ``\mathbf{L}`` captures the structure of a graph by combining information from the adjacency and degree matrices

`` \mathbf{C}_{ij} = \exp \left[ -2 \cdot \arccos \left( \frac{|\mathbf{V}_i^\top \mathbf{V}_j|}{\|\mathbf{V}_i\|_2 \cdot \|\mathbf{V}_j\|_2} \right) \right], \quad \text{for $i, j = 1,\dots,MN$} ``

``\mathbf{A} = \mathbf{Z} + \mathbf{Z}^T``

``\mathbf{D} = \operatorname{diag}(\mathbf{d})\quad \text{where } \mathbf{d}_i = \sum_{j=1}^{MN}A_{ij},\quad \text{for $i = 1,\dots,MN$}.``

``\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}``
"""

# ╔═╡ d4b67203-333b-45e0-91b2-4fa8d1a8ba78
model = fit(data[mask, :]', n_clusters; max_nz=15)

# ╔═╡ d61311e5-b471-44a8-9969-9ffbb713356e
labels = model.assignments

# ╔═╡ 6b6a0558-ada2-4193-a5f3-a4878421f914
md"""
### Confusion Matrix -- Clustering Results
"""

# ╔═╡ b66ad2d1-3159-4cd0-9609-9f7c151e8b87
md"""
#### Relabel the clusters to compare it with the ground truth
"""

# ╔═╡ c100daa2-5ecb-41ac-80de-ed580548bc15
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 1,
	2 => 6,
	3 => 7,
	4 => 8,
	5 => 2,
	6 => 9,
	7 => 5,
	8 => 4,
	9 => 3
),
	"PaviaUni" => Dict(
	0 => 0,
	1 => 5,
	2 => 8,
	3 => 3,
	4 => 9,
	5 => 1,
	6 => 6,
	7 => 4,
	8 => 2,
	9 => 7,
)
)

# ╔═╡ c0af6dcc-a318-4a85-a828-be61fc28cc85
relabel_keys = relabel_maps[Location]

# ╔═╡ 5d9b5b61-93cf-4824-a8d0-8ec136f0902b
D_relabel = [relabel_keys[label] for label in labels]

# ╔═╡ 57fe2828-a673-400d-961a-a9d36948ba54
md"""
### Ground Truth Vs. Clustering Result
"""

# ╔═╡ c6c5735d-240c-4ebf-a940-531e4f3ace81
with_theme() do

	# Create figure
	fig = Figure(; size=(700, 650))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# subgrid = fig[1, 1] = GridLayout()

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
	
	hm1 = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,1], hm1, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="TSC Clustering Results - $Location", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm2 = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,2], hm2, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ 60465472-3dae-482b-9630-84dbaa425966
begin
	ground_labels_re = filter(x -> x != 0, gt_labels) #Filter out the background pixel label
	true_labels_re = length(ground_labels_re)
	predicted_labels_re = n_clusters

	confusion_matrix_re = zeros(Float64, true_labels_re, predicted_labels_re) #Initialize a confusion matrix filled with zeros
	cluster_results_re = fill(NaN32, size(data)[1:2]) #Clustering algorithm results

	# clu_assign, idx = spec_aligned, spec_clustering_idx
	

	cluster_results_re[mask] .= D_relabel

	for (label_idx, label) in enumerate(ground_labels_re)
	
		label_indices = findall(gt_data .== label)
	
		cluster_values = [cluster_results_re[idx] for idx in label_indices]
		t_pixels = length(cluster_values)
		cluster_counts = [count(==(cluster), cluster_values) for cluster in 1:n_clusters]
		confusion_matrix_re[label_idx, :] .= [count / t_pixels * 100 for count in cluster_counts]
	end
end

# ╔═╡ b2d17b14-ab7f-4b4f-9403-42ec4d9e5506
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="Confusion Matrix - TSC Clustering - $Location")
	hm = heatmap!(ax, permutedims(confusion_matrix_re), colormap=:viridis)
	pm = permutedims(confusion_matrix_re)

	for i in 1:true_labels_re, j in 1:predicted_labels_re
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm)
	fig
end

# ╔═╡ Cell order:
# ╟─0bdfe2ea-7c33-11f0-3423-e9a930a7eff5
# ╟─4ab3db82-595e-4cba-99e3-651b069ba7d9
# ╟─9fa11256-e680-4735-993c-84fa584bc65f
# ╟─bd10a546-ef6f-44c5-a840-97fc07d1da12
# ╠═eb33cfc1-58c4-4f34-a70c-a07928e02039
# ╠═58baac76-e492-4c7d-8d84-a5a571c81c7d
# ╠═f265ae93-bd07-4fcc-8adb-b6aa6d135f07
# ╠═a23d06a4-e0de-458e-afe5-d7989342dc37
# ╠═92620cae-f1bd-4864-a795-d2a8c8dbf16e
# ╠═c8376c69-f1e6-459f-aedd-766e2d84cb3a
# ╟─ba3ecda0-aa6b-46c3-80f4-eb8ab9a8a4bd
# ╠═683658a8-7fb4-4cf2-9af5-dea630aaf5d5
# ╠═d6236062-97f9-45a5-96fc-54865dc7adfa
# ╠═63d1a002-ee29-4e6c-99b5-33f0b035501b
# ╠═d51cba8e-2131-4480-b585-471ca84c529d
# ╠═5c796783-68fc-4a7e-b4e2-e483a9296f3a
# ╟─670896f7-f26e-4d8f-8973-b0f857c39f3c
# ╠═7e32f893-16f4-468c-bf6c-fd4f7cfa34f1
# ╟─8858eaba-a317-4635-9f05-f28755672cea
# ╠═b90b3cff-d287-4738-a108-20b676359b22
# ╟─b6cd76d8-1b07-48bd-af93-a3cd991756ae
# ╠═1f97e07d-c586-4c60-91d6-ba44f39f7886
# ╠═3f891d4f-1631-4455-9cb5-7bef91d5fb68
# ╟─77ec0819-7c35-46ca-ab4e-2f66d719a8a5
# ╠═eef28252-7264-4970-90b2-60737c610e29
# ╟─fa18556c-63cb-4f1c-a261-d3b98964c5d0
# ╠═87e5df0a-862f-4e95-afea-427b85771b47
# ╟─84a49020-fd8b-4ab6-b096-bf87b09712f3
# ╠═5db6119b-537c-4d05-81e5-49b120f5f34d
# ╠═420cb46e-66dc-4b09-bb3d-ca02b5c5322b
# ╟─ced71964-16de-44bd-b34a-98054a0ea185
# ╟─6de6a857-cf65-4c59-841d-6d4c9353e617
# ╟─2763c03a-8470-41ce-a991-5aa8762ca971
# ╠═0e14a739-a926-45b4-8df1-96090d257bfd
# ╠═d4b67203-333b-45e0-91b2-4fa8d1a8ba78
# ╠═d61311e5-b471-44a8-9969-9ffbb713356e
# ╟─6b6a0558-ada2-4193-a5f3-a4878421f914
# ╟─b66ad2d1-3159-4cd0-9609-9f7c151e8b87
# ╠═c100daa2-5ecb-41ac-80de-ed580548bc15
# ╠═c0af6dcc-a318-4a85-a828-be61fc28cc85
# ╠═5d9b5b61-93cf-4824-a8d0-8ec136f0902b
# ╟─57fe2828-a673-400d-961a-a9d36948ba54
# ╠═c6c5735d-240c-4ebf-a940-531e4f3ace81
# ╠═60465472-3dae-482b-9630-84dbaa425966
# ╠═b2d17b14-ab7f-4b4f-9403-42ec4d9e5506
