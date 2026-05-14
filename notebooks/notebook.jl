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

# ╔═╡ 295bbe52-d1f7-4880-84df-639bf6be23b6
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ f34d4731-9e7e-4624-bdc6-12f0b9734f11
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ a02a82ec-84f5-4429-aa36-3f1017f74332
begin
    include(joinpath(@__DIR__, "..", "TSC_Julia", "tsc.jl"))
    using .TSC
end

# ╔═╡ baadc009-9be1-43bd-bc10-89610755f347
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

# ╔═╡ e6c94de6-7b69-4003-b956-b603d218a4ea
md"""
### This Notebook demonstrates the implementation of Thresholded Subspace Clustering (TSC) Clustering Algorithm on Pavia Dataset
"""

# ╔═╡ b59f17d2-8d4e-443e-83df-e70c8e45958b
md"""
#### Project Setup

> We start by activating the project environment and importing the required Julia packages.
"""

# ╔═╡ d9722d73-f0c1-4e8b-a099-b54c6ca89dec
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ 6e2611f1-e56a-455a-be2f-66bfdfb2f778
filepath = abspath(joinpath(@__DIR__, "..", "MAT Files", "$Location.mat"))

# ╔═╡ 24ec9d2f-5531-4bd8-b31e-c2245e002c74
gt_filepath = abspath(joinpath(@__DIR__, "..", "GT Files", "$Location.mat"))

# ╔═╡ 7bbd8a1c-c57b-4305-9198-40feba9b542c
CACHEDIR = abspath(joinpath(@__DIR__, "..", "cache_files", "Aerial Datasets"))

# ╔═╡ c1b19a5f-f387-4d11-a8cf-2c0b6c83904c
md"""
#### Defined cachet function to cache runs
"""

# ╔═╡ 96572a79-3e0c-47c3-8016-bd141745ee81
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 1f8f3f29-b666-4986-8ff7-b8021a24cacb
vars = matread(filepath)

# ╔═╡ 7d031d4c-62b0-41c4-9a98-969850c65648
vars_gt = matread(gt_filepath)

# ╔═╡ b126efef-20d2-4c39-b414-b5b030cbe457
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 0220a0ef-755c-41e4-a555-18c5a58099f6
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 023b3b11-7f99-4abe-93a3-e026b8a6168f
md"""
### Hyperspectral Image Cube - $Location
"""

# ╔═╡ f291fa32-a21d-466b-af5b-6cdebca7f1f8
data = vars[data_key]

# ╔═╡ 1bc7120d-5484-4b41-aab3-1c9a9a43629e
md"""
### Ground Truth Data
"""

# ╔═╡ 781721b0-a273-45fd-89e6-c6a0d9f58bc0
gt_data = vars_gt[gt_key]

# ╔═╡ 0a319270-951b-47e1-a4ea-24d4d8258864
md"""
### Ground Truth Labels
"""

# ╔═╡ 31801973-d6c1-4a33-8181-27b598676b86
gt_labels = sort(unique(gt_data))

# ╔═╡ 3a998668-be40-415d-bf67-503e01cae001
bg_indices = findall(gt_data .== 0)

# ╔═╡ eab3ca40-ea6f-4e42-9f2d-97116d67764f
md"""
### Define mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 3f366f2c-4b9e-4512-bdc1-1e9be8ebc13f
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ ed664957-8c6d-4899-b8d5-0ebd859fc9b3
md"""
##### Number of clusters equivalent to the number of unique labels from the ground truth data
"""

# ╔═╡ 8bd009fb-4ea1-455e-9f7e-f8d81e844a73
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 0f63ae9e-0a50-4739-bd3a-85a4ee35c809
md"""
#### Slider to choose the band of the image
"""

# ╔═╡ d34de99a-766f-4137-8b48-f71fb5293117
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 094787c3-585e-4d96-9f96-bc0c80665b31
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

# ╔═╡ 480f6b51-ca0a-4f5e-ac86-1e2ee40ab06c
md"""
> ###### Thresholded Subspace Clustering (TSC) Algorithm treats data points as nodes in the graph which are then clustered using techniques from spectral graph theory [Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416; Heckel, Reinhard, and Helmut Bölcskei. "Robust subspace clustering via thresholding" IEEE Transactions on Information Theory, 61(11), 6320–6342, 2015.] 
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

# ╔═╡ d0d303b6-1448-4e90-9a00-4cdde4b0f9f3
size(data[mask, :]', 2)

# ╔═╡ 75260ec6-391b-4ce3-a084-034c1651a161
isqrt(148152)

# ╔═╡ 3cb67359-8491-4fb8-b85d-2f8070baf9c3
model = fit(data[mask, :]', n_clusters; max_nz=10, kmeans_nruns=100)

# ╔═╡ 9778bc47-7b27-4d37-bf47-82737c759c8b
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ c48d26fe-82e4-4869-a48f-8c6dd66ec56a
clusters = model.assignments

# ╔═╡ 2f9623ce-e12d-44c0-a092-fb837864b92b
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 1,
	2 => 8,
	3 => 6,
	4 => 9,
	5 => 7,
	6 => 2,
	7 => 3,
	8 => 5,
	9 => 4,
),
	"PaviaUni" => Dict(
	0 => 0,
	1 => 3,
	2 => 4,
	3 => 1,
	4 => 9,
	5 => 7,
	6 => 6,
	7 => 5,
	8 => 8,
	9 => 2,
)
)

# ╔═╡ d00a0185-afcf-4753-85ec-7c331c687507
relabel_keys = relabel_maps[Location]

# ╔═╡ 9cbc9294-87be-4d9c-a194-c19df153f79f
D_relabel = [relabel_keys[label] for label in clusters]

# ╔═╡ af686d74-e702-416d-b99e-6738eda01491
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
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="KSS Clustering Results - $Location", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm2 = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,2], hm2, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ 9623c9ec-59e2-4050-b18e-e1a36765a506
# new_assignments = aligned_assignments(model.assignments)

# ╔═╡ Cell order:
# ╟─baadc009-9be1-43bd-bc10-89610755f347
# ╟─e6c94de6-7b69-4003-b956-b603d218a4ea
# ╟─b59f17d2-8d4e-443e-83df-e70c8e45958b
# ╠═295bbe52-d1f7-4880-84df-639bf6be23b6
# ╠═f34d4731-9e7e-4624-bdc6-12f0b9734f11
# ╠═d9722d73-f0c1-4e8b-a099-b54c6ca89dec
# ╠═6e2611f1-e56a-455a-be2f-66bfdfb2f778
# ╠═24ec9d2f-5531-4bd8-b31e-c2245e002c74
# ╠═7bbd8a1c-c57b-4305-9198-40feba9b542c
# ╠═c1b19a5f-f387-4d11-a8cf-2c0b6c83904c
# ╠═96572a79-3e0c-47c3-8016-bd141745ee81
# ╠═1f8f3f29-b666-4986-8ff7-b8021a24cacb
# ╠═7d031d4c-62b0-41c4-9a98-969850c65648
# ╠═b126efef-20d2-4c39-b414-b5b030cbe457
# ╠═0220a0ef-755c-41e4-a555-18c5a58099f6
# ╟─023b3b11-7f99-4abe-93a3-e026b8a6168f
# ╠═f291fa32-a21d-466b-af5b-6cdebca7f1f8
# ╟─1bc7120d-5484-4b41-aab3-1c9a9a43629e
# ╠═781721b0-a273-45fd-89e6-c6a0d9f58bc0
# ╟─0a319270-951b-47e1-a4ea-24d4d8258864
# ╠═31801973-d6c1-4a33-8181-27b598676b86
# ╠═3a998668-be40-415d-bf67-503e01cae001
# ╟─eab3ca40-ea6f-4e42-9f2d-97116d67764f
# ╠═3f366f2c-4b9e-4512-bdc1-1e9be8ebc13f
# ╟─ed664957-8c6d-4899-b8d5-0ebd859fc9b3
# ╠═8bd009fb-4ea1-455e-9f7e-f8d81e844a73
# ╟─0f63ae9e-0a50-4739-bd3a-85a4ee35c809
# ╠═d34de99a-766f-4137-8b48-f71fb5293117
# ╠═094787c3-585e-4d96-9f96-bc0c80665b31
# ╟─480f6b51-ca0a-4f5e-ac86-1e2ee40ab06c
# ╠═a02a82ec-84f5-4429-aa36-3f1017f74332
# ╠═d0d303b6-1448-4e90-9a00-4cdde4b0f9f3
# ╠═75260ec6-391b-4ce3-a084-034c1651a161
# ╠═3cb67359-8491-4fb8-b85d-2f8070baf9c3
# ╠═9778bc47-7b27-4d37-bf47-82737c759c8b
# ╠═c48d26fe-82e4-4869-a48f-8c6dd66ec56a
# ╠═2f9623ce-e12d-44c0-a092-fb837864b92b
# ╠═d00a0185-afcf-4753-85ec-7c331c687507
# ╠═9cbc9294-87be-4d9c-a194-c19df153f79f
# ╠═af686d74-e702-416d-b99e-6738eda01491
# ╠═9623c9ec-59e2-4050-b18e-e1a36765a506
