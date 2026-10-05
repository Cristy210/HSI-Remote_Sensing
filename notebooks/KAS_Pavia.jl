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

# ╔═╡ b8800e73-a755-4a83-ab98-d121f8e9d80c
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ 0b1a39fa-0f1d-4573-90e1-b95660499d38
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ProgressLogging, Dates, Logging, MAT

# ╔═╡ dd843d8a-db29-42cc-850d-3ac1e7727459
begin
    include(joinpath(@__DIR__, "..", "KAS_Julia", "KAS.jl"))
    using .KAS
end

# ╔═╡ cad9c15b-b89d-45ca-9478-5a9c0cb769a5
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

# ╔═╡ 1e4668ad-33c3-4750-9eb4-2c487e69150b
md"""
### This Notebook demonstrates the implementation of K-Subspaces Clustering (KSS) Clustering Algorithm on Pavia Dataset
"""

# ╔═╡ c34461b9-d2a4-496d-abe6-bd8b17d8029a
md"""
#### Project Setup

> We start by activating the project environment and importing the required Julia packages.
"""

# ╔═╡ 461d3eda-2599-4a93-9a8c-b3b20d530082
@bind Location Select(["Pavia",])

# ╔═╡ 53510449-6e63-4efe-9b63-50e5b2f31012
filepath = abspath(joinpath(@__DIR__, "..", "MAT Files", "$Location.mat"))

# ╔═╡ a92b1778-110a-4bb0-93c5-d8ff287c101d
gt_filepath = abspath(joinpath(@__DIR__, "..", "GT Files", "$Location.mat"))

# ╔═╡ 5b5b2faf-8aea-4c1b-a864-880de8397772
CACHEDIR = abspath(joinpath(@__DIR__, "..", "cache_files", "Aerial Datasets"))

# ╔═╡ fb63e744-ff38-4143-bd99-757dc2919513
md"""
#### Defined cachet function to cache runs
"""

# ╔═╡ abded5c9-31dd-48c6-86d5-d2b319c7767e
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 498eb728-5e1b-4847-8ae2-a8182f84488b
vars = matread(filepath)

# ╔═╡ b6689d9c-b213-4c25-a0d3-099374b22e21
vars_gt = matread(gt_filepath)

# ╔═╡ 156d323f-f5e5-43c6-baeb-12ced16c2aec
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 8a1c8a30-0f9a-400a-ac34-1cdafde6e7ee
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 6068e7d5-da06-4b81-a67e-5a7813bf9ed3
md"""
### Hyperspectral Image Cube - $Location
"""

# ╔═╡ e06950c9-7fbe-45ff-9546-d73146242661
data = vars[data_key]

# ╔═╡ e2a2d12a-4a67-4902-a7e5-bbfcc3978566
md"""
### Ground Truth Data
"""

# ╔═╡ 6ef32c32-78b1-4bf9-aa51-1421ed88a327
gt_data = vars_gt[gt_key]

# ╔═╡ b22638fd-a261-49cc-a52b-166cf5173e2e
md"""
### Ground Truth Labels
"""

# ╔═╡ acf5522b-e642-4dc9-bf1a-e75f3d14d8b1
gt_labels = sort(unique(gt_data))

# ╔═╡ 809d8618-b089-491e-a15d-6aae3a68cf70
bg_indices = findall(gt_data .== 0)

# ╔═╡ a07edf2a-09c0-4177-9ca7-cb80eba042be
md"""
### Define mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 9b43c42e-2479-477d-af73-b4888ffee575
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ b01e4773-fe99-493e-819c-e113173a3ed5
md"""
##### Number of clusters equivalent to the number of unique labels from the ground truth data
"""

# ╔═╡ d9e95bbe-d34b-4edc-9d1d-57c0be5679b0
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ dff09346-864d-4cf7-b591-fe1dd2773e0a
md"""
#### Slider to choose the band of the image
"""

# ╔═╡ b9779add-a9ec-4835-850c-0d2aab39d1c9
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 879391cb-14b5-44eb-8291-e8d3c6ac5a50
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

# ╔═╡ 4d193f05-1ab4-4308-9074-4d1f779a9645
K = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# ╔═╡ 3d1f2541-fd65-4640-9d63-9f5dddeff30e
# data[mask, :]

# ╔═╡ c949586a-bb46-4b85-aa24-0d989190f46e
md"""
#### Fit the K-Affine spaces model
"""

# ╔═╡ b3b7e9ef-813e-48b2-94fc-c72869744b09
model = fit(data[mask, :]', K)

# ╔═╡ 5b187ae4-8ef9-4578-b5bd-e0cdb4ef9aea
md"""
#### Affine space Basis
"""

# ╔═╡ db785f7e-6a3b-4e3a-a10b-a1814c70d44c
subspace_basis = model.U

# ╔═╡ 00175646-2cff-415d-966b-afc46220f971
md"""
#### Labels
"""

# ╔═╡ c54875bf-7bfc-4993-98f5-0753fed6a907
labels = model.c

# ╔═╡ 3ee3dbdd-0050-4ee9-962c-109d16cba65b
md"""
#### Total Cost
"""

# ╔═╡ fef765ab-de21-49b8-8e39-e2a11d965a18
totalcost = model.totalcost

# ╔═╡ 11d85411-904f-4626-b3a0-1b2e3ebef362
md"""
#### Converged
"""

# ╔═╡ 6b74369b-5bc4-48bf-b94e-cc7d6f19816c
converged = model.converged

# ╔═╡ 52e2ac37-023d-4d99-bb32-07ac36bc3606
md"""
#### Pixel count for each label
"""

# ╔═╡ 22962796-594e-463b-ac23-ff6be3f669ae
counts = model.counts

# ╔═╡ c81b00d8-9771-4f3a-adc4-faaeb908d728
md"""
#### Relabel the clusters to compare it with the ground truth
"""

# ╔═╡ d76866f6-b71f-4158-8163-977c111024b0
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 8,
	2 => 4,
	3 => 5,
	4 => 7,
	5 => 9,
	6 => 3,
	7 => 2,
	8 => 6,
	9 => 1
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

# ╔═╡ 40b0c793-dd34-4707-879d-65abc28f37ec
relabel_keys = relabel_maps[Location]

# ╔═╡ baa6acb7-6fb8-4c84-8d61-5cb32ff19744
D_relabel = [relabel_keys[label] for label in labels]

# ╔═╡ 77166d97-f79b-45da-a630-e3d18eb8a35a
md"""
### Ground Truth Vs. Clustering Result
"""

# ╔═╡ e635ae66-7cdc-43af-9a77-513a3883e008
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
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="KAS Clustering Results - $Location", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm2 = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,2], hm2, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ Cell order:
# ╟─cad9c15b-b89d-45ca-9478-5a9c0cb769a5
# ╟─1e4668ad-33c3-4750-9eb4-2c487e69150b
# ╟─c34461b9-d2a4-496d-abe6-bd8b17d8029a
# ╠═b8800e73-a755-4a83-ab98-d121f8e9d80c
# ╠═0b1a39fa-0f1d-4573-90e1-b95660499d38
# ╠═461d3eda-2599-4a93-9a8c-b3b20d530082
# ╠═53510449-6e63-4efe-9b63-50e5b2f31012
# ╠═a92b1778-110a-4bb0-93c5-d8ff287c101d
# ╠═5b5b2faf-8aea-4c1b-a864-880de8397772
# ╟─fb63e744-ff38-4143-bd99-757dc2919513
# ╠═abded5c9-31dd-48c6-86d5-d2b319c7767e
# ╠═498eb728-5e1b-4847-8ae2-a8182f84488b
# ╠═b6689d9c-b213-4c25-a0d3-099374b22e21
# ╠═156d323f-f5e5-43c6-baeb-12ced16c2aec
# ╠═8a1c8a30-0f9a-400a-ac34-1cdafde6e7ee
# ╟─6068e7d5-da06-4b81-a67e-5a7813bf9ed3
# ╠═e06950c9-7fbe-45ff-9546-d73146242661
# ╟─e2a2d12a-4a67-4902-a7e5-bbfcc3978566
# ╠═6ef32c32-78b1-4bf9-aa51-1421ed88a327
# ╟─b22638fd-a261-49cc-a52b-166cf5173e2e
# ╠═acf5522b-e642-4dc9-bf1a-e75f3d14d8b1
# ╠═809d8618-b089-491e-a15d-6aae3a68cf70
# ╠═a07edf2a-09c0-4177-9ca7-cb80eba042be
# ╠═9b43c42e-2479-477d-af73-b4888ffee575
# ╠═b01e4773-fe99-493e-819c-e113173a3ed5
# ╠═d9e95bbe-d34b-4edc-9d1d-57c0be5679b0
# ╠═dff09346-864d-4cf7-b591-fe1dd2773e0a
# ╠═b9779add-a9ec-4835-850c-0d2aab39d1c9
# ╠═879391cb-14b5-44eb-8291-e8d3c6ac5a50
# ╠═dd843d8a-db29-42cc-850d-3ac1e7727459
# ╠═4d193f05-1ab4-4308-9074-4d1f779a9645
# ╠═3d1f2541-fd65-4640-9d63-9f5dddeff30e
# ╠═c949586a-bb46-4b85-aa24-0d989190f46e
# ╠═b3b7e9ef-813e-48b2-94fc-c72869744b09
# ╠═5b187ae4-8ef9-4578-b5bd-e0cdb4ef9aea
# ╠═db785f7e-6a3b-4e3a-a10b-a1814c70d44c
# ╠═00175646-2cff-415d-966b-afc46220f971
# ╠═c54875bf-7bfc-4993-98f5-0753fed6a907
# ╠═3ee3dbdd-0050-4ee9-962c-109d16cba65b
# ╠═fef765ab-de21-49b8-8e39-e2a11d965a18
# ╠═11d85411-904f-4626-b3a0-1b2e3ebef362
# ╠═6b74369b-5bc4-48bf-b94e-cc7d6f19816c
# ╠═52e2ac37-023d-4d99-bb32-07ac36bc3606
# ╠═22962796-594e-463b-ac23-ff6be3f669ae
# ╠═c81b00d8-9771-4f3a-adc4-faaeb908d728
# ╠═d76866f6-b71f-4158-8163-977c111024b0
# ╠═40b0c793-dd34-4707-879d-65abc28f37ec
# ╠═baa6acb7-6fb8-4c84-8d61-5cb32ff19744
# ╠═77166d97-f79b-45da-a630-e3d18eb8a35a
# ╠═e635ae66-7cdc-43af-9a77-513a3883e008
