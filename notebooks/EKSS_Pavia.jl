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

# ╔═╡ 7e58ee95-6276-4133-9a69-bab79adf22c1
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ aacd15ee-d9d1-44c9-b74d-e19f11c624d3
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ProgressLogging, Dates, Logging, MAT, CacheVariables

# ╔═╡ 9c682225-c728-43fd-bfe6-7aa645658a0b
begin
    include(joinpath(@__DIR__, "..", "EKSS_Julia", "ekss.jl"))
    using .EKSS
end

# ╔═╡ f064b053-ebb5-49da-be64-409b2f670be0
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

# ╔═╡ 8a1fbe11-b0d5-4900-9247-cbf6953389b0
md"""
### This Notebook demonstrates the implementation of Ensemble K-Subspaces Clustering (EKSS) Clustering Algorithm on Pavia Dataset
"""

# ╔═╡ 3fa65339-cd96-4cfb-8663-05b10646ad5b
md"""
#### Project Setup

> We start by activating the project environment and importing the required Julia packages.
"""

# ╔═╡ 564c316a-a762-481e-bae1-5f96199bbe5a
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ bb111521-cd8a-404f-84d3-2fba9b52e236
filepath = abspath(joinpath(@__DIR__, "..", "MAT Files", "$Location.mat"))

# ╔═╡ 173c4379-4513-46a9-a09b-c3226fe9a54d
gt_filepath = abspath(joinpath(@__DIR__, "..", "GT Files", "$Location.mat"))

# ╔═╡ 0bd2cf71-329d-418f-90a7-8f9e7a20872e
CACHEDIR = abspath(joinpath(@__DIR__, "..", "cache_files", "Aerial Datasets"))

# ╔═╡ ff448999-94a1-419c-9d51-7f144b0c96b0
md"""
#### Defined cachet function to cache runs
"""

# ╔═╡ 890a9c97-0b20-4cc5-b006-5f94104787a6
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 7dc8381e-41ff-49d7-bcd6-b5ad144cc15c
vars = matread(filepath)

# ╔═╡ f1943287-7fbb-44d4-a037-801c5726a7ee
vars_gt = matread(gt_filepath)

# ╔═╡ 566a8470-a12a-4ee8-abbc-6918489e75ca
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ ad519d54-7d7f-453b-9aa9-95719138bb00
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 8e4fe12d-272c-45cf-b9ca-e0af48d35b9d
md"""
### Hyperspectral Image Cube - $Location
"""

# ╔═╡ c2d05464-1a7b-4926-b335-b95f1e9b56d9
data = vars[data_key]

# ╔═╡ de4df40c-71c0-4452-9b13-f3dc8fca53a8
md"""
### Ground Truth Data
"""

# ╔═╡ 1d562e6a-4975-40ba-bf49-128aaba1ec40
gt_data = vars_gt[gt_key]

# ╔═╡ 19876584-7aac-4145-b4fc-4812539c4547
md"""
### Ground Truth Labels
"""

# ╔═╡ 6b497645-8d10-44fc-831b-190ea780118d
gt_labels = sort(unique(gt_data))

# ╔═╡ 0bbf9db6-7225-4075-8ab1-11d3507de0f8
bg_indices = findall(gt_data .== 0)

# ╔═╡ 382e5182-5771-4946-a408-fa280e46817e
md"""
### Define mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 838ef749-8c22-41b2-b4d9-9647e44d6364
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 2766502b-4c3b-4c5e-bfad-9c1acefcb6fd
md"""
##### Number of clusters equivalent to the number of unique labels from the ground truth data
"""

# ╔═╡ 78144c74-6406-4394-bb3f-f7f35e755847
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ f917bf5c-0db4-401e-86f7-3568ee3466fd
md"""
#### Slider to choose the band of the image
"""

# ╔═╡ c50aeb77-adf7-458d-a632-68b5f92f47d5
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 02bebeff-2f55-458a-ba57-0aed5b144949
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

# ╔═╡ 1227b4a1-6bee-4dd8-af46-a7be4f66893d
Threads.nthreads()

# ╔═╡ 32d645f8-4799-4f4e-bd72-39bf39bde29a
@bind q PlutoUI.Slider(500:2000; show_value=true, default=1802)

# ╔═╡ 63fbc23c-10d2-489b-a5ea-ffeaad4ae30e
@bind nruns PlutoUI.Slider(20:50; show_value=true, default=20)

# ╔═╡ 3f5d5f80-1705-4741-8ce5-c1efc327542f
@bind Kbar PlutoUI.Slider(3:8; show_value=true, default=7)

# ╔═╡ d26cb582-f4d7-4ac3-88ac-6e10f8f1678c
labels = cache(joinpath(CACHEDIR, "EKSS_$(Location)_$(q)_$(nruns)_$(Kbar)_labels.bson")) do
    model = fit(data[mask, :]', 1, n_clusters;
        Kbar=Kbar,
		parallel=true,
        nruns=nruns,
        q=q,
    )

    model.assignments
end

# ╔═╡ 75a0aa1b-46db-4f18-b3c7-8d2ae2abb4a4
# labels = model.assignments

# ╔═╡ c0ab09e7-8a15-46db-bd2f-3c919db3e3f3
unique(labels)

# ╔═╡ 7cdb0e0d-e193-49a2-aa0f-27d363fa98ff
# N = size(model.coassoc, 1)

# ╔═╡ 6b852205-3791-49d9-bb73-599aa4b9484d
# max(300, round(Int, 0.01 * N))

# ╔═╡ a13da7f6-b291-4373-b891-feb948fe58b4
# degrees = vec(sum(model.coassoc; dims=2))

# ╔═╡ 1343229c-5a94-4725-b915-f622d1e4ffc1
# minimum(degrees), maximum(degrees), count(==(0), degrees)

# ╔═╡ 9e9dc36d-b48c-4844-a5e8-e4c71ab4bb22
md"""
#### Relabel the clusters to compare it with the ground truth
"""

# ╔═╡ a46391de-49cc-4bad-856a-8e8597956d54
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 8,
	2 => 1,
	3 => 7,
	4 => 2,
	5 => 9,
	6 => 5,
	7 => 6,
	8 => 3,
	9 => 4
),
	"PaviaUni" => Dict(
	0 => 0,
	1 => 9,
	2 => 1,
	3 => 2,
	4 => 5,
	5 => 4,
	6 => 6,
	7 => 8,
	8 => 3,
	9 => 7,
)
)

# ╔═╡ e8d00032-8d42-438d-b525-3a70f288c4dd
relabel_keys = relabel_maps[Location]

# ╔═╡ 13db233f-5f6c-49dd-abfe-ce3b71d41d82
D_relabel = [relabel_keys[label] for label in labels]

# ╔═╡ becb1539-f2a7-4584-a58a-b51bbd68fc3b
md"""
### Ground Truth Vs. Clustering Result
"""

# ╔═╡ e2ad7609-1831-49a9-b1db-3ba32f6b8421
with_theme() do

	# Create figure
	fig = Figure(; size=(950, 650))
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# subgrid = fig[1, 1] = GridLayout()

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
	
	hm1 = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,1], hm1, tellwidth=false, vertical=false)

	# Show cluster map
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="EKSS Clustering Results - $Location", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm2 = heatmap!(ax, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(fig[2,2], hm2, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ 6f03411f-a2fc-417a-b683-8e0659ff025a
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

# ╔═╡ 4e274694-4451-4836-acbf-bc0b1eff79df
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="Confusion Matrix - EKSS Clustering - $Location")
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
# ╟─f064b053-ebb5-49da-be64-409b2f670be0
# ╟─8a1fbe11-b0d5-4900-9247-cbf6953389b0
# ╟─3fa65339-cd96-4cfb-8663-05b10646ad5b
# ╠═7e58ee95-6276-4133-9a69-bab79adf22c1
# ╠═aacd15ee-d9d1-44c9-b74d-e19f11c624d3
# ╠═564c316a-a762-481e-bae1-5f96199bbe5a
# ╠═bb111521-cd8a-404f-84d3-2fba9b52e236
# ╠═173c4379-4513-46a9-a09b-c3226fe9a54d
# ╠═0bd2cf71-329d-418f-90a7-8f9e7a20872e
# ╟─ff448999-94a1-419c-9d51-7f144b0c96b0
# ╠═890a9c97-0b20-4cc5-b006-5f94104787a6
# ╠═7dc8381e-41ff-49d7-bcd6-b5ad144cc15c
# ╠═f1943287-7fbb-44d4-a037-801c5726a7ee
# ╠═566a8470-a12a-4ee8-abbc-6918489e75ca
# ╠═ad519d54-7d7f-453b-9aa9-95719138bb00
# ╟─8e4fe12d-272c-45cf-b9ca-e0af48d35b9d
# ╠═c2d05464-1a7b-4926-b335-b95f1e9b56d9
# ╟─de4df40c-71c0-4452-9b13-f3dc8fca53a8
# ╠═1d562e6a-4975-40ba-bf49-128aaba1ec40
# ╠═19876584-7aac-4145-b4fc-4812539c4547
# ╠═6b497645-8d10-44fc-831b-190ea780118d
# ╠═0bbf9db6-7225-4075-8ab1-11d3507de0f8
# ╟─382e5182-5771-4946-a408-fa280e46817e
# ╠═838ef749-8c22-41b2-b4d9-9647e44d6364
# ╟─2766502b-4c3b-4c5e-bfad-9c1acefcb6fd
# ╠═78144c74-6406-4394-bb3f-f7f35e755847
# ╟─f917bf5c-0db4-401e-86f7-3568ee3466fd
# ╠═c50aeb77-adf7-458d-a632-68b5f92f47d5
# ╠═02bebeff-2f55-458a-ba57-0aed5b144949
# ╠═9c682225-c728-43fd-bfe6-7aa645658a0b
# ╠═1227b4a1-6bee-4dd8-af46-a7be4f66893d
# ╠═32d645f8-4799-4f4e-bd72-39bf39bde29a
# ╠═63fbc23c-10d2-489b-a5ea-ffeaad4ae30e
# ╠═3f5d5f80-1705-4741-8ce5-c1efc327542f
# ╠═d26cb582-f4d7-4ac3-88ac-6e10f8f1678c
# ╠═75a0aa1b-46db-4f18-b3c7-8d2ae2abb4a4
# ╠═c0ab09e7-8a15-46db-bd2f-3c919db3e3f3
# ╠═7cdb0e0d-e193-49a2-aa0f-27d363fa98ff
# ╠═6b852205-3791-49d9-bb73-599aa4b9484d
# ╠═a13da7f6-b291-4373-b891-feb948fe58b4
# ╠═1343229c-5a94-4725-b915-f622d1e4ffc1
# ╟─9e9dc36d-b48c-4844-a5e8-e4c71ab4bb22
# ╠═a46391de-49cc-4bad-856a-8e8597956d54
# ╠═e8d00032-8d42-438d-b525-3a70f288c4dd
# ╠═13db233f-5f6c-49dd-abfe-ce3b71d41d82
# ╟─becb1539-f2a7-4584-a58a-b51bbd68fc3b
# ╠═e2ad7609-1831-49a9-b1db-3ba32f6b8421
# ╠═6f03411f-a2fc-417a-b683-8e0659ff025a
# ╠═4e274694-4451-4836-acbf-bc0b1eff79df
