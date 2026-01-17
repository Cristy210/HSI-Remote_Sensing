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

# ╔═╡ 00e482e6-52b3-4107-a7dc-eb75bb794182
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ 5cf97e7a-b1e3-42de-a750-a14e965f3621
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ProgressLogging, Dates, Logging, MAT

# ╔═╡ 25681354-a7cc-4495-a285-f33e4041828e
begin
    include(joinpath(@__DIR__, "..", "KSS_Julia", "KSS.jl"))
    using .KSS
end

# ╔═╡ b479f491-12fc-45e5-acee-8bc56171d68d
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

# ╔═╡ a0ed7b8e-6bd5-468c-9cda-2bc880a2949e
md"""
### This Notebook demonstrates the implementation of K-Subspaces Clustering (KSS) Clustering Algorithm on Pavia Dataset
"""

# ╔═╡ 3859c670-cc5a-4a6d-a726-9be7bbb7439b
md"""
#### Project Setup

> We start by activating the project environment and importing the required Julia packages.
"""

# ╔═╡ b7bd2ec1-d507-4898-abc6-ff3e24cd8fa4
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ 5710e281-81cb-4003-96af-4c96451ac689
filepath = abspath(joinpath(@__DIR__, "..", "MAT Files", "$Location.mat"))

# ╔═╡ d9a48313-efd5-4b4a-84ec-2716f621fbbe
gt_filepath = abspath(joinpath(@__DIR__, "..", "GT Files", "$Location.mat"))

# ╔═╡ 23c10baf-52df-438e-bb62-47baa9c42370
CACHEDIR = abspath(joinpath(@__DIR__, "..", "cache_files", "Aerial Datasets"))

# ╔═╡ 22678137-22a2-41d8-ab8f-2eec73bd8f9c
md"""
#### Defined cachet function to cache runs
"""

# ╔═╡ f8d95fab-e21f-4d95-9d87-019d88f61a9f
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 86d6e7d8-fc17-4cfc-825c-2e8ad2f978eb
vars = matread(filepath)

# ╔═╡ 146358e4-8da4-4ebd-a37c-6b1a7477cc59
vars_gt = matread(gt_filepath)

# ╔═╡ 70d0c6aa-cab1-44e5-98ca-d3cbf64799fe
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ c9db3733-bb99-450b-a110-b9ce4a349ebf
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 2c4560af-dc2a-4c42-aa8b-dda98e25be05
md"""
### Hyperspectral Image Cube - $Location
"""

# ╔═╡ b38f61b0-e6b6-435d-9417-4d4613368b5b
data = vars[data_key]

# ╔═╡ 44b9a078-f19a-4e45-9fe1-fa27f5d40ee9
md"""
### Ground Truth Data
"""

# ╔═╡ 25fa3169-b154-4e79-b8dd-8dddf7acc73c
gt_data = vars_gt[gt_key]

# ╔═╡ d3beb7e9-2dde-4f1d-9bc4-03a5c4142d64
md"""
### Ground Truth Labels
"""

# ╔═╡ 86bb34e0-199d-4996-911b-9e820c119d5f
gt_labels = sort(unique(gt_data))

# ╔═╡ 05a18305-8463-4f23-8139-19ef14ba3915
bg_indices = findall(gt_data .== 0)

# ╔═╡ 26a20e61-f629-4879-bfe0-b16dc24a4cd5
md"""
### Define mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ a3c618c2-0ec5-4b13-9197-11a9a40ad0ab
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ f611ff56-3583-47d4-b359-dd06991d9e98
md"""
##### Number of clusters equivalent to the number of unique labels from the ground truth data
"""

# ╔═╡ c7ea8f94-65e9-4ba3-a449-847584d519e7
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 331b8031-756a-43df-845b-cff3f84e85f6
md"""
#### Slider to choose the band of the image
"""

# ╔═╡ c7015ad6-7631-4414-a830-cd48bef09385
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 44a1fda0-64ee-4ec2-acb2-6494c3e23f84
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

# ╔═╡ 656337cc-98b9-4792-a41d-6c217f254ba7
md"""
> ###### K-Subspaces (KSS) assumes that the data points lies in an union of low-dimensional subspaces present within the original feature space and each data point could be represented using other data points that are lying in the same subspace. 
>
>###### The basis vectors for each subspace are formed by the number of left singular vectors from the Singular Value Decomposition (SVD). 

"""

# ╔═╡ 233b336e-c10b-4940-8bd8-f749d3dca1a7
md"""
`` 
\mathbf{U}_k^{\mathrm{dim}_k} = \hat{\mathbf{U}}[:, 1 : \mathrm{dim}_k]
\quad \text{where} \quad 
\mathcal{Y}_k = \hat{\mathbf{U}}\hat{\Sigma}\hat{\mathbf{V}}^{\top} 
\ \text{is an SVD}, 
\quad \text{for } k = 1, \ldots, K,
\quad d = \{ \mathrm{dim}_1, \dots, \mathrm{dim}_k \} 
 ``
"""

# ╔═╡ f69306b9-8512-4e1e-86e1-1be462460456
md"""
``
\text{classify}(\mathbf{y}) =
\underset{k \in \{1,\ldots,K\}}{\arg\min}
\ \bigl\lVert \mathbf{y} -
\mathbf{U}_k^{\mathrm{dim}_k}
(\mathbf{U}_k^{\mathrm{dim}_k})^{\top}
\mathbf{y} \bigr\rVert_2
``
"""

# ╔═╡ e699f5a9-064b-4956-865e-61550233d27d
K = [2, 2, 2, 2, 2, 2, 2, 2, 2]

# ╔═╡ 91c971c4-55cc-4265-8674-28736ebad4d3
# data[mask, :]

# ╔═╡ c8788cc0-0898-44db-a954-04382cebb9db
md"""
#### Fit the K-Subspace model
"""

# ╔═╡ 34253612-b62e-4990-a904-aea0141a5282
model = fit(data[mask, :]', K)

# ╔═╡ 8b89d5e3-b5f5-408c-887a-11c3c29a5394
md"""
#### Subspace Basis
"""

# ╔═╡ 28f97be3-40be-4dd1-9c6d-2ec403217082
subspace_basis = model.U

# ╔═╡ bafe3d75-88fd-4465-a1e8-e3e7bc8d1539
md"""
#### Labels
"""

# ╔═╡ d64aae0e-7675-4be2-8d47-5f99c6491f7d
labels = model.c

# ╔═╡ 3a21881f-5e38-4c9f-8204-4fcc047c4d4a
md"""
#### Total Cost
"""

# ╔═╡ 775a4119-8355-4fba-9983-ca59dd7bee8a
totalcost = model.totalcost

# ╔═╡ 9757fec6-0d5b-42a9-ae1c-6c98a9efbde9
md"""
#### Converged
"""

# ╔═╡ 2537f938-126a-4d32-adf6-6c8b234ce1c0
converged = model.converged

# ╔═╡ b1f29dd5-1231-4f88-820c-daade6001b19
md"""
#### Pixel count for each label
"""

# ╔═╡ 44d83fee-3dc3-4800-96ac-1799909795e7
counts = model.counts

# ╔═╡ b769d789-bf2a-420b-808f-a1d511d5884b
md"""
#### Relabel the clusters to compare it with the ground truth
"""

# ╔═╡ ffa237cf-85c6-4d01-b656-82092fd2856d
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 6,
	2 => 4,
	3 => 2,
	4 => 7,
	5 => 1,
	6 => 8,
	7 => 5,
	8 => 3,
	9 => 9
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

# ╔═╡ 3eec9db8-f1ad-40ce-bd2c-adf134e3ded4
relabel_keys = relabel_maps[Location]

# ╔═╡ 8e8577cc-50be-406e-83fd-1526bddc9029
D_relabel = [relabel_keys[label] for label in labels]

# ╔═╡ 4fe5fe46-697e-4201-88ea-09c38cd7370d
md"""
### Ground Truth Vs. Clustering Result
"""

# ╔═╡ f78fa1e2-215f-40f2-8e71-98c2a3176698
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

# ╔═╡ 2c44b212-2379-4be4-891c-591fb60b0eae
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

# ╔═╡ ea40eab7-a990-4fd7-977d-0084288e9345
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="Confusion Matrix - KSS Clustering - $Location")
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
# ╟─b479f491-12fc-45e5-acee-8bc56171d68d
# ╟─a0ed7b8e-6bd5-468c-9cda-2bc880a2949e
# ╟─3859c670-cc5a-4a6d-a726-9be7bbb7439b
# ╠═00e482e6-52b3-4107-a7dc-eb75bb794182
# ╠═5cf97e7a-b1e3-42de-a750-a14e965f3621
# ╠═b7bd2ec1-d507-4898-abc6-ff3e24cd8fa4
# ╠═5710e281-81cb-4003-96af-4c96451ac689
# ╠═d9a48313-efd5-4b4a-84ec-2716f621fbbe
# ╠═23c10baf-52df-438e-bb62-47baa9c42370
# ╟─22678137-22a2-41d8-ab8f-2eec73bd8f9c
# ╠═f8d95fab-e21f-4d95-9d87-019d88f61a9f
# ╠═86d6e7d8-fc17-4cfc-825c-2e8ad2f978eb
# ╠═146358e4-8da4-4ebd-a37c-6b1a7477cc59
# ╠═70d0c6aa-cab1-44e5-98ca-d3cbf64799fe
# ╠═c9db3733-bb99-450b-a110-b9ce4a349ebf
# ╟─2c4560af-dc2a-4c42-aa8b-dda98e25be05
# ╠═b38f61b0-e6b6-435d-9417-4d4613368b5b
# ╟─44b9a078-f19a-4e45-9fe1-fa27f5d40ee9
# ╠═25fa3169-b154-4e79-b8dd-8dddf7acc73c
# ╠═d3beb7e9-2dde-4f1d-9bc4-03a5c4142d64
# ╠═86bb34e0-199d-4996-911b-9e820c119d5f
# ╠═05a18305-8463-4f23-8139-19ef14ba3915
# ╟─26a20e61-f629-4879-bfe0-b16dc24a4cd5
# ╠═a3c618c2-0ec5-4b13-9197-11a9a40ad0ab
# ╟─f611ff56-3583-47d4-b359-dd06991d9e98
# ╠═c7ea8f94-65e9-4ba3-a449-847584d519e7
# ╟─331b8031-756a-43df-845b-cff3f84e85f6
# ╠═c7015ad6-7631-4414-a830-cd48bef09385
# ╠═44a1fda0-64ee-4ec2-acb2-6494c3e23f84
# ╟─656337cc-98b9-4792-a41d-6c217f254ba7
# ╟─233b336e-c10b-4940-8bd8-f749d3dca1a7
# ╟─f69306b9-8512-4e1e-86e1-1be462460456
# ╠═25681354-a7cc-4495-a285-f33e4041828e
# ╠═e699f5a9-064b-4956-865e-61550233d27d
# ╠═91c971c4-55cc-4265-8674-28736ebad4d3
# ╟─c8788cc0-0898-44db-a954-04382cebb9db
# ╠═34253612-b62e-4990-a904-aea0141a5282
# ╟─8b89d5e3-b5f5-408c-887a-11c3c29a5394
# ╠═28f97be3-40be-4dd1-9c6d-2ec403217082
# ╟─bafe3d75-88fd-4465-a1e8-e3e7bc8d1539
# ╠═d64aae0e-7675-4be2-8d47-5f99c6491f7d
# ╟─3a21881f-5e38-4c9f-8204-4fcc047c4d4a
# ╠═775a4119-8355-4fba-9983-ca59dd7bee8a
# ╟─9757fec6-0d5b-42a9-ae1c-6c98a9efbde9
# ╠═2537f938-126a-4d32-adf6-6c8b234ce1c0
# ╟─b1f29dd5-1231-4f88-820c-daade6001b19
# ╠═44d83fee-3dc3-4800-96ac-1799909795e7
# ╟─b769d789-bf2a-420b-808f-a1d511d5884b
# ╠═ffa237cf-85c6-4d01-b656-82092fd2856d
# ╠═3eec9db8-f1ad-40ce-bd2c-adf134e3ded4
# ╠═8e8577cc-50be-406e-83fd-1526bddc9029
# ╟─4fe5fe46-697e-4201-88ea-09c38cd7370d
# ╠═f78fa1e2-215f-40f2-8e71-98c2a3176698
# ╠═2c44b212-2379-4be4-891c-591fb60b0eae
# ╠═ea40eab7-a990-4fd7-977d-0084288e9345
