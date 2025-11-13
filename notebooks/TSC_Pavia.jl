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
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 58baac76-e492-4c7d-8d84-a5a571c81c7d
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

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
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 92620cae-f1bd-4864-a795-d2a8c8dbf16e
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ c8376c69-f1e6-459f-aedd-766e2d84cb3a
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

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

# ╔═╡ 56ae6711-4645-4981-81a5-4ddc5c5fc521


# ╔═╡ 29242190-f547-4fe5-abcd-2dac57200163


# ╔═╡ c9697307-e424-42e3-9e61-100e9d2eb61b
md"""
### Affinity Matrix
"""

# ╔═╡ 967709d5-ac33-4ec0-b2d2-c50d72dc8017
begin
function affinity(X::Matrix; max_nz=10, chunksize=isqrt(size(X,2)),
	func = c -> exp(-2*acos(clamp(c,-1,1))))

	# Compute normalized spectra (so that inner product = cosine of angle)
	X = mapslices(normalize, X; dims=1)

	# Find nonzero values (in chunks)
	C_buf = similar(X, size(X,2), chunksize)    # pairwise cosine buffer
	s_buf = Vector{Int}(undef, size(X,2))       # sorting buffer
	nz_list = @withprogress mapreduce(vcat, enumerate(Iterators.partition(1:size(X,2), chunksize))) do (chunk_idx, chunk)
		# Compute cosine angles (for chunk) and store in appropriate buffer
		C_chunk = length(chunk) == chunksize ? C_buf : similar(X, size(X,2), length(chunk))
		mul!(C_chunk, X', view(X, :, chunk))

		# Zero out all but `max_nz` largest values
		nzs = map(chunk, eachcol(C_chunk)) do col, c
			idx = partialsortperm!(s_buf, c, 1:max_nz; rev=true)
			collect(idx), fill(col, max_nz), func.(view(c,idx))
		end

		# Log progress and return
		@logprogress chunk_idx/cld(size(X,2),chunksize)
		return nzs
	end

	# Form and return sparse array
	rows = reduce(vcat, getindex.(nz_list, 1))
	cols = reduce(vcat, getindex.(nz_list, 2))
	vals = reduce(vcat, getindex.(nz_list, 3))
	return sparse([rows; cols],[cols; rows],[vals; vals])
end
affinity(cube::Array{<:Real,3}; kwargs...) =
	affinity(permutedims(reshape(cube, :, size(cube,3))); kwargs...)
end

# ╔═╡ 4ce6a047-e4f8-41a2-9c4a-94643e8ce737


# ╔═╡ efb0c7d5-28f3-44cc-a521-3bb1464da64a
max_nz = 10

# ╔═╡ 033d4da1-eba3-4c59-b945-2d59b81906e6
A = cachet(joinpath(CACHEDIR, "Affinity_$Location$max_nz.bson")) do
	affinity(permutedims(data[mask, :]); max_nz)
end

# ╔═╡ 8b2fa6e6-04ac-4b7b-a29c-f1c38dbcfdb1
md"""
#### Eigenvector embedding: Compute the `k` eigenvectors of the normalized Laplacian, where `k` equals the number of clusters.
"""

# ╔═╡ 0fa5a2ea-1a4d-4ed5-9ae1-a0a3280d0bfe
function embedding(A, k; seed=0)
	# Set seed for reproducibility
	Random.seed!(seed)

	# Compute node degrees and form Laplacian
	d = vec(sum(A; dims=2))
	Dsqrinv = sqrt(inv(Diagonal(d)))
	L = Symmetric(I - (Dsqrinv * A) * Dsqrinv)

	# Compute eigenvectors
	decomp, history = partialschur(L; nev=k, which=:SR)
	@info history

	return mapslices(normalize, decomp.Q; dims=2)
end

# ╔═╡ dae400bf-ea56-4ae8-9f0c-56ea45ef2652
V = embedding(A, n_clusters)

# ╔═╡ 2d467a37-fcf8-4b63-8604-64216f64cb57
md"""
#### Run multiple K-Means runs with different random seeds and pick the clustering result with the  best run (lowest cost)
"""

# ╔═╡ f38c1c54-4e37-4c75-a67d-1b4378de0227
function batchkmeans(X, k, args...; nruns=100, kwargs...)
	runs = @withprogress map(1:nruns) do idx
		# Run K-means
		Random.seed!(idx)  # set seed for reproducibility
		result = with_logger(NullLogger()) do
			kmeans(X, k, args...; kwargs...)
		end

		# Log progress and return result
		@logprogress idx/nruns
		return result
	end

	# Print how many converged
	nconverged = count(run -> run.converged, runs)
	@info "$nconverged/$nruns runs converged"

	# Return runs sorted best to worst
	return sort(runs; by=run->run.totalcost)
end

# ╔═╡ ac511c25-b9bd-40a9-b03d-6032faac6781
spec_clusterings = batchkmeans(permutedims(V), n_clusters; maxiter=1000)

# ╔═╡ a8f26017-5e2a-4991-a840-68495ccece02
costs = [spec_clusterings[i].totalcost for i in 1:100]

# ╔═╡ 4d85331b-65bb-46fa-8a8f-3634ee7eacc9
min_index = argmin(costs)

# ╔═╡ d4cb1d09-890a-4c8a-a5d0-ddc8f6d166eb
aligned_assignments(clusterings, baseperm=1:maximum(first(clusterings).assignments)) = map(clusterings) do clustering
	# New labels determined by simple heuristic that aims to align different clusterings
	thresh! = a -> (a[a .< 0.2*sum(a)] .= 0; a)
	alignments = [thresh!(a) for a in eachrow(counts(clusterings[1], clustering))]
	new_labels = sortperm(alignments[baseperm]; rev=true)

	# Return assignments with new labels
	return [new_labels[l] for l in clustering.assignments]
end

# ╔═╡ 5414cbcb-c322-4536-b9c3-edd9fca2ca11
spec_aligned = aligned_assignments(spec_clusterings)

# ╔═╡ 6b6a0558-ada2-4193-a5f3-a4878421f914
md"""
### Confusion Matrix -- Clustering Results
"""

# ╔═╡ d21295be-cbd5-4135-b2e3-7925ad133581
@bind spec_clustering_idx PlutoUI.Slider(1:length(spec_clusterings); show_value=true)

# ╔═╡ db8c4072-c07b-41b5-8baf-237665a85bfd
begin
	ground_labels = filter(x -> x != 0, gt_labels) #Filter out the background pixel label
	true_labels = length(ground_labels)
	predicted_labels = n_clusters

	confusion_matrix = zeros(Float64, true_labels, predicted_labels) #Initialize a confusion matrix filled with zeros
	cluster_results = fill(NaN32, size(data)[1:2]) #Clustering algorithm results

	clu_assign, idx = spec_aligned, spec_clustering_idx

	cluster_results[mask] .= clu_assign[idx]

	for (label_idx, label) in enumerate(ground_labels)
	
		label_indices = findall(gt_data .== label)
	
		cluster_values = [cluster_results[idx] for idx in label_indices]
		t_pixels = length(cluster_values)
		cluster_counts = [count(==(cluster), cluster_values) for cluster in 1:n_clusters]
		confusion_matrix[label_idx, :] .= [count / t_pixels * 100 for count in cluster_counts]
	end
end

# ╔═╡ aaf81673-a329-406e-abd0-2041a6875355
with_theme() do
	fig = Figure(; size=(600, 700))
	ax = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels, yticks = 1:true_labels)
	hm = heatmap!(ax, permutedims(confusion_matrix), colormap=:viridis)
	pm = permutedims(confusion_matrix)

	for i in 1:true_labels, j in 1:predicted_labels
        value = round(pm[i, j], digits=1)
        text!(ax, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=14)
    end
	Colorbar(fig[1, 2], hm)
	fig
end;

# ╔═╡ 0a6643cf-fd62-43e3-a427-e3b4b435d047
relabel_maps = Dict(
	"Pavia" => Dict(
	0 => 0,
	1 => 8,
	2 => 3,
	3 => 1,
	4 => 6,
	5 => 7,
	6 => 2,
	7 => 9,
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

# ╔═╡ bb45da10-c4c3-42c9-99f7-977cd13bcfab
relabel_keys = relabel_maps[Location]

# ╔═╡ 17bd5e56-1fad-42ac-8d5d-bc6ccc26a9a3
D_relabel = [relabel_keys[label] for label in spec_aligned[1]]

# ╔═╡ 9c2d52df-5704-4542-b237-0ed299bf51f6
md"""
### Confusion Matrix -- Best Clustering Result
"""

# ╔═╡ 6859158c-1078-4c1a-8cd0-28e31329cf46
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

# ╔═╡ 62e7fc9b-180a-44d1-bc77-fbd48b49d58b
with_theme() do
	assignments, idx = spec_aligned, spec_clustering_idx
	

	# Create figure
	fig = Figure(; size=(1200, 600))
	supertitle = Label(fig[0, 1:3], "Thresholded Subspace Clustering (TSC) Results on $Location Dataset & Confusion Matrix", fontsize=20, halign=:center, valign=:top)
	colors = Makie.Colors.distinguishable_colors(n_clusters + 1)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))
	grid_1 = GridLayout(fig[1, 1]; nrow=2, ncol=1)
	grid_2 = GridLayout(fig[1, 2]; nrow=2, ncol=1)
	grid_3 = GridLayout(fig[1, 3]; nrow=2, ncol=1)

	# Show data
	ax1 = Axis(grid_1[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=15)
	hm = heatmap!(ax1, permutedims(gt_data); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(grid_1[2,1], hm, tellwidth=false, vertical=false)

	# Show cluster map
	ax2 = Axis(grid_2[1,1]; aspect=DataAspect(), yreversed=true, title="TSC Results - $Location", titlesize=15)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= D_relabel
	hm = heatmap!(ax2, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, 9))
	Colorbar(grid_2[2,1], hm, tellwidth=false, vertical=false)

	#Show Confusion Matrix
	ax3 = Axis(grid_3[1, 1]; aspect=DataAspect(), yreversed=true, xlabel = "Predicted Labels", ylabel = "True Labels", xticks = 1:predicted_labels_re, yticks = 1:true_labels_re, title="TSC - $Location - Confusion Matrix")
	hm = heatmap!(ax3, permutedims(confusion_matrix_re), colormap=:viridis)
	pm = permutedims(confusion_matrix_re)

	for i in 1:true_labels_re, j in 1:predicted_labels_re
        value = round(pm[i, j], digits=1)
        text!(ax3, i - 0.02, j - 0.1, text = "$value", color=:black, align = (:center, :center), fontsize=10)
    end
	Colorbar(grid_3[2, 1], hm, tellwidth=false, vertical=false)
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
# ╠═ba3ecda0-aa6b-46c3-80f4-eb8ab9a8a4bd
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
# ╟─56ae6711-4645-4981-81a5-4ddc5c5fc521
# ╟─29242190-f547-4fe5-abcd-2dac57200163
# ╟─c9697307-e424-42e3-9e61-100e9d2eb61b
# ╠═967709d5-ac33-4ec0-b2d2-c50d72dc8017
# ╟─4ce6a047-e4f8-41a2-9c4a-94643e8ce737
# ╠═efb0c7d5-28f3-44cc-a521-3bb1464da64a
# ╠═033d4da1-eba3-4c59-b945-2d59b81906e6
# ╟─8b2fa6e6-04ac-4b7b-a29c-f1c38dbcfdb1
# ╠═0fa5a2ea-1a4d-4ed5-9ae1-a0a3280d0bfe
# ╠═dae400bf-ea56-4ae8-9f0c-56ea45ef2652
# ╟─2d467a37-fcf8-4b63-8604-64216f64cb57
# ╠═f38c1c54-4e37-4c75-a67d-1b4378de0227
# ╠═ac511c25-b9bd-40a9-b03d-6032faac6781
# ╠═a8f26017-5e2a-4991-a840-68495ccece02
# ╠═4d85331b-65bb-46fa-8a8f-3634ee7eacc9
# ╠═d4cb1d09-890a-4c8a-a5d0-ddc8f6d166eb
# ╠═5414cbcb-c322-4536-b9c3-edd9fca2ca11
# ╟─6b6a0558-ada2-4193-a5f3-a4878421f914
# ╠═d21295be-cbd5-4135-b2e3-7925ad133581
# ╠═db8c4072-c07b-41b5-8baf-237665a85bfd
# ╠═aaf81673-a329-406e-abd0-2041a6875355
# ╠═0a6643cf-fd62-43e3-a427-e3b4b435d047
# ╠═bb45da10-c4c3-42c9-99f7-977cd13bcfab
# ╠═17bd5e56-1fad-42ac-8d5d-bc6ccc26a9a3
# ╟─9c2d52df-5704-4542-b237-0ed299bf51f6
# ╠═6859158c-1078-4c1a-8cd0-28e31329cf46
# ╠═62e7fc9b-180a-44d1-bc77-fbd48b49d58b
