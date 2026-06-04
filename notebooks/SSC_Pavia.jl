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

# ╔═╡ b72d8930-62dc-49bc-be81-9119d0e53598
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ 0cf92e28-a2d7-4c27-a870-1dfbe39564b6
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 670ca92c-4f16-48c4-97b4-969e77517fc7
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

# ╔═╡ 20c45faa-ea8e-4bcb-b644-dfa415fc3aca
md"""
### This Notebook demonstrates the implementation of Sparse Subspace Clustering (SSC) Clustering Algorithm on Pavia Dataset
"""

# ╔═╡ 1609c6c4-7e97-4e53-bb90-33da3163cc0e
md"""
#### Project Setup

> We start by activating the project environment and importing the required Julia packages.
"""

# ╔═╡ 7ea5e9b1-a1fa-4e19-be45-ae785adbff3e
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ b2bcba95-51de-4192-bbb7-8cb0e630240c
filepath = abspath(joinpath(@__DIR__, "..", "MAT Files", "$Location.mat"))

# ╔═╡ b06025b5-947f-4d69-9b15-95c84b9d02f3
gt_filepath = abspath(joinpath(@__DIR__, "..", "GT Files", "$Location.mat"))

# ╔═╡ da2e5198-17c0-4aee-aae5-c5ebe87009dd
CACHEDIR = abspath(joinpath(@__DIR__, "..", "cache_files", "Aerial Datasets"))

# ╔═╡ 217dd71e-9e43-4697-9d32-74472c06893e
md"""
#### Defined cachet function to cache runs
"""

# ╔═╡ da933dbc-8386-40a5-9b78-91a6c3aea62f
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ bbfd82d7-de7c-42fd-804c-475c2484f002
vars = matread(filepath)

# ╔═╡ edb208a7-a07e-4c05-8a72-69dc30d1670c
vars_gt = matread(gt_filepath)

# ╔═╡ b4810770-a369-4cc1-b721-a4b777a4e2a7
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 080e4a8b-0b88-41a6-b5dd-82d2cb19f617
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 8e1c6c99-03e0-4771-a73a-5751372c4177
md"""
### Hyperspectral Image Cube - $Location
"""

# ╔═╡ 9d0df7b8-5f63-4175-89a6-c7322e524e87
data = vars[data_key]

# ╔═╡ 5b33cb7e-af24-46d1-9162-7db0c792bb13
md"""
### Ground Truth Data
"""

# ╔═╡ 0f83d4d0-03c9-42ad-bbc8-b76fe44d212c
gt_data = vars_gt[gt_key]

# ╔═╡ 7ca305f2-b9d1-4d14-8ce1-4c46ae8d44e3
md"""
### Ground Truth Labels
"""

# ╔═╡ 6f844aa8-7e03-46cb-8372-803ee93d79fa
gt_labels = sort(unique(gt_data))

# ╔═╡ b03bbc39-fe20-43b3-85d7-a5efc9421fa3
bg_indices = findall(gt_data .== 0)

# ╔═╡ e869fe28-e8b2-4367-8263-70db7dff73b1
md"""
### Define mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ bfe3ad3f-f76a-4008-8892-e99f298826d9
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 45171f79-0956-4fc5-b7f3-2b7d2d29d74f
md"""
##### Number of clusters equivalent to the number of unique labels from the ground truth data
"""

# ╔═╡ 1a4760a7-a508-4501-b1e7-d59142beeec0
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ 2848bea9-1835-478a-b9c4-4858fd6e1ce6
md"""
#### Slider to choose the band of the image
"""

# ╔═╡ 604c3923-5b29-4119-941b-27c8920f9e0a
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ f535b2b6-e953-462c-9597-8df39a495d5e
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

# ╔═╡ 85248f4d-a80a-4fd6-a16b-5bc4c8c78f06
function ssc(X)
end

# ╔═╡ Cell order:
# ╟─670ca92c-4f16-48c4-97b4-969e77517fc7
# ╟─20c45faa-ea8e-4bcb-b644-dfa415fc3aca
# ╟─1609c6c4-7e97-4e53-bb90-33da3163cc0e
# ╠═b72d8930-62dc-49bc-be81-9119d0e53598
# ╠═0cf92e28-a2d7-4c27-a870-1dfbe39564b6
# ╠═7ea5e9b1-a1fa-4e19-be45-ae785adbff3e
# ╠═b2bcba95-51de-4192-bbb7-8cb0e630240c
# ╠═b06025b5-947f-4d69-9b15-95c84b9d02f3
# ╠═da2e5198-17c0-4aee-aae5-c5ebe87009dd
# ╟─217dd71e-9e43-4697-9d32-74472c06893e
# ╠═da933dbc-8386-40a5-9b78-91a6c3aea62f
# ╠═bbfd82d7-de7c-42fd-804c-475c2484f002
# ╠═edb208a7-a07e-4c05-8a72-69dc30d1670c
# ╠═b4810770-a369-4cc1-b721-a4b777a4e2a7
# ╠═080e4a8b-0b88-41a6-b5dd-82d2cb19f617
# ╟─8e1c6c99-03e0-4771-a73a-5751372c4177
# ╠═9d0df7b8-5f63-4175-89a6-c7322e524e87
# ╟─5b33cb7e-af24-46d1-9162-7db0c792bb13
# ╠═0f83d4d0-03c9-42ad-bbc8-b76fe44d212c
# ╠═7ca305f2-b9d1-4d14-8ce1-4c46ae8d44e3
# ╠═6f844aa8-7e03-46cb-8372-803ee93d79fa
# ╠═b03bbc39-fe20-43b3-85d7-a5efc9421fa3
# ╠═e869fe28-e8b2-4367-8263-70db7dff73b1
# ╠═bfe3ad3f-f76a-4008-8892-e99f298826d9
# ╟─45171f79-0956-4fc5-b7f3-2b7d2d29d74f
# ╠═1a4760a7-a508-4501-b1e7-d59142beeec0
# ╟─2848bea9-1835-478a-b9c4-4858fd6e1ce6
# ╠═604c3923-5b29-4119-941b-27c8920f9e0a
# ╠═f535b2b6-e953-462c-9597-8df39a495d5e
# ╠═85248f4d-a80a-4fd6-a16b-5bc4c8c78f06
