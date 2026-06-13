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

# ╔═╡ 28a39b59-9ff1-46ab-b33e-61a16af00d6a
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ dc9ce868-3aa0-41c2-b13b-0cc3335d0826
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ProgressLogging, Dates, Logging, MAT

# ╔═╡ c20b3f4a-113e-4624-bc21-303dee03f4c8
begin
    include(joinpath(@__DIR__, "..", "KSS_Julia", "kss.jl"))
    using .KSS
end

# ╔═╡ b006a964-d069-4ef1-b60b-2e92aa852f4b
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

# ╔═╡ 396abda9-0d09-41f9-9bf5-19ad539b5ad5
md"""
### This Notebook demonstrates the implementation of Clustering Algorithms on different hyperspectral images
"""

# ╔═╡ e34d314d-9a57-4baf-b0ee-752c48dffcdc
md"""
#### Project Setup

> We start by activating the project environment and importing the required Julia packages.
"""

# ╔═╡ df197226-9df0-426a-957e-9c79c35c123a
data_dir = abspath(joinpath(@__DIR__, "..", "CZ_hsdb"))

# ╔═╡ dde1c83f-edb9-4772-9314-9ecca8929b8e
mat_files = sort(glob("*.mat", data_dir))

# ╔═╡ c15abf1b-47fd-47bb-ad0d-108a3a312445
file_names = basename.(mat_files)

# ╔═╡ 4be3c9cb-0762-44cb-b91c-efe49700f461
@bind selected_file Select(file_names)

# ╔═╡ 39690c33-8429-4d24-a116-63a9b5fc2a76
filepath = joinpath(data_dir, selected_file)

# ╔═╡ 41cf517a-5613-4198-b751-77018d49a98a
vars = matread(filepath)

# ╔═╡ 1c2e6823-3860-4f1e-976d-88a6f8e32629
data = vars["ref"]

# ╔═╡ e84b4b6d-e1c2-45d0-8a8b-99a988df2fe8
gt_data = vars["lbl"]

# ╔═╡ dbffc8b4-ae0c-47e7-9199-7ab98596e487
gt_labels = sort(unique(gt_data))

# ╔═╡ d80527c8-d9b0-469f-8f04-970e0b0b6fb2
bg_indices = findall(gt_data .== 0)

# ╔═╡ b3a60ac8-87e2-42e5-ba2d-d7246df2899c
md"""
### Define mask to remove the background pixels, i.e., pixels labeled zero
"""

# ╔═╡ 19fbb4cc-f7d3-4986-bcba-2f536310da34
begin
	mask = trues(size(data, 1), size(data, 2))
	for idx in bg_indices
		x, y = Tuple(idx)
		mask[x, y] = false
	end
end

# ╔═╡ 60983ea0-b10b-4d61-8768-e3dd330225de
md"""
#### Slider to choose the band of the image
"""

# ╔═╡ aa056475-bc60-4977-b486-516286845466
@bind band PlutoUI.Slider(1:size(data, 3), show_value=true)

# ╔═╡ 7148dde0-7893-4286-beca-eb6f22ebd6ba
with_theme() do
	fig = Figure(; size=(750, 600))
	# labels = length(unique(gt_data))
	# colors = Makie.Colors.distinguishable_colors(n_clusters+1)
	ax = Axis(fig[1, 1], aspect=DataAspect(), title ="Image, Band - $band", yreversed=true)
	ax1 = Axis(fig[1, 2], aspect=DataAspect(), title ="Masked Image", yreversed=true)
	image!(ax, permutedims(data[:, :, band]))
	H, W, B = size(data)
	clustermap = zeros(eltype(data), H, W, B)
	clustermap[mask, :] .= data[mask, :]
	hm = heatmap!(ax1, permutedims(clustermap[:, :, band]))
	Colorbar(fig[2,2], hm, tellwidth=false, vertical=false)
	fig
end

# ╔═╡ 82f2a6e3-52cd-4c0f-b455-960bc560b802
@bind n_clusters PlutoUI.Slider(1:7; default=4, show_value=true)

# ╔═╡ c3c73321-4e8f-42a7-97cc-a7fffff22f84
model = fit(data[mask, :]', fill(1, n_clusters))

# ╔═╡ bf06f85d-f10d-4ad6-8818-17365f7f3801
assignments = model.c

# ╔═╡ 2859470e-1446-45ac-a377-cafa6f7a6a17
unique(assignments)

# ╔═╡ 8ce185c4-2ac7-4602-9a9b-8b3f1a29dc6d
with_theme() do

	# Create figure
	fig = Figure(; size=(1200, 450))
	colors = Makie.Colors.distinguishable_colors(4)
	# colors_re = Makie.Colors.distinguishable_colors(length(re_labels))

	# subgrid = fig[1, 1] = GridLayout()

	# Show data
	ax = Axis(fig[1,1]; aspect=DataAspect(), yreversed=true, title="Ground Truth", titlesize=20)
	
	image!(ax, permutedims(data[:, :, 22]))
	# Colorbar(fig[2,1], hm1, tellwidth=false, vertical=false)

	# Show cluster map
	ax1 = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="KSS Clustering Results", titlesize=20)
	clustermap = fill(0, size(data)[1:2])
	clustermap[mask] .= assignments
	hm2 = heatmap!(ax1, permutedims(clustermap); colormap=Makie.Categorical(colors), colorrange=(0, 3))
	Colorbar(fig[2,2], hm2, tellwidth=false, vertical=false)
	
	fig
end

# ╔═╡ fed5f3b5-d22e-4b8c-8ae9-ad58d2bc940e
colors = Makie.Colors.distinguishable_colors(4)

# ╔═╡ Cell order:
# ╟─b006a964-d069-4ef1-b60b-2e92aa852f4b
# ╟─396abda9-0d09-41f9-9bf5-19ad539b5ad5
# ╟─e34d314d-9a57-4baf-b0ee-752c48dffcdc
# ╠═28a39b59-9ff1-46ab-b33e-61a16af00d6a
# ╠═dc9ce868-3aa0-41c2-b13b-0cc3335d0826
# ╠═df197226-9df0-426a-957e-9c79c35c123a
# ╠═dde1c83f-edb9-4772-9314-9ecca8929b8e
# ╠═c15abf1b-47fd-47bb-ad0d-108a3a312445
# ╠═4be3c9cb-0762-44cb-b91c-efe49700f461
# ╠═39690c33-8429-4d24-a116-63a9b5fc2a76
# ╠═41cf517a-5613-4198-b751-77018d49a98a
# ╠═1c2e6823-3860-4f1e-976d-88a6f8e32629
# ╠═e84b4b6d-e1c2-45d0-8a8b-99a988df2fe8
# ╠═dbffc8b4-ae0c-47e7-9199-7ab98596e487
# ╠═d80527c8-d9b0-469f-8f04-970e0b0b6fb2
# ╟─b3a60ac8-87e2-42e5-ba2d-d7246df2899c
# ╠═19fbb4cc-f7d3-4986-bcba-2f536310da34
# ╟─60983ea0-b10b-4d61-8768-e3dd330225de
# ╠═aa056475-bc60-4977-b486-516286845466
# ╠═7148dde0-7893-4286-beca-eb6f22ebd6ba
# ╠═c20b3f4a-113e-4624-bc21-303dee03f4c8
# ╠═82f2a6e3-52cd-4c0f-b455-960bc560b802
# ╠═c3c73321-4e8f-42a7-97cc-a7fffff22f84
# ╠═bf06f85d-f10d-4ad6-8818-17365f7f3801
# ╠═2859470e-1446-45ac-a377-cafa6f7a6a17
# ╠═8ce185c4-2ac7-4602-9a9b-8b3f1a29dc6d
# ╠═fed5f3b5-d22e-4b8c-8ae9-ad58d2bc940e
