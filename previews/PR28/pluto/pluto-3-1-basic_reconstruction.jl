### A Pluto.jl notebook ###
# v0.20.28

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

# ╔═╡ ce857766-c939-11f1-a743-c7bd01aef4da
begin
	# Layout
	using PlutoUI
	using PlutoUI: Slider   # GLMakie also exports a `Slider`
	using PlutoTeachingTools
	using HypertextLiteral

	# Plot
	using GLMakie

	# Reconstruction
	using MAT
	using FFTW
	using NFFT
	using NFFTTools
	using LinearAlgebra
	using Statistics
	using Random

	PlutoUI.TableOfContents()
end

# ╔═╡ ce848b7e-c939-11f1-b0e6-0b6babc5505a
ChooseDisplayMode()

# ╔═╡ ce857734-c939-11f1-a910-9de016fca185
html"""
	<h1 style="text-align:center">
		Graduate Program - Biomedical MRI
	</h1>
	<div style="text-align:center">
		<p style="font-weight:bold; font-size: 35px; font-variant: small-caps; margin: 0px">
			Lesson 3-1: Basic image reconstruction
		</p>
		<p style="font-size: 30px; font-variant: small-caps; margin: 0px">
			Aurélien Trotier
		</p>
		<p style="font-size: 20px;">
			CRMSB - Université de Bordeaux / CNRS
		</p>
	</div>
"""

# ╔═╡ ce857782-c939-11f1-b207-a90fed7c07e9
md"""
# 1. Introduction

During an MRI acquisition, the gradients encode the position of the spins into the **phase** of the signal. The signal received at time $t$ is:

$$s(t) = \int m(\mathbf{r})\, e^{-i2\pi\, \mathbf{k}(t)\cdot\mathbf{r}}\, d\mathbf{r}
\qquad \text{with} \qquad
\mathbf{k}(t) = \frac{\gamma}{2\pi}\int_0^t \mathbf{G}(\tau)\, d\tau$$

The signal is the **Fourier transform** of the transverse magnetization $m(\mathbf{r})$, evaluated at the position $\mathbf{k}(t)$. The space in which the data are acquired is called the **k-space**. Reconstructing an image therefore means computing an **inverse Fourier transform**.

When the k-space is sampled on a Cartesian grid ($N_x \times N_y$ points separated by $\Delta k_x, \Delta k_y$), the inverse Fourier transform becomes an inverse **discrete** Fourier transform, computed with the **FFT**. When it is not (radial, spiral...), we need a **non-uniform FFT (NUFFT)**.

In this practical we only use low-level operations (`fft`, `ifft`, `nfft`) to understand:

1. how the information is distributed in k-space (center *vs* periphery),
2. what sets the **spatial resolution**,
3. what **zero-filling** does (and does not do),
4. what sets the **field of view (FOV)** and where the **fold-over artefact** comes from,
5. how the **point spread function (PSF)** summarizes all these effects,
6. how to reconstruct **non-Cartesian** data with a NUFFT.

Parallel imaging (GRAPPA, SENSE), iterative reconstruction and compressed sensing are studied in the *advanced reconstruction* practical.
"""

# ╔═╡ ce85778e-c939-11f1-a186-e98bf7b3f0a9
md"""
# 2. The data

The dataset is a 2D axial brain slice acquired with an **8-channel head coil**. The k-space is **fully sampled** on a Cartesian grid. The file contains:

- `kspace_8ch`: the complex k-space of each coil, size $N_x \times N_y \times N_c$
- `map_8ch`: the sensitivity map of each coil (same size)

!!! warning "Path to the data"
    The file `kspace_brain_8ch.mat` is expected in `ressources/data/`. Change `data_path` below if you stored it somewhere else.
"""

# ╔═╡ ce8577a2-c939-11f1-a52f-4f78ed5cd323
data_path = joinpath(@__DIR__, "..", "ressources", "data", "kspace_brain_8ch.mat")

# ╔═╡ ce8577ac-c939-11f1-9e03-513f5cbe81e4
begin
	vars = matread(data_path)
	kspace_8ch = vars["kspace_8ch"]
	maps_8ch = vars["map_8ch"]
	Nx, Ny, Nc = size(kspace_8ch)
end

# ╔═╡ ce8577b6-c939-11f1-8693-11da8d5aab7f
md"""
The original header of the acquisition is not available. We will **assume an in-plane pixel size of 1 mm × 1 mm**, which gives the following field of view (in mm):
"""

# ╔═╡ ce8577ca-c939-11f1-8929-fb5c8d4b616a
begin
	pix = (1.0, 1.0)          # pixel size (mm) along x and y
	FOV = (Nx, Ny) .* pix     # field of view (mm)
end

# ╔═╡ ce8577d4-c939-11f1-ae8b-c7f121718b79
md"""
## 2.1 Centered FFT

The FFT algorithm assumes that the **first** element of the array is the zero frequency ($k=0$). In MRI, the k-space is stored with $k=0$ in the **center** of the matrix (at index $N/2+1$), and we also want the center of the image ($x = 0$) in the center of the matrix.

We therefore define a **centered** inverse FFT (k-space → image) and a centered FFT (image → k-space), applied along the first two dimensions only (the 3rd one is the coil dimension):
"""

# ╔═╡ ce8577de-c939-11f1-a210-174d90077116
begin
	"Centered 2D inverse FFT along dimensions 1 and 2 : k-space → image"
	ifft2c(k) = fftshift(ifft(ifftshift(k, (1, 2)), (1, 2)), (1, 2))

	"Centered 2D FFT along dimensions 1 and 2 : image → k-space"
	fft2c(x) = fftshift(fft(ifftshift(x, (1, 2)), (1, 2)), (1, 2))
end

# ╔═╡ ce857810-c939-11f1-8cab-7b3e76ed2de2
img_8ch = ifft2c(kspace_8ch);

# ╔═╡ ce857826-c939-11f1-8d67-917d7dacd765
md"""
Coil : $(@bind coil Slider(1:Nc, default=1, show_value=true))
"""

# ╔═╡ ce857842-c939-11f1-805e-112a1a7dbcdc
md"""
## 2.2 Coil combination

To simplify the rest of the practical, we combine the 8 coils into a **single complex image**. The two classical methods are:

- **Sum of squares (SoS)**: $\quad m_{SoS} = \sqrt{\sum_l |m_l|^2}$ — simple, but the phase is lost.
- **Sensitivity-weighted combination** (Roemer): knowing the sensitivities $C_l$, the least-squares estimate is

$$\hat{m}(x,y) = \frac{\sum_l C_l^*(x,y)\, m_l(x,y)}{\sum_l |C_l(x,y)|^2}$$

which keeps the phase. We use it to build a single complex image `img` and its k-space `ksp = fft2c(img)`: **`ksp` is the k-space of a virtual single coil with a uniform sensitivity, used in all the following sections.**
"""

# ╔═╡ ce857858-c939-11f1-b49f-e5f2d5617604
begin
	img_sos = sqrt.(sum(abs2.(img_8ch), dims=3))[:, :, 1]
	img = sum(conj.(maps_8ch) .* img_8ch, dims=3)[:, :, 1] ./ (sum(abs2.(maps_8ch), dims=3)[:, :, 1] .+ eps())
	ksp = fft2c(img)
end;

# ╔═╡ ce85787e-c939-11f1-9ec4-e9effdd86dbf
md"""
# 3. What does a point of the k-space represent?

The inverse Fourier transform writes the image as a **sum of complex exponentials** (plane waves):

$$m(x,y) = \sum_{k_x}\sum_{k_y} K(k_x,k_y)\, e^{i2\pi(k_x x + k_y y)}$$

Each k-space sample is the **weight** of one plane wave. Let's look at the plane wave associated with a single point: we create a k-space full of zeros with only one non-zero sample and reconstruct it.

Position of the point, relative to the k-space center (in number of samples):
"""

# ╔═╡ ce857886-c939-11f1-ae95-1d3d04969db2
md"""
``i_{k_x}`` : $(@bind ikx Slider(-20:20, default=4, show_value=true))

``i_{k_y}`` : $(@bind iky Slider(-20:20, default=0, show_value=true))

phase of the sample (rad) : $(@bind φ_point Slider(0:π/8:2π, default=0, show_value=true))
"""

# ╔═╡ ce85789c-c939-11f1-8812-c1c7e9b81f3e
begin
	K_point = zeros(ComplexF64, Nx, Ny)
	K_point[Nx÷2+1+ikx, Ny÷2+1+iky] = cis(φ_point)
	img_point = ifft2c(K_point)
end;

# ╔═╡ ce8578ce-c939-11f1-bc56-033bb57825ff
md"""
## 3.1 Center *vs* periphery of the k-space

The image is the sum of all these plane waves. Let's reconstruct the image using **only the center**, or **only the periphery**, of the k-space. The mask is an ellipse whose radius is given as a fraction of $k_{max}$.
"""

# ╔═╡ ce8578d8-c939-11f1-bb73-ad8dd5b1409b
md"""
Part of the k-space kept : $(@bind center_mode Select(["center" => "center only", "periphery" => "periphery only"]))

Radius (fraction of ``k_{max}``) : $(@bind r_center Slider(0.02:0.02:1.0, default=0.1, show_value=true))
"""

# ╔═╡ ce8578e2-c939-11f1-9717-b9a7a2992306
begin
	kx_norm = ((1:Nx) .- (Nx ÷ 2 + 1)) ./ (Nx / 2)   # k_x / k_max ∈ [-1, 1)
	ky_norm = ((1:Ny) .- (Ny ÷ 2 + 1)) ./ (Ny / 2)
	mask_center = (kx_norm .^ 2 .+ (ky_norm .^ 2)') .<= r_center^2
	mask_cp = center_mode == "center" ? mask_center : .!mask_center

	ksp_cp = ksp .* mask_cp
	img_cp = ifft2c(ksp_cp)
	energy_cp = sum(abs2, ksp_cp) / sum(abs2, ksp)
end;

# ╔═╡ ce857914-c939-11f1-ac3e-09bdb3a2341f
md"""
# 4. Spatial resolution

The finest detail that can be represented is set by the **highest spatial frequency acquired**, $k_{max}$. To simulate an acquisition at lower resolution, we keep only the central $n_x \times n_y$ samples of the k-space (same $\Delta k$, so same FOV) and reconstruct them with an `ifft` of size $n_x \times n_y$.
"""

# ╔═╡ ce85791e-c939-11f1-8191-43781599eea0
begin
	"Extract the central `n[1] × n[2]` samples of a centered 2D k-space"
	function crop_center(K, n)
		N = size(K)
		idx = ntuple(d -> (N[d] ÷ 2 + 1 - n[d] ÷ 2) .+ (0:n[d]-1), 2)
		return K[idx...]
	end

	"Place a centered 2D k-space in the middle of a `n[1] × n[2]` matrix of zeros"
	function zero_pad(K, n)
		N = size(K)
		Kp = zeros(eltype(K), n)
		idx = ntuple(d -> (n[d] ÷ 2 + 1 - N[d] ÷ 2) .+ (0:N[d]-1), 2)
		Kp[idx...] .= K
		return Kp
	end
end;

# ╔═╡ ce857932-c939-11f1-a5eb-d7d23aae8f03
md"""
``n_x`` : $(@bind nx_res Slider(2:2:Nx, default=Nx÷2, show_value=true))

``n_y`` : $(@bind ny_res Slider(2:2:Ny, default=Ny÷2, show_value=true))
"""

# ╔═╡ ce85793c-c939-11f1-a977-c97d73a69d1d
begin
	ksp_low = crop_center(ksp, (nx_res, ny_res))
	# ifft divides by n_x n_y instead of N_x N_y : rescale to keep the same intensity
	img_low = ifft2c(ksp_low) .* (nx_res * ny_res) / (Nx * Ny)
	pix_low = FOV ./ (nx_res, ny_res)
end;

# ╔═╡ ce857978-c939-11f1-adf3-c3c0894ef2f2
md"""
# 5. Zero-filling

Zero-filling consists in **adding zeros around the acquired k-space** to increase the matrix size before the `ifft`. It is used by default on most scanners ("interpolation" option).

Here we "acquire" a low-resolution k-space and zero-fill it by an integer factor.
"""

# ╔═╡ ce8579a0-c939-11f1-be9c-87d81721c12a
md"""
Acquired matrix : $(@bind zf_acq Select([4 => "1/4 of the reference matrix", 2 => "1/2 of the reference matrix"]))

Zero-filling factor : $(@bind zf_factor Select([1, 2, 4], default=2))
"""

# ╔═╡ ce8579aa-c939-11f1-9c19-758b9191c856
begin
	n_acq = (Nx ÷ zf_acq, Ny ÷ zf_acq)
	n_zf = n_acq .* zf_factor

	ksp_acq = crop_center(ksp, n_acq)
	img_acq = ifft2c(ksp_acq) .* prod(n_acq) / (Nx * Ny)

	ksp_zf = zero_pad(ksp_acq, n_zf)
	img_zf = ifft2c(ksp_zf) .* prod(n_zf) / (Nx * Ny)
end;

# ╔═╡ ce8579dc-c939-11f1-9f54-63c420564585
md"""
# 6. Field of view and fold-over artefact

## 6.1 Skipping k-space lines

The FOV is set by the spacing between two k-space samples:

$$FOV = \frac{1}{\Delta k}$$

A common way to accelerate an acquisition is to acquire only **one line out of $R$**. The spacing becomes $R\,\Delta k$ and the FOV is divided by $R$. What happens to the part of the object which is outside the new FOV?
"""

# ╔═╡ ce8579e6-c939-11f1-8fac-a5d9fb66055e
md"""
Acceleration factor ``R`` : $(@bind R_us Select([1, 2, 4, 5], default=2))

Undersampled direction : $(@bind dim_us Select([2 => "y (anterior-posterior)", 1 => "x (left-right)"]))
"""

# ╔═╡ ce8579ee-c939-11f1-a45f-5f9b1f3a7fd2
begin
	N_us = size(ksp, dim_us)
	c_us = N_us ÷ 2 + 1
	lines_us = findall(i -> mod(i - c_us, R_us) == 0, 1:N_us)  # acquired lines (always including k = 0)

	# (a) what the scanner does : only the acquired lines → smaller matrix, FOV / R
	ksp_us = collect(selectdim(ksp, dim_us, lines_us))
	img_us = ifft2c(ksp_us)
	FOV_us = dim_us == 1 ? (FOV[1] / R_us, FOV[2]) : (FOV[1], FOV[2] / R_us)

	# (b) the missing lines are replaced by zeros → full matrix
	mask_us = zeros(size(ksp))
	selectdim(mask_us, dim_us, lines_us) .= 1
	img_us_zero = ifft2c(ksp .* mask_us)
end;

# ╔═╡ ce857a18-c939-11f1-a74c-9b44cd187e00
"""
    fold(m, R, dim)

Fold the image `m` along the dimension `dim` with an acceleration factor `R`:
sum of the `R` copies of the image shifted by `FOV/R`, restricted to the central `N/R` pixels.
"""
function fold(m, R, dim)
	N = size(m, dim)
	n = N ÷ R
	shift(s) = ntuple(d -> d == dim ? s * n : 0, ndims(m))
	m_sum = sum(circshift(m, shift(s)) for s in 0:R-1)
	c = N ÷ 2 + 1
	return collect(selectdim(m_sum, dim, (c - n ÷ 2) .+ (0:n-1)))
end

# ╔═╡ ce857a20-c939-11f1-81f3-399e2f152067
md"""
Maximum difference between `fold(img, R, dim)` and the image (a) : **$(maximum(abs.(fold(img, R_us, dim_us) .- img_us)))** (to compare with ``\max|m|`` = $(round(maximum(abs, img), sigdigits=3)))
"""

# ╔═╡ ce857a4a-c939-11f1-bc45-4db948920f84
md"""
## 6.2 Any FOV with a NUFFT

With the Cartesian data we can only divide the FOV by an integer. To simulate an acquisition with **any** FOV and **any** resolution, we can compute the Fourier transform of the reference image at arbitrary k-space positions with a **non-uniform FFT** (`NFFT.jl`):

$$K(\mathbf{k}_j) = \sum_{\mathbf{r}} m(\mathbf{r})\, e^{-i2\pi \mathbf{k}_j\cdot\mathbf{r}}$$

`plan_nfft(nodes, N)` takes the positions $\mathbf{k}_j$ in a $2 \times M$ matrix, in **cycles per pixel** of the reference image (range $[-0.5, 0.5)$). `p * x` is the forward transform and `adjoint(p) * y` the adjoint one.

We sample a Cartesian grid with $\Delta k = 1/FOV$ and $k_{max} = 1/(2\Delta x)$, then reconstruct it with a simple `ifft`.
"""

# ╔═╡ ce857a5e-c939-11f1-9d15-3d6253c39ecc
md"""
``FOV_x`` (mm) : $(@bind FOVx_sim Slider(60:5:250, default=120, show_value=true))

``FOV_y`` (mm) : $(@bind FOVy_sim Slider(60:5:300, default=220, show_value=true))

resolution ``\Delta x = \Delta y`` (mm) : $(@bind res_sim Select([1.0, 1.5, 2.0, 3.0]))
"""

# ╔═╡ ce857a68-c939-11f1-a5dd-ed94bb1d9e87
begin
	n_sim = 2 .* floor.(Int, (FOVx_sim, FOVy_sim) ./ (2 * res_sim))   # matrix size

	# Cartesian grid with Δk = 1/FOV (mm⁻¹), converted in cycles / reference pixel
	kx_sim = ((0:n_sim[1]-1) .- n_sim[1] ÷ 2) ./ FOVx_sim .* pix[1]
	ky_sim = ((0:n_sim[2]-1) .- n_sim[2] ÷ 2) ./ FOVy_sim .* pix[2]
	nodes_sim = permutedims(hcat(vec(kx_sim .* ones(n_sim[2])'), vec(ones(n_sim[1]) .* ky_sim')))

	p_sim = plan_nfft(nodes_sim, size(img))
	ksp_sim = reshape(p_sim * img, n_sim)   # "acquisition"
	img_sim = ifft2c(ksp_sim) .* prod(n_sim) .* prod(pix) ./ (FOVx_sim * FOVy_sim)
end;

# ╔═╡ ce857a90-c939-11f1-9b09-23bfc459b72f
md"""
# 7. Point Spread Function (PSF)

In practice, we never acquire the full continuous k-space. The acquired data can be written as the "true" k-space $M(\mathbf{k})$ multiplied by a **sampling function** $S(\mathbf{k})$ (1 where a sample is acquired, 0 elsewhere, possibly with weights):

$$K_{acq}(\mathbf{k}) = S(\mathbf{k})\, M(\mathbf{k})$$

From the **convolution theorem**, the reconstructed image is the true image convolved with the inverse Fourier transform of the sampling function:

$$m_{acq}(\mathbf{r}) = \left(PSF \circledast m\right)(\mathbf{r}) \qquad \text{with} \qquad PSF = \mathcal{F}^{-1}\{S\}$$

The **PSF** is the image of a single point: it summarizes the effects of the sampling on the image (resolution, ringing, fold-over...).
"""

# ╔═╡ ce857a9a-c939-11f1-80ab-c17e89ae7479
md"""
Sampling pattern : $(@bind psf_pattern Select(["full" => "full k-space", "crop" => "central crop (low resolution)", "hann" => "central crop + Hann filter", "regular" => "regular undersampling along y", "random" => "random undersampling along y"], default="crop"))

Crop fraction of ``k_{max}`` (crop / hann) : $(@bind psf_frac Slider(0.1:0.05:1.0, default=0.5, show_value=true))

Acceleration ``R`` (regular / random) : $(@bind psf_R Slider(1:6, default=2, show_value=true))
"""

# ╔═╡ ce857aa4-c939-11f1-8dfe-f97efd30ec03
"""
    sampling_mask(pattern, N; frac, R, ncenter=16, seed=1)

Sampling function `S(k)` of size `N` for the different sampling patterns.
"""
function sampling_mask(pattern, N; frac=0.5, R=2, ncenter=16, seed=1)
	kx = ((1:N[1]) .- (N[1] ÷ 2 + 1)) ./ (N[1] / 2)    # k / k_max
	ky = ((1:N[2]) .- (N[2] ÷ 2 + 1)) ./ (N[2] / 2)
	S = zeros(N)
	if pattern == "full"
		S .= 1
	elseif pattern == "crop"
		S .= (abs.(kx) .<= frac) .* (abs.(ky) .<= frac)'
	elseif pattern == "hann"
		hann = k -> abs(k) <= frac ? 0.5 + 0.5 * cos(π * k / frac) : 0.0
		S .= hann.(kx) .* hann.(ky)'
	elseif pattern == "regular"
		S[:, mod.((1:N[2]) .- (N[2] ÷ 2 + 1), R) .== 0] .= 1
	elseif pattern == "random"
		lines = rand(MersenneTwister(seed), N[2]) .< 1 / R
		lines[(N[2]÷2+1-ncenter÷2):(N[2]÷2+ncenter÷2)] .= true   # always acquire the center
		S[:, lines] .= 1
	end
	return S
end

# ╔═╡ ce857ab8-c939-11f1-8afe-bb076db1a2bd
begin
	S_psf = sampling_mask(psf_pattern, (Nx, Ny); frac=psf_frac, R=psf_R)

	# PSF on the image grid
	psf = ifft2c(S_psf)
	psf = psf ./ psf[Nx÷2+1, Ny÷2+1]

	# PSF on a finer grid (zero-filling ×8 of the sampling function !) to see its shape
	zf_psf = 8
	psf_fine = ifft2c(zero_pad(S_psf, zf_psf .* (Nx, Ny)))
	psf_fine = psf_fine ./ psf_fine[zf_psf*Nx÷2+1, zf_psf*Ny÷2+1]
	prof_x = psf_fine[:, zf_psf*Ny÷2+1]
	prof_y = psf_fine[zf_psf*Nx÷2+1, :]

	img_psf = ifft2c(S_psf .* ksp)
end;

# ╔═╡ ce857aea-c939-11f1-a6de-5fcf82392f04
let
	x0, y0 = Nx ÷ 2 + 1 + 20, Ny ÷ 2 + 1 + 30   # any pixel
	psf_raw = ifft2c(S_psf)                     # not normalized
	cx, cy = Nx ÷ 2 + 1, Ny ÷ 2 + 1             # position of r = 0 in the PSF
	acc = zero(ComplexF64)
	for x in 1:Nx, y in 1:Ny
		# PSF(r0 - r) with a circular (periodic) indexing
		acc += img[x, y] * psf_raw[mod(cx + x0 - x - 1, Nx) + 1, mod(cy + y0 - y - 1, Ny) + 1]
	end
	(direct_sum = acc, from_fft = img_psf[x0, y0])
end

# ╔═╡ ce857afe-c939-11f1-96f1-37d7a88bb764
md"""
## 7.2 A gallery of sampling patterns

Let's compare the PSF of the most common sampling patterns, all with an acceleration of about $R = 4$ (except the full k-space):

- **regular**: one line out of 4 (classical parallel imaging),
- **radial**: 64 spokes, reconstructed with the adjoint NUFFT + DCF (see section 8),
- **random**: samples drawn uniformly in the k-space,
- **random + center**: the same, with a fully sampled center,
- **Poisson disc + center**: samples separated by a *minimum distance* which increases with $|\mathbf{k}|$ (variable density), with a fully sampled center.

The last three patterns undersample **both** directions of the k-space: in practice they are used for the two phase-encoding directions $(k_y, k_z)$ of a **3D** acquisition (in 2D, only whole lines can be skipped).

To see the effect of the PSF clearly, we also use a simple phantom: a small bright square and a larger gray square.
"""

# ╔═╡ ce857b08-c939-11f1-ab72-e71f83fada7d
"""
    random_mask(N; R=4, ncenter=(0, 0), seed=1)

Random sampling mask of size `N` (probability `1/R` for each sample), with a fully sampled `ncenter` region.
"""
function random_mask(N; R=4, ncenter=(0, 0), seed=1)
	M = rand(MersenneTwister(seed), N...) .< 1 / R
	c = N .÷ 2 .+ 1
	M[c[1]-ncenter[1]÷2:c[1]+ncenter[1]÷2-1, c[2]-ncenter[2]÷2:c[2]+ncenter[2]÷2-1] .= true
	return M
end

# ╔═╡ ce857b12-c939-11f1-b39b-4d4677e9d605
"""
    poisson_mask(N; R=4, ncenter=(16, 16), α=2.0, seed=1)

Variable-density Poisson-disc sampling mask of size `N`. The samples are drawn in a random order
and accepted only if no other accepted sample is closer than `r(k) = r₀ (1 + α |k|)`.
The radius `r₀` is adjusted (bisection) to reach the acceleration factor `R`.
A region of `ncenter` samples is fully sampled.
"""
function poisson_mask(N; R=4, ncenter=(16, 16), α=2.0, seed=1)
	kx = ((1:N[1]) .- (N[1] ÷ 2 + 1)) ./ (N[1] / 2)
	ky = ((1:N[2]) .- (N[2] ÷ 2 + 1)) ./ (N[2] / 2)
	kr = sqrt.(kx .^ 2 .+ (ky .^ 2)') ./ sqrt(2)          # 0 at the center, 1 in the corners
	c = N .÷ 2 .+ 1

	function draw(r0)
		M = falses(N)
		w = ceil(Int, r0 * (1 + α))
		for li in randperm(MersenneTwister(seed), prod(N))
			i, j = Tuple(CartesianIndices(N)[li])
			r = r0 * (1 + α * kr[i, j])
			# reject the candidate if an accepted sample is closer than r
			ok = true
			for di in -w:w, dj in -w:w
				ii, jj = i + di, j + dj
				if 1 <= ii <= N[1] && 1 <= jj <= N[2] && M[ii, jj] && di^2 + dj^2 < r^2
					ok = false
					break
				end
			end
			ok && (M[i, j] = true)
		end
		M[c[1]-ncenter[1]÷2:c[1]+ncenter[1]÷2-1, c[2]-ncenter[2]÷2:c[2]+ncenter[2]÷2-1] .= true
		return M
	end

	lo, hi = 0.3, 10.0
	for _ in 1:20
		mid = (lo + hi) / 2
		prod(N) / count(draw(mid)) > R ? (hi = mid) : (lo = mid)
	end
	return draw((lo + hi) / 2)
end

# ╔═╡ ce857b30-c939-11f1-96ca-95af124a0140
begin
	# simple phantom : a small bright square and a larger gray square
	phantom = zeros(ComplexF64, Nx, Ny)
	phantom[25:32, 175:182] .= 1.0
	phantom[85:110, 70:95] .= 0.5
end;

# ╔═╡ ce857b88-c939-11f1-9f1d-890cd9c23883
md"""
# 8. Non-Cartesian imaging: radial sampling and NUFFT

In radial imaging, each readout ("spoke") is a line passing through the center of the k-space with a different angle $\theta$. The samples are **not on a Cartesian grid**, so the FFT cannot be used directly. The reconstruction uses the **adjoint NUFFT** (also called *gridding*):

$$\hat{m}(\mathbf{r}) = \sum_j w_j\, K(\mathbf{k}_j)\, e^{i2\pi\mathbf{k}_j\cdot\mathbf{r}}$$

where $w_j$ is the **density compensation function (DCF)**: the center of the k-space is sampled much more densely than the periphery, so each sample must be weighted by the area it represents. For radial sampling, $w_j \propto |\mathbf{k}_j|$ (the "ramp" filter).

We simulate the radial acquisition from the reference image with the forward NUFFT.
"""

# ╔═╡ ce857b9e-c939-11f1-8bb6-ef9d2571f3a0
begin
	"Radial trajectory : `n_spokes` spokes of `Nread` samples over 180°, in cycles/pixel (k_max = 0.5)"
	function radial_nodes(n_spokes, Nread)
		r = ((0:Nread-1) .- Nread ÷ 2) ./ Nread        # position along the spoke
		θ = π .* (0:n_spokes-1) ./ n_spokes             # spokes uniformly distributed over 180°
		return permutedims(hcat(vec(r .* cos.(θ)'), vec(r .* sin.(θ)')))
	end

	"Analytical DCF of the radial trajectory : area of the k-space represented by each sample (ramp |k|)"
	function ramp_dcf(n_spokes, Nread)
		r = ((0:Nread-1) .- Nread ÷ 2) ./ Nread
		# the center (r = 0) is shared by all the spokes : disk of radius Δr/2
		return vec(max.(abs.(r), 1 / (4Nread)) .* ones(n_spokes)') .* (π / (n_spokes * Nread))
	end
end

# ╔═╡ ce857ba8-c939-11f1-a470-b7dc508c51d0
md"""
Number of spokes : $(@bind n_spokes Slider(8:2:420, default=64, show_value=true))

Density compensation : $(@bind dcf_choice Select(["ramp" => "analytical ramp |k|", "sdc" => "numerical (NFFTTools.sdc)", "none" => "none"]))
"""

# ╔═╡ ce857bb2-c939-11f1-882f-47c6ef0e25cb
begin
	Nread = 256                                     # samples per spoke
	nodes_rad = radial_nodes(n_spokes, Nread)
	p_rad = plan_nfft(nodes_rad, size(img))
	data_rad = p_rad * img                          # radial "acquisition"

	w_ramp = ramp_dcf(n_spokes, Nread)              # analytical DCF
	w_sdc = sdc(p_rad, iters=10)                    # numerical DCF (section 8.1)
	w_sdc .*= sum(w_ramp) / sum(w_sdc)              # same normalization as the ramp
	w_none = fill(1 / length(data_rad), length(data_rad))
	w_used = Dict("ramp" => w_ramp, "sdc" => w_sdc, "none" => w_none)[dcf_choice]

	img_rad = adjoint(p_rad) * (w_used .* data_rad)
	psf_rad = adjoint(p_rad) * complex.(w_used)     # image of a point : all the samples = 1
	psf_rad = psf_rad ./ maximum(abs, psf_rad)
	n_nyquist = ceil(Int, π / 2 * Nread)
end;

# ╔═╡ ce857be4-c939-11f1-8740-db92fc4b4f98
md"""
## 8.1 Numerical density compensation

The ramp $|\mathbf{k}|$ is only valid for an ideal radial trajectory. For a general trajectory (spiral, variable-density, trajectory measured with gradient errors...) there is no simple formula and the DCF is computed **numerically**. `NFFTTools.jl` provides `sdc(p; iters)` which implements the iterative method of Pipe & Menon (MRM 1999). Starting from $w = 1$, the weights are updated as

$$w_j \leftarrow \frac{w_j}{\left(w \circledast \phi\right)(\mathbf{k}_j)}$$

where $w \circledast \phi$ is the sampling density estimated by convolving the weighted samples with the interpolation kernel $\phi$ of the NUFFT. At convergence, the weighted density is uniform. It only needs the trajectory (the NUFFT plan):

```julia
w_sdc = sdc(p_rad, iters=10)
```

The figure below compares the three options for the current number of spokes. The error is computed inside the head, after an optimal intensity scaling.
"""

# ╔═╡ ce857c2c-c939-11f1-afdd-75cc4547f687
md"""
# 9. Summary

| Parameter of the k-space | Effect on the image |
|:--|:--|
| Center of the k-space (low frequencies) | contrast, global shape, most of the energy |
| Periphery (high frequencies) | edges, fine details |
| Extent $k_{max}$ | spatial resolution $\Delta x = 1/(2k_{max}) = FOV/N$ |
| Spacing $\Delta k$ | field of view $FOV = 1/\Delta k$ |
| $\Delta k$ too large (object > FOV) | fold-over (aliasing) |
| Zeros added around the k-space | interpolation (smaller pixels) **without** resolution gain |
| Abrupt truncation at $k_{max}$ | Gibbs ringing (sinc PSF) — reduced by filtering, at the cost of blurring |
| Sampling function $S(\mathbf{k})$ | image $=$ object $\circledast$ PSF, with $PSF = \mathcal{F}^{-1}\{S\}$ |
| Regular / random / variable-density sampling | coherent fold-over / incoherent noise-like artefact |
| Non-Cartesian sampling | NUFFT + density compensation (analytical or numerical); undersampling gives streaks |
"""

# ╔═╡ ce857c34-c939-11f1-8a90-dbf15df7b4a9
md"""
# Helper functions
"""

# ╔═╡ ce857c48-c939-11f1-b7d9-25caffb70cd2
begin
	"Pixel center coordinates (mm) of a matrix of size `n` with a field of view `fov`, 0 at index n÷2+1"
	pixcoords(n, fov) = ((1:n) .- (n ÷ 2 + 1)) .* (fov / n)

	"Display an image with physical axes (mm)"
	function show_img!(gp, A; title="", fov=FOV, colormap=:grays, colorrange=extrema(A))
		nx, ny = size(A)
		ax = Axis(gp; title, aspect=DataAspect(), xlabel="x (mm)", ylabel="y (mm)")
		heatmap!(ax, pixcoords(nx, fov[1]), pixcoords(ny, fov[2]), A; colormap, colorrange)
		return ax
	end

	"Display a k-space (log scale by default) with axes in sample index relative to the center"
	function show_ksp!(gp, K; title="", logscale=true)
		nx, ny = size(K)
		ax = Axis(gp; title, aspect=DataAspect(), xlabel="i_kx", ylabel="i_ky")
		if logscale
			kref = log10(maximum(abs, ksp))
			heatmap!(ax, (1:nx) .- (nx ÷ 2 + 1), (1:ny) .- (ny ÷ 2 + 1), log10.(abs.(K) .+ eps()); colormap=:inferno, colorrange=(kref - 5, kref))
		else
			heatmap!(ax, (1:nx) .- (nx ÷ 2 + 1), (1:ny) .- (ny ÷ 2 + 1), abs.(K); colormap=:inferno)
		end
		return ax
	end

	"Display a matrix without axes (gallery figures)"
	function panel!(gp, A; colormap=:grays, colorrange=extrema(A))
		ax = Axis(gp, aspect=DataAspect())
		hidedecorations!(ax)
		heatmap!(ax, A; colormap, colorrange)
		return ax
	end

	"Binary k-space image of non-Cartesian `nodes` (cycles/pixel) on a grid of size `N`"
	function rasterize(nodes, N)
		M = zeros(N)
		for j in axes(nodes, 2)
			i1 = clamp(round(Int, nodes[1, j] * N[1]) + N[1] ÷ 2 + 1, 1, N[1])
			i2 = clamp(round(Int, nodes[2, j] * N[2]) + N[2] ÷ 2 + 1, 1, N[2])
			M[i1, i2] = 1
		end
		return M
	end

	"Pixels inside the head (where the sensitivity maps are defined)"
	head_mask = dropdims(sum(abs2, maps_8ch, dims=3), dims=3) .> 0

	"Relative error inside the head after the optimal (complex) scaling of `x`"
	function rel_err(x, ref; mask=head_mask)
		a, b = x[mask], ref[mask]
		α = dot(a, b) / dot(a, a)
		return norm(α .* a .- b) / norm(b)
	end

	"Format a fraction as a percentage"
	pct(x) = "$(round(100 * x, digits=1)) %"

	"Full width at half maximum (same unit as `dx`) of the magnitude of a 1D profile around its maximum"
	function fwhm(p, dx)
		a = abs.(p)
		c = argmax(a)
		half = a[c] / 2
		l = c
		while l > 1 && a[l-1] >= half
			l -= 1
		end
		r = c
		while r < length(a) && a[r+1] >= half
			r += 1
		end
		return (r - l + 1) * dx
	end
end;

# ╔═╡ ce85782e-c939-11f1-bff0-ef3fa59f69b9
let
	f = Figure(size=(1100, 330))
	show_ksp!(f[1, 1], kspace_8ch[:, :, coil]; title="k-space (log₁₀|k|)")
	show_img!(f[1, 2], abs.(img_8ch[:, :, coil]); title="magnitude")
	show_img!(f[1, 3], angle.(img_8ch[:, :, coil]); title="phase", colormap=:twilight)
	show_img!(f[1, 4], abs.(maps_8ch[:, :, coil]); title="sensitivity map", colormap=:viridis)
	f
end

# ╔═╡ ce857860-c939-11f1-82b6-971f5a9beced
let
	f = Figure(size=(1100, 330))
	show_img!(f[1, 1], img_sos; title="Sum of squares")
	show_img!(f[1, 2], abs.(img); title="Sensitivity-weighted : |img|")
	show_img!(f[1, 3], angle.(img); title="angle(img)", colormap=:twilight)
	show_ksp!(f[1, 4], ksp; title="ksp (log₁₀|k|)")
	f
end

# ╔═╡ ce8578a6-c939-11f1-970e-c1398aaafbe2
let
	f = Figure(size=(900, 380))
	ax = show_ksp!(f[1, 1], K_point; title="k-space (one sample)", logscale=false)
	xlims!(ax, -25, 25); ylims!(ax, -25, 25)
	show_img!(f[1, 2], real.(img_point); title="real part of the image")
	show_img!(f[1, 3], imag.(img_point); title="imaginary part of the image")
	f
end

# ╔═╡ ce8578ea-c939-11f1-b17a-27a0b19abb62
let
	f = Figure(size=(1000, 380))
	show_ksp!(f[1, 1], ksp_cp; title="masked k-space")
	show_img!(f[1, 2], abs.(img_cp); title="image ($(center_mode) only)")
	show_img!(f[1, 3], abs.(img); title="reference")
	Label(f[0, :], "Fraction of the k-space samples : $(round(100 * mean(mask_cp), digits=1)) %   —   fraction of the energy : $(round(100 * energy_cp, digits=1)) %", fontsize=16)
	f
end

# ╔═╡ ce857946-c939-11f1-9851-5f5972b926b2
let
	f = Figure(size=(1100, 420))
	ax = show_ksp!(f[1, 1], zero_pad(ksp_low, (Nx, Ny)); title="acquired k-space")
	show_img!(f[1, 2], abs.(img_low); title="matrix $(nx_res)×$(ny_res) — pixel $(round(pix_low[1], digits=2))×$(round(pix_low[2], digits=2)) mm")
	show_img!(f[1, 3], abs.(img); title="reference : matrix $(Nx)×$(Ny)")

	# profile along x through the center of the image
	ax = Axis(f[1, 4], title="profile along x (y = 0)", xlabel="x (mm)", ylabel="|m|")
	lines!(ax, pixcoords(Nx, FOV[1]), abs.(img[:, Ny÷2+1]), label="reference", color=:gray60)
	stairs!(ax, pixcoords(nx_res, FOV[1]), abs.(img_low[:, ny_res÷2+1]), step=:center, label="low resolution", color=:red)
	axislegend(ax, position=:cb)
	colsize!(f.layout, 4, Relative(0.3))
	f
end

# ╔═╡ ce8579c0-c939-11f1-a208-cd73da949c4b
let
	zoom_x, zoom_y = (-50, 0), (20, 80)  # zoom region (mm)
	f = Figure(size=(1100, 700))
	show_ksp!(f[1, 1], ksp_acq; title="acquired k-space $(n_acq[1])×$(n_acq[2])")
	show_ksp!(f[2, 1], ksp_zf; title="zero-filled k-space $(n_zf[1])×$(n_zf[2])")
	for (i, (im_, t)) in enumerate([(img_acq, "acquired, no zero-filling"), (img_zf, "zero-filled ×$(zf_factor)"), (img, "reference $(Nx)×$(Ny)")])
		show_img!(f[1, i+1], abs.(im_); title=t)
		ax = show_img!(f[2, i+1], abs.(im_); title="zoom", fov=FOV)
		limits!(ax, zoom_x..., zoom_y...)
	end
	f
end

# ╔═╡ ce8579fa-c939-11f1-b192-dbaeb4a40b9e
let
	f = Figure(size=(1100, 420))
	show_ksp!(f[1, 1], ksp .* mask_us; title="acquired lines (R = $(R_us))")
	show_img!(f[1, 2], abs.(img_us); fov=FOV_us, title="(a) acquired lines only : FOV $(FOV_us[1])×$(FOV_us[2]) mm")
	ax = show_img!(f[1, 3], abs.(img_us_zero); title="(b) missing lines set to 0")
	show_img!(f[1, 4], abs.(img); title="reference")
	f
end

# ╔═╡ ce857a72-c939-11f1-b36a-f79e772f31a3
let
	f = Figure(size=(1000, 450))
	ax1 = show_img!(f[1, 1], abs.(img); title="object (reference)")
	ax2 = show_img!(f[1, 2], abs.(img_sim); fov=(FOVx_sim, FOVy_sim), title="FOV $(FOVx_sim)×$(FOVy_sim) mm — matrix $(n_sim[1])×$(n_sim[2])")
	# draw the simulated FOV on the object, and the object FOV on the image
	poly!(ax1, Rect(-FOVx_sim / 2, -FOVy_sim / 2, FOVx_sim, FOVy_sim), color=:transparent, strokecolor=:red, strokewidth=2)
	poly!(ax2, Rect(-FOV[1] / 2, -FOV[2] / 2, FOV[1], FOV[2]), color=:transparent, strokecolor=:cyan, strokewidth=1, linestyle=:dash)
	lim = max(FOVx_sim, FOVy_sim, FOV...) / 2
	for ax in (ax1, ax2)
		limits!(ax, -lim, lim, -lim, lim)
	end
	f
end

# ╔═╡ ce857ac4-c939-11f1-929b-2f48a7563010
let
	f = Figure(size=(1100, 750))
	ax = Axis(f[1, 1], title="sampling function S(k)", aspect=DataAspect(), xlabel="i_kx", ylabel="i_ky")
	heatmap!(ax, (1:Nx) .- (Nx ÷ 2 + 1), (1:Ny) .- (Ny ÷ 2 + 1), S_psf, colormap=:grays, colorrange=(0, 1))
	show_img!(f[1, 2], log10.(abs.(psf) .+ 1e-6); title="log₁₀|PSF|", colormap=:inferno, colorrange=(-3, 0))
	show_img!(f[1, 3], abs.(img_psf); title="image = PSF ⊛ object")

	x_f, y_f = pixcoords(zf_psf * Nx, FOV[1]), pixcoords(zf_psf * Ny, FOV[2])
	for (i, (coord, prof, name)) in enumerate([(x_f, prof_x, "x"), (y_f, prof_y, "y")])
		ax = Axis(f[2, i], title="PSF profile along $name — FWHM = $(round(fwhm(prof, step(coord)), digits=2)) mm", xlabel="$name (mm)")
		lines!(ax, coord, real.(prof), label="real")
		lines!(ax, coord, abs.(prof), label="magnitude", linestyle=:dash)
		hlines!(ax, [0, 0.5], color=:gray, linestyle=:dot)
		xlims!(ax, -15, 15)
		axislegend(ax)
	end
	ax = Axis(f[2, 3], title="|PSF| along y (full FOV)", xlabel="y (mm)", yscale=log10)
	lines!(ax, y_f, abs.(prof_y) .+ 1e-6)
	ylims!(ax, 1e-4, 1.5)
	f
end

# ╔═╡ ce857b3a-c939-11f1-8b41-d7ad9a8add09
gallery = let
	c = Ny ÷ 2 + 1
	regular = Float64.(repeat(permutedims([mod(y - c, 4) == 0 for y in 1:Ny]), Nx, 1))

	# Cartesian patterns : PSF and images with the FFT
	cart = [("full", ones(Nx, Ny)),
		("regular R = 4", regular),
		("random", Float64.(random_mask((Nx, Ny); R=4))),
		("random + center", Float64.(random_mask((Nx, Ny); R=4, ncenter=(16, 16)))),
		("Poisson disc + center", Float64.(poisson_mask((Nx, Ny); R=4)))]
	out = [(name=n, mask=M, psf=ifft2c(M), phantom=ifft2c(M .* fft2c(phantom)), brain=ifft2c(M .* fft2c(img))) for (n, M) in cart]

	# radial pattern : PSF and images with the NUFFT (64 spokes ≈ 403/64 ≈ 6 times below Nyquist)
	nodes = radial_nodes(64, 256)
	p = plan_nfft(nodes, (Nx, Ny))
	w = ramp_dcf(64, 256)
	rad = (name="radial 64 spokes", mask=rasterize(nodes, (Nx, Ny)), psf=adjoint(p) * complex.(w),
		phantom=adjoint(p) * (w .* (p * phantom)), brain=adjoint(p) * (w .* (p * img)))
	insert!(out, 3, rad)
	out
end;

# ╔═╡ ce857b62-c939-11f1-8748-cbaa912e1a07
let
	f = Figure(size=(1300, 980))
	for (j, g) in enumerate(gallery)
		Label(f[0, j], g.name, fontsize=15, tellwidth=false)
		panel!(f[1, j], g.mask; colorrange=(0, 1))
		panel!(f[2, j], log10.(abs.(g.psf) ./ maximum(abs, g.psf) .+ 1e-6); colormap=:inferno, colorrange=(-3, 0))
		panel!(f[3, j], abs.(g.phantom) ./ maximum(abs, g.phantom); colorrange=(0, 1))
		panel!(f[4, j], abs.(g.brain) ./ maximum(abs, g.brain); colorrange=(0, 1))
	end
	for (i, t) in enumerate(["k-space", "log₁₀|PSF|", "phantom", "brain"])
		Label(f[i, 0], t, rotation=π / 2, fontsize=15, tellheight=false)
	end
	f
end

# ╔═╡ ce857bba-c939-11f1-991e-ddfecdb72a7b
let
	f = Figure(size=(1100, 420))
	ax = Axis(f[1, 1], title="trajectory ($(n_spokes) spokes)", aspect=DataAspect(), xlabel="k_x (cycles/pixel)", ylabel="k_y")
	sel = 1:max(1, length(data_rad) ÷ 20000):length(data_rad)
	scatter!(ax, nodes_rad[1, sel], nodes_rad[2, sel], markersize=2, color=log10.(abs.(data_rad[sel]) .+ 1), colormap=:inferno)
	ax_psf = show_img!(f[1, 2], log10.(abs.(psf_rad) .+ 1e-6); title="log₁₀|PSF|", colormap=:inferno, colorrange=(-3, 0))
	R_free = n_spokes / (π * 0.5)   # radius (mm) of the streak-free region : 1/Δk_θ = N_spokes / (π k_max), k_max = 0.5 mm⁻¹
	lines!(ax_psf, R_free .* cos.(0:0.05:2π+0.05), R_free .* sin.(0:0.05:2π+0.05), color=:cyan, linestyle=:dash)
	show_img!(f[1, 3], abs.(img_rad); title="adjoint NUFFT — DCF : $(dcf_choice)")
	show_img!(f[1, 4], abs.(img); title="reference")
	Label(f[0, :], "Nyquist criterion for $(Nread) samples per spoke : N_spokes ≥ π/2 × $(Nread) ≈ $(n_nyquist) spokes", fontsize=16)
	f
end

# ╔═╡ ce857bee-c939-11f1-b3ab-d3a3e2a9a292
let
	f = Figure(size=(1300, 430))
	ax = Axis(f[1, 1], title="DCF along one spoke", xlabel="k (cycles/pixel)", ylabel="w")
	r = ((0:Nread-1) .- Nread ÷ 2) ./ Nread
	lines!(ax, r, w_ramp[1:Nread], label="ramp |k|")
	lines!(ax, r, w_sdc[1:Nread], label="sdc", linestyle=:dash)
	axislegend(ax, position=:ct)
	for (j, (name, w)) in enumerate([("none", w_none), ("ramp |k|", w_ramp), ("sdc", w_sdc)])
		rec = adjoint(p_rad) * (w .* data_rad)
		show_img!(f[1, j+1], abs.(rec); title="DCF $(name) — error $(pct(rel_err(rec, img)))")
	end
	f
end

# ╔═╡ ce857c52-c939-11f1-a1a6-0357bdfc8bdc
begin
	hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))
	tip(text) = Markdown.MD(Markdown.Admonition("note", "Tip", [text]))
	note(text) = Markdown.MD(Markdown.Admonition("note", "note", [text]))
	important(text) = Markdown.MD(Markdown.Admonition("example", "Important", [text]))
	question(text) = Markdown.MD(Markdown.Admonition("danger", "Question", [text]))
	answer_folded(text) = @htl("""
	<details class="admonition info" style="background-color: #f1f8e9 !important; border-left-color: #2e7d32 !important; display: block; margin: 1em 0; padding: 0; border-left-style: solid; border-left-width: .4rem; border-radius: .2rem; box-shadow: 0 .2rem .5rem rgba(0,0,0,.05), 0 0 .05rem rgba(0,0,0,.1); overflow: hidden;">
	  <summary class="admonition-title" style="cursor: pointer; font-weight: bold; padding: .6rem 1rem .6rem 2rem; background-color: rgba(46, 125, 50, 0.1); color: #1b5e20; border-bottom: 1px solid rgba(46, 125, 50, 0.1); user-select: none;">
	    Answer
	  </summary>
	  <div class="admonition-content" style="padding: 1rem;">
	    $text
	  </div>
	</details>
	""")
end

# ╔═╡ ce8577f4-c939-11f1-a9bb-9166ec6331c2
md"""
**Q1** Why do we need `ifftshift` **before** and `fftshift` **after** the `ifft`? What would the image look like without them?
""" |> question

# ╔═╡ ce8577fc-c939-11f1-a11f-7de7678552e2
answer_folded(md"""
`ifftshift` moves the k-space center ($k=0$, index $N/2+1$) to index 1, as expected by the FFT. `fftshift` then moves the image center from index 1 back to the middle of the matrix.

Without the final `fftshift`, the image is cut in 4 quarters which are swapped (the center of the object ends up in the corners). Without the first `ifftshift`, the k-space center is not at index 1: this is equivalent to multiplying the image by a phase ramp $e^{i\pi n}=(-1)^n$, i.e. a checkerboard phase. The magnitude is unchanged but the phase is wrong. Try it!
""")

# ╔═╡ ce857806-c939-11f1-9616-179a45c8bcc4
md"""
**Q2** Reconstruct the image of each coil. Display, for one coil, the k-space (in log scale) and the magnitude and phase of the image. Why do we display the k-space on a **log** scale?
""" |> question

# ╔═╡ ce857838-c939-11f1-b726-3f74f2978559
answer_folded(md"""
The k-space values span several orders of magnitude: the center is about $10^4$ times larger than the periphery. On a linear scale we would only see a bright dot in the middle.

Each coil "sees" the part of the brain close to it: the magnitude follows the sensitivity map, and each coil has a different phase.
""")

# ╔═╡ ce8578b0-c939-11f1-b1f5-b3380a4c20db
md"""
**Q3** How are the orientation and the wavelength of the stripes related to the position of the sample in k-space? Compute the wavelength for $i_{k_x}=4,\ i_{k_y}=0$ and check on the figure.

**Q4** What is the effect of the phase of the sample?

**Q5** Which image corresponds to the central point ($i_{k_x}=i_{k_y}=0$)? What does the value of `ksp` at the center represent?
""" |> question

# ╔═╡ ce8578b8-c939-11f1-bddc-a5d5b58861ff
answer_folded(md"""
**Q3** The stripes are perpendicular to the vector $\mathbf{k}$. The spacing between two samples is $\Delta k = 1/FOV$, so the sample $i_k$ corresponds to $k = i_k/FOV$ and the wavelength is

$$\lambda = \frac{1}{|\mathbf{k}|} = \frac{FOV}{|i_k|}$$

i.e. there are exactly $|i_k|$ periods in the FOV. For $i_{k_x}=4$: $\lambda = 160/4 = 40$ mm. The further from the center, the finer the stripes.

**Q4** The phase of the sample **shifts** the stripes (a phase of $2\pi$ shifts them by one wavelength).

**Q5** The center ($\mathbf{k}=0$) is a constant image. `ksp` at the center is proportional to the **sum (mean) of the image** — this is why it is by far the largest value of the k-space.
""")

# ╔═╡ ce857900-c939-11f1-ad5a-0b1bf52472e0
md"""
**Q6** Keep only the center (radius 0.1). What information is preserved? What is lost?

**Q7** Keep only the periphery (radius 0.1). What do you see?

**Q8** What fraction of the samples, and what fraction of the energy, does the central region (radius 0.1) contain?
""" |> question

# ╔═╡ ce85790a-c939-11f1-90b8-51a41f09adb6
answer_folded(md"""
**Q6** The center contains the **low spatial frequencies**: the global shape and the **contrast** between tissues are preserved, but the image is blurred (fine details and edges are lost).

**Q7** The periphery contains the **high spatial frequencies**: only the **edges and fine details** remain; the contrast (flat regions) disappears.

**Q8** About 0.8 % of the samples contain ≈ 80 % of the energy of the signal. This is why the contrast of a sequence is mainly determined by the moment the center of the k-space is acquired (e.g. the *effective echo time* in turbo spin echo).
""")

# ╔═╡ ce85795c-c939-11f1-aa1c-2970d6f1a825
md"""
**Q9** For a matrix of $80 \times 110$, compute the pixel size and $k_{max}$ (in mm⁻¹). Give the general relation between $\Delta x$, $FOV$, $N$ and $k_{max}$.

**Q10** Choose $n_x = 160$ and $n_y = 40$. What is the shape of the pixels? In which direction is the image blurred?

**Q11** At low resolution, look at the edges of the skull: what is this artefact? (we will explain it in section 7)
""" |> question

# ╔═╡ ce85796e-c939-11f1-9079-855d3bdfce42
answer_folded(md"""
**Q9** $\Delta x = FOV/N = 160/80 = 2$ mm and $\Delta y = 220/110 = 2$ mm. The k-space extends from $-k_{max}$ to $k_{max}$ with $N$ samples spaced by $\Delta k = 1/FOV$:

$$\Delta x = \frac{FOV}{N} = \frac{1}{N\, \Delta k} = \frac{1}{2\, k_{max}}$$

so $k_{max} = 1/(2 \times 2) = 0.25$ mm⁻¹. The FOV does not change: it is set by $\Delta k$ which we did not modify.

**Q10** The pixels are rectangular: 1 mm × 5.5 mm. The image is blurred only along $y$. Rectangular pixels are often used clinically to save time along the phase encoding direction (scan time ∝ number of phase encoding lines).

**Q11** These oscillations near sharp edges are the **Gibbs (truncation) artefact**: the abrupt truncation of the k-space at $k_{max}$ convolves the image with a sinc function (see section 7).
""")

# ╔═╡ ce8579c8-c939-11f1-ac73-bf48286d4b77
md"""
**Q12** What is the pixel size after a zero-filling of 2? Of 4?

**Q13** Compare the zoomed zero-filled image with the reference. Has the spatial resolution improved? Why?

**Q14** What is zero-filling useful for, then?
""" |> question

# ╔═╡ ce8579d2-c939-11f1-9d0a-1b2f9e90735d
answer_folded(md"""
**Q12** The FOV is unchanged and the matrix is multiplied by the zero-filling factor: the pixel size is divided by 2 (or 4).

**Q13** No. The image looks smoother (no more visible "blocks") but it is **not sharper**: no new information was acquired, $k_{max}$ is the same. The true resolution ($1/(2k_{max})$) is unchanged; zero-filling only **interpolates** the image (with a sinc kernel) on a finer grid.

**Q14** It gives a nicer display (less pixelated), reduces partial-volume effects due to the *position* of the pixel grid, and is useful before resampling/registration or to sample the PSF finely (see section 7). Be careful: the "reconstructed resolution" written on the scanner may include this interpolation!
""")

# ╔═╡ ce857a0e-c939-11f1-99fd-41e428b75ba4
md"""
**Q15** What is the FOV for $R=2$ along $y$? Along $x$? Explain the artefact in image (a).

**Q16** Write an equation linking the value of a pixel of the folded image $m_R(x,y)$ to the reference image $m(x,y)$ (for an undersampling along $y$). Implement it in the function `fold` below and check that you obtain exactly the image (a).

**Q17** What is the link between images (a) and (b)?

**Q18** On a scanner, fold-over appears along the **phase encoding** direction but almost never along the **readout** direction. Why?
""" |> question

# ╔═╡ ce857a40-c939-11f1-9dd8-7d91fc5f916b
answer_folded(md"""
**Q15** $FOV_y = 220/2 = 110$ mm and $FOV_x = 160/2 = 80$ mm. The object is larger than the new FOV: the parts outside the FOV are not lost, they **fold back** on the opposite side of the image (aliasing / wrap-around artefact).

**Q16** The image is the sum of the copies of the object shifted by the new FOV:

$$m_R(x,y) = \sum_{n=0}^{R-1} m\!\left(x,\ y + n\,\frac{FOV_y}{R}\right)$$

For $R=2$: $m_2(x,y) = m(x,y) + m(x, y + FOV_y/2)$. One equation, two unknowns: this is the problem solved by **parallel imaging** (SENSE) with several coils.

**Q17** Sampling the k-space with a spacing $R\,\Delta k$ makes the image **periodic** with a period $FOV/R$. Image (b) shows this periodic image over the full FOV (with $R$ identical replicas): its central part is exactly image (a). The scanner only reconstructs one period.

**Q18** Along the readout, sampling more points does not cost time (it only requires a shorter dwell time / larger bandwidth with the same gradient area). Scanners therefore **oversample by 2** the readout and discard the outer part of the image. Along the phase encoding direction each line costs one TR, so the FOV is chosen as small as possible.
""")

# ╔═╡ ce857a7c-c939-11f1-ad5a-35bb6b40a2ce
md"""
**Q19** Find the smallest FOV along $x$ and along $y$ for which there is no fold-over.

**Q20** What happens when the FOV is larger than the object? Is it a problem?

**Q21** For a fixed FOV, change the resolution: what changes? How would the scan time change if $y$ were the phase encoding direction?
""" |> question

# ╔═╡ ce857a86-c939-11f1-8a1f-8593fd7e73e7
answer_folded(md"""
**Q19** The FOV must contain the whole object (including the skull and the fat): roughly $FOV_x \gtrsim 145$ mm and $FOV_y \gtrsim 190$ mm. Below these values the object folds back.

**Q20** No artefact, the image simply contains noise/background around the object. But for the same resolution, the matrix (and thus the number of phase encoding lines and the scan time) increases.

**Q21** The FOV is unchanged but the matrix size $N = FOV/\Delta x$ changes. Scan time ∝ number of phase encoding lines $= FOV_y/\Delta y$: going from 2 mm to 1 mm doubles the scan time (and the SNR per voxel drops because the voxel is smaller).
""")

# ╔═╡ ce857acc-c939-11f1-83ec-677a6a4fee19
md"""
**Q22** *Full k-space*: what is the PSF? Is the FWHM exactly the pixel size?

**Q23** *Central crop* (0.5): describe the PSF. What is its FWHM? How does it explain the resolution loss **and** the Gibbs ringing observed in section 4?

**Q24** *Crop + Hann filter*: compare to the previous case. What is the trade-off?

**Q25** *Regular undersampling* ($R=2$, then 3, 4): what appears in the PSF? Link it to section 6.

**Q26** *Random undersampling*: how is the artefact modified? Why is it interesting for **compressed sensing**?
""" |> question

# ╔═╡ ce857ad6-c939-11f1-86b4-0bfe79f16c07
answer_folded(md"""
**Q22** The PSF of the full k-space is a 2D **sinc** (Dirichlet kernel) which is 1 at the center and 0 at all the other pixels: on the image grid it is a Dirac, the image is perfect. On a finer grid we see that its FWHM is ≈ 1.2 $\Delta x$: the actual resolution is slightly worse than the pixel size.

**Q23** A sinc twice as wide (FWHM ≈ 1.2 × 2 mm ≈ 2.4 mm): the convolution blurs the image (resolution loss). The sinc has **negative side lobes** (−22 % for the first one) which create oscillations near sharp edges: the **Gibbs ringing** (truncation artefact).

**Q24** The Hann filter smoothly reduces the high frequencies: the side lobes almost disappear (no ringing) but the main lobe is wider (FWHM ≈ 2 $\Delta x$, i.e. ≈ 4 mm instead of 2.4 mm for a crop of 0.5): **less ringing, more blurring**. This is the "k-space filter" option on scanners.

**Q25** The PSF contains **$R$ peaks** separated by $FOV/R$ (the sampling function is a Dirac comb of period $R\Delta k$, its Fourier transform is a Dirac comb of period $FOV/R$). Convolving the object with this PSF superimposes shifted copies of the object: this is the **fold-over** artefact.

**Q26** With a random sampling, the energy of the replicas is spread over the whole PSF as low-level, **incoherent**, noise-like side lobes. The artefact looks like noise instead of structured copies of the object. Compressed sensing uses this property: an incoherent artefact can be removed by a sparsity-promoting reconstruction (e.g. wavelets, total variation), whereas coherent fold-over cannot.
""")

# ╔═╡ ce857ae0-c939-11f1-ba89-cfec836b98c6
md"""
**Q27** (to go further) Check the convolution theorem for one pixel: compute $m_{acq}(\mathbf{r}_0) = \sum_{\mathbf{r}} m(\mathbf{r})\, PSF(\mathbf{r}_0 - \mathbf{r})$ with a direct (circular) sum, and compare to `img_psf`.
""" |> question

# ╔═╡ ce857b6c-c939-11f1-9e2d-d11cdd7f6fa3
md"""
**Q28** Which patterns produce a **coherent** artefact (structured copies of the object) and which ones an **incoherent** (noise-like) artefact? Is the width of the central peak of the PSF (the resolution) modified?

**Q29** Compare "random" and "random + center". Why is it so important to fully sample the center of the k-space?

**Q30** Why does the Poisson-disc pattern use a variable density? What is the advantage of imposing a minimum distance between the samples compared to a purely random draw?
""" |> question

# ╔═╡ ce857b80-c939-11f1-9ad6-bbc7a58c7cb9
answer_folded(md"""
**Q28** *Regular*: $R$ sharp replicas of the peak → coherent fold-over. *Radial*: streaks, spread around the object. *Random* and *Poisson disc*: a sharp central peak on top of a low-level, noise-like background → incoherent artefact. In all cases the central peak keeps the same width: the resolution is set by $k_{max}$, not by the number of samples.

**Q29** The center contains most of the energy (≈ 80 % in 0.8 % of the samples, section 3.1). Missing central samples creates large low-frequency errors: the gray square becomes blotchy and the contrast is wrong. With a fully sampled center, the energy of the artefact is much lower and only affects the fine details. The fully sampled center is also used as **calibration** data (GRAPPA kernel, sensitivity maps — see the advanced reconstruction practical).

**Q30** The variable density follows the energy distribution of the k-space: many samples near the center where the signal is strong, few at the periphery where it is weak → less artefact energy for the same number of samples. The minimum distance avoids clusters and large holes in the k-space: the artefact is more uniform and is pushed away from the central peak of the PSF. This is the classical sampling for **compressed sensing** (e.g. `bart poisson`).
""")

# ╔═╡ ce857bc6-c939-11f1-ac31-1b687634357a
md"""
**Q31** Reconstruct the image **without** density compensation (`none`). What do you observe? Look at the PSF and explain.

**Q32** With the DCF, decrease the number of spokes (e.g. 32). What is the artefact? Compare it to the Cartesian fold-over.

**Q33** The cyan circle shows the region where the PSF is free of streaks. Its radius is $N_{spokes}/(\pi k_{max})$. Explain this formula by computing the distance $\Delta k_\theta$ between two neighboring spokes at the edge of the k-space (remember that for Cartesian sampling the PSF replicas are at a distance $1/\Delta k$).

**Q34** How many spokes are needed to fulfill the Nyquist criterion everywhere? Compare with the number of lines of a Cartesian acquisition with the same resolution.
""" |> question

# ╔═╡ ce857bda-c939-11f1-9de5-336c6a3b0b1b
answer_folded(md"""
**Q31** Without DCF the center of the k-space is over-weighted (all the spokes pass through it): the image is very **blurred**. The PSF has a very wide main lobe ($\propto 1/|\mathbf{r}|$): the low frequencies dominate.

**Q32** **Streaks** appear, radiating from the object. As for Cartesian undersampling, the FOV becomes smaller than the object, but the artefact is spread around the object as streaks instead of a coherent copy — this is why radial imaging is robust to undersampling (and to motion).

**Q33** Along a spoke the spacing is $\Delta k_r = 1/FOV$. Between two spokes, the distance at $k_{max}$ is the arc length $\Delta k_\theta = \pi k_{max} / N_{spokes}$ (the spokes cover 180°, i.e. $\pi$ rad, and each spoke crosses the whole diameter). As for Cartesian sampling (replicas of the PSF at a distance $1/\Delta k$), the first streaks of the PSF appear at a distance $1/\Delta k_\theta = N_{spokes}/(\pi k_{max})$. With $k_{max} = 0.5$ mm⁻¹ the radius is $2N_{spokes}/\pi$ mm (≈ 41 mm for 64 spokes): an object smaller than this diameter is (almost) free of streaks.

**Q34** We need $\Delta k_\theta \le \Delta k_r$: $N_{spokes} \ge \frac{\pi}{2} N_{read} \approx 403$ spokes for $N_{read} = 256$, whereas a Cartesian acquisition would need 256 lines: a fully sampled radial acquisition needs $\pi/2 \approx 1.57$ times more readouts.
""")

# ╔═╡ ce857bf8-c939-11f1-98bc-edc3fcf01984
md"""
**Q35** Compare the ramp and the `sdc` weights along a spoke. Where do they differ?

**Q36** Compare the images and the errors for 64 and for 400 spokes.

**Q37** Why is a numerical DCF necessary for spiral trajectories?
""" |> question

# ╔═╡ ce857c16-c939-11f1-8d7e-e77216bef749
answer_folded(md"""
**Q35** Near the center, both are proportional to $|\mathbf{k}|$ (with a different value at $k=0$, where the ramp uses an ad-hoc value). With 64 spokes, the `sdc` weights **saturate** beyond $|k| \approx 0.25$ cycles/pixel: there, the spokes are separated by more than the width of the interpolation kernel (the periphery is undersampled), so `sdc` "sees" each sample as isolated and gives it a constant weight, whereas the ramp keeps increasing. Close to the Nyquist criterion (≈ 400 spokes), the two curves coincide except at the very center.

**Q36** With 64 spokes, `sdc` down-weights the undersampled periphery, which reduces the streaks: the error is slightly lower (≈ 15.5 % vs 16.9 %). With 400 spokes, the ramp (the exact answer for this ideal radial trajectory) gives a lower error (≈ 6 % vs 8 %), the difference coming from the center weights. Without DCF, the image stays blurred whatever the number of spokes.

**Q37** For a spiral, the density depends on the design of the trajectory (number of interleaves, variable density, gradient limits) and on the gradient imperfections (measured trajectory). There is no simple analytical formula, but the iterative method works for **any** trajectory.
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"
NFFT = "efe261a4-0d2b-5849-be55-fc731d526b0d"
NFFTTools = "7424e34d-94f7-41d6-98a0-85abaf1b6c91"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
FFTW = "~1.10.0"
GLMakie = "~0.13.13"
HypertextLiteral = "~1.0.0"
MAT = "~0.12.1"
NFFT = "~0.14.3"
NFFTTools = "~0.3.0"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "128ea62e2945b7b9f857230d33d8f303922fe0f2"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractNFFTs]]
deps = ["LinearAlgebra", "Printf", "Requires", "ScopedValues"]
git-tree-sha1 = "27aa55535187ed2e62aff2883399a443944d0f55"
uuid = "7f219486-4aa7-41d6-80a7-e08ef20ceed7"
version = "0.9.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractNFFTs.extensions]
    AbstractNFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "dee8388a9bec89c8713ad176f96960ae2f98598a"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.4"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.BasicInterpolators]]
deps = ["LinearAlgebra", "Memoize", "Random"]
git-tree-sha1 = "3f7be532673fc4a22825e7884e9e0e876236b12a"
uuid = "26cce99e-4866-4b6d-ab74-862489e035e0"
version = "0.7.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkSplitters]]
git-tree-sha1 = "1c52c8e2673edc030191177ff1aee42d25149acb"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.2.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "7bc84b769c1d384315e7b5c4ac03a6c303e6cf35"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.8"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a958ab3a40c755563f5e1405c0846cb0446bf19d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.131"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9741b45e1189f878eefe5789e6833dbdefc5adfa"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "af06f66cca2b698ab9c482de55977ff8178d025e"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.6"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "PrecompileTools", "Printf", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "3e1770a9d85b8bd1767431b959d657dc8825d78d"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.13.13"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "364685f5ffde25deb1bbcfd5bb278a5c6b7a9b37"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.11"

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

    [deps.GeometryBasics.weakdeps]
    Extents = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "26e37af34e8ccb7a8358bd8de1619c7bcb526738"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.4"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "aws_c_s3_jll", "dlfcn_win32_jll", "libaec_jll", "mpif_jll"]
git-tree-sha1 = "58d79d3a61d2fdeda667fead57546ae22a20b27a"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "2.2.3+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "cbbd99d516990d05c02aa132aa077b2bf50ae323"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.15.0+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "31bb6c92405c084617facc1d7ed9eb6c402d061e"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.30"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "ff294afb9a15d31d8d7422da138844641a73135f"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.11"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "0db6aea5b64caa1c47e9fdd27394f0296f05e8bf"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.15"

    [deps.IntervalSets.extensions]
    IntervalSetsMakieExt = "Makie"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "4657a834b01ce00f2b94546095c4350617b9af2c"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.10.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b7970cef8ae1c990ba0c09cd8bdc1145e006632f"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "22.1.7+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "b85e2797b2409570e84c4de46238c0ed5f6476ae"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.2"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MAT]]
deps = ["CodecZlib", "Dates", "HDF5", "OrderedCollections", "PooledArrays", "SparseArrays", "StringEncodings", "Tables"]
git-tree-sha1 = "f407947457464f24f6cf34b744e2f86a7929a93e"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.12.1"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MPIABI_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "42e2cada9f35500b4d227106ce2b60e85d9f15bc"
uuid = "b5ada748-db0f-5fc0-8972-9331c762740c"
version = "1.0.1+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "08cb47f95a2f1b795e43a9e4f9783e68ecdc1ca6"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "5.0.3+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8e98d5d80b87403c311fd51e8455d4546ba7a5f8"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.12"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "09a99227205419c00e734f413ee98c18bc1b02e4"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.7+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "f2c8715d05bf10f9d4dc354e69dee30b6be53239"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.13"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "c009236e222df68e554c7ce5c720e4a33cc0c23f"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.5.3"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "ac6cb1d8807a05cf1acc9680e09d2294f9d33956"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.8"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.NFFT]]
deps = ["AbstractNFFTs", "BasicInterpolators", "Distributed", "FFTW", "LinearAlgebra", "OhMyThreads", "PrecompileTools", "Printf", "Random", "Reexport", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "9a1ecb7604507a6c0c77723b93dba335b5029cb1"
uuid = "efe261a4-0d2b-5849-be55-fc731d526b0d"
version = "0.14.3"

    [deps.NFFT.extensions]
    NFFTGPUArraysExt = ["Adapt", "GPUArrays"]

    [deps.NFFT.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"

[[deps.NFFTTools]]
deps = ["AbstractFFTs", "AbstractNFFTs", "FFTW", "LinearAlgebra", "NFFT"]
git-tree-sha1 = "6307fd141ca6df79fb88bbb9021e588ed71ba3e7"
uuid = "7424e34d-94f7-41d6-98a0-85abaf1b6c91"
version = "0.3.0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "ae638151254d159c4740d21a0820f17b7f74eb43"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.7"
weakdeps = ["Markdown", "ProgressMeter"]

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"
    ProgressMeterExt = "ProgressMeter"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "38a93f17e431141c6470bb67a88952a7c4f0e928"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "fd09db52a90efaae33a3d91f0bc71a22c429e8f1"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.14+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "0dba69ef957d790d41484ca7c20212c5ac806738"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.14+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7126b66b721a605a2fec966a2874c5ed53258eb3"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "90b41ced6bacd8c01bd05da8aed35c5458891749"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "5005266de4bfe50e53ff44a5cb5c540b6e47a254"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "e6ffa7bd4f8d0b8744d18f219304738d952f9a37"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.11"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "de349d5585559b692f518b6d3fc7c179d64b2364"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.23"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "b814d5005d6a529d740ffe06f8a86396f6501138"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.9.2"

    [deps.StructUtils.extensions]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "1f0f9f401753701a7e4113b5056ca38d33875b55"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.29.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b826aedca5fce2e3c91904e338c279e5c1a2da74"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.15.4+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.aws_c_auth_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_http_jll", "aws_c_sdkutils_jll"]
git-tree-sha1 = "8cab83c96af80a1be968251ce1a0548a7545484d"
uuid = "2b3700d1-4306-52e2-a478-c162f0c514be"
version = "0.9.6+0"

[[deps.aws_c_cal_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "22c0f42f4a1f0dc5dcfa8fd267c4ac407c455e7a"
uuid = "70f11efc-bab2-57f1-b0f3-22aad4e67c4b"
version = "0.9.13+0"

[[deps.aws_c_common_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a759cb9bf456ad792cc7898a81ae333cce9ef02a"
uuid = "73048d1d-b8c4-5092-a58d-866c5e8d1e50"
version = "0.12.6+0"

[[deps.aws_c_compression_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "7910c72f45f44afd297c39fe43b99c56d5ed22ec"
uuid = "73a04cd5-f3d7-5bac-9290-e8adb709f224"
version = "0.3.2+0"

[[deps.aws_c_http_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_compression_jll", "aws_c_io_jll"]
git-tree-sha1 = "3fb8685778068de502c72fec5dd8075e037cee15"
uuid = "3254fc65-9028-534d-aa9d-d76d128babc6"
version = "0.10.15+0"

[[deps.aws_c_io_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_common_jll", "s2n_tls_jll"]
git-tree-sha1 = "7e481d474b2087ee8bbf55b81bf9119f21e396d9"
uuid = "13c41daa-f319-5298-b5eb-5754e0170d52"
version = "0.26.3+0"

[[deps.aws_c_s3_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_auth_jll", "aws_c_common_jll", "aws_c_http_jll", "aws_checksums_jll", "s2n_tls_jll"]
git-tree-sha1 = "3e9917ab25114feba657e71be41cad068b9f6595"
uuid = "bd1f34fb-993f-5903-a121-aaf302eed6d4"
version = "0.11.5+0"

[[deps.aws_c_sdkutils_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "c43dfba2c1ab9ea9f02f2c80e86fa16f6460244e"
uuid = "1282aa60-004d-510b-9f52-12498d409daa"
version = "0.2.4+1"

[[deps.aws_checksums_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "2570c8e23f4771a087b12a47edcaaa670ac05a01"
uuid = "b2a88e68-78e7-5e94-8c20-c02986ec140e"
version = "0.2.10+0"

[[deps.dlfcn_win32_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e141d67ffe550eadfb5af1bdbdaf138031e4805f"
uuid = "c4b69c83-5512-53e3-94e6-de98773c479f"
version = "1.4.2+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60f4792734488db6f42e2c7699f1d4594780bd03"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.7+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef17c47d22224aaecc76e597ab21a072e025cf7b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.14.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.mpif_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML"]
git-tree-sha1 = "a06fcd368cfe6fe2c0eb7b63320d4d27ddcd010d"
uuid = "9aeb927a-4695-514f-a259-621a69f20ec0"
version = "1.0.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ed2d8d4ad96c03d3abe3382cdbbdea9489b63636"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2023.1.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.s2n_tls_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d6cfe0874f96962fe4b20ed8bdb71c467e09c2c6"
uuid = "cddc5d3d-934d-5d3a-9747-62fc12ea3f48"
version = "1.7.11+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─ce848b7e-c939-11f1-b0e6-0b6babc5505a
# ╟─ce857734-c939-11f1-a910-9de016fca185
# ╟─ce857766-c939-11f1-a743-c7bd01aef4da
# ╟─ce857782-c939-11f1-b207-a90fed7c07e9
# ╟─ce85778e-c939-11f1-a186-e98bf7b3f0a9
# ╠═ce8577a2-c939-11f1-a52f-4f78ed5cd323
# ╠═ce8577ac-c939-11f1-9e03-513f5cbe81e4
# ╟─ce8577b6-c939-11f1-8693-11da8d5aab7f
# ╠═ce8577ca-c939-11f1-8929-fb5c8d4b616a
# ╟─ce8577d4-c939-11f1-ae8b-c7f121718b79
# ╠═ce8577de-c939-11f1-a210-174d90077116
# ╟─ce8577f4-c939-11f1-a9bb-9166ec6331c2
# ╟─ce8577fc-c939-11f1-a11f-7de7678552e2
# ╟─ce857806-c939-11f1-9616-179a45c8bcc4
# ╠═ce857810-c939-11f1-8cab-7b3e76ed2de2
# ╟─ce857826-c939-11f1-8d67-917d7dacd765
# ╠═ce85782e-c939-11f1-bff0-ef3fa59f69b9
# ╟─ce857838-c939-11f1-b726-3f74f2978559
# ╟─ce857842-c939-11f1-805e-112a1a7dbcdc
# ╠═ce857858-c939-11f1-b49f-e5f2d5617604
# ╠═ce857860-c939-11f1-82b6-971f5a9beced
# ╟─ce85787e-c939-11f1-9ec4-e9effdd86dbf
# ╟─ce857886-c939-11f1-ae95-1d3d04969db2
# ╠═ce85789c-c939-11f1-8812-c1c7e9b81f3e
# ╠═ce8578a6-c939-11f1-970e-c1398aaafbe2
# ╟─ce8578b0-c939-11f1-b1f5-b3380a4c20db
# ╟─ce8578b8-c939-11f1-bddc-a5d5b58861ff
# ╟─ce8578ce-c939-11f1-bc56-033bb57825ff
# ╟─ce8578d8-c939-11f1-bb73-ad8dd5b1409b
# ╠═ce8578e2-c939-11f1-9717-b9a7a2992306
# ╠═ce8578ea-c939-11f1-b17a-27a0b19abb62
# ╟─ce857900-c939-11f1-ad5a-0b1bf52472e0
# ╟─ce85790a-c939-11f1-90b8-51a41f09adb6
# ╟─ce857914-c939-11f1-ac3e-09bdb3a2341f
# ╠═ce85791e-c939-11f1-8191-43781599eea0
# ╟─ce857932-c939-11f1-a5eb-d7d23aae8f03
# ╠═ce85793c-c939-11f1-a977-c97d73a69d1d
# ╠═ce857946-c939-11f1-9851-5f5972b926b2
# ╟─ce85795c-c939-11f1-aa1c-2970d6f1a825
# ╟─ce85796e-c939-11f1-9079-855d3bdfce42
# ╟─ce857978-c939-11f1-adf3-c3c0894ef2f2
# ╟─ce8579a0-c939-11f1-be9c-87d81721c12a
# ╠═ce8579aa-c939-11f1-9c19-758b9191c856
# ╠═ce8579c0-c939-11f1-a208-cd73da949c4b
# ╟─ce8579c8-c939-11f1-ac73-bf48286d4b77
# ╟─ce8579d2-c939-11f1-9d0a-1b2f9e90735d
# ╟─ce8579dc-c939-11f1-9f54-63c420564585
# ╟─ce8579e6-c939-11f1-8fac-a5d9fb66055e
# ╠═ce8579ee-c939-11f1-a45f-5f9b1f3a7fd2
# ╠═ce8579fa-c939-11f1-b192-dbaeb4a40b9e
# ╟─ce857a0e-c939-11f1-99fd-41e428b75ba4
# ╠═ce857a18-c939-11f1-a74c-9b44cd187e00
# ╟─ce857a20-c939-11f1-81f3-399e2f152067
# ╟─ce857a40-c939-11f1-9dd8-7d91fc5f916b
# ╟─ce857a4a-c939-11f1-bc45-4db948920f84
# ╟─ce857a5e-c939-11f1-9d15-3d6253c39ecc
# ╠═ce857a68-c939-11f1-a5dd-ed94bb1d9e87
# ╠═ce857a72-c939-11f1-b36a-f79e772f31a3
# ╟─ce857a7c-c939-11f1-ad5a-35bb6b40a2ce
# ╟─ce857a86-c939-11f1-8a1f-8593fd7e73e7
# ╟─ce857a90-c939-11f1-9b09-23bfc459b72f
# ╟─ce857a9a-c939-11f1-80ab-c17e89ae7479
# ╠═ce857aa4-c939-11f1-8dfe-f97efd30ec03
# ╠═ce857ab8-c939-11f1-8afe-bb076db1a2bd
# ╠═ce857ac4-c939-11f1-929b-2f48a7563010
# ╟─ce857acc-c939-11f1-83ec-677a6a4fee19
# ╟─ce857ad6-c939-11f1-86b4-0bfe79f16c07
# ╟─ce857ae0-c939-11f1-ba89-cfec836b98c6
# ╠═ce857aea-c939-11f1-a6de-5fcf82392f04
# ╟─ce857afe-c939-11f1-96f1-37d7a88bb764
# ╠═ce857b08-c939-11f1-ab72-e71f83fada7d
# ╠═ce857b12-c939-11f1-b39b-4d4677e9d605
# ╠═ce857b30-c939-11f1-96ca-95af124a0140
# ╠═ce857b3a-c939-11f1-8b41-d7ad9a8add09
# ╠═ce857b62-c939-11f1-8748-cbaa912e1a07
# ╟─ce857b6c-c939-11f1-9e2d-d11cdd7f6fa3
# ╟─ce857b80-c939-11f1-9ad6-bbc7a58c7cb9
# ╟─ce857b88-c939-11f1-9f1d-890cd9c23883
# ╠═ce857b9e-c939-11f1-8bb6-ef9d2571f3a0
# ╠═ce857ba8-c939-11f1-a470-b7dc508c51d0
# ╠═ce857bb2-c939-11f1-882f-47c6ef0e25cb
# ╠═ce857bba-c939-11f1-991e-ddfecdb72a7b
# ╟─ce857bc6-c939-11f1-ac31-1b687634357a
# ╟─ce857bda-c939-11f1-9de5-336c6a3b0b1b
# ╟─ce857be4-c939-11f1-8740-db92fc4b4f98
# ╠═ce857bee-c939-11f1-b3ab-d3a3e2a9a292
# ╟─ce857bf8-c939-11f1-98bc-edc3fcf01984
# ╟─ce857c16-c939-11f1-8d7e-e77216bef749
# ╟─ce857c2c-c939-11f1-afdd-75cc4547f687
# ╟─ce857c34-c939-11f1-8a90-dbf15df7b4a9
# ╟─ce857c48-c939-11f1-b7d9-25caffb70cd2
# ╟─ce857c52-c939-11f1-a1a6-0357bdfc8bdc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
