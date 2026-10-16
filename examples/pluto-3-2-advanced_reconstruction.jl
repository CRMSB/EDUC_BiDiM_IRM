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

# ╔═╡ 54da8ca0-c93c-11f1-95ce-1dc2aba07a20
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
	using LinearOperators
	using Wavelets
	using LinearAlgebra
	using Statistics
	using Random

	PlutoUI.TableOfContents()
end

# ╔═╡ 54d999bc-c93c-11f1-8cef-9d217749574d
ChooseDisplayMode()

# ╔═╡ 54da8c6e-c93c-11f1-af5b-2355f654e538
html"""
	<h1 style="text-align:center">
		Graduate Program - Biomedical MRI
	</h1>
	<div style="text-align:center">
		<p style="font-weight:bold; font-size: 35px; font-variant: small-caps; margin: 0px">
			Lesson 3-2: Advanced reconstruction
		</p>
		<p style="font-size: 30px; font-variant: small-caps; margin: 0px">
			Aurélien Trotier
		</p>
		<p style="font-size: 20px;">
			CRMSB - Université de Bordeaux / CNRS
		</p>
	</div>
"""

# ╔═╡ 54da8cbe-c93c-11f1-bdd6-d574ba01661e
md"""
# 1. Introduction

In the *basic reconstruction* practical, we saw that skipping k-space lines reduces the FOV and creates **fold-over**, and that the PSF of the sampling pattern describes the artefact. With a single coil, the missing information is lost.

With a **multi-channel coil**, each coil sees the object through its own sensitivity $C_l(\mathbf{r})$:

$$y_l(\mathbf{k}) = \int C_l(\mathbf{r})\, m(\mathbf{r})\, e^{-i2\pi\mathbf{k}\cdot\mathbf{r}}\, d\mathbf{r} + n_l(\mathbf{k})$$

The coils provide additional **spatial encoding**, which can replace some of the gradient encoding. In this practical we use the same brain dataset (8 coils, fully sampled) and simulate accelerated acquisitions to study:

1. **GRAPPA**: fill the missing k-space lines using the correlations between the coils (k-space method),
2. **SENSE**: unfold the aliased image using the sensitivity maps (image-domain method),
3. **iterative SENSE**: write the acquisition with **linear operators** and solve a least-squares problem (works for any sampling pattern, Cartesian or **radial**),
4. **compressed sensing**: add a **sparsity** (wavelet) regularization to reconstruct from even fewer samples.
"""

# ╔═╡ 54da8cd2-c93c-11f1-b75b-03162c86c6e8
md"""
# 2. The data

The file contains `kspace_8ch` (complex k-space of each coil, $N_x \times N_y \times N_c$) and `map_8ch` (the sensitivity maps). As in the basic practical, we assume a pixel size of 1 mm × 1 mm.

!!! warning "Path to the data"
    The file `kspace_brain_8ch.mat` is expected in `ressources/data/`. Change `data_path` below if you stored it somewhere else.
"""

# ╔═╡ 54da8ce4-c93c-11f1-aeb0-75a99f4e362d
data_path = joinpath(@__DIR__, "..", "ressources", "data", "kspace_brain_8ch.mat")

# ╔═╡ 54da8cf0-c93c-11f1-8dbe-e57c9d2fde44
begin
	vars = matread(data_path)
	kspace_8ch = vars["kspace_8ch"]
	maps_8ch = vars["map_8ch"]
	Nx, Ny, Nc = size(kspace_8ch)
	pix = (1.0, 1.0)          # pixel size (mm)
	FOV = (Nx, Ny) .* pix     # field of view (mm)
end;

# ╔═╡ 54da8d04-c93c-11f1-8ef1-697f4a8aa850
begin
	"Centered 2D inverse FFT along dimensions 1 and 2 : k-space → image"
	ifft2c(k) = fftshift(ifft(ifftshift(k, (1, 2)), (1, 2)), (1, 2))

	"Centered 2D FFT along dimensions 1 and 2 : image → k-space"
	fft2c(x) = fftshift(fft(ifftshift(x, (1, 2)), (1, 2)), (1, 2))
end

# ╔═╡ 54da8d0e-c93c-11f1-b3f2-df8514ad3fd6
md"""
The **reference** image is obtained from the fully sampled data with the sensitivity-weighted coil combination

$$m(x,y) = \frac{\sum_l C_l^*(x,y)\, m_l(x,y)}{\sum_l |C_l(x,y)|^2}$$

All the accelerated reconstructions will be compared to it. Note that it contains noise: an error of a few percent is unavoidable.
"""

# ╔═╡ 54da8d16-c93c-11f1-93de-9300e30ea956
begin
	img_8ch = ifft2c(kspace_8ch)
	img = sum(conj.(maps_8ch) .* img_8ch, dims=3)[:, :, 1] ./ (sum(abs2.(maps_8ch), dims=3)[:, :, 1] .+ eps())
	ksp = fft2c(img)
end;

# ╔═╡ 54da8d22-c93c-11f1-8bfe-4f63bf1f8774
md"""
Coil : $(@bind coil Slider(1:Nc, default=1, show_value=true))
"""

# ╔═╡ 54da8d48-c93c-11f1-a907-b3fddb4d9fff
md"""
# 3. GRAPPA

## 3.1 Coil sensitivities in k-space

In the basic practical we used a virtual single coil. With a multi-channel coil, the image seen by coil $l$ is the object multiplied by the sensitivity of the coil:

$$m_l(\mathbf{r}) = C_l(\mathbf{r})\; m(\mathbf{r}) \qquad\Longleftrightarrow\qquad K_l(\mathbf{k}) = \left(\hat{C}_l \circledast M\right)(\mathbf{k})$$

A product in the image domain is a **convolution** in k-space. The sensitivity maps are smooth (they vary slowly in space), so their Fourier transform $\hat{C}_l$ is concentrated on **a few samples** around $\mathbf{k}=0$: it is a small convolution kernel.
"""

# ╔═╡ 54da8d54-c93c-11f1-817f-2356cf14b3ec
md"""
Coil : $(@bind coil_g Slider(1:Nc, default=1, show_value=true))
"""

# ╔═╡ 54da8d72-c93c-11f1-b1c2-b337b195e653
md"""
Let's zoom on $|\hat{C}_l|$ for the 8 coils (±12 samples around the center). This is what happens to **a single k-space sample of the object**: in the k-space of each coil, its signal "bleeds" over its neighbors, with a different pattern (amplitude and phase) for each coil. The title gives the radius containing 95 % of the energy of $\hat{C}_l$.
"""

# ╔═╡ 54da8d7c-c93c-11f1-bae0-57709288b8de
let
	f = Figure(size=(1100, 560))
	kr = sqrt.((((1:Nx) .- (Nx ÷ 2 + 1)) .^ 2) .+ ((((1:Ny) .- (Ny ÷ 2 + 1)) .^ 2)'))
	order = sortperm(vec(kr))
	w = 12
	cx, cy = Nx ÷ 2 + 1, Ny ÷ 2 + 1
	for l in 1:Nc
		C_hat = fft2c(maps_8ch[:, :, l])
		energy = cumsum(vec(abs2.(C_hat))[order]) ./ sum(abs2, C_hat)
		r95 = vec(kr)[order][findfirst(>=(0.95), energy)]
		ax = Axis(f[(l-1)÷4+1, (l-1)%4+1], title="coil $l : 95 % within r = $(round(r95, digits=1))", aspect=DataAspect())
		heatmap!(ax, -w:w, -w:w, abs.(C_hat[cx-w:cx+w, cy-w:cy+w]), colormap=:inferno)
		hidedecorations!(ax)
	end
	f
end

# ╔═╡ 54da8d86-c93c-11f1-b8d9-698760ce6441
md"""
## 3.2 Why does GRAPPA work?

**Consequences:**

1. In each coil, a k-space sample is a **linear combination of its neighbors** in the "true" k-space $M$, with weights given by $\hat{C}_l$.
2. The 8 coils see the same $M$ through 8 different kernels: the coil k-spaces are **locally correlated**, and a sample of coil $l$ can be predicted from the neighboring samples of **all** the coils.
3. This relation is a convolution: it is the **same everywhere** in k-space (shift-invariant).

**GRAPPA** (Griswold et al., MRM 2002) exploits this: the weights of the linear combination (the *GRAPPA kernel*) are learned on a small fully sampled region in the center of the k-space (the **ACS** lines, *auto-calibration signal*), then applied everywhere to fill the missing lines:

$$K_l(k_x, k_y) = \sum_{c=1}^{N_c}\ \sum_{b \in \{k_y - o,\ k_y - o + R\}}\ \sum_{d=-1}^{1} w_{l,c,b,d}\; K_c(k_x + d,\ b)$$

Here, the kernel uses the 2 acquired lines surrounding the missing line (which is at an offset $o$ from the previous acquired line), 3 points along $k_x$ and all the coils.
"""

# ╔═╡ 54da8d9a-c93c-11f1-b83c-d53f82639a83
"""
    grappa(kus, acq, acs_lines; R=2)

GRAPPA reconstruction of a multi-coil k-space `kus` (Nx × Ny × Nc) undersampled along y.
- `acq` : `true` for the acquired lines (including the ACS lines)
- `acs_lines` : range of the fully sampled calibration lines
Kernel : the 2 acquired lines surrounding the missing one × 3 points along kx × all the coils.
"""
function grappa(kus, acq, acs_lines; R=2)
	Nx, Ny, Nc = size(kus)
	k = copy(kus)
	acs = kus[:, acs_lines, :]
	na = length(acs_lines)
	source(K, x, y, off) = vcat(vec(K[x-1:x+1, y-off, :]), vec(K[x-1:x+1, y-off+R, :]))

	for off in 1:R-1   # position of the missing line after the previous acquired line
		# 1. calibration : learn the weights on the ACS, where all the lines are known
		src = [source(acs, x, y, off) for y in 1+off:na-R+off for x in 2:Nx-1]
		tgt = [acs[x, y, :] for y in 1+off:na-R+off for x in 2:Nx-1]
		W = permutedims(reduce(hcat, src)) \ permutedims(reduce(hcat, tgt))   # least squares

		# 2. synthesis : fill the missing lines with the same weights
		for y in 1:Ny
			(!acq[y] && y - off >= 1 && y - off + R <= Ny && acq[y-off] && acq[y-off+R]) || continue
			for x in 2:Nx-1
				k[x, y, :] = transpose(source(k, x, y, off)) * W
			end
		end
	end
	return k
end

# ╔═╡ 54da8da4-c93c-11f1-b5ca-437a022c6e2f
md"""
Acceleration ``R`` : $(@bind R_grappa Select([2, 3, 4], default=3))

Number of ACS lines : $(@bind n_acs Slider(8:2:40, default=24, show_value=true))

Number of coils used : $(@bind nc_grappa Slider(1:Nc, default=Nc, show_value=true))
"""

# ╔═╡ 54da8df4-c93c-11f1-886b-3976f917b448
md"""
# 4. SENSE

## 4.1 Unfolding in the image domain

SENSE (Pruessmann et al., MRM 1999) works directly on the **folded images**. With one line out of $R$ along $y$, the image of coil $l$ is (basic practical, section 6):

$$m_{R,l}(x, y) = \sum_{s=0}^{R-1} C_l\!\left(x, y_s\right) m\!\left(x, y_s\right) \qquad y_s = y + s\,\frac{FOV_y}{R}$$

For each folded pixel, we have $N_c$ equations (one per coil) and $R$ unknowns (the $R$ superimposed pixels):

$$\underbrace{\begin{bmatrix} m_{R,1}(x,y) \\ \vdots \\ m_{R,N_c}(x,y) \end{bmatrix}}_{\mathbf{a}}
=
\underbrace{\begin{bmatrix}
C_1(x,y_0) & \cdots & C_1(x,y_{R-1}) \\
\vdots & & \vdots \\
C_{N_c}(x,y_0) & \cdots & C_{N_c}(x,y_{R-1})
\end{bmatrix}}_{C}
\underbrace{\begin{bmatrix} m(x,y_0) \\ \vdots \\ m(x,y_{R-1}) \end{bmatrix}}_{\mathbf{m}}$$

If $N_c \ge R$, it is solved in the least-squares sense: $\hat{\mathbf{m}} = (C^H C)^{-1} C^H \mathbf{a}$.

## 4.2 Noise amplification: the g-factor

The noise is amplified by the inversion. The SNR of the SENSE image is

$$SNR_{R} = \frac{SNR_{full}}{g\,\sqrt{R}} \qquad\text{with}\qquad g_i = \sqrt{\left[(C^HC)^{-1}\right]_{ii}\,\left[C^HC\right]_{ii}} \;\ge 1$$

$\sqrt{R}$ comes from the shorter acquisition, and the **geometry factor** $g$ from the conditioning of $C$: $g = 1$ when the coils "see" the superimposed pixels very differently, and $g \gg 1$ when their sensitivities are similar.
"""

# ╔═╡ 54da8e08-c93c-11f1-b397-5b6b6769f472
"""
    sense_unfold(img_fold, maps, R)

Cartesian SENSE along y. `img_fold` : folded coil images (Nx × Ny/R × Nc), `maps` : sensitivities (Nx × Ny × Nc).
Returns the unfolded image and the g-factor map.
"""
function sense_unfold(img_fold, maps, R)
	Nx, Ny, Nc = size(maps)
	n = Ny ÷ R
	start = Ny ÷ 2 + 1 - n ÷ 2        # line of the full FOV corresponding to the first line of the reduced FOV
	m = zeros(ComplexF64, Nx, Ny)
	g = zeros(Nx, Ny)
	for j in 1:n, x in 1:Nx
		# the R pixels of the full FOV which fold onto the pixel (x, j)
		y0 = start + j - 1
		ys = [mod(y0 - 1 - s * n, Ny) + 1 for s in 0:R-1]
		ys = filter(y -> any(!iszero, maps[x, y, :]), ys)   # only the pixels inside the object
		isempty(ys) && continue
		C = transpose(maps[x, ys, :])                       # Nc × (number of superimposed pixels)
		CᴴC = C' * C
		m[x, ys] = CᴴC \ (C' * img_fold[x, j, :])
		g[x, ys] = sqrt.(real.(diag(inv(CᴴC)) .* diag(CᴴC)))
	end
	return m, g
end

# ╔═╡ 54da8e12-c93c-11f1-a970-c142d744ad5f
md"""
Acceleration ``R`` (along y) : $(@bind R_sense Select([2, 4, 5], default=4))
"""

# ╔═╡ 54da8e1e-c93c-11f1-9567-c57586f8ffe5
sense_res = let
	c = Ny ÷ 2 + 1
	lines = findall(y -> mod(y - c, R_sense) == 0, 1:Ny)   # acquired lines (including k = 0)
	img_fold = ifft2c(kspace_8ch[:, lines, :])             # folded coil images, FOV_y / R
	m, g = sense_unfold(img_fold, maps_8ch, R_sense)
	(img_fold=img_fold, m=m, g=g)
end;

# ╔═╡ 54da8e4c-c93c-11f1-b9db-716aa2b57f7f
md"""
# 5. Iterative SENSE with linear operators

## 5.1 The encoding operator

All the acquisitions can be written as a **linear model**:

$$\mathbf{y} = E\,\mathbf{x} + \mathbf{n} \qquad\text{with}\qquad E = P\,F\,S$$

-  $\mathbf{x}$: the image we want ($N = N_x N_y$ unknowns),
-  $S$: **coil sensitivities** — multiplies the image by each $C_l$ ($N \to N N_c$),
-  $F$: **Fourier transform** of each coil image ($N N_c \to N N_c$),
-  $P$: **sampling mask** — keeps only the acquired k-space samples ($N N_c \to M$),
-  $\mathbf{y}$: the $M$ acquired samples, $\mathbf{n}$ the noise.

$E$ is a huge matrix: for $R = 4$, $M \times N \approx 70\,000 \times 35\,000$ complex numbers ≈ 40 GB! We never build it. We only need **functions** that compute $E\mathbf{x}$ and the adjoint $E^H\mathbf{y}$. `LinearOperators.jl` wraps such functions into objects that behave like matrices (product `*`, adjoint `'`, composition...). This is how MRIReco.jl (and BART) work.
"""

# ╔═╡ 54da8e62-c93c-11f1-845a-d3c11184a704
"Build a `LinearOperator` (m × n) from a forward function `f` and its adjoint `fadj`"
op(f, fadj, m, n) = LinearOperator(ComplexF64, m, n, false, false,
	(res, v) -> (res .= f(v)), nothing, (res, v) -> (res .= fadj(v)))

# ╔═╡ 54da8e76-c93c-11f1-911e-a1c36ab41f54
begin
	Npix = Nx * Ny

	# S : image → coil images (multiplication by the sensitivities)
	S_op = op(x -> vec(maps_8ch .* reshape(x, Nx, Ny)),
		y -> vec(sum(conj.(maps_8ch) .* reshape(y, Nx, Ny, Nc), dims=3)), Npix * Nc, Npix)

	# F : unitary centered FFT of each coil image (Fᴴ = F⁻¹)
	F_op = op(x -> vec(fft2c(reshape(x, Nx, Ny, Nc))) ./ sqrt(Npix),
		y -> vec(ifft2c(reshape(y, Nx, Ny, Nc))) .* sqrt(Npix), Npix * Nc, Npix * Nc)
end;

# ╔═╡ 54da8e7e-c93c-11f1-b39f-f3534cf7f871
md"""
### Sampling masks

Iterative reconstruction is not limited to regular Cartesian undersampling. We will use three masks: regular lines, random lines, and a 2D **variable-density Poisson-disc** mask with a fully sampled center (the typical mask for the two phase-encoding directions of a 3D acquisition, and for compressed sensing):
"""

# ╔═╡ 54da8e8a-c93c-11f1-a168-099deb42c77c
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

# ╔═╡ 54da8e9e-c93c-11f1-8583-b3297c716647
md"""
Sampling mask : $(@bind mask_choice Select(["poisson" => "Poisson disc + center (2D)", "random1d" => "random lines + center (1D)", "regular" => "regular lines + 16 ACS lines (1D)"]))

Acceleration ``R`` : $(@bind R_it Select([2, 3, 4, 6, 8, 12], default=4))
"""

# ╔═╡ 54da8ed0-c93c-11f1-9e81-3b1966ff321f
md"""
## 5.2 Iterative reconstruction: CG-SENSE

The simplest reconstruction is the adjoint $\hat{\mathbf{x}} = E^H\mathbf{y}$ (zero-filling, inverse FFT, coil combination with $C_l^*$). It is only correct if $E^HE = I$, which is not the case with undersampling.

Instead, we look for the image which best explains the acquired data:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}}\ \|E\mathbf{x} - \mathbf{y}\|_2^2 + \lambda\|\mathbf{x}\|_2^2
\quad\Longleftrightarrow\quad
(E^H E + \lambda I)\,\hat{\mathbf{x}} = E^H \mathbf{y}$$

$E^HE + \lambda I$ is Hermitian positive (semi-)definite: this linear system is solved with the **conjugate gradient** (CG) algorithm, which only needs to apply $E$ and $E^H$ at each iteration (Pruessmann et al., MRM 2001: *CG-SENSE*). Here is a complete implementation:
"""

# ╔═╡ 54da8ee4-c93c-11f1-b3d1-6d111f205923
"""
    cg_sense(E, y; iters=20, λ=0.0, callback=x -> nothing)

Conjugate gradient on the normal equations (EᴴE + λI) x = Eᴴy.
Returns the image and the relative residual at each iteration.
`callback(x)` is called after each iteration (e.g. to monitor the error).
"""
function cg_sense(E, y; iters=20, λ=0.0, callback=x -> nothing)
	b = E' * y
	x = zeros(ComplexF64, length(b))
	r = copy(b)                 # residual b - A x, with x = 0
	p = copy(r)                 # search direction
	rr = real(dot(r, r))
	res = [sqrt(rr)]
	for _ in 1:iters
		Ap = E' * (E * p) .+ λ .* p
		α = rr / real(dot(p, Ap))
		x .+= α .* p
		r .-= α .* Ap
		rr_new = real(dot(r, r))
		p .= r .+ (rr_new / rr) .* p
		rr = rr_new
		push!(res, sqrt(rr))
		callback(x)
	end
	return x, res ./ res[1]
end

# ╔═╡ 54da8eee-c93c-11f1-b8a1-f38901d15ad0
md"""
Number of CG iterations : $(@bind n_iter Slider(1:60, default=15, show_value=true))

Regularization ``\lambda`` : $(@bind λ_cg Select([0.0, 0.1, 1.0, 10.0]))
"""

# ╔═╡ 54da8f34-c93c-11f1-b1e8-fbbac5095288
md"""
## 5.3 Non-Cartesian CG-SENSE: just change the Fourier operator

Nothing in the CG-SENSE algorithm depends on the Cartesian sampling: it only needs to apply $E$ and $E^H$. For a **radial** acquisition, we keep the coil operator $S$ and the solver, and only replace $P\,F$ (FFT + mask) by a **NUFFT** operator $\mathcal{N}$, applied to each coil image:

$$E_{cart} = P\,F\,S \qquad\longrightarrow\qquad E_{radial} = W^{1/2}\,\mathcal{N}\,S$$

$W$ is the **density compensation** (DCF, see the basic practical, section 8). In the iterative reconstruction it is used as a **weighting** of the least-squares problem, $\min_\mathbf{x}\|W^{1/2}(\mathcal{N}S\mathbf{x} - \mathbf{y})\|_2^2$: it acts as a **preconditioner** which makes all the spatial frequencies converge at the same speed (without it, the over-sampled center of the k-space dominates).

We simulate a multi-coil radial acquisition by applying the forward NUFFT to the fully sampled **coil images** (so the data contain the real noise of each coil).
"""

# ╔═╡ 54da8f3e-c93c-11f1-a609-55bb1df4808f
begin
	"Radial trajectory : `n_spokes` spokes of `Nread` samples over 180°, in cycles/pixel (k_max = 0.5)"
	function radial_nodes(n_spokes, Nread)
		r = ((0:Nread-1) .- Nread ÷ 2) ./ Nread
		θ = π .* (0:n_spokes-1) ./ n_spokes
		return permutedims(hcat(vec(r .* cos.(θ)'), vec(r .* sin.(θ)')))
	end

	"Analytical DCF of the radial trajectory (ramp |k|)"
	function ramp_dcf(n_spokes, Nread)
		r = ((0:Nread-1) .- Nread ÷ 2) ./ Nread
		return vec(max.(abs.(r), 1 / (4Nread)) .* ones(n_spokes)') .* (π / (n_spokes * Nread))
	end
end

# ╔═╡ 54da8f54-c93c-11f1-812d-f12292c2ded8
"""
    nufft_op(p, w)

NUFFT operator for all the coils, weighted by √w : coil images (Nx·Ny·Nc) → radial samples (M·Nc).
This is the only operator that changes compared to the Cartesian case (`F_op` + `P_op`).
"""
function nufft_op(p, w)
	sw = sqrt.(w)
	M = length(w)
	forward(x) = vec(reduce(hcat, [sw .* (p * reshape(x, Nx, Ny, Nc)[:, :, l]) for l in 1:Nc]))
	adjoint_(y) = vec(cat([adjoint(p) * (sw .* reshape(y, M, Nc)[:, l]) for l in 1:Nc]...; dims=3))
	return op(forward, adjoint_, M * Nc, Npix * Nc)
end

# ╔═╡ 54da8f5c-c93c-11f1-999f-b5a9ea9fd3dd
md"""
Number of spokes : $(@bind n_spokes_it Slider(2:8:256, default=64, show_value=true)) (Nyquist : ≈ 403 spokes)

Number of CG iterations : $(@bind n_iter_rad Slider(1:40, default=8, show_value=true))
"""

# ╔═╡ 54da8fb4-c93c-11f1-bc9e-73366be25a02
md"""
# 6. Compressed sensing

## 6.1 Images are compressible

An MRI image is **compressible**: in a suitable basis (e.g. **wavelets**), most of the coefficients are close to zero, and the image can be described with a small fraction of them (this is the principle of JPEG 2000). Let's keep only the largest wavelet coefficients of the reference image.

The wavelet transform (Daubechies 4, 4 levels, `Wavelets.jl`) needs a size divisible by $2^4$: the image is zero-padded to 256 × 256. The transform is **orthogonal**, so $\Psi^H\Psi = I$.
"""

# ╔═╡ 54da8fc0-c93c-11f1-9c06-83ab1ec9435e
begin
	wt = wavelet(WT.db4)        # Daubechies 4 wavelet
	wl_levels = 4
	wl_size = (256, 256)        # zero-padded size, divisible by 2^levels

	"Wavelet coefficients of an image (zero-padded to `wl_size`)"
	function wl_forward(x)
		p = zeros(ComplexF64, wl_size)
		p[1:Nx, 1:Ny] .= reshape(x, Nx, Ny)
		return vec(dwt(p, wt, wl_levels))
	end

	"Image from wavelet coefficients (inverse transform + crop)"
	wl_adjoint(w) = vec(idwt(reshape(w, wl_size), wt, wl_levels)[1:Nx, 1:Ny])

	# Ψᴴ : wavelet coefficients → image
	Ψᴴ = op(wl_adjoint, wl_forward, Nx * Ny, prod(wl_size))
end;

# ╔═╡ 54da8fca-c93c-11f1-a71a-b127ec7d5448
md"""
Fraction of the wavelet coefficients kept : $(@bind wl_keep Select([0.02, 0.05, 0.1, 0.2, 0.5], default=0.1))
"""

# ╔═╡ 54da9006-c93c-11f1-a83c-235ec5f31cad
md"""
## 6.2 Reconstruction with a sparsity constraint

Compressed sensing (Lustig et al., MRM 2007) combines three ingredients:

1. a **compressible** image (sparse in a transform $\Psi$),
2. an **incoherent** sampling: the undersampling artefact must look like noise (random or Poisson-disc sampling, see the PSF in the basic practical),
3. a **non-linear** reconstruction that promotes sparsity:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}}\ \frac{1}{2}\|E\mathbf{x} - \mathbf{y}\|_2^2 + \lambda\|\Psi\mathbf{x}\|_1$$

The $\ell_1$ norm is not differentiable: CG cannot be used. We write the image with its wavelet coefficients, $\mathbf{x} = \Psi^H\mathbf{w}$, and solve for $\mathbf{w}$ with **FISTA** (Beck & Teboulle 2009), a proximal gradient algorithm. Each iteration is

1. a **gradient step** on the data consistency: $\mathbf{u} = \mathbf{z} - \frac{1}{L} A^H(A\mathbf{z} - \mathbf{y})$, with $A = E\,\Psi^H$,
2. a **soft-thresholding** of the coefficients (proximal operator of the $\ell_1$ norm): $\mathbf{w} = \mathbf{u}\, \max\!\left(0, 1 - \frac{\lambda / L}{|\mathbf{u}|}\right)$,
3. a **momentum** step (extrapolation between $\mathbf{w}$ and its previous value).

The step $1/L$ uses the Lipschitz constant $L = \|A\|^2$ (largest eigenvalue of $A^HA$), estimated with a few power iterations.
"""

# ╔═╡ 54da9018-c93c-11f1-a0fe-0db6b4eab876
"Largest eigenvalue of AᴴA (power iteration)"
function power_iteration(A; iters=20)
	v = normalize(randn(MersenneTwister(1), ComplexF64, size(A, 2)))
	for _ in 1:iters
		v = normalize(A' * (A * v))
	end
	return real(dot(v, A' * (A * v)))
end

# ╔═╡ 54da9024-c93c-11f1-93c9-6b2bcbc3049a
"""
    fista(A, y; λ, L, iters=100)

Solve min_w ½‖A w - y‖² + λ‖w‖₁ with FISTA. `L` : Lipschitz constant ‖A‖².
"""
function fista(A, y; λ, L, iters=100)
	w = zeros(ComplexF64, size(A, 2))
	z = copy(w)
	t = 1.0
	for _ in 1:iters
		u = z .- (A' * (A * z .- y)) ./ L                                  # gradient step
		w_new = u .* max.(0, 1 .- (λ / L) ./ max.(abs.(u), eps()))          # soft-thresholding
		t_new = (1 + sqrt(1 + 4t^2)) / 2
		z = w_new .+ ((t - 1) / t_new) .* (w_new .- w)                       # momentum
		w, t = w_new, t_new
	end
	return w
end

# ╔═╡ 54da902e-c93c-11f1-8e29-511db98cbabf
md"""
The compressed sensing reconstruction uses the **same mask and data** as section 5 (choose the mask and $R$ there). $\lambda$ is given relative to $\max|A^H\mathbf{y}|$.

Outside the head the sensitivity maps are 0: the data do not constrain the image there, and the faint halo visible around the head only comes from the wavelet regularization. The errors are computed inside the head.

Regularization ``\lambda`` (relative) : $(@bind λ_cs Select([0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], default=3e-4))

FISTA iterations : $(@bind n_fista Slider(10:10:200, default=100, show_value=true))
"""

# ╔═╡ 54da906a-c93c-11f1-80b9-b7a34870bd46
md"""
!!! tip "Going further: MRIReco.jl"
    Everything written by hand in this practical is available in **MRIReco.jl**: the operators (`FFTOp`, `NFFTOp` for non-Cartesian data, `SensitivityOp`, the sampling operator, `WaveletOp`) are built with `LinearOperators.jl`, and **RegularizedLeastSquares.jl** provides the solvers (CG, FISTA, ADMM, ...) and the regularizations (ℓ2, ℓ1-wavelet, total variation...). A reconstruction is configured with a dictionary of parameters, roughly:

    ```julia
    params = Dict{Symbol,Any}()
    params[:reco] = "multiCoil"
    params[:reconSize] = (Nx, Ny)
    params[:senseMaps] = maps
    params[:solver] = FISTA
    params[:reg] = L1Regularization(λ)
    params[:sparseTrafo] = "Wavelet"
    params[:iterations] = 100
    img = reconstruction(acqData, params)
    ```

    (check the documentation of your MRIReco.jl version for the exact parameter names).
"""

# ╔═╡ 54da907e-c93c-11f1-a0fa-61a4eba1af9a
md"""
# 7. Summary

| Method | Domain | Calibration | Sampling | Limit |
|:--|:--|:--|:--|:--|
| GRAPPA | k-space: missing lines = linear combination of the neighbors of all the coils | ACS lines | regular (Cartesian) | noise / residual aliasing at high $R$ |
| SENSE | image: unfold $R$ superimposed pixels with the sensitivities | sensitivity maps | regular (Cartesian) | g-factor: $SNR_R = SNR/(g\sqrt{R})$ |
| Iterative SENSE | least squares $\min\|E\mathbf{x}-\mathbf{y}\|^2$ with $E = PFS$, solved by CG | sensitivity maps | **any** (random, radial with NUFFT + DCF...) | ill-conditioning → stop early or regularize |
| Compressed sensing | $\min \frac{1}{2}\|E\mathbf{x}-\mathbf{y}\|^2 + \lambda\|\Psi\mathbf{x}\|_1$, solved by FISTA | sensitivity maps + $\lambda$ | **incoherent** (Poisson disc, random) | choice of $\lambda$, loss of small details |
"""

# ╔═╡ 54da908a-c93c-11f1-bb44-0163e9446bd9
md"""
# Helper functions
"""

# ╔═╡ 54da909c-c93c-11f1-a7b7-ef3fc66604eb
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

	"Random selection of k-space lines (exact acceleration `R`) with `ncenter` central lines"
	function random_lines(Ny; R=4, ncenter=16, seed=1)
		c = Ny ÷ 2 + 1
		acquired = falses(Ny)
		acquired[c-ncenter÷2:c+ncenter÷2-1] .= true
		others = shuffle(MersenneTwister(seed), findall(!, acquired))
		acquired[others[1:max(0, round(Int, Ny / R) - ncenter)]] .= true
		return acquired
	end

	"Sum-of-squares of multi-coil images"
	sos_img(x) = sqrt.(dropdims(sum(abs2, x, dims=3), dims=3))

	"Sum-of-squares image of a multi-coil k-space"
	sos(k) = sqrt.(dropdims(sum(abs2.(ifft2c(k)), dims=3), dims=3))

	"Normalized root mean square error"
	nrmse(a, ref) = norm(a .- ref) / norm(ref)

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

# ╔═╡ 54da8d36-c93c-11f1-8b23-a3dbb0ef61d2
let
	f = Figure(size=(1100, 330))
	show_img!(f[1, 1], abs.(img_8ch[:, :, coil]); title="coil $(coil) : magnitude")
	show_img!(f[1, 2], angle.(img_8ch[:, :, coil]); title="coil $(coil) : phase", colormap=:twilight)
	show_img!(f[1, 3], abs.(maps_8ch[:, :, coil]); title="sensitivity map", colormap=:viridis)
	show_img!(f[1, 4], abs.(img); title="reference (combined)")
	f
end

# ╔═╡ 54da8d5e-c93c-11f1-829c-15665bf7805c
let
	C = maps_8ch[:, :, coil_g]
	C_hat = fft2c(C)
	f = Figure(size=(1100, 760))
	show_img!(f[1, 1], abs.(img); title="object m")
	show_img!(f[1, 2], abs.(C); title="×   sensitivity of coil $(coil_g) : C")
	show_img!(f[1, 3], abs.(C .* img); title="=   coil image : C · m")
	show_ksp!(f[2, 1], ksp; title="M = FT(m)")
	ax = Axis(f[2, 2]; title="⊛   Ĉ = FT(C)", aspect=DataAspect(), xlabel="i_kx", ylabel="i_ky")
	heatmap!(ax, (1:Nx) .- (Nx ÷ 2 + 1), (1:Ny) .- (Ny ÷ 2 + 1), log10.(abs.(C_hat) ./ maximum(abs, C_hat) .+ 1e-8); colormap=:inferno, colorrange=(-5, 0))
	show_ksp!(f[2, 3], fft2c(C .* img); title="=   coil k-space : K = Ĉ ⊛ M")
	f
end

# ╔═╡ 54da8dae-c93c-11f1-988a-510dc142db4c
grappa_res = let
	k_coils = kspace_8ch[:, :, 1:nc_grappa]
	c = Ny ÷ 2 + 1
	acq = [mod(y - c, R_grappa) == 0 for y in 1:Ny]
	acs = (c-n_acs÷2):(c+n_acs÷2-1)
	acq[acs] .= true

	k_us = k_coils .* reshape(acq, 1, Ny, 1)
	k_g = grappa(k_us, acq, acs; R=R_grappa)

	ref, im_us, im_g = sos(k_coils), sos(k_us), sos(k_g)
	(k_us=k_us, k_grappa=k_g, img_ref=ref, img_us=im_us, img_grappa=im_g,
		R_eff=Ny / count(acq), err_us=nrmse(im_us, ref), err_grappa=nrmse(im_g, ref))
end;

# ╔═╡ 54da8dba-c93c-11f1-8081-c1271096db05
let
	g = grappa_res
	f = Figure(size=(1300, 440))
	show_ksp!(f[1, 1], g.k_us[:, :, 1]; title="acquired k-space (coil 1)")
	show_img!(f[1, 2], g.img_us; title="zero-filled SoS — NRMSE $(pct(g.err_us))")
	show_ksp!(f[1, 3], g.k_grappa[:, :, 1]; title="GRAPPA k-space (coil 1)")
	show_img!(f[1, 4], g.img_grappa; title="GRAPPA SoS — NRMSE $(pct(g.err_grappa))")
	show_img!(f[1, 5], 5 .* abs.(g.img_grappa .- g.img_ref); title="|GRAPPA − reference| × 5", colorrange=(0, maximum(g.img_ref)))
	Label(f[0, :], "R = $(R_grappa), $(n_acs) ACS lines → effective acceleration $(round(g.R_eff, digits=2)), $(nc_grappa) coil(s)", fontsize=16)
	f
end

# ╔═╡ 54da8e30-c93c-11f1-a952-51438a32940d
let
	s = sense_res
	f = Figure(size=(1300, 450))
	show_img!(f[1, 1], sos_img(s.img_fold); fov=(FOV[1], FOV[2] / R_sense), title="folded (SoS), FOV_y = $(FOV[2] / R_sense) mm")
	show_img!(f[1, 2], abs.(s.m); title="SENSE — error $(pct(rel_err(s.m, img)))")
	show_img!(f[1, 3], abs.(img); title="reference")
	show_img!(f[1, 4], 5 .* abs.(s.m .- img) .* head_mask; title="|SENSE − reference| × 5", colorrange=(0, maximum(abs, img)))
	ax = show_img!(f[1, 5], s.g; title="g-factor (max $(round(maximum(s.g), digits=1)))", colormap=:viridis, colorrange=(1, max(2, quantile(s.g[head_mask], 0.99))))
	Colorbar(f[1, 6], colormap=:viridis, limits=(1, max(2, quantile(s.g[head_mask], 0.99))))
	f
end

# ╔═╡ 54da8ea8-c93c-11f1-bdf9-1baa03fd9dca
begin
	mask_it = let
		if mask_choice == "poisson"
			poisson_mask((Nx, Ny); R=R_it)
		elseif mask_choice == "random1d"
			repeat(permutedims(random_lines(Ny; R=R_it, ncenter=16)), Nx, 1)
		else
			c = Ny ÷ 2 + 1
			acquired = [mod(y - c, R_it) == 0 || abs(y - c) < 8 for y in 1:Ny]
			repeat(permutedims(acquired), Nx, 1)
		end
	end

	# P : keep the acquired samples of every coil
	P_op = opRestriction(findall(vec(repeat(mask_it, 1, 1, Nc))), Npix * Nc)

	# encoding operator and acquired data
	E = P_op * F_op * S_op
	y_it = P_op * vec(kspace_8ch) ./ sqrt(Npix)
	R_eff_it = Npix / count(mask_it)
end;

# ╔═╡ 54da8ec6-c93c-11f1-bd02-4d6839dd2942
let
	x = randn(ComplexF64, size(E, 2))
	y = randn(ComplexF64, size(E, 1))
	(lhs=dot(E * x, y), rhs=dot(x, E' * y))
end

# ╔═╡ 54da8f02-c93c-11f1-b9d6-5d191ce3f186
begin
	x_adj = reshape(E' * y_it, Nx, Ny)
	x_cg_vec, res_cg = cg_sense(E, y_it; iters=n_iter, λ=λ_cg)
	x_cg = reshape(x_cg_vec, Nx, Ny)
end;

# ╔═╡ 54da9038-c93c-11f1-8384-9d9cfd41fcbf
begin
	A_cs = E * Ψᴴ                         # data ← image ← wavelet coefficients
	L_cs = power_iteration(A_cs)
	λ_abs = λ_cs * maximum(abs, A_cs' * y_it)
	w_cs = fista(A_cs, y_it; λ=λ_abs, L=L_cs, iters=n_fista)
	x_cs = reshape(Ψᴴ * w_cs, Nx, Ny)
end;

# ╔═╡ 54da8f0c-c93c-11f1-8317-29e6c30a6170
let
	f = Figure(size=(1300, 760))
	ax = Axis(f[1, 1], title="mask — effective R = $(round(R_eff_it, digits=2))", aspect=DataAspect())
	heatmap!(ax, mask_it, colormap=:grays)
	hidedecorations!(ax)
	show_img!(f[1, 2], abs.(x_adj); title="adjoint Eᴴy — error $(pct(rel_err(x_adj, img)))")
	show_img!(f[1, 3], abs.(x_cg); title="CG-SENSE $(n_iter) it. — error $(pct(rel_err(x_cg, img)))")
	show_img!(f[1, 4], abs.(img); title="reference (fully sampled)")

	ax = Axis(f[2, 1:2], title="convergence", xlabel="iteration", ylabel="relative residual", yscale=log10)
	scatterlines!(ax, 0:n_iter, res_cg)
	show_img!(f[2, 3], 5 .* abs.(x_cg .- img) .* head_mask; title="|CG-SENSE − reference| × 5", colorrange=(0, maximum(abs, img)))
	show_img!(f[2, 4], dropdims(sum(abs2, maps_8ch, dims=3), dims=3); title="Σ|C_l|²", colormap=:viridis)
	f
end

# ╔═╡ 54da8f66-c93c-11f1-863a-5d9628192d88
radial_res = let
	Nread = 256
	nodes = radial_nodes(n_spokes_it, Nread)
	p = plan_nfft(nodes, (Nx, Ny))
	w = ramp_dcf(n_spokes_it, Nread)

	# multi-coil radial "acquisition" : forward NUFFT of each fully sampled coil image
	data = reduce(hcat, [p * img_8ch[:, :, l] for l in 1:Nc])        # M × Nc

	# gridding : adjoint NUFFT with DCF of each coil, then coil combination
	# (= Eᴴ y with DCF, normalized by Σ|C_l|² as in the basic coil combination)
	E_dcf = nufft_op(p, w) * S_op
	y_dcf = vec(sqrt.(w) .* data)
	x_grid = reshape(E_dcf' * y_dcf, Nx, Ny) ./ max.(dropdims(sum(abs2, maps_8ch, dims=3), dims=3), eps())

	# CG-SENSE with and without DCF : same solver as in section 5.2 !
	err_dcf, err_nodcf = Float64[], Float64[]
	x_dcf, _ = cg_sense(E_dcf, y_dcf; iters=n_iter_rad, callback=x -> push!(err_dcf, rel_err(reshape(x, Nx, Ny), img)))
	E_nodcf = nufft_op(p, ones(length(w))) * S_op
	x_nodcf, _ = cg_sense(E_nodcf, vec(data); iters=n_iter_rad, callback=x -> push!(err_nodcf, rel_err(reshape(x, Nx, Ny), img)))

	(x_grid=x_grid, x_dcf=reshape(x_dcf, Nx, Ny), x_nodcf=reshape(x_nodcf, Nx, Ny), err_dcf=err_dcf, err_nodcf=err_nodcf)
end;

# ╔═╡ 54da8f86-c93c-11f1-8f88-c7b9d75e0b4b
let
	r = radial_res
	f = Figure(size=(1300, 760))
	show_img!(f[1, 1], abs.(r.x_grid); title="gridding (adjoint + DCF) — error $(pct(rel_err(r.x_grid, img)))")
	show_img!(f[1, 2], abs.(r.x_nodcf); title="CG-SENSE without DCF, $(n_iter_rad) it. — $(pct(rel_err(r.x_nodcf, img)))")
	show_img!(f[1, 3], abs.(r.x_dcf); title="CG-SENSE with DCF, $(n_iter_rad) it. — $(pct(rel_err(r.x_dcf, img)))")
	show_img!(f[1, 4], abs.(img); title="reference (fully sampled Cartesian)")
	ax = Axis(f[2, 1:2], title="error vs iteration ($(n_spokes_it) spokes)", xlabel="iteration", ylabel="relative error")
	scatterlines!(ax, 1:n_iter_rad, r.err_nodcf, label="without DCF")
	scatterlines!(ax, 1:n_iter_rad, r.err_dcf, label="with DCF (preconditioner)")
	hlines!(ax, [rel_err(r.x_grid, img)], color=:gray, linestyle=:dash, label="gridding")
	axislegend(ax)
	show_img!(f[2, 3], 5 .* abs.(r.x_grid .- img) .* head_mask; title="|gridding − reference| × 5", colorrange=(0, maximum(abs, img)))
	show_img!(f[2, 4], 5 .* abs.(r.x_dcf .- img) .* head_mask; title="|CG-SENSE − reference| × 5", colorrange=(0, maximum(abs, img)))
	f
end

# ╔═╡ 54da8fd4-c93c-11f1-a9b2-63906804daf7
let
	w = Ψᴴ' * vec(img)                                    # wavelet coefficients of the reference
	thr = sort(abs.(w), rev=true)[round(Int, wl_keep * length(w))]
	w_kept = w .* (abs.(w) .>= thr)
	img_kept = reshape(Ψᴴ * w_kept, Nx, Ny)

	f = Figure(size=(1200, 450))
	ax = Axis(f[1, 1], title="|wavelet coefficients| (log)", aspect=DataAspect())
	heatmap!(ax, log10.(abs.(reshape(w, wl_size)) .+ 1e-3), colormap=:inferno)
	hidedecorations!(ax)
	ax = Axis(f[1, 2], title="sorted |coefficients|", xlabel="fraction of the coefficients", yscale=log10)
	ws = sort(abs.(w), rev=true)
	lines!(ax, (1:length(ws)) ./ length(ws), ws .+ 1e-6)
	vlines!(ax, [wl_keep], color=:red, linestyle=:dash)
	ylims!(ax, 1e-2, maximum(ws) * 2)
	show_img!(f[1, 3], abs.(img_kept); title="$(round(Int, 100wl_keep)) % of the coefficients — error $(pct(rel_err(img_kept, img)))")
	show_img!(f[1, 4], 5 .* abs.(img_kept .- img) .* head_mask; title="|difference| × 5", colorrange=(0, maximum(abs, img)))
	f
end

# ╔═╡ 54da904c-c93c-11f1-9c56-953c4f96df1e
let
	f = Figure(size=(1300, 760))
	ax = Axis(f[1, 1], title="mask — effective R = $(round(R_eff_it, digits=2))", aspect=DataAspect())
	heatmap!(ax, mask_it, colormap=:grays)
	hidedecorations!(ax)
	show_img!(f[1, 2], abs.(x_cg); title="CG-SENSE $(n_iter) it. — error $(pct(rel_err(x_cg, img)))")
	show_img!(f[1, 3], abs.(x_cs); title="CS wavelet — error $(pct(rel_err(x_cs, img)))\n$(round(100 * mean(abs.(w_cs) .> 0), digits=1)) % non-zero wavelet coefficients")
	show_img!(f[1, 4], abs.(img); title="reference (fully sampled)")
	show_img!(f[2, 2], 5 .* abs.(x_cg .- img) .* head_mask; title="|CG-SENSE − reference| × 5", colorrange=(0, maximum(abs, img)))
	show_img!(f[2, 3], 5 .* abs.(x_cs .- img) .* head_mask; title="|CS − reference| × 5", colorrange=(0, maximum(abs, img)))
	f
end

# ╔═╡ 54da90a6-c93c-11f1-8355-a79763784662
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

# ╔═╡ 54da8dd6-c93c-11f1-ad65-2597a5f84e2c
md"""
**Q1** Why is $\hat{C}_l$ concentrated around the center of the k-space? Over how many samples does the signal of one k-space sample "bleed"?

**Q2** Explain why a missing k-space sample can be estimated from the acquired neighboring samples of all the coils. Why are the weights learned in the center (ACS) valid everywhere in the k-space?

**Q3** Run GRAPPA with $R$ = 2, 3 and 4. How does the quality evolve? Why does the effective acceleration differ from $R$?

**Q4** Use a single coil. Does GRAPPA still work? Why? What is the minimal number of coils for a given $R$?

**Q5** Reduce the number of ACS lines to 8. What happens?
""" |> question

# ╔═╡ 54da8de0-c93c-11f1-ba44-976dc4f0e5c0
answer_folded(md"""
**Q1** The sensitivity varies slowly in space: it only contains **low spatial frequencies**, so its Fourier transform is compact (95 % of the energy within a radius of a few samples). Each k-space sample of the object is spread over this neighborhood in the k-space of every coil. The size of the GRAPPA kernel must match the support of $\hat{C}_l$. (The maps used here are set to 0 outside the head, which adds high frequencies: a true coil sensitivity is even smoother.)

**Q2** Each coil k-space is $M$ convolved with a different compact kernel. A missing sample of coil $l$ depends on $M$ in a small neighborhood, and this neighborhood is also "seen", through 8 different combinations, by the acquired samples of the 8 coils around the missing one. With enough coils, the missing value is (approximately) a linear combination of these acquired samples. Because the relation is a convolution, it is shift-invariant: the same weights apply everywhere, so they can be learned where all the data are known (ACS).

**Q3** $R = 2$: almost perfect. $R = 3$: slight residual aliasing and noise. $R = 4$: clear residual aliasing and noise amplification: the coils do not differ enough along $y$ to separate 4 superimposed pixels (*g-factor*). The ACS lines are acquired in addition to the regular lines, so the effective acceleration $N_y / N_{acquired}$ is lower than $R$.

**Q4** No. With one coil, each sample gives a single combination of $M$, and the missing samples cannot be recovered (1 equation for $R$ unknowns, as for the single-coil fold-over of the basic practical): the kernel only interpolates and the aliasing remains. Parallel imaging requires $N_c \ge R$, and in practice $N_c \gg R$ for a good g-factor.

**Q5** The kernel is learned with fewer equations, from the very center of the k-space only: the weights are poorly estimated (and sensitive to noise), so residual aliasing appears. Conversely, many ACS lines improve the calibration but reduce the effective acceleration.
""")

# ╔═╡ 54da8e3a-c93c-11f1-ac05-fbf2e91b057f
md"""
**Q6** How many equations and unknowns does each folded pixel give? What is the condition on the number of coils?

**Q7** Why do we only keep, in `sense_unfold`, the superimposed pixels located inside the object? What would happen otherwise?

**Q8** Compare the error and the g-factor for $R$ = 2, 4 and 5. Where is the g-factor highest? Relate it to the position of the coils.

**Q9** Compare SENSE and GRAPPA: in which domain do they work, and what calibration information do they need?
""" |> question

# ╔═╡ 54da8e44-c93c-11f1-b03b-bbdeac87405a
answer_folded(md"""
**Q6** $N_c$ equations (one per coil) and $R$ unknowns: we need $N_c \ge R$, and in practice $N_c$ clearly larger than $R$ to keep a good conditioning.

**Q7** Outside the object the sensitivity maps are 0 (and the signal is only noise): keeping these pixels would add unknowns that are not constrained by the data, making $C^HC$ singular or badly conditioned. Using the **support** of the object reduces the number of unknowns and improves the unfolding (this is also why the edges of the FOV have a lower g-factor).

**Q8** $R=2$: error ≈ 6 %, $g$ close to 1. $R=4$: error ≈ 25 %, $g$ up to ≈ 6. $R=5$: the noise explodes ($g$ up to ≈ 20). The g-factor is highest in the **center** of the head, far from all the coils, where the sensitivities of the coils are similar and smooth: the superimposed pixels cannot be distinguished.

**Q9** GRAPPA works in **k-space** and needs **ACS lines** (a fully sampled central region) to learn its kernel; it gives one image per coil (combined afterwards). SENSE works in the **image domain** and needs explicit **sensitivity maps** (from a calibration scan or estimated from the ACS, e.g. ESPIRiT); errors in the maps directly create artefacts. Both rely on the same physics: the spatial variation of the coil sensitivities.
""")

# ╔═╡ 54da8eb0-c93c-11f1-990e-f3130ff681bf
md"""
**Q10** Check that `E'` is really the adjoint of `E` (*dot test*): for any $\mathbf{x}$ and $\mathbf{y}$, $\langle E\mathbf{x}, \mathbf{y}\rangle = \langle \mathbf{x}, E^H\mathbf{y}\rangle$.
""" |> question

# ╔═╡ 54da8f16-c93c-11f1-b052-9712134da4cc
md"""
**Q11** Look at the adjoint reconstruction $E^H\mathbf{y}$ for the 3 masks. Why is it not a good image? Relate it to the PSF of the mask (basic practical, section 7) and to the map $\sum_l |C_l|^2$.

**Q12** Run CG-SENSE with 1, 5, 15 and 60 iterations. What is recovered first? What happens with many iterations at $R = 8$?

**Q13** For $R = 4$, compare the regular 1D mask and the Poisson disc 2D mask. Which one gives the lowest error? Why?

**Q14** What is the effect of $\lambda$?

**Q15** How would you modify $E$ to reconstruct the radial data of the basic practical (section 8 of basic reconstruction)? What would you add to perform a **compressed sensing** reconstruction?
""" |> question

# ╔═╡ 54da8f2a-c93c-11f1-8fa1-195c1476b01f
answer_folded(md"""
**Q11** $E^H\mathbf{y} = S^H F^H P^H \mathbf{y}$: the missing samples are replaced by zeros, so each coil image is convolved with the PSF of the mask (fold-over or incoherent artefacts), and the coil combination with $C_l^*$ weights the image by $\sum_l |C_l|^2$ (intensity shading). Even with a full k-space, $E^HE = \sum_l|C_l|^2 \neq I$: the adjoint is not the inverse.

**Q12** The first iterations recover the large structures (low frequencies, which carry most of the energy), then the details, and the artefacts decrease. The problem is ill-conditioned: after a while, CG starts fitting the noise, which is amplified (*semi-convergence*). With the Poisson mask, the error is minimal around 15 iterations (≈ 13 % at $R=4$) and increases again with 60 iterations (≈ 21 % at $R=4$, ≈ 43 % at $R=8$). We must stop early or regularize.

**Q13** The Poisson disc mask gives a lower error (≈ 13 % vs ≈ 17 % at 15 iterations), although the regular mask acquires more samples because of its ACS lines (effective $R = 3.3$). With a 2D undersampling, the aliasing is spread in both directions: each pixel is mixed with pixels far away in $x$ **and** $y$, where the coil sensitivities are very different, and the artefact is incoherent. With a 1D mask, the $R$ superimposed pixels are aligned along $y$ only, where the coils differ less: the problem is worse conditioned (higher g-factor).

**Q14** $\lambda$ penalizes the energy of the image (Tikhonov regularization): it limits the noise amplification and stabilizes the iterations, at the cost of a bias (lower intensity, residual artefacts) when it is too large. It is a trade-off between noise and artefacts/bias. For instance at $R = 8$ (regular mask, 60 iterations), $\lambda = 1$ keeps the error at ≈ 24 % instead of ≈ 44 %.

**Q15** For radial data, replace $P\,F$ by a NUFFT operator for each coil (forward: `p * x`, adjoint: `adjoint(p) * y`), weighted by $\sqrt{w}$ with the DCF as a preconditioner: this is done in section 5.3. For compressed sensing, add a sparsity term $\lambda\|\Psi\mathbf{x}\|_1$ ($\Psi$ = wavelets or finite differences / total variation): the problem is no longer quadratic and CG is replaced by a proximal algorithm (ISTA/FISTA, ADMM), with an incoherent sampling (Poisson disc).
""")

# ╔═╡ 54da8f98-c93c-11f1-a353-77e3c7271248
md"""
**Q16** Compare the code of `nufft_op` with `F_op` and `P_op`. Which parts of the reconstruction did not change?

**Q17** With 64 spokes, compare the gridding reconstruction and CG-SENSE (with DCF). What happened to the streaks? Why?

**Q18** Compare the convergence with and without DCF. Why does the DCF accelerate the convergence?

**Q19** Decrease the number of spokes to 32, then increase it to 256. How do the gridding and CG-SENSE errors evolve?
""" |> question

# ╔═╡ 54da8fa2-c93c-11f1-99f6-1319df48809e
answer_folded(md"""
**Q16** Only the Fourier part changes: the FFT + mask (`F_op`, `P_op`) is replaced by a NUFFT for each coil (`p * x` and `adjoint(p) * y`), weighted by $\sqrt{w}$. The coil operator `S_op`, the composition $E = \dots\, S$, the solver `cg_sense` and the comparison tools are exactly the same. This modularity is the main advantage of the operator formulation (and how MRIReco.jl or BART handle any trajectory).

**Q17** Gridding is a single application of $E^H$: the undersampling streaks remain (≈ 12 % error with 64 spokes). CG-SENSE enforces **data consistency** for all the coils at once: the coil sensitivities provide the missing information, and most of the streaks disappear (≈ 9 % after 5–10 iterations).

**Q18** Without DCF, the center of the k-space (crossed by all the spokes) dominates $E^HE$: the first iterations only recover a blurred image and CG needs ≈ 25 iterations to reach its best error. The DCF equalizes the contribution of all the spatial frequencies ($E^HE$ becomes closer to the identity): the best error is reached after only 5–10 iterations. It is a **preconditioner**. Beware: with the DCF, the error increases again after ≈ 10 iterations (semi-convergence, noise amplification, as in section 5.2): we must stop early or regularize.

**Q19** Gridding: ≈ 24 % (32 spokes), 12 % (64), 7 % (128), 6 % (256). CG-SENSE (best number of iterations): ≈ 14 %, 9 %, 7 %, 5 %. The gain of the iterative reconstruction is large when the data are strongly undersampled (streaks) and becomes small close to the Nyquist criterion, where gridding is already almost exact.
""")

# ╔═╡ 54da8fe6-c93c-11f1-aed0-e911b1c80686
md"""
**Q20** What fraction of the wavelet coefficients is needed to describe the image with an error below 10 %? What is lost first?
""" |> question

# ╔═╡ 54da8ff2-c93c-11f1-8b6c-65164c5a07c8
answer_folded(md"""
**Q20** About 10 % of the coefficients give an error ≈ 7 % (5 % → ≈ 13 %, 20 % → ≈ 3 %). The small coefficients mostly describe the **noise** and the finest textures: when they are removed, the image is first denoised, then the small details are blurred.
""")

# ╔═╡ 54da9056-c93c-11f1-8897-41a7a005ad1f
md"""
**Q21** With the Poisson-disc mask and $R = 8$, compare CG-SENSE (at its best number of iterations) and the CS reconstruction. Look at the error maps.

**Q22** Change $\lambda$. What happens when it is too small? Too large?

**Q23** Use the regular mask (1D) with $R = 8$. Does compressed sensing still work? Why?

**Q24** The error is computed against the fully sampled image, which is itself noisy. Why can this underestimate the quality of the CS reconstruction?
""" |> question

# ╔═╡ 54da9060-c93c-11f1-97b4-dd5312912684
answer_folded(md"""
**Q21** CS gives a lower error (≈ 14 % vs ≈ 18 % at $R = 8$; ≈ 11 % vs ≈ 15 % at $R = 6$; ≈ 18 % vs ≈ 22 % at $R = 12$). The CG-SENSE error map contains noise amplification and incoherent aliasing everywhere; the sparsity constraint removes most of this noise-like component.

**Q22** $\lambda = 0$: plain least squares (equivalent to CG-SENSE, but FISTA converges more slowly). Too small: noise and aliasing remain. Too large: too many coefficients are set to zero, the image becomes smooth and "blocky" (wavelet artefacts), and the small details and the contrast of small structures are lost. $\lambda$ is a trade-off between artefacts/noise and fidelity, usually tuned on a reference dataset.

**Q23** Much less: CS only improves the error slightly (≈ 21 % vs ≈ 24 % for CG-SENSE), mostly by denoising, and stays far behind CS with the Poisson mask (≈ 14 %), although the regular mask acquires more samples (effective $R \approx 5.4$ because of the ACS lines). The regular undersampling produces a **coherent** artefact (sharp copies of the object), which is as sparse as the object itself: the $\ell_1$ term cannot distinguish the replicas from the true image. Compressed sensing needs an **incoherent** sampling.

**Q24** CS also removes part of the **noise** present in the data: the CS image can be closer to the true (noise-free) object than the reference is, but this denoising appears as a "difference" with the noisy reference. A fair comparison would use a noise-free simulation or several repetitions.
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearOperators = "5c8ed15e-5a4c-59e4-a42b-c7e8811fb125"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"
NFFT = "efe261a4-0d2b-5849-be55-fc731d526b0d"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Wavelets = "29a6e085-ba6d-5f35-a997-948ac2efa89a"

[compat]
FFTW = "~1.10.0"
GLMakie = "~0.13.13"
HypertextLiteral = "~1.0.0"
LinearOperators = "~2.14.3"
MAT = "~0.12.1"
NFFT = "~0.14.3"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.83"
Wavelets = "~0.10.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "4d1b9ab6743267fc57eacd91cbbdd4d3ec0f1f7b"

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

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

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

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "a65cfc2999988f5ba09fc4bd8049e3ed914e5a04"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.6"
weakdeps = ["OffsetArrays"]

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

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

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

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

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

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

[[deps.LinearOperators]]
deps = ["FastClosures", "LinearAlgebra", "Printf", "SparseArrays", "TimerOutputs"]
git-tree-sha1 = "99a44fc6443bfe118cfe64ca059136bc85e64ccb"
uuid = "5c8ed15e-5a4c-59e4-a42b-c7e8811fb125"
version = "2.14.3"

    [deps.LinearOperators.extensions]
    LinearOperatorsAMDGPUExt = "AMDGPU"
    LinearOperatorsCUDAExt = "CUDA"
    LinearOperatorsChainRulesCoreExt = "ChainRulesCore"
    LinearOperatorsJLArraysExt = "JLArrays"
    LinearOperatorsLDLFactorizationsExt = "LDLFactorizations"
    LinearOperatorsMetalExt = "Metal"
    LinearOperatorsOpNormExt = ["Arpack", "TSVD", "GenericLinearAlgebra"]

    [deps.LinearOperators.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GenericLinearAlgebra = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    LDLFactorizations = "40e66cde-538c-5869-a4ad-c39174c6795b"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    TSVD = "9449cd9e-2762-5aa3-a617-5413e99d722e"

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

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "9aa2cf5b3a0f55ef0ac4f26c8b92cc6d08b06fbf"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.3"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"
    PolynomialsRecipesBaseExt = "RecipesBase"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "99fd8f32ab6728ff205cddd24e2d9ea6cbb498d3"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.0"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

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

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

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

[[deps.TimerOutputs]]
deps = ["ExprTools", "PrettyTables", "Printf", "Tables"]
git-tree-sha1 = "2244393f04554e24b54896b430b2ab3850a2232d"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "1.2.2"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

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

[[deps.Wavelets]]
deps = ["DSP", "LinearAlgebra", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "e50c5e157177002ebe6bd921596db92ae9e0c73b"
uuid = "29a6e085-ba6d-5f35-a997-948ac2efa89a"
version = "0.10.2"

    [deps.Wavelets.extensions]
    WaveletsGPUExt = ["GPUArrays", "KernelAbstractions"]

    [deps.Wavelets.weakdeps]
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"

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
# ╟─54d999bc-c93c-11f1-8cef-9d217749574d
# ╟─54da8c6e-c93c-11f1-af5b-2355f654e538
# ╟─54da8ca0-c93c-11f1-95ce-1dc2aba07a20
# ╟─54da8cbe-c93c-11f1-bdd6-d574ba01661e
# ╟─54da8cd2-c93c-11f1-b75b-03162c86c6e8
# ╠═54da8ce4-c93c-11f1-aeb0-75a99f4e362d
# ╠═54da8cf0-c93c-11f1-8dbe-e57c9d2fde44
# ╠═54da8d04-c93c-11f1-8ef1-697f4a8aa850
# ╟─54da8d0e-c93c-11f1-b3f2-df8514ad3fd6
# ╠═54da8d16-c93c-11f1-93de-9300e30ea956
# ╟─54da8d22-c93c-11f1-8bfe-4f63bf1f8774
# ╠═54da8d36-c93c-11f1-8b23-a3dbb0ef61d2
# ╟─54da8d48-c93c-11f1-a907-b3fddb4d9fff
# ╟─54da8d54-c93c-11f1-817f-2356cf14b3ec
# ╠═54da8d5e-c93c-11f1-829c-15665bf7805c
# ╟─54da8d72-c93c-11f1-b1c2-b337b195e653
# ╠═54da8d7c-c93c-11f1-bae0-57709288b8de
# ╟─54da8d86-c93c-11f1-b8d9-698760ce6441
# ╠═54da8d9a-c93c-11f1-b83c-d53f82639a83
# ╟─54da8da4-c93c-11f1-b5ca-437a022c6e2f
# ╠═54da8dae-c93c-11f1-988a-510dc142db4c
# ╠═54da8dba-c93c-11f1-8081-c1271096db05
# ╟─54da8dd6-c93c-11f1-ad65-2597a5f84e2c
# ╟─54da8de0-c93c-11f1-ba44-976dc4f0e5c0
# ╟─54da8df4-c93c-11f1-886b-3976f917b448
# ╠═54da8e08-c93c-11f1-b397-5b6b6769f472
# ╠═54da8e12-c93c-11f1-a970-c142d744ad5f
# ╠═54da8e1e-c93c-11f1-9567-c57586f8ffe5
# ╠═54da8e30-c93c-11f1-a952-51438a32940d
# ╟─54da8e3a-c93c-11f1-ac05-fbf2e91b057f
# ╟─54da8e44-c93c-11f1-b03b-bbdeac87405a
# ╠═54da8e4c-c93c-11f1-b9db-716aa2b57f7f
# ╠═54da8e62-c93c-11f1-845a-d3c11184a704
# ╠═54da8e76-c93c-11f1-911e-a1c36ab41f54
# ╟─54da8e7e-c93c-11f1-b39f-f3534cf7f871
# ╠═54da8e8a-c93c-11f1-a168-099deb42c77c
# ╟─54da8e9e-c93c-11f1-8583-b3297c716647
# ╠═54da8ea8-c93c-11f1-bdf9-1baa03fd9dca
# ╟─54da8eb0-c93c-11f1-990e-f3130ff681bf
# ╠═54da8ec6-c93c-11f1-bd02-4d6839dd2942
# ╟─54da8ed0-c93c-11f1-9e81-3b1966ff321f
# ╠═54da8ee4-c93c-11f1-b3d1-6d111f205923
# ╟─54da8eee-c93c-11f1-b8a1-f38901d15ad0
# ╠═54da8f02-c93c-11f1-b9d6-5d191ce3f186
# ╠═54da8f0c-c93c-11f1-8317-29e6c30a6170
# ╟─54da8f16-c93c-11f1-b052-9712134da4cc
# ╟─54da8f2a-c93c-11f1-8fa1-195c1476b01f
# ╟─54da8f34-c93c-11f1-b1e8-fbbac5095288
# ╠═54da8f3e-c93c-11f1-a609-55bb1df4808f
# ╠═54da8f54-c93c-11f1-812d-f12292c2ded8
# ╟─54da8f5c-c93c-11f1-999f-b5a9ea9fd3dd
# ╠═54da8f66-c93c-11f1-863a-5d9628192d88
# ╠═54da8f86-c93c-11f1-8f88-c7b9d75e0b4b
# ╟─54da8f98-c93c-11f1-a353-77e3c7271248
# ╟─54da8fa2-c93c-11f1-99f6-1319df48809e
# ╟─54da8fb4-c93c-11f1-bc9e-73366be25a02
# ╠═54da8fc0-c93c-11f1-9c06-83ab1ec9435e
# ╠═54da8fca-c93c-11f1-a71a-b127ec7d5448
# ╠═54da8fd4-c93c-11f1-a9b2-63906804daf7
# ╟─54da8fe6-c93c-11f1-aed0-e911b1c80686
# ╟─54da8ff2-c93c-11f1-8b6c-65164c5a07c8
# ╟─54da9006-c93c-11f1-a83c-235ec5f31cad
# ╠═54da9018-c93c-11f1-a0fe-0db6b4eab876
# ╠═54da9024-c93c-11f1-93c9-6b2bcbc3049a
# ╟─54da902e-c93c-11f1-8e29-511db98cbabf
# ╠═54da9038-c93c-11f1-8384-9d9cfd41fcbf
# ╠═54da904c-c93c-11f1-9c56-953c4f96df1e
# ╟─54da9056-c93c-11f1-8897-41a7a005ad1f
# ╟─54da9060-c93c-11f1-97b4-dd5312912684
# ╟─54da906a-c93c-11f1-80b9-b7a34870bd46
# ╟─54da907e-c93c-11f1-a0fa-61a4eba1af9a
# ╟─54da908a-c93c-11f1-bb44-0163e9446bd9
# ╟─54da909c-c93c-11f1-a7b7-ef3fc66604eb
# ╟─54da90a6-c93c-11f1-8355-a79763784662
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
