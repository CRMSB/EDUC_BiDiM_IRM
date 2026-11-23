### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 6d40c51a-e341-11f1-b726-dfca5c95d502
begin
	# Layout
	using PlutoUI
	using PlutoUI: Slider   # CairoMakie also exports a `Slider`
	using PlutoTeachingTools
	using HypertextLiteral

	# Plot (CairoMakie : no OpenGL needed)
	using CairoMakie

	# Sequence design, simulation, reconstruction
	using KomaMRI
	using KomaMRI.PulseDesigner
	using FFTW

	PlutoUI.TableOfContents()
end

# ╔═╡ 6d3fcda4-e341-11f1-bc12-59a517d73b55
ChooseDisplayMode()

# ╔═╡ 6d40c4ca-e341-11f1-8792-3d0d4367c887
html"""
	<h1 style="text-align:center">
		Graduate Program - Biomedical MRI
	</h1>
	<div style="text-align:center">
		<p style="font-weight:bold; font-size: 35px; font-variant: small-caps; margin: 0px">
			Lesson 4-4: Practical — writing an EPI sequence
		</p>
		<p style="font-size: 30px; font-variant: small-caps; margin: 0px">
			Aurélien Trotier
		</p>
		<p style="font-size: 20px;">
			CRMSB - Université de Bordeaux / CNRS
		</p>
	</div>
"""

# ╔═╡ 6d40c536-e341-11f1-88f0-11ddef63b873
md"""
!!! info "How to work on this practical"
    * This time, you write the sequence **on your own**: each section gives the goal of a step, you add the cells.
    * Use the notebooks **4-1** (Pulseq basics: events, dead times, timing, labels) and **4-3** (simulation and reconstruction) as references.
    * Check each step before going to the next one: `check_timing(seq, sys)`, `plot_seq`, `get_kspace` / `plot_kspace`, and compare with the figures of section 1.
    * The first execution of the notebook takes several minutes (compilation of KomaMRI).
"""

# ╔═╡ 6d40c54c-e341-11f1-9b88-717fd00ece5e
md"""
# 1. Introduction

## 1.1 Echo planar imaging (EPI)

In the gradient-echo sequence of lesson 4-1, each excitation acquires **one** k-space line: an image of ``N_y`` lines needs ``N_y`` TRs. In **EPI** (Mansfield, 1977), a **single excitation** acquires the **whole k-space** (single-shot EPI): after the excitation, the readout gradient is switched back and forth, and each lobe acquires one line.

* The readout gradient **alternates its sign** (``+G_x``, ``-G_x``, ``+G_x``...): the odd lines are acquired from ``-k_{x,max}`` to ``+k_{x,max}``, the even lines in the opposite direction.
* Between two readout lobes, a short phase-encoding gradient, the **blip**, moves the trajectory by ``+\Delta k_y`` to the next line.
* Before the train, the **prephasers** bring the trajectory to the first corner of k-space: ``-k_{x,max}`` along the readout (half the area of a readout lobe) and ``-k_{y,max}`` along the phase encoding.
* The time between two lines is the **echo spacing** ESP. **TE** is the time between the center of the RF pulse and the acquisition of the **center of k-space** (``k_x = k_y = 0``), i.e. the middle of line ``N_y/2+1``: it sets the contrast.

!!! warning "When is the ADC open?"
    In this practical, the ADC is open **only on the flat top** (plateau) of each readout lobe: the samples are then equally spaced in ``k_x``, as in the gradient-echo sequence of lesson 4-1. During the ramps, the trajectory still moves in ``k_x`` but no sample is acquired (this time is lost). Sampling on the ramps as well (**ramp sampling**) shortens the echo spacing but requires regridding the data: it is the bonus at the end of the practical.

## 1.2 Timing diagram

The figure shows a schematic single-shot gradient-echo EPI with ``N_y = 8`` lines. The color of each readout lobe matches the color of the corresponding line in the k-space figure of section 1.3.
"""

# ╔═╡ 6d40c568-e341-11f1-8c84-8d544e224da0
begin
	# schematic EPI (arbitrary units) shared by the two figures
	Ny_demo = 8                              # number of lines of the schematic
	ramp, flat, blipdur = 0.25, 1.5, 0.4     # readout ramp, readout flat top, blip block
	ro_dur = 2ramp + flat                    # readout block
	ESP_demo = ro_dur + blipdur              # echo spacing
	t0_demo = 3.0                            # start of the first readout block
	tro(i) = t0_demo + (i - 1) * ESP_demo    # start of the readout block of line i
	TE_demo = tro(Ny_demo ÷ 2 + 1) + ro_dur / 2   # center of the line k_y = 0
	line_colors = cgrad(:viridis, Ny_demo, categorical=true)
end;

# ╔═╡ 6d40c57e-e341-11f1-9861-bd12a1a601b9
let
	f = Figure(size=(1300, 700))
	ax = Axis(f[1, 1], xlabel="time (arbitrary units)", yticksvisible=false, yticklabelsvisible=false,
		title="Single-shot gradient-echo EPI (schematic, $(Ny_demo) lines)")
	hidespines!(ax, :l, :r, :t)
	ax.ygridvisible = false
	ax.xgridvisible = false

	rows = (RF=10.0, Gz=7.5, Gy=5.0, Gx=2.5, ADC=0.0)
	for (name, y) in pairs(rows)
		hlines!(ax, [y], color=(:gray, 0.6), linewidth=1)
		text!(ax, -2.2, y, text=string(name), align=(:right, :center), fontsize=16, font=:bold)
	end

	# trapezoid helper
	trap(a, b, A, y0; r=0.15) = ([a, a + r, b - r, b], y0 .+ [0, A, A, 0])
	function g!(a, b, A, row; color=:steelblue, r=0.15)
		x, y = trap(a, b, A, row; r)
		poly!(ax, Point2f.(x, y), color=(color, 0.5), strokecolor=color, strokewidth=1.5)
	end

	# ---- blocks
	tend = tro(Ny_demo) + ro_dur
	blocks = [(-1.2, 1.2, "excitation"), (1.2, 2.4, "prephasers"), (2.4, t0_demo, "TE\ndelay")]
	for i in 1:Ny_demo
		push!(blocks, (tro(i), tro(i) + ro_dur, "readout\n$(i)"))
		i < Ny_demo && push!(blocks, (tro(i) + ro_dur, tro(i) + ESP_demo, "blip"))
	end
	push!(blocks, (tend, tend + 1.4, "spoiler /\nTR delay"))
	for (a, b, name) in blocks
		isdelay = occursin("delay", name)
		poly!(ax, Rect(a, -3.4, b - a, 1.1), color=isdelay ? (:orange, 0.5) : (:gray, 0.15), strokecolor=:gray, strokewidth=1)
		text!(ax, (a + b) / 2, -2.85, text=name, align=(:center, :center), fontsize=10, rotation=name == "blip" ? π / 2 : 0.0)
	end
	text!(ax, -2.2, -2.85, text="blocks", align=(:right, :center), fontsize=16, font=:bold)

	# ---- RF
	t = range(-1, 1, 200)
	lines!(ax, t, rows.RF .+ 1.4 .* sinc.(2 .* t) .* (0.5 .+ 0.5 .* cos.(π .* t)), color=:black, linewidth=2)
	text!(ax, 0, rows.RF + 1.7, text="α", align=(:center, :bottom), fontsize=16)

	# ---- Gz : slice selection + rephaser
	g!(-1.2, 1.2, 0.8, rows.Gz)
	g!(1.3, 2.3, -0.9, rows.Gz)

	# ---- Gy : prephaser to -k_y,max, then blips
	g!(1.3, 2.3, -1.0, rows.Gy, color=:darkgreen)
	text!(ax, 1.8, rows.Gy - 1.2, text=L"-k_{y,\max}", align=(:center, :top), fontsize=12, color=:darkgreen)
	for i in 1:Ny_demo-1
		a = tro(i) + ro_dur
		g!(a, a + blipdur, 0.5, rows.Gy, color=:darkgreen, r=blipdur / 2)    # triangle
	end
	text!(ax, tro(1) + ro_dur + blipdur / 2, rows.Gy + 0.7, text=L"\text{blips } +\Delta k_y", align=(:left, :bottom), fontsize=12, color=:darkgreen)

	# ---- Gx : prephaser (-half area of a lobe), alternating readout lobes
	g!(1.3, 2.3, -0.75, rows.Gx, color=:firebrick)
	text!(ax, 1.8, rows.Gx - 0.95, text="−A/2", align=(:center, :top), fontsize=12, color=:firebrick)
	for i in 1:Ny_demo
		s = isodd(i) ? 1 : -1
		g!(tro(i), tro(i) + ro_dur, s * 0.75, rows.Gx, color=line_colors[i], r=ramp)
		text!(ax, tro(i) + ro_dur / 2, rows.Gx + s * 0.95, text=s > 0 ? "+A" : "−A", align=(:center, s > 0 ? :bottom : :top), fontsize=11)
	end

	# ---- ADC : flat top only, echo of each line
	for i in 1:Ny_demo
		a = tro(i)
		poly!(ax, Rect(a + ramp, rows.ADC - 0.25, flat, 0.5), color=(line_colors[i], 0.45), strokecolor=line_colors[i])
		for (c, d) in ((a, a + ramp), (a + ramp + flat, a + ro_dur))       # ramps : no ADC
			poly!(ax, Rect(c, rows.ADC - 0.25, d - c, 0.5), color=(:red, 0.12), strokecolor=(:red, 0.4))
		end
		amp = 0.25 + 1.3 * exp(-((i - (Ny_demo ÷ 2 + 1)) / 1.6)^2)
		tt = range(-flat / 2, flat / 2, 150)
		lines!(ax, a + ro_dur / 2 .+ tt, rows.ADC .+ amp .* sinc.(2.2 .* tt), color=:black, linewidth=1.2)
	end
	text!(ax, tro(1) + ro_dur / 2, rows.ADC - 0.45, text="ADC on the\nflat top", align=(:center, :top), fontsize=11)
	text!(ax, tro(2) + ramp / 2, rows.ADC - 0.45, text="ramps:\nno ADC", align=(:center, :top), fontsize=11, color=:red)
	text!(ax, TE_demo, rows.ADC + 1.65, text=L"k_y = 0", align=(:center, :bottom), fontsize=12)

	# ---- timing : TE and ESP
	ytop = rows.RF + 3.0
	arrows2d!(ax, [Point2f(TE_demo / 2, ytop)], [Vec2f(TE_demo / 2 - 0.1, 0)], tipwidth=8, tiplength=8, color=:black)
	arrows2d!(ax, [Point2f(TE_demo / 2, ytop)], [Vec2f(-TE_demo / 2 + 0.1, 0)], tipwidth=8, tiplength=8, color=:black)
	text!(ax, TE_demo / 2, ytop + 0.2, text="TE: RF center → center of k-space (middle of line $(Ny_demo ÷ 2 + 1))", align=(:center, :bottom), fontsize=13)
	c1, c2 = tro(6) + ro_dur / 2, tro(7) + ro_dur / 2
	yesp = rows.ADC - 1.3
	linesegments!(ax, [Point2f(c1, yesp - 0.2), Point2f(c1, rows.ADC), Point2f(c2, yesp - 0.2), Point2f(c2, rows.ADC)], color=:gray30, linestyle=:dot)
	arrows2d!(ax, [Point2f((c1 + c2) / 2, yesp)], [Vec2f((c2 - c1) / 2 - 0.05, 0)], tipwidth=7, tiplength=7, color=:gray30)
	arrows2d!(ax, [Point2f((c1 + c2) / 2, yesp)], [Vec2f(-(c2 - c1) / 2 + 0.05, 0)], tipwidth=7, tiplength=7, color=:gray30)
	text!(ax, (c1 + c2) / 2, yesp - 0.15, text="ESP", align=(:center, :top), fontsize=13, color=:gray30)
	arrows2d!(ax, [Point2f((tro(1) + tend) / 2, rows.RF - 1.0)], [Vec2f((tend - tro(1)) / 2 - 0.1, 0)], tipwidth=7, tiplength=7, color=:gray50)
	arrows2d!(ax, [Point2f((tro(1) + tend) / 2, rows.RF - 1.0)], [Vec2f(-(tend - tro(1)) / 2 + 0.1, 0)], tipwidth=7, tiplength=7, color=:gray50)
	text!(ax, (tro(1) + tend) / 2 - 3.0, rows.RF - 0.9, text=L"\text{readout train: } N_y \times \text{ESP}", align=(:center, :bottom), fontsize=12, color=:gray50)
	vlines!(ax, [0, TE_demo], color=(:black, 0.3), linestyle=:dot)

	xlims!(ax, -4.2, tend + 1.6)
	ylims!(ax, -3.8, ytop + 1.0)
	f
end

# ╔═╡ 6d40c592-e341-11f1-8211-ef27254f7262
md"""
Notes on the diagram:

* The readout prephaser is **negative** (gradient echo): its area is half the **total** area of a readout lobe (ramps included), so that ``k_x = 0`` is reached in the middle of each lobe.
* The blips are small triangles: their area is ``\Delta k_y = 1/FOV_y``. The phase-encoding prephaser has an area of ``-N_y/2 \cdot \Delta k_y``.
* In this first version, each readout lobe (gradient + ADC + label) and each blip are **separate blocks**. (In optimized EPI sequences, the blip is played during the ramps of the readout gradient.)
* The TE delay is placed **before** the train: TE is reached in the middle of the readout of line ``k_y = 0`` (line ``N_y/2+1``).

## 1.3 k-space trajectory

The same schematic in k-space. Each colored segment is a flat top (ADC open, dots = samples); the gray segments are the ramps (the trajectory moves, but nothing is acquired), the green arrows are the blips.
"""

# ╔═╡ 6d40c59c-e341-11f1-b549-97e6ffa22440
let
	Nx_demo = 8
	e = ramp / flat                         # k_x covered by one ramp (in units of k_x,max)
	kmax = 1.0
	kys = (-Ny_demo ÷ 2):(Ny_demo ÷ 2 - 1)
	xs = ((-Nx_demo ÷ 2):(Nx_demo ÷ 2 - 1)) ./ (Nx_demo ÷ 2)   # samples : sample Nx/2 + 1 at k_x = 0

	f = Figure(size=(900, 650))
	ax = Axis(f[1, 1], xlabel=L"k_x / k_{x,\max}", ylabel=L"k_y / \Delta k_y", 
		title="EPI k-space trajectory (flat-top sampling)", yticks=kys)

	# prephasers : from k = 0 to the first corner
	xstart = -(kmax + e)
	lines!(ax, [0, xstart], [0, kys[1]], color=:gray40, linestyle=:dash, linewidth=2)
	text!(ax, -0.97, kys[1] + 0.5, text="prephasers", align=(:left, :center), fontsize=13, color=:gray40)

	for (i, ky) in enumerate(kys)
		s = isodd(i) ? 1 : -1
		c = line_colors[i]
		# ramps (no ADC)
		lines!(ax, [-s * (kmax + e), -s * kmax], [ky, ky], color=:gray60, linewidth=3)
		lines!(ax, [s * kmax, s * (kmax + e)], [ky, ky], color=:gray60, linewidth=3)
		# flat top (ADC) and samples
		lines!(ax, [-s * kmax, s * kmax], [ky, ky], color=c, linewidth=3)
		scatter!(ax, s .* xs, fill(ky, Nx_demo), color=c, markersize=9, strokecolor=:black, strokewidth=0.5)
		arrows2d!(ax, [Point2f(-0.15s, ky)], [Vec2f(0.3s, 0)], tipwidth=10, tiplength=10, color=c)
		text!(ax, -s * (kmax + e) - 0.05s, ky, text="$(i)", align=(s > 0 ? :right : :left, :center), fontsize=12, color=c)
		# blip to the next line
		if i < Ny_demo
			arrows2d!(ax, [Point2f(s * (kmax + e), ky)], [Vec2f(0, 1)], tipwidth=8, tiplength=8, color=:darkgreen, shaftwidth=2)
		end
	end
	scatter!(ax, [0], [0], marker=:star5, markersize=20, color=:red)
	text!(ax, 0.05, 0.15, text="k = 0 at TE", align=(:left, :bottom), fontsize=13, color=:red)
	text!(ax, -(kmax + e / 2), kys[end] + 0.45, text="ramp:\nno ADC", align=(:center, :bottom), fontsize=11, color=:gray40)
	text!(ax, kmax + e + 0.08, kys[1] + 0.5, text="blip", align=(:left, :center), fontsize=12, color=:darkgreen)
	xlims!(ax, -1.6, 1.6)
	ylims!(ax, kys[1] - 0.8, kys[end] + 1.3)
	f
end

# ╔═╡ 6d40c5c4-e341-11f1-8aea-ab1ad8121987
md"""
# 2. Scanner and parameters

These values are given. EPI needs strong and fast gradients: the limits are those of a recent clinical scanner.
"""

# ╔═╡ 6d40c5da-e341-11f1-a624-e93098e13746
begin
	γ̄ = 42.577478518e6       # Hz/T

	sys = Scanner(limits=HardwareLimits(
		Gmax=40e-3,               # maximum gradient amplitude (T/m)
		Smax=150.0,               # maximum slew rate (T/m/s)
		B1=20e-6,                 # maximum RF amplitude (T)
		GR_Δt=10e-6,              # gradient raster time (s)
		RF_Δt=1e-6,               # RF raster time (s)
		ADC_Δt=100e-9,            # ADC raster time (s)
		DUR_Δt=10e-6,             # block duration raster (s)
		RF_dead_time=100e-6,      # (s)
		RF_ring_down_time=20e-6,  # (s)
		ADC_dead_time=10e-6,      # (s)
	))

	FOV = [256e-3, 256e-3]    # field of view (m)
	Nx, Ny = 64, 64           # matrix
	slice_thickness = 5e-3    # m
	flip_angle = 90           # degrees
	TE = 40e-3                # s: time of the center of k-space
	TR = 1.0                 # s (single shot: one excitation per image)
	rf_duration = 3e-3        # s
	dwell = 5e-6              # s: ADC dwell time
	Δk = 1 ./ FOV             # k-space step (1/m)
end;

# ╔═╡ 6d40c5e2-e341-11f1-979c-87f84feb4717
let
	bwpx = round(1 / (Nx * dwell), digits=1)
	tflat = Nx * dwell * 1e6
	md"""
	| derived value | |
	|:--|--:|
	| readout bandwidth | $(bwpx) Hz/pixel |
	| flat top of a readout lobe ``N_x \times`` dwell | $(tflat) µs |
	| area of the flat top ``N_x \Delta k_x`` | $(round(Nx * Δk[1], digits=1)) m⁻¹ |
	| area of a blip ``\Delta k_y`` | $(round(Δk[2], digits=2)) m⁻¹ |
	"""
end

# ╔═╡ 6d40c5f6-e341-11f1-a1cd-9f8288aa30d5
md"""
# 3. Step 1 — Excitation

Create the slice-selective excitation pulse, its slice-selection gradient and the slice rephaser.
"""

# ╔═╡ 3af6bb29-e5d7-45cb-8587-1455f0540bfd
# 👉 excitation pulse, slice-selection gradient and rephaser

# ╔═╡ 6d40c600-e341-11f1-b484-c11f2c12b480
md"""
# 4. Step 2 — Readout lobes and ADC (flat-top sampling)

Create the positive and negative readout lobes and the ADC, which is open **only on the flat top**.

**Notes :** double-check the ADC\_dead\_time
"""

# ╔═╡ 943a016f-7d74-40ca-9f29-9ba8861c74ce
# 👉 readout lobe gx_ro and ADC

# ╔═╡ 6d40c60c-e341-11f1-a88a-5f8ae03f3672
md"""
# 5. Step 3 — Prephasers readout

Create the readout prephaser:
- You will need to compute the gradient area with : `area(gx_ro)`

**Note:** area(gx_ro) is in s*T/m -> you need to multiply by  γ̄ to return to 1/m

Check the results with a dummy sequence:
  - the sequence 
  - k-space trajectory 
  - sampling of central point

"""

# ╔═╡ 24875edf-e1ca-422d-bfc3-6740b040beb7
# 👉 readout prephaser gx_pre

# ╔═╡ a3e1593a-6f1a-4afc-b376-90e48cbe74af
md"""
Let's check with a dummy seq, and the k-space and check the central point ?
"""

# ╔═╡ 0ace3cfe-6f97-4256-ab49-387f6d4cbb47
# 👉 dummy sequence: prephaser + one readout

# ╔═╡ 70f23a01-2068-4a4b-abd7-c8359ae44364
# 👉 plot the sequence

# ╔═╡ 3a5463dc-0a13-4a79-a0e3-863efceb56af
# 👉 plot the k-space trajectory

# ╔═╡ a14bc1d9-b4b2-4bfb-941d-8269c2f463d6
md"""
# 6. Step 4 — Compute phase-encoding gradient and blip gradients

- The phase-encoding is necessary to reach the lower k-space line
- The blip gradient will move the k-space line by 1 one line up.

Build a second dummy sequence to verify the k-space (check first with Ny = 1)
"""

# ╔═╡ 96eda4f8-bca3-4ec6-9123-921d6069c858
# 👉 phase-encoding prephaser gy_pre and blip gy_blip

# ╔═╡ 27560390-b334-4d46-acc3-0d7a38f8340b
# 👉 dummy sequence: excitation, prephasers and the Ny readout lobes + blips

# ╔═╡ a62c51c8-156b-4daa-8163-a2bba8e9a4dd
# 👉 plot the sequence

# ╔═╡ 27f81b63-1939-4160-bbec-484224bcad8f
# 👉 plot the k-space trajectory

# ╔═╡ 20ed442d-5471-40d8-927c-0301cad90c67
md"""
# 7. Step 5 — Add spoiling along Z axis

We want to apply a gradient that imposes a dephasing of 8π across the slice thickness.

The equation that expresses the dephasing occurring across the slice thickness is:

$$Δ ϕ = γ Δx \int G(t) dt = γ Δx\  \text{area\_SI}(G)$$

area_SI is in $s*T/m$ and `maketrapezoid` expects a value in 1/m.
"""

# ╔═╡ 75039254-830c-4e8c-835f-09b5c6534809
# 👉 spoiler gradient gz_spoil

# ╔═╡ 6d40c614-e341-11f1-af2f-674da767a6eb
md"""
# 8. Step 6 — Timing: TE / TR

Compute the echo spacing, the minimum TE and the delays needed to reach the TE and the TR of section 2.

1. build a dummy seq and use the function `dur(seq[i],sys)` or dur.(seq,Ref(sys))` to extract all block duration
2. What is the minimum TE ? Double check how to measure the delay to k=0 !
3. Add a delay (block should be a multiple to the gradient raster : `round_to_raster(t, sys.limits.GR_Δt)  # Round to raster`
4. Now calculate the TR

**Note:** This a complex step find the way you prefer to check that. 
- You can also add label to help you
- take a look at the KomaMRI documentation to see how ADC is defined (a.delay is different to the gx\_ro.rise\_time)
- Double check which sample is the center one
- You can change the number Nx / Ny to validate your test

![](https://juliahealth.org/KomaMRI.jl/dev/assets/event-shapes-adc.CjfDymPf.svg)

"""

# ╔═╡ 80d9a2a5-bf91-44f4-9a19-967a6e501d8a
# 👉 dummy sequence to measure the block durations

# ╔═╡ fe173716-bf67-46fd-9845-6598ea8d3b9f
# 👉 minimum TE and TE delay

# ╔═╡ ea655124-a778-466c-aa70-b37d6d191c3e
md"""
Dummy sequence for TR_min with the correct TE. You can calculate the total delay of the sequence.
"""

# ╔═╡ c9bccb97-f7d7-4881-895f-964ce8655eff
# 👉 dummy sequence with the TE delay

# ╔═╡ 908e8ab3-56d5-4b58-8ae4-27ed68b90056
# 👉 check the timing

# ╔═╡ eb43101d-f62d-4363-9056-b797113c94ad
# 👉 minimum TR and TR delay

# ╔═╡ 6d6b8345-ca80-4076-8567-c5fb6638656b
# 👉 your code here

# ╔═╡ 6d40c628-e341-11f1-8b3c-6d0a9b8410b7
md"""
# 9. Step 7 — The sequence

Write the function that builds the EPI sequence (with the `LIN` label of each line)
"""

# ╔═╡ 743202ef-1ce6-41df-9148-a349d56ae705
# 👉 the complete EPI sequence seq_EPI

# ╔═╡ c385f52f-c689-4353-9cbb-0b7298c1f640
# 👉 check the timing

# ╔═╡ e636c72c-b92b-45a8-aeeb-982dfc791259
# 👉 your code here

# ╔═╡ 7ac8f190-feb5-4147-98f5-51f4fdca266e
# 👉 uncomment to download your sequence
# begin
# 	seq_file = joinpath(mktempdir(), "epi.seq")
# 	write_seq(seq_EPI, seq_file; sys)
# 	DownloadButton(read(seq_file), "epi.seq")
# end

# ╔═╡ 6d40c632-e341-11f1-8311-0b8ea754e05a
md"""
# 10. Step 8 — Checking the k-space trajectory and delay **again**

- find the adc point closer to k=0 (argmin )
```julia
_, k_adc = get_kspace(seq)
t_adc = get_adc_sampling_times(seq)
i0 = argmin(abs2.(k_adc[:, 1]) .+ abs2.(k_adc[:, 2]))
```

- calculate the effective time of this point

**Note:** might be a little bit off due to raster_time rounding)
"""

# ╔═╡ 23d91f41-e094-4184-8c2d-9b33272725e8
# 👉 measure the TE

# ╔═╡ 6d40c63e-e341-11f1-a75c-4f132a025bfd
md"""
# 11. Step 9 — Simulation

Simulate your sequence on the brain phantom and get the raw data.

```julia
	obj_full = brain_phantom2D()
	sim_params = merge(KomaMRICore.default_sim_params(), Dict("return_type" => "raw"))
	raw = simulate(obj_full, seq_EPI, sys; sim_params)
```

and plot the signal : `plot_signal(raw; height=350)`
"""

# ╔═╡ a93ee69c-d836-4df1-b2c2-6a9c5453aa84
# 👉 simulation

# ╔═╡ 8f90ff19-cdf6-4714-9d9f-3dc5bbd40faf
# 👉 plot the signal

# ╔═╡ 6d40c646-e341-11f1-a736-33e633defab2
md"""
# 12. Step 10 — Reconstruction

1. Extract the k-space from the raw struct with the following function:
```julia
function epi_kspace(raw)
	k = zeros(ComplexF64, Nx, Ny)
	for p in raw.profiles
		lin = Int(p.head.idx.kspace_encode_step_1)
		s = p.data[:, 1]
		k[:, lin+1] = s
	end
	return k
end
```

2. perform a reconstruction using fft / fftshift.

3. Analyze the results and fix it if you find an issue
"""

# ╔═╡ d5a931e0-e969-4e93-a7cb-af813418c58e
# 👉 reconstruction

# ╔═╡ a1ba2dd4-5333-4174-8732-c894123c84ae
md"""
You can also perform the reconstruction directly with MRIReco which will automatically take care of the reverse because a nufft acquisition is performed

```julia
img_mrireco = reconstruction(AcquisitionData(raw), Dict{Symbol,Any}(:reco => "direct", :reconSize => (Nx, Ny)))[:, :, 1]
```
"""

# ╔═╡ 8f02770c-48b2-4168-90e4-4e05754889ed
# 👉 reconstruction with MRIReco

# ╔═╡ 027d46bd-3da2-4124-8441-10f498a77cc8
# 👉 display the image

# ╔═╡ 6d40c65a-e341-11f1-a1c6-7715d519c056
md"""
# 13. Step 11 — EPI artefacts

1. What artefact appears in the image?
2. Show a way to prove it.
"""

# ╔═╡ 90e3753b-a178-44e0-bbc9-feb30514eec3
# 👉 your code here

# ╔═╡ 8a777633-60da-4975-aff3-53860d3c38a3
# 👉 your code here

# ╔═╡ 6d40c66c-e341-11f1-8c7d-b978d7d7ba26
md"""
# Helper functions
"""

# ╔═╡ 6d40c682-e341-11f1-be53-5327a3b749f1
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

# ╔═╡ 6d40c5b0-e341-11f1-9c65-a111c7ddb628
md"""
**Q1** The odd and the even lines are acquired in **opposite** directions. What must the reconstruction do before the 2D FFT? What happens in the image if the odd and even lines are slightly shifted in ``k_x`` (e.g. due to a gradient delay)?

**Q2** The time between two adjacent samples is the dwell time along ``k_x`` but ESP along ``k_y``. What is the bandwidth per pixel along the phase-encoding direction? Which direction is the most sensitive to off-resonance (fat, ``B_0`` inhomogeneities)?
""" |> question

# ╔═╡ 67db9f76-6da0-4569-bdc6-b950aa5cfc56
md"""
Useful functions :
- `make_trapezoid`
- `make_adc`
""" |> hint

# ╔═╡ a188067b-b192-4b21-b020-327b6a28daf1
md"""
Useful functions :
- `make_trapezoid`
- Don't forget to multiply by γ̄
- Plot the k-space and check that one of the ADC point position is kx=0/ky=0
- `plot_seq(seq)`
- `plot_kspace(seq)`
""" |> hint

# ╔═╡ 423de740-e3f2-41e2-8ba0-575619c6dce5
md"""
- use rf.delay and rf.center (check the definition in documentation)
- The timing value on the plot starts at the beginning of the sequence. If you want to get the TE value, don't forget to remove rf.delay and rf.center.
- Change the matrix size / TE to make your life easy (N=8 is a sweet spot)
""" |> hint

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
KomaMRI = "6a340f8b-2cdf-4c04-99be-4953d9b66d0a"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
CairoMakie = "~0.15.17"
FFTW = "~1.10.0"
HypertextLiteral = "~1.0.0"
KomaMRI = "~0.15.0"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "d7b8f1c245d076641a147ef4e0c1e3feb8517d86"

[[deps.ANSIColoredPrinters]]
git-tree-sha1 = "574baf8110975760d391c710b6341da1afa48d8c"
uuid = "a4c015fc-c6ff-483c-b24f-f7ea428134e9"
version = "0.0.1"

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

[[deps.AcceleratedKernels]]
deps = ["ArgCheck", "GPUArraysCore", "KernelAbstractions", "Markdown", "UnsafeAtomics"]
git-tree-sha1 = "f75592c5dc92f7f75e544b156eba7165d4418193"
uuid = "6a4ca0a5-0e36-4168-a932-d9be78d558f1"
version = "0.4.4"

    [deps.AcceleratedKernels.extensions]
    AcceleratedKernelsoneAPIExt = "oneAPI"

    [deps.AcceleratedKernels.weakdeps]
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "8c0bc12397dabf5d2dfc4b5763ec463854cc44d3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.46"

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
git-tree-sha1 = "eec8a0ad4e4b673afba213931504c360fffb8f97"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.8.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptiveArrayPools]]
deps = ["Preferences", "Printf", "Random"]
git-tree-sha1 = "86d0ae34bc46373170064ef33c12a303994ab736"
uuid = "4f381ef7-9af0-4cbe-99d4-cf36d7b0f233"
version = "0.3.12"

    [deps.AdaptiveArrayPools.extensions]
    AdaptiveArrayPoolsCUDAExt = "CUDA"
    AdaptiveArrayPoolsMetalExt = "Metal"

    [deps.AdaptiveArrayPools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"

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

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "1aec1ff0dcaa83a484e7f72097562fa3102c6722"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.30.2"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceFillArraysExt = "FillArrays"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceGPUArraysCoreTrackerExt = ["GPUArraysCore", "Tracker"]
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "0aa371c875e24cf773c87b2f50ab03e4545748af"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.7.0"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"

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

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BasicInterpolators]]
deps = ["LinearAlgebra", "Memoize", "Random"]
git-tree-sha1 = "3f7be532673fc4a22825e7884e9e0e876236b12a"
uuid = "26cce99e-4866-4b6d-ab74-862489e035e0"
version = "0.7.1"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Bonito]]
deps = ["ANSIColoredPrinters", "Base64", "CodecZlib", "Colors", "CommonMark", "Dates", "Deno_jll", "HTTP", "Hyperscript", "JSON", "LinearAlgebra", "Markdown", "MbedTLS", "MsgPack", "Observables", "OrderedCollections", "PrecompileTools", "Random", "RelocatableFolders", "SHA", "Scratch", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "189a245332edd1515bf3c291a7659e4458c323e2"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "5.3.0"

    [deps.Bonito.extensions]
    BonitoDocumenterExt = ["Documenter", "MarkdownAST"]

    [deps.Bonito.weakdeps]
    Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
    MarkdownAST = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "0db159493e83ecb2d122f9c40138d89b70d43963"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.1.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "FreeType2_jll", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "2d9e0c818e6e5af309192a5b3638c8848c7119b5"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.17"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "0cfb779e24fdced77a05df1dbe99200821a9f563"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+1"

[[deps.ChainRulesCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "295fa9b898db109bc56df62c0fc20bd105cec00e"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.8"
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
git-tree-sha1 = "060adaf7a5c2e14bca025edf0eea0206ffdebd87"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.32.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "af523e59de94de05b9b367a2ca4d63233aacd222"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.5"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "96fce5c80e0834e96ab6d24fb384fd945c9cb2d4"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.2"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "0737a3e59f9d94957f5b177fa2ca40901636b85b"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.4"

[[deps.CommonMark]]
deps = ["PrecompileTools"]
git-tree-sha1 = "7c8fe02c7eb6fe22e89d3990123a2a931ca8fcfb"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "1.0.4"

    [deps.CommonMark.extensions]
    CommonMarkMarkdownASTExt = "MarkdownAST"
    CommonMarkMarkdownExt = "Markdown"

    [deps.CommonMark.weakdeps]
    Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
    MarkdownAST = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonWorldInvalidations]]
deps = ["PrecompileTools"]
git-tree-sha1 = "7c4ea0adfb7238bf4678aa36325863fab6163f6f"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.2.2"

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
git-tree-sha1 = "b56df0c58a9d37b4a926f84f3f4f283a5987680e"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.9"

[[deps.ConstructionBase]]
git-tree-sha1 = "b1bc10667320dd5367aff450a992c951ea38fbbc"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.1"
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

[[deps.CpuId]]
deps = ["Markdown", "PrecompileTools"]
git-tree-sha1 = "5dbd5150b96f29553fc3b553c6bfb0eadf7ac204"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.3"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "dd73b345b24e52d7b57550b69113261ef2d79c92"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.7"
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

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "e1e28f37cbdca4b594ad86ecfe2ee8f25796ebff"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.7.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "478d3450b1cc56521d1d99ad923acf43f8813c6d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.134"

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

[[deps.Electron]]
deps = ["Artifacts", "Base64", "FilePaths", "JSON", "RelocatableFolders", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1a80218bb51d9f59e7a31adbbe7983500d07027"
uuid = "a1bb12fb-d4d1-54b4-b10a-ee7951ef7ad3"
version = "7.0.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f2659a735f9360fedfa29925812380f322679e93"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.10.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "d2e49e7efd29719d6f28b891b0e0e159daa9d2b4"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.11"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "f4d904e7ff5aadd8e7fe9ab8544a0a295dfa8904"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.3"

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

[[deps.FastInterpolations]]
deps = ["AdaptiveArrayPools", "HelpPlots", "LinearAlgebra", "Preferences", "Printf"]
git-tree-sha1 = "f7bcd496f0a2e4e1174e7e1f667f27fb671fd7df"
uuid = "9ea80cae-fc13-4c00-8066-6eaedb12f34b"
version = "0.4.20"

    [deps.FastInterpolations.extensions]
    FastInterpolationsChainRulesCoreExt = "ChainRulesCore"
    FastInterpolationsColorVectorSpaceExt = ["ColorTypes", "ColorVectorSpace"]
    FastInterpolationsEnzymeExt = "Enzyme"
    FastInterpolationsForwardDiffExt = "ForwardDiff"
    FastInterpolationsRecipesBaseExt = "RecipesBase"
    FastInterpolationsSymbolicsExt = ["Symbolics", "SymbolicUtils"]

    [deps.FastInterpolations.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
    ColorVectorSpace = "c3611d14-8923-5661-9e6a-0046d554d3a4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

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
git-tree-sha1 = "831f86fde78ab7f6470d6ba4d1161a11fb6b4ec0"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.2"
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
git-tree-sha1 = "e1e9ba3a8f5d71e6ae698f138a1362aeabb42e9f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.18.3+0"

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
git-tree-sha1 = "e894be022a2eba58463939f1bf1f7f7006f27f52"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.19+0"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "416b521e1158dd93cd99b3adcd872085726b36a1"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "f60436514e6d73f8ee069ae4ee0ff038fe96b69d"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.2"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "462ab5e6578928370f632705e96af82f6c1c09a1"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "ec46c5825710fa1a15d468acb2d93cc939a7a5fe"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.13"

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
git-tree-sha1 = "a3efbbc027441271444dcd0c0a46f2d119dc4329"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "6.1.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "ef70da5e123a06a29e2d6ddff0f09985bc226491"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.3"

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
git-tree-sha1 = "d791a1c4c3c8f383b248a799be31105d00310204"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "2.2.4+0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "EnumX", "Logging", "PrecompileTools", "Random", "Reseau", "SHA", "TranscodingStreams", "URIs", "UUIDs", "Zlib_jll"]
git-tree-sha1 = "70b131ae84fb6ff84d4e70ba1648fb5747e339a0"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "2.15.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "59d266e426ab9695f625fe2b6a51c2d3a04f1ecd"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.15001.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HelpPlots]]
git-tree-sha1 = "4b09f61ba5ab9242250855d44cf8747d0d43b981"
uuid = "5880b2e8-ad0a-48c9-bacd-30dc9c1357e0"
version = "1.1.1"

    [deps.HelpPlots.extensions]
    HelpPlotsExt = "Plots"

    [deps.HelpPlots.weakdeps]
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Preferences", "Static"]
git-tree-sha1 = "af9ab7d1f70739a47f03be78771ebda38c3c71bf"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.18"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "550c21420b96b46f2f0abe8fc4c9430cd55649e3"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.15.0+3"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "c28dceeadc1c9c699656a3ac9d63f79cfaa17472"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.31"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

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
git-tree-sha1 = "f0f005f997dfb8c5fe23920d99458a9619873893"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.10"

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
git-tree-sha1 = "1c531bf0f8a5c60a340926e058fd3f209b5eef5d"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.12"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticMakieExt = "Makie"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "0db6aea5b64caa1c47e9fdd27394f0296f05e8bf"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.15"
weakdeps = ["Makie", "Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsMakieExt = "Makie"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "123311cf9834c6ee6a8347bba9af68f255615391"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.11.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "59545b0a2b27208b0650df0a46b8e3019f85055b"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.4"

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
git-tree-sha1 = "06d90f699285d055e34f0b014636a75a17ff2252"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.13.4"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]
    JSONSerializationExt = ["Serialization"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e06b451acfb1c6fa5ea753b5042207226271f455"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "2e1baab820f43c198e2a247c7f061d0b53e27e8d"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.46"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "19fa81989298fc2fb103fee29b03c5284f17e8a5"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.13"

[[deps.KomaMRI]]
deps = ["Artifacts", "AxisArrays", "Bonito", "Electron", "FFTW", "KomaMRICore", "KomaMRIFiles", "KomaMRIPlots", "MAT", "MRIReco", "MsgPack", "Observables", "PlotlyBase", "PrecompileTools", "Preferences", "Reexport", "TOML"]
git-tree-sha1 = "319d619dc977b8938df71002409f90da6ce1a5ba"
uuid = "6a340f8b-2cdf-4c04-99be-4953d9b66d0a"
version = "0.15.0"

[[deps.KomaMRIBase]]
deps = ["DSP", "FFTW", "Interpolations", "LinearAlgebra", "MAT", "MRIBase", "Parameters", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8606e160d47abb369d1b48585bc7cb1ca662c434"
uuid = "d0bc0b20-b151-4d03-b2a4-6ca51751cb9c"
version = "0.15.2"
weakdeps = ["Unitful"]

    [deps.KomaMRIBase.extensions]
    KomaMRIBaseUnitfulExt = "Unitful"

[[deps.KomaMRICore]]
deps = ["AcceleratedKernels", "Adapt", "Functors", "KernelAbstractions", "KomaMRIBase", "PrecompileTools", "ProgressMeter", "Reexport", "ThreadsX"]
git-tree-sha1 = "5bca758d01b34dcc62e852e10bed0d49b34450f8"
uuid = "4baa4f4d-2ae9-40db-8331-a7d1080e3f4e"
version = "0.14.0"

    [deps.KomaMRICore.extensions]
    KomaAMDGPUExt = "AMDGPU"
    KomaCUDAExt = "CUDA"
    KomaMetalExt = "Metal"
    KomaoneAPIExt = "oneAPI"

    [deps.KomaMRICore.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.KomaMRIFiles]]
deps = ["FileIO", "HDF5", "InteractiveUtils", "KomaMRIBase", "MAT", "MD5", "MRIFiles", "PrecompileTools", "Printf", "Reexport", "SHA"]
git-tree-sha1 = "4d5992a0a4332726237ec3305edb59b018449689"
uuid = "fcf631a6-1c7e-4e88-9e64-b8888386d9dc"
version = "0.12.0"

[[deps.KomaMRIPlots]]
deps = ["Artifacts", "Bonito", "Interpolations", "KomaMRIBase", "MAT", "PlotlyBase", "PlotlyKaleido", "QMRIColors"]
git-tree-sha1 = "6d631f747cebd970ba4bf12e8abc7f033878f767"
uuid = "76db0263-63f3-4d26-bb9a-5dba378db904"
version = "0.14.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d7c65411dda4bfa0beb7490476122b4f805cf31"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.3.0+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "ScopedValues", "Unicode"]
git-tree-sha1 = "41c2ce7ec5a14c89b93dc6a5853bf2a91d7a8583"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "10.4.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "39def0b4521e0ef2bf26960d8965729f496bb4ec"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.56+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

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

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

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
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "aa971a09f0f1fe92fe772713a564aa48abe510df"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.3"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LinearOperatorCollection]]
deps = ["Adapt", "InteractiveUtils", "LinearAlgebra", "LinearOperators", "OhMyThreads", "Random", "Reexport", "SparseArrays"]
git-tree-sha1 = "280cae8a51e4c45a8364d7bf36a4892b353101d8"
uuid = "a4a2c56f-fead-462a-a3ab-85921a5f2575"
version = "2.5.1"

    [deps.LinearOperatorCollection.extensions]
    LinearOperatorFFTWExt = "FFTW"
    LinearOperatorGPUArraysExt = "GPUArrays"
    LinearOperatorNFFTExt = ["AbstractNFFTs", "FFTW"]
    LinearOperatorNonuniformFFTsExt = ["AbstractNFFTs", "NonuniformFFTs", "FFTW"]
    LinearOperatorRadonKAExt = "RadonKA"
    LinearOperatorWaveletExt = "Wavelets"

    [deps.LinearOperatorCollection.weakdeps]
    AbstractNFFTs = "7f219486-4aa7-41d6-80a7-e08ef20ceed7"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    NFFT = "efe261a4-0d2b-5849-be55-fc731d526b0d"
    NonuniformFFTs = "cd96f58b-6017-4a02-bb9e-f4d81626177f"
    RadonKA = "86de8297-835b-47df-b249-c04e8db91db5"
    Wavelets = "29a6e085-ba6d-5f35-a997-948ac2efa89a"

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
git-tree-sha1 = "52702dc5aba6392de19d734750c37ffb08515dc0"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.5"

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

[[deps.MD5]]
deps = ["Random", "SHA"]
git-tree-sha1 = "1576f756617d31eb397a4a517b68562fd28dc2b4"
uuid = "6ac74813-4b46-53a4-afec-0b5dc9d7885c"
version = "0.2.3"

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

[[deps.MRIBase]]
deps = ["AbstractNFFTs", "LinearAlgebra", "NFFTTools"]
git-tree-sha1 = "085cc7b8873c752001e178b693e0aa25f8c0246a"
uuid = "f7771a9a-6e57-4e71-863b-6e4b6a2f17df"
version = "0.4.6"

[[deps.MRIFiles]]
deps = ["FileIO", "HDF5", "LightXML", "LinearAlgebra", "MRIBase", "Printf", "Reexport"]
git-tree-sha1 = "ee51abab4b541393d95be8b7e7f16f618e561456"
uuid = "5a6f062f-bf45-497d-b654-ad17aae2a530"
version = "0.3.2"

[[deps.MRIOperators]]
deps = ["Adapt", "LinearAlgebra", "LinearOperatorCollection", "LinearOperators", "MRIBase", "NFFT", "OhMyThreads", "Printf", "Reexport", "StatsBase", "TSVD", "Wavelets"]
git-tree-sha1 = "0abba0db38406c7a1e8b444b163ff50b90b696fb"
uuid = "fb1137e3-90a6-46ce-a672-6e1e53d120f2"
version = "0.4.3"

    [deps.MRIOperators.extensions]
    MRIOperatorsGPUArraysExt = ["GPUArrays", "KernelAbstractions"]

    [deps.MRIOperators.weakdeps]
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"

[[deps.MRIReco]]
deps = ["AxisArrays", "LinearAlgebra", "MRIBase", "MRIOperators", "OhMyThreads", "PrecompileTools", "ProgressMeter", "Random", "Reexport", "RegularizedLeastSquares", "Unitful"]
git-tree-sha1 = "a2a2c5e9648c53579e1e0dde222d49ea4b488057"
uuid = "bdf86e05-2d2b-5731-a332-f3fe1f9e047f"
version = "0.9.2"

    [deps.MRIReco.extensions]
    MRIRecoGPUArraysExt = "GPUArrays"

    [deps.MRIReco.weakdeps]
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FastInterpolations", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "8ff1cd11162828e9f517e32d83b49d2af63391c6"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.17"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

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

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "0e4b139a1f0f0b8c054fcdfd3be5f610cfb8c787"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.2.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "3350d35d0ebcf7ccee0c66290b8e95d82419ef6f"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.5+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "d9d20d6eb2a3290ba5c70a1c013b45305cc39c9e"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

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
deps = ["AbstractFFTs", "AbstractNFFTs", "FFTW", "LinearAlgebra"]
git-tree-sha1 = "e20ac21560b314a7c38c4cd2fdcece34c66f36b6"
uuid = "7424e34d-94f7-41d6-98a0-85abaf1b6c91"
version = "0.2.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "676c0f1c1c34224d6869675047a54b42fdc7b8c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.5"

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
git-tree-sha1 = "750d0f2b5aac238d50dfee5f7a44fae3175a3305"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.18.0"
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
git-tree-sha1 = "a73a90d87e3f941f5084d34d56934c2fb59dd9b1"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.9"
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
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "OpenJPH_jll", "Zlib_jll", "Zstd_jll", "libdeflate_jll"]
git-tree-sha1 = "afa66dc1246b28b68600d053715e00a379268980"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.5.3+0"

[[deps.OpenJPH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "1b1f3cabb1130670812b848f4623d69d578ea805"
uuid = "95196869-6d61-5868-8eee-c38c33804e0c"
version = "0.32.0+0"

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
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "0ed04c372da78ff5b98e35e3bd4f4ca9931299cc"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.13.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "e5ae52d1f4f19a9591d615cb0f1ef8d415b40933"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.6"

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
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "83585da0006b18bea4731b042017567252156895"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.10.0"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlotlyKaleido]]
deps = ["Artifacts", "Base64", "JSON", "Kaleido_jll"]
git-tree-sha1 = "c7a270d11881c0709052f8e3def3d60d551e99f3"
uuid = "f2990250-8cf9-495f-b13a-cce12b45703c"
version = "2.3.1"

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
git-tree-sha1 = "d86ea3a5364b306b8547e02741830e69c2d04448"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.6"

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
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "eeac4c9337f876c3c32b8da732141953567f57c9"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.1"

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
deps = ["Printf"]
git-tree-sha1 = "ce5ba3edfdeba107a721748d53b5f5887f415853"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.13.0"

    [deps.ProgressMeter.extensions]
    ProgressMeterDistributedExt = "Distributed"
    ProgressMeterIJuliaExt = "IJulia"

    [deps.ProgressMeter.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QMRIColors]]
deps = ["Colors", "DelimitedFiles"]
git-tree-sha1 = "77d0d5f446d544beaa568b837fade390d9d38721"
uuid = "2bec176e-9e8c-4764-a62f-295118d1ec05"
version = "1.0.1"

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

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0c0dcffc464cda2ff33fc8ffc1dd1bdd53cd98be"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.4.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "02d31ad62838181c1a3a5fd23a1ce5914a643601"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.3"

[[deps.RegularizedLeastSquares]]
deps = ["InteractiveUtils", "IterativeSolvers", "LinearAlgebra", "LinearOperatorCollection", "LinearOperators", "OhMyThreads", "Random", "ShiftedArrays", "SparseArrays", "Statistics", "StatsBase", "VectorizationBase"]
git-tree-sha1 = "b70c3c3e9ad3b13fec0cc40310ec152cd3fea4d8"
uuid = "1e9c538a-f78c-5de5-8ffb-0b6dbe892d23"
version = "0.16.13"

    [deps.RegularizedLeastSquares.extensions]
    RegularizedLeastSquaresCUDAExt = "CUDA"
    RegularizedLeastSquaresGPUArraysExt = ["GPUArrays", "KernelAbstractions"]

    [deps.RegularizedLeastSquares.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"

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

[[deps.Reseau]]
deps = ["NetworkOptions", "OpenSSL_jll", "PrecompileTools", "Random", "SHA"]
git-tree-sha1 = "13d0465bbd89b45183c0363b635dc82d5843593f"
uuid = "802f3686-a58f-41ce-bb0c-3c43c75bba36"
version = "1.6.6"

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
git-tree-sha1 = "15f266d7b3039ca0608d2c53bb32225b96932310"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.14"

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
git-tree-sha1 = "0651e0ac84b2027c5f227c058d37f81b5be373bd"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.8.2"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SciMLPublic]]
deps = ["PrecompileTools"]
git-tree-sha1 = "74685afb51732a464fbce79a72708f7c4203ceb7"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.3.0"

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
git-tree-sha1 = "89880a2cca5cabc68c9023f58e544beda1d2fc1f"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.2.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "57aa595158717ef165e6f5ab639fe2e3178c0a2b"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

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
git-tree-sha1 = "2c79185e5261b159474903598c571ddd9f12ff7b"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.6"

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
git-tree-sha1 = "d1a08b3cb5d1470a2b6dc98ba644f1436d5a740f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "474a5283ad435618090122872eea6a8165ea6bcf"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.4.6"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "SciMLPublic", "Static"]
git-tree-sha1 = "2a635e15d5035c53b345077c947f31ff91744078"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.10.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "8a8a52f887f38d44014e24ee46b8a7311c625f6d"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.24"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "bf7e2d0d9ed9ac2a652fd598c0f784910cc759f2"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.5"

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
git-tree-sha1 = "4daaddbe37aa80b4551bf7c56adf53e7cfcdf35a"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.4"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    AdaptExt = "Adapt"
    GPUArraysCoreExt = "GPUArraysCore"
    KernelAbstractionsExt = "KernelAbstractions"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"
    StaticArraysExt = "StaticArrays"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "b04bfab37484defe300b2f9ed64469abab2431ff"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.12.0"

    [deps.StructUtils.extensions]
    StructUtilsDBInterfaceExt = ["DBInterface", "Tables"]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DBInterface = "a10d1c49-ce27-4219-8d33-6db1a4562965"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
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

[[deps.TSVD]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "c39caef6bae501e5607a6caf68dd9ac6e8addbcb"
uuid = "9449cd9e-2762-5aa3-a617-5413e99d722e"
version = "0.4.4"

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

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.ThreadsX]]
deps = ["Accessors", "ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "SplittablesBase", "Transducers"]
git-tree-sha1 = "70bd8244f4834d46c3d68bd09e7792d8f571ef04"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.12"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TimerOutputs]]
deps = ["ExprTools", "PrettyTables", "Printf", "Tables"]
git-tree-sha1 = "30350f9ca2e26d58ed7abab3461494d42e0c223e"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "1.2.3"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "eb3b85ca904a1d1647b15acc091b113b8874b648"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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

[[deps.UnsafeAtomics]]
deps = ["LLVM"]
git-tree-sha1 = "74ac5e0e702972da953ffc981094617070be582d"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.4.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsGPUCompilerExt = "GPUCompiler"

    [deps.UnsafeAtomics.weakdeps]
    GPUCompiler = "61eb1bfa-7361-4325-ad38-22787b887f55"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "807a234dc5e6132dd6cf4c9317ca0917c4001ab3"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.74"

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

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

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
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

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

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

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
git-tree-sha1 = "7945f94b714d20b55aee2510b511afee0ee17ad2"
uuid = "2b3700d1-4306-52e2-a478-c162f0c514be"
version = "1.0.1+0"

[[deps.aws_c_cal_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "60a8d3fafed6f7ed8b3114793885a03330aa8e51"
uuid = "70f11efc-bab2-57f1-b0f3-22aad4e67c4b"
version = "1.0.0+0"

[[deps.aws_c_common_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f5ce8191ad02a5a4127d37ab5f474aee5cf91ad4"
uuid = "73048d1d-b8c4-5092-a58d-866c5e8d1e50"
version = "1.0.3+1"

[[deps.aws_c_compression_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "6efd591b9fae7919266d3b342a18a6e6280850f9"
uuid = "73a04cd5-f3d7-5bac-9290-e8adb709f224"
version = "1.0.0+1"

[[deps.aws_c_http_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_compression_jll", "aws_c_io_jll"]
git-tree-sha1 = "e4108b4a2201cb9987ddd36714fdcfd572d6bb24"
uuid = "3254fc65-9028-534d-aa9d-d76d128babc6"
version = "1.0.0+0"

[[deps.aws_c_io_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_common_jll", "s2n_tls_jll"]
git-tree-sha1 = "b816756403a8db80e32fc23a4a1aae240d2168ca"
uuid = "13c41daa-f319-5298-b5eb-5754e0170d52"
version = "1.0.0+0"

[[deps.aws_c_s3_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_auth_jll", "aws_c_common_jll", "aws_c_http_jll", "aws_checksums_jll", "s2n_tls_jll"]
git-tree-sha1 = "1ebf5d4fa9c5248c3c1066124ef41d06de05a03a"
uuid = "bd1f34fb-993f-5903-a121-aaf302eed6d4"
version = "1.3.1+0"

[[deps.aws_c_sdkutils_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "0162699e5aa99420f04a9606a9d7b38a872a81ee"
uuid = "1282aa60-004d-510b-9f52-12498d409daa"
version = "1.0.0+0"

[[deps.aws_checksums_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "ee1f5427815c98355b0b0454b5fe79806ffdf935"
uuid = "b2a88e68-78e7-5e94-8c20-c02986ec140e"
version = "1.0.1+1"

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
git-tree-sha1 = "31e4c16405815484827f75a38fb7eb17432e72e3"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.8+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "21e05cef1411bfe6b22a1ca8cbdc64a297d4b447"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdeflate_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "11c07205bc3b3b7833b8fb36beb4e4764d0126a7"
uuid = "46979653-d7f6-5232-b59e-dd310c4598de"
version = "1.26.0+1"

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
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "e067c8bae65bb40866552296a2d5b4dc65b8928f"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.80.702+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "21cbc8fc2ac0bb182ed461e556d9c062ae6a3164"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.24.1+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "52d3b9475133c3bc8c0a7f90f18bfc8cdb5a443a"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.1+0"

[[deps.mpif_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML"]
git-tree-sha1 = "52e322f7a6d418576cdeb043dd2a7f7ad9a8abd0"
uuid = "9aeb927a-4695-514f-a259-621a69f20ec0"
version = "1.0.1+0"

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
git-tree-sha1 = "6f9611b1b99beb089154f19a4031579f3a818cbd"
uuid = "cddc5d3d-934d-5d3a-9747-62fc12ea3f48"
version = "1.7.12+0"

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
"""

# ╔═╡ Cell order:
# ╟─6d3fcda4-e341-11f1-bc12-59a517d73b55
# ╟─6d40c4ca-e341-11f1-8792-3d0d4367c887
# ╟─6d40c51a-e341-11f1-b726-dfca5c95d502
# ╟─6d40c536-e341-11f1-88f0-11ddef63b873
# ╟─6d40c54c-e341-11f1-9b88-717fd00ece5e
# ╟─6d40c568-e341-11f1-8c84-8d544e224da0
# ╟─6d40c57e-e341-11f1-9861-bd12a1a601b9
# ╟─6d40c592-e341-11f1-8211-ef27254f7262
# ╟─6d40c59c-e341-11f1-b549-97e6ffa22440
# ╟─6d40c5b0-e341-11f1-9c65-a111c7ddb628
# ╟─6d40c5c4-e341-11f1-8aea-ab1ad8121987
# ╠═6d40c5da-e341-11f1-a624-e93098e13746
# ╠═6d40c5e2-e341-11f1-979c-87f84feb4717
# ╟─6d40c5f6-e341-11f1-a1cd-9f8288aa30d5
# ╠═3af6bb29-e5d7-45cb-8587-1455f0540bfd
# ╟─6d40c600-e341-11f1-b484-c11f2c12b480
# ╟─67db9f76-6da0-4569-bdc6-b950aa5cfc56
# ╠═943a016f-7d74-40ca-9f29-9ba8861c74ce
# ╟─6d40c60c-e341-11f1-a88a-5f8ae03f3672
# ╟─a188067b-b192-4b21-b020-327b6a28daf1
# ╠═24875edf-e1ca-422d-bfc3-6740b040beb7
# ╟─a3e1593a-6f1a-4afc-b376-90e48cbe74af
# ╠═0ace3cfe-6f97-4256-ab49-387f6d4cbb47
# ╠═70f23a01-2068-4a4b-abd7-c8359ae44364
# ╠═3a5463dc-0a13-4a79-a0e3-863efceb56af
# ╟─a14bc1d9-b4b2-4bfb-941d-8269c2f463d6
# ╠═96eda4f8-bca3-4ec6-9123-921d6069c858
# ╠═27560390-b334-4d46-acc3-0d7a38f8340b
# ╠═a62c51c8-156b-4daa-8163-a2bba8e9a4dd
# ╠═27f81b63-1939-4160-bbec-484224bcad8f
# ╟─20ed442d-5471-40d8-927c-0301cad90c67
# ╠═75039254-830c-4e8c-835f-09b5c6534809
# ╟─6d40c614-e341-11f1-af2f-674da767a6eb
# ╟─423de740-e3f2-41e2-8ba0-575619c6dce5
# ╠═80d9a2a5-bf91-44f4-9a19-967a6e501d8a
# ╠═fe173716-bf67-46fd-9845-6598ea8d3b9f
# ╟─ea655124-a778-466c-aa70-b37d6d191c3e
# ╠═c9bccb97-f7d7-4881-895f-964ce8655eff
# ╠═908e8ab3-56d5-4b58-8ae4-27ed68b90056
# ╠═eb43101d-f62d-4363-9056-b797113c94ad
# ╠═6d6b8345-ca80-4076-8567-c5fb6638656b
# ╟─6d40c628-e341-11f1-8b3c-6d0a9b8410b7
# ╠═743202ef-1ce6-41df-9148-a349d56ae705
# ╠═c385f52f-c689-4353-9cbb-0b7298c1f640
# ╠═e636c72c-b92b-45a8-aeeb-982dfc791259
# ╠═7ac8f190-feb5-4147-98f5-51f4fdca266e
# ╟─6d40c632-e341-11f1-8311-0b8ea754e05a
# ╠═23d91f41-e094-4184-8c2d-9b33272725e8
# ╟─6d40c63e-e341-11f1-a75c-4f132a025bfd
# ╠═a93ee69c-d836-4df1-b2c2-6a9c5453aa84
# ╠═8f90ff19-cdf6-4714-9d9f-3dc5bbd40faf
# ╟─6d40c646-e341-11f1-a736-33e633defab2
# ╠═d5a931e0-e969-4e93-a7cb-af813418c58e
# ╟─a1ba2dd4-5333-4174-8732-c894123c84ae
# ╠═8f02770c-48b2-4168-90e4-4e05754889ed
# ╠═027d46bd-3da2-4124-8441-10f498a77cc8
# ╟─6d40c65a-e341-11f1-a1c6-7715d519c056
# ╠═90e3753b-a178-44e0-bbc9-feb30514eec3
# ╠═8a777633-60da-4975-aff3-53860d3c38a3
# ╟─6d40c66c-e341-11f1-8c7d-b978d7d7ba26
# ╟─6d40c682-e341-11f1-be53-5327a3b749f1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
