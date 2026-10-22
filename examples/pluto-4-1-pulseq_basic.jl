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

# ╔═╡ b2a7c86e-c95c-11f1-adb2-e5af245fc1b8
begin
	# Layout
	using PlutoUI
	using PlutoUI: Slider   # GLMakie also exports a `Slider`
	using PlutoTeachingTools
	using HypertextLiteral
	using Printf

	# Plot
	using GLMakie

	# Sequence design (Pulseq) and simulation
	using KomaMRI
	using KomaMRI.PulseDesigner
	using FFTW                # FFT for the reconstruction

	PlutoUI.TableOfContents()
end

# ╔═╡ b2a6b80c-c95c-11f1-b594-532a3821be5c
ChooseDisplayMode()

# ╔═╡ b2a7c80a-c95c-11f1-b301-7118b80de909
html"""
	<h1 style="text-align:center">
		Graduate Program - Biomedical MRI
	</h1>
	<div style="text-align:center">
		<p style="font-weight:bold; font-size: 35px; font-variant: small-caps; margin: 0px">
			Lesson 4-1: Writing a Pulseq sequence
		</p>
		<p style="font-size: 30px; font-variant: small-caps; margin: 0px">
			Aurélien Trotier
		</p>
		<p style="font-size: 20px;">
			CRMSB - Université de Bordeaux / CNRS
		</p>
	</div>
"""

# ╔═╡ b2a7c882-c95c-11f1-95ad-7b6f6a918114
md"""
# 1. Introduction

## 1.1 What is Pulseq?

[Pulseq](https://pulseq.github.io) is an open, **vendor-independent** file format (`.seq`) to describe an MRI sequence. The sequence is written on a computer (MATLAB, Python with pypulseq, or Julia with KomaMRI) and executed on the scanner by an *interpreter* sequence (available for Siemens, GE and Philips).

A Pulseq sequence is a list of **blocks** played one after the other. A block contains **at most one event per channel**:

| Channel | Event |
|:--|:--|
| RF | RF pulse (amplitude and phase shape) |
| Gx, Gy, Gz | gradient (trapezoid or arbitrary shape) |
| ADC | data acquisition (number of samples, dwell time) |
| delay | block duration (when the block only waits) |

All the events of a block start at the **beginning of the block** (each one can have its own `delay`), and the block lasts as long as its **longest event**. There is no notion of "the gradient continues in the next block": everything happening at the same time must be in the same block.

## 1.2 Pulseq in Julia: KomaMRI

KomaMRI.jl reads, writes and **simulates** Pulseq sequences. Its `PulseDesigner` module provides functions with the same names as in MATLAB Pulseq and pypulseq:

| function | creates |
|:--|:--|
| `make_sinc_pulse` | a slice-selective sinc RF pulse + its slice-selection and rephasing gradients |
| `make_trapezoid` | a trapezoidal gradient |
| `make_adc` | an ADC event |
| `make_delay` | a delay (empty block) |
| `@addblock seq += (...)` | adds a block to the sequence |
| `check_timing` | checks rasters, dead times and ring-down |
| `write_seq` | writes the `.seq` file |

The same sequence object can then be **simulated** with KomaMRI (next part of the course) and run on the scanner.

## 1.3 Units

Pulseq does not use T/m for the gradients but frequencies: the gyromagnetic ratio is included in the amplitudes.

| quantity | Pulseq unit | conversion |
|:--|:--|:--|
| gradient amplitude | Hz/m | $G_{[Hz/m]} = \bar{\gamma}\, G_{[T/m]}$ |
| gradient area (k-space position) | 1/m | $k = \bar{\gamma}\int G\,dt$ |
| slew rate | Hz/m/s | $S_{[Hz/m/s]} = \bar{\gamma}\, S_{[T/m/s]}$ |

with $\bar{\gamma} = \gamma/2\pi = 42.577$ MHz/T for the proton. In `make_*` functions, **plain numbers are in Pulseq units** (Hz/m, 1/m), but KomaMRI stores the events in SI units (T/m).
"""

# ╔═╡ b2a7c8dc-c95c-11f1-b376-1f4337d7986f
γ̄ = 42.577478518e6   # Hz/T, proton gyromagnetic ratio / 2π

# ╔═╡ b2a7c8ee-c95c-11f1-a495-131e2d17ea4b
md"""
# Part I — The subtleties

# 2. The scanner limits

Before designing anything, we must describe the hardware. Each value is a **hard constraint**: a sequence violating it will be rejected by the interpreter on the scanner (or will damage the hardware).
"""

# ╔═╡ b2a7c8fa-c95c-11f1-b402-4bb9161851ee
sys = Scanner(limits=HardwareLimits(
	Gmax=20e-3,               # maximum gradient amplitude (T/m)
	Smax=100.0,               # maximum slew rate (T/m/s)
	B1=20e-6,                 # maximum RF amplitude (T)
	GR_Δt=10e-6,              # gradient raster time (s)
	RF_Δt=1e-6,               # RF raster time (s)
	ADC_Δt=100e-9,            # ADC raster time (s)
	DUR_Δt=10e-6,             # block duration raster (s)
	RF_dead_time=100e-6,      # (s)
	RF_ring_down_time=20e-6,  # (s)
	ADC_dead_time=10e-6,      # (s)
));

# ╔═╡ b2a7c90e-c95c-11f1-9565-81482728c9a2
md"""
| limit | value | meaning |
|:--|:--|:--|
| `Gmax` | $(sys.limits.Gmax * 1e3) mT/m = $(round(sys.limits.Gmax * γ̄ / 1e3, digits=1)) kHz/m | maximum gradient amplitude |
| `Smax` | $(sys.limits.Smax) T/m/s | maximum slew rate: shortest rise time ``G_{max}/S_{max}`` = $(round(sys.limits.Gmax / sys.limits.Smax * 1e6)) µs |
| `B1` | $(sys.limits.B1 * 1e6) µT | maximum RF amplitude |
| `GR_Δt` | $(sys.limits.GR_Δt * 1e6) µs | **gradient raster**: all gradient times (delay, rise, flat, fall) are multiples of it |
| `RF_Δt` | $(sys.limits.RF_Δt * 1e6) µs | RF raster: RF samples and delays |
| `ADC_Δt` | $(sys.limits.ADC_Δt * 1e9) ns | ADC raster: the dwell time is a multiple of it |
| `DUR_Δt` | $(sys.limits.DUR_Δt * 1e6) µs | block duration raster |
| `RF_dead_time` | $(sys.limits.RF_dead_time * 1e6) µs | minimum time between the block start and the RF start |
| `RF_ring_down_time` | $(sys.limits.RF_ring_down_time * 1e6) µs | minimum time between the RF end and the block end |
| `ADC_dead_time` | $(sys.limits.ADC_dead_time * 1e6) µs | minimum time before the first ADC sample and after the last one |

These are the values used in the Pulseq demo sequences for Siemens scanners, with conservative gradient limits (a 3 T clinical scanner typically allows 40–80 mT/m and 150–200 T/m/s). The dead times are explained in section 4.
"""

# ╔═╡ b2a7c918-c95c-11f1-8e28-537c2a69520e
md"""
# 3. Computing a trapezoidal gradient

## 3.1 The gradient area

A gradient moves the k-space position by its **area** (zeroth moment):

$$\Delta k = \bar{\gamma}\int G(t)\,dt$$

For a trapezoid of amplitude ``A``, rise time ``t_r``, flat time ``t_f`` and fall time ``t_d``:

$$\text{area} = A\left(t_f + \frac{t_r + t_d}{2}\right)$$

In Pulseq units (``A`` in Hz/m), the area is directly in 1/m, i.e. in k-space units.

## 3.2 The shortest gradient for a given area

To reach an area as fast as possible, we use the maximum slew rate ``S_{max}`` and, if possible, the maximum amplitude ``G_{max}``:

* **ramps**: ``t_r = t_d = A / S_{max}``;
* if the area is **small**, ``G_{max}`` is never reached: the gradient is a **triangle** with ``A = \sqrt{\text{area}\cdot S_{max}}`` and ``t_r = \sqrt{\text{area}/S_{max}}``. This happens when ``\text{area} \le G_{max}^2/S_{max}``;
* otherwise, it is a **trapezoid** with ``A = G_{max}``, ``t_r = G_{max}/S_{max}`` and ``t_f = \text{area}/G_{max} - t_r``.

## 3.3 The raster

The scanner updates the gradients on a fixed time grid, the **gradient raster** (`GR_Δt` = $(sys.limits.GR_Δt * 1e6) µs): ``t_r``, ``t_f`` and ``t_d`` must be multiples of it. The times computed above are therefore **rounded up** to the raster, and the amplitude is **recomputed** so that the area is exact:

$$A = \frac{\text{area}}{t_f + (t_r + t_d)/2}$$

(the amplitude is then slightly below ``G_{max}``). Here is a direct implementation:
"""

# ╔═╡ b2a7c92c-c95c-11f1-afb8-99fbd77fa14c
"""
    my_trapezoid(area; sys)

Shortest symmetric trapezoid with a given `area` (1/m), computed by hand.
Returns the amplitude (Hz/m) and the rise, flat and fall times (s).
"""
function my_trapezoid(area; sys=sys)
	Gmax = sys.limits.Gmax * γ̄               # Hz/m
	Smax = sys.limits.Smax * γ̄               # Hz/m/s
	Δt = sys.limits.GR_Δt
	ceil_raster(t) = ceil(t / Δt - 1e-9) * Δt   # round up to the gradient raster

	if abs(area) <= Gmax^2 / Smax
		# triangle : Gmax is never reached
		rise = ceil_raster(sqrt(abs(area) / Smax))
		flat = 0.0
	else
		# trapezoid : ramps at full slew rate, flat top at Gmax
		rise = ceil_raster(Gmax / Smax)
		flat = ceil_raster(abs(area) / Gmax - rise)
	end
	amplitude = area / (flat + rise)        # exact area : A (flat + (rise + fall)/2), fall = rise
	return (amplitude=amplitude, rise=rise, flat=flat, fall=rise)
end

# ╔═╡ b2a7c94a-c95c-11f1-88f1-1f4e98d61b7c
md"""
Area (1/m) : $(@bind area_demo Slider(-3000:50:3000, default=800, show_value=true))
"""

# ╔═╡ b2a7c954-c95c-11f1-9ba2-930890cad9f3
let
	m = my_trapezoid(area_demo)
	g = make_trapezoid(; area=area_demo, sys)    # KomaMRI (Pulseq units for plain numbers)

	f = Figure(size=(1000, 380))
	ax = Axis(f[1, 1], title="area = $(area_demo) 1/m : $(m.flat == 0 ? "triangle" : "trapezoid")", xlabel="time (µs)", ylabel="G (mT/m)")
	t = [0, m.rise, m.rise + m.flat, m.rise + m.flat + m.fall] .* 1e6
	lines!(ax, t, [0, m.amplitude, m.amplitude, 0] ./ γ̄ .* 1e3, linewidth=3, label="my_trapezoid")
	tk = [0, g.rise, g.rise + g.T, g.rise + g.T + g.fall] .* 1e6
	scatter!(ax, tk, [0, g.A, g.A, 0] .* 1e3, color=:red, markersize=12, label="make_trapezoid")
	# gradient raster
	raster = 0:sys.limits.GR_Δt*1e6:t[end]
	scatter!(ax, raster, zeros(length(raster)), color=:gray, markersize=4, label="gradient raster")
	hlines!(ax, [sys.limits.Gmax, -sys.limits.Gmax] .* 1e3, color=:black, linestyle=:dash)
	axislegend(ax, position=:rb)
	f
end

# ╔═╡ b2a7c968-c95c-11f1-b5a5-cd4613f3144b
let
	m = my_trapezoid(area_demo)
	g = make_trapezoid(; area=area_demo, sys)
	md"""
	|  | amplitude (Hz/m) | amplitude (mT/m) | rise (µs) | flat (µs) | fall (µs) | area (1/m) |
	|:--|--:|--:|--:|--:|--:|--:|
	| `my_trapezoid` | $(round(m.amplitude, digits=1)) | $(round(m.amplitude / γ̄ * 1e3, digits=3)) | $(round(m.rise * 1e6)) | $(round(m.flat * 1e6)) | $(round(m.fall * 1e6)) | $(round(m.amplitude * (m.flat + m.rise), digits=2)) |
	| `make_trapezoid` | $(round(g.A * γ̄, digits=1)) | $(round(g.A * 1e3, digits=3)) | $(round(g.rise * 1e6)) | $(round(g.T * 1e6)) | $(round(g.fall * 1e6)) | $(round(area(g) * γ̄, digits=2)) |
	"""
end

# ╔═╡ b2a7c99a-c95c-11f1-b049-379aa68a6ac7
md"""
## 3.4 Other ways to define a trapezoid

`make_trapezoid` accepts several combinations of parameters:

| call | use |
|:--|:--|
| `make_trapezoid(; area, sys)` | shortest gradient with a given area (spoilers...) |
| `make_trapezoid(; area, duration, sys)` | given area **and** total duration: several gradients in the same block can be synchronized (prephasers on x, y, z) |
| `make_trapezoid(; flat_area, flat_time, sys)` | given area **during the flat top** only: the **readout** gradient, where the ADC samples the flat top |
| `make_trapezoid(; amplitude, flat_time, sys)` | given amplitude (e.g. slice selection) |
"""

# ╔═╡ b2a7c9ae-c95c-11f1-9171-712569d1bf98
md"""
# 4. Dead times and ring-down

## 4.1 Why dead times?

The RF transmitter and the receiver are not ideal switches:

* **`RF_dead_time`** (≈ 100 µs): before playing an RF pulse, the system must *unblank* the RF power amplifier, switch the transmit/receive (T/R) switch to transmit and set the frequency and phase of the synthesizer. The RF pulse can therefore **not start at the beginning of a block**: its delay must be at least `RF_dead_time`. The gradients are not concerned: they can start during this time (e.g. the ramp of the slice-selection gradient).

* **`RF_ring_down_time`** (≈ 20 µs): after the pulse, the energy stored in the resonant RF coil decays exponentially ("ringing"), and the T/R switch must go back to receive. Nothing else may happen during this time: the **block must last at least `RF_ring_down_time` after the end of the RF**. Otherwise the next block (e.g. an ADC) would record the ringing.

* **`ADC_dead_time`** (≈ 10 µs): the receiver needs time to set its frequency/phase **before** the first sample and to process/transfer the data **after** the last one. The ADC must start at least `ADC_dead_time` after the beginning of the block and end at least `ADC_dead_time` before its end.

## 4.2 ADC samples: center of the dwell

In Pulseq, an ADC with `delay` $d$ and dwell $\Delta t$ divides the time into $N$ intervals and takes **one sample in the middle of each interval**:

$$t_i = d + \left(i - \tfrac{1}{2}\right)\Delta t, \qquad i = 1 \dots N$$

The dead time is counted from the beginning of the first interval ($d \ge$ `ADC_dead_time`). (KomaMRI stores the time of the first sample, i.e. `adc.delay` $= d + \Delta t/2$.)

## 4.3 What `make_sinc_pulse` and `make_adc` do for you

Let's create an excitation pulse and look at its timing:
"""

# ╔═╡ b2a7c9b8-c95c-11f1-afe3-7b8269c3c999
rf_ex, gz_ex, gzr_ex = make_sinc_pulse(15 * π / 180; duration=3e-3, slice_thickness=5e-3,
	apodization=0.5, time_bw_product=4, sys, use=Excitation());

# ╔═╡ b2a7c9c0-c95c-11f1-b02f-879108e5bfc9
md"""
| event | delay | duration | end |
|:--|--:|--:|--:|
| RF | $(round(rf_ex.delay * 1e6, digits=1)) µs | $(round(rf_ex.T * 1e6, digits=1)) µs | $(round((rf_ex.delay + rf_ex.T) * 1e6, digits=1)) µs (+ $(sys.limits.RF_ring_down_time * 1e6) µs ring-down = $(round(dur(rf_ex, sys) * 1e6, digits=1)) µs) |
| slice gradient | $(round(gz_ex.delay * 1e6, digits=1)) µs | rise $(round(gz_ex.rise * 1e6)) + flat $(round(gz_ex.T * 1e6)) + fall $(round(gz_ex.fall * 1e6)) µs | $(round(dur(gz_ex) * 1e6, digits=1)) µs |
"""

# ╔═╡ b2a7c9d6-c95c-11f1-9cff-17765a8dd7bb
let
	f = Figure(size=(1000, 420))
	ax = Axis(f[1, 1], title="excitation block", xlabel="time in the block (µs)", ylabel="|B1| (µT)")
	ax2 = Axis(f[1, 1], ylabel="Gz (mT/m)", yaxisposition=:right, ylabelcolor=:darkorange, yticklabelcolor=:darkorange)
	hidespines!(ax2); hidexdecorations!(ax2)

	rf_end = rf_ex.delay + rf_ex.T
	block_end = max(dur(gz_ex), dur(rf_ex, sys))
	vspan!(ax, [0], [sys.limits.RF_dead_time * 1e6], color=(:red, 0.2), label="RF dead time")
	vspan!(ax, [rf_end * 1e6], [(rf_end + sys.limits.RF_ring_down_time) * 1e6], color=(:purple, 0.3), label="RF ring-down")
	t_rf = rf_ex.delay .+ range(0, rf_ex.T, length(rf_ex.A))
	lines!(ax, t_rf .* 1e6, abs.(rf_ex.A) .* 1e6, color=:black, label="RF")
	g = gz_ex
	lines!(ax2, [0, g.delay, g.delay + g.rise, g.delay + g.rise + g.T, dur(g)] .* 1e6, [0, 0, g.A, g.A, 0] .* 1e3, color=:darkorange, linewidth=2)
	vlines!(ax, [block_end * 1e6], color=:gray, linestyle=:dash, label="block end")
	axislegend(ax, position=:lt)
	f
end

# ╔═╡ b2a7c9fe-c95c-11f1-99a2-437c95c5f6bb
md"""
## 4.4 Checking the timing: `check_timing`

`check_timing(seq, sys)` checks every block of a sequence: raster alignment, RF dead time, RF ring-down (the ring-down is counted as part of the RF event, which must fit in the block), ADC dead times. It returns `nothing` if everything is correct, and throws an error otherwise. `write_seq` runs the same checks before writing the file.

Let's build a few (wrong) blocks:
"""

# ╔═╡ b2a7ca12-c95c-11f1-9311-0d2387798c29
"Run `check_timing` on a sequence and display the result"
function timing_report(seq)
	try
		check_timing(seq, sys)
		return Markdown.MD(Markdown.Admonition("tip", "check_timing", [md"✔ the timing is correct"]))
	catch e
		return Markdown.MD(Markdown.Admonition("danger", "check_timing", [Markdown.Paragraph(["✘ " * sprint(showerror, e)])]))
	end
end

# ╔═╡ b2a7ca1c-c95c-11f1-b52b-fb0ae9d31373
md"""
**(a) RF without dead time**: we force the RF delay to 0.
"""

# ╔═╡ b2a7ca24-c95c-11f1-89d4-4b255badd83d
let
	rf_bad = deepcopy(rf_ex)
	rf_bad.delay = 0.0
	seq = Sequence(sys)
	@addblock seq += (rf_bad, make_delay(5e-3))
	timing_report(seq)
end

# ╔═╡ b2a7ca30-c95c-11f1-89e6-cf21b2dddef9
md"""
**(b) Block ending at the end of the RF**: a non-selective block where the duration (`make_delay`) stops right after the RF.
"""

# ╔═╡ b2a7ca44-c95c-11f1-b641-aba3fef475b1
let
	rf_end = rf_ex.delay + rf_ex.T
	block = ceil(rf_end / sys.limits.DUR_Δt) * sys.limits.DUR_Δt          # block end just after the RF
	seq = Sequence(sys)
	@addblock seq += (rf_ex, make_delay(block))
	md"""
	RF end : $(round(rf_end * 1e6, digits=1)) µs — block end : $(round(block * 1e6, digits=1)) µs

	$(timing_report(seq))
	"""
end

# ╔═╡ b2a7ca4e-c95c-11f1-95e5-6d9b6524cef4
md"""
**(c) Same block with the ring-down time**:
"""

# ╔═╡ b2a7ca58-c95c-11f1-99cd-a14f92735061
let
	block = ceil((rf_ex.delay + rf_ex.T + sys.limits.RF_ring_down_time) / sys.limits.DUR_Δt) * sys.limits.DUR_Δt
	seq = Sequence(sys)
	@addblock seq += (rf_ex, make_delay(block))
	timing_report(seq)
end

# ╔═╡ b2a7ca62-c95c-11f1-8d3e-61d62645f6bb
md"""
**(d) ADC ending at the end of the block**: an ADC of 64 samples (dwell 20 µs) in a block which stops exactly at the last sample.
"""

# ╔═╡ b2a7ca76-c95c-11f1-b029-d501ed25c85c
let
	adc = make_adc(64; dwell=20e-6, delay=sys.limits.ADC_dead_time, sys)
	block = sys.limits.ADC_dead_time + 64 * 20e-6
	seq = Sequence(sys)
	@addblock seq += (adc, make_delay(block))
	timing_report(seq)
end

# ╔═╡ b2a7ca96-c95c-11f1-a573-131ae63ea07b
md"""
# 5. Choosing the gradients for a given FOV and resolution

## 5.1 Readout gradient

During the readout, the k-space position increases linearly: $k_x(t) = \bar{\gamma} G_x t$. Between two ADC samples (dwell $\Delta t$), it moves by $\bar{\gamma} G_x \Delta t$, which must be equal to the k-space step $\Delta k = 1/FOV$ (basic reconstruction practical):

$$\bar{\gamma}\, G_x\, \Delta t = \frac{1}{FOV} \quad\Longrightarrow\quad G_x = \frac{1}{\bar{\gamma}\; FOV\; \Delta t}$$

The **resolution** is set by the total area of the flat top (the k-space extent):

$$N_x \Delta k = \frac{N_x}{FOV} = \frac{1}{\Delta x} \qquad \text{(flat area of the readout gradient)}$$

and the readout duration is $T_{acq} = N_x \Delta t$. The **receiver bandwidth** is

$$BW = \frac{1}{\Delta t}, \qquad BW_{pixel} = \frac{1}{N_x \Delta t} = \frac{1}{T_{acq}}$$

The choice is a trade-off: a high bandwidth gives a short readout (short TE, less distortion and chemical shift) but a stronger gradient and a lower SNR ($SNR \propto 1/\sqrt{BW}$). Since $G_x \le G_{max}$, there is a **minimum FOV** for a given dwell time: $FOV_{min} = 1/(\bar{\gamma} G_{max}\Delta t)$.

In Pulseq: `make_trapezoid(; flat_area=Nx/FOV, flat_time=Nx*dwell, sys)`, and `make_adc(Nx; duration=Nx*dwell, delay=gx.rise, sys)`.
"""

# ╔═╡ b2a7caa8-c95c-11f1-b640-9748fae0b005
md"""
FOV (mm) : $(@bind fov_ro Slider(50:10:400, default=256, show_value=true))

Matrix ``N_x`` : $(@bind n_ro Select([64, 128, 256, 512], default=128))

Bandwidth per pixel (Hz/pixel) : $(@bind bw_px Slider(50:10:1500, default=400, show_value=true))
"""

# ╔═╡ b2a7cab2-c95c-11f1-a8a3-21950f15c6cd
let
	FOVm = fov_ro * 1e-3
	dwell = round(1 / (bw_px * n_ro) / sys.limits.ADC_Δt) * sys.limits.ADC_Δt   # on the ADC raster
	G_Hz = 1 / (FOVm * dwell)                   # Hz/m
	G_mT = G_Hz / γ̄ * 1e3
	ok = G_Hz <= sys.limits.Gmax * γ̄
	readout = try
		flat = ceil(n_ro * dwell / sys.limits.GR_Δt - 1e-9) * sys.limits.GR_Δt   # flat time on the gradient raster
		g = make_trapezoid(; flat_area=n_ro / FOVm, flat_time=flat, sys)
		"$(round(dur(g) * 1e3, digits=3)) ms (ramps $(round(g.rise * 1e6)) µs)"
	catch e
		"✘ " * first(sprint(showerror, e), 150)
	end
	md"""
	| | |
	|:--|:--|
	| pixel size ``\Delta x = FOV/N_x`` | $(round(fov_ro / n_ro, digits=2)) mm |
	| k-space step ``\Delta k = 1/FOV`` | $(round(1 / FOVm, digits=2)) m⁻¹ |
	| flat area ``N_x/FOV`` | $(round(n_ro / FOVm, digits=1)) m⁻¹ |
	| dwell time ``\Delta t`` | $(round(dwell * 1e6, digits=2)) µs (receiver BW $(round(1 / dwell / 1e3, digits=1)) kHz) |
	| acquisition time ``T_{acq}`` | $(round(n_ro * dwell * 1e3, digits=3)) ms |
	| readout amplitude ``G_x`` | **$(round(G_mT, digits=2)) mT/m** = $(round(G_Hz / 1e3, digits=1)) kHz/m — $(round(100 * G_mT / (sys.limits.Gmax * 1e3), digits=1)) % of Gmax $(ok ? "✔" : "✘ **too high**")|
	| readout gradient (`make_trapezoid`) | $(readout) |
	| minimum FOV for this dwell | $(round(1 / (sys.limits.Gmax * γ̄ * dwell) * 1e3, digits=1)) mm |
	"""
end

# ╔═╡ b2a7cada-c95c-11f1-94e5-5b67829f2f8e
md"""
## 5.2 Readout prephaser: where is k = 0?

Before the readout, a **prephaser** gradient moves to $-k_{max}$ so that the echo ($k_x = 0$) occurs in the middle of the readout. The usual choice is

$$\text{area}_{pre} = -\frac{\text{area}(G_x)}{2}$$

(the total area of the readout, ramps included: the echo is then at the center of the flat top). But the ADC samples are in the **middle of the dwell intervals**: with $N_x$ even, the center of the flat top falls **between** samples $N_x/2$ and $N_x/2+1$, and $k = 0$ is **not sampled**.

The FFT used for the reconstruction expects samples at $k = -N/2, \dots, N/2 - 1$ (times $\Delta k$), with $k = 0$ at index $N/2 + 1$ (basic reconstruction practical, section 2.1). We therefore shift the prephaser by half a k-space step:

$$\text{area}_{pre} = -\frac{\text{area}(G_x)}{2} - \frac{\Delta k}{2}$$

Let's check with the k-space positions computed by KomaMRI (`get_kspace`) for $N_x = 8$:
"""

# ╔═╡ b2a7cae4-c95c-11f1-8612-2da66b1cdfc6
let
	FOVm, N = 0.256, 8
	Δk = 1 / FOVm
	gx = make_trapezoid(; flat_area=N * Δk, flat_time=N * 40e-6, sys)
	adc = make_adc(N; duration=gx.T, delay=gx.rise, sys)

	f = Figure(size=(1000, 300))
	ax = Axis(f[1, 1], xlabel="kₓ (1/Δk)", yticks=([1, 2], ["−area/2 − Δk/2", "−area/2"]), title="k-space position of the ADC samples")
	for (row, shift) in ((2, 0.0), (1, Δk / 2))
		gx_pre = make_trapezoid(; area=-area(gx) * γ̄ / 2 - shift, duration=1e-3, sys)
		seq = Sequence(sys)
		@addblock seq += (x=gx_pre,)
		@addblock seq += (x=gx, adc)
		_, k_adc = get_kspace(seq)
		scatter!(ax, k_adc[:, 1] ./ Δk, fill(row, N), markersize=14)
		text!(ax, k_adc[N÷2+1, 1] / Δk, row + 0.25, text="sample N/2+1", align=(:center, :bottom), fontsize=12)
	end
	vlines!(ax, [0], color=:black, linestyle=:dash)
	ylims!(ax, 0.5, 2.8)
	f
end

# ╔═╡ b2a7caee-c95c-11f1-ab9f-9f14b204ab1c
md"""
## 5.3 Phase encoding

Along ``y``, each TR acquires one k-space line: the phase-encoding gradient (played during the prephaser block) has an area

$$\text{area}_y(j) = \left(j - \frac{N_y}{2}\right)\Delta k_y, \qquad j = 0, \dots, N_y - 1$$

from ``-N_y\Delta k_y/2`` to ``(N_y/2 - 1)\Delta k_y``, again with ``k_y = 0`` at index ``N_y/2 + 1``. In practice we create **one** trapezoid for the largest area ``N_y\Delta k_y/2`` and **scale** its amplitude for each line (`scale * gy_pre`): the timing is identical for every TR.

## 5.4 Same block, same duration

The readout prephaser, the phase encoding and the slice rephaser are played **at the same time**, in the same block. To keep the block short, we compute the shortest duration needed by each of them, take the **longest**, and create all the gradients with this common `duration` (`make_trapezoid(; area, duration, sys)`).

## 5.5 Slice selection

`make_sinc_pulse` also computes the slice-selection gradient: a sinc of duration ``T_{rf}`` and time-bandwidth product ``TBW`` has a bandwidth ``BW_{rf} = TBW/T_{rf}``, and the slice thickness ``\Delta z`` requires

$$G_z = \frac{BW_{rf}}{\bar{\gamma}\,\Delta z}$$

and a **rephaser** of area ``-\frac{1}{2}`` × (area of ``G_z`` from the RF center to its end) (see the slice selection practical).
"""

# ╔═╡ b2a7cb02-c95c-11f1-8e23-f373e37fe610
md"""
For the excitation pulse of section 4.3 (``T_{rf}`` = 3 ms, TBW = 4, ``\Delta z`` = 5 mm): ``BW_{rf}`` = $(round(4 / 3e-3, digits=1)) Hz, expected ``G_z`` = $(round(4 / 3e-3 / 5e-3 / 1e3, digits=1)) kHz/m — `make_sinc_pulse` : $(round(gz_ex.A * γ̄ / 1e3, digits=1)) kHz/m = $(round(gz_ex.A * 1e3, digits=2)) mT/m.

Rephaser area : $(round(area(gzr_ex) * γ̄, digits=1)) m⁻¹ = $(round(area(gzr_ex) / area(gz_ex), digits=3)) × area(``G_z``).
"""

# ╔═╡ b2a7cb34-c95c-11f1-99b2-25ec21bd0719
md"""
# Part II — A gradient echo sequence

# 6. Gradient echo without spoiling

We now write a 2D **gradient echo** sequence. Each TR contains 4 blocks:

1. excitation: slice-selective sinc pulse + $G_z$,
2. prephasers: readout prephaser ($G_x$), phase encoding ($G_y$), slice rephaser ($G_z$),
3. a delay to reach the desired TE,
4. readout: $G_x$ + ADC,
5. a delay to reach the desired TR.

No spoiling is applied for now (no RF spoiling, no gradient spoiler): we will see the consequences with the simulation.

## 6.1 Parameters
"""

# ╔═╡ b2a7cb3e-c95c-11f1-aac0-2b86d26acaa3
begin
	FOV = [256e-3, 256e-3]             # m
	Nx, Ny = 32, 32         # matrix
	slice_thickness = 5e-3    # m
	flip_angle = 15           # degrees
	TE = 8e-3                 # s
	TR = 1000e-3                # s
	rf_duration = 3e-3        # s
	dwell = 20e-6             # s : BW = 1/(Nx dwell) = 390.6 Hz/pixel
	Δk = 1 ./ FOV              # m⁻¹
end;

# ╔═╡ b2a7cb48-c95c-11f1-a683-c180cbc4be6c
md"""
## 6.2 The events

**Excitation** (`make_sinc_pulse` returns the RF, the slice-selection gradient and its rephaser):
"""

# ╔═╡ b2a7cb52-c95c-11f1-8666-f12aba2c1e56
rf, gz, gz_reph = make_sinc_pulse(flip_angle * π / 180; duration=rf_duration, slice_thickness,
	apodization=0.5, time_bw_product=4, sys, use=Excitation());

# ╔═╡ b2a7cb5a-c95c-11f1-8bc0-ef9043b00f45
md"""
**Readout**: the flat area is set by the resolution and the flat time by the dwell time (section 5.1). Note the **double raster constraint**: the dwell time must be a multiple of the ADC raster (100 ns) **and** the flat time $N_x \Delta t$ a multiple of the gradient raster (10 µs). With $N_x = 128$, a bandwidth of exactly 400 Hz/pixel gives $N_x\Delta t$ = 2.5 ms but $\Delta t$ = 19.53 µs is not on the ADC raster; we choose $\Delta t$ = 20 µs (390.6 Hz/pixel, $T_{acq}$ = 2.56 ms). The ADC starts at the end of the ramp:
"""

# ╔═╡ b2a7cb66-c95c-11f1-85d7-73d8fbfb97c6
begin
	gx = make_trapezoid(; flat_area=Nx * Δk[1], flat_time=Nx * dwell, sys)
	adc = make_adc(Nx; dwell, delay=gx.rise, sys)
end;

# ╔═╡ b2a7cb70-c95c-11f1-a6f0-e349bc4a9fe3
md"""
**Prephasers** (sections 5.2 – 5.4): we compute the shortest duration of each gradient, and use the longest one for both.
"""

# ╔═╡ b2a7cb84-c95c-11f1-aeaa-e1a87ba5e519
begin
	area_gx_pre = -area(gx) * γ̄ / 2 - Δk[1] / 2     # 1/m, k = 0 on sample Nx/2 + 1
	area_gy_pre = Ny / 2 * Δk[2]                   # largest phase-encoding area (1/m)

	t_pre = max(dur(make_trapezoid(; area=area_gx_pre, sys)),
		dur(make_trapezoid(; area=area_gy_pre, sys)),
		dur(gz_reph))
	gx_pre = make_trapezoid(; area=area_gx_pre, duration=t_pre, sys)
	gy_pre = make_trapezoid(; area=area_gy_pre, duration=t_pre, sys)

	# scaling of gy_pre for each line : area = (j - Ny/2) Δk, j = 0 … Ny-1
	pe_scales = ((0:Ny-1) .- Ny / 2) ./ (Ny / 2)
end;

# ╔═╡ b2a7cb8c-c95c-11f1-baa4-03914e74062a
md"""
| event | amplitude | duration |
|:--|--:|--:|
| slice selection ``G_z`` | $(round(gz.A * 1e3, digits=2)) mT/m | $(round(dur(gz) * 1e3, digits=3)) ms |
| slice rephaser | $(round(gz_reph.A * 1e3, digits=2)) mT/m | $(round(dur(gz_reph) * 1e3, digits=3)) ms |
| readout ``G_x`` | $(round(gx.A * 1e3, digits=2)) mT/m | $(round(dur(gx) * 1e3, digits=3)) ms (dwell $(round(dwell * 1e6, digits=2)) µs) |
| readout prephaser | $(round(gx_pre.A * 1e3, digits=2)) mT/m | $(round(dur(gx_pre) * 1e3, digits=3)) ms |
| phase encoding (max) | $(round(gy_pre.A * 1e3, digits=2)) mT/m | $(round(dur(gy_pre) * 1e3, digits=3)) ms |
"""

# ╔═╡ b2a7cb98-c95c-11f1-bd4c-cfc9ae8bdb32
md"""
## 6.3 Timing: TE and TR

TE is the time between the **center of the RF pulse** and the **echo** (the sample at $k = 0$). To compute it, we first build the blocks **without** the TE and TR delays and read their durations (`seq.DUR`, which accounts for all the events of the block):

$$TE = \underbrace{\left(D_1 - t_{rf}\right)}_{\text{end of excitation}} + D_2 + \underbrace{d_{TE}}_{\text{TE delay}} + t_{echo}$$

with $D_1, D_2$ the durations of the excitation and prephaser blocks, $t_{rf}$ the time of the RF center in block 1, and $t_{echo}$ the time of the sample $N_x/2+1$ in the readout block. The minimum TE is obtained for $d_{TE} = 0$. Then

$$TR = D_1 + D_2 + d_{TE} + D_4 + d_{TR}$$
"""

# ╔═╡ b2a7cba2-c95c-11f1-b84f-afd731d7eb16
begin
	# one TR without delays, to measure the block durations
	tr_blocks = Sequence(sys)
	@addblock tr_blocks += (rf, z=gz)
	@addblock tr_blocks += (x=gx_pre, y=gy_pre, z=gz_reph)
	@addblock tr_blocks += (x=gx, adc)
	D1, D2, D4 = tr_blocks.DUR

	t_rf = rf.delay + rf.center                         # RF center in block 1
	t_echo = adc.delay + (Nx ÷ 2) * adc.T / (Nx - 1)    # sample Nx/2 + 1 in the readout block (KomaMRI : first sample at adc.delay)

	TE_min = (D1 - t_rf) + D2 + t_echo
	TR_min = D1 + D2 + (TE - TE_min) + D4
	@assert TE >= TE_min "TE too short : minimum TE = $(TE_min * 1e3) ms"
	@assert TR >= TR_min "TR too short : minimum TR = $(TR_min * 1e3) ms"

	delay_TE = round_to_raster(TE - TE_min, sys.limits.DUR_Δt)
	delay_TR = round_to_raster(TR - (D1 + D2 + delay_TE + D4), sys.limits.DUR_Δt)
end;

# ╔═╡ b2a7cbac-c95c-11f1-a44b-25cd878d14d2
md"""
| | |
|:--|--:|
| excitation block ``D_1`` | $(round(D1 * 1e3, digits=3)) ms (RF center at $(round(t_rf * 1e3, digits=3)) ms) |
| prephaser block ``D_2`` | $(round(D2 * 1e3, digits=3)) ms |
| readout block ``D_4`` | $(round(D4 * 1e3, digits=3)) ms (echo at $(round(t_echo * 1e3, digits=3)) ms) |
| **minimum TE** | **$(round(TE_min * 1e3, digits=3)) ms** |
| TE delay ``d_{TE}`` | $(round(delay_TE * 1e3, digits=3)) ms |
| TR delay ``d_{TR}`` | $(round(delay_TR * 1e3, digits=3)) ms |
"""

# ╔═╡ b2a7cbd4-c95c-11f1-bdad-81e11426b550
md"""
## 6.4 Building the sequence
"""

# ╔═╡ b2a7cbde-c95c-11f1-8229-775a3bdbee6e
begin
	seq = Sequence(sys)
	
	@addblock for scale in (pe_scales)		
		seq += (rf, z=gz)                                  # 1. excitation
		seq += (x=gx_pre, y=scale * gy_pre, z=gz_reph)     # 2. prephasers
		seq += make_delay(delay_TE)   					   # 3. TE delay
		seq += (x=gx, adc) 
		# 4. readout
		seq += make_delay(delay_TR)                        # 5. TR delay
	end
	seq
end

# ╔═╡ b2a7cbe8-c95c-11f1-b69e-9fd8c45f287d
timing_report(seq)

# ╔═╡ b2a7cbf2-c95c-11f1-a385-5b5368d941e2
md"""
The sequence diagram of the first two TRs (interactive plot: zoom with the mouse):
"""

# ╔═╡ b2a7cbfe-c95c-11f1-92c9-4d1a2fd1b794
plot_seq(seq; range=[0, 2 * TR * 1e3], height=450)

# ╔═╡ b2a7cc10-c95c-11f1-95b1-27e339be39ae
md"""
## 6.5 Checking the k-space trajectory

`get_kspace` integrates the gradients and returns the k-space position of every ADC sample. Each line should cover $k_x \in [-N_x/2, N_x/2 - 1]\,\Delta k$, and the lines should be separated by $\Delta k_y = \Delta k$:
"""

# ╔═╡ b2a7cc1a-c95c-11f1-9144-453f1b8e9b43
let
	k_traj, k_adc = get_kspace(seq)
	kx = k_adc[:, 1] ./ Δk[1]
	ky = k_adc[:, 2] ./ Δk[2]

	kx_traj = k_traj[:, 1] ./ Δk[1]
	ky_traj = k_traj[:, 2] ./ Δk[2]

	f = Figure(size=(800, 800))
	ax = Axis(f[1, 1], title="ADC samples (one dot out of 4 lines)", xlabel="kₓ / Δk", ylabel="k_y / Δk", aspect=DataAspect())
	lines!(ax, kx_traj, ky_traj,color=:blue)
	scatter!(ax, kx, ky, marker=:x,markersize=10,color=:red)
	f
end

# ╔═╡ 86833d7a-c977-11f1-84af-aae81d03624e
md"""
## 6.6 Verifying TE and TR

The TE and TR delays were computed with formulas (section 6.3). It is good practice to **measure** the timing on the final sequence, as we would do with pypulseq (`seq.get_block(i).block_duration`):

* **TR**: sum of the durations of the 5 blocks of each TR (`seq.DUR` contains the duration of every block);
* **TE**: time between the **center of each RF pulse** (start of the block, `get_block_start_times`, + RF delay + RF center) and the **echo**, i.e. the ADC sample $N_x/2+1$ of the following readout (`get_adc_sampling_times`). We also check with `get_kspace` that this sample is really at $k_x = 0$.

Note that TE can only be adjusted by steps of the **block duration raster** (`DUR_Δt` = 10 µs): the TE delay is rounded to this raster, so the measured TE may differ from the requested one by up to 5 µs.
"""

# ╔═╡ 86833e74-c977-11f1-84af-aae81d03624e
let
	n_blocks_TR = 5                                     # blocks in one TR
	t_block = get_block_start_times(seq)                # start time of each block (s)
	t_adc = get_adc_sampling_times(seq)                 # time of each ADC sample (s)
	_, k_adc = get_kspace(seq)

	# TR : sum of the durations of the blocks of each TR
	TR_measured = [sum(seq.DUR[(i-1)*n_blocks_TR+1:i*n_blocks_TR]) for i in 1:Ny]

	# TE : from the center of each RF pulse to the sample Nx/2 + 1 of the following readout
	rf_blocks = findall(i -> is_RF_on(seq.RF[1, i]), 1:length(seq))
	t_rf_centers = [t_block[i] + seq.RF[1, i].delay + seq.RF[1, i].center for i in rf_blocks]
	t_echoes = reshape(t_adc, Nx, Ny)[Nx÷2+1, :]
	TE_measured = t_echoes .- t_rf_centers

	# the sample used as echo must be at kx = 0
	kx_echo = reshape(k_adc[:, 1], Nx, Ny)[Nx÷2+1, :]

	@assert all(abs.(TR_measured .- TR) .< 1e-9) "TR is not good"
	@assert all(abs.(TE_measured .- TE) .<= sys.limits.DUR_Δt / 2) "TE is not good"
	@assert all(abs.(kx_echo) .< 1e-3 * Δk[1]) "the echo sample is not at kx = 0"

	md"""
	| | requested | measured (all the TRs) | max. error |
	|:--|--:|--:|--:|
	| TR | $(TR * 1e3) ms | $(round(minimum(TR_measured) * 1e3, digits=4)) – $(round(maximum(TR_measured) * 1e3, digits=4)) ms | $(round(maximum(abs.(TR_measured .- TR)) * 1e6, sigdigits=2)) µs |
	| TE | $(TE * 1e3) ms | $(round(minimum(TE_measured) * 1e3, digits=4)) – $(round(maximum(TE_measured) * 1e3, digits=4)) ms | $(round(maximum(abs.(TE_measured .- TE)) * 1e6, sigdigits=2)) µs |
	| ``k_x`` of the echo sample | 0 | max. abs. value : $(round(maximum(abs, kx_echo) / Δk[1], sigdigits=2)) ``\Delta k_x`` | |

	✔ $(length(rf_blocks)) TRs checked : TE and TR are correct.
	"""
end

# ╔═╡ b2a7cc24-c95c-11f1-9605-952e838cfd25
md"""
## 6.7 Writing the `.seq` file

`write_seq` checks the hardware limits and the timing again, then writes the file. The `[DEFINITIONS]` section stores the rasters and dead times, so that the interpreter on the scanner can check that the file matches its hardware.
"""

# ╔═╡ b2a7cc30-c95c-11f1-a2dc-cdc5fe014b84
begin
	seq_file = joinpath(mktempdir(), "gre_no_spoiling.seq")
	write_seq(seq, seq_file; sys)
	DownloadButton(read(seq_file), "gre_no_spoiling.seq")
end

# ╔═╡ b2a7cc42-c95c-11f1-905e-0b7badd5fc44
let
	lines = readlines(seq_file)
	i = findfirst(==("[BLOCKS]"), lines)
	Markdown.MD(Markdown.Code("", join(lines[1:i+8], "\n")))
end

# ╔═╡ 5de42076-c974-11f1-96f8-aae81d03624e
md"""
# Part III — Simulation with KomaMRI

# 7. Simulating the gradient echo

KomaMRI solves the **Bloch equations** for every spin of a digital phantom, with the **same `Sequence` object** as the one written in the `.seq` file: what we simulate is exactly what the scanner would play (gradients, RF, ADC timing, dead times...).

## 7.1 The phantom

A phantom is a set of spins (isochromats), each with a position ``(x, y, z)``, a proton density ``\rho``, relaxation times ``T_1``, ``T_2``, ``T_2^*`` and an off-resonance ``\Delta\omega`` (e.g. the chemical shift of fat). We first use the 2D brain phantom of KomaMRI with its **standard parameters** (`brain_phantom2D()`: one isochromat every 2 mm) and the default simulation parameters, with the long TR of section 6.1. We will see in section 8 when this density of isochromats is not sufficient.
"""

# ╔═╡ 5de420d0-c974-11f1-96f8-aae81d03624e
obj = brain_phantom2D()     # standard phantom : one isochromat every 2 mm

# ╔═╡ 5de42102-c974-11f1-96f8-aae81d03624e
let
	f = Figure(size=(1200, 420))
	for (j, (name, val, unit)) in enumerate([("ρ", obj.ρ, ""), ("T₁", obj.T1 .* 1e3, " (ms)"), ("T₂", obj.T2 .* 1e3, " (ms)")])
		ax = Axis(f[1, 2j-1], title=name * unit, aspect=DataAspect(), xlabel="x (mm)", ylabel="y (mm)")
		sc = scatter!(ax, obj.x .* 1e3, obj.y .* 1e3, color=val, markersize=2, colormap=:viridis)
		Colorbar(f[1, 2j], sc)
		# field of view of the sequence
		poly!(ax, Rect(-FOV[1] / 2 * 1e3, -FOV[2] / 2 * 1e3, FOV[1] * 1e3, FOV[2] * 1e3), color=:transparent, strokecolor=:red, strokewidth=2)
	end
	Label(f[0, :], "$(length(obj)) spins — red : FOV of the sequence ($(FOV[1] * 1e3) × $(FOV[2] * 1e3) mm)", fontsize=16)
	f
end

# ╔═╡ 5de42134-c974-11f1-96f8-aae81d03624e
md"""
## 7.2 Simulation

`simulate(obj, seq, sys)` returns the signal acquired by the ADC. With `return_type = "mat"`, the result is an array (samples × coils) in the order of acquisition: the $N_x$ samples of the first line, then the second line...
"""

# ╔═╡ 5de42148-c974-11f1-96f8-aae81d03624e
begin
	sim_params = KomaMRICore.default_sim_params()
	sim_params["return_type"] = "mat"
	signal = simulate(obj, seq, sys; sim_params)
end;

# ╔═╡ 5de4215c-c974-11f1-96f8-aae81d03624e
md"""
We can also simulate the **file** we wrote: `read_seq` reads the `.seq` file back into a `Sequence`. Both signals must be identical (up to the numerical precision of the file):
"""

# ╔═╡ 5de42170-c974-11f1-96f8-aae81d03624e
begin
	seq_from_file = read_seq(seq_file)
	signal_file = simulate(obj, seq_from_file, sys; sim_params)
	md"Relative difference between the simulation of `seq` and of the `.seq` file : $(round(maximum(abs, signal_file .- signal) / maximum(abs, signal), sigdigits=2))"
end

# ╔═╡ 5de42184-c974-11f1-96f8-aae81d03624e
md"""
## 7.3 Reconstruction

The acquisition is Cartesian: the signal is reshaped into an $N_x \times N_y$ k-space and reconstructed with a centered inverse FFT (basic reconstruction practical). The sample $N_x/2+1$ of the line $N_y/2+1$ is $k = 0$, as designed in sections 5.2 and 5.3.
"""

# ╔═╡ 5de42198-c974-11f1-96f8-aae81d03624e
begin
	"Centered 2D inverse FFT : k-space → image"
	ifft2c(k) = fftshift(ifft(ifftshift(k)))

	kspace_sim = reshape(signal[:, 1], Nx, Ny)    # column j = line j (phase-encoding step)
	img_sim = ifft2c(kspace_sim)
end;

# ╔═╡ 5de421ac-c974-11f1-96f8-aae81d03624e
let
	# pixel centers in mm (0 at index N/2 + 1)
	xs = ((1:Nx) .- (Nx ÷ 2 + 1)) .* (FOV[1] / Nx) .* 1e3
	ys = ((1:Ny) .- (Ny ÷ 2 + 1)) .* (FOV[2] / Ny) .* 1e3

	f = Figure(size=(1200, 450))
	ax = Axis(f[1, 1], title="k-space (log |signal|)", xlabel="kₓ / Δkₓ", ylabel="k_y / Δk_y", aspect=DataAspect())
	heatmap!(ax, (1:Nx) .- (Nx ÷ 2 + 1), (1:Ny) .- (Ny ÷ 2 + 1), log10.(abs.(kspace_sim) .+ 1e-3), colormap=:inferno)
	ax = Axis(f[1, 2], title="image $(Nx) × $(Ny) — pixel $(round(FOV[1] / Nx * 1e3, digits=1)) × $(round(FOV[2] / Ny * 1e3, digits=1)) mm", xlabel="x (mm)", ylabel="y (mm)", aspect=DataAspect())
	heatmap!(ax, xs, ys, abs.(img_sim), colormap=:grays)
	ax = Axis(f[1, 3], title="phantom (ρ) in the same FOV", xlabel="x (mm)", ylabel="y (mm)", aspect=DataAspect())
	scatter!(ax, obj.x .* 1e3, obj.y .* 1e3, color=obj.ρ, markersize=2, colormap=:grays)
	limits!(ax, -FOV[1] / 2 * 1e3, FOV[1] / 2 * 1e3, -FOV[2] / 2 * 1e3, FOV[2] / 2 * 1e3)
	f
end

# ╔═╡ 092d29d0-cd5f-11f1-86a2-aae81d03624e
let
	TRms = TR * 1e3
	tscan = round(Ny * TR, digits=1)
	tscan256 = round(256 * TR / 60, digits=1)
	md"""
	With TR = $(TRms) ms, the image is correct: TR is much longer than the ``T_2`` of the tissues (``T_2`` ≤ 330 ms in this phantom), so the transverse magnetization has **decayed before the next excitation**. Each TR starts again from a purely longitudinal magnetization: the relaxation acts as a **natural ``T_2`` spoiling**.
	
	But the acquisition lasts ``N_y \times TR`` = $(tscan) s for a 32 × 32 image, i.e. $(tscan256) min for 256 lines! To go faster, we must shorten TR... and then the transverse magnetization does not have time to disappear.
	"""
end

# ╔═╡ 092d2a34-cd5f-11f1-86a2-aae81d03624e
md"""
# 8. Short TR: spoiling

## 8.1 A short TR without spoiling

To build the different versions of the sequence, we write a function that reuses the events of section 6 and adds the options studied below (gradient spoiler, RF spoiling, dummy scans, labels). The TE is unchanged; the TR delay is recomputed for the new TR.
"""

# ╔═╡ 092d2a5c-cd5f-11f1-86a2-aae81d03624e
begin
	# gradient spoiler along z : n_spoil cycles of dephasing across the slice thickness
	n_spoil = 2
	gz_spoil = make_trapezoid(; area=n_spoil / slice_thickness, sys)

	"""
	    gre_sequence(; TR, flip=flip_angle, spoiler=false, rf_spoiling=false, φ0=117, n_dummy=0, labels=false, rep=nothing, order=1:Ny)

	Gradient echo with the events of section 6.
	- `spoiler` : gradient spoiler `gz_spoil` at the end of each TR
	- `rf_spoiling` : quadratic RF phase increment φₙ = φ0 n(n+1)/2 (degrees), applied to the RF and to the ADC
	- `n_dummy` : number of dummy scans (TRs without ADC) before the acquisition
	- `flip` : flip angle (degrees) : the RF pulse of section 6 is scaled
	- `labels` : Pulseq label LIN (line number) on each readout
	- `rep` : if given, the label REP (repetition) is also set to `rep`
	- `order` : order of acquisition of the phase-encoding lines
	"""
	function gre_sequence(; TR=TR, flip=flip_angle, spoiler=false, rf_spoiling=false, φ0=117, n_dummy=0, labels=false, rep=nothing, order=1:Ny)
		d_TR = round_to_raster(TR - (D1 + D2 + delay_TE + D4), sys.limits.DUR_Δt)
		@assert d_TR >= 0 "TR too short"
		@assert !spoiler || d_TR >= dur(gz_spoil) "TR too short for the spoiler"

		seq = Sequence(sys)
		lines = vcat(fill(0, n_dummy), collect(order))        # 0 = dummy scan
		for (n, j) in enumerate(lines)
			# RF spoiling : same phase for the RF pulse and the ADC
			φ = rf_spoiling ? deg2rad(φ0 * (n - 1) * n / 2) : 0.0
			rf_n = deepcopy(rf);  rf_n.ϕ = φ;  rf_n.A .*= flip / flip_angle
			adc_n = deepcopy(adc); adc_n.ϕ = φ
			scale = j == 0 ? 0.0 : pe_scales[j]

			@addblock seq += (rf_n, z=gz)
			@addblock seq += (x=gx_pre, y=scale * gy_pre, z=gz_reph)
			@addblock seq += make_delay(delay_TE)
			if j == 0
				@addblock seq += (make_delay(D4), x=gx)              # dummy scan : no ADC
			elseif labels && rep !== nothing
				@addblock seq += (x=gx, adc_n, make_label(:SET, :LIN, j - 1), make_label(:SET, :REP, rep))
			elseif labels
				@addblock seq += (x=gx, adc_n, make_label(:SET, :LIN, j - 1))
			else
				@addblock seq += (x=gx, adc_n)
			end
			if spoiler
				@addblock seq += (make_delay(d_TR), z=gz_spoil)
			else
				@addblock seq += make_delay(d_TR)
			end
		end
		return seq
	end

	"Image of a gradient echo simulation (signal returned with return_type = \"mat\")"
	gre_image(signal) = ifft2c(reshape(signal[:, 1], Nx, Ny))

	TR_short = 20e-3
end;

# ╔═╡ 092d2a70-cd5f-11f1-86a2-aae81d03624e
begin
	seq_short = gre_sequence(; TR=TR_short)
	img_short_std = gre_image(simulate(obj, seq_short, sys; sim_params))
end;

# ╔═╡ 092d2a7a-cd5f-11f1-86a2-aae81d03624e
let
	f = Figure(size=(900, 420))
	for (j, (im, t)) in enumerate([(img_sim, "TR = $(TR * 1e3) ms"), (img_short_std, "TR = $(TR_short * 1e3) ms, no spoiler")])
		ax = Axis(f[1, j], title=t, aspect=DataAspect())
		heatmap!(ax, abs.(im), colormap=:grays)
		hidedecorations!(ax)
	end
	Label(f[0, :], "standard phantom : $(length(obj)) spins", fontsize=16)
	f
end

# ╔═╡ 092d2a8e-cd5f-11f1-86a2-aae81d03624e
md"""
With a short TR, the image is corrupted. Part of this artefact is **physical**: the transverse magnetization left at the end of a TR is tilted by the next pulses and creates additional echoes (stimulated echoes) which add to the signal of the following lines.

But part of it is **numerical**. Our sequence is in fact **not balanced**: at each TR, the gradients leave a net area (moment) which dephases the spins inside each voxel:
"""

# ╔═╡ 092d2a98-cd5f-11f1-86a2-aae81d03624e
let
	m_x = (area(gx_pre) + area(gx)) * γ̄              # net readout area per TR (1/m)
	m_z = (area(gz) + area(gz_reph)) * γ̄             # net slice-selection area per TR (1/m)
	cyclesz = round(m_z * slice_thickness, digits=1)
	thicknessmm = slice_thickness * 1e3
	md"""
	| axis | net area per TR | dephasing across one voxel |
	|:--|--:|--:|
	| ``x`` (readout) | $(round(m_x, digits=1)) m⁻¹ | $(round(m_x * FOV[1] / Nx, digits=1)) cycles over a pixel of $(FOV[1] / Nx * 1e3) mm |
	| ``z`` (slice) | $(round(m_z, digits=1)) m⁻¹ | $(cyclesz) cycles over the slice thickness of $(thicknessmm) mm |
	"""
end

# ╔═╡ 092d2aa2-cd5f-11f1-86a2-aae81d03624e
md"""
In a real voxel, there is a **continuum** of spins: this dephasing partially destroys the transverse magnetization (it is a "natural" gradient spoiling). In the simulation, each voxel contains only a few isochromats: the standard phantom has a spin every 2 mm, i.e. 4 isochromats along each direction of an 8 mm pixel, and above all a **single** isochromat along ``z`` (2D phantom) for 2 cycles of dephasing across the slice. The dephasing is not sampled correctly and the simulation creates **artificial echoes**.

## 8.2 Increasing the density of isochromats

To simulate the dephasing inside the slice correctly, we create a phantom with several isochromats **along $z$**, distributed over the slice thickness (copies of the 2D phantom at different $z$, with $\rho$ divided by the number of copies):
"""

# ╔═╡ 092d2aac-cd5f-11f1-86a2-aae81d03624e
"""
    phantom_z(obj, Nz)

Copies of the 2D phantom `obj` at `Nz` positions along z, distributed over the slice thickness.
"""
function phantom_z(obj, Nz)
	zs = (((1:Nz) .- 0.5) ./ Nz .- 0.5) .* slice_thickness
	N = length(obj)
	return Phantom(name="brain, $(Nz) isochromats along z",
		x=repeat(obj.x, Nz), y=repeat(obj.y, Nz), z=repeat(zs, inner=N),
		ρ=repeat(obj.ρ, Nz) ./ Nz, T1=repeat(obj.T1, Nz), T2=repeat(obj.T2, Nz),
		T2s=repeat(obj.T2s, Nz), Δw=repeat(obj.Δw, Nz))
end

# ╔═╡ 092d2ac0-cd5f-11f1-86a2-aae81d03624e
begin
	Nz_iso = 20
	obj_z = phantom_z(obj, Nz_iso)
	img_short_z = gre_image(simulate(obj_z, seq_short, sys; sim_params))
end;

# ╔═╡ 092d2aca-cd5f-11f1-86a2-aae81d03624e
let
	f = Figure(size=(900, 420))
	for (j, (im, t)) in enumerate([(img_short_std, "$(length(obj)) spins (standard)"), (img_short_z, "$(length(obj_z)) spins ($(Nz_iso) along z)")])
		ax = Axis(f[1, j], title=t, aspect=DataAspect())
		heatmap!(ax, abs.(im), colormap=:grays)
		hidedecorations!(ax)
	end
	Label(f[0, :], "same sequence : TR = $(TR_short * 1e3) ms, no spoiler", fontsize=16)
	f
end

# ╔═╡ 092d2ae8-cd5f-11f1-86a2-aae81d03624e
md"""
## 8.3 Gradient spoiling and RF spoiling: a single-voxel experiment

The goal of a **spoiled** gradient echo (FLASH, SPGR) is to reach the steady state where **no transverse magnetization survives** from one TR to the next. The signal is then given by the **Ernst equation**:

$$S = M_0 \sin\alpha\,\frac{1 - E_1}{1 - E_1\cos\alpha}\,e^{-TE/T_2^*}, \qquad E_1 = e^{-TR/T_1}$$

Two tools are used:

* **gradient spoiling**: a strong gradient at the end of each TR dephases the spins over ``n`` cycles inside each voxel: area ``= n/\Delta z`` for a spoiler along ``z`` (here $(n_spoil) cycles, $(round(area(gz_spoil) * γ̄, digits=0)) m⁻¹). Alone, it is **not sufficient**: the dephasing is the same at every TR, and part of the magnetization is refocused by the following pulses (stimulated echoes).
* **RF spoiling**: the phase of the RF pulse changes at every TR with a **quadratic** increment, ``\varphi_n = \varphi_0\,n(n+1)/2`` with ``\varphi_0 = 117°`` (Zur et al., MRM 1991). The echoes of the previous excitations no longer add coherently. The ADC must be demodulated with the **same phase** (`adc.ϕ = φₙ`), otherwise the phase of the signal changes at each line.

To see the effect clearly, we simulate a **single voxel** (gray matter, ``T_1`` = 833 ms, ``T_2`` = 83 ms) described by ``N_{iso}`` isochromats along ``z``, with a non-selective pulse, and we follow the signal over 150 TRs.
"""

# ╔═╡ 092d2af2-cd5f-11f1-86a2-aae81d03624e
md"""
Isochromats in the voxel ``N_{iso}`` : $(@bind N_iso Slider([1, 4, 8, 16, 32, 64, 128, 256], default=64, show_value=true))

RF phase increment ``\varphi_0`` (°) : $(@bind φ0_demo Slider(0:1:180, default=117, show_value=true))
"""

# ╔═╡ 092d2b06-cd5f-11f1-86a2-aae81d03624e
voxel_signals = let
	T1v, T2v, n_TR = 0.833, 0.083, 150
	vox = Phantom(name="voxel", x=zeros(N_iso), y=zeros(N_iso),
		z=(((1:N_iso) .- 0.5) ./ N_iso .- 0.5) .* slice_thickness,
		ρ=fill(1 / N_iso, N_iso), T1=fill(T1v, N_iso), T2=fill(T2v, N_iso))

	rf_b = make_block_pulse(flip_angle * π / 180; duration=0.5e-3, sys)     # non-selective pulse
	adc_1 = make_adc(1; dwell=10e-6, delay=sys.limits.ADC_dead_time, sys)
	d_te = round_to_raster(TE - dur(rf_b, sys), sys.limits.DUR_Δt)
	d_tr = round_to_raster(TR_short - dur(rf_b, sys) - d_te - 100e-6, sys.limits.DUR_Δt)

	function voxel_seq(; spoiler, rf_spoiling)
		s = Sequence(sys)
		for n in 1:n_TR
			φ = rf_spoiling ? deg2rad(φ0_demo * (n - 1) * n / 2) : 0.0
			r = deepcopy(rf_b);  r.ϕ = φ
			a = deepcopy(adc_1); a.ϕ = φ
			@addblock s += (r,)
			@addblock s += make_delay(d_te)
			@addblock s += (a, make_delay(100e-6))
			if spoiler
				@addblock s += (make_delay(d_tr), z=gz_spoil)
			else
				@addblock s += make_delay(d_tr)
			end
		end
		return s
	end

	cases = [("no spoiling (no gradient)", false, false), ("gradient spoiler", true, false), ("gradient + RF spoiling", true, true)]
	signals = [abs.(vec(simulate(vox, voxel_seq(; spoiler=sp, rf_spoiling=rs), sys; sim_params)[:, 1])) for (_, sp, rs) in cases]

	E1 = exp(-TR_short / T1v)
	α = flip_angle * π / 180
	ernst = sin(α) * (1 - E1) / (1 - E1 * cos(α)) * exp(-TE / T2v)
	(names=first.(cases), signals=signals, ernst=ernst)
end;

# ╔═╡ 092d2b10-cd5f-11f1-86a2-aae81d03624e
let
	f = Figure(size=(1000, 420))
	ax = Axis(f[1, 1], title="single voxel, TR = $(TR_short * 1e3) ms, α = $(flip_angle)°, $(N_iso) isochromats, φ₀ = $(φ0_demo)°", xlabel="TR number", ylabel="|signal| (M₀ = 1)")
	for (name, s) in zip(voxel_signals.names, voxel_signals.signals)
		lines!(ax, 1:length(s), s, label=name * " — last : $(round(s[end], digits=4))", linewidth=2)
	end
	hlines!(ax, [voxel_signals.ernst], color=:black, linestyle=:dash, label="Ernst equation : $(round(voxel_signals.ernst, digits=4))")
	axislegend(ax, position=:rt)
	f
end

# ╔═╡ 092d2b2e-cd5f-11f1-86a2-aae81d03624e
md"""
## 8.4 Spoiled gradient echo images

We now add the gradient spoiler, then the RF spoiling, to the imaging sequence, and simulate them with the dense phantom (TR = $(TR_short * 1e3) ms):
"""

# ╔═╡ 092d2b38-cd5f-11f1-86a2-aae81d03624e
begin
	img_spoil_grad = gre_image(simulate(obj_z, gre_sequence(; TR=TR_short, spoiler=true), sys; sim_params))
	img_spoil_full = gre_image(simulate(obj_z, gre_sequence(; TR=TR_short, spoiler=true, rf_spoiling=true), sys; sim_params))
end;

# ╔═╡ 092d2b42-cd5f-11f1-86a2-aae81d03624e
let
	f = Figure(size=(1300, 400))
	ims = [(img_sim, "TR = $(TR * 1e3) ms (natural T₂ spoiling)"), (img_short_z, "TR = $(TR_short * 1e3) ms, no spoiler"),
		(img_spoil_grad, "TR = $(TR_short * 1e3) ms, gradient spoiler"), (img_spoil_full, "TR = $(TR_short * 1e3) ms, gradient + RF spoiling")]
	for (j, (im, t)) in enumerate(ims)
		ax = Axis(f[1, j], title=t, aspect=DataAspect(), titlesize=13)
		heatmap!(ax, abs.(im), colormap=:grays)
		hidedecorations!(ax)
	end
	f
end

# ╔═╡ 092d2b6a-cd5f-11f1-86a2-aae81d03624e
md"""
# 9. Dummy scans

## 9.1 The approach to the steady state

The Ernst equation describes the **steady state**. At the beginning of the acquisition, the magnetization starts from equilibrium ($M_z = M_0$): the first TRs give a much higher signal, and the signal converges towards the steady state with an apparent relaxation time

$$\frac{1}{T_1^*} = \frac{1}{T_1} - \frac{\ln(\cos\alpha)}{TR}$$

Let's look at the spoiled signal of each tissue over the TRs (single-voxel simulations, gradient + RF spoiling). We use 400 isochromats: with a long ``T_2`` (CSF), the transverse magnetization of many previous TRs survives, dephased by several times ``n_{spoil}`` cycles, and many isochromats are needed to sample these phases (with 64 isochromats the CSF curve oscillates artificially).
"""

# ╔═╡ 092d2b74-cd5f-11f1-86a2-aae81d03624e
transients = let
	n_TR = 250
	tissues = [("WM", 0.500, 0.070), ("GM", 0.833, 0.083), ("CSF", 2.569, 0.329)]
	N = 400        # many isochromats : with a long T₂ (CSF), the echoes of many previous TRs survive
	rf_b = make_block_pulse(flip_angle * π / 180; duration=0.5e-3, sys)
	adc_1 = make_adc(1; dwell=10e-6, delay=sys.limits.ADC_dead_time, sys)
	d_te = round_to_raster(TE - dur(rf_b, sys), sys.limits.DUR_Δt)
	d_tr = round_to_raster(TR_short - dur(rf_b, sys) - d_te - 100e-6, sys.limits.DUR_Δt)
	s = Sequence(sys)
	for n in 1:n_TR
		φ = deg2rad(117 * (n - 1) * n / 2)
		r = deepcopy(rf_b);  r.ϕ = φ
		a = deepcopy(adc_1); a.ϕ = φ
		@addblock s += (r,)
		@addblock s += make_delay(d_te)
		@addblock s += (a, make_delay(100e-6))
		@addblock s += (make_delay(d_tr), z=gz_spoil)
	end
	map(tissues) do (name, T1v, T2v)
		vox = Phantom(name=name, x=zeros(N), y=zeros(N), z=(((1:N) .- 0.5) ./ N .- 0.5) .* slice_thickness,
			ρ=fill(1 / N, N), T1=fill(T1v, N), T2=fill(T2v, N))
		sig = abs.(vec(simulate(vox, s, sys; sim_params)[:, 1]))
		T1s = 1 / (1 / T1v - log(cos(flip_angle * π / 180)) / TR_short)
		(name=name, signal=sig, T1_star=T1s)
	end
end;

# ╔═╡ 092d2b7e-cd5f-11f1-86a2-aae81d03624e
let
	f = Figure(size=(1000, 420))
	ax = Axis(f[1, 1], title="gradient + RF spoiled signal, TR = $(TR_short * 1e3) ms, α = $(flip_angle)°", xlabel="TR number", ylabel="|signal| (M₀ = 1)", yscale=log10)
	for t in transients
		lines!(ax, 1:length(t.signal), t.signal, linewidth=2, label="$(t.name) : T₁* = $(round(t.T1_star * 1e3)) ms ≈ $(round(t.T1_star / TR_short, digits=1)) TR")
	end
	vspan!(ax, [0.5], [Ny + 0.5], color=(:orange, 0.2), label="the $(Ny) lines of the image (no dummy scan)")
	axislegend(ax, position=:rt)
	f
end

# ╔═╡ 092d2b88-cd5f-11f1-86a2-aae81d03624e
md"""
## 9.2 Dummy scans

**Dummy scans** are TRs played **without acquisition** (no ADC) before the first line: the magnetization approaches the steady state before the k-space is acquired. In `gre_sequence`, they are TRs with a zero phase encoding and a delay instead of the ADC (`n_dummy`).
"""

# ╔═╡ 092d2b92-cd5f-11f1-86a2-aae81d03624e
md"""
Number of dummy scans : $(@bind n_dummy_demo Select([0, 10, 50, 150], default=50))
"""

# ╔═╡ 092d2ba6-cd5f-11f1-86a2-aae81d03624e
img_dummy = gre_image(simulate(obj_z, gre_sequence(; TR=TR_short, spoiler=true, rf_spoiling=true, n_dummy=n_dummy_demo), sys; sim_params));

# ╔═╡ 092d2bb0-cd5f-11f1-86a2-aae81d03624e
let
	f = Figure(size=(1200, 420))
	ax = Axis(f[1, 1], title="no dummy scan", aspect=DataAspect())
	heatmap!(ax, abs.(img_spoil_full), colormap=:grays); hidedecorations!(ax)
	ax = Axis(f[1, 2], title="$(n_dummy_demo) dummy scans", aspect=DataAspect())
	heatmap!(ax, abs.(img_dummy), colormap=:grays); hidedecorations!(ax)
	ax = Axis(f[1, 3], title="|difference|", aspect=DataAspect())
	heatmap!(ax, abs.(abs.(img_dummy) .- abs.(img_spoil_full)), colormap=:inferno); hidedecorations!(ax)
	f
end

# ╔═╡ 401e7702-7ff8-47b0-b310-761e3e59acf9
md"""
# 10. KomaMRI built-in functions

KomaMRI has a lot of built-in functions for visualization and reconstruction. Let's see them  rapidly.

The return_type can be "mat" or "raw" :
- mat stores only the signal
- raw creates a MRD format that can be used by MRIReco for the reconstruction
"""

# ╔═╡ 6afc141b-bafd-40e1-a865-958c8c8944f2
# a copy of the simulation parameters with return_type = "raw" (do not modify `sim_params` : it is used by other cells)
signal_raw = simulate(obj, seq, sys; sim_params=merge(sim_params, Dict("return_type" => "raw")))

# ╔═╡ f601219a-8b52-46f0-997b-ad9f83c479f5
plot_seq(seq; range=[0, to_SI(TR) * 1e3], slider=true, height=320)

# ╔═╡ d443eb8c-7062-4bf1-9578-52a590cb31de
 plot_kspace(seq; view_2d=true, width=400, height=400)

# ╔═╡ f4253c9c-9f5a-4802-b5e4-0607936616d1
plot_signal(signal_raw)

# ╔═╡ 092d31a0-cd5f-11f1-86a2-aae81d03624e
md"""
## 10.1 Automatic reconstruction

With `return_type = "raw"`, the simulation returns an ISMRMRD `RawAcquisitionData`. It can be converted to `AcquisitionData` and reconstructed by **MRIReco.jl** (`reconstruction`). For the data simulated by KomaMRI, the trajectory (the k-space position of each sample) is computed from the sequence: MRIReco uses it to place the samples.
"""

# ╔═╡ 718d4132-1f4e-4458-86e7-19569f00020a
let
	acq = AcquisitionData(signal_raw)
	rec_params = Dict{Symbol,Any}(:reco => "direct", :reconSize => (Nx, Ny))
	img_rec = reconstruction(acq, rec_params)
	plot_image(abs.(img_rec[:, :, 1]); title="MRIReco reconstruction")
end

# ╔═╡ 092d3696-cd5f-11f1-86a2-aae81d03624e
md"""
## 10.2 Labels

On the scanner, the raw data does not contain the k-space position of each sample: the reconstruction must know **which line** each readout corresponds to. In Pulseq, this information is given by **labels**, extensions attached to the blocks (`make_label`):

| label | meaning |
|:--|:--|
| `LIN` | phase-encoding line |
| `PAR` | partition (second phase encoding in 3D) |
| `SLC` | slice |
| `AVG`, `REP`, `SET`, `ECO`... | average, repetition, echo... |

`make_label(:SET, :LIN, j)` sets the counter to `j`, `make_label(:INC, :LIN, 1)` increments it. The value of the counters at the time of each ADC is written in the header of the corresponding profile (`profile.head.idx.kspace_encode_step_1` for `LIN`).

To show why they are needed, we acquire the lines in a **centric** order ($k_y = 0$ first, then $\pm\Delta k_y$, ...) with the `LIN` label:
"""

# ╔═╡ 092d36aa-cd5f-11f1-86a2-aae81d03624e
begin
	centric_order = vec(permutedims(hcat(Ny÷2+1:Ny, Ny÷2:-1:1)))   # 17, 16, 18, 15, ...
	seq_centric = gre_sequence(; TR=TR, labels=true, order=centric_order)
	raw_centric = simulate(obj, seq_centric, sys; sim_params=merge(sim_params, Dict("return_type" => "raw")))
end;

# ╔═╡ 092d36be-cd5f-11f1-86a2-aae81d03624e
md"""
Order of acquisition of the lines : $(join(centric_order[1:8], ", ")), …

`LIN` label of the first profiles : $(join([Int(p.head.idx.kspace_encode_step_1) for p in raw_centric.profiles[1:8]], ", ")), … (0-based)
"""

# ╔═╡ 092d36c8-cd5f-11f1-86a2-aae81d03624e
let
	# (a) without the labels : the profiles are put in the order of acquisition
	k_naive = reduce(hcat, [p.data[:, 1] for p in raw_centric.profiles])

	# (b) with the labels : each profile is put at its line
	k_sorted = zeros(ComplexF64, Nx, Ny)
	for p in raw_centric.profiles
		k_sorted[:, p.head.idx.kspace_encode_step_1+1] = p.data[:, 1]
	end

	# (c) MRIReco : uses the trajectory computed by KomaMRI
	img_mrireco = reconstruction(AcquisitionData(raw_centric), Dict{Symbol,Any}(:reco => "direct", :reconSize => (Nx, Ny)))

	f = Figure(size=(1200, 420))
	for (j, (im, t)) in enumerate([(ifft2c(k_naive), "(a) order of acquisition"), (ifft2c(k_sorted), "(b) sorted with the LIN label"), (img_mrireco[:, :, 1], "(c) MRIReco (trajectory)")])
		ax = Axis(f[1, j], title=t, aspect=DataAspect())
		heatmap!(ax, abs.(im), colormap=:grays)
		hidedecorations!(ax)
	end
	f
end

# ╔═╡ 092d36f0-cd5f-11f1-86a2-aae81d03624e
md"""
## 10.3 Splitting the data with labels

Labels are also used to **split the data** of a single sequence into several images: repetitions (`REP`), contrasts (`SET`, `ECO`), slices (`SLC`), averages (`AVG`)...

As an example, one sequence acquires the same slice twice with two different flip angles (5° and 30°, gradient + RF spoiled, 30 dummy scans each), the second acquisition being labeled `REP = 1`. The reconstruction reads the `REP` and `LIN` counters of each profile to build two k-spaces:
"""

# ╔═╡ 092d36fa-cd5f-11f1-86a2-aae81d03624e
begin
	flips_rep = [5, 30]
	seq_2contrasts = gre_sequence(; TR=TR_short, flip=flips_rep[1], spoiler=true, rf_spoiling=true, n_dummy=30, labels=true, rep=0) +
		gre_sequence(; TR=TR_short, flip=flips_rep[2], spoiler=true, rf_spoiling=true, n_dummy=30, labels=true, rep=1)
	raw_2contrasts = simulate(obj_z, seq_2contrasts, sys; sim_params=merge(sim_params, Dict("return_type" => "raw")))
end;

# ╔═╡ 092d3704-cd5f-11f1-86a2-aae81d03624e
let
	# one k-space per repetition, each profile at its line
	n_rep = maximum(p.head.idx.repetition for p in raw_2contrasts.profiles) + 1
	k_rep = zeros(ComplexF64, Nx, Ny, n_rep)
	for p in raw_2contrasts.profiles
		k_rep[:, p.head.idx.kspace_encode_step_1+1, p.head.idx.repetition+1] = p.data[:, 1]
	end

	f = Figure(size=(900, 420))
	for r in 1:n_rep
		ax = Axis(f[1, r], title="REP = $(r - 1) : α = $(flips_rep[r])°", aspect=DataAspect())
		heatmap!(ax, abs.(ifft2c(k_rep[:, :, r])), colormap=:grays)
		hidedecorations!(ax)
	end
	Label(f[0, :], "$(length(raw_2contrasts.profiles)) profiles split with the REP and LIN labels", fontsize=16)
	f
end

# ╔═╡ b2a7cc6a-c95c-11f1-a8e7-e95f1c97f46e
md"""
# Helper functions
"""

# ╔═╡ b2a7cc74-c95c-11f1-a21d-0d0a834a9b18
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

# ╔═╡ b2a7c97c-c95c-11f1-9636-7533730aa3f8
md"""
**Q1** With the limits of `sys`, compute by hand the shortest gradient for an area of 1000 m⁻¹. Check with the slider.

**Q2** Below which area is the shortest gradient a triangle? Check with the slider.

**Q3** Why is the amplitude of `make_trapezoid` slightly smaller than $G_{max}$ for a large area?

**Q4** In KomaMRI, what is stored in the field `g.T` of a trapezoid? In which unit is `g.A`?
""" |> question

# ╔═╡ b2a7c986-c95c-11f1-855e-a3c70c4c0605
answer_folded(md"""
**Q1** $G_{max} = 20$ mT/m $= 851.5$ kHz/m, $S_{max} = 100$ T/m/s $= 4.258\times10^6$ kHz/m/s. Rise time $G_{max}/S_{max} = 200$ µs (already on the raster). Flat time $= 1000/851\,550 - 200\ \text{µs} = 974$ µs, rounded up to **980 µs**. Amplitude $= 1000/(980 + 200)\ \text{µs} = 847.5$ kHz/m $= 19.90$ mT/m. Total duration: 1.38 ms.

**Q2** When $\text{area} \le G_{max}^2/S_{max} = 851\,550^2 / 4.258\times10^9 \approx 170$ m⁻¹.

**Q3** The times are rounded **up** to the 10 µs raster: the gradient is slightly longer than needed, so a slightly lower amplitude gives the exact area.

**Q4** `g.T` is the duration of the **flat top** (the total duration is `delay + rise + T + fall`, given by `dur(g)`), and `g.A` is in **T/m** (SI). Multiply by $\bar{\gamma}$ to get Hz/m.
""")

# ╔═╡ b2a7c9ea-c95c-11f1-ab03-237fa49d7a79
md"""
**Q5** Why is the RF delay $(round(rf_ex.delay * 1e6, digits=1)) µs and not 0? Why is it not exactly 100 µs?

**Q6** The slice-selection gradient starts *before* the RF dead time ends. Is this a problem? Why does it start at $(round(gz_ex.delay * 1e6)) µs?

**Q7** What sets the duration of this block? Would the ring-down time be a problem if the slice gradient were shorter?
""" |> question

# ╔═╡ b2a7c9f2-c95c-11f1-9be7-8fbf74634392
answer_folded(md"""
**Q5** `make_sinc_pulse` automatically sets the RF delay to (at least) `RF_dead_time` = 100 µs. The extra 0.5 µs comes from the RF raster: like the ADC, the RF samples are placed in the **middle** of each 1 µs raster interval, and KomaMRI stores the time of the first sample.

**Q6** No: the dead time only concerns the RF chain, the gradients can be played. The gradient is delayed so that its **ramp ends exactly when the RF starts** (delay = RF dead time − rise time = 100 − 70 = 30 µs): the RF is played on the flat top, as required for a correct slice profile.

**Q7** The longest event: here the fall of the slice gradient ends after RF end + ring-down. If the gradient ended earlier, the block would have to last at least until RF end + ring-down: the ring-down then directly increases the minimum TE.
""")

# ╔═╡ b2a7ca80-c95c-11f1-90a4-19d1896109c5
md"""
**Q8** Fix block (d). What is the shortest possible duration of this block?
""" |> question

# ╔═╡ b2a7ca8a-c95c-11f1-80c0-75b0f0f37730
answer_folded(md"""
**Q8** The block must contain: `ADC_dead_time` before the first dwell interval, the $64 \times 20$ µs of sampling and `ADC_dead_time` after: $10 + 1280 + 10 = 1300$ µs (already a multiple of the 10 µs block raster).

```julia
@addblock seq += (adc, make_delay(1300e-6))
```
""")

# ╔═╡ b2a7cac8-c95c-11f1-bbe1-6b9147fb0243
md"""
**Q9** For FOV = 256 mm, $N_x = 128$ and 400 Hz/pixel, compute by hand the dwell time and the readout amplitude in mT/m.

**Q10** Keep $N_x$ and the bandwidth and decrease the FOV. What happens? What can you change to reach FOV = 100 mm?

**Q11** At constant FOV and bandwidth per pixel, how does the readout amplitude change with $N_x$? And the readout duration?
""" |> question

# ╔═╡ b2a7cad0-c95c-11f1-b49a-4fe529f62e22
answer_folded(md"""
**Q9** $T_{acq} = 1/400 = 2.5$ ms, $\Delta t = 2.5\ \text{ms}/128 = 19.53$ µs (rounded to the 100 ns raster: 19.5 µs). $G_x = 1/(0.256 \times 19.5\times10^{-6}) = 200$ kHz/m $= 4.70$ mT/m.

**Q10** $G_x \propto 1/FOV$: below $FOV_{min}$ ≈ 59 mm (here) the required amplitude exceeds $G_{max}$. To reach a smaller FOV we must **increase the dwell time** (decrease the bandwidth): longer readout, more SNR, but longer TE.

**Q11** $G_x = BW_{pixel}\,N_x/(\bar{\gamma}\,FOV)$: doubling $N_x$ (higher resolution) at constant bandwidth per pixel **doubles** the amplitude; $T_{acq} = 1/BW_{pixel}$ is unchanged. At constant dwell time instead, the amplitude is unchanged but the readout lasts twice longer.
""")

# ╔═╡ b2a7cb0c-c95c-11f1-8445-d70d1b45d7e0
md"""
**Q12** The rephaser must cancel the area of $G_z$ played **after the center of the RF pulse**. Show that, for a pulse centered on the flat top and symmetric ramps, it is exactly $-\frac{1}{2}$ area($G_z$).
""" |> question

# ╔═╡ b2a7cb20-c95c-11f1-9667-e33b18fb0dff
answer_folded(md"""
**Q12** After the RF center, $G_z$ plays half of its flat top and its fall ramp: $A\,(t_f/2 + t_d/2)$. The total area is $A\,(t_f + t_r/2 + t_d/2)$. With $t_r = t_d$, half of it is $A\,(t_f/2 + t_r/2)$: the same value. This is no longer true for an asymmetric pulse (center not in the middle of the flat top) or different ramps: the rephaser must then be computed from the RF center (`center_pos` in `make_sinc_pulse`).
""")

# ╔═╡ b2a7cbc0-c95c-11f1-85ec-b9982577b5bd
md"""
**Q13** Which parameters would you change to decrease the minimum TE? List at least three, with their drawback.
""" |> question

# ╔═╡ b2a7cbca-c95c-11f1-89e4-eb4abf2d76e9
answer_folded(md"""
**Q13**
* a **shorter RF pulse** (or an asymmetric pulse with the center closer to its end): higher $G_z$ and B1 amplitude, worse slice profile for a shorter duration at constant TBW;
* a **higher bandwidth** (shorter readout, earlier echo): lower SNR;
* **stronger / faster gradients** for the prephasers (limited by $G_{max}$, $S_{max}$ and peripheral nerve stimulation);
* a **partial echo** (asymmetric readout: fewer samples before $k = 0$), reconstructed with partial Fourier;
* a lower resolution (smaller prephaser and phase-encoding areas).

The dead times and the ring-down are also part of the minimum TE but cannot be changed.
""")

# ╔═╡ 86833ea6-c977-11f1-84af-aae81d03624e
md"""
**Q15 bis** Set TE = 8.003 ms in section 6.1. What is the measured TE? Why? What happens if you set TE = 3 ms?
""" |> question

# ╔═╡ 86833eba-c977-11f1-84af-aae81d03624e
answer_folded(md"""
**Q15 bis** The TE delay is a block duration: it is rounded to the 10 µs block raster, so the measured TE is 8.00 ms (error 3 µs, accepted by the check). TE = 3 ms is shorter than the minimum TE: the `@assert` of section 6.3 stops the notebook with the minimum possible TE.
""")

# ╔═╡ b2a7cc4c-c95c-11f1-9fda-393344f7885d
md"""
**Q14** In the `[BLOCKS]` section, each line is a block. Identify the columns (duration, RF, GX, GY, GZ, ADC...). How many different gradient events (`[TRAP]` section) do you expect for ``N_y`` = $(Ny) phase-encoding lines? Check in the file.

**Q15** This sequence has no spoiling: what happens to the transverse magnetization at the end of each TR with a short TR (e.g. TR = 20 ms, ``T_2^*`` of the brain ≈ 50 ms)? And with the TR of section 6.1? What will it do to the image? (We will check it with the simulation.)
""" |> question

# ╔═╡ b2a7cc62-c95c-11f1-b2f5-7ffa87d0a69a
answer_folded(md"""
**Q14** The columns are: block number, block duration (in units of the block raster), RF, GX, GY, GZ, ADC and extension IDs (0 = no event). Pulseq stores each **distinct** event only once (in the `[RF]`, `[TRAP]`, `[ADC]` sections) and the blocks refer to them by their ID (0 = no event). The RF pulse, the ADC, the slice gradient, the slice rephaser, the readout and its prephaser are the same in every TR: one entry each. Each phase-encoding gradient has a different amplitude: ``N_y - 1`` entries (the central line has a zero amplitude, i.e. no event). In total ≈ ``N_y + 3`` trapezoids, while the file contains ``5 N_y`` blocks.

**Q15** With TR < T₂*, the transverse magnetization has not decayed at the end of the TR. It is excited again by the next pulse and contributes to the following echoes: the signal does not reach a simple spoiled steady state, and it depends on the phase accumulated during each TR (different for each phase-encoding step and each position). This produces **artefacts** (bands, ghosts) and an incorrect contrast. This is why gradient echo sequences use a **gradient spoiler** and **RF spoiling** (next parts). With a long TR (TR ≫ ``T_2``, e.g. 1 s), the transverse magnetization has decayed before the next pulse: the relaxation itself spoils the signal (section 7).
""")

# ╔═╡ 5de42120-c974-11f1-96f8-aae81d03624e
md"""
**Q16** Compare the size of the phantom with the FOV of the sequence (red rectangle). Is the FOV large enough? What would happen with ``FOV_y`` = 128 mm (try it in section 6.1)?
""" |> question

# ╔═╡ 5de421c0-c974-11f1-96f8-aae81d03624e
md"""
**Q17** The pixel size is ``FOV/N`` = 8 mm. How would you halve it, and what would it cost (gradients, scan time, SNR)?

**Q18** The image is reconstructed with a simple FFT. Which choice made in section 5.2 ensures that the image is centered, without a linear phase?
""" |> question

# ╔═╡ 5de421de-c974-11f1-96f8-aae81d03624e
answer_folded(md"""
**Q16** The phantom extends over ≈ 170 mm along ``x`` and ≈ 205 mm along ``y``: the FOV of 256 × 256 mm contains it, there is no fold-over. With ``FOV_y`` = 128 mm, the parts of the head outside the FOV **fold over** onto the opposite side of the image (basic reconstruction practical, section 6): the FOV along ``y`` must be at least ≈ 210 mm.

**Q17** Double ``N_x`` and ``N_y`` at the same FOV: the readout flat area (``N_x/FOV``) and the phase-encoding areas are doubled (stronger or longer gradients, longer readout or higher bandwidth, longer minimum TE), the scan time ``N_y \times TR`` is doubled and the SNR per voxel is divided by ≈ 2√2 (voxel 4× smaller, acquisition 2× longer).

**Q18** The readout prephaser was shifted by ``-\Delta k/2`` so that the sample ``N_x/2+1`` is exactly ``k_x = 0``, and the phase-encoding areas go from ``-N_y/2`` to ``N_y/2-1`` (in ``\Delta k_y``) with ``k_y = 0`` on line ``N_y/2+1``. This is the convention of the centered FFT: no half-pixel shift (linear phase) appears in the image.
""")

# ╔═╡ 07f5df3d-05f3-499a-973d-182e46028055
md"""
**Q22** Implement the 100 dummy scans and check the results.

**Q23** Increase the TR to 100 ms and 1000 ms. What happens on the image ? Can you explain what happens ?
""" |> question

# ╔═╡ fc16d7c2-12a2-43b0-909e-278ee5d69f53
answer_folded(md"""
**Q22** The image quality is still bad.

**Q23** For TR = 1000 ms, we obtain a "good" image quality.	The long TR creates a natural spoiling, all the transverse magnetization is spoiled due to the $T_2$ effect.	  
""")

# ╔═╡ 092d2ad4-cd5f-11f1-86a2-aae81d03624e
md"""
**Q19** The two images were obtained with **the same sequence**. Which one is closer to what the scanner would give? Why?

**Q20** How many isochromats per voxel are needed to simulate a dephasing of $n$ cycles across the voxel?
""" |> question

# ╔═╡ 092d2ade-cd5f-11f1-86a2-aae81d03624e
answer_folded(md"""
**Q19** The second one. On the scanner, each voxel contains billions of spins: the dephasing created by the unbalanced gradients is a continuous distribution of phases. In the simulation, the phases of a few isochromats can **re-align** after some TRs and create echoes which do not exist in reality. The artefact of the first image is mostly **numerical**.

**Q20** The phase must be sampled finely over each cycle: at least ≈ 5–10 isochromats per cycle, i.e. ≈ $10n$ isochromats along the direction of the dephasing (see the single-voxel experiment below). Here, $n_{spoil}$ = 2 cycles along $z$ with 20 isochromats.
""")

# ╔═╡ 092d2b1a-cd5f-11f1-86a2-aae81d03624e
md"""
**Q21** With 64 isochromats and $\varphi_0$ = 117°, compare the steady-state signal of the three cases with the Ernst equation. Which one is a "spoiled" gradient echo?

**Q22** Decrease the number of isochromats to 4 or 8. What happens to the gradient-spoiled signals? Relate it to section 8.2.

**Q23** Change $\varphi_0$. Is every value suitable? Try 0°, 90°, 117° and 180°.

**Q24** Without any gradient, the signal does not decay to the Ernst value. What is this sequence?
""" |> question

# ╔═╡ 092d2b24-cd5f-11f1-86a2-aae81d03624e
answer_folded(md"""
**Q21** Only **gradient + RF spoiling** reaches the Ernst value (≈ 0.098 vs 0.098). The gradient spoiler alone gives a different steady state (≈ 0.107, +9 %): the stimulated echoes still contribute. This difference depends on $T_2$: the contrast is no longer a pure $T_1$ weighting.

**Q22** With too few isochromats, the dephasing over $n$ cycles is badly sampled: the isochromats re-phase and the signals are wrong (the gradient + RF spoiled signal no longer matches the Ernst equation). The simulation needs enough isochromats per cycle of dephasing (≈ 10).

**Q23** No. With $\varphi_0$ = 0° there is no RF spoiling; with 180° the phase alternates and the sequence behaves like a balanced sequence. Only some increments (117°, 50°, ...) make the stimulated echoes add incoherently; 117° (Zur 1991) gives a signal very close to the Ernst equation for most $T_1$/$T_2$.

**Q24** All the isochromats see the same field: nothing dephases the transverse magnetization, which is fully refocused at each TR. This is a **balanced SSFP** (bSSFP / TrueFISP) on resonance: its steady state depends on $T_2/T_1$, not on the Ernst equation.
""")

# ╔═╡ 092d2b56-cd5f-11f1-86a2-aae81d03624e
md"""
**Q25** Compare the contrast of the long-TR image and of the spoiled short-TR image (gray matter, white matter, CSF). Explain with the Ernst equation.

**Q26** Why is the difference between the "no spoiler" and "gradient spoiler" images small in this simulation?
""" |> question

# ╔═╡ 092d2b60-cd5f-11f1-86a2-aae81d03624e
answer_folded(md"""
**Q25** With TR = 1 s ≫ TR of the spoiled sequence, $E_1 \approx 0$ for most tissues: the signal ≈ $M_0 \sin\alpha$, a **proton density** weighting (CSF, with the highest $\rho$, is bright). With TR = 20 ms and α = 15°, the longitudinal magnetization does not recover: tissues with a short ``T_1`` (white matter) give more signal than tissues with a long ``T_1`` (CSF, ``T_1`` ≈ 2.6 s, dark): **``T_1`` weighting**. In these images the ventricles are not dark yet: the lines are acquired during the approach to the steady state, when the CSF signal is still high (section 9, with dummy scans).

**Q26** Because the "unspoiled" sequence is already **not balanced**: the readout and the slice-selection gradients leave a net area at each TR (section 8.1), which acts as a gradient spoiler. Adding the spoiler only increases this dephasing. The main difference comes from the **RF spoiling**, which removes the contribution of the stimulated echoes (≈ 10 % on the signal).
""")

# ╔═╡ 092d2bba-cd5f-11f1-86a2-aae81d03624e
md"""
**Q27** Using the curves of section 9.1, how many dummy scans are needed for the white matter and the gray matter to be within a few percent of their steady state? And for the CSF? What does it cost?

**Q28** Without dummy scans, the line ``k_y = 0`` (TR number $(Ny ÷ 2 + 1)) is acquired during the transient. What is the effect on the contrast and on the image? Why is the problem worse with a **centric** phase-encoding order?
""" |> question

# ╔═╡ 092d2bce-cd5f-11f1-86a2-aae81d03624e
answer_folded(md"""
**Q27** The signal converges with a time constant ``T_1^*``: after ``3\,T_1^*`` it is within 5 % of the steady state. Here ``T_1^*`` ≈ 13 TR for the WM and 17 TR for the GM: ≈ 40–50 dummy scans. For the CSF (``T_1`` = 2.6 s), ``T_1^*`` ≈ 24 TR and its signal starts much higher than its steady state: ≈ 70 dummy scans are needed. Each dummy scan costs one TR (here 20 ms): 50 dummy scans = 1 s, negligible for a large matrix but not for a 32-line acquisition.

**Q28** The contrast of the image is mainly set by the center of the k-space: acquired during the transient, the signal is higher than the steady state, and the contrast is closer to a proton density weighting (the CSF is too bright). The modulation of the signal along $k_y$ also creates **blurring/ghosting** along the phase-encoding direction (the PSF is the Fourier transform of the modulation). With a centric order, the center of the k-space is acquired **first**, when the signal is the furthest from the steady state: dummy scans are essential.
""")

# ╔═╡ 092d36d2-cd5f-11f1-86a2-aae81d03624e
md"""
**Q29** Explain the three images. Why do we need the labels on a real scanner, although image (c) is correct without them?

**Q30** With dummy scans, the first TRs have no ADC. Do the labels change? Why is it important to put the label in the **readout** block?
""" |> question

# ╔═╡ 092d36e6-cd5f-11f1-86a2-aae81d03624e
answer_folded(md"""
**Q29** (a) The profiles are placed in the order of acquisition, as if the order were linear: the lines are scrambled and the image is meaningless. (b) Each profile is placed at the line given by its `LIN` label: the image is correct. (c) KomaMRI computes the k-space position of each sample from the sequence and MRIReco uses this trajectory. The raw data acquired on a scanner (e.g. through the Pulseq interpreter) **do not contain the trajectory**: the reconstruction (vendor, Gadgetron, your own script) relies on the counters (`LIN`, `PAR`, `SLC`, ...) to sort the data.

**Q30** No: dummy scans have no ADC, so no profile is written, and the label of each readout still gives its line. The label must be set in the block of the ADC (or before it): its value at the time of the ADC is written in the profile header. A label placed after the readout would be associated with the next line.
""")

# ╔═╡ 092d3718-cd5f-11f1-86a2-aae81d03624e
md"""
**Q31** Explain the contrast of the two images with the Ernst equation. Which image could be used to estimate a $T_1$ map, and how (variable flip angle method)?
""" |> question

# ╔═╡ 092d3722-cd5f-11f1-86a2-aae81d03624e
answer_folded(md"""
**Q31** At 5°, the signal $\approx M_0 \sin\alpha$ hardly depends on $T_1$ (proton density weighting: the CSF is bright). At 30°, the longitudinal magnetization of the tissues with a long $T_1$ does not recover between the pulses ($T_1$ weighting: the CSF is dark, the white matter bright). The **two** images together give $T_1$: writing the Ernst equation as $S/\sin\alpha = E_1\, S/\tan\alpha + M_0(1 - E_1)$, the slope between the two flip angles gives $E_1 = e^{-TR/T_1}$ in each voxel (DESPOT1 / variable flip angle method).
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
KomaMRI = "6a340f8b-2cdf-4c04-99be-4953d9b66d0a"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
FFTW = "~1.10.0"
GLMakie = "~0.13.13"
HypertextLiteral = "~1.0.0"
KomaMRI = "~0.14.0"
PlutoTeachingTools = "~0.4.7"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "e6d1d6527c371c84a213a230f564d2f116d11166"

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
git-tree-sha1 = "aa7d720632c3bf29efff118c744c67f8431dff66"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.6.0"

    [deps.Atomix.extensions]
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

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
git-tree-sha1 = "f947a8b5967714bf5b7d82070b647631de1f10d9"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "5.2.0"

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

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

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

[[deps.Electron]]
deps = ["Base64", "FilePaths", "JSON", "Pkg", "RelocatableFolders", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "a1fdd6f48f8519dcfe995ae93cd12f60cb2e1593"
uuid = "a1bb12fb-d4d1-54b4-b10a-ee7951ef7ad3"
version = "6.1.1"

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

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "1ac2813982db52b974c9343124ca61adbf297316"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.3"

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

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "0b2b3e8b8fb7c1f5ddbf498be2eb2b3b37c9c0ed"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.1"

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

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "EnumX", "PrecompileTools", "Random", "Reseau", "SHA", "URIs", "UUIDs", "Zlib_jll"]
git-tree-sha1 = "d6e390d1515836237b837fb7b6fca7ba1d752db6"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "2.6.5"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Preferences", "Static"]
git-tree-sha1 = "af9ab7d1f70739a47f03be78771ebda38c3c71bf"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.18"

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
weakdeps = ["Makie", "Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsMakieExt = "Makie"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

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

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "e51d16f9dcff134c4ecf79e7cfa764c59d2d648e"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.45"

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
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.KomaMRI]]
deps = ["Artifacts", "Bonito", "Electron", "FFTW", "KomaMRICore", "KomaMRIFiles", "KomaMRIPlots", "MAT", "MRIReco", "MsgPack", "Observables", "PlotlyBase", "PrecompileTools", "Preferences", "Reexport"]
git-tree-sha1 = "cd57064bc9fd256ddf7b6ee5088a3c680082ad95"
uuid = "6a340f8b-2cdf-4c04-99be-4953d9b66d0a"
version = "0.14.0"

[[deps.KomaMRIBase]]
deps = ["DSP", "FFTW", "Interpolations", "LinearAlgebra", "MAT", "MRIBase", "Parameters", "PrecompileTools", "Reexport"]
git-tree-sha1 = "95a0e40d4b370db2c110818bed9e889f640e5f3f"
uuid = "d0bc0b20-b151-4d03-b2a4-6ca51751cb9c"
version = "0.14.0"
weakdeps = ["Unitful"]

    [deps.KomaMRIBase.extensions]
    KomaMRIBaseUnitfulExt = "Unitful"

[[deps.KomaMRICore]]
deps = ["AcceleratedKernels", "Adapt", "Functors", "KernelAbstractions", "KomaMRIBase", "PrecompileTools", "ProgressMeter", "Reexport", "ThreadsX"]
git-tree-sha1 = "6bd66ee0e1f7ecad71123a691b36149446ca0bdd"
uuid = "4baa4f4d-2ae9-40db-8331-a7d1080e3f4e"
version = "0.13.0"

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
git-tree-sha1 = "7f4d93ba268774dedfb720a75ac611ac8476232f"
uuid = "fcf631a6-1c7e-4e88-9e64-b8888386d9dc"
version = "0.11.6"

[[deps.KomaMRIPlots]]
deps = ["Interpolations", "KomaMRIBase", "MAT", "PlotlyBase", "PlotlyKaleido", "QMRIColors"]
git-tree-sha1 = "97ab7ceb2a4f8ccbbfab195f6474984b88414235"
uuid = "76db0263-63f3-4d26-bb9a-5dba378db904"
version = "0.13.1"

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

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "23803295083335b8b122ad45adfd817a75c0671d"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "10.1.1"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "b5469a728c2f4ceab4b088025e89f7033bedd0b5"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.52+0"

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
deps = ["InteractiveUtils", "LinearAlgebra", "LinearOperators", "OhMyThreads", "Random", "Reexport", "SparseArrays"]
git-tree-sha1 = "7cb4604fcc046f068f2a926c38a80aa9aa378fc2"
uuid = "a4a2c56f-fead-462a-a3ab-85921a5f2575"
version = "2.4.2"

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
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "f2c8715d05bf10f9d4dc354e69dee30b6be53239"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.13"

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
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

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

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "c009236e222df68e554c7ce5c720e4a33cc0c23f"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.5.3"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

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
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

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

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

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
git-tree-sha1 = "143c6bc769ca26f5ba6047a4409423f523b5cdab"
uuid = "1e9c538a-f78c-5de5-8ffb-0b6dbe892d23"
version = "0.16.12"

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
git-tree-sha1 = "aaee74fab545e9af4b21d10c2203c93c636c4ae0"
uuid = "802f3686-a58f-41ce-bb0c-3c43c75bba36"
version = "1.4.1"

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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

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
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

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
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

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
# ╟─b2a6b80c-c95c-11f1-b594-532a3821be5c
# ╟─b2a7c80a-c95c-11f1-b301-7118b80de909
# ╟─b2a7c86e-c95c-11f1-adb2-e5af245fc1b8
# ╟─b2a7c882-c95c-11f1-95ad-7b6f6a918114
# ╠═b2a7c8dc-c95c-11f1-b376-1f4337d7986f
# ╟─b2a7c8ee-c95c-11f1-a495-131e2d17ea4b
# ╠═b2a7c8fa-c95c-11f1-b402-4bb9161851ee
# ╟─b2a7c90e-c95c-11f1-9565-81482728c9a2
# ╟─b2a7c918-c95c-11f1-8e28-537c2a69520e
# ╠═b2a7c92c-c95c-11f1-afb8-99fbd77fa14c
# ╠═b2a7c94a-c95c-11f1-88f1-1f4e98d61b7c
# ╠═b2a7c954-c95c-11f1-9ba2-930890cad9f3
# ╠═b2a7c968-c95c-11f1-b5a5-cd4613f3144b
# ╟─b2a7c97c-c95c-11f1-9636-7533730aa3f8
# ╟─b2a7c986-c95c-11f1-855e-a3c70c4c0605
# ╟─b2a7c99a-c95c-11f1-b049-379aa68a6ac7
# ╟─b2a7c9ae-c95c-11f1-9171-712569d1bf98
# ╠═b2a7c9b8-c95c-11f1-afe3-7b8269c3c999
# ╟─b2a7c9c0-c95c-11f1-b02f-879108e5bfc9
# ╠═b2a7c9d6-c95c-11f1-9cff-17765a8dd7bb
# ╟─b2a7c9ea-c95c-11f1-ab03-237fa49d7a79
# ╟─b2a7c9f2-c95c-11f1-9be7-8fbf74634392
# ╟─b2a7c9fe-c95c-11f1-99a2-437c95c5f6bb
# ╠═b2a7ca12-c95c-11f1-9311-0d2387798c29
# ╟─b2a7ca1c-c95c-11f1-b52b-fb0ae9d31373
# ╠═b2a7ca24-c95c-11f1-89d4-4b255badd83d
# ╟─b2a7ca30-c95c-11f1-89e6-cf21b2dddef9
# ╠═b2a7ca44-c95c-11f1-b641-aba3fef475b1
# ╟─b2a7ca4e-c95c-11f1-95e5-6d9b6524cef4
# ╠═b2a7ca58-c95c-11f1-99cd-a14f92735061
# ╟─b2a7ca62-c95c-11f1-8d3e-61d62645f6bb
# ╠═b2a7ca76-c95c-11f1-b029-d501ed25c85c
# ╟─b2a7ca80-c95c-11f1-90a4-19d1896109c5
# ╟─b2a7ca8a-c95c-11f1-80c0-75b0f0f37730
# ╟─b2a7ca96-c95c-11f1-a573-131ae63ea07b
# ╟─b2a7caa8-c95c-11f1-b640-9748fae0b005
# ╠═b2a7cab2-c95c-11f1-a8a3-21950f15c6cd
# ╟─b2a7cac8-c95c-11f1-bbe1-6b9147fb0243
# ╟─b2a7cad0-c95c-11f1-b49a-4fe529f62e22
# ╟─b2a7cada-c95c-11f1-94e5-5b67829f2f8e
# ╠═b2a7cae4-c95c-11f1-8612-2da66b1cdfc6
# ╟─b2a7caee-c95c-11f1-ab9f-9f14b204ab1c
# ╟─b2a7cb02-c95c-11f1-8e23-f373e37fe610
# ╟─b2a7cb0c-c95c-11f1-8445-d70d1b45d7e0
# ╟─b2a7cb20-c95c-11f1-9667-e33b18fb0dff
# ╟─b2a7cb34-c95c-11f1-99b2-25ec21bd0719
# ╠═b2a7cb3e-c95c-11f1-aac0-2b86d26acaa3
# ╟─b2a7cb48-c95c-11f1-a683-c180cbc4be6c
# ╠═b2a7cb52-c95c-11f1-8666-f12aba2c1e56
# ╟─b2a7cb5a-c95c-11f1-8bc0-ef9043b00f45
# ╠═b2a7cb66-c95c-11f1-85d7-73d8fbfb97c6
# ╟─b2a7cb70-c95c-11f1-a6f0-e349bc4a9fe3
# ╠═b2a7cb84-c95c-11f1-aeaa-e1a87ba5e519
# ╟─b2a7cb8c-c95c-11f1-baa4-03914e74062a
# ╟─b2a7cb98-c95c-11f1-bd4c-cfc9ae8bdb32
# ╠═b2a7cba2-c95c-11f1-b84f-afd731d7eb16
# ╠═b2a7cbac-c95c-11f1-a44b-25cd878d14d2
# ╟─b2a7cbc0-c95c-11f1-85ec-b9982577b5bd
# ╟─b2a7cbca-c95c-11f1-89e4-eb4abf2d76e9
# ╟─b2a7cbd4-c95c-11f1-bdad-81e11426b550
# ╠═b2a7cbde-c95c-11f1-8229-775a3bdbee6e
# ╠═b2a7cbe8-c95c-11f1-b69e-9fd8c45f287d
# ╟─b2a7cbf2-c95c-11f1-a385-5b5368d941e2
# ╠═b2a7cbfe-c95c-11f1-92c9-4d1a2fd1b794
# ╟─b2a7cc10-c95c-11f1-95b1-27e339be39ae
# ╠═b2a7cc1a-c95c-11f1-9144-453f1b8e9b43
# ╟─86833d7a-c977-11f1-84af-aae81d03624e
# ╠═86833e74-c977-11f1-84af-aae81d03624e
# ╟─86833ea6-c977-11f1-84af-aae81d03624e
# ╟─86833eba-c977-11f1-84af-aae81d03624e
# ╟─b2a7cc24-c95c-11f1-9605-952e838cfd25
# ╟─b2a7cc30-c95c-11f1-a2dc-cdc5fe014b84
# ╠═b2a7cc42-c95c-11f1-905e-0b7badd5fc44
# ╟─b2a7cc4c-c95c-11f1-9fda-393344f7885d
# ╟─b2a7cc62-c95c-11f1-b2f5-7ffa87d0a69a
# ╟─5de42076-c974-11f1-96f8-aae81d03624e
# ╠═5de420d0-c974-11f1-96f8-aae81d03624e
# ╠═5de42102-c974-11f1-96f8-aae81d03624e
# ╟─5de42120-c974-11f1-96f8-aae81d03624e
# ╟─5de42134-c974-11f1-96f8-aae81d03624e
# ╠═5de42148-c974-11f1-96f8-aae81d03624e
# ╟─5de4215c-c974-11f1-96f8-aae81d03624e
# ╠═5de42170-c974-11f1-96f8-aae81d03624e
# ╟─5de42184-c974-11f1-96f8-aae81d03624e
# ╠═5de42198-c974-11f1-96f8-aae81d03624e
# ╠═5de421ac-c974-11f1-96f8-aae81d03624e
# ╠═5de421c0-c974-11f1-96f8-aae81d03624e
# ╟─5de421de-c974-11f1-96f8-aae81d03624e
# ╟─07f5df3d-05f3-499a-973d-182e46028055
# ╟─fc16d7c2-12a2-43b0-909e-278ee5d69f53
# ╟─092d29d0-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2a34-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2a5c-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2a70-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2a7a-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2a8e-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2a98-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2aa2-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2aac-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2ac0-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2aca-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2ad4-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2ade-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2ae8-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2af2-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2b06-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2b10-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2b1a-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2b24-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2b2e-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2b38-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2b42-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2b56-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2b60-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2b6a-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2b74-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2b7e-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2b88-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2b92-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2ba6-cd5f-11f1-86a2-aae81d03624e
# ╠═092d2bb0-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2bba-cd5f-11f1-86a2-aae81d03624e
# ╟─092d2bce-cd5f-11f1-86a2-aae81d03624e
# ╟─401e7702-7ff8-47b0-b310-761e3e59acf9
# ╟─6afc141b-bafd-40e1-a865-958c8c8944f2
# ╠═f601219a-8b52-46f0-997b-ad9f83c479f5
# ╠═d443eb8c-7062-4bf1-9578-52a590cb31de
# ╠═f4253c9c-9f5a-4802-b5e4-0607936616d1
# ╟─092d31a0-cd5f-11f1-86a2-aae81d03624e
# ╠═718d4132-1f4e-4458-86e7-19569f00020a
# ╟─092d3696-cd5f-11f1-86a2-aae81d03624e
# ╠═092d36aa-cd5f-11f1-86a2-aae81d03624e
# ╠═092d36be-cd5f-11f1-86a2-aae81d03624e
# ╠═092d36c8-cd5f-11f1-86a2-aae81d03624e
# ╟─092d36d2-cd5f-11f1-86a2-aae81d03624e
# ╟─092d36e6-cd5f-11f1-86a2-aae81d03624e
# ╟─092d36f0-cd5f-11f1-86a2-aae81d03624e
# ╠═092d36fa-cd5f-11f1-86a2-aae81d03624e
# ╠═092d3704-cd5f-11f1-86a2-aae81d03624e
# ╟─092d3718-cd5f-11f1-86a2-aae81d03624e
# ╟─092d3722-cd5f-11f1-86a2-aae81d03624e
# ╟─b2a7cc6a-c95c-11f1-a8e7-e95f1c97f46e
# ╟─b2a7cc74-c95c-11f1-a21d-0d0a834a9b18
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
