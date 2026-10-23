# AGENTS.md — working on EDUC_BiDiM_IRM

Teaching material (Graduate Program – Biomedical MRI, CRMSB Bordeaux) published as a
VitePress website built with Documenter. The courses are **Pluto notebooks** (and a few
Jupyter notebooks) embedded in documentation pages. Courses are written in **English**.

## Repository layout

| path | content |
|:--|:--|
| `examples/` | the notebooks (source of truth): `pluto-X-Y-name.jl`, `jupyter-X-Y-name.ipynb`, and their exported `.html` |
| `docs/src/*.md` | one page per course, embedding the exported HTML in an `<iframe>` |
| `docs/src/public/pluto/` | exported Pluto HTML + `.jl` copies (served at `./pluto/...`) |
| `docs/src/public/jupyter/` | exported Jupyter HTML (served at `./jupyter/...`) |
| `docs/make.jl` | Documenter config: the `pages=[...]` list is the website menu |
| `docs/export_jupyter.sh` | Jupyter → HTML export script (nbconvert through `uvx`) |
| `src/template.jl` | Pluto template for a new course (header, helper functions) |
| `ressources/` | raw material (data, old scripts, figures) — **untracked**, not published |

Numbering: `X-Y` = part-lesson (1 = overview, 2 = NMR/MRI simulation, 3 = reconstruction,
4 = sequence programming). Variants use a letter, e.g. `4-1b` = solution of `4-1`.

## How the website is deployed

1. **Notebooks are NOT executed in CI.** Their HTML must be exported locally and
   committed in `docs/src/public/` (see below).
2. CI (`.github/workflows/CI.yml`, job `docs`) runs `docs/make.jl` and deploys to the
   `gh-pages` branch on every push to `main` (`devurl = "dev"`).
3. So: export HTML → copy to `docs/src/public/` → add/update the `.md` page → add it to
   `make.jl` → check locally → commit and push (**only when the user asks**).

### Export a Pluto notebook

Export **only the notebook you changed** (`export_directory` re-runs every notebook of
`examples/`, which takes a very long time):

```julia
# from the repository root
using Pkg; Pkg.activate("docs")
using PlutoSliderServer
PlutoSliderServer.export_notebook("examples/pluto-4-1b-pulseq_practical.jl")   # → examples/*.html
```

Then copy the `.html` **and** the `.jl` to `docs/src/public/pluto/`. Do not overwrite
`index.html` / `pluto_export.json` unless all the notebooks were re-exported.

To export everything (slow): `PlutoSliderServer.export_directory("examples/")`, then copy
the content of `examples/` to `docs/src/public/pluto/`.

### Export a Jupyter notebook

```bash
docs/export_jupyter.sh examples/my_notebook.ipynb      # outputs saved in the notebook
docs/export_jupyter.sh --execute examples/my.ipynb     # re-run first (kernel deps needed)
```

Names without spaces. Add an "Open in Colab" badge on its page
(`https://colab.research.google.com/github/CRMSB/EDUC_BiDiM_IRM/blob/main/examples/<file>.ipynb`).

### Documentation page

````markdown
```@raw html
<div style="height: 100vh; width: 100%; overflow: auto; border: 1px solid #ddd; border-radius: 8px;">
  <iframe src="./pluto/pluto-X-Y-name.html" style="width: 100%; height: 100%; border: 0;"></iframe>
</div>
```
````

A solution can be put **below** the exercise on the same page (`## Solution` + a second
iframe) instead of a separate menu entry. Add new pages to `pages=[...]` in `docs/make.jl`.

### Check locally

```julia
julia --project=docs docs/make.jl          # full build (the deployment step is skipped locally)
# or, with live reload:
using LiveServer; servedocs(; buildfoldername="build/1")
```

The built site is in `docs/build/1/` (e.g. `docs/build/1/pluto/` must contain the HTML).

## Writing / editing Pluto notebooks

- Start from `src/template.jl` (title HTML block, `ChooseDisplayMode()`, packages cell with
  `PlutoUI.TableOfContents()`, helper functions at the end: `question`, `answer_folded`...).
- Questions: `md"""**Q1** ...""" |> question`; answers in `answer_folded(md"""...""")`.
  For exercises use PlutoTeachingTools checks: `correct()`, `keep_working(; text=md"...")`,
  `still_missing()`, and `missing` placeholders marked 👉.
- Notebooks are **self-contained** (packages are resolved by Pluto from the embedded
  Project/Manifest). Do not add packages with `Pkg` in a notebook environment activated with
  `Pluto.activate_notebook_environment` — it is synced with the notebook file.
- When the user edits a notebook in Pluto, edit **their file** cell by cell (by cell UUID);
  do not regenerate it from an old source. Keep a backup copy before large edits.
- Plotting: prefer **CairoMakie** for new notebooks (GLMakie needs an OpenGL context and
  crashes when the screen is locked / on headless runs; it is used in the older notebooks).
  `PlutoUI` and Makie both export `Slider`: add `using PlutoUI: Slider`.
- Pluto rules: one global definition per name across cells; do not mutate a global defined
  in another cell (e.g. use `merge(sim_params, Dict(...))`); functions referencing globals
  create dependencies — avoid cycles (put small generic helpers in their own cell).

### Markdown interpolation pitfalls (`md"""..."""`)

- Inside a cell with `$(...)` interpolation, write inline math as ``` ``x`` ``` (not `$x$`).
- Two `_` or two `*` in different interpolations of the same paragraph are parsed as
  emphasis → precompute values in a `let` with simple names (no underscores).
- `($(` (parenthesis right before an interpolation) can break the interpolation: reword.
- A `|` inside a table cell breaks the table (`|k_x|`): write "abs. value".
- Do not put a Makie figure inside a `Markdown.MD`: return it from its own cell.

## Running and testing notebooks headless

- Run a notebook with Pluto (outputs, errors) — e.g. a small script with
  `Pluto.ServerSession()` + `Pluto.SessionActions.open(session, path; run_async=false)`,
  then inspect `c.errored` / `c.output.body` of each cell.
- Run it as a script: `Pluto.activate_notebook_environment(nb); include(nb)` (cells are in
  topological order in the file). Paths must be absolute; `@bind` takes default values.
- An exported HTML embeds the notebook state (base64 msgpack, `data:;base64,...`):
  decode with `Pluto.unpack` to check that no cell errored.
- Long simulations stop while the Mac sleeps; check `uptime`/elapsed times before
  concluding that something hangs.

## Domain notes (KomaMRI ≥ 0.14, Pulseq)

- Pulseq-style API in `KomaMRI.PulseDesigner` (`make_sinc_pulse`, `make_trapezoid`,
  `make_adc`, `make_delay`, `make_label`); plain numbers are Pulseq units (Hz/m, 1/m),
  events are stored in SI (T/m). `g.T` = flat-top duration; `dur(g)` = total.
- `@addblock seq += (rf, z=gz)`; the tuple syntax only works directly after `+=`
  (no ternary). `check_timing(seq, sys)` throws on dead-time / ring-down / raster errors;
  `write_seq` also checks the hardware limits (B1, Gmax, Smax).
- ADC samples are at the centre of each dwell (`adc.delay` in Koma = Pulseq delay + dwell/2).
  Gradient echo: prephaser `-area(gx)/2 - Δk/2` puts sample N/2+1 at k = 0. Spin echo: keep
  the echo at the centre of the readout and shift the ADC by −dwell/2 instead (CPMG).
- Labels (`LIN`, `ECO`, `REP`...) end up in `raw.profiles[i].head.idx`
  (`kspace_encode_step_1`, `contrast`, `repetition`). MRIReco's reconstruction of KomaMRI
  raw data uses the trajectory, not the labels.
- Spoiling/crushers need many isochromats per voxel along the dephasing direction
  (≈ 10 per cycle; build a phantom with copies along z). Brain phantom: drop `Δw`/`T2s`
  when they are not needed (simulation ~300× slower with them).

## Conventions

- Answer the user in the language they write in; course content in English.
- Do not commit or push unless asked. Do not delete user files; keep backups of edited
  notebooks outside the repository.
