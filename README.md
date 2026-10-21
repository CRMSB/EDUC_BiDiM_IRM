# EDUC_BiDiM_IRM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://aTrotier.github.io/EDUC_BiDiM_IRM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://aTrotier.github.io/EDUC_BiDiM_IRM.jl/dev/)
[![Build Status](https://github.com/aTrotier/EDUC_BiDiM_IRM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/aTrotier/EDUC_BiDiM_IRM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/aTrotier/EDUC_BiDiM_IRM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/aTrotier/EDUC_BiDiM_IRM.jl)


## How to update the website with this repository
### Write a course

An example is available in `src/template.jl`

### Pluto notebooks
Pluto notebooks are stored in the `examples` folder.

1. They should be converted into a html 
```julia
using Pkg
Pkg.activate("docs")
using 
PlutoSliderServer.export_directory("examples/")
```
and the content of the example folder should be copied to `docs/src/public`.

2. Each html should be included in a markdown page (see `docs/1-2-installation.md`)
````
```@raw html
<div style="height: 100vh; width: 100%; overflow: auto; border: 1px solid #ddd; border-radius: 8px;">
  <iframe src="./pluto-1-2-installation.html" style="width: 100%; height: 100%; border: 0;"></iframe>
</div>
```
````
3. the .md files should be added in the `make.jl`

### Jupyter notebooks
Jupyter notebooks (`.ipynb`) are also stored in the `examples` folder (use a name without spaces).

1. Export them into html (written to `docs/src/public/jupyter`):
```bash
docs/export_jupyter.sh                              # all the notebooks of examples/
docs/export_jupyter.sh examples/my_notebook.ipynb   # only one notebook
docs/export_jupyter.sh --execute ...                # re-run the notebook before the export
```
The script uses `uvx` (from [uv](https://docs.astral.sh/uv/)) to run `nbconvert` in a temporary environment. Without `--execute`, the outputs saved in the notebook are exported: run and save the notebook first.

2. Each html should be included in a markdown page, with a link to open the notebook in Google Colab (see `docs/src/4-2-hands_on_bruker_pulseq.md`):
````
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CRMSB/EDUC_BiDiM_IRM/blob/main/examples/my_notebook.ipynb)

```@raw html
<div style="height: 100vh; width: 100%; overflow: auto; border: 1px solid #ddd; border-radius: 8px;">
  <iframe src="./jupyter/my_notebook.html" style="width: 100%; height: 100%; border: 0;"></iframe>
</div>
```
````
3. the .md files should be added in the `make.jl`

## Check the results

```julia
using LiveServer
servedocs(; buildfoldername="build/1")
```


