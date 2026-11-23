# Practical — writing an EPI sequence

Write a single-shot gradient-echo EPI sequence with Pulseq (KomaMRI), simulate it, reconstruct the image and study the EPI artefacts. This time you write the sequence on your own: each section gives the goal of a step.

1. Click on **Edit or run this notebook** (top right of the notebook below) and download the notebook file (`.jl`).
2. Open it with Pluto on your computer (see [Installation](1-2-installation.md)). The first execution takes several minutes.
3. Fill in the cells marked 👉 and check each step (`check_timing`, `plot_seq`, `plot_kspace`) before moving on to the next one.
4. If needed, check the solution at the bottom of this webpage.

```@raw html
<div style="height: 100vh; width: 100%; overflow: auto; border: 1px solid #ddd; border-radius: 8px;">
  <iframe src="./pluto/pluto-4-4-epi_practical.html" style="width: 100%; height: 100%; border: 0;"></iframe>
</div>
```

## Solution

```@raw html
<div style="height: 100vh; width: 100%; overflow: auto; border: 1px solid #ddd; border-radius: 8px;">
  <iframe src="./pluto/pluto-4-4b-epi_practical.html" style="width: 100%; height: 100%; border: 0;"></iframe>
</div>
```
