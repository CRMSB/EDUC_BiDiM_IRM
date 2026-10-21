#!/usr/bin/env bash
# Export Jupyter notebooks to standalone HTML pages for the documentation.
#
# Usage (from anywhere):
#   docs/export_jupyter.sh                       # all the notebooks of examples/
#   docs/export_jupyter.sh examples/my.ipynb     # only some notebooks
#   docs/export_jupyter.sh --execute ...         # re-run the notebooks before the export
#
# The HTML files are written to docs/src/public/jupyter/ and can be embedded in a
# documentation page with an <iframe> (see docs/src/4-2-hands_on_bruker_pulseq.md).
#
# nbconvert is run in a temporary environment with `uvx` (nothing is installed in
# your Python environments). To use another installation, set NBCONVERT, e.g.
#   NBCONVERT="conda run -n bruker_pulseq jupyter nbconvert" docs/export_jupyter.sh
#
# Without --execute, the outputs saved in the notebook are exported as they are:
# run the notebook and save it before the export.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/src/public/jupyter"
NBCONVERT="${NBCONVERT:-uvx --from nbconvert --with ipykernel jupyter-nbconvert}"

EXECUTE=()
NOTEBOOKS=()
for arg in "$@"; do
    case "$arg" in
        --execute) EXECUTE=(--execute) ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) NOTEBOOKS+=("$arg") ;;
    esac
done

# default : every notebook of examples/
if [ ${#NOTEBOOKS[@]} -eq 0 ]; then
    shopt -s nullglob
    NOTEBOOKS=("$ROOT"/examples/*.ipynb)
fi
if [ ${#NOTEBOOKS[@]} -eq 0 ]; then
    echo "No notebook found in $ROOT/examples" >&2
    exit 1
fi

mkdir -p "$OUT"
for nb in "${NOTEBOOKS[@]}"; do
    echo "Exporting $nb"
    # --embed-images : the images are stored inside the HTML file (a single file to publish)
    $NBCONVERT --to html --embed-images ${EXECUTE[@]+"${EXECUTE[@]}"} --output-dir "$OUT" "$nb"
done
echo "HTML files written to $OUT"
