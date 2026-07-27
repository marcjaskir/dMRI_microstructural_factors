#!/bin/bash
#SBATCH --job-name=grad_voxelwise
#SBATCH --output=logs/gradients_voxelwise_%j.out
#SBATCH --error=logs/gradients_voxelwise_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# gradients_voxelwise SLURM wrapper
#
# Stage 1 (factor scores): ~288 subjects × 26 scalars; moderate I/O, ~16–32 GB RAM.
# Stage 2 (Laplacian): loads (288 × 966k) float32 subject matrix (~1.1 GB) plus
#   embed affinity (~15k nodes default); request 64–128 GB for safety.
#
# Usage:
#   sbatch run_gradients_voxelwise.slurm                           # voxelwise loadings, both stages
#   sbatch run_gradients_voxelwise.slurm factor_scores             # stage 1 only
#   sbatch run_gradients_voxelwise.slurm neuromaps                 # stage 3 only
#   FACTOR_LOADING_SOURCE=regionwise sbatch run_gradients_voxelwise.slurm
#
# FACTOR_LOADING_SOURCE: voxelwise (default) or regionwise
# CSF_MODES: with_csf,no_csf (default both), with_csf, no_csf, or both
# EMBED_STRIDE / EMBED_TOP_K / MAX_EMBED_VOXELS: Laplacian subsampling
#
# Output layout (under GRAD_ROOT):
#   voxelwise_with_csf/stride{S}_topk{K}_max{M}/
#   voxelwise_no_csf/stride{S}_topk{K}_max{M}/
#   regionwise/stride{S}_topk{K}_max{M}/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${STRUCTURAL_TRACTOMETRY_ROOT:-__PROJECT_ROOT__}"
GRAD_ROOT="${GRADIENTS_VOXELWISE_OUTPUT_DIR:-${ROOT}/derivatives/analysis/gradients_voxelwise}"
FACTOR_LOADING_SOURCE="${FACTOR_LOADING_SOURCE:-voxelwise}"
CSF_MODES="${CSF_MODES:-with_csf,no_csf}"
EMBED_STRIDE="${EMBED_STRIDE:-2}"
EMBED_TOP_K="${EMBED_TOP_K:-200}"
MAX_EMBED_VOXELS="${MAX_EMBED_VOXELS:-15000}"
STAGE="${1:-all}"

COMMON_ARGS=(
  --factor-loading-source "${FACTOR_LOADING_SOURCE}"
  --output-dir "${GRAD_ROOT}"
  --csf-modes "${CSF_MODES}"
)
GRADIENT_ARGS=(
  "${COMMON_ARGS[@]}"
  --embed-stride "${EMBED_STRIDE}"
  --embed-top-k "${EMBED_TOP_K}"
  --max-embed-voxels "${MAX_EMBED_VOXELS}"
)

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

run_factor_scores() {
  echo "=== Stage 1: per-subject factor score maps (${FACTOR_LOADING_SOURCE}, csf=${CSF_MODES}) ==="
  python compute_factor_scores.py "${COMMON_ARGS[@]}"
}

run_gradients() {
  echo "=== Stage 2: Laplacian G1/G2 + figures (${FACTOR_LOADING_SOURCE}, csf=${CSF_MODES}, embed=${EMBED_STRIDE}/${EMBED_TOP_K}/${MAX_EMBED_VOXELS}) ==="
  python compute_gradients.py "${GRADIENT_ARGS[@]}" --figures
}

run_neuromaps() {
  echo "=== Stage 3: neuromaps spatial correlations + lollipop figures (csf=${CSF_MODES}, embed=${EMBED_STRIDE}/${EMBED_TOP_K}/${MAX_EMBED_VOXELS}) ==="
  echo "Requires: pip install -r requirements-neuromaps.txt"
  python compute_neuromaps_correlations.py \
    "${GRADIENT_ARGS[@]}" \
    --figures \
    -v
}

case "${STAGE}" in
  factor_scores|scores|1)
    run_factor_scores
    ;;
  gradients|gradient|2)
    run_gradients
    ;;
  neuromaps|3)
    run_neuromaps
    ;;
  all|"")
    run_factor_scores
    run_gradients
    ;;
  *)
    echo "Unknown stage: ${STAGE} (use factor_scores, gradients, neuromaps, or all)"
    exit 1
    ;;
esac

echo "Finished gradients_voxelwise (${STAGE}, source=${FACTOR_LOADING_SOURCE}, csf=${CSF_MODES}, root=${GRAD_ROOT})"
