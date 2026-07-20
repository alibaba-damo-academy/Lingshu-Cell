#!/bin/bash
# ============================================================================
# PARSE PBMC Unconditional Generation — Inference + Evaluation
#
# Reproduces the unconditional PBMC tissue generation experiment from the
# Lingshu-Cell technical report (arXiv:2603.25240).
#
# Prerequisites:
#   - Model checkpoint: parse_pbmc_uncond_85M.pth (from HuggingFace)
#   - Real PBMC h5ad for evaluation (see README for data preparation)
#   - Python environment with requirements.txt installed
#
# Usage:
#   MODEL_PATH=/path/to/parse_pbmc_uncond_85M.pth \
#   bash scripts/run_infer_and_eval-parse_pbmc_uncond.sh
# ============================================================================
set -euo pipefail

# --- Required ---
MODEL_PATH="${MODEL_PATH:?'Set MODEL_PATH to the checkpoint .pth file'}"

# --- Optional overrides ---
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
STEPS="${STEPS:-256}"
SEED="${SEED:-50000}"
SUBTASK="${SUBTASK:-1of1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-output/parse_pbmc_uncond}"

# --- Evaluation (optional) ---
REAL_DATA="${REAL_DATA:-}"  # Path to real PBMC h5ad for evaluation
EVAL_METRICS="${EVAL_METRICS:-umap,expressed_genes,distribution,linregress}"

export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== PARSE PBMC Unconditional Inference ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "STEPS: $STEPS"
echo "SEED: $SEED"

# --- Inference ---
python -m workflows.infer.main \
    --config-name infer/legacy_parse_pbmc_uncond \
    checkpoint.model_path="$MODEL_PATH" \
    output.workdir="$OUTPUT_DIR" \
    task.subtask="$SUBTASK" \
    task.base_seed="$SEED" \
    task.batch_size="$BATCH_SIZE" \
    "task.condition_csv='n_cells
${NUM_SAMPLES}'" \
    sampler.steps="$STEPS"

echo "=== Inference complete ==="

# --- Evaluation (if real data is provided) ---
if [ -n "$REAL_DATA" ]; then
    # Find the latest sample directory
    SAMPLE_DIR=$(ls -td "$OUTPUT_DIR"/samples/parse_pbmc_uncond/*/ 2>/dev/null | head -1)
    if [ -z "$SAMPLE_DIR" ]; then
        echo "WARNING: No sample directory found for evaluation."
        exit 0
    fi

    echo "=== Running evaluation ==="
    echo "SAMPLE_DIR: $SAMPLE_DIR"
    echo "REAL_DATA: $REAL_DATA"

    python -m workflows.eval.main \
        sample_dir="$SAMPLE_DIR" \
        real_data="$REAL_DATA" \
        convert_ensembl_to_name=true \
        gene_id_name_file=data/vcc/gene_id_name.csv \
        "metrics=[$EVAL_METRICS]"

    echo "=== Evaluation complete ==="
else
    echo "Skipping evaluation (REAL_DATA not set)."
    echo "To run evaluation, set REAL_DATA=/path/to/pbmc_real.h5ad"
fi
