#!/bin/bash
# ============================================================================
# VCC Gene Perturbation — Inference + Evaluation
#
# Reproduces the VCC CRISPRi perturbation experiment from the
# Lingshu-Cell technical report (arXiv:2603.25240).
# Uses clue-based biology prior injection + classifier-free guidance.
#
# Prerequisites:
#   - Model checkpoint: vcc_perturbation_85M.pth (from HuggingFace)
#   - ESM gene embeddings: gene_embeddings.npz (from HuggingFace)
#   - Clue CSV: whole_dataset_stringent_clue.csv (from HuggingFace)
#   - VCC evaluation data (see README for data access via arc-virtual-cell-atlas)
#   - cell-eval tool: pip install cell-eval
#
# Usage:
#   MODEL_PATH=/path/to/vcc_perturbation_85M.pth \
#   GENE_EMBEDDING_FILE=/path/to/gene_embeddings.npz \
#   CLUE_CSV=/path/to/whole_dataset_stringent_clue.csv \
#   bash scripts/run_infer_and_eval-vcc_perturbation.sh
# ============================================================================
set -euo pipefail

# --- Required ---
MODEL_PATH="${MODEL_PATH:?'Set MODEL_PATH to vcc_perturbation_85M.pth'}"
GENE_EMBEDDING_FILE="${GENE_EMBEDDING_FILE:?'Set GENE_EMBEDDING_FILE to gene_embeddings.npz'}"

# --- Optional overrides ---
CLUE_CSV="${CLUE_CSV:-}"  # Path to whole_dataset_stringent_clue.csv (empty = no clue)
STEPS="${STEPS:-3}"
CFG_SCALE="${CFG_SCALE:-2.0}"
SUBTASK="${SUBTASK:-1of1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONDITION_CSV="${CONDITION_CSV:-data/vcc/pert_counts_Test.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-output/vcc_perturbation}"

# --- Evaluation (optional) ---
CONTROL_H5AD="${CONTROL_H5AD:-}"  # Path to VCC control (non-targeting) h5ad for merge
VCC_TEST_VALIDATION_H5AD="${VCC_TEST_VALIDATION_H5AD:-}"
VCC_TEST_BASELINE_DIR="${VCC_TEST_BASELINE_DIR:-}"

export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== VCC Gene Perturbation Inference ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "GENE_EMBEDDING_FILE: $GENE_EMBEDDING_FILE"
echo "CLUE_CSV: ${CLUE_CSV:-'(disabled)'}"
echo "STEPS: $STEPS, CFG_SCALE: $CFG_SCALE"

# --- Inference ---
python -m workflows.infer.main \
    --config-name infer/legacy_vcc \
    checkpoint.model_path="$MODEL_PATH" \
    output.workdir="$OUTPUT_DIR" \
    task.subtask="$SUBTASK" \
    task.batch_size="$BATCH_SIZE" \
    task.condition_csv="$CONDITION_CSV" \
    model.config.gene_embedding_file="$GENE_EMBEDDING_FILE" \
    sampler.steps="$STEPS" \
    sampler.cfg_scale="$CFG_SCALE" \
    sampler.clue_csv="$CLUE_CSV"

echo "=== Inference complete ==="

# --- Merge with control cells (required for VCC evaluation) ---
if [ -n "$CONTROL_H5AD" ]; then
    SAMPLE_DIR=$(ls -td "$OUTPUT_DIR"/samples/vcc_legacy_infer/*/ 2>/dev/null | head -1)
    if [ -z "$SAMPLE_DIR" ]; then
        echo "WARNING: No sample directory found."
        exit 0
    fi

    MERGED_H5AD="$SAMPLE_DIR/merged_with_control.h5ad"
    echo "=== Merging with control cells ==="
    python workflows/utils/merge_with_control.py \
        --predict "$SAMPLE_DIR/merged.h5ad" \
        --control "$CONTROL_H5AD" \
        --output "$MERGED_H5AD"

    echo "Merged output: $MERGED_H5AD"

    # --- VCC Evaluation ---
    if [ -n "$VCC_TEST_VALIDATION_H5AD" ] && [ -n "$VCC_TEST_BASELINE_DIR" ]; then
        CELL_EVAL_CONDA="${CELL_EVAL_CONDA:-}"
        echo "=== Running VCC cell-eval ==="
        export VCC_TEST_VALIDATION_H5AD VCC_TEST_BASELINE_DIR

        python -m workflows.eval.main \
            --config-name eval/vcc \
            sample_dir="$SAMPLE_DIR" \
            "metrics=[vcc_test]" \
            "external.vcc_test.params.conda_env=$CELL_EVAL_CONDA"

        echo "=== VCC Evaluation complete ==="
    else
        echo "Skipping VCC evaluation (VCC_TEST_VALIDATION_H5AD or VCC_TEST_BASELINE_DIR not set)."
    fi
else
    echo "Skipping merge and evaluation (CONTROL_H5AD not set)."
    echo "To run VCC evaluation, provide CONTROL_H5AD, VCC_TEST_VALIDATION_H5AD, VCC_TEST_BASELINE_DIR."
fi
