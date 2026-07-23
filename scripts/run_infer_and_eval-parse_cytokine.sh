#!/bin/bash
# ============================================================================
# PARSE Cytokine Perturbation — Inference + Evaluation
#
# Reproduces the PARSE cytokine conditional generation experiment from the
# Lingshu-Cell technical report (arXiv:2603.25240).
# Runs per-donor inference with cfg_general_control (equivalent to old cfg_PBS).
#
# Prerequisites:
#   - Model checkpoint: parse_cytokine_85M.pth (from HuggingFace)
#   - Condition CSVs: data/parse_cytokine/Donor{2,6,9,10}_conditions.csv
#   - PARSE test data for evaluation (see README for data preparation)
#   - cell-eval tool: pip install cell-eval
#
# Usage:
#   MODEL_PATH=/path/to/parse_cytokine_85M.pth \
#   DONOR=Donor10 \
#   bash scripts/run_infer_and_eval-parse_cytokine.sh
# ============================================================================
set -euo pipefail

# --- Required ---
MODEL_PATH="${MODEL_PATH:?'Set MODEL_PATH to parse_cytokine_85M.pth'}"
DONOR="${DONOR:?'Set DONOR (e.g., Donor10, Donor2, Donor6, Donor9)'}"

# --- Optional overrides ---
STEPS="${STEPS:-3}"
CFG_SCALE="${CFG_SCALE:-1.0}"
SUBTASK="${SUBTASK:-1of1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CONDITION_CSV="${CONDITION_CSV:-data/parse_cytokine/${DONOR}_conditions.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-output/parse_cytokine}"

# --- Evaluation (optional) ---
PARSE_TEST_DATA_DIR="${PARSE_TEST_DATA_DIR:-}"  # Directory containing PARSE_test_<Donor>.h5ad + baseline_general/

export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== PARSE Cytokine Inference (${DONOR}) ==="
echo "MODEL_PATH: $MODEL_PATH"
echo "CONDITION_CSV: $CONDITION_CSV"
echo "STEPS: $STEPS, CFG_SCALE: $CFG_SCALE"

# --- Inference ---
python -m workflows.infer.main \
    --config-name infer/legacy_parse_cytokine \
    checkpoint.model_path="$MODEL_PATH" \
    output.workdir="$OUTPUT_DIR" \
    name="parse_cytokine_${DONOR}" \
    task.subtask="$SUBTASK" \
    task.batch_size="$BATCH_SIZE" \
    task.condition_csv="$CONDITION_CSV" \
    sampler.steps="$STEPS" \
    sampler.cfg_scale="$CFG_SCALE"

echo "=== Inference complete ==="

# --- Merge with control + Evaluation ---
if [ -n "$PARSE_TEST_DATA_DIR" ]; then
    SAMPLE_DIR=$(ls -td "$OUTPUT_DIR"/samples/parse_cytokine_"${DONOR}"/*/ 2>/dev/null | head -1)
    if [ -z "$SAMPLE_DIR" ]; then
        echo "WARNING: No sample directory found."
        exit 0
    fi

    # Merge: append PBS control cells from the donor's test h5ad
    DONOR_H5AD="${PARSE_TEST_DATA_DIR}/PARSE_test_${DONOR}.h5ad"
    MERGED_H5AD="$SAMPLE_DIR/all_samples_lowerbound_for_cell_eval.h5ad"

    if [ -f "$DONOR_H5AD" ]; then
        echo "=== Merging with PBS control cells from ${DONOR_H5AD} ==="
        python workflows/utils/merge_with_control.py \
            --predict "$SAMPLE_DIR/merged.h5ad" \
            --control "$DONOR_H5AD" \
            --output "$MERGED_H5AD" \
            --target_obs_key cytokine \
            --control_value PBS \
            --gene_id_name_csv data/vcc/gene_id_name.csv \
            --convert_predict name_to_id

        echo "Merged output: $MERGED_H5AD"

        # Run cell-eval
        CELL_EVAL_CONDA="${CELL_EVAL_CONDA:-}"
        echo "=== Running PARSE cell-eval ==="
        export PARSE_TEST_DATA_DIR
        python -m workflows.eval.main \
            --config-name eval/vcc \
            sample_dir="$SAMPLE_DIR" \
            "metrics=[parse_cond_test]" \
            "external.parse_cond_test.params.conda_env=$CELL_EVAL_CONDA" \
            "external.parse_cond_test.params.donor_id=${DONOR}"

        echo "=== Evaluation complete ==="
    else
        echo "WARNING: Donor h5ad not found: $DONOR_H5AD"
        echo "Skipping merge and evaluation."
    fi
else
    echo "Skipping evaluation (PARSE_TEST_DATA_DIR not set)."
    echo "To run evaluation, set PARSE_TEST_DATA_DIR=/path/to/PARSE_test_data"
fi
