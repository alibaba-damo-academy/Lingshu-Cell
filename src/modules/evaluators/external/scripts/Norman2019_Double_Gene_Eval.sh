#!/bin/bash
# Norman2019 double-gene perturbation evaluation via cell-eval.
#
# Runs cell-eval run + cell-eval score on a predicted h5ad (with control
# cells already merged and target_gene columns combined).
#
# Usage (called by vcc_score.py evaluator framework):
#   bash Norman2019_Double_Gene_Eval.sh \
#       --predict /path/to/merged_with_control.h5ad \
#       --validation /path/to/real_validation.h5ad \
#       --real-de /path/to/real_de.csv \
#       --base-agg /path/to/agg_results.csv \
#       --outdir /path/to/output \
#       --threads 20 \
#       --profile vcc_general \
#       --pert-col target_gene \
#       --control-pert non-perturbation

set -e

# ==== Defaults ====
DEFAULT_THREADS=20
DEFAULT_PROFILE="vcc_general"
DEFAULT_PERT_COL="target_gene"
DEFAULT_CONTROL_PERT="non-perturbation"
DEFAULT_BASE_OUTDIR="./cell-eval-outdir_score"

# ==== Initialize ====
PREDICT_PATH=""
VALIDATION_PATH=""
REAL_DE_PATH=""
BASE_AGG=""
THREADS="$DEFAULT_THREADS"
PROFILE="$DEFAULT_PROFILE"
PERT_COL="$DEFAULT_PERT_COL"
CONTROL_PERT="$DEFAULT_CONTROL_PERT"
OUTDIR=""

# ==== Usage ====
usage() {
    echo "Usage: $0 --predict PATH --validation PATH --real-de PATH [options]"
    echo ""
    echo "Required:"
    echo "  --predict PATH           Predicted h5ad (with control cells)"
    echo "  --validation PATH        Real validation h5ad"
    echo "  --real-de PATH           Real DE CSV (from cell-eval baseline)"
    echo ""
    echo "Optional:"
    echo "  --base-agg PATH          Baseline agg CSV (for cell-eval score)"
    echo "  --threads N              Thread count (default: $DEFAULT_THREADS)"
    echo "  --profile NAME           cell-eval profile (default: $DEFAULT_PROFILE)"
    echo "  --pert-col NAME          Perturbation column (default: $DEFAULT_PERT_COL)"
    echo "  --control-pert NAME      Control label (default: $DEFAULT_CONTROL_PERT)"
    echo "  --outdir PATH            Output directory"
    echo "  -h, --help               Show this help"
    exit 1
}

# ==== Parse arguments ====
while [[ $# -gt 0 ]]; do
    case "$1" in
        --predict)
            PREDICT_PATH="$2"
            shift 2
            ;;
        --validation)
            VALIDATION_PATH="$2"
            shift 2
            ;;
        --real-de)
            REAL_DE_PATH="$2"
            shift 2
            ;;
        --base-agg)
            BASE_AGG="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --pert-col)
            PERT_COL="$2"
            shift 2
            ;;
        --control-pert)
            CONTROL_PERT="$2"
            shift 2
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# ==== Validate ====
if [[ -z "$PREDICT_PATH" ]]; then
    echo "ERROR: --predict is required" >&2
    usage
fi
if [[ -z "$VALIDATION_PATH" ]]; then
    echo "ERROR: --validation is required" >&2
    usage
fi
if [[ -z "$REAL_DE_PATH" ]]; then
    echo "ERROR: --real-de is required" >&2
    usage
fi

# ==== Auto-generate OUTDIR ====
if [[ -z "$OUTDIR" ]]; then
    filename=$(basename "$PREDICT_PATH")
    prefix="${filename%%.*}"
    OUTDIR="${DEFAULT_BASE_OUTDIR}/${prefix}"
fi

# ==== Print config ====
echo "========== Norman2019 Double Gene Eval =========="
echo "Predict:     $PREDICT_PATH"
echo "Validation:  $VALIDATION_PATH"
echo "Real DE:     $REAL_DE_PATH"
echo "Base agg:    ${BASE_AGG:-<no score step>}"
echo "Threads:     $THREADS"
echo "Profile:     $PROFILE"
echo "pert-col:    $PERT_COL"
echo "control:     $CONTROL_PERT"
echo "Output:      $OUTDIR"
echo "================================================="

# ==== Create output dir ====
mkdir -p "$OUTDIR"

# ==== Run cell-eval run ====
echo "=== cell-eval run ==="
cell-eval run \
    --pert-col "$PERT_COL" \
    --control-pert "$CONTROL_PERT" \
    -ap "$PREDICT_PATH" \
    -ar "$VALIDATION_PATH" \
    -dr "$REAL_DE_PATH" \
    --num-threads "$THREADS" \
    --profile "$PROFILE" \
    -o "$OUTDIR"

# ==== Run cell-eval score (optional) ====
if [[ -n "$BASE_AGG" ]] && [[ -f "$BASE_AGG" ]]; then
    AGG_FILE=$(find "$OUTDIR" -name "*agg_results.csv" | head -1)
    if [[ -n "$AGG_FILE" ]]; then
        echo "=== cell-eval score ==="
        cell-eval score \
            --user-input "$AGG_FILE" \
            --base-input "$BASE_AGG" \
            -o "$OUTDIR/baseline_diff.csv"
    else
        echo "WARNING: agg_results.csv not found in $OUTDIR, skipping score."
    fi
fi

echo "Evaluation complete: $OUTDIR"
