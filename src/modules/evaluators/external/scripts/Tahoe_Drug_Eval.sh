#!/bin/bash
set -e

# ==== 默认值 ====
DEFAULT_THREADS=40
DEFAULT_PROFILE="vcc_general"
DEFAULT_PERT_COL="drug__dose"
DEFAULT_CONTROL_PERT="DMSO_TF__0.0 uM"
DEFAULT_CELLTYPE_COL="cell_line_id"
DEFAULT_BASE_OUTDIR="./cell-eval-outdir_score"

# ==== 初始化参数 ====
PREDICT_PATH=""
VALIDATION_PATH=""
REAL_DE_PATH=""
BASE_AGG=""
THREADS="$DEFAULT_THREADS"
PROFILE="$DEFAULT_PROFILE"
PERT_COL="$DEFAULT_PERT_COL"
CONTROL_PERT="$DEFAULT_CONTROL_PERT"
CELLTYPE_COL="$DEFAULT_CELLTYPE_COL"
OUTDIR=""

# ==== 帮助信息 ====
usage() {
    echo "用法: $0 --predict PATH --validation PATH --real-de PATH [选项]"
    echo ""
    echo "必选参数:"
    echo "  --predict PATH           预测数据文件(.h5ad, 含 control)"
    echo "  --validation PATH        验证集文件(.h5ad, 真实细胞 + control)"
    echo "  --real-de PATH           Real DE CSV (由 cell-eval baseline 生成)"
    echo ""
    echo "可选参数 (带默认值):"
    echo "  --base-agg PATH          基线 agg 文件（用于 cell-eval score）"
    echo "  --threads N              线程数         (默认: $DEFAULT_THREADS)"
    echo "  --profile NAME           Profile 名称   (默认: $DEFAULT_PROFILE)"
    echo "  --pert-col NAME          扰动列名       (默认: $DEFAULT_PERT_COL)"
    echo "  --control-pert NAME      Control 标识   (默认: $DEFAULT_CONTROL_PERT)"
    echo "  --celltype-col NAME      Cell type 列名 (默认: $DEFAULT_CELLTYPE_COL)"
    echo "  --outdir PATH            输出目录       (默认: ${DEFAULT_BASE_OUTDIR}/<预测文件前缀>)"
    echo "  -h, --help               显示此帮助信息"
    exit 1
}

# ==== 参数解析 ====
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
        --celltype-col)
            CELLTYPE_COL="$2"
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
            echo "未知参数: $1"
            usage
            ;;
    esac
done

# ==== 参数检查 ====
if [[ -z "$PREDICT_PATH" ]]; then
    echo "错误: 必须指定 --predict 参数"
    usage
fi
if [[ -z "$VALIDATION_PATH" ]]; then
    echo "错误: 必须指定 --validation 参数"
    usage
fi
if [[ -z "$REAL_DE_PATH" ]]; then
    echo "错误: 必须指定 --real-de 参数"
    usage
fi

# ==== 自动生成 OUTDIR ====
if [[ -z "$OUTDIR" ]]; then
    filename=$(basename "$PREDICT_PATH")
    prefix="${filename%%.*}"
    OUTDIR="${DEFAULT_BASE_OUTDIR}/${prefix}"
fi

# ==== 打印参数 ====
echo "========== Tahoe Drug Eval =========="
echo "预测数据:    $PREDICT_PATH"
echo "验证数据:    $VALIDATION_PATH"
echo "Real DE:     $REAL_DE_PATH"
echo "基线agg:     ${BASE_AGG:-<不执行 score>}"
echo "线程数:      $THREADS"
echo "Profile:     $PROFILE"
echo "pert-col:    $PERT_COL"
echo "control:     $CONTROL_PERT"
echo "celltype:    $CELLTYPE_COL"
echo "输出目录:    $OUTDIR"
echo "====================================="

# ==== 创建目录 ====
mkdir -p "$OUTDIR"

# ==== 运行 cell-eval run ====
echo "=== 运行 cell-eval run ==="
cell-eval run \
    --pert-col "$PERT_COL" \
    --control-pert "$CONTROL_PERT" \
    --celltype-col "$CELLTYPE_COL" \
    -ap "$PREDICT_PATH" \
    -ar "$VALIDATION_PATH" \
    -dr "$REAL_DE_PATH" \
    --num-threads "$THREADS" \
    --profile "$PROFILE" \
    -o "$OUTDIR"

# ==== 运行 cell-eval score (可选) ====
if [[ -n "$BASE_AGG" ]] && [[ -f "$BASE_AGG" ]]; then
    # cell-eval may prefix output with celltype name; find the actual agg file.
    AGG_FILE=$(find "$OUTDIR" -name "*agg_results.csv" | head -1)
    if [[ -n "$AGG_FILE" ]]; then
        echo "=== 运行 cell-eval score ==="
        cell-eval score \
            --user-input "$AGG_FILE" \
            --base-input "$BASE_AGG" \
            -o "$OUTDIR/baseline_diff.csv"
    else
        echo "WARNING: agg_results.csv not found in $OUTDIR, skipping score."
    fi
fi

echo "✅ 评估完成: $OUTDIR"
