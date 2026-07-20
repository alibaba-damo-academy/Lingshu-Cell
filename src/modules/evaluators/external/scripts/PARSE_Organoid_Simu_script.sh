#!/bin/bash
set -e

# ==== 默认值 ====
DEFAULT_EXP_ID="CD-2-2-cytok-3-2-v1"
DEFAULT_DONOR_ID="Donor1"

BASE_TESTSET="${PARSE_ORGANOID_DATA_DIR:?'Set PARSE_ORGANOID_DATA_DIR to the Pre_exp directory'}"
BASELINE_BASE="${PARSE_ORGANOID_DATA_DIR:?'Set PARSE_ORGANOID_DATA_DIR'}/baseline_general"

DEFAULT_THREADS=40
DEFAULT_PROFILE="vcc_general"
DEFAULT_BASE_OUTDIR="./cell-eval-outdir_score"

# ==== 初始化参数 ====
PREDICT_PATH=""
EXP_ID="$DEFAULT_EXP_ID"
DONOR_ID="$DEFAULT_DONOR_ID"

VALIDATION_PATH=""
REAL_DE_PATH=""
BASE_AGG=""

THREADS="$DEFAULT_THREADS"
PROFILE="$DEFAULT_PROFILE"
OUTDIR=""

# ==== 帮助信息 ====
usage() {
    echo "用法: $0 --predict PATH [选项]"
    echo ""
    echo "必选参数:"
    echo "  --predict PATH           预测数据文件(.h5ad)"
    echo ""
    echo "可选参数:"
    echo "  --exp-id ID              实验ID (默认: $DEFAULT_EXP_ID，例如 CD-2-2-cytok-3-2-v1)"
    echo "  --donor-id ID            Donor ID (默认: $DEFAULT_DONOR_ID，例如 Donor1)"
    echo "  --validation PATH        验证集文件(默认: Pre_exp/PARSE_<exp-id>-test_split/PARSE_<exp-id>-test_<donor-id>.h5ad)"
    echo "  --real-de PATH           Real DE CSV(默认: baseline_general/<exp-id>_<donor-id>/real_de.csv)"
    echo "  --base-agg PATH          基线agg文件(默认: baseline_general/<exp-id>_<donor-id>/agg_results.csv)"
    echo "  --threads N              线程数(默认: $DEFAULT_THREADS)"
    echo "  --profile NAME           Profile名称(默认: $DEFAULT_PROFILE)"
    echo "  --outdir PATH            输出目录(默认: ${DEFAULT_BASE_OUTDIR}/<预测文件前缀>/<exp-id>_<donor-id>)"
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
        --exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        --donor-id)
            DONOR_ID="$2"
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

EXP_DONOR_ID="${EXP_ID}_${DONOR_ID}"

# ==== 根据 exp-id + donor-id 自动生成默认路径（若用户没显式传入） ====
if [[ -z "$VALIDATION_PATH" ]]; then
    VALIDATION_PATH="${BASE_TESTSET}/PARSE_${EXP_ID}-test_split/PARSE_${EXP_ID}-test_${DONOR_ID}.h5ad"
fi

if [[ -z "$REAL_DE_PATH" ]]; then
    REAL_DE_PATH="${BASELINE_BASE}/${EXP_DONOR_ID}/real_de.csv"
fi

if [[ -z "$BASE_AGG" ]]; then
    BASE_AGG="${BASELINE_BASE}/${EXP_DONOR_ID}/agg_results.csv"
fi

# ==== 自动根据预测文件生成 OUTDIR（如果没传） ====
if [[ -z "$OUTDIR" ]]; then
    filename=$(basename "$PREDICT_PATH")
    prefix="${filename%%.*}"
    OUTDIR="${DEFAULT_BASE_OUTDIR}/${prefix}/${EXP_DONOR_ID}"
fi

# ==== 打印参数 ====
echo "========== 参数 =========="
echo "实验ID:      $EXP_ID"
echo "Donor ID:    $DONOR_ID"
echo "预测数据:    $PREDICT_PATH"
echo "验证数据:    $VALIDATION_PATH"
echo "Real DE:     $REAL_DE_PATH"
echo "基线agg:     $BASE_AGG"
echo "线程数:      $THREADS"
echo "Profile:     $PROFILE"
echo "输出目录:    $OUTDIR"
echo "=========================="

mkdir -p "$OUTDIR"

echo "=== 运行 cell-eval run ==="
cell-eval run \
    -ap "$PREDICT_PATH" \
    -ar "$VALIDATION_PATH" \
    -dr "$REAL_DE_PATH" \
    --num-threads "$THREADS" \
    --profile "$PROFILE" \
    --pert-col cytokine \
    --control-pert PBS \
    -o "$OUTDIR"

echo "=== 运行 cell-eval score ==="
cell-eval score \
    --user-input "$OUTDIR/agg_results.csv" \
    --base-input "$BASE_AGG" \
    -o "$OUTDIR/baseline_diff_vcc_score.csv"

echo "✅ 所有任务完成"

