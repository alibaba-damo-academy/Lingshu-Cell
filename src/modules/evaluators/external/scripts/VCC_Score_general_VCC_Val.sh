#!/bin/bash
set -e

# ==== 默认值 ====
DEFAULT_VALIDATION_PATH="${VCC_VAL_VALIDATION_H5AD:?'Set VCC_VAL_VALIDATION_H5AD to the validation adata h5ad path'}"
DEFAULT_REAL_DE_PATH="${VCC_VAL_BASELINE_DIR:?'Set VCC_VAL_BASELINE_DIR to VCC_Val_Data/VCC_Eval/baseline directory'}/real_de.csv"
DEFAULT_BASE_AGG="${VCC_VAL_BASELINE_DIR:?'Set VCC_VAL_BASELINE_DIR'}/agg_results.csv"
DEFAULT_THREADS=40
DEFAULT_PROFILE="vcc_general"
DEFAULT_BASE_OUTDIR="./cell-eval-outdir_score"  # 注意这里改成了 BASE_OUTDIR

# ==== 初始化参数 ====
PREDICT_PATH=""
VALIDATION_PATH="$DEFAULT_VALIDATION_PATH"
REAL_DE_PATH="$DEFAULT_REAL_DE_PATH"
BASE_AGG="$DEFAULT_BASE_AGG"
THREADS="$DEFAULT_THREADS"
PROFILE="$DEFAULT_PROFILE"
OUTDIR=""   # 会在解析完 predict 自动生成

# ==== 帮助信息 ====
usage() {
    echo "用法: $0 --predict 路径 [选项]"
    echo ""
    echo "必选参数:"
    echo "  --predict PATH           预测数据文件(.h5ad)"
    echo ""
    echo "可选参数 (带默认值):"
    echo "  --validation PATH        验证集文件    (默认: $DEFAULT_VALIDATION_PATH)"
    echo "  --real-de PATH           Real DE CSV   (默认: $DEFAULT_REAL_DE_PATH)"
    echo "  --base-agg PATH          基线agg文件   (默认: $DEFAULT_BASE_AGG)"
    echo "  --threads N              线程数         (默认: $DEFAULT_THREADS)"
    echo "  --profile NAME           Profile名称    (默认: $DEFAULT_PROFILE)"
    echo "  --outdir PATH            输出目录       (默认: ${DEFAULT_BASE_OUTDIR}/<预测文件前缀>)"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --predict adata_predict.h5ad"
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

# ==== 自动根据预测文件生成 OUTDIR（如果没传） ====
if [[ -z "$OUTDIR" ]]; then
    filename=$(basename "$PREDICT_PATH")   # 例如 adata_predict.h5ad
    prefix="${filename%%.*}"               # 去掉第一个 '.' 及之后 => adata_predict
    OUTDIR="${DEFAULT_BASE_OUTDIR}/${prefix}"
fi

# ==== 打印参数 ====
echo "========== 参数 =========="
echo "预测数据:    $PREDICT_PATH"
echo "验证数据:    $VALIDATION_PATH"
echo "Real DE:     $REAL_DE_PATH"
echo "基线agg:     $BASE_AGG"
echo "线程数:      $THREADS"
echo "Profile:     $PROFILE"
echo "输出目录:    $OUTDIR"
echo "=========================="

# ==== 创建目录 ====
mkdir -p "$OUTDIR"

# ==== 运行 cell-eval run ====
echo "=== 运行 cell-eval run ==="
cell-eval run \
    -ap "$PREDICT_PATH" \
    -ar "$VALIDATION_PATH" \
    -dr "$REAL_DE_PATH" \
    --num-threads "$THREADS" \
    --profile "$PROFILE" \
    -o "$OUTDIR"

# ==== 运行 cell-eval score ====
echo "=== 运行 cell-eval score ==="
cell-eval score \
    --user-input "$OUTDIR/agg_results.csv" \
    --base-input "$BASE_AGG" \
    -o "$OUTDIR/baseline_diff_vcc_score.csv"

echo "✅ 所有任务完成"

