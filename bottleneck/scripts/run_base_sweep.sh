#!/usr/bin/env bash
# Overnight base-model sweep (no VN).
#
# Models:
#   - TransformerConv (lr=1e-4, heads=2)
#   - GAT (lr=2e-4, heads=2)
#   - SetTransformer (heads=4, dim=32) at both lr=1e-3 and lr=1e-4
#
# Configs (n, dim):
#   - TransformerConv / GAT: (50, 256), (50, 1024), (100, 256), (200, 256), (200, 1024)
#   - SetTransformer: n in {50, 100, 200}, dim fixed to 32 (model default)
#
# Runs are executed sequentially. On per-run failure (e.g. OOM on
# n=200/dim=1024 TransformerConv) we log it and continue.

set -u  # error on unset vars; DO NOT set -e (we want to continue past failures)

SWEEP_NAME="base_sweep_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="sro-base-sweep"
RESULTS_CSV="results/${SWEEP_NAME}.csv"
LOG_DIR="logs/${SWEEP_NAME}"

cd "$(dirname "$0")/.."  # cd to bottleneck/
mkdir -p "results" "$LOG_DIR"

SUMMARY_LOG="${LOG_DIR}/_summary.log"
echo "Sweep start: $(date)" | tee "$SUMMARY_LOG"
echo "Results CSV: $RESULTS_CSV" | tee -a "$SUMMARY_LOG"
echo "Per-run logs in: $LOG_DIR/" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# Common flags shared by all runs.
COMMON=(
    --task_type two --star_variant connected
    --seed 1 --target_acc 1.01
    --max_epochs 200
    --num_train_samples 7000 --num_test_samples 700
    --lr_schedule none
    --dropout 0.0
    --wandb --wandb_project "$WANDB_PROJECT"
    --results_csv "$RESULTS_CSV"
)

# Counter for progress.
TOTAL_RUNS=16
RUN_IDX=0

run_one() {
    local tag="$1"
    shift
    RUN_IDX=$((RUN_IDX + 1))
    local t_start=$(date +%s)
    local log_file="${LOG_DIR}/${tag}.log"
    echo "[${RUN_IDX}/${TOTAL_RUNS}] $(date '+%F %T')  ${tag}  -> ${log_file}" | tee -a "$SUMMARY_LOG"
    if python train.py "$@" > "$log_file" 2>&1; then
        local t_end=$(date +%s)
        local elapsed=$((t_end - t_start))
        echo "    OK   (${elapsed}s)" | tee -a "$SUMMARY_LOG"
    else
        local ec=$?
        local t_end=$(date +%s)
        local elapsed=$((t_end - t_start))
        echo "    FAIL (${elapsed}s, exit=${ec})  -- continuing to next run" | tee -a "$SUMMARY_LOG"
        # Append a grep of OOM / error summary to the summary log for quick scan.
        tail -n 20 "$log_file" | sed 's/^/        /' >> "$SUMMARY_LOG"
    fi
}

# -----------------------------------------------------------------------------
# TransformerConv runs: lr=1e-4, heads=2, vary (n, dim).
# -----------------------------------------------------------------------------
for NDIM in "50 256" "50 1024" "100 256" "200 256" "200 1024"; do
    set -- $NDIM; N=$1; DIM=$2
    run_one "TransformerConv_h2_lr1e-4_n${N}_d${DIM}" \
        --model_type Transformer \
        --start "$N" --end "$((N+1))" \
        --dim "$DIM" --heads 2 --lr 1e-4 \
        "${COMMON[@]}"
done

# -----------------------------------------------------------------------------
# GAT runs: lr=2e-4, heads=2, vary (n, dim).
# -----------------------------------------------------------------------------
for NDIM in "50 256" "50 1024" "100 256" "200 256" "200 1024"; do
    set -- $NDIM; N=$1; DIM=$2
    run_one "GAT_h2_lr2e-4_n${N}_d${DIM}" \
        --model_type GAT \
        --start "$N" --end "$((N+1))" \
        --dim "$DIM" --heads 2 --lr 2e-4 \
        "${COMMON[@]}"
done

# -----------------------------------------------------------------------------
# SetTransformer runs: heads=4, dim=32 (fixed), vary n and lr.
# -----------------------------------------------------------------------------
for N in 50 100 200; do
    for LR in 1e-3 1e-4; do
        run_one "SetTransformer_h4_d32_lr${LR}_n${N}" \
            --model_type SetTransformer \
            --start "$N" --end "$((N+1))" \
            --dim 32 --heads 4 --lr "$LR" \
            "${COMMON[@]}"
    done
done

echo "" | tee -a "$SUMMARY_LOG"
echo "Sweep end: $(date)" | tee -a "$SUMMARY_LOG"
echo "Aggregated results CSV: $RESULTS_CSV" | tee -a "$SUMMARY_LOG"
