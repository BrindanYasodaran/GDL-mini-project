#!/usr/bin/env bash
set -u

SWEEP_NAME="vn_strategies_$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="sro-vn-strategies"
RESULTS_CSV="results/${SWEEP_NAME}.csv"
LOG_DIR="logs/${SWEEP_NAME}"

cd "$(dirname "$0")/.."  # cd to bottleneck/
mkdir -p "results" "$LOG_DIR"

SUMMARY_LOG="${LOG_DIR}/_summary.log"
echo "Sweep start: $(date)"                                        | tee "$SUMMARY_LOG"
echo "Results CSV: $RESULTS_CSV"                                   | tee -a "$SUMMARY_LOG"
echo "Per-run logs: $LOG_DIR/"                                     | tee -a "$SUMMARY_LOG"
echo "W&B project: $WANDB_PROJECT"                                 | tee -a "$SUMMARY_LOG"
echo ""                                                            | tee -a "$SUMMARY_LOG"

N_BASE=100
COMMON=(
    --task_type two --star_variant connected
    --start "$N_BASE" --end "$((N_BASE + 1))"
    --model_type GAT --heads 2 --lr 2e-4
    --target_acc 0.999 --max_epochs 100
    --num_train_samples 7000 --num_test_samples 700
    --lr_schedule none
    --dropout 0.0
    --eval_every 1
    --wandb --wandb_project "$WANDB_PROJECT"
    --results_csv "$RESULTS_CSV"
)

PROB_VN_DEFAULTS=(
    --prob_vn
    --num_vn 25
    --vn_tau_start 0.1 --vn_tau_end 0.1
    --vn_tau_anneal_epochs 0 --vn_tau_schedule constant
)

TOTAL_RUNS=23
RUN_IDX=0

run_one() {
    local tag="$1"; shift
    local group="$1"; shift
    local tags=()
    while [ "$1" != "--" ]; do
        tags+=("$1")
        shift
    done
    shift

    RUN_IDX=$((RUN_IDX + 1))
    local t_start=$(date +%s)
    local log_file="${LOG_DIR}/${tag}.log"
    echo "[${RUN_IDX}/${TOTAL_RUNS}] $(date '+%F %T')  ${tag}  group=${group}  tags=[${tags[*]}]" \
        | tee -a "$SUMMARY_LOG"
    if python train.py \
        --wandb_group "$group" \
        --wandb_tags "${tags[@]}" \
        "$@" > "$log_file" 2>&1; then
        local t_end=$(date +%s)
        echo "    OK   ($((t_end - t_start))s)" | tee -a "$SUMMARY_LOG"
    else
        local ec=$?
        local t_end=$(date +%s)
        echo "    FAIL ($((t_end - t_start))s, exit=${ec}) -- continuing" | tee -a "$SUMMARY_LOG"
        tail -n 20 "$log_file" | sed 's/^/        /' >> "$SUMMARY_LOG"
    fi
}

echo "=== Part 1: Primary comparison (3 strategies x 3 seeds) ===" | tee -a "$SUMMARY_LOG"
for SEED in 1 2 3; do
    run_one \
        "P1_DPW_d64_s${SEED}" \
        "P1_DPW_d64" \
        "P1" "DPW" "d_router_64" "m25" "tau_fixed_0.1" "dim1024" "seed_replicates" \
        -- \
        --dim 1024 --seed "$SEED" \
        --vn_router decoupled --vn_d_router 64 \
        "${PROB_VN_DEFAULTS[@]}" "${COMMON[@]}"

    run_one \
        "P1_TPW_s${SEED}" \
        "P1_TPW" \
        "P1" "TPW" "m25" "tau_fixed_0.1" "dim1024" "seed_replicates" \
        -- \
        --dim 1024 --seed "$SEED" \
        --vn_router tied \
        "${PROB_VN_DEFAULTS[@]}" "${COMMON[@]}"

    run_one \
        "P1_APW_s${SEED}" \
        "P1_APW" \
        "P1" "APW" "m25" "tau_fixed_0.1" "dim1024" "seed_replicates" \
        -- \
        --dim 1024 --seed "$SEED" \
        --vn_router adaptive \
        "${PROB_VN_DEFAULTS[@]}" "${COMMON[@]}"
done

echo "=== Part 2A: DPW d_router ablation ===" | tee -a "$SUMMARY_LOG"
for DR in 16 256; do
    run_one \
        "P2A_DPW_d${DR}_s1" \
        "P2A_DPW_d${DR}" \
        "P2A" "DPW" "d_router_${DR}" "m25" "tau_fixed_0.1" "dim1024" "ablation" \
        -- \
        --dim 1024 --seed 1 \
        --vn_router decoupled --vn_d_router "$DR" \
        "${PROB_VN_DEFAULTS[@]}" "${COMMON[@]}"
done

echo "=== Part 2B: APW temperature annealing ===" | tee -a "$SUMMARY_LOG"
run_one \
    "P2B_APW_anneal_exp5.0-0.1_50ep_s1" \
    "P2B_APW_anneal_exp" \
    "P2B" "APW" "m25" "tau_anneal" "anneal_exp" "dim1024" "ablation" \
    -- \
    --dim 1024 --seed 1 \
    --vn_router adaptive \
    --prob_vn --num_vn 25 \
    --vn_tau_start 5.0 --vn_tau_end 0.1 \
    --vn_tau_anneal_epochs 50 --vn_tau_schedule exp \
    "${COMMON[@]}"

run_one \
    "P2B_APW_anneal_lin2.0-0.1_30ep_s1" \
    "P2B_APW_anneal_lin" \
    "P2B" "APW" "m25" "tau_anneal" "anneal_linear" "dim1024" "ablation" \
    -- \
    --dim 1024 --seed 1 \
    --vn_router adaptive \
    --prob_vn --num_vn 25 \
    --vn_tau_start 2.0 --vn_tau_end 0.1 \
    --vn_tau_anneal_epochs 30 --vn_tau_schedule linear \
    "${COMMON[@]}"

echo "=== Part 2C: TPW temperature annealing ===" | tee -a "$SUMMARY_LOG"
run_one \
    "P2C_TPW_anneal_exp5.0-0.1_50ep_s1" \
    "P2C_TPW_anneal_exp" \
    "P2C" "TPW" "m25" "tau_anneal" "anneal_exp" "dim1024" "ablation" \
    -- \
    --dim 1024 --seed 1 \
    --vn_router tied \
    --prob_vn --num_vn 25 \
    --vn_tau_start 5.0 --vn_tau_end 0.1 \
    --vn_tau_anneal_epochs 50 --vn_tau_schedule exp \
    "${COMMON[@]}"

echo "=== Part 2D: oracle routing topline ===" | tee -a "$SUMMARY_LOG"
run_one \
    "P2D_oracle_s1" \
    "P2D_oracle" \
    "P2D" "oracle" "m25" "dim1024" "topline" \
    -- \
    --dim 1024 --seed 1 \
    --vn_router adaptive \
    --oracle_routing \
    "${PROB_VN_DEFAULTS[@]}" "${COMMON[@]}"

echo "=== Part 2E: no-VN baseline (GAT) ===" | tee -a "$SUMMARY_LOG"
run_one \
    "P2E_GAT_noVN_s1" \
    "P2E_GAT_noVN" \
    "P2E" "no_vn" "baseline" "dim1024" \
    -- \
    --dim 1024 --seed 1 \
    "${COMMON[@]}"

echo "=== Part 2F: dim vs num_vn sweep (APW) ===" | tee -a "$SUMMARY_LOG"
for DIM in 256 1024; do
    for M in 1 5 10 25; do
        if [ "$DIM" = "1024" ] && [ "$M" = "25" ]; then
            echo "    (skip dim=1024 m=25 -- already in Part 1 seed=1)" | tee -a "$SUMMARY_LOG"
            continue
        fi
        run_one \
            "P2F_APW_d${DIM}_m${M}_s1" \
            "P2F_APW_d${DIM}_m${M}" \
            "P2F" "APW" "dim${DIM}" "m${M}" "tau_fixed_0.1" "dim_vs_m" \
            -- \
            --dim "$DIM" --seed 1 \
            --vn_router adaptive \
            --prob_vn --num_vn "$M" \
            --vn_tau_start 0.1 --vn_tau_end 0.1 \
            --vn_tau_anneal_epochs 0 --vn_tau_schedule constant \
            "${COMMON[@]}"
    done
done

echo ""                                                         | tee -a "$SUMMARY_LOG"
echo "Sweep end: $(date)"                                       | tee -a "$SUMMARY_LOG"
echo "Aggregated results CSV: $RESULTS_CSV"                     | tee -a "$SUMMARY_LOG"
echo "Completed ${RUN_IDX}/${TOTAL_RUNS} runs."                 | tee -a "$SUMMARY_LOG"
