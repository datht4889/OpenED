#!/bin/bash
# Seven distillation baselines with a shared task0 checkpoint and the f12_pl protocol.
set -euo pipefail

cd "$(dirname "$0")/../../.." || exit 1
PERM=${PERM:-0}
SEED=${SEED:-42}
PROTOCOL=${PROTOCOL:-v2}
GPU=${GPU:-0}
RESUME=${RESUME:-0}
DATA_PREFIX=${DATA_PREFIX:-ace_b10_perm}
SHARED="dist_shared_task0_perm${PERM}_${PROTOCOL}_s${SEED}"

if [ ! -f "results/qwen3/ced/${SHARED}/.complete" ]; then
    [ ! -e "results/qwen3/ced/${SHARED}" ] || {
        echo "incomplete shared task0 already exists: ${SHARED}"
        exit 1
    }
    echo "===== shared task0 ${SHARED} $(date) ====="
    bash scripts/qwen/ced/run_ced_v2.sh \
        --run-name "${SHARED}" --mode sft --data-prefix "${DATA_PREFIX}" --perm "${PERM}" \
        --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" \
        --bs 2 --acc 16 --greedy 1 --gpus "${GPU}" --end-task 0 \
        > "logs_${SHARED}.log" 2>&1
fi

run_dist () {  # $1=method label $2=kd-type ($3...=optional runner flags)
    local METHOD=$1; local KD_TYPE=$2; shift 2
    local RUN_NAME="dist_${METHOD}_perm${PERM}_${PROTOCOL}_s${SEED}"
    local RUN_ROOT="results/qwen3/ced/${RUN_NAME}"
    if [ -f "${RUN_ROOT}/.complete" ]; then
        echo "SKIP complete ${RUN_NAME}"
        return
    fi
    local START_TASK=1
    local RESUME_ARGS=()
    if [ -e "${RUN_ROOT}" ]; then
        [ "${RESUME}" = "1" ] || {
            echo "partial run exists; set RESUME=1: ${RUN_NAME}"
            exit 1
        }
        START_TASK=$(${HOME}/miniconda3/envs/mta/bin/python -c \
            "import json; print(json.load(open('${RUN_ROOT}/run_manifest.json'))['completed_task'] + 1)")
        RESUME_ARGS+=(--resume)
    fi
    echo "===== ${RUN_NAME} (${KD_TYPE}) $(date) ====="
    bash scripts/qwen/ced/run_ced_v2.sh --run-name "${RUN_NAME}" --mode ce_kd --data-prefix "${DATA_PREFIX}" --perm "${PERM}" \
        --kd-type "${KD_TYPE}" --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" \
        --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 2 --acc 16 \
        --greedy 1 --gpus "${GPU}" --start-task "${START_TASK}" \
        --task0-source-run "${SHARED}" "${RESUME_ARGS[@]}" "$@" \
        >> "logs_${RUN_NAME}.log" 2>&1
    [ -f "results/qwen3/ced/${RUN_NAME}/.complete" ] || {
        echo "missing completion marker: ${RUN_NAME}"
        exit 1
    }
    echo "DONE ${RUN_NAME}"
}

run_dist kd kd
run_dist rkl rkl
run_dist sfkl sfkl
run_dist srkl srkl
run_dist csd csd
run_dist distillm adaptive-srkl \
    --extra "--student-gen --gen-do-sample --gen-top-p 1.0 --gen-temperature 1.0 --gen-num-beams 1 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000"
run_dist amid adaptive-amid \
    --extra "--student-gen --gen-do-sample --gen-top-p 1.0 --gen-temperature 1.0 --gen-num-beams 1 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5"
echo "ALL DISTILL DONE $(date)"
