#!/bin/bash
# Orchestrates the CRE (continual relation extraction) baseline sweep for one dataset:
# tokenizes each requested permutation, then launches the distillation queue (7 methods)
# and the CL-LoRA queue (8 methods) each on its own GPU, in the background.
#
# Rule enforced by scripts/qwen/cre/run_cre_*.sh: at most ONE queue per GPU. This script
# respects that by giving the distillation queue and the CL-LoRA queue separate GPUs and
# running each one's permutations sequentially inside a single background process, so a
# GPU never has two queues racing for its memory between tasks.
#
# Usage:
#   bash run.sh <tacred|fewrel> ["perms"] [gpu_dist] [gpu_cllora]
#
#   bash run.sh tacred                     # perms 0-4, dist on gpu0, CL-LoRA on gpu1
#   bash run.sh fewrel "0 1 2"             # only perm0-2
#   bash run.sh tacred "0" 0 1             # single perm, explicit GPUs
#
# Logs: logs_cre_dist_<ds>.log and logs_cre_cllora_<ds>.log in the repo root.
# Safe to re-run: prep_cre.sh and both queue scripts skip work that already completed.
set -uo pipefail

DS=${1:?dataset required: tacred|fewrel}
PERMS=${2:-"0 1 2 3 4"}
GPU_DIST=${3:-0}
GPU_CLLORA=${4:-1}

cd "$(dirname "$0")"
case "${DS}" in
    tacred|fewrel) ;;
    *) echo "unknown dataset '${DS}' (expected tacred|fewrel)"; exit 1 ;;
esac

echo "[run.sh] dataset=${DS} perms='${PERMS}' gpu_dist=${GPU_DIST} gpu_cllora=${GPU_CLLORA}"

# Tokenize every requested permutation up front. The distillation queue reads
# processed_data/ (task0 is always plain CE, tokenized here); the CL-LoRA engine reads
# the raw JSONL directly and does not need this step, but it is cheap (a few minutes
# per permutation) so we do it once regardless.
for P in ${PERMS}; do
    bash scripts/qwen/cre/prep_cre.sh "${DS}" "${P}" || { echo "[run.sh] tokenize failed for perm${P}"; exit 1; }
done

# Each queue is its own detached process (setsid nohup), so it survives this shell
# exiting. Arguments are passed positionally rather than via environment variables,
# since a backgrounded `bash -c` does not inherit unexported shell variables.
run_dist_queue () {   # $1=dataset $2=gpu $3..=perms
    local ds=$1 gpu=$2; shift 2
    for p in "$@"; do
        bash scripts/qwen/cre/run_cre_dist.sh "${ds}" "${p}" "${gpu}"
    done
}
run_cllora_queue () {  # $1=dataset $2=gpu $3..=perms
    local ds=$1 gpu=$2; shift 2
    for p in "$@"; do
        bash scripts/qwen/cre/run_cre_cllora.sh "${ds}" "${p}" "${gpu}"
    done
}

setsid nohup bash -c "$(declare -f run_dist_queue); run_dist_queue \"\$@\"" _ \
    "${DS}" "${GPU_DIST}" ${PERMS} \
    > "logs_cre_dist_${DS}.log" 2>&1 < /dev/null &
DIST_PID=$!
setsid nohup bash -c "$(declare -f run_cllora_queue); run_cllora_queue \"\$@\"" _ \
    "${DS}" "${GPU_CLLORA}" ${PERMS} \
    > "logs_cre_cllora_${DS}.log" 2>&1 < /dev/null &
CLLORA_PID=$!

echo "[run.sh] distillation queue running on gpu${GPU_DIST}, pid ${DIST_PID} -> logs_cre_dist_${DS}.log"
echo "[run.sh] CL-LoRA queue running on gpu${GPU_CLLORA}, pid ${CLLORA_PID} -> logs_cre_cllora_${DS}.log"
echo "[run.sh] tail -f logs_cre_dist_${DS}.log logs_cre_cllora_${DS}.log"
