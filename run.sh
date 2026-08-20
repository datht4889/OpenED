#!/bin/bash
# Orchestrates the baseline sweep for one dataset, either family:
#   CRE  (continual relation extraction): tacred, fewrel
#   CED  (continual event detection):     maven, rams, geneva  (ace runs the same way too)
# Tokenizes each requested permutation, then launches the distillation queue (7 methods)
# and the CL-LoRA queue (8 methods) each on its own GPU, in the background.
#
# Rule enforced by the per-family runners: at most ONE queue per GPU. This script respects
# that by giving the distillation queue and the CL-LoRA queue separate GPUs and running each
# one's permutations sequentially inside a single background process, so a GPU never has two
# queues racing for its memory between tasks.
#
# Usage:
#   bash run.sh                                                     # run everything (see below)
#   bash run.sh <tacred|fewrel|maven|rams|geneva> ["perms"] [gpu_dist] [gpu_cllora] [queue]
#
#   bash run.sh rams                       # perms 0-4, dist on gpu0, CL-LoRA on gpu1, both queues
#   bash run.sh geneva "0 1 2"             # only perm0-2
#   bash run.sh maven "0" 0 1              # single perm, explicit GPUs
#   bash run.sh rams "3 4" 0 1 cllora      # catch up missing perms on ONE queue only
#
# No-argument mode (`bash run.sh`): runs TACRED and FewRel first (finish CRE) with the full
# perm range 0-4 -- the per-run skip logic already baked into
# scripts/qwen/cre/run_cre_{dist,cllora}.sh (a completion marker check per method+perm) means
# whatever CRE work already finished is left untouched and only the missing perms/methods
# actually train -- then RAMS and GENEVA in full (nothing has ever run for them, so "in full"
# and "only what's missing" are the same thing). Datasets run ONE AT A TIME (dist+CL-LoRA
# concurrently within a dataset, next dataset only after both queues of the current one finish)
# so at most one queue ever sits on a given GPU. Order: tacred, fewrel, rams, geneva. Override
# with RUN_ALL_DATASETS="ds1 ds2 ...".
#
# queue (5th arg, default "both"): dist|cllora|both. Use this instead of running both queues
# when only one baseline family is missing for that dataset -- launching the other queue
# anyway would retrain and immediately discard a shared task0 checkpoint for nothing (CRE's
# run_cre_dist.sh deletes it at the end of every full sweep and recreates it if missing; the
# CED dist_queue.sh keeps it but running it again just to reuse an existing task0 is wasted
# work if CL-LoRA is the only thing missing). Not available in no-argument mode (always both).
#
# Logs: logs_{cre_dist,cre_cllora,ced_dist,ced_cllora}_<ds>.log in the repo root; the
# no-argument mode additionally logs its own progress to logs_run_all.log.
# Safe to re-run: every step below skips work that already completed (each run's own
# resume/skip-if-complete logic), whichever family the dataset belongs to.
set -uo pipefail

cd "$(dirname "$0")"
CLLORA_METHODS="inclora olora tree inflora epi migu gainlora_o gainlora_inf"

family_of () {  # $1=dataset -> echoes cre|ced
    case "$1" in
        tacred|fewrel) echo cre ;;
        maven|rams|geneva|ace) echo ced ;;
        *) return 1 ;;
    esac
}

# Each queue is its own detached process (setsid nohup at the call site), so it survives
# this shell exiting. Arguments are passed positionally rather than via environment
# variables, since a backgrounded `bash -c` does not inherit unexported shell variables.
run_dist_queue () {   # $1=family $2=dataset $3=gpu $4..=perms
    local family=$1 ds=$2 gpu=$3; shift 3
    for p in "$@"; do
        if [ "${family}" = "cre" ]; then
            bash scripts/qwen/cre/run_cre_dist.sh "${ds}" "${p}" "${gpu}"
        else
            PERM=${p} GPU=${gpu} DATA_PREFIX="${ds}_b10_perm" bash scripts/qwen/ced/dist_queue.sh
        fi
        # dist_queue.sh (CED) has no top-level FAILED marker of its own -- it dies silently
        # under set -e -- so log it here regardless of family, or a crashed perm just looks
        # like the queue quietly moved on.
        local rc=$?
        [ "${rc}" -eq 0 ] || echo "[run.sh] FAILED dist queue: family=${family} ds=${ds} perm=${p} (exit ${rc}), see the per-run log above/logs_*_${ds}.log"
    done
}
run_cllora_queue () {  # $1=family $2=dataset $3=gpu $4=methods $5..=perms
    local family=$1 ds=$2 gpu=$3 methods=$4; shift 4
    for p in "$@"; do
        if [ "${family}" = "cre" ]; then
            bash scripts/qwen/cre/run_cre_cllora.sh "${ds}" "${p}" "${gpu}"
        else
            DATA_ROOT="data/${ds}_b10_perm${p}" bash scripts/qwen/ced/run_all_cllora.sh "${gpu}" ${methods}
        fi
        local rc=$?
        [ "${rc}" -eq 0 ] || echo "[run.sh] FAILED cllora queue: family=${family} ds=${ds} perm=${p} (exit ${rc}), see the per-run log above/logs_*_${ds}.log"
    done
}

check_data () {  # $1=family $2=dataset $3=perms -- tokenizes both families; CED tokenization
                  # is NOT done by run_ced_v2.sh itself (it only tokenizes PL/balance side-data,
                  # never the base task data -- confirmed the hard way against a real RAMS run),
                  # so this has to run tools/process_data.py per task itself, same as the
                  # hand-written maven_dist_perm14.sh that has been doing this successfully.
    local family=$1 ds=$2; shift 2
    if [ "${family}" = "cre" ]; then
        for p in "$@"; do
            bash scripts/qwen/cre/prep_cre.sh "${ds}" "${p}" || { echo "[run.sh] tokenize failed for ${ds} perm${p}"; return 1; }
        done
    else
        local PY=${PY:-python3}
        for p in "$@"; do
            [ -s "data/${ds}_b10_perm${p}/streams.json" ] || {
                echo "[run.sh] missing data/${ds}_b10_perm${p}/streams.json -- build it first (tools/build_maven_perms.py --src data/${ds} --out-prefix ${ds}_b10_perm)"
                return 1
            }
            for t in 0 1 2 3 4; do
                local out="processed_data/${ds}_b10_perm${p}/${t}"
                [ -d "${out}/qwen" ] && [ -n "$(ls -A "${out}/qwen" 2>/dev/null)" ] && continue
                PYTHONPATH=. ${PY} tools/process_data.py \
                    --data-dir "data/${ds}_b10_perm${p}/${t}/" --processed-data-dir "${out}" \
                    --model-path Qwen/Qwen3-0.6B --data-process-workers 4 \
                    --max-prompt-length 460 --t-max-prompt-length 640 \
                    --dev-num 1000 --model-type qwen > "/tmp/tok_${ds}_p${p}t${t}.log" 2>&1 || {
                    echo "[run.sh] tokenize failed for ${ds} perm${p} task${t}, see /tmp/tok_${ds}_p${p}t${t}.log"
                    tail -10 "/tmp/tok_${ds}_p${p}t${t}.log"
                    return 1
                }
            done
        done
    fi
}

if [ $# -eq 0 ]; then
    # ---- no-argument mode: everything, one dataset at a time ----
    run_all () {
        local datasets=${RUN_ALL_DATASETS:-"tacred fewrel rams geneva"}
        local perms="0 1 2 3 4"
        for ds in ${datasets}; do
            local family; family=$(family_of "${ds}") || { echo "[run.sh:all] unknown dataset ${ds}, skipping"; continue; }
            echo "[run.sh:all] === ${ds} (${family}) start $(date -Iseconds) ==="
            check_data "${family}" "${ds}" ${perms} || { echo "[run.sh:all] === ${ds} data check failed, skipping ==="; continue; }
            ( run_dist_queue "${family}" "${ds}" 0 ${perms} ) > "logs_${family}_dist_${ds}.log" 2>&1 &
            local dpid=$!
            ( run_cllora_queue "${family}" "${ds}" 1 "${CLLORA_METHODS}" ${perms} ) > "logs_${family}_cllora_${ds}.log" 2>&1 &
            local cpid=$!
            wait "${dpid}" "${cpid}"
            echo "[run.sh:all] === ${ds} done $(date -Iseconds) ==="
        done
        echo "[run.sh:all] ALL DATASETS DONE $(date -Iseconds)"
    }
    setsid nohup bash -c "$(declare -f family_of run_dist_queue run_cllora_queue check_data run_all); run_all" \
        > logs_run_all.log 2>&1 < /dev/null &
    echo "[run.sh] running everything (rams, geneva, tacred, fewrel; one dataset at a time), pid $! -> logs_run_all.log"
    echo "[run.sh] tail -f logs_run_all.log"
    exit 0
fi

DS=$1
PERMS=${2:-"0 1 2 3 4"}
GPU_DIST=${3:-0}
GPU_CLLORA=${4:-1}
QUEUE=${5:-both}

FAMILY=$(family_of "${DS}") || { echo "unknown dataset '${DS}' (expected tacred|fewrel|maven|rams|geneva)"; exit 1; }
case "${QUEUE}" in
    dist|cllora|both) ;;
    *) echo "unknown queue '${QUEUE}' (expected dist|cllora|both)"; exit 1 ;;
esac

echo "[run.sh] dataset=${DS} family=${FAMILY} perms='${PERMS}' gpu_dist=${GPU_DIST} gpu_cllora=${GPU_CLLORA} queue=${QUEUE}"
check_data "${FAMILY}" "${DS}" ${PERMS} || exit 1

if [ "${QUEUE}" = "dist" ] || [ "${QUEUE}" = "both" ]; then
    setsid nohup bash -c "$(declare -f run_dist_queue); run_dist_queue \"\$@\"" _ \
        "${FAMILY}" "${DS}" "${GPU_DIST}" ${PERMS} \
        > "logs_${FAMILY}_dist_${DS}.log" 2>&1 < /dev/null &
    DIST_PID=$!
    echo "[run.sh] distillation queue running on gpu${GPU_DIST}, pid ${DIST_PID} -> logs_${FAMILY}_dist_${DS}.log"
fi
if [ "${QUEUE}" = "cllora" ] || [ "${QUEUE}" = "both" ]; then
    setsid nohup bash -c "$(declare -f run_cllora_queue); run_cllora_queue \"\$@\"" _ \
        "${FAMILY}" "${DS}" "${GPU_CLLORA}" "${CLLORA_METHODS}" ${PERMS} \
        > "logs_${FAMILY}_cllora_${DS}.log" 2>&1 < /dev/null &
    CLLORA_PID=$!
    echo "[run.sh] CL-LoRA queue running on gpu${GPU_CLLORA}, pid ${CLLORA_PID} -> logs_${FAMILY}_cllora_${DS}.log"
fi
echo "[run.sh] tail -f logs_${FAMILY}_dist_${DS}.log logs_${FAMILY}_cllora_${DS}.log"
