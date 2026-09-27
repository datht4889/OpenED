#!/usr/bin/env bash
# Bring this repo up on a host and prove it works, then hand over to run.sh.
#
#   bash project_commands.sh                    # env + unzip + checks + smoke run
#   SKIP_INSTALL=1 bash project_commands.sh     # deps already installed
#   SKIP_SMOKE=1   bash project_commands.sh     # setup only, no GPU needed
#   RUN_SWEEP=1    bash project_commands.sh     # ... and launch the real sweep at the end
#
# Everything here is idempotent: re-running skips what is already in place. Training itself
# lives in run.sh; this file is the path from a fresh clone to a first working run.
#
# Knobs (all optional):
#   VENV          venv to activate, e.g. /mnt/local/uvenvs/opened  (default: use the current
#                 environment, whatever `python` already resolves to)
#   PY / ENV_BIN  interpreter and env bin/ for the runners (default: derived from `python`)
#   SMOKE_GPU     GPU for the smoke run (default 0)
#   SMOKE_LIMIT   rows per split in the smoke run (default 64)
#   SKIP_INSTALL / SKIP_SMOKE / RUN_SWEEP as above
set -euo pipefail
cd "$(dirname "$0")"

step () { echo; echo "=== $* ==="; }
have () { [ -e "$1" ]; }

# ---------------------------------------------------------------- 1. environment
step "1. environment"
if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    echo "activated ${VENV}"
else
    echo "no VENV given, using the current environment"
fi
PY=${PY:-$(command -v python || command -v python3)}
[ -x "${PY}" ] || { echo "no python found; set PY or activate an env"; exit 1; }
ENV_BIN=${ENV_BIN:-$(dirname "${PY}")}
export PY ENV_BIN
echo "PY=${PY}"
echo "ENV_BIN=${ENV_BIN}"
"${PY}" -V

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1, not touching dependencies"
elif "${PY}" -c "import torch, transformers, peft" 2>/dev/null; then
    echo "torch/transformers/peft already importable, skipping install"
else
    # opened.txt is the uv pin list for hosts that cannot reach GitHub (its en_core_web_sm
    # line was a GitHub wheel URL and has been moved to download.txt as a zip).
    req=opened.txt; [ -f "${req}" ] || req=requirements.txt
    echo "installing from ${req}"
    "${PY}" -m pip install -r "${req}"
fi

# ---------------------------------------------------------------- 2. spaCy model
step "2. spaCy model (en_core_web_sm)"
if have en_core_web_sm/meta.json; then
    echo "already unpacked: $("${PY}" -c "import json;m=json.load(open('en_core_web_sm/meta.json'));print(m['lang']+'_'+m['name'], m['version'])")"
elif have en_core_web_sm.zip; then
    # Python's zipfile, not unzip: the image has no unzip binary and no sudo.
    "${PY}" -c "import zipfile; zipfile.ZipFile('en_core_web_sm.zip').extractall('.')"
    rm -rf __MACOSX en_core_web_sm.zip          # the zip was packed on a Mac
    echo "unpacked -> ./en_core_web_sm  (spacy.load(\"en_core_web_sm\") works from this directory)"
else
    echo "no en_core_web_sm.zip here -- download.txt fetches it; skipping (nothing in this repo imports spaCy)"
fi

# ---------------------------------------------------------------- 3. data
step "3. data"
missing=0
for f in data/maven_b10_perm0/streams.json data/maven_b10_perm0/0/train.jsonl \
         data/rams_b10_perm0/streams.json data/rams_b10_perm0/0/train.jsonl; do
    if have "${f}"; then
        printf '  %-44s ok\n' "${f}"
    else
        printf '  %-44s MISSING\n' "${f}"
        missing=1
    fi
done
for f in data/tacred_perm0/streams.json data/fewrel_perm0/streams.json data/tacred_groups/groups.json; do
    have "${f}" && printf '  %-44s ok\n' "${f}" || printf '  %-44s absent (CRE, fine if this host only runs CED)\n' "${f}"
done
[ "${missing}" = "0" ] || {
    echo "CED splits are missing -- fetch them with download.txt (they land in these exact"
    echo "directory names, nothing to rename), then re-run this script."
    exit 1
}

# ---------------------------------------------------------------- 4. smoke run
step "4. smoke run"
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
    echo "SKIP_SMOKE=1, not training"
else
    SMOKE_GPU=${SMOKE_GPU:-0}
    SMOKE_LIMIT=${SMOKE_LIMIT:-64}
    # A fresh host usually has no Qwen3-0.6B in its HF cache, and the CL-LoRA path defaults
    # to offline, which would fail with a cache miss instead of downloading.
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

    echo "4a. tokenizer path: rams perm0, all 5 tasks, no GPU"
    bash run.sh rams "0" "${SMOKE_GPU}" "${SMOKE_GPU}" prep

    echo
    echo "4b. training path: one CL-LoRA task on gpu${SMOKE_GPU}, ${SMOKE_LIMIT} rows, 1 epoch"
    rm -rf results/smoke_cllora
    bash scripts/qwen/ced/run_cllora.sh \
        --method inclora --data-root data/rams_b10_perm0 \
        --num-tasks 5 --end-task 0 --epochs 1 --limit "${SMOKE_LIMIT}" \
        --gpu "${SMOKE_GPU}" --py "${PY}" --save results/smoke_cllora
    echo "smoke run OK -> results/smoke_cllora (throwaway, delete whenever)"
fi

# ---------------------------------------------------------------- 5. the real sweep
step "5. sweep"
SWEEP_CMD='MISSING_PLAN="rams:0 1 2 3 4:both" bash run.sh'
if [ "${RUN_SWEEP:-0}" = "1" ]; then
    echo "launching: ${SWEEP_CMD}"
    env MISSING_PLAN="rams:0 1 2 3 4:both" bash run.sh
    echo "tail -f logs_run_all.log"
else
    cat <<EOF
not launched (pass RUN_SWEEP=1 to launch from here). When you are ready:

  ${SWEEP_CMD}
  tail -f logs_run_all.log

results/ is gitignored, so on a fresh host no completion markers arrive with the repo and
nothing is skipped: a bare \`bash run.sh\` would retrain MAVEN perm0-1 as well. Either copy
the old host's results/ over first, or keep MISSING_PLAN narrowed as above. run.sh's header
has the rest, including GPU_DIST_ALL / GPU_CLLORA_ALL when gpu0+gpu1 are not the free pair.

Collect afterwards:  python tools/ced_collect.py --host-label <label> [--upload]
EOF
fi
