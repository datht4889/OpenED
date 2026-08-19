# OpenED — Continual Event Detection (CED) & Continual Relation Extraction (CRE)

Qwen3-0.6B trained continually across tasks (LoRA, teacher = previous task's checkpoint).
Two baseline families — distillation (KD/RKL/SFKL/SRKL/CSD/DistiLLM/AMiD) and CL-LoRA
(IncLoRA/O-LoRA/TreeLoRA/InfLoRA/EPI/MIGU/GainLoRA×2) — across two dataset families:

- **CED** (event detection, 5 tasks/permutation): ACE, MAVEN, RAMS, GENEVA
- **CRE** (relation extraction, 10 tasks/permutation): TACRED, FewRel

All datasets use 5 permutations (SharpSeq stream orders).

## Setup

```bash
conda create -n mta python=3.12 && conda activate mta
pip install -r requirements.txt
export CUDA_HOME=$HOME/miniconda3/envs/mta   # needed for deepspeed to detect nvcc
```

## Run

```bash
bash run.sh                                                      # everything, see below
bash run.sh <tacred|fewrel|maven|rams|geneva> ["perms"] [gpu_dist] [gpu_cllora]
```

`bash run.sh` with **no arguments** runs RAMS and GENEVA in full, then TACRED and FewRel —
one dataset at a time (dist + CL-LoRA concurrently within a dataset, next dataset only
starts once both queues of the current one finish, so no GPU ever gets two queues). For
TACRED/FewRel this only trains what is actually missing: `run_cre_dist.sh` /
`run_cre_cllora.sh` check a completion marker per method+perm and skip whatever already
finished, so re-running the full perm range 0-4 is a no-op for anything already done.
Progress: `tail -f logs_run_all.log` (plus the per-dataset logs listed below once a
dataset starts). Override the dataset list/order with `RUN_ALL_DATASETS="ds1 ds2 ..."`.

One call runs one dataset end to end: tokenizes/checks the requested permutations, then
launches the distillation queue (7 methods) on `gpu_dist` and the CL-LoRA queue (8 methods)
on `gpu_cllora`, each working through its permutations in the background. No other setup
step is needed.

```bash
bash run.sh rams                   # perms 0-4, distillation on gpu0, CL-LoRA on gpu1
bash run.sh geneva "0 1 2"         # only perm0-2
bash run.sh maven "0" 0 1          # single perm, explicit GPU assignment
bash run.sh tacred                 # CRE dataset, same interface

tail -f logs_ced_dist_rams.log logs_ced_cllora_rams.log
```

CED datasets (maven/rams/geneva) need their perm split built once before the first run —
CRE datasets tokenize automatically inside `run.sh`, CED datasets just need the split to
already exist on disk:

```bash
python tools/build_maven_perms.py --src data/rams --out-prefix rams_b10_perm --cap 10
python tools/build_maven_perms.py --src data/geneva --out-prefix geneva_b10_perm --cap 10
```

(the script is dataset-agnostic despite the filename — it only assumes the
`{system_prompt, user_prompt, response}` record schema shared by ACE/MAVEN/RAMS/GENEVA)

Notes:
- `bash run.sh` with no arguments errors out — the dataset is required, and each call
  covers exactly one dataset. Run once per dataset you need. On a 2-GPU box that means
  running them one after the other, since each call already claims both GPUs (one queue
  per GPU — see below).
- Never run two queues on the same GPU. The memory/disk guards in the per-family runners
  (`scripts/qwen/cre/run_cre_*.sh` for CRE, `scripts/qwen/ced/{dist_queue,run_all_cllora}.sh`
  for CED) are snapshots, not reservations, and two queues racing for one card's memory
  between tasks will OOM each other.
- Safe to re-run: every step (tokenize/prep, both queue scripts) checks for its own
  completion marker before starting and skips finished work, so a re-run after any
  interruption just resumes.

## Results

```bash
python tools/ced_collect.py --host-label <label> [--upload]
```
