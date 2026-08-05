# OpenED — Continual Relation Extraction (CRE)

Qwen3-0.6B trained continually across relation-extraction tasks (LoRA, teacher = previous
task's checkpoint). Two baseline families — distillation (KD/RKL/SFKL/SRKL/CSD/DistiLLM/AMiD)
and CL-LoRA (IncLoRA/O-LoRA/TreeLoRA/InfLoRA/EPI/MIGU/GainLoRA×2) — on TACRED and FewRel,
5 permutations each, 10 tasks per permutation.

## Setup

```bash
conda create -n mta python=3.12 && conda activate mta
pip install -r requirements.txt
export CUDA_HOME=$HOME/miniconda3/envs/mta   # needed for deepspeed to detect nvcc
```

## Run

```bash
bash run.sh <tacred|fewrel> ["perms"] [gpu_dist] [gpu_cllora]
```

One call runs one dataset end to end: tokenizes the requested permutations, then launches
the distillation queue (7 methods) on `gpu_dist` and the CL-LoRA queue (8 methods) on
`gpu_cllora`, each working through its permutations in the background. No other setup step
is needed.

```bash
bash run.sh tacred                 # perms 0-4, distillation on gpu0, CL-LoRA on gpu1
bash run.sh fewrel "0 1 2"         # only perm0-2
bash run.sh tacred "0" 0 1         # single perm, explicit GPU assignment

tail -f logs_cre_dist_tacred.log logs_cre_cllora_tacred.log
```

Notes:
- `bash run.sh` with no arguments errors out — the dataset is required, and each call
  covers exactly one dataset. For both, run twice: `bash run.sh tacred` then
  `bash run.sh fewrel`. On a 2-GPU box that means running them one after the other, since
  each call already claims both GPUs (one queue per GPU — see below).
- Never run two queues on the same GPU. The memory/disk guards in
  `scripts/qwen/cre/run_cre_*.sh` are snapshots, not reservations, and two queues racing
  for one card's memory between tasks will OOM each other.
- Safe to re-run: every step (`prep_cre.sh`, both queue scripts) checks for its own
  completion marker before starting and skips finished work, so a re-run after any
  interruption just resumes.

## Results

```bash
python tools/ced_collect.py --host-label <label> [--upload]
```
