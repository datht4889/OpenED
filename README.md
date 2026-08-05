# OpenED — Continual Event/Relation Extraction

Qwen3-0.6B student trained continually across tasks (LoRA, teacher = previous-task
checkpoint). Two baseline families: **distillation** (KD/RKL/SFKL/SRKL/CSD/DistiLLM/AMiD)
and **CL-LoRA** (IncLoRA/O-LoRA/TreeLoRA/InfLoRA/EPI/MIGU/GainLoRA×2). Four datasets:
ACE, MAVEN (event detection) and TACRED, FewRel (continual relation extraction, CRE).

## Setup

```bash
conda create -n mta python=3.12 && conda activate mta
pip install -r requirements.txt
export CUDA_HOME=$HOME/miniconda3/envs/mta   # needed for deepspeed to detect nvcc
```

One GPU per run everywhere below; nothing here uses multi-GPU/DDP.

## Data layout

```
data/<dataset>_perm<0-4>/streams.json          # task types, in stream order
data/<dataset>_perm<0-4>/<task_id>/{train,dev,test}.jsonl
```

Each record is `{system_prompt, user_prompt, response}`; `response` is
`{"events": [[trigger_or_subject, type_or_relation, args_or_object, description]]}`.
CRE reuses this exact schema — a relation is encoded as a single-argument "event".

- **ACE / MAVEN**: 5 tasks/perm. MAVEN's 168 relation types were split into 5
  frequency-balanced streams (`tools/build_maven_perms.py`); ACE's split comes from
  SharpSeq.
- **TACRED / FewRel**: 10 tasks/perm, built from order-independent shards
  (`data/<ds>_groups/`) via `tools/build_re_perms.py`. Add more permutations with:
  ```bash
  python tools/build_re_perms.py --dataset tacred --perms 1 2 3 4
  python tools/build_re_perms.py --dataset tacred --gate   # verify the assembly rule
  ```

## Running ACE / MAVEN

```bash
# tokenize once per dataset+perm (writes processed_data/)
bash scripts/qwen/ced/run_ced_v2.sh --run-name t0 --data-prefix ace_b10_perm --perm 0 \
    --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --bs 2 --acc 16 --greedy 1 \
    --gpus "0" --start-task 0 --end-task 0            # shared task0 (plain CE)

# one distillation baseline, tasks 1..4, teacher = the shared task0 above
bash scripts/qwen/ced/run_ced_v2.sh --run-name rkl0 --mode ce_kd --data-prefix ace_b10_perm \
    --perm 0 --kd-type rkl --w-span 0 --rank 16 --alpha 64 --epochs 5 --bs 2 --acc 16 \
    --greedy 1 --gpus "0" --start-task 1 --end-task 4

# one CL-LoRA baseline, all 5 tasks (standalone engine, reads raw JSONL directly)
bash scripts/qwen/ced/run_cllora.sh --method tree --data-root data/ace_b10_perm0 \
    --rank 16 --alpha 64 --epochs 5 --gpu 1 --protocol v2
```

`--kd-type` ∈ `fkl rkl sfkl srkl csd adaptive-srkl(=DistiLLM) adaptive-amid(=AMiD)`.
`--method` ∈ `inclora olora tree inflora epi migu gainlora_o gainlora_inf`.
DistiLLM/AMiD need `--extra "--student-gen --init-threshold 0.0 --loss-eps 0.1 --capacity 1000"`.

## Running CRE (TACRED / FewRel)

`run.sh` tokenizes the requested permutations and launches both baseline families, one
queue per GPU, each queue working through its permutations in sequence:

```bash
bash run.sh tacred                 # perms 0-4, distillation on gpu0, CL-LoRA on gpu1
bash run.sh fewrel "0 1 2"         # only perm0-2
bash run.sh tacred "0" 0 1         # single perm, explicit GPU assignment

tail -f logs_cre_dist_tacred.log logs_cre_cllora_tacred.log
```

Never run two queues on the same GPU: the memory/disk guards in
`scripts/qwen/cre/run_cre_*.sh` are snapshots, not reservations, and two queues racing
for one card's memory between tasks will OOM each other.

## Collecting results

```bash
python tools/ced_collect.py --host-label <label> [--upload]   # writes summary.json per run
```

Rerunning any script is safe — every runner checks for its own completion marker
(`task<last>/log.txt` or `.complete`) before starting and skips finished work.
