"""Build the pool for the final calibration epoch of task t (CL-DETR-style).

Two modes (--dist):
  uniform  — legacy F4: up to N sentences per type, all types forced equal.
             Known-bad: teaches a uniform type prior, far from the test
             distribution (F4 run collapsed to 0.318).
  matched  — CL-DETR-style: the pool keeps a REALISTIC type distribution.
             Old types come from the replay exemplar rows already present in
             the (staged) train data — taken as-is, no re-balancing; the new
             task contributes a small seeded sample (--per-type ROWS selected
             per new type; multi-label rows picked for one type may add a few
             incidental mentions of another) so the calibration pass cannot
             erase what was just learned.

Usage:
  python tools/ced_balance_pool.py --data-dir <staged task dir> --streams s.json \
      --task-id T --dist matched --per-type 10 --out <dir>
dev/test copied unchanged.
"""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--dist", choices=["uniform", "matched"], default="uniform")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "train.jsonl"))]

    if args.dist == "matched":
        streams = json.load(open(args.streams))
        old_types = set()
        for s in streams[:args.task_id]:
            old_types.update(s)
        new_types = set(streams[args.task_id])

        # old side: every replay exemplar row (all its event types are old) —
        # kept verbatim, natural proportions, no re-balancing
        pool, seen = [], set()
        new_by_type = defaultdict(list)
        for r in rows:
            events = json.loads(r["response"]).get("events", [])
            types = {e[1] for e in events if isinstance(e, list) and len(e) >= 2}
            if types and types <= old_types:
                key = json.dumps(r, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    pool.append(r)
            else:
                for ty in types & new_types:
                    new_by_type[ty].append(r)

        # new side: small seeded sample per new type so calibration cannot
        # erase the task just learned
        for ty, cand in new_by_type.items():
            random.shuffle(cand)
            for r in cand[:args.per_type]:
                key = json.dumps(r, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    pool.append(r)
        random.shuffle(pool)
    else:
        by_type = defaultdict(list)
        for r in rows:
            events = json.loads(r["response"]).get("events", [])
            for ty in {e[1] for e in events}:
                by_type[ty].append(r)

        pool, seen = [], set()
        for ty, cand in by_type.items():
            random.shuffle(cand)
            for r in cand[:args.per_type]:
                key = id(r)
                if key not in seen:
                    seen.add(key)
                    pool.append(r)
        random.shuffle(pool)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ["dev.jsonl", "test.jsonl"]:
        shutil.copy(os.path.join(args.data_dir, split), os.path.join(args.out, split))
    print(f"balance pool ({args.dist}): {len(pool)} rows")


if __name__ == "__main__":
    main()
