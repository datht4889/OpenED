"""R6: teacher pseudo-labeling for CED.

The previous-task teacher runs inference on the CURRENT task's train inputs and
detects events of OLD types (the ones stripped from this task's gold responses
— the main measured source of forgetting: 138-242 sentences/task on ACE).
Detected old-type events are merged into the gold responses, restoring
supervision the split removed.

Rules:
- only events whose type is in streams[<task_id]
- trigger text must appear verbatim in the input sentence
- skip if the gold response already has an event with the same (trigger, type)
- replay exemplars (all-old-type responses) are left untouched

Usage:
  python tools/ced_pseudo_label.py --teacher <merged_dir> --data-dir data/ace_b10_perm0/1 \
      --streams data/ace_b10_perm0/streams.json --task-id 1 --out data/r6_run/1 [--gpu 0]
Writes train.jsonl (augmented) + dev/test copied unchanged, plus pl_stats.json.
"""
import argparse
import json
import os
import re
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def input_text_of(user_prompt):
    m = re.search(r"Given an input text: ?\n?(.*?)\n\nYour task", user_prompt, re.S)
    return m.group(1).strip() if m else None


def parse_events(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("events", [])
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--lexicon-filter", type=int, default=1,
                    help="1: only accept (trigger,type) pairs seen in old tasks' gold train data")
    args = ap.parse_args()

    streams = json.load(open(args.streams))
    old_types = set()
    for s in streams[:args.task_id]:
        old_types.update(s)
    assert old_types, "task 0 has no old types — pseudo-labeling starts at task 1"

    # trigger lexicon from old tasks' gold annotations: stripped triggers are real
    # ACE triggers so they re-occur in old train data; hallucinated pairs don't.
    lexicon = set()
    if args.lexicon_filter:
        base = os.path.dirname(os.path.normpath(args.data_dir))
        for t_prev in range(args.task_id):
            p = os.path.join(base, str(t_prev), "train.jsonl")
            for line in open(p):
                for e in json.loads(json.loads(line)["response"]).get("events", []):
                    if e[1] in old_types:
                        lexicon.add((e[0].lower(), e[1]))
        print(f"lexicon: {len(lexicon)} (trigger,type) pairs from tasks 0..{args.task_id-1}")

    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "train.jsonl"))]

    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                 device_map={"": device})
    model.eval()

    # candidates: rows whose gold contains at least one NEW-type event (skip
    # replay exemplars and pure no-event rows keeps teacher calls low-risk)
    cand_idx = []
    for i, r in enumerate(rows):
        gold = json.loads(r["response"]).get("events", [])
        types = {e[1] for e in gold}
        if types - old_types:
            cand_idx.append(i)

    n_aug_rows = 0
    n_aug_events = 0
    for b in range(0, len(cand_idx), args.batch_size):
        idxs = cand_idx[b:b + args.batch_size]
        prompts = []
        for i in idxs:
            r = rows[i]
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": r["system_prompt"]},
                 {"role": "user", "content": r["user_prompt"]}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False))
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        for i, text in zip(idxs, texts):
            r = rows[i]
            sent = input_text_of(r["user_prompt"]) or ""
            gold = json.loads(r["response"]).get("events", [])
            gold_keys = {(e[0], e[1]) for e in gold}
            added = []
            for e in parse_events(text):
                if not isinstance(e, list) or len(e) < 2:
                    continue
                trig, ty = e[0], e[1]
                if ty not in old_types:
                    continue
                if not isinstance(trig, str) or trig not in sent:
                    continue
                if (trig, ty) in gold_keys:
                    continue
                if args.lexicon_filter and (trig.lower(), ty) not in lexicon:
                    continue
                args_clean = []
                if len(e) > 2 and isinstance(e[2], list):
                    for a in e[2]:
                        if isinstance(a, list) and len(a) >= 2 \
                                and isinstance(a[0], str) and isinstance(a[1], str):
                            args_clean.append([a[0], a[1]])
                ev = [trig, ty, args_clean,
                      e[3] if len(e) > 3 and isinstance(e[3], str) else ""]
                added.append(ev)
                gold_keys.add((trig, ty))
            if added:
                rows[i]["response"] = json.dumps({"events": gold + added})
                n_aug_rows += 1
                n_aug_events += len(added)
        print(f"pseudo-label {min(b + args.batch_size, len(cand_idx))}/{len(cand_idx)} "
              f"(+{n_aug_events} events on {n_aug_rows} rows)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ["dev.jsonl", "test.jsonl"]:
        shutil.copy(os.path.join(args.data_dir, split), os.path.join(args.out, split))
    stats = {"task_id": args.task_id, "candidates": len(cand_idx),
             "aug_rows": n_aug_rows, "aug_events": n_aug_events}
    with open(os.path.join(args.out, "pl_stats.json"), "w") as f:
        json.dump(stats, f)
    print("STATS", json.dumps(stats))


if __name__ == "__main__":
    main()
