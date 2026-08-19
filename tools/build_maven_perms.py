"""Build MAVEN CED task splits for the 5 stream permutations.

MAVEN counterpart of build_ced_perms.py. Two differences drive a separate script:
  * MAVEN ships as a single generated-format split (data/maven/{train,dev,test}.jsonl,
    records are {system_prompt, user_prompt, response}) instead of the raw ACE schema,
    so tasks are carved by filtering each record's event list rather than re-parsing raw.
  * MAVEN has 168 event types and no published task assignment here, so the streams are
    built by a frequency-balanced greedy split (deterministic given --seed) and written to
    streams.json. Swap in a published split by passing --streams-file.

Output matches what the CED runners expect:
    data/<out-prefix><p>/streams.json
    data/<out-prefix><p>/<t>/{train,dev,test}.jsonl

Usage:
    python tools/build_maven_perms.py --cap 10 --out-prefix maven_b10_perm
    python tools/build_maven_perms.py --perms 0 --dry-run
"""
import argparse
import json
import os
import random
from collections import Counter

# same stream orders used for ACE (SharpSeq), applied to the MAVEN streams
PERM = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def events_of(row):
    resp = row["response"]
    if isinstance(resp, str):
        resp = json.loads(resp)
    return resp.get("events", [])


def with_events(row, events):
    out = dict(row)
    out["response"] = json.dumps({"events": events}, ensure_ascii=False)
    return out


def keep(row, types):
    """Record restricted to `types`; empty list means the sentence is a negative here."""
    return [e for e in events_of(row) if len(e) >= 2 and e[1] in types]


def build_streams(train_rows, n_streams, seed):
    """Greedy longest-processing-time split so every stream carries a similar volume.

    Balancing matters more here than for ACE: MAVEN frequencies span 3 to 2678 examples,
    so a naive alphabetical or random split would make some tasks an order of magnitude
    larger than others and confound the continual-learning comparison.
    """
    freq = Counter()
    for row in train_rows:
        for e in events_of(row):
            if len(e) >= 2:
                freq[e[1]] += 1
    rnd = random.Random(seed)
    # sort by count desc, ties broken deterministically-but-arbitrarily via the seed
    order = sorted(freq, key=lambda t: (-freq[t], rnd.random()))
    buckets, loads = [[] for _ in range(n_streams)], [0] * n_streams
    for t in order:
        i = loads.index(min(loads))
        buckets[i].append(t)
        loads[i] += freq[t]
    return [sorted(b) for b in buckets], freq, loads


def build_task(train_rows, dev_rows, test_rows, task_types, seen_types, buffer, cap, rnd):
    """One task of one permutation.

    train = sentences having an event of this task's types (events filtered to those types)
            + a 10% sample of sentences with no event of these types (negatives)
            + the replay buffer accumulated from earlier tasks of this permutation
    dev/test = sentences having an event among all types seen so far (cumulative eval)
    Returns (train, dev, test, new_exemplars).
    """
    pos, neg = [], []
    per_type = {}
    for row in train_rows:
        evs = keep(row, task_types)
        if evs:
            rec = with_events(row, evs)
            pos.append(rec)
            for e in evs:                       # first `cap` per type become exemplars
                bucket = per_type.setdefault(e[1], [])
                if len(bucket) < cap:
                    bucket.append(rec)
        else:
            neg.append(with_events(row, []))

    train = list(pos)
    train.extend(rnd.sample(neg, min(len(neg), len(pos) // 10)))
    train.extend(buffer)

    def cumulative(rows):
        out = []
        for row in rows:
            evs = keep(row, seen_types)
            if evs:
                out.append(with_events(row, evs))
        return out

    exemplars = [r for v in per_type.values() for r in v]
    return train, cumulative(dev_rows), cumulative(test_rows), exemplars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/maven")
    ap.add_argument("--out-prefix", default="maven_b10_perm")
    ap.add_argument("--cap", type=int, default=10, help="exemplars kept per event type")
    ap.add_argument("--n-streams", type=int, default=5)
    ap.add_argument("--perms", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--streams-file", help="reuse an existing streams.json instead of rebuilding")
    ap.add_argument("--dry-run", action="store_true", help="print the split, write nothing")
    args = ap.parse_args()

    train_rows = load_jsonl(f"{args.src}/train.jsonl")
    dev_rows = load_jsonl(f"{args.src}/dev.jsonl")
    test_rows = load_jsonl(f"{args.src}/test.jsonl")
    print(f"loaded train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)}")

    if args.streams_file:
        streams = json.load(open(args.streams_file))
        freq, loads = Counter(), [0] * len(streams)
    else:
        streams, freq, loads = build_streams(train_rows, args.n_streams, args.seed)
    print(f"streams: sizes={[len(s) for s in streams]} train-events={loads}")
    for i, s in enumerate(streams):
        print(f"  stream {i} ({len(s)} types, {loads[i]} events): {s[:6]}{' ...' if len(s) > 6 else ''}")

    if args.dry_run:
        return

    for p in args.perms:
        order = PERM[p]
        rnd = random.Random(args.seed + p)
        out_root = f"data/{args.out_prefix}{p}"
        os.makedirs(out_root, exist_ok=True)
        perm_streams = [streams[i] for i in order]
        with open(f"{out_root}/streams.json", "w", encoding="utf-8") as f:
            json.dump(perm_streams, f)

        buffer, seen = [], set()
        print(f"\n=== perm {p} (stream order {order}, cap {args.cap}) ===")
        for t, task_types in enumerate(perm_streams):
            seen.update(task_types)
            train, dev, test, exemplars = build_task(
                train_rows, dev_rows, test_rows, set(task_types), set(seen),
                buffer, args.cap, rnd)
            out_dir = f"{out_root}/{t}"
            os.makedirs(out_dir, exist_ok=True)
            for name, rows in [("train", train), ("dev", dev), ("test", test)]:
                with open(f"{out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            buffer.extend(exemplars)
            print(f"task {t}: streams[{order[t]}] ({len(task_types)} types) "
                  f"train={len(train)} dev={len(dev)} test={len(test)} buffer_after={len(buffer)}")


if __name__ == "__main__":
    main()
