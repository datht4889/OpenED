#!/usr/bin/env python3
"""Push the continual-learning splits in data/ to one Hugging Face dataset repo per corpus.

Each repo gets the same shape, whatever the local directory names were:

    raw/{train,dev,test}.jsonl      the corpus before it is cut into tasks
    perm<k>/streams.json            the label groups of that permutation, in task order
    perm<k>/<task>/{train,dev,test}.jsonl
    groups/                         CRE only: the label groups the permutations draw from
    README.md                       dataset card

    python tools/push_hf_datasets.py --dry-run                  # print the plan, touch nothing
    python tools/push_hf_datasets.py --datasets maven rams      # push two of them
    python tools/push_hf_datasets.py --public --datasets maven  # default is private

Reads HF_TOKEN / HF_USERNAME from the environment (../.env holds both in this checkout).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# corpus -> (raw dir, {published dir: local dir}, licence note for the card)
LAYOUT = {
    "ace": dict(
        raw="ace",
        perms={"perm0": "ace"},          # only one ordering is checked in; perm1-4 live on the servers
        tasks_in_raw=True,
        licence="LDC2006T06 (ACE 2005). Redistribution is NOT permitted: keep this repo private/gated.",
        restricted=True,
    ),
    "maven": dict(raw="maven", perms={f"perm{k}": f"maven_b10_perm{k}" for k in range(5)},
                  licence="MAVEN is released by THUNLP under its own terms; check before making public.",
                  restricted=False),
    "rams": dict(raw="rams", perms={f"perm{k}": f"rams_b10_perm{k}" for k in range(5)},
                 licence="RAMS (AI2) is distributed under its own terms; check before making public.",
                 restricted=False),
    "geneva": dict(raw="geneva", perms={f"perm{k}": f"geneva_b10_perm{k}" for k in range(5)},
                   licence="GENEVA derives from FrameNet; FrameNet's licence carries over.",
                   restricted=False),
    "tacred": dict(raw="tacred", perms={f"perm{k}": f"tacred_perm{k}" for k in range(5)},
                   groups="tacred_groups",
                   licence="LDC2018T24 (TACRED). Redistribution is NOT permitted: keep this repo private/gated.",
                   restricted=True),
    "fewrel": dict(raw="fewrel", perms={f"perm{k}": f"fewrel_perm{k}" for k in range(5)},
                   groups="fewrel_groups",
                   licence="FewRel is released by THUNLP under its own terms; check before making public.",
                   restricted=False),
}

CARD = """---
license: other
task_categories:
- text-generation
language:
- en
tags:
- continual-learning
- {family_tag}
- event-extraction
---

# {name} — continual learning splits

The {name} corpus cut into the task streams used by the OpenED continual-learning
experiments ({family}, {ntask} tasks per permutation, {nperm} permutation{plural}).

## Layout

```
raw/{{train,dev,test}}.jsonl        the corpus before it is cut into tasks
perm<k>/streams.json               the label groups of permutation k, in task order
perm<k>/<task>/{{train,dev,test}}.jsonl
{groups_line}```

`train.jsonl` of task *t* holds that task's labels plus a replay buffer of at most 10
examples per earlier label. `test.jsonl` is **cumulative**: every label seen up to and
including task *t*, so a single number measures both what was learned and what was kept.

## Record format

One JSON object per line, the prompt format the generative models are trained on:

```json
{{"system_prompt": "...", "user_prompt": "Given an input text: ...", "response": "{{\\"events\\": [...]}}"}}
```

## Licence

{licence}

These files are a re-packaging of the original corpus: the licence of the source data
applies unchanged, and this repo grants nothing beyond it.
"""


def stage(name: str, spec: dict, out: Path) -> list[str]:
    """Build the published tree under `out` with hardlinks; return a printable file plan."""
    plan = []
    if out.exists():
        shutil.rmtree(out)

    def put(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        plan.append(f"{dst.relative_to(out)}  <-  {src.relative_to(DATA)}")

    raw = DATA / spec["raw"]
    for split in ("train", "dev", "test"):
        f = raw / f"{split}.jsonl"
        if f.exists():
            put(f, out / "raw" / f"{split}.jsonl")

    for pub, local in spec["perms"].items():
        src = DATA / local
        streams = src / "streams.json"
        if streams.exists():
            put(streams, out / pub / "streams.json")
        for task in sorted(d for d in src.iterdir() if d.is_dir() and d.name.isdigit()):
            for split in ("train", "dev", "test"):
                f = task / f"{split}.jsonl"
                if f.exists():
                    put(f, out / pub / task.name / f"{split}.jsonl")

    if spec.get("groups"):
        src = DATA / spec["groups"]
        for f in sorted(src.rglob("*")):
            if f.is_file():
                put(f, out / "groups" / f.relative_to(src))

    nperm = len(spec["perms"])
    ntask = len({p.parent.name for p in out.glob("perm0/*/train.jsonl")}) or "?"
    family = "CRE / continual relation extraction" if spec.get("groups") else "CED / continual event detection"
    (out / "README.md").write_text(CARD.format(
        name=name.upper(),
        family=family,
        family_tag="relation-extraction" if spec.get("groups") else "event-detection",
        ntask=ntask,
        nperm=nperm,
        plural="" if nperm == 1 else "s",
        groups_line="groups/                            the label groups the permutations draw from\n" if spec.get("groups") else "",
        licence=spec["licence"],
    ))
    plan.append("README.md  <-  generated")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LAYOUT), choices=list(LAYOUT))
    ap.add_argument("--namespace", default=os.environ.get("HF_USERNAME"))
    ap.add_argument("--prefix", default="cl-", help="repo name prefix (default: cl-<dataset>)")
    ap.add_argument("--public", action="store_true", help="create public repos (default: private)")
    ap.add_argument("--allow-restricted-public", action="store_true",
                    help="publish ACE/TACRED publicly despite their LDC licences")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage-dir", default=os.environ.get("STAGE_DIR", "/tmp/hf_stage"))
    a = ap.parse_args()

    if not a.namespace:
        sys.exit("no --namespace and no HF_USERNAME in the environment")
    token = os.environ.get("HF_TOKEN")
    if not a.dry_run and not token:
        sys.exit("no HF_TOKEN in the environment")

    if a.public:
        blocked = [d for d in a.datasets if LAYOUT[d]["restricted"]]
        if blocked and not a.allow_restricted_public:
            sys.exit(f"refusing --public for licence-restricted corpora: {', '.join(blocked)}\n"
                     f"ACE05 is LDC2006T06 and TACRED is LDC2018T24; neither licence allows\n"
                     f"redistribution. Push them privately, drop them from --datasets, or pass\n"
                     f"--allow-restricted-public if you have decided to publish them anyway.")
        if blocked:
            print(f"!! publishing licence-restricted corpora publicly: {', '.join(blocked)}")

    api = None
    if not a.dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

    for name in a.datasets:
        spec = LAYOUT[name]
        repo_id = f"{a.namespace}/{a.prefix}{name}"
        out = Path(a.stage_dir) / name
        plan = stage(name, spec, out)
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**20
        print(f"\n=== {repo_id}  ({'public' if a.public else 'private'})  "
              f"{len(plan)} files, {size:.0f} MB")
        for line in plan[:6]:
            print("   ", line)
        if len(plan) > 6:
            print(f"    ... {len(plan) - 6} more")
        if a.dry_run:
            continue
        api.create_repo(repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out),
                          commit_message=f"{name} continual-learning splits")
        print(f"    pushed -> https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
