"""Verify the --peft-path adapter-reload path builds the exact trained structure.

Run on a server with a real adapter checkpoint (CPU is fine):
  python tools/verify_peft_path.py --base Qwen/Qwen3-0.6B --adapter <ckpt dir> \
      --r 16 --alpha 64

Reports whether the rebuilt LoraConfig (as used in utils.get_model with
--peft-path) produces the same set of adapter parameter names as the trained
adapter, and whether load_state_dict is lossless.
"""
import argparse

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def adapter_keys(model):
    return {k for k in model.state_dict() if "lora" in k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=64)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    trained = PeftModel.from_pretrained(base, args.adapter)
    trained_keys = adapter_keys(trained)
    state = dict(trained.state_dict().items())
    print(f"trained adapter: {len(trained_keys)} lora params")
    del trained

    for label, cfg in [
        ("NO target_modules (old buggy path)",
         LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.r, lora_alpha=args.alpha, lora_dropout=0.1)),
        ("WITH 7 target_modules (fixed path)",
         LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.r, lora_alpha=args.alpha, lora_dropout=0.1,
                    target_modules=TARGETS)),
    ]:
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
        rebuilt = get_peft_model(base, cfg)
        keys = adapter_keys(rebuilt)
        missing = trained_keys - keys
        extra = keys - trained_keys
        print(f"\n[{label}]")
        print(f"  rebuilt lora params: {len(keys)} | missing vs trained: {len(missing)} | extra: {len(extra)}")
        if missing:
            print("  e.g. missing:", sorted(missing)[:3])
        try:
            rebuilt.load_state_dict(state)
            print("  load_state_dict(strict=True): OK")
        except Exception as e:
            print(f"  load_state_dict(strict=True): FAIL — {str(e)[:200]}")
        del rebuilt, base


if __name__ == "__main__":
    main()
