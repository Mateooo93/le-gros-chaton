#!/usr/bin/env python3
"""Verify the trajectory-SFT adapter on a GPU (run on Kaggle T4).

Loads the EXACT training-time stack — base (4-bit, fp16 compute) + Fable5
adapter + trajectory LoRA nested on top (the saved keys are
base_model.model.base_model.model... because get_peft_model wrapped an
already-PEFT model) — then checks the two behaviors trajectory SFT is meant
to bake in:

  1. TOOL-CALL FORMAT: does the model emit ```tool\nargs``` blocks with
     valid tool names (and not collapse to empty/plain-prose responses)?
  2. SELF-AWARENESS: does the model emit state-sheets ([STATE] ...) and
     self-review lines without being prompted with the state-sheet template?

Usage (on Kaggle/GPU box):
    HF_TOKEN=... python3 colab/verify_adapter.py \
        [--base Qwen/Qwen3.5-9B]
        [--fable mateo0093/le-gros-chaton-qwen]
        [--traj mateo0093/le-gros-chaton-qwen-traj-sft]
        [--n 12] [--max-new 384]
"""
import argparse
import json
import os
import re
import sys
import time

TRACE_TASK = """Fix the failing test in src/parser.py. Start by exploring the repo structure, 
run the tests, then fix the bug and verify. The parser mishandles nested 
brackets in string literals."""

TRACE_TOOLS = """You have these tools:
  list_dir    : list files in a directory
  read_file   : read a file
  search_code : search for a pattern
  run_test    : run a test (e.g. pytest tests/test_x.py)
  finish      : submit your answer (put your summary + diff in the args)
"""


def log(*a):
    print("[verify-adapter]", *a, flush=True)


def load_stack(base: str, fable: str, traj: str):
    """Replicate trajectory_sft.py's load: base 4-bit -> Fable5 -> LoRA on top.

    Training did: PeftModel.from_pretrained(base, fable) then
    get_peft_model(inner, LoraConfig). The saved traj weights carry the
    DOUBLE prefix (base_model.model.base_model.model...) — i.e. the outer
    PeftModel nesting. We rebuild that exact structure and load weights via
    load_state_dict, checking that every saved lora tensor lands.

    Returns (model, tokenizer) — PeftModel (traj) nested over Fable5.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    log(f"loading base '{base}' (4-bit fp16 compute) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="cuda:0",
        trust_remote_code=True, torch_dtype=torch.float16,
    )
    log(f"attaching Fable5 adapter '{fable}' ...")
    model = PeftModel.from_pretrained(model, fable)

    # --- rebuild the OUTER trajectory LoRA exactly as training built it ---
    traj_cfg = json.load(open(os.path.join(traj, "adapter_config.json")))
    log(f"attaching trajectory LoRA (r={traj_cfg.get('r')}, "
        f"alpha={traj_cfg.get('lora_alpha')}, "
        f"{len(traj_cfg.get('target_modules', []))} target modules) ...")
    model = get_peft_model(model, LoraConfig(
        r=traj_cfg.get("r", 16),
        lora_alpha=traj_cfg.get("lora_alpha", 32),
        target_modules=list(traj_cfg.get("target_modules", [])),
        lora_dropout=traj_cfg.get("lora_dropout", 0.0),
        bias=traj_cfg.get("bias", "none"),
    ))
    log("loading trained trajectory LoRA weights ...")
    sd = torch.load(os.path.join(traj, "adapter_model.safetensors"),
                    map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_lora = [k for k in missing if "lora" in k]
    log(f"  saved={len(sd)} tensors | lora keys missing={len(missing_lora)} "
        f"| unexpected={len(unexpected)}")
    if missing_lora:
        log("  WARN: trained lora weights did NOT land — adapter is NOT applied!")
    else:
        log("  OK: every trained lora tensor landed (double-nested keys match).")
    model.eval()
    return model, tok


def gen(model, tok, prompt: str, max_new: int, temp: float = 0.7) -> str:
    import torch
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=True,
            temperature=temp, top_p=0.95, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def check_tool_format(model, tok, n: int, max_new: int) -> dict:
    import torch  # noqa
    valid_tools = {"list_dir", "read_file", "search_code", "run_test", "finish"}
    tasks = [
        "Read the file main.py and search for the bug in the parser.",
        "List the files in src/ and read the first one.",
        "Run the tests to see what's failing.",
        "Search for 'TODO' across the codebase, then read the relevant file.",
        "Fix a bug: start by exploring the repo structure.",
    ]
    results = {"total": 0, "any_call": 0, "valid_format": 0, "valid_tool": 0,
               "empty": 0, "unknown_tools": {}, "samples": []}
    for i in range(n):
        task = tasks[i % len(tasks)]
        out = gen(model, tok, task, max_new)
        calls = re.findall(r"```(\w+)\s*\n(.*?)```", out, re.DOTALL)
        calls = [(t, a.strip()[:60]) for t, a in calls]
        results["total"] += 1
        if not out.strip():
            results["empty"] += 1
        if calls:
            results["any_call"] += 1
            if all(c[0] in valid_tools for c in calls):
                results["valid_tool"] += 1
            else:
                for c in calls:
                    if c[0] not in valid_tools:
                        results["unknown_tools"][c[0]] = results["unknown_tools"].get(c[0], 0) + 1
        if len(calls) == 1 and calls[0][0] in valid_tools:
            results["valid_format"] += 1
        results["samples"].append({"task": task, "calls": [c[0] for c in calls],
                                   "out": out[:160]})
        log(f"  [{i+1}/{n}] calls={[c[0] for c in calls]}")
    return results


def check_self_awareness(model, tok, n: int, max_new: int) -> dict:
    results = {"total": 0, "state_sheet": 0, "self_review": 0, "samples": []}
    for i in range(n):
        out = gen(model, tok, TRACE_TASK, max_new, temp=0.9)
        results["total"] += 1
        has_state = bool(re.search(r"\[STATE\]", out)) or \
            bool(re.search(r"(?i)\bknown\b.*\btried\b.*\bfailed\b.*\bnext\b", out[:600]))
        has_review = bool(re.search(r"(?i)self-?review|self review|what (went|i'?d|i )", out))
        results["state_sheet"] += int(has_state)
        results["self_review"] += int(has_review)
        results["samples"].append({"state": has_state, "review": has_review, "out": out[:200]})
        log(f"  [{i+1}/{n}] state_sheet={has_state} self_review={has_review}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--fable", default="mateo0093/le-gros-chaton-qwen")
    ap.add_argument("--traj", default="mateo0093/le-gros-chaton-qwen-traj-sft")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--json-out", default="verify_results.json")
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load_stack(args.base, args.fable, args.traj)
    log(f"stack loaded in {time.time()-t0:.0f}s | "
        f"VRAM {torch.cuda.memory_allocated()/1e9:.2f} GiB")

    log("--- tool-call format ---")
    tf = check_tool_format(model, tok, args.n, args.max_new)
    log("--- self-awareness ---")
    sa = check_self_awareness(model, tok, max(1, args.n // 2), args.max_new)

    res = {"tool_format": tf, "self_awareness": sa,
           "stack": {"base": args.base, "fable": args.fable, "traj": args.traj}}
    with open(args.json_out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"saved -> {args.json_out}")
    log("=== TOOL FORMAT: {}% any-call, {}% valid-tool ({} prompts)".format(
        100 * tf["any_call"] / max(1, tf["total"]),
        100 * tf["valid_tool"] / max(1, tf["total"]), tf["total"]))
    log("=== SELF-AWARENESS: {}% state-sheet, {}% self-review ({} rollouts)".format(
        100 * sa["state_sheet"] / max(1, sa["total"]),
        100 * sa["self_review"] / max(1, sa["total"]), sa["total"]))


if __name__ == "__main__":
    import torch  # noqa: E402
    main()
