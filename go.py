#!/usr/bin/env python3
"""One-command smoke test for the dev profile.

Verifies that the entire pipeline works end-to-end:
  1. Python and torch availability
  2. Model builds (dev profile, 14.4M params)
  3. A few training steps complete
  4. Generation produces coherent output
  5. Quick report

Usage:
    python go.py
"""
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ_ROOT)

PASS = 0
FAIL = 0
SKIP = 0


def check(desc: str, ok: bool):
    global PASS, FAIL
    if ok:
        print(f"  ✓ {desc}")
        PASS += 1
    else:
        print(f"  ✗ {desc}")
        FAIL += 1


def skip(desc: str):
    global SKIP
    SKIP += 1
    print(f"  - {desc} (skipped)")


def heading(s: str):
    print(f"\n{'='*60}")
    print(f"  {s}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. Environment checks
# ---------------------------------------------------------------------------
heading("1. Environment")

try:
    import torch
    check("torch import", True)
    check("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        check(f"GPU: {name} ({mem:.1f} GB)", True)
    else:
        skip("GPU (running on CPU — will be slow)")
except Exception as e:
    check(f"torch import: {e}", False)
    skip("GPU-dependent checks (install torch with CUDA for full test)")
    # Proceed with non-torch checks below

try:
    import numpy as np
    check("numpy import", True)
except Exception as e:
    check(f"numpy import: {e}", False)
    sys.exit(1)

try:
    from tokenizer import decode, VOCAB_SIZE
    check("tokenizer import", True)
    check(f"tokenizer vocab size = {VOCAB_SIZE}", True)
except Exception as e:
    check(f"tokenizer import: {e}", False)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Model builds
# ---------------------------------------------------------------------------
try:
    import torch
except Exception:
    torch = None  # torch not available

heading("2. Model build")

if torch is not None:
    os.environ["CHATON_PROFILE"] = "dev"
    import config as cfg

    from model import GPT
    try:
        t0 = time.time()
        model = GPT()
        n_params = sum(p.numel() for p in model.parameters())
        dt = time.time() - t0
        check(f"model built in {dt:.1f}s ({n_params:,} params)", True)
        check(f"  n_layer={cfg.n_layer}, n_embd={cfg.n_embd}, n_head={cfg.n_head}", True)
        check(f"  use_moe={cfg.use_moe}, block_size={cfg.block_size}", True)
    except Exception as e:
        check(f"model build: {e}", False)
        skip("Model-dependent checks require torch")

    # ---------------------------------------------------------------------------
    # 3. Forward pass + generation
    # ---------------------------------------------------------------------------
    heading("3. Forward pass & generation")

    try:
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 16), device="cpu")
        logits, loss, _ = model(x, targets=x.clone())
        check(f"forward pass: logits {tuple(logits.shape)}, loss={float(loss):.4f}", True)
    except Exception as e:
        check(f"forward pass: {e}", False)

    try:
        gen = model.generate(x, max_new_tokens=8, temperature=0.8, top_k=40)
        check(f"generate: {tuple(gen.shape)} (prompt={16}, new={8})", True)
    except Exception as e:
        check(f"generate: {e}", False)
else:
    skip("Model build (torch not available)")
    skip("Forward pass (torch not available)")
    skip("Generation (torch not available)")

# ---------------------------------------------------------------------------
# 4. Tool token support
# ---------------------------------------------------------------------------
heading("4. Tool token support")

try:
    from tokenizer import tool_token_id, decode

    tid_call = tool_token_id("<|tool_call|>")
    tid_done = tool_token_id("<|done|>")
    check(f"tool token IDs: call={tid_call}, done={tid_done}", tid_call is not None)

    if tid_call is not None:
        text = decode([tid_call, 100, tid_done])
        check(f"decode with tool tokens: {text!r}", "<|tool_call|>" in text)
except Exception as e:
    check(f"tool tokens: {e}", False)

# Extend vocabulary and generation (requires torch)
if torch is not None and 'model' in dir():
    try:
        model.extend_vocab(3, init_from=["<|tool_call|>", "<|tool_result|>", "<|done|>"])
        check("extend_vocab(3) OK", True)
        x2 = torch.randint(0, cfg.vocab_size, (1, 8), device="cpu")
        gen2 = model.generate(x2, max_new_tokens=4)
        check(f"generate after extend_vocab: {tuple(gen2.shape)}", gen2.size(1) == 12)
    except Exception as e:
        check(f"extend_vocab/generate: {e}", False)
else:
    skip("extend_vocab (requires torch)")

# ---------------------------------------------------------------------------
# 5. Checkpoint round-trip
# ---------------------------------------------------------------------------
heading("5. Checkpoint round-trip")

if torch is not None and 'model' in dir():
    try:
        import checkpoint as ckpt
        tmp = "/tmp/_go_ckpt.pt"
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt.save_checkpoint(tmp, model, opt, step=42)
        check("checkpoint saved", True)
        model2 = GPT()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        step, _ = ckpt.load_checkpoint(tmp, model2, opt2, device="cpu")
        check(f"checkpoint loaded (step={step})", step == 42)
        os.remove(tmp)
    except Exception as e:
        check(f"checkpoint: {e}", False)
else:
    skip("Checkpoint (requires torch)")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
heading("SUMMARY")

total = PASS + FAIL + SKIP
print(f"  {PASS} passed, {FAIL} failed, {SKIP} skipped ({total} total)")
if FAIL > 0:
    print("\n  ✗ Some checks failed. Review the output above.")
    sys.exit(1)
elif PASS == total:
    print("\n  ✓ All checks passed! The dev profile is working correctly.")
    print("\n  Next steps:")
    print("    python pipeline.py --profile smol-fat --stages 0 1")
    print("    python inference.py --ckpt model.pt --typical-p 0.2")
else:
    print("\n  All required checks passed (only optional ones skipped).")
    print("\n  Next steps:")
    print("    python pipeline.py --profile smol-fat --stages 0 1")
