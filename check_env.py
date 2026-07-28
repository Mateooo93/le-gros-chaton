#!/usr/bin/env python3
"""Environment readiness check for Le Gros Chaton training.

Verifies that the system meets all requirements for training, evaluation,
and inference.  Reports pass/fail/skip for each check.

Usage:
    python check_env.py
"""
import os
import shutil
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ_ROOT)

PASS = 0
FAIL = 0
SKIP = 0


def check(desc: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        print(f"  ✓ {desc}")
        PASS += 1
    else:
        print(f"  ✗ {desc}  {detail}")
        FAIL += 1


def skip(desc: str, reason: str = ""):
    global SKIP
    SKIP += 1
    r = f" ({reason})" if reason else ""
    print(f"  - {desc}{r}")


def heading(s: str):
    print(f"\n{'='*60}")
    print(f"  {s}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------
heading("1. Python")

check("Python >= 3.11", sys.version_info >= (3, 11), f"got {sys.version_info.major}.{sys.version_info.minor}")
check("pip available", shutil.which("pip") is not None)
check("make available", shutil.which("make") is not None)
check("git available", shutil.which("git") is not None)

# ---------------------------------------------------------------------------
# 2. PyTorch + CUDA
# ---------------------------------------------------------------------------
heading("2. PyTorch & CUDA")

try:
    import torch
    check("torch import", True)
    check("torch version", torch.__version__ >= "2.1", f"got {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    check("CUDA available", cuda_ok)
    if cuda_ok:
        n_gpu = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        check(f"GPU: {name} ({mem:.1f} GB, {n_gpu} device(s))", True)
    else:
        skip("GPU (running on CPU — training will be extremely slow)")
except ImportError:
    check("torch import", False, "not installed — run: pip install torch>=2.1")
except OSError as e:
    check(f"torch: CUDA library issue ({e}) ", False,
          "torch is installed but CUDA libraries are missing. "
          "Try: pip uninstall torch && pip install torch>=2.1 --index-url https://download.pytorch.org/whl/cu118")
    skip("CUDA check (torch not available)")
except Exception as e:
    check(f"torch: {e}", False)

# ---------------------------------------------------------------------------
# 3. Dependencies
# ---------------------------------------------------------------------------
heading("3. Python dependencies")

deps = [
    ("numpy", "numpy>=1.24"),
    ("tiktoken", "tiktoken>=0.5"),
    ("datasets", "datasets>=2.14"),
    ("tokenizers", "tokenizers>=0.15"),
    ("huggingface_hub", "huggingface_hub>=0.19"),
]
for name, spec in deps:
    try:
        __import__(name)
        check(f"{name}", True)
    except ImportError:
        check(f"{name}", False, f"not installed — run: pip install {spec}")

dev_deps = [
    ("pytest", "pytest>=7.0"),
    ("ruff", "ruff>=0.4"),
    ("black", "black>=24.0"),
]
for name, spec in dev_deps:
    try:
        __import__(name)
        check(f"{name}", True)
    except ImportError:
        skip(f"{name}", f"optional — install with: pip install {spec}")

# ---------------------------------------------------------------------------
# 4. Tokenizer
# ---------------------------------------------------------------------------
heading("4. Tokenizer")

try:
    from tokenizer import VOCAB_SIZE, encode, decode, tool_token_id
    check(f"tokenizer loaded (vocab={VOCAB_SIZE})", True)
    ids = encode("Hello, world!")
    check(f"encode round-trip: '{ids}' → '{decode(ids)}'", decode(ids) == "Hello, world!")
    tid = tool_token_id("<|tool_call|>")
    check(f"tool token <|tool_call|> = {tid}", tid is not None)
except Exception as e:
    check(f"tokenizer: {e}", False)

# ---------------------------------------------------------------------------
# 5. Secrets (HF token)
# ---------------------------------------------------------------------------
heading("5. Secrets")

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token and os.path.exists(os.path.join(PROJ_ROOT, "gpus.md")):
    with open(os.path.join(PROJ_ROOT, "gpus.md")) as f:
        for line in f:
            if "=" in line and ("HF_TOKEN" in line or "hf_token" in line.lower()):
                hf_token = line.split("=", 1)[1].strip()
                break
if hf_token and hf_token.startswith("hf_"):
    check("HF_TOKEN set", True)
    skip("HF_TOKEN value (not printed for security)")
else:
    skip("HF_TOKEN", "not found — checkpoint push/pull will fail without it")

hf_repo = os.environ.get("CHATON_HF_REPO", "")
if hf_repo:
    check("CHATON_HF_REPO set", True)
else:
    skip("CHATON_HF_REPO", "not set — checkpoint Hub sync disabled")

# ---------------------------------------------------------------------------
# 6. Modal (optional)
# ---------------------------------------------------------------------------
heading("6. Modal cloud (optional)")

try:
    import modal
    check("modal import", True)
    skip("Modal config", "verified at launch time")
except ImportError:
    skip("modal", "not installed — run: pip install modal")

# ---------------------------------------------------------------------------
# 7. Disk space
# ---------------------------------------------------------------------------
heading("7. Disk space")

try:
    import shutil
    total, used, free = shutil.disk_usage(PROJ_ROOT)
    free_gb = free / (2**30)
    check(f"free disk space: {free_gb:.1f} GB", free_gb >= 10,
          f"need at least 10 GB for datasets and checkpoints, have {free_gb:.1f}")
except Exception:
    skip("disk space check (shutil not available)")

# ---------------------------------------------------------------------------
# 8. Config validation
# ---------------------------------------------------------------------------
heading("8. Configuration")

try:
    os.environ["CHATON_PROFILE"] = os.environ.get("CHATON_PROFILE", "dev")
    import config as cfg
    issues = cfg.validate()
    if issues:
        for i in issues:
            check(f"config: {i}", False)
    else:
        check("config validation", True, "no issues detected")
    check(f"profile: {cfg.PROFILE}", True)
    check(f"architecture: {cfg.n_layer}L, {cfg.n_embd}D, {cfg.n_head}H", True)
except Exception as e:
    check(f"config: {e}", False)

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
heading("SUMMARY")

total = PASS + FAIL + SKIP
print(f"  {PASS} passed, {FAIL} failed, {SKIP} skipped ({total} total)")

if FAIL > 0:
    print("\n  ✗ Some checks failed. Fix the reported issues.")
    sys.exit(1)
elif PASS == total:
    print("\n  ✓ All checks passed! Ready for training.")
    print("\n  Next steps:")
    print("    python go.py              # Smoke test")
    print("    CHATON_PROFILE=dev python train.py  # Quick training test")
    print("    CHATON_PROFILE=smol-fat python train.py  # Real training")
else:
    print("\n  ✓ All required checks passed.")
    print("\n  Next steps:")
    print("    python go.py              # Smoke test")
    print("    CHATON_PROFILE=dev python train.py  # Quick training test")
