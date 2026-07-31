# Kaggle training script — Qwen3.5-9B SFT on Fable5
# Pushed and run via the Kaggle API from run_kaggle.py
import os
import subprocess
import sys

import traceback

LOG = "/tmp/train_log.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")

try:
    # 1. Install deps (Qwen3.5 needs latest transformers)
    log("=== Installing deps ===")
    # IMPORTANT: Kaggle sometimes assigns P100 (sm_60).  Modern torch wheels
    # dropped sm_60, so pin CUDA 11.8 build which still supports it.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "torch==2.4.1+cu118", "--index-url", "https://download.pytorch.org/whl/cu118"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/huggingface/transformers.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "accelerate", "peft", "bitsandbytes", "trl", "datasets", "tiktoken", "tokenizers"])
    log("deps installed (torch cu118 for P100 compat)")

    # 2. Clone the repo
    log("=== Cloning repo ===")
    if not os.path.exists("le-gros-chaton"):
        subprocess.check_call(["git", "clone",
            "https://github.com/Mateooo93/le-gros-chaton.git"])
    os.chdir("le-gros-chaton")
    log("repo cloned")

    # 3. Set HF token from env
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    log(f"HF_TOKEN set: {bool(hf_token)}")

    # 4. Run training (Phase 1: SFT)
    limit = os.environ.get("TRAIN_LIMIT", "10000")
    log(f"=== Training (limit={limit}) ===")
    r = subprocess.run([sys.executable, "train_qwen.py",
        "--sft-only", "--limit", limit],
        capture_output=True, text=True, timeout=8*3600)
    log("STDOUT:\n" + r.stdout[-3000:])
    log("STDERR:\n" + r.stderr[-3000:])
    log(f"TRAIN_RC={r.returncode}")
    if r.returncode != 0:
        raise RuntimeError(f"train failed rc={r.returncode}")

    log("=== TRAINING COMPLETE ===")

except Exception as e:
    log("EXCEPTION: " + repr(e))
    log(traceback.format_exc())
    # Copy log to output dir so we can download it
    os.makedirs("/kaggle/working", exist_ok=True)
    try:
        import shutil
        shutil.copy(LOG, "/kaggle/working/train_log.txt")
    except Exception:
        pass
    raise

# Copy log to output for download
try:
    import shutil
    os.makedirs("/kaggle/working", exist_ok=True)
    shutil.copy(LOG, "/kaggle/working/train_log.txt")
except Exception:
    pass
