# Kaggle training script — Qwen3.5-9B SFT on Fable5
# Pushed and run via the Kaggle API from run_kaggle.py
import os
import subprocess
import sys

# 1. Install deps (Qwen3.5 needs latest transformers)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/huggingface/transformers.git"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "accelerate", "peft", "bitsandbytes", "trl", "datasets", "tiktoken", "tokenizers"])

# 2. Clone the repo
if not os.path.exists("le-gros-chaton"):
    subprocess.check_call(["git", "clone",
        "https://github.com/Mateooo93/le-gros-chaton.git"])
os.chdir("le-gros-chaton")

# 3. Set HF token from Kaggle secret
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# 4. Run training (Phase 1: SFT)
#   limit from env, default 10000 (fits in 9hr session)
limit = os.environ.get("TRAIN_LIMIT", "10000")
subprocess.check_call([sys.executable, "train_qwen.py",
    "--sft-only", "--limit", limit])

print("=== TRAINING COMPLETE ===")
