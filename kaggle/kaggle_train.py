# Kaggle training script — Qwen3.5-9B SFT on Fable5
# Pushed and run via the Kaggle API from run_kaggle.py
import os
import subprocess
import sys

import traceback

LOG = "/tmp/train_log.txt"

# Reduce CUDA fragmentation (helps on 16GB T4s, esp. with grad accumulation)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(str(msg) + "\n")


def sft_adapter_args() -> list[str]:
    """Return ['--adapter', repo] so SFT continues from the last trained adapter.

    The base repo `{who}/le-gros-chaton-qwen` holds the last completed SFT
    adapter (10k rows so far, then 16k once this run saves its final model).
    """
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        return []
    try:
        from huggingface_hub import HfApi
        who = HfApi(token=tok).whoami()["name"]
        return ["--adapter", f"{who}/le-gros-chaton-qwen"]
    except Exception as e:
        print(f"[warn] cannot derive HF repo for adapter: {e}")
        return []


def sft_resume_args() -> list[str]:
    """Return ['--resume-sft', repo] if HF_TOKEN is set, else [].

    Points train_qwen.py at the private Hub repo that train_sft uploads its
    every-20% checkpoints to, so a re-run after a Kaggle disconnect resumes
    from the latest checkpoint instead of starting over.
    """
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        return []
    try:
        from huggingface_hub import HfApi
        who = HfApi(token=tok).whoami()["name"]
        return ["--resume-sft", f"{who}/le-gros-chaton-qwen-sft-ckpt"]
    except Exception as e:
        print(f"[warn] cannot derive HF repo for resume: {e}")
        return []


def rlvr_args() -> list[str]:
    """Build the Phase 2 (RLVR) command args.

    Starts from the Phase 1 SFT adapter on the Hub, optionally resumes RLVR
    checkpoints, and uploads RLVR checkpoints to a separate Hub repo.
    """
    tok = os.environ.get("HF_TOKEN", "")
    steps = os.environ.get("RLVR_STEPS", "100")
    group = os.environ.get("RLVR_GROUP", "4")
    save_every = os.environ.get("RLVR_SAVE_EVERY", "25")
    args = ["--rlvr-only", "--rlvr-steps", steps, "--group-size", group,
            "--rlvr-save-every", save_every]
    if tok:
        try:
            from huggingface_hub import HfApi
            who = HfApi(token=tok).whoami()["name"]
            args += ["--adapter", f"{who}/le-gros-chaton-qwen",
                     "--resume-rlvr", f"{who}/le-gros-chaton-qwen-rlvr-ckpt"]
        except Exception as e:
            print(f"[warn] cannot derive HF repo for RLVR: {e}")
    return args


try:
    # 1. Install deps (Qwen3.5 needs transformers >= 5.14.1 for 'qwen3_5' support)
    log("=== Installing deps ===")
    # Keep Kaggle's preinstalled torch (2.10+cu128) — it supports sm_75 (T4)
    # and ships torch.distributed.tensor.DTensor, which transformers 5.x imports
    # at startup. The old torch==2.4.1+cu118 pin was only needed for the P100
    # (sm_60) and breaks with ImportError: cannot import name 'DTensor'.
    # transformers 5.14.1 requires torch>=2.4 (satisfied) and tokenizers in
    # [0.22.0,0.23.0]; 0.22.0 needs huggingface-hub<1.0 which conflicts with
    # transformers' huggingface-hub>=1.5, so use 0.22.1 (<2.0).
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "transformers==5.14.1", "tokenizers==0.22.1", "safetensors",
        "accelerate", "peft", "bitsandbytes", "trl", "datasets", "tiktoken"])
    log("deps installed (preinstalled torch + transformers 5.14.1 + tokenizers 0.22.1)")

    # 2. Clone the repo
    log("=== Cloning repo ===")
    if not os.path.exists("le-gros-chaton"):
        subprocess.check_call(["git", "clone",
            "https://github.com/Mateooo93/le-gros-chaton.git"])
    os.chdir("le-gros-chaton")
    log("repo cloned")

    # 3. Set HF token (run_kaggle.py injects the literal here at push time)
    hf_token = os.environ.get("HF_TOKEN", "")
    log(f"HF_TOKEN set: {bool(hf_token)}")

    # 4. Run training — PHASE 1 (SFT) by default, PHASE 2 (RLVR) when
    #    TRAIN_PHASE=rlvr. Stream stdout/stderr live to the Kaggle cell AND
    #    tee it to /kaggle/working/train_log.txt so we can `kaggle kernels
    #    output` mid-run to see progress.
    limit = os.environ.get("TRAIN_LIMIT", "10000")
    phase = os.environ.get("TRAIN_PHASE", "sft").strip().lower()
    log(f"=== Training (phase={phase}, limit={limit}) ===")
    out_dir = "/kaggle/working"
    os.makedirs(out_dir, exist_ok=True)
    live_log = os.path.join(out_dir, "train_log.txt")
    # Seed the live log with everything logged so far, then stream.
    try:
        import shutil as _sh
        _sh.copy(LOG, live_log)
    except Exception:
        pass

    if phase == "rlvr":
        cmd = [sys.executable, "-u", "train_qwen.py"] + rlvr_args()
    else:
        # SFT: continue from the trained SFT adapter (mateo0093/le-gros-chaton-qwen)
        # rather than starting from a random LoRA. --resume-sft is NOT used:
        # the SFT ckpt repo's checkpoint-* layout only exists for Kaggle
        # disconnect resume within the same run, and its trainer step counter
        # would not match this run's data size.
        cmd = [sys.executable, "-u", "train_qwen.py",
               "--sft-only", "--limit", limit, "--batch-size",
               os.environ.get("SFT_BATCH", "1")] + sft_adapter_args() + sft_resume_args()
    log("CMD: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    tail = []
    with open(live_log, "a") as lf:
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            lf.write(line + "\n"); lf.flush()
            tail.append(line)
            if len(tail) > 400:
                tail.pop(0)
    rc = proc.wait()
    log("=== train_qwen.py output tail ===")
    log("\n".join(tail))
    log(f"TRAIN_RC={rc}")
    if rc != 0:
        raise RuntimeError(f"train failed rc={rc}")

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
