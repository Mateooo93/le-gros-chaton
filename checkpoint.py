"""Resumable checkpoints + HuggingFace Hub push/pull for VM-hopping.

The problem: free-tier VMs (Colab/Kaggle/Modal) die after hours. To pretrain
"le fat chaton" across many short VM sessions, a checkpoint must save MORE than
just weights — it must save enough that a fresh VM can continue training
identically:
  - model state_dict (weights)
  - optimizer state_dict (AdamW m/v per param — without these, momentum resets)
  - the current step (so the LR schedule and iter count continue, not restart)
  - the scheduler/optimizer settings (lr_max, warmup, etc.)
  - a snapshot of config + tokenizer so a fresh VM rebuilds the right architecture

save_checkpoint / load_checkpoint handle the local file. push_hub / pull_hub
ship it to/from the HuggingFace Hub (free, git-based, survives any VM dying).

Env:
  HF_TOKEN   -> your huggingface token (set in the VM's env / secrets)
  CHATON_HF_REPO -> e.g. "yourname/le-fat-chaton-ckpt" (create the repo first on HF)

Typical VM-hop loop:
  train.py runs a chunk  ->  checkpoint.save_checkpoint(...)  ->  checkpoint.push_hub(...)
  [new VM]
  checkpoint.pull_hub(...)  ->  step, optim, model = checkpoint.load_checkpoint(...)
  train.py resumes from that step
"""
import os
import json
import torch
import config as cfg

CKPT_PATH = os.environ.get("CHATON_CKPT_PATH", "checkpoint.pt")
HF_REPO = os.environ.get("CHATON_HF_REPO", "")      # e.g. "mateo/le-fat-chaton-ckpt"
HF_FILENAME = "checkpoint.pt"


def save_checkpoint(path, model, optimizer, step, scaler=None, extra=None):
    """Write a resumable checkpoint to `path` (local disk)."""
    payload = {
        "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")},
        "extra": extra or {},
    }
    torch.save(payload, path)
    print(f"[ckpt] saved step {step} -> {path}")


def load_checkpoint(path, model, optimizer, scaler=None, device="cuda"):
    """Restore model + optimizer + step (+ scaler) from a checkpoint.
    Returns the step to resume from. Model/optimizer must already be built."""
    payload = torch.load(path, map_location=device, weights_only=False)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    step = payload["step"]
    print(f"[ckpt] restored step {step} from {path}")
    return int(step)


# --- HuggingFace Hub sync (optional; only if HF_REPO + huggingface_hub installed) ---
def _hub():
    if not HF_REPO:
        raise RuntimeError("set CHATON_HF_REPO env var (e.g. 'you/le-fat-chaton-ckpt')")
    from huggingface_hub import HfApi
    return HfApi()


def push_hub(path=None):
    """Upload the checkpoint to the HF Hub (overwrites latest)."""
    path = path or CKPT_PATH
    api = _hub()
    api.upload_file(path_or_fileobj=path, path_in_repo=HF_FILENAME, repo_id=HF_REPO,
                    token=os.environ.get("HF_TOKEN"))
    print(f"[ckpt] pushed {path} -> {HF_REPO}:{HF_FILENAME}")


def pull_hub(path=None):
    """Download the latest checkpoint from the HF Hub to local disk."""
    path = path or CKPT_PATH
    api = _hub()
    # download into the SAME directory as the target path so the move below works
    api.hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME,
                        local_dir=os.path.dirname(os.path.abspath(path)) or ".",
                        token=os.environ.get("HF_TOKEN"))
    # hf_hub_download writes to <local_dir>/<HF_FILENAME>; move to our path if needed
    dl = os.path.join(os.path.dirname(os.path.abspath(path)) or ".", HF_FILENAME)
    if dl != path and os.path.exists(dl):
        os.replace(dl, path)
    print(f"[ckpt] pulled {HF_REPO}:{HF_FILENAME} -> {path}")
    return path


if __name__ == "__main__":
    # quick self-test: save a dummy ckpt, reload the step, confirm round-trip
    from model import GPT
    m = GPT().to("cpu")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    save_checkpoint("/tmp/_test_ckpt.pt", m, opt, step=1234)
    step = load_checkpoint("/tmp/_test_ckpt.pt", m, opt, device="cpu")
    print("round-trip step:", step, "(expect 1234)")
    os.remove("/tmp/_test_ckpt.pt")