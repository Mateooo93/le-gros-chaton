"""Modal launch script for Qwen3.5-9B + Fable5 training.

Continues from the Kaggle-trained SFT adapter (mateo0093/le-gros-chaton-qwen),
so Kaggle GPU time is never wasted — Modal finishes the remaining dataset rows.

Budget estimate (L4 24GB @ $0.80/hr, 9B QLoRA):
  python modal_train_qwen.py --full           # finish 160K rows  (~$5-8)
  python modal_train_qwen.py --full --rlvr    # + RLVR            (~$8-12)

Usage:
  # Continue SFT from Kaggle's adapter: Kaggle trained rows [0, 40000),
  # Modal trains [40000, 160000) and resumes the optimizer from the last
  # Kaggle checkpoint.
  python modal_train_qwen.py --full --sft-start 40000 \
      --resume-sft mateo0093/le-gros-chaton-qwen-sft-ckpt

  # Everything including RLVR:
  python modal_train_qwen.py --full --sft-start 40000 --rlvr \
      --resume-sft mateo0093/le-gros-chaton-qwen-sft-ckpt
"""
import os
import argparse
import modal

app = modal.App("le-gros-chaton-qwen")

# transformers 5.14.1 is required for the 'qwen3_5' hybrid-attention VLM
# (older releases throw KeyError('qwen3_5')). tokenizers 0.22.1 satisfies
# transformers' [0.22.0, 0.23.0] range without the huggingface-hub<1.0
# conflict that 0.22.0 has.
image = (
    # nvidia/cuda devel base provides nvcc (needed to build causal-conv1d
    # from source — debian_slim has no CUDA toolkit). CUDA 12.4 is
    # binary-compatible with torch 2.10+cu128 at runtime.
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch==2.10.0",
        "transformers==5.14.1",
        "tokenizers==0.22.1",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "datasets",
        "tiktoken",
        "safetensors",
    )
    # Fast kernels for the hybrid linear-attention layers (24/32 layers are
    # linear-attention). Without these, transformers falls back to a slow torch
    # path (the log says "The fast path is not available...").
    .run_commands(
        "pip install wheel setuptools ninja && pip install causal-conv1d --no-build-isolation || true",
        "pip install 'flash-linear-attention[cuda]' --no-build-isolation || true",
    )
    .add_local_dir(".", "/root/proj", ignore=["__pycache__", ".git", "*.pt", "*.bin"])
)

HF_SECRET = os.environ.get("CHATON_HF_SECRET", "chaton-hf")


@app.function(
    gpu="L4",  # $0.80/hr — best value for 9B QLoRA; use A100 for RLVR speed
    image=image,
    timeout=86400,
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def train(limit: int | None = None, sft_start: int | None = None,
          sft_only: bool = True, resume_sft: str = "mateo0093/le-gros-chaton-qwen-sft-ckpt",
          adapter: str | None = None, model_name: str = "Qwen/Qwen3.5-9B",
          trajectory_sft: bool = False, eff_batch: int = 16):
    os.chdir("/root/proj")
    # Reduce CUDA fragmentation (helped avoid OOMs on T4/L4 in testing)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Resolve adapter + sft_start (same logic as the local main()): if adapter
    # is None, pull the newest checkpoint-* from the resume repo and derive
    # sft_start = step * eff_batch (rows already trained).
    token = os.environ.get("HF_TOKEN", "")
    if adapter is None:
        try:
            from huggingface_hub import snapshot_download, HfApi
            api = HfApi(token=token)
            files = api.list_repo_files(resume_sft)
            ck_dirs = sorted({f.split("/")[0] for f in files
                              if f.split("/")[0].startswith("checkpoint-")},
                             key=lambda d: int(d.split("-")[1]))
            if ck_dirs:
                latest = ck_dirs[-1]
                print(f"[modal] Resuming from {resume_sft}/{latest}")
                local = snapshot_download(
                    repo_id=resume_sft, token=token,
                    allow_patterns=[f"{latest}/**"],
                    ignore_patterns=["*.bin", "optimizer.pt"],
                )
                nested = os.path.join(local, latest, "sft")
                adapter = nested if os.path.isdir(nested) else os.path.join(local, latest)
                step = int(latest.split("-")[1])
                if sft_start is None:
                    sft_start = step * eff_batch
                    print(f"[modal] Derived sft_start from checkpoint step: "
                          f"{step} x {eff_batch} = {sft_start} rows")
            else:
                adapter = "mateo0093/le-gros-chaton-qwen"
                if sft_start is None:
                    sft_start = 0
        except Exception as e:
            adapter = "mateo0093/le-gros-chaton-qwen"
            if sft_start is None:
                sft_start = 0
            print(f"[modal] Could not resolve adapter ({e}) — using base adapter")
    if sft_start is None:
        sft_start = 0
    print(f"[modal] FINAL: adapter={adapter} sft_start={sft_start} "
          f"limit={limit} sft_only={sft_only} trajectory={trajectory_sft}")

    # train_qwen.py reads HF_TOKEN from env for checkpoint pull/upload.
    import subprocess, sys
    cmd = [sys.executable, "-u", "train_qwen.py",
           "--model", model_name,
           "--adapter", adapter,          # start from Kaggle's SFT adapter
           "--sft-start", str(sft_start),  # skip rows already trained on Kaggle
           "--limit", str(limit) if limit else "160000",
           "--sft-epochs", "1",
           "--max-length", "512",
           "--batch-size", "1"]  # eff batch 8 (grad-accum 8); batch 4 @ 1024 OOMs L4
    if trajectory_sft:
        # Long-context trajectory SFT: batch 1 (grad-accum 8) to fit 16K+ ctx;
        # raise --trajectory-ctx toward 256K as GPU allows.
        cmd += ["--trajectory-sft", "--trajectory-ctx", "16384"]
    if resume_sft:
        cmd += ["--resume-sft", resume_sft]
    if not sft_only:
        cmd += ["--rlvr-steps", "200", "--group-size", "4",
                "--rlvr-save-every", "25"]
    else:
        cmd += ["--sft-only"]
    print("[modal] CMD:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        raise RuntimeError(f"train_qwen.py failed rc={r.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Full 160K dataset (default: 10K)")
    parser.add_argument("--sft-start", type=int, default=None,
                        help="Row offset — rows [0, start) already trained. "
                             "If omitted, auto-read from the adapter's "
                             "sft_progress.json (written by the Kaggle run)")
    parser.add_argument("--rlvr", action="store_true", help="Include RLVR phase")
    parser.add_argument("--trajectory-sft", action="store_true",
                        help="SFT on agent trajectories (assistant-token loss, "
                             "needs agent_traces_full.jsonl in the repo)")
    parser.add_argument("--resume-sft", default="mateo0093/le-gros-chaton-qwen-sft-ckpt",
                        help="HF repo with SFT checkpoints to resume from")
    parser.add_argument("--adapter", default=None,
                        help="Pretrained LoRA adapter to start from (local dir or HF "
                             "repo id). If omitted, auto-resolves to the newest "
                             "checkpoint-* in --resume-sft (nested sft/ adapter).")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--eff-batch", type=int, default=16,
                        help="Effective batch used by the resumed run (batch x grad_accum "
                             "x GPUs). Used to derive sft_start from the checkpoint step.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")

    # Auto-detect continuation state:
    # 1. newest checkpoint-* subdir in --resume-sft (or the sft/ adapter nested
    #    inside it) is the starting adapter
    # 2. sft_start = checkpoint_step * eff_batch (rows already trained) unless
    #    explicitly overridden or sft_progress.json says otherwise
    if args.adapter is None:
        try:
            from huggingface_hub import snapshot_download, HfApi
            api = HfApi(token=token)
            files = api.list_repo_files(args.resume_sft)
            import re as _re
            ck_dirs = sorted({f.split("/")[0] for f in files
                              if f.split("/")[0].startswith("checkpoint-")},
                             key=lambda d: int(d.split("-")[1]))
            if ck_dirs:
                latest = ck_dirs[-1]
                print(f"[modal] Resuming from {args.resume_sft}/{latest}")
                local = snapshot_download(
                    repo_id=args.resume_sft, token=token,
                    allow_patterns=[f"{latest}/**"],
                    ignore_patterns=["*.bin", "optimizer.pt"],
                )
                # PEFT multi-adapter layout: checkpoint-800/sft/adapter_config.json
                nested = os.path.join(local, latest, "sft")
                args.adapter = nested if os.path.isdir(nested) else os.path.join(local, latest)
                step = int(latest.split("-")[1])
                if args.sft_start is None:
                    args.sft_start = step * args.eff_batch
                    print(f"[modal] Derived sft_start from checkpoint step: "
                          f"{step} x {args.eff_batch} = {args.sft_start} rows")
            else:
                # No checkpoints: fall back to the base adapter repo (flat layout)
                args.adapter = "mateo0093/le-gros-chaton-qwen"
                if args.sft_start is None:
                    args.sft_start = 0
                    print("[modal] No checkpoints found — using base adapter, start=0")
        except Exception as e:
            args.adapter = "mateo0093/le-gros-chaton-qwen"
            if args.sft_start is None:
                args.sft_start = 0
            print(f"[modal] Could not resolve adapter ({e}) — using base adapter")

    print(f"[modal] FINAL: adapter={args.adapter} sft_start={args.sft_start}")

    limit = None if args.full else 10000
    train.remote(
        limit=limit,
        sft_start=args.sft_start,
        sft_only=not args.rlvr,
        resume_sft=args.resume_sft,
        adapter=args.adapter,
        model_name=args.model,
        trajectory_sft=args.trajectory_sft,
        eff_batch=args.eff_batch,
    )


if __name__ == "__main__":
    main()
