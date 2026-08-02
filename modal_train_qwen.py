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
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1+cu118",
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
    .add_local_dir(".", "/root/proj", ignore=["__pycache__", ".git", "*.pt", "*.bin"])
)

HF_SECRET = os.environ.get("CHATON_HF_SECRET", "chaton-hf")


@app.function(
    gpu="L4",  # $0.80/hr — best value for 9B QLoRA; use A100 for RLVR speed
    image=image,
    timeout=86400,
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def train(limit: int | None, sft_start: int, sft_only: bool,
          resume_sft: str, adapter: str, model_name: str):
    os.chdir("/root/proj")

    # train_qwen.py reads HF_TOKEN from env for checkpoint pull/upload.
    import subprocess, sys
    cmd = [sys.executable, "-u", "train_qwen.py",
           "--model", model_name,
           "--adapter", adapter,          # start from Kaggle's SFT adapter
           "--sft-start", str(sft_start),  # skip rows already trained on Kaggle
           "--limit", str(limit) if limit else "160000",
           "--batch-size", "4",           # fits L4 24GB
           "--sft-epochs", "1",
           "--max-length", "1024"]
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
    parser.add_argument("--sft-start", type=int, default=0,
                        help="Row offset — rows [0, start) already trained on Kaggle")
    parser.add_argument("--rlvr", action="store_true", help="Include RLVR phase")
    parser.add_argument("--resume-sft", default="mateo0093/le-gros-chaton-qwen-sft-ckpt",
                        help="HF repo with SFT checkpoints to resume from")
    parser.add_argument("--adapter", default="mateo0093/le-gros-chaton-qwen",
                        help="Pretrained LoRA adapter to start from")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    limit = None if args.full else 10000
    train.remote(
        limit=limit,
        sft_start=args.sft_start,
        sft_only=not args.rlvr,
        resume_sft=args.resume_sft,
        adapter=args.adapter,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
