"""Modal launch script for Qwen3.5-9B + Fable5 training.

Uses your $30 Modal budget efficiently:

  python modal_train_qwen.py                                        # SFT (default)
  python modal_train_qwen.py --full                                 # Full 160K rows (~$10)
  python modal_train_qwen.py --rlvr                                 # SFT + RLVR (~$15)
  python modal_train_qwen.py --model Qwen/Qwen3.5-32B --full        # 32B on A100 ($20)
"""
import os
import modal

app = modal.App("le-gros-chaton-qwen")

# Build image with all deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.45",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "datasets",
        "tiktoken",
        "tokenizers",
    )
    .add_local_dir(".", "/root/proj", ignore=["__pycache__", ".git", "*.pt", "*.bin"])
)

# Modal secret for HF token
app.secret = modal.Secret.from_name("chaton-hf")

@app.function(
    gpu="L4",  # L4 is $0.80/hr — best value for 9B QLoRA
    image=image,
    timeout=86400,
    secrets=[modal.Secret.from_name("chaton-hf")],
)
def train(limit: int = None, sft_only: bool = True, model_name: str = "Qwen/Qwen3.5-9B"):
    os.chdir("/root/proj")
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    os.environ["CHATON_HF_REPO"] = "mateo0093/le-fat-chaton-ckpt"

    from train_qwen import load_model, load_fable5_dataset, train_sft, train_rlvr

    model, tokenizer = load_model(model_name)

    # Phase 1: SFT
    print("[modal] Phase 1: SFT on Fable5 dataset")
    dataset = load_fable5_dataset(limit=limit)
    sft_path = train_sft(
        model, tokenizer, dataset,
        out_dir="/root/models/qwen_sft",
        batch_size=4,  # Fits L4 24GB
    )

    if not sft_only:
        # Phase 2: RLVR
        print("[modal] Phase 2: RLVR")
        from eval.humaneval_loader import load as load_humaneval
        problems = load_humaneval(limit=10)
        train_rlvr(
            model, tokenizer, problems,
            out_dir="/root/models/qwen_rlvr",
            n_steps=200,
        )

    # Save to HF Hub
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo("mateo0093/le-gros-chaton-qwen", private=True, exist_ok=True)
    api.upload_folder(
        folder_id="mateo0093/le-gros-chaton-qwen",
        folder_path="/root/models",
        path_in_repo=".",
    )
    print("[modal] Model pushed to HF Hub: mateo0093/le-gros-chaton-qwen")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Full 160K dataset")
    parser.add_argument("--rlvr", action="store_true", help="Include RLVR phase")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()

    limit = None if args.full else 10000
    train.remote(limit=limit, sft_only=not args.rlvr, model_name=args.model)
