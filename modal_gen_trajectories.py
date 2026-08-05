"""Modal launch script for PHASE 2a: diverse trajectory generation.

Runs gen_trajectories.py on a GPU using the trained Le Gros Chaton SFT adapter
(checkpoint-6200 = the 91% fat cat), with CREATIVITY sampling enabled:
5 high-temperature solutions per task, only keeping verified + NOVEL ones
(n-gram overlap filter). The resulting agent_traces_full.jsonl is uploaded
to HF so the trajectory SFT (phase 2b) can consume it.

Usage:
    python modal_gen_trajectories.py --n 100 --samples 5 --temp 0.9
    python modal_gen_trajectories.py --n 200 --samples 3 --temp 0.9 \
        --adapter mateo0093/le-gros-chaton-qwen
"""
import os
import argparse
import modal

app = modal.App("le-gros-chaton-trajectories")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
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
        "gitpython",
    )
    .run_commands(
        "pip install wheel setuptools ninja && pip install causal-conv1d --no-build-isolation || true",
        "pip install 'flash-linear-attention[cuda]' --no-build-isolation || true",
    )
    .add_local_dir(".", "/root/proj", ignore=["__pycache__", ".git", "*.pt", "*.bin"])
)

HF_SECRET = os.environ.get("CHATON_HF_SECRET", "chaton-hf")


@app.function(
    gpu="L4",
    image=image,
    timeout=86400,
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def gen(n: int = 50, samples: int = 5, temp: float = 0.9,
        novelty_thresh: float = 0.5, use_4bit: bool = True,
        adapter: str = "mateo0093/le-gros-chaton-qwen",
        out: str = "agent_traces_full.jsonl",
        upload_repo: str = "mateo0093/le-gros-chaton-traces"):
    os.chdir("/root/proj")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # make_repo() runs git init/commit — the container has no git identity.
    import subprocess as _sp
    _sp.run(["git", "config", "--global", "user.email", "chaton@modal.local"])
    _sp.run(["git", "config", "--global", "user.name", "Le Gros Chaton"])

    import subprocess, sys, json
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "")

    # Download the adapter if it's an HF repo id (fat cat SFT weights).
    local_adapter = adapter
    if "://" not in adapter and "/" in adapter and not os.path.isdir(adapter):
        from huggingface_hub import snapshot_download
        print(f"[gen-modal] Pulling adapter {adapter}...")
        local_adapter = snapshot_download(repo_id=adapter, token=token,
                                          ignore_patterns=["*.bin", "optimizer.pt"])
        # PEFT multi-adapter layout may nest under sft/
        nested = os.path.join(local_adapter, "sft")
        if os.path.isdir(nested):
            local_adapter = nested
        print(f"[gen-modal] adapter at {local_adapter}")

    cmd = [
        sys.executable, "-u", "gen_trajectories.py",
        "--n", str(n),
        "--model", "Qwen/Qwen3.5-9B",
        "--ckpt", local_adapter,
        "--samples", str(samples),
        "--temp", str(temp),
        "--novelty-thresh", str(novelty_thresh),
        "--out", out,
    ]
    if use_4bit:
        cmd += ["--use-4bit"]
    print("[gen-modal] CMD:", " ".join(cmd))
    try:
        r = subprocess.run(cmd)
    except Exception as e:
        # Dump the real error somewhere we can read it (Modal swallows the
        # traceback when a subprocess dies mid-run).
        import traceback as _tb
        err = _tb.format_exc()
        print("[gen-modal] SUBPROCESS CRASHED:", err)
        if token:
            from huggingface_hub import HfApi
            HfApi(token=token).upload_file(
                path_or_fileobj=err.encode(),
                path_in_repo="gen_traces_error.txt",
                repo_id=upload_repo, repo_type="dataset")
        raise
    if r.returncode != 0:
        raise RuntimeError(f"gen_trajectories.py failed rc={r.returncode}")

    # Upload traces to HF so phase 2b (trajectory SFT on Modal) can pull them.
    if token and os.path.isfile(out):
        api = HfApi(token=token)
        api.create_repo(upload_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=out, path_in_repo=out,
                        repo_id=upload_repo, repo_type="dataset")
        print(f"[gen-modal] ✓ Uploaded {out} -> {upload_repo}")

        # Count what we got (verified / novel / self-reviewed).
        n_ok = n_self = 0
        with open(out) as f:
            for line in f:
                e = json.loads(line)
                n_ok += 1 if e.get("verified") else 0
                n_self += 1 if e.get("self_review") else 0
        print(f"[gen-modal] traces: {n_ok} verified / {n_self} with self-review")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Tasks to attempt")
    parser.add_argument("--samples", type=int, default=5,
                        help="Diversity samples per task (creativity)")
    parser.add_argument("--temp", type=float, default=0.9,
                        help="Sampling temperature for diverse solutions")
    parser.add_argument("--novelty-thresh", type=float, default=0.5,
                        help="Max n-gram overlap with kept solutions to stay novel")
    parser.add_argument("--adapter", default="mateo0093/le-gros-chaton-qwen",
                        help="SFT adapter to generate with (fat cat weights)")
    parser.add_argument("--out", default="agent_traces_full.jsonl")
    parser.add_argument("--upload-repo", default="mateo0093/le-gros-chaton-traces")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    gen.remote(
        n=args.n, samples=args.samples, temp=args.temp,
        novelty_thresh=args.novelty_thresh,
        use_4bit=not args.no_4bit, adapter=args.adapter,
        out=args.out, upload_repo=args.upload_repo,
    )


if __name__ == "__main__":
    main()
