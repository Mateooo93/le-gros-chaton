"""Serve Le Gros Chaton (Qwen3.5-9B base or adapter) as an OpenAI-compatible
endpoint on Modal for Terminal-Bench 2.0 evaluation.

This is the recommended model backend for ``eval/tbench_eval.py`` on boxes
without enough local VRAM/disk for the weights (it also keeps the local disk
free of ~20GB of model files). It uses vLLM, so it serves the SAME OpenAI
``/v1/chat/completions`` API the harness speaks, and later evals (traj_sft
adapter, final RLVR) reuse it unchanged by switching the env vars.

Usage:
    # deploy (builds the vllm image on Modal's infra, ~5-10 min first time)
    SERVE_MODEL=Qwen/Qwen3.5-9B VLLM_API_KEY=<token> \
        .venv/bin/python modal_serve_qwen.py

    # or with an adapter merged via PEFT (trajectory-SFT / RLVR checkpoints):
    SERVE_MODEL=Qwen/Qwen3.5-9B SERVE_ADAPTER=mateo0093/le-gros-chaton-qwen \
        VLLM_API_KEY=<token> .venv/bin/python modal_serve_qwen.py

    # print the endpoint URL + auth, then run the eval:
    python eval/tbench_eval.py --run --model-server <url> \
        --model-name Qwen/Qwen3.5-9B --model-api-key <token> \
        --label "Qwen3.5-9B-baseline" --adapter base

Env vars:
    SERVE_MODEL   HF model id to serve (default Qwen/Qwen3.5-9B)
    SERVE_ADAPTER HF LoRA adapter to merge (optional; peft merge at boot)
    SERVE_GPU     modal GPU string (default L4; A10G/A100-40GB for adapter+
                  fp16 merge headroom)
    VLLM_API_KEY  optional bearer token required by the client
    HF_TOKEN      HF token (or use the 'chaton-hf' Modal secret)

The deployed endpoint is a Modal web server: each container idles up to
``container_idle_timeout`` seconds before scaling to zero, so the pilot costs
only the GPU time actually used (~$0.80/hr on L4).
"""
import os

import modal

app = modal.App("le-gros-chaton-serve")

SERVE_MODEL = os.environ.get("SERVE_MODEL", "Qwen/Qwen3.5-9B")
SERVE_ADAPTER = os.environ.get("SERVE_ADAPTER", "")
SERVE_GPU = os.environ.get("SERVE_GPU", "L4")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "huggingface_hub", "peft")
)

HF_SECRET = os.environ.get("CHATON_HF_SECRET", "chaton-hf")


def _serve_argv(model_dir: str) -> list[str]:
    argv = [
        "vllm", "serve", model_dir,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.9",
        "--trust-remote-code",
        "--disable-log-requests",
        "--served-model-name", SERVE_MODEL,
    ]
    if VLLM_API_KEY:
        argv += ["--api-key", VLLM_API_KEY]
    return argv


@app.function(
    image=image,
    gpu=SERVE_GPU,
    scaledown_window=1200,
    timeout=86400,
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, label="le-gros-chaton-qwen")
def web() -> None:
    import subprocess
    import time

    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(SERVE_MODEL)
    if SERVE_ADAPTER:
        # Merge the LoRA adapter into a copy so vLLM serves the adapted model.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[serve] merging adapter {SERVE_ADAPTER} into {SERVE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        merged = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype="auto", trust_remote_code=True),
            SERVE_ADAPTER,
        )
        merged = merged.merge_and_unload()
        merged_dir = "/root/merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        model_dir = merged_dir
        print(f"[serve] adapter merged, serving {merged_dir}")

    argv = _serve_argv(model_dir)
    print(f"[serve] starting: {' '.join(argv[:6])} ...")
    proc = subprocess.Popen(argv)
    try:
        while True:
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        proc.terminate()


if __name__ == "__main__":
    print(f"[serve] deploying SERVE_MODEL={SERVE_MODEL} "
          f"SERVE_ADAPTER={SERVE_ADAPTER or '(none)'} GPU={SERVE_GPU}")
    with app.run():
        try:
            url = web.get_web_url()
        except Exception:
            url = None
        print(f"[serve] endpoint: {url}")
        print(f"[serve] auth:     {'Bearer ' + VLLM_API_KEY if VLLM_API_KEY else 'none'}")
        if url:
            # Probe the endpoint so the GPU container boots and the model loads.
            import time, urllib.request
            for attempt in range(120):
                try:
                    with urllib.request.urlopen(url + "/health", timeout=10) as r:
                        if r.status == 200:
                            print(f"[serve] container warm after {attempt * 15}s")
                            break
                except Exception:
                    time.sleep(15)
        print("[serve] deployment complete. Run the eval with --model-server "
              f"{url} --model-name {SERVE_MODEL}")
