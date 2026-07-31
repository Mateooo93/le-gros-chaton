"""Local inference server for the fine-tuned 9B coding agent.

Loads the base Qwen3.5-9B + LoRA adapters, quantizes to 4-bit, and serves
inference via a simple HTTP API or CLI.  Designed to run on a laptop/desktop
GPU (8-16GB) locally.

Usage (CLI):
    python serve_qwen.py --ckpt qwen_coding_agent --prompt "write a fib function"

Usage (HTTP):
    python serve_qwen.py --ckpt qwen_coding_agent --serve --port 8000
    curl -X POST localhost:8000/generate -d '{"prompt": "def fib(n):"}'
"""
import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def load_model(model_name: str, ckpt_path: str | None = None,
               use_4bit: bool = True, use_8bit: bool = False):
    """Load base model + LoRA adapter, quantized for local running."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        quant = BitsAndBytesConfig(load_in_8bit=True)

    print(f"[serve] Loading base {model_name}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if ckpt_path and os.path.isdir(ckpt_path):
        print(f"[serve] Loading LoRA adapter from {ckpt_path}...")
        model = PeftModel.from_pretrained(model, ckpt_path)
        model.eval()
        print("[serve] LoRA adapter applied")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path or model_name,
                                              trust_remote_code=True)
    print(f"[serve] Loaded in {time.time() - t0:.1f}s")

    # Memory report
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"[serve] GPU memory used: {used:.1f} GB")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new: int = 512,
             temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Generate a completion."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            temperature=temperature, top_p=top_p, do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def serve(model, tokenizer, port: int = 8000):
    """Start an HTTP server."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        print("[serve] Install fastapi: pip install fastapi uvicorn")
        return

    app = FastAPI(title="Le Gros Chaton 9B")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "Qwen3.5-9B + LoRA"}

    @app.post("/generate")
    async def generate_endpoint(req: Request):
        body = await req.json()
        prompt = body.get("prompt", "")
        max_new = body.get("max_new_tokens", 512)
        temperature = body.get("temperature", 0.7)
        if not prompt:
            return JSONResponse({"error": "prompt required"}, status_code=400)
        t0 = time.time()
        text = generate(model, tokenizer, prompt, max_new, temperature)
        elapsed = time.time() - t0
        return {"text": text, "elapsed_s": round(elapsed, 2),
                "tokens": len(tokenizer.encode(text))}

    import uvicorn
    print(f"[serve] Listening on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    parser = argparse.ArgumentParser(description="Local 9B coding agent server")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None, help="LoRA adapter dir")
    parser.add_argument("--prompt", default=None, help="CLI mode: generate once")
    parser.add_argument("--serve", action="store_true", help="HTTP server mode")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-quant", action="store_true",
                        help="Skip 4-bit quantization")
    args = parser.parse_args()

    import torch
    model, tokenizer = load_model(
        args.model, args.ckpt,
        use_4bit=not args.no_quant, use_8bit=False,
    )

    if args.serve:
        serve(model, tokenizer, port=args.port)
    elif args.interactive:
        print("[serve] Interactive mode. Type 'quit' to exit.")
        print("Type 'clear' to reset conversation.\n")
        history = []
        while True:
            user = input(">>> ")
            if user.strip().lower() in ("quit", "exit"):
                break
            if user.strip().lower() == "clear":
                history = []
                print("(conversation cleared)")
                continue
            if not user.strip():
                continue
            history.append({"role": "user", "content": user})
            # Simple 2-turn context window for the local model
            prompt_msgs = history[-6:]
            prompt = "".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                for m in prompt_msgs
            ) + "<|im_start|>assistant\n"
            t0 = time.time()
            text = generate(model, tokenizer, prompt, args.max_new, args.temperature)
            elapsed = time.time() - t0
            print(f"  {text}")
            print(f"  [{elapsed:.1f}s]\n")
            history.append({"role": "assistant", "content": text})
    elif args.prompt:
        t0 = time.time()
        text = generate(model, tokenizer, args.prompt, args.max_new, args.temperature)
        elapsed = time.time() - t0
        print(text)
        print(f"\n[{elapsed:.1f}s]")
    else:
        print("[serve] Provide --prompt, --interactive, or --serve")


if __name__ == "__main__":
    main()
