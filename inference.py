"""Unified inference engine for Chaton.

Centralises model loading, device management, generation parameters, and
tool-token handling so that ``chat.py``, ``agent/loop.py``, and any other
inference consumer share the same logic and benefit from all features
(tool tokens, typical sampling, RoPE scaling, KV cache compression, etc.).

Usage:
    engine = InferenceEngine(\"model.pt\")
    tokens = engine.generate(\"def hello():\\n\")
    tokens = engine.generate(\"def hello():\\n\", typical_p=0.2, max_new=256)
    text = engine.generate_text(\"def hello():\\n\")
    text = engine.chat([{\"role\": \"user\", \"content\": \"Write fib\"}])
"""
import os
import time
import torch
from tokenizer import encode, decode, tool_token_id, tool_token_name
from model import GPT
import config as cfg


class InferenceEngine:
    """Centralised model loading and text generation.

    Handles:
      - Checkpoint loading (via ``GPT.from_checkpoint``)
      - Vocab extension for tool tokens
      - KV-cache-optimised generation
      - Chat template formatting
      - All sampling strategies (top-k, top-p, typical_p)
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str | None = None,
        compile_model: bool = True,
        extend_vocab: bool = True,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[inference] loading checkpoint from {ckpt_path}")
        t0 = time.time()
        self.model = GPT.from_checkpoint(ckpt_path, device=self.device)
        print(f"[inference] model loaded in {time.time() - t0:.1f}s")

        # Extend vocabulary with tool tokens for the agent loop.
        if extend_vocab and not hasattr(self.model, "_vocab_extended"):
            from tokenizer import tool_token_id
            if tool_token_id("<|tool_call|>") is not None:
                init_from = ["<|tool_call|>", "<|tool_result|>", "<|done|>"]
                self.model.extend_vocab(3, init_from=init_from)
            self.model._vocab_extended = True

        if compile_model and self.device == "cuda":
            self.model = torch.compile(self.model)
            print("[inference] model compiled")

        self.model.eval()
        self._caches = None  # KV cache kept across chat turns
        self._past_len = 0

    def clear_cache(self):
        """Reset the internal KV cache (start a fresh conversation)."""
        self._caches = None
        self._past_len = 0

    @torch.no_grad()
    def generate(
        self,
        prompt: str | torch.Tensor | list[int],
        max_new: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        typical_p: float = 0.0,
        repetition_penalty: float = 1.2,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate token IDs from a prompt.

        Args:
            prompt: string, list of ints, or (1, T) tensor of token IDs.
            … (sampling params match ``model.generate``)

        Returns:
            (1, T_out) tensor of generated token IDs (prompt + new tokens).
        """
        # Normalise prompt to (1, T) tensor.
        if isinstance(prompt, str):
            prompt_ids = encode(prompt)
        elif isinstance(prompt, list):
            prompt_ids = prompt
        else:
            prompt_ids = None

        if prompt_ids is not None:
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        else:
            idx = prompt.clone().to(self.device)

        # Trim to block_size.
        if idx.size(1) > cfg.block_size:
            idx = idx[:, -cfg.block_size:]

        kv_caches = self._caches if use_cache else None
        rope_offset = self._past_len if use_cache else 0

        out, new_caches = self.model.generate(
            idx,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            repetition_penalty=repetition_penalty,
            kv_caches=kv_caches,
            return_caches=True,
        )

        if use_cache:
            self._caches = new_caches
            # Track total position (cached + new tokens) for RoPE offset.
            # After cache compression (StreamingLLM), this may be approximate
            # relative to the compressed length — see model.py for details.
            self._past_len = new_caches[0][0].size(1) if (new_caches and new_caches[0][0].dim() == 3) else (new_caches[0][0].size(2) if new_caches else out.size(1))

        self._last_tokens = out[0].tolist()

        return out

    def generate_text(
        self,
        prompt: str,
        max_new: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        typical_p: float = 0.0,
        **kwargs,
    ) -> str:
        """Generate decoded text from a string prompt."""
        out = self.generate(
            prompt, max_new=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, typical_p=typical_p, **kwargs,
        )
        return decode(out[0].tolist())

    def chat(
        self,
        messages: list[dict],
        max_new: int = 512,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        typical_p: float = 0.2,
        **kwargs,
    ) -> str:
        """Multi-turn chat with a message history.

        *messages* is a list of dicts with ``role`` (system|user|assistant)
        and ``content`` keys.  Messages are formatted with the chat template:

            <|system|>System prompt here<|user|>User message here<|assistant|>

        Returns the assistant's text response.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>{content}")
        parts.append("<|assistant|>")
        prompt_str = "".join(parts)
        return self.generate_text(
            prompt_str, max_new=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, typical_p=typical_p, **kwargs,
        )


def _main():
    """Simple interactive chat entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Inference with Chaton")
    parser.add_argument("--ckpt", default="model.pt", help="Checkpoint path")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--typical-p", type=float, default=0.0,
                        help="Typical sampling threshold (0=off, 0.2=code)")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON with metadata (non-interactive only)")
    args = parser.parse_args()

    engine = InferenceEngine(args.ckpt, device=args.device)

    if args.prompt:
        import time
        t0 = time.time()
        text = engine.generate_text(
            args.prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            typical_p=args.typical_p,
        )
        elapsed = time.time() - t0
        tok_count = len(getattr(engine, "_last_tokens", []))
        if tok_count:
            from tokenizer import encode as tok_encode
            prompt_tokens = len(tok_encode(args.prompt))
            new_tokens = max(tok_count - prompt_tokens, 0)
            tok_s = round(new_tokens / max(elapsed, 1e-6), 1)
        else:
            new_tokens = 0
            tok_s = 0.0
        if args.json:
            import json
            print(json.dumps({
                "text": text,
                "elapsed_s": round(elapsed, 3),
                "new_tokens": new_tokens,
                "tok_s": tok_s,
                "prompt": args.prompt,
            }))
        else:
            print(text)
        return

    print("Chat with Chaton.  Type 'quit' to exit, /clear to reset cache.\n")
    engine.clear_cache()
    history: list[dict] = [{"role": "system",
                            "content": "You are a helpful coding assistant."}]

    while True:
        user = input(">>> ")
        if user.strip().lower() in ("quit", "exit"):
            break
        if user.strip() == "/clear":
            engine.clear_cache()
            history = history[:1]
            print("(cache reset)\n")
            continue
        if not user.strip():
            continue

        history.append({"role": "user", "content": user})
        t0 = time.time()
        reply = engine.chat(
            history, max_new=args.max_new, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, typical_p=args.typical_p,
        )
        elapsed = time.time() - t0
        print(f"  {reply}")
        print(f"  ({elapsed:.1f}s, {len(reply)} chars)\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    _main()