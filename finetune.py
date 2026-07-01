"""Instruction fine-tuning.

Two data sources, picked by CHATON_INSTRUCT env var:
  "local"  (default) -> finetune_data.json (your ~48 hand-written Q&A pairs)
  "dolly"            -> databricks-dolly-15k, filtered to short pairs (~5-8k)
The model loads base model.pt and learns to FOLLOW the chat TEMPLATE
("### Human: ...  ### Assistant: ...") with loss only on the assistant turn.
Be honest: this teaches FORMAT (structure/tone/stop), NOT reasoning at 26M.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from model import GPT
import config as cfg
from tokenizer import encode, decode, EOT_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCT_SOURCE = os.environ.get("CHATON_INSTRUCT", "local")

# chat template wrapping. ends each turn with EOT so the model learns to stop.
HUMAN = "### Human:"
ASSIST = "### Assistant:"


def _load_examples():
    if INSTRUCT_SOURCE == "dolly":
        from datasets import load_dataset
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        out = []
        for ex in ds:
            p = (ex.get("instruction") or "").strip()
            if ex.get("context"):
                p = (p + "\n" + ex["context"].strip()).strip()
            r = (ex.get("response") or "").strip()
            # keep short pairs that will fit block_size comfortably
            if len(p) + len(r) < 600 and p and r:
                out.append({"prompt": p, "response": r})
        print(f"[ft] dolly: kept {len(out)} short pairs")
        return out
    else:
        with open("finetune_data.json", "r", encoding="utf-8") as f:
            ex = json.load(f)
        # adapt the Q: / A: local format into the chat template here for consistency
        return ex


examples = _load_examples()


def _format_prompt(text):
    """Wrap a user prompt in the chat template. The local finetune_data uses
    'Q: ...\\nA:' already; for dolly we build '### Human: ... ### Assistant:'."""
    if INSTRUCT_SOURCE == "dolly":
        return f"{HUMAN} {text}\n{ASSIST}"
    return text   # local pairs already shaped as "Q: ...\nA:"


def get_finetune_batch(batch_size, block_size):
    xs, ys = [], []
    for _ in range(batch_size):
        ex = examples[torch.randint(len(examples), (1,)).item()]
        prompt = _format_prompt(ex["prompt"])
        prompt_ids = encode(prompt)
        response_ids = encode(ex["response"]) + [EOT_TOKEN]
        full = (prompt_ids + response_ids)[: block_size + 1]

        x = full[:-1]
        y = full[1:].copy()
        # mask the prompt positions (the model should only be scored on the response)
        prompt_len = len(prompt_ids)
        for i in range(min(prompt_len - 1, len(y))):
            y[i] = -100
        xs.append(x)
        ys.append(y)

    for i in range(len(xs)):
        pad = block_size - len(xs[i])
        if pad > 0:
            xs[i] = xs[i] + [EOT_TOKEN] * pad
            ys[i] = ys[i] + [-100] * pad

    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device))


if __name__ == "__main__":
    print("Using device:", device)
    print(f"Loaded {len(examples)} instruct examples (source={INSTRUCT_SOURCE})")

    model = GPT().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    print("Loaded base model.pt")

    # gentle, lower lr so we don't clobber base knowledge
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    max_iters = 600
    ft_batch = 8
    grad_accum = 2

    model.train()
    for step in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(grad_accum):
            x, y = get_finetune_batch(ft_batch, cfg.block_size)
            logits, loss, _ = model(x, targets=y)
            (loss / grad_accum).backward()
            accum += loss.item()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {accum/grad_accum:.4f}")

    torch.save(model.state_dict(), "model_finetuned.pt")
    print("Saved fine-tuned model to model_finetuned.pt")