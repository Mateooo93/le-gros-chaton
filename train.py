import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
from model import GPT
import config as cfg
# NOTE: imports data2 (streaming memmap pipeline) if present, else falls back to data.py.
try:
    from data2 import get_batch
    _DATA = "data2 (streaming memmap)"
except Exception:
    from data import get_batch
    _DATA = "data.py (fallback)"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, "| data:", _DATA)

model = GPT().to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))


def get_lr(step):
    """warmup 0->lr_max, then cosine decay lr_max->lr_min."""
    if step < cfg.warmup_iters:
        return cfg.lr_max * step / cfg.warmup_iters
    decay = cfg.max_iters - cfg.warmup_iters
    progress = (step - cfg.warmup_iters) / max(1, decay)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    return cfg.lr_min + coeff * (cfg.lr_max - cfg.lr_min)


@torch.no_grad()
def estimate_loss(eval_iters=None):
    eval_iters = eval_iters or cfg.eval_iters
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, cfg.micro_batch, cfg.block_size)
            with torch.autocast(device_type=device, dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# 4. The training loop with GRADIENT ACCUMULATION.
#    We run `grad_accum` forward/backward passes summing (scaled) gradients,
#    then ONE optimizer step. Effective batch = micro_batch * grad_accum.
for step in range(cfg.max_iters):
    # set the LR for this step (warmup + cosine)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for _ in range(cfg.grad_accum):
        x, y = get_batch("train", cfg.micro_batch, cfg.block_size)
        with torch.autocast(device_type=device, dtype=torch.float16):
            _, loss, _ = model(x, targets=y)
        # scale by 1/grad_accum so the accumulated grad averages (not sums)
        scaler.scale(loss / cfg.grad_accum).backward()
        accum_loss += loss.item()

    # clip the UNSCALED gradient norm to grad_clip (unscale first for scaler)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    if step % cfg.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:4d}  lr {lr:.2e}  train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")

model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
torch.save(model_to_save.state_dict(), "model.pt")
print("Saved model to model.pt")