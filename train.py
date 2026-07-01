import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
from model import GPT
import config as cfg
import checkpoint as ckpt
# --- data source: wikitext by default, CODE for the fat coding pretrain.
#     CHATON_DATA=wikitext (default) -> data2.py (streaming memmap, wikitext-2/103)
#     CHATON_DATA=code              -> data_code.py (smollm-corpus/stack-v2/starcoderdata)
#     Both expose get_batch(split, batch_size, block_size) so the swap is clean. ---
_DATA_CHOICE = os.environ.get("CHATON_DATA", "wikitext").lower()
if _DATA_CHOICE == "code":
    from data_code import get_batch
    _DATA = "data_code (streaming code corpus)"
else:
    try:
        from data2 import get_batch
        _DATA = "data2 (streaming memmap wikitext)"
    except Exception:
        from data import get_batch
        _DATA = "data.py (fallback wikitext)"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, "| data:", _DATA)

model = GPT().to(device)
# torch.compile optional: great on A100s, but on a T4 its autotune buffers eat
# scarce memory and the compile+autotune is slow. CHATON_COMPILE=0 to disable.
if os.environ.get("CHATON_COMPILE", "1") == "1":
    model = torch.compile(model)
else:
    print("[train] torch.compile DISABLED (CHATON_COMPILE=0)")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

# --- VM-hopping: optionally pull the latest checkpoint from HF Hub and resume.
#     Set CHATON_HF_REPO + HF_TOKEN in the env to use this. If a local ckpt
#     exists, load that instead. start_step = where we resume from.
#     We check the checkpoint's saved config (n_embd/n_layer/use_moe/...) against
#     the current profile BEFORE loading — if they don't match (e.g. a stale dev
#     checkpoint left on the Hub from a different profile), skip cleanly instead
#     of throwing a wall of state_dict errors. This is the VM-hopping robustness
#     fix: a checkpoint from a DIFFERENT architecture is not resumable. ---
start_step = 0
if os.environ.get("CHATON_RESUME", "0") == "1":
    try:
        if os.environ.get("CHATON_HF_REPO"):
            ckpt.pull_hub()   # download latest from HF Hub
        if os.path.exists(ckpt.CKPT_PATH):
            # peek at the saved config snapshot before loading the heavy state
            ck_peek = torch.load(ckpt.CKPT_PATH, map_location="cpu", weights_only=False)
            saved_cfg = ck_peek.get("config", {}) if isinstance(ck_peek, dict) else {}
            # the architecture-defining fields that must match to be resumable
            _ARCH_KEYS = ("n_embd", "n_layer", "n_head", "n_kv_head",
                          "vocab_size", "use_moe", "n_expert", "n_shared_expert",
                          "mlp_type")
            mism = {k: (saved_cfg.get(k), getattr(cfg, k)) for k in _ARCH_KEYS
                    if saved_cfg.get(k) != getattr(cfg, k, None) and k in saved_cfg}
            if saved_cfg and mism:
                print(f"[train] checkpoint is from a different architecture "
                      f"({len(mism)} field(s) mismatch: "
                      f"{ {k: f'{a}->{b}' for k,(a,b) in list(mism.items())[:4]} }); "
                      f"not resumable -> starting fresh at step 0")
            else:
                start_step = ckpt.load_checkpoint(ckpt.CKPT_PATH, model, optimizer, scaler, device)
                print(f"[train] resuming from step {start_step}")
    except Exception as e:
        print(f"[train] resume attempted but failed ({str(e)[:200]}...); starting fresh at step 0")

ckpt_interval = int(os.environ.get("CHATON_CKPT_INTERVAL", "500"))  # save every N outer steps


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
for step in range(start_step, cfg.max_iters):
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

    # --- periodic checkpoint + push to HF Hub (VM-hopping). Saves every
    #     ckpt_interval outer steps so a disconnect loses at most that much. ---
    if (step + 1) % ckpt_interval == 0:
        ckpt.save_checkpoint(ckpt.CKPT_PATH, model, optimizer, step + 1, scaler)
        if os.environ.get("CHATON_HF_REPO"):
            try:
                ckpt.push_hub()
            except Exception as e:
                print(f"[train] hub push failed ({e}); local ckpt still saved")

model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
torch.save(model_to_save.state_dict(), "model.pt")
print("Saved model to model.pt")