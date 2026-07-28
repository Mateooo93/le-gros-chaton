"""Training entrypoint with resumable checkpoints and VM-hopping.

Usage:
  CHATON_PROFILE=dev python train.py
  CHATON_PROFILE=smol-fat CHATON_DATA=code CHATON_RESUME=1 python train.py

Imports `get_batch` from the data module chosen by $CHATON_DATA.
Exposes a `train()` function called from __main__ so it can also be imported
and called programmatically (e.g. by future RFT/RLVR scripts).
"""
import os

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from model import GPT
import config as cfg
import checkpoint as ckpt
from log import ExperimentLog

# --- Deterministic seeding ---------------------------------------------------
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)


# --- Data source: wikitext by default, CODE for the fat coding pretrain.
#     CHATON_DATA=code      (default) -> data_code.py (smollm-corpus/stack-v2/starcoderdata)
#     CHATON_DATA=wikitext            -> data2.py (streaming memmap, wikitext-2/103)
#     Both expose get_batch(split, batch_size, block_size) so the swap is clean. ---
_DATA_CHOICE = os.environ.get("CHATON_DATA", "code").lower()
if _DATA_CHOICE == "code":
    from data_code import get_batch  # type: ignore
    _DATA = "data_code (streaming code corpus)"
elif _DATA_CHOICE == "wikitext":
    try:
        from data2 import get_batch  # type: ignore
        _DATA = "data2 (streaming memmap wikitext)"
    except ImportError:
        from data import get_batch  # type: ignore
        _DATA = "data.py (fallback wikitext)"
else:
    raise ValueError(f"unknown CHATON_DATA={_DATA_CHOICE!r}; expected 'wikitext' or 'code'")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device, "| data:", _DATA)

# --- Config validation -------------------------------------------------------
_issues = cfg.validate()
if _issues:
    print("\n[config] ⚠ validation warnings:")
    for i, issue in enumerate(_issues, 1):
        print(f"  {i}. {issue}")
    print()


def get_lr(step: int) -> float:
    """Learning rate at *step*.

    ``cosine``: warmup linear 0→lr_max, then cosine decay lr_max→lr_min.
    ``wsd``:    warmup linear 0→lr_max, stable at lr_max for the middle,
                then linear cooldown lr_max→0 over the final ``cooldown_iters``
                steps.  Better for VM-hopping (unknown total steps).
    """
    if step < cfg.warmup_iters:
        return cfg.lr_max * step / cfg.warmup_iters

    if cfg.lr_schedule == "wsd":
        cooldown_start = cfg.max_iters - cfg.cooldown_iters
        if step >= cooldown_start:
            # Linear cooldown: lr_max → 0
            progress = (step - cooldown_start) / max(1, cfg.cooldown_iters)
            return cfg.lr_max * (1.0 - progress)
        return cfg.lr_max  # stable plateau

    # Cosine decay (default)
    decay = cfg.max_iters - cfg.warmup_iters
    progress = (step - cfg.warmup_iters) / max(1, decay)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 → 0
    return cfg.lr_min + coeff * (cfg.lr_max - cfg.lr_min)


@torch.no_grad()
def estimate_loss(eval_iters: int | None = None) -> dict[str, float]:
    """Run the model on train/val splits and return mean losses."""
    eval_iters = eval_iters or cfg.eval_iters
    out: dict[str, float] = {}
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


class ThroughputTracker:
    """Lightweight training throughput instrumentation.

    Tracks tokens/s, FLOPs/s, GPU memory, and ETA.  Logged to the experiment
    logger at each eval interval so throughput regressions (e.g. from a slow
    dataloader, GPU throttling, memory fragmentation) are visible across runs.
    """

    def __init__(self):
        self.t0 = time.time()
        self.tok_count = 0
        self.flop_count = 0
        self.step_count = 0

    def step(self, tokens: int):
        """Call *after* each optimizer step.  *tokens* = B * block_size."""
        self.tok_count += tokens
        self.step_count += 1
        n_params = sum(p.numel() for p in model.parameters())
        self.flop_count += 6 * n_params * tokens

    def snapshot(self) -> dict:
        """Current throughput metrics (snapshot, not cumulative)."""
        elapsed = time.time() - self.t0
        tok_s = self.tok_count / max(elapsed, 1e-9)
        n_params = sum(p.numel() for p in model.parameters())
        flops_s = 6 * n_params * tok_s
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e9
        remaining = cfg.max_iters - self.step_count
        eta_s = remaining * (elapsed / max(self.step_count, 1))
        prog_pct = 100.0 * self.step_count / max(cfg.max_iters, 1)
        return {
            "elapsed_h": round(elapsed / 3600, 2),
            "tok_s": int(tok_s),
            "flops_s": int(flops_s),
            "gpu_mem_gb": round(gpu_mem, 2),
            "eta_s": int(eta_s),
            "prog_pct": round(prog_pct, 1),
        }


def train() -> None:
    """Build model, optimizer, scaler; optionally resume; run the training loop.

    The loop writes checkpoints every `ckpt_interval` steps and pushes to
    HuggingFace Hub if $CHATON_HF_REPO is set.  On completion, saves a bare
    state_dict to ``model.pt``.
    """
    global model, optimizer, scaler  # noqa: PLW0603 — set for inference scripts
    # --- build model ---------------------------------------------------------
    model = GPT(gradient_checkpointing=cfg.gradient_checkpointing).to(device)
    if os.environ.get("CHATON_COMPILE", "1") == "1":
        model = torch.compile(model)
    else:
        print("[train] torch.compile DISABLED (CHATON_COMPILE=0)")

    # Parameter groups with selective weight decay.
    # No decay for: biases, 1D weights (all norms), embedding tables.
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or "embed" in name or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    print(f"[train] {len(decay)} param groups with decay, {len(no_decay)} without")
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr_max)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # --- Exponential Moving Average (EMA) of model weights ------------------
    # EMA smooths training noise by maintaining a shadow copy of parameters
    # that is updated as: ema = decay * ema + (1 - decay) * param.
    # During evaluation, the EMA weights usually give better results.
    # Disable with CHATON_EMA_DECAY=0 or CHATON_EMA_DECAY= (empty).
    ema_decay = float(os.environ.get("CHATON_EMA_DECAY", "0.999"))
    ema_shadow: dict[str, torch.Tensor] = {}
    if ema_decay > 0:
        unwrapped = getattr(model, "_orig_mod", model)
        for name, p in unwrapped.named_parameters():
            if p.requires_grad:
                ema_shadow[name] = p.detach().clone()
        print(f"[train] EMA enabled (decay={ema_decay}, {len(ema_shadow)} params)")

    def _swap_ema(restore: bool = False):
        """Swap model weights with EMA shadow (if EMA enabled)."""
        if not ema_shadow:
            return
        unwrapped = getattr(model, "_orig_mod", model)
        for name, p in unwrapped.named_parameters():
            if name in ema_shadow:
                if restore:
                    # Restore original: swap back
                    tmp = p.data.clone()
                    p.data.copy_(ema_shadow[name])
                    ema_shadow[name] = tmp
                else:
                    # Save EMA: hold current weights in shadow, put EMA in model
                    tmp = p.data.clone()
                    p.data.copy_(ema_shadow[name])
                    ema_shadow[name] = tmp

    def _update_ema():
        """Update EMA shadow after an optimizer step."""
        if not ema_shadow:
            return
        unwrapped = getattr(model, "_orig_mod", model)
        with torch.no_grad():
            for name, p in unwrapped.named_parameters():
                if name in ema_shadow and p.requires_grad:
                    ema_shadow[name].mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

    # --- VM-hopping: pull checkpoint and resume ------------------------------
    # Checks the checkpoint's saved config against the current profile BEFORE
    # loading — if they don't match (e.g. a stale dev checkpoint from a different
    # architecture), skip cleanly instead of throwing state_dict errors.
    start_step = 0
    if os.environ.get("CHATON_RESUME", "0") == "1":
        try:
            if os.environ.get("CHATON_HF_REPO"):
                ckpt.pull_hub()
            if os.path.exists(ckpt.CKPT_PATH):
                ck_peek = torch.load(
                    ckpt.CKPT_PATH, map_location="cpu", weights_only=False
                )
                saved_cfg = (
                    ck_peek.get("config", {}) if isinstance(ck_peek, dict) else {}
                )
                mism = {
                    k: (saved_cfg.get(k), getattr(cfg, k, None))
                    for k in cfg.ARCH_KEYS
                    if saved_cfg.get(k) != getattr(cfg, k, None) and k in saved_cfg
                }
                if saved_cfg and mism:
                    print(
                        f"[train] checkpoint is from a different architecture "
                        f"({len(mism)} field(s) mismatch: "
                        f"{ {k: f'{a}→{b}' for k, (a, b) in list(mism.items())[:4]} }); "
                        f"not resumable → starting fresh at step 0"
                    )
                else:
                    start_step, ckpt_extra = ckpt.load_checkpoint(
                        ckpt.CKPT_PATH, model, optimizer, scaler, device
                    )
                    print(f"[train] resuming from step {start_step}")
                    # Restore EMA shadow from checkpoint extra
                    if ema_shadow and ckpt_extra.get("ema"):
                        saved_ema = ckpt_extra["ema"]
                        for k in list(ema_shadow.keys()):
                            if k in saved_ema:
                                ema_shadow[k] = saved_ema[k].to(device)
                        print(f"[train] restored EMA shadow ({len(saved_ema)} params)")
        except Exception as e:
            print(
                f"[train] resume attempted but failed ({str(e)[:200]}...); "
                f"starting fresh at step 0"
            )

    ckpt_interval = int(os.environ.get("CHATON_CKPT_INTERVAL", "500"))

    # --- experiment logger ---
    run_name = time.strftime("train_%Y%m%d_%H%M%S")
    log_dir = os.environ.get("CHATON_LOG_DIR", os.path.join(PROJ_ROOT, "runs", run_name))
    log = ExperimentLog(log_dir)
    log.write({
        "_type": "meta_run_start",
        "profile": cfg.PROFILE,
        "lr_schedule": cfg.lr_schedule,
        "max_iters": cfg.max_iters,
        "micro_batch": cfg.micro_batch,
        "grad_accum": cfg.grad_accum,
    })

    tracker = ThroughputTracker()

    # --- training loop -------------------------------------------------------
    for step in range(start_step, cfg.max_iters):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch("train", cfg.micro_batch, cfg.block_size)
            with torch.autocast(device_type=device, dtype=torch.float16):
                _, loss, _ = model(x, targets=y)
            scaler.scale(loss / cfg.grad_accum).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)

        # Gradient clipping: for MoE models, clip each expert separately to
        # prevent one expert's noisy gradients from dominating the norm and
        # starving all other experts.  Falls back to global clip for dense.
        if cfg.use_moe:
            # Per-expert gradient clipping: access the underlying model
            # (unwrap torch.compile wrapper if present).
            unwrapped = getattr(model, "_orig_mod", model)
            if hasattr(unwrapped, "blocks"):
                expert_params: list[list[torch.nn.Parameter]] = []
                other_params: list[torch.nn.Parameter] = []
                for block in unwrapped.blocks:
                    if hasattr(block.mlp, "experts"):
                        for e in block.mlp.experts:
                            expert_params.append(list(e.parameters()))
                    else:
                        other_params.extend(block.mlp.parameters())
            # Per-expert clip
            for ep in expert_params:
                if any(p.grad is not None for p in ep):
                    torch.nn.utils.clip_grad_norm_(ep, cfg.grad_clip)
            # Global clip for non-expert params
            if other_params:
                torch.nn.utils.clip_grad_norm_(other_params, cfg.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        _update_ema()

        tracker.step(tokens=cfg.micro_batch * cfg.block_size)

        # Capture aux losses from the last micro-batch (before estimate_loss
        # overwrites them with eval forward passes).
        aux_l = float(getattr(model, "last_aux_loss", 0.0))
        z_l = float(getattr(model, "last_z_loss", 0.0))

        if step % cfg.eval_interval == 0:
            _swap_ema()  # use EMA weights for eval
            losses = estimate_loss()
            _swap_ema(restore=True)  # restore training weights
            perf = tracker.snapshot()
            print(
                f"step {step:4d}  lr {lr:.2e}  "
                f"train loss {losses['train']:.4f}  val loss {losses['val']:.4f}  "
                f"{perf['tok_s']:,} tok/s  "
                f"mem {perf['gpu_mem_gb']:.1f} GB  "
                f"aux {aux_l:.4f}  z {z_l:.6f}  "
                f"eta {perf['eta_s']}s  ({perf['prog_pct']:.0f}%)"
            )
            log.write({
                "step": step,
                "loss": losses["train"],
                "val_loss": losses["val"],
                "lr": lr,
                "aux_loss": aux_l,
                "z_loss": z_l,
                **perf,
            })

        if (step + 1) % ckpt_interval == 0:
            ckpt.save_checkpoint(
                ckpt.CKPT_PATH, model, optimizer, step + 1, scaler,
                extra={"ema": ema_shadow} if ema_shadow else None,
            )
            if os.environ.get("CHATON_HF_REPO"):
                try:
                    ckpt.push_hub()
                except Exception as e:
                    print(f"[train] hub push failed ({e}); local ckpt still saved")

    # --- final save ----------------------------------------------------------
    # Save EMA weights if enabled (they produce better eval results).
    if ema_shadow:
        _swap_ema()  # put EMA weights into model
        model_to_save = getattr(model, "_orig_mod", model)
        torch.save(model_to_save.state_dict(), "model.pt")
        _swap_ema(restore=True)  # restore training weights
        print("Saved EMA model to model.pt")
    else:
        model_to_save = getattr(model, "_orig_mod", model)
        torch.save(model_to_save.state_dict(), "model.pt")
        print("Saved model to model.pt")

    # --- close log ---
    if ema_shadow:
        _swap_ema()
    final_losses = estimate_loss()
    if ema_shadow:
        _swap_ema(restore=True)
    log.close(summary={
        "final_step": step + 1,
        "final_loss": final_losses["train"],
        "final_val_loss": final_losses["val"],
    })


if __name__ == "__main__":
    import sys
    if "--info" in sys.argv:
        # Print effective configuration and exit
        cfg.validate()
        print(f"Profile:       {cfg.PROFILE}")
        print(f"Architecture:  {cfg.n_layer} layers, {cfg.n_embd} embd, "
              f"{cfg.n_head} heads, {cfg.n_kv_head} KV heads")
        print(f"Vocabulary:    {cfg.vocab_size} tokens, block={cfg.block_size}")
        print(f"MoE:           {cfg.use_moe}, {cfg.n_expert} experts, "
              f"top-{cfg.n_expert_top}, shared={cfg.n_shared_expert}")
        print(f"MLP:           {cfg.mlp_type}, GQA={cfg.n_kv_head != cfg.n_head}, "
              f"MLA={getattr(cfg, 'kv_latent_dim', 0)}")
        print(f"Params:        {sum(p.numel() for p in GPT().parameters()):,}")
        print(f"Training:      lr={cfg.lr_max}, batch={cfg.micro_batch * cfg.grad_accum}, "
              f"iters={cfg.max_iters}")
        print(f"Schedule:      {cfg.lr_schedule}, warmup={cfg.warmup_iters}, "
              f"cooldown={cfg.cooldown_iters}")
        print(f"Data:          {os.environ.get('CHATON_DATA', 'code')}")
        print(f"RoPE scale:    {getattr(cfg, 'rope_scaling_scale', 1.0)}")
        print(f"EMA decay:     {os.environ.get('CHATON_EMA_DECAY', '0.999')}")
        print(f"Grad clip:     {cfg.grad_clip}")
        print(f"Grad ckpt:     {cfg.gradient_checkpointing}")
        print(f"Dynamic top-k: {getattr(cfg, 'dynamic_topk', False)}")
        print(f"Quantile bal:  {getattr(cfg, 'quantile_balance', False)}")
        print(f"SiTU act:      {getattr(cfg, 'use_situ', False)}")
        print(f"Latent MoE:    {getattr(cfg, 'moe_latent_dim', 0)}")
        print(f"Learned res:   {getattr(cfg, 'learned_residual', False)}")
        sys.exit(0)
    train()