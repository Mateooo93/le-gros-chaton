"""All the numbers that define the model + training, in one place.

Switch between profiles with CHATON_PROFILE:
  dev  (default) -> 14.4M dense for the 2070 (architecture dev + smoke tests)
  smol-fat       -> 290M/120M-active MoE for pipeline proof runs
  fat            -> "le fat chaton": 10.5B/3.83B-active MoE for the cloud pretrain

Everything else is env-overridable so each VM can tune batch/lr/iters without
editing this file.

Exports:
  ARCH_KEYS  — tuple of architecture-defining field names, used by checkpoint.py
               for clean config snapshots and arch-mismatch detection.
"""
import os as _os

# --- Seed for reproducibility ---
SEED = int(_os.environ.get("CHATON_SEED", "42"))

PROFILE = _os.environ.get("CHATON_PROFILE", "dev")

if PROFILE == "smol-fat":
    # --- proof-of-concept MoE: 290.0M total / 120.1M active. Trains on a
    #     single L4 24GB in ~20-40 min for ~1000 steps. Proves the pipeline
    #     cheaply before committing to the full fat profile. Uses SwiGLU +
    #     GQA + shared expert so training dynamics mirror ``fat``. ---
    vocab_size   = 50257
    block_size   = 2048
    n_layer      = 12
    n_head       = 8
    n_embd       = 512
    use_moe      = True
    n_expert     = 8
    n_expert_top = 2
    moe_aux_loss = 0.01
    mlp_type     = "swiglu"   # gated MLP (Llama/Qwen/DeepSeek standard)
    n_kv_head    = 4          # GQA: 4 KV heads shared across 8 Q heads
    n_shared_expert = 1       # DeepSeek-style always-on shared expert
    use_qk_norm  = True       # QK-normalisation for long-context stability
    gradient_checkpointing = True  # recompute activations on backward
elif PROFILE == "fat":
    # --- le fat chaton: MoE, 10.47B total, 3.83B active per token ---
    # 16 experts x ~655M each, top-2 + shared expert.
    # Architecture: SwiGLU, GQA (8 KV heads), 1 shared expert + 16 routed experts.
    vocab_size   = 50257      # keep rich code vocab (don't shrink for a coder)
    block_size   = 4096       # repo-scale context (RoPE; can YaRN-extend later)
    n_layer      = 32
    n_head       = 16
    n_embd       = 2048       # head_dim = 128
    use_moe      = True
    n_expert     = 8
    n_expert_top = 2
    moe_aux_loss = 0.01
    mlp_type     = "swiglu"   # gated MLP (Llama/Qwen standard) — better per param
    n_kv_head    = 8          # GQA: 8 KV heads shared across 16 Q heads
    n_shared_expert = 1       # DeepSeek-style always-on shared expert
    use_qk_norm  = True       # QK-normalisation for long-context stability
    gradient_checkpointing = True  # recompute activations on backward
else:
    # --- dev profile: tiny, fits 8GB 2070, proves the architecture ---
    vocab_size   = 50257
    block_size   = 512
    n_layer      = 6
    n_head       = 4
    n_embd       = 256
    use_moe      = False      # flip True to test the MoE on the 2070
    n_expert     = 8
    n_expert_top = 2
    moe_aux_loss = 0.01
    mlp_type     = "gelu"     # switch to "swiglu" to test the new MLP on the 2070
    n_kv_head    = n_head     # MHA by default; set < n_head to test GQA
    n_shared_expert = 0       # set 1 to test shared-expert MoE on the 2070
    use_qk_norm  = True       # set False to test without QK-norm on the 2070
    gradient_checkpointing = False  # small model fits without checkpointing
    dynamic_topk = False       # enable for dev-testing dynamic routing

# --- Training (env-overridable per VM) ---
micro_batch  = int(_os.environ.get("CHATON_MICRO_BATCH", "8"))
grad_accum   = int(_os.environ.get("CHATON_GRAD_ACCUM", "4"))
batch_size   = micro_batch * grad_accum

# block_size can be shrunk per-VM to fit memory (the profile default is the
# max; smaller = fewer activations). Env-override so a tight T4 can drop to 1024
# without editing profiles.
if _os.environ.get("CHATON_BLOCK_SIZE"):
    block_size = int(_os.environ["CHATON_BLOCK_SIZE"])

lr_max       = float(_os.environ.get("CHATON_LR_MAX", "3e-4"))
lr_min       = float(_os.environ.get("CHATON_LR_MIN", "3e-5"))
warmup_iters = int(_os.environ.get("CHATON_WARMUP", "300"))
max_iters    = int(_os.environ.get("CHATON_MAX_ITERS", "8000"))
# LR schedule: "cosine" (standard) or "wsd" (warmup-stable-decay).
# WSD is better for VM-hopping: train at full LR until cooldown starts.
lr_schedule  = _os.environ.get("CHATON_LR_SCHEDULE", "cosine")
# WSD cooldown: LR linearly decays to zero over this many final steps.
# Only used when lr_schedule="wsd". 0 = no cooldown (constant LR after warmup).
cooldown_iters = int(_os.environ.get("CHATON_COOLDOWN_ITERS", "500"))
eval_interval = int(_os.environ.get("CHATON_EVAL_INTERVAL", "250"))
grad_clip    = float(_os.environ.get("CHATON_GRAD_CLIP", "1.0"))
eval_iters   = int(_os.environ.get("CHATON_EVAL_ITERS", "50"))
# Selective weight decay (AdamW).  Standard: 0.1 for linear/embed, 0.0 for norms/biases.
weight_decay = float(_os.environ.get("CHATON_WEIGHT_DECAY", "0.1"))
# KV cache compression (StreamingLLM-style).  When the cache exceeds
# *max_cache_len*, keep the first *cache_n_sink* tokens + last *n_local* tokens.
# This bounds cache memory during long agent rollouts.
max_cache_len = int(_os.environ.get("CHATON_MAX_CACHE_LEN", "0")) or block_size
cache_n_sink  = int(_os.environ.get("CHATON_CACHE_N_SINK", "4"))
# Quantile-balanced routing (Kimi K3-style).  Replaces auxiliary load-balancing
# loss with a deterministic quantile threshold per expert.  Each expert receives
# roughly the same number of tokens, eliminating the need for aux_loss or bias.
quantile_balance = _os.environ.get("CHATON_QUANTILE_BALANCE", "").lower() in ("1", "true", "yes")
# SiTU activation (Sigmoid Tanh Unit, Kimi K3-style).  Replaces SiLU in SwiGLU
# to avoid dead-neuron pathology in rarely-activated MoE experts.
use_situ = _os.environ.get("CHATON_USE_SITU", "").lower() in ("1", "true", "yes")
# Latent MoE routing (Kimi K3-style).  Token is compressed before routing,
# then decompressed after expert computation.  Reduces routing computation
# and communication in multi-GPU settings.  0 = disabled.
moe_latent_dim = int(_os.environ.get("CHATON_MOE_LATENT_DIM", "0"))
# Learned residual scaling (AttnRes-inspired, Kimi K3).  Each layer learns
# a scaling factor for its attention residual connection, allowing the model
# to control information flow across depth.
learned_residual = _os.environ.get("CHATON_LEARNED_RESIDUAL", "").lower() in ("1", "true", "yes")
# Routing-Free MoE (arXiv:2604.00801).  Eliminates centralized routers, Softmax,
# and TopK.  Each expert independently decides its own activation via internal
# low-rank gate.  Default: False (uses standard top-k routing).
use_routing_free = _os.environ.get("CHATON_USE_ROUTING_FREE", "").lower() in ("1", "true", "yes")
# Multi-Head Latent Attention (DeepSeek-V2): KV cache compression via
# low-rank latent.  kv_latent_dim=0 disables MLA (uses standard GQA).
# Recommended: kv_latent_dim=4*head_dim for ~4x cache compression.
kv_latent_dim = int(_os.environ.get("CHATON_KV_LATENT_DIM", "0"))
# RoPE scaling for context extension (NTK-aware YaRN).
# scale > 1.0 extends effective context without retraining.
#   rope_scaling_scale=2.0  -> 2x block_size context (e.g., 2048 → 4096)
#   rope_scaling_scale=4.0  -> 4x block_size context (e.g., 2048 → 8192)
rope_scaling_scale = float(_os.environ.get("CHATON_ROPE_SCALE", "1.0"))

# MoE auxiliary loss coefficients
moe_aux_loss = float(_os.environ.get("CHATON_MOE_AUX_LOSS", str(moe_aux_loss)))
# Z-loss (DeepSeek-MoE): penalises extreme router logits to stabilise training.
#   z_loss = mean(log(sum(exp(gate_logits)))^2)
# Typical range: 0.001 (DeepSeek-V2) — 0.01 (stronger, for unstable runs).
moe_z_loss  = float(_os.environ.get("CHATON_MOE_Z_LOSS", "0.001"))

# --- Architecture-defining keys used by checkpoint.py for config snapshot and
#     arch-mismatch detection. The canonical list; checkpoint.py imports this
#     instead of guessing from dir(cfg). ---
ARCH_KEYS = (
    "n_embd", "n_layer", "n_head", "n_kv_head",
    "vocab_size", "use_moe", "n_expert", "n_expert_top",
    "n_shared_expert", "mlp_type", "block_size", "use_qk_norm",
    "kv_latent_dim", "quantile_balance", "use_situ", "moe_latent_dim", "learned_residual", "use_routing_free",
)

# --- allow architecture flags to be flipped via env without editing this file
#     (lets us test SwiGLU/GQA/shared-expert on the 2070 dev profile quickly).
#     Empty/unset = keep the profile default, so the smol-fat run (which passes
#     no arch env vars) is never disturbed. ---
if _os.environ.get("CHATON_MLP_TYPE"):
    mlp_type = _os.environ["CHATON_MLP_TYPE"]
if _os.environ.get("CHATON_N_KV_HEAD"):
    n_kv_head = int(_os.environ["CHATON_N_KV_HEAD"])
if _os.environ.get("CHATON_N_SHARED_EXPERT"):
    n_shared_expert = int(_os.environ["CHATON_N_SHARED_EXPERT"])
if _os.environ.get("CHATON_USE_QK_NORM", "").lower() in ("0", "false", "no"):
    use_qk_norm = False
if _os.environ.get("CHATON_GRAD_CKPT", "").lower() in ("0", "false", "no"):
    gradient_checkpointing = False
if _os.environ.get("CHATON_DYNAMIC_TOPK", "").lower() in ("1", "true", "yes"):
    dynamic_topk = True
    dynamic_topk_threshold = float(
        _os.environ.get("CHATON_DYNAMIC_TOPK_THRESHOLD", "2.0")
    )


# --- Config validation -------------------------------------------------------
def validate() -> list[str]:
    """Check the resolved config for common misconfigurations.

    Returns a list of human-readable warnings.  An empty list means
    everything looks sane.  Call this before building the model or starting
    training.
    """
    issues: list[str] = []

    # Architecture constraints
    if n_embd % n_head != 0:
        issues.append(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
    if n_head % n_kv_head != 0:
        issues.append(f"n_head ({n_head}) must be a multiple of n_kv_head ({n_kv_head})")
    if use_moe and n_expert_top >= n_expert:
        issues.append(
            f"n_expert_top ({n_expert_top}) must be < n_expert ({n_expert})"
        )
    if mlp_type not in ("gelu", "swiglu"):
        issues.append(f"unknown mlp_type {mlp_type!r}; expected 'gelu' or 'swiglu'")
    if kv_latent_dim > 0 and kv_latent_dim % head_dim != 0:
        issues.append(
            f"kv_latent_dim ({kv_latent_dim}) should be a multiple of "
            f"head_dim ({head_dim}) for optimal MLA performance"
        )

    # Training constraints
    if micro_batch * grad_accum <= 0:
        issues.append(
            f"effective batch size ({batch_size}) must be positive "
            f"(micro_batch={micro_batch}, grad_accum={grad_accum})"
        )
    if lr_max <= 0:
        issues.append(f"lr_max ({lr_max}) must be positive")
    if warmup_iters >= max_iters:
        issues.append(
            f"warmup_iters ({warmup_iters}) >= max_iters ({max_iters}) — "
            "model will never leave warmup"
        )
    if block_size < 64:
        issues.append(f"block_size ({block_size}) is very small; sequences may be mostly padding")

    # Schedule validation
    if lr_schedule not in ("cosine", "wsd"):
        issues.append(f"unknown lr_schedule {lr_schedule!r}; expected 'cosine' or 'wsd'")
    if lr_schedule == "wsd" and cooldown_iters <= 0:
        issues.append("wsd schedule requires cooldown_iters > 0 "
                      "(set CHATON_COOLDOWN_ITERS)")
    if lr_schedule == "wsd" and cooldown_iters >= max_iters - warmup_iters:
        issues.append(
            f"wsd cooldown_iters ({cooldown_iters}) >= stable phase "
            f"({max_iters - warmup_iters}); cooldown would overlap warmup"
        )

    # Profile-specific warnings
    if PROFILE == "fat" and block_size < 2048:
        issues.append(
            f"fat profile with block_size={block_size} — "
            "recommend ≥2048 for repo-scale context"
        )
    if PROFILE == "dev" and use_moe:
        issues.append(
            "dev profile with MoE enabled — this is fine for testing but "
            "the dense MLP is more representative of the dev profile's scale"
        )

    return issues