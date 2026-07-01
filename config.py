"""All the numbers that define the model + training, in one place.

Switch between profiles with CHATON_PROFILE:
  dev  (default) -> tiny model for the 2070 (architecture dev + smoke tests)
  fat            -> "le fat chaton": ~8B-total / ~2B-active MoE for the cloud pretrain

Everything else is env-overridable so each VM can tune batch/lr/iters without
editing this file.
"""
import os as _os

PROFILE = _os.environ.get("CHATON_PROFILE", "dev")

if PROFILE == "smol-fat":
    # --- proof-of-concept MoE: ~1B total / ~0.3B active. Trains on a single
    #     A100 40GB in ~20-40 min for ~1000 steps. Proves the pipeline cheaply
    #     before committing to the full 9B. Real proportions, just smaller. ---
    vocab_size   = 50257
    block_size   = 2048
    n_layer      = 12
    n_head       = 8
    n_embd       = 512
    use_moe      = True
    n_expert     = 8
    n_expert_top = 2
    moe_aux_loss = 0.01
    mlp_type     = "gelu"     # keep smol-fat unchanged for the durability proof run
    n_kv_head    = n_head     # MHA (no GQA) — preserve current behavior
    n_shared_expert = 0       # no shared expert — preserve current behavior
elif PROFILE == "fat":
    # --- le fat chaton: MoE, ~8B total, ~2B active per token ---
    # 8 experts x ~1.1B each, top-2 -> runs at ~2B FLOPs but knows ~8B.
    # Architecture upgrades vs the smol-fat proof: SwiGLU (better per-param),
    # GQA (8 KV heads for 16 Q heads -> 2x smaller KV cache for long agent
    # context decode), 1 shared expert (common knowledge, always active) +
    # 8 routed experts. These are the Qwen3-Coder / DeepSeek-Coder-V2 choices.
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

# --- Training (env-overridable per VM) ---
micro_batch  = int(_os.environ.get("CHATON_MICRO_BATCH", "8"))
grad_accum   = int(_os.environ.get("CHATON_GRAD_ACCUM", "4"))
batch_size   = micro_batch * grad_accum

lr_max       = float(_os.environ.get("CHATON_LR_MAX", "3e-4"))
lr_min       = 3e-5
warmup_iters = int(_os.environ.get("CHATON_WARMUP", "300"))
max_iters    = int(_os.environ.get("CHATON_MAX_ITERS", "8000"))
eval_interval = 250
grad_clip    = 1.0
eval_iters   = int(_os.environ.get("CHATON_EVAL_ITERS", "50"))

# --- allow architecture flags to be flipped via env without editing this file
#     (lets us test SwiGLU/GQA/shared-expert on the 2070 dev profile quickly).
#     Empty/unset = keep the profile default, so the smol-fat run (which passes
#     no arch env vars) is never disturbed. ---
if _os.environ.get("CHATON_MLP_TYPE"):
    mlp_type = _os.environ["CHATON_MLP_TYPE"]
if _os.environ.get("CHATON_N_KV_HEAD"):
    n_kv_head = int(_os.environ["CHATON_N_KV_HEAD"])
elif "n_kv_head" not in dir():
    n_kv_head = n_head
if _os.environ.get("CHATON_N_SHARED_EXPERT"):
    n_shared_expert = int(_os.environ["CHATON_N_SHARED_EXPERT"])
elif "n_shared_expert" not in dir():
    n_shared_expert = 0