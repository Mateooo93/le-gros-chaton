# All the numbers that define your model, in one place.
# Change these here, and everything else picks them up automatically.

# --- Data ---
vocab_size   = 50257   # GPT-2 vocab size (from tokenizer.py)
block_size   = 512     # context length (raised 256->512: SDPA makes this fit)

# --- Model size (~17.7M params; widen AFTER the custom-vocab capstone) ---
n_layer      = 6       # number of transformer blocks
n_head       = 4       # attention heads (must divide n_embd evenly)
n_embd       = 256     # dimension of each token's vector

# --- Training ---
# Effective batch = micro_batch * grad_accum. On 8GB use micro_batch=8,accum=4 (=32).
# On a bigger GPU (or Colab T4 16GB) raise micro_batch (env override below).
import os as _os
micro_batch  = int(_os.environ.get("CHATON_MICRO_BATCH", "8"))
grad_accum   = int(_os.environ.get("CHATON_GRAD_ACCUM", "4"))
batch_size   = micro_batch * grad_accum

# LR schedule: warmup from 0 -> lr_max, then cosine decay -> lr_min.
lr_max       = float(_os.environ.get("CHATON_LR_MAX", "3e-4"))
lr_min       = 3e-5
warmup_iters = 300     # linear warmup steps (kills the early instability)
max_iters    = int(_os.environ.get("CHATON_MAX_ITERS", "8000"))
eval_interval = 250    # how often to measure train+val loss
grad_clip    = 1.0     # clip gradient global norm (stability insurance)

eval_iters   = 50      # batches averaged per eval