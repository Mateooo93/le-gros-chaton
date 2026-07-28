"""FLOP, parameter, and memory profile analyzer for le fat chaton.

Computes the full resource model for any config profile WITHOUT building
the model (no torch needed for allocation numbers).  Provides:

  - Parameter count: total, active (top-k routed), per-component breakdown
  - FLOPs per token: forward FLOPs (training ≈ 2× forward, inference = 1× forward)
  - Memory estimate: model weights, activations, KV cache (per sequence)
  - Throughput estimate: max tokens/sec on common GPUs (A100, A6000, T4)
  - Cost estimate: USD per 1B tokens trained, per 1M tokens served

USAGE
-----
  python profile_analyzer.py                    # current profile
  python profile_analyzer.py --profile fat      # explicit profile
  python profile_analyzer.py --profile dev      # compare
  python profile_analyzer.py --profile smol-fat --gpu A100 --batch 4

OUTPUT
------
  le fat chaton — resource profile
  ════════════════════════════════════════════════
  Profile:  smol-fat
  Params:   2,048.3M total / 770.5M active (37.6%)
  FLOPs:    512.7 GFLOPs/token forward
  Memory:
    Model weights:  7,687 MiB  (FP16)
    Activations:    1,024 MiB  (micro_batch=4, block_size=2048)
    KV cache:       240 MiB    (batch=1, block_size=4096)
  Throughput (A100 80GB):
    Training:       ~42,000 tokens/s  (micro_batch=4, grad_accum=4)
    Inference:      ~168,000 tokens/s
  Cost:
    Training 1B tokens: ~$1.20 (A100, $2/hr)
    Inference 10M tokens: ~$0.02 (A100)
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any


PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# ---------------------------------------------------------------------------
# GPU specs (peak FLOPs for common training GPUs)
# ---------------------------------------------------------------------------

GPU_SPECS: dict[str, dict[str, float]] = {
    # FP16 TFLOPS (tensor core), HBM (GB), HBM bandwidth (GB/s)
    "A100":  {"tflops_fp16": 312.0, "hbm_gb": 80, "bandwidth_gbs": 2039},
    "A100_40": {"tflops_fp16": 312.0, "hbm_gb": 40, "bandwidth_gbs": 1555},
    "A6000": {"tflops_fp16": 151.0, "hbm_gb": 48, "bandwidth_gbs": 768},
    "A5000": {"tflops_fp16": 91.0, "hbm_gb": 24, "bandwidth_gbs": 576},
    "A4000": {"tflops_fp16": 38.0, "hbm_gb": 16, "bandwidth_gbs": 448},
    "RTX4090": {"tflops_fp16": 165.0, "hbm_gb": 24, "bandwidth_gbs": 1008},
    "RTX3090": {"tflops_fp16": 71.0, "hbm_gb": 24, "bandwidth_gbs": 936},
    "RTX6000": {"tflops_fp16": 91.0, "hbm_gb": 48, "bandwidth_gbs": 768},
    "T4":     {"tflops_fp16": 65.0, "hbm_gb": 16, "bandwidth_gbs": 320},
    "V100":   {"tflops_fp16": 125.0, "hbm_gb": 32, "bandwidth_gbs": 900},
    "L4":     {"tflops_fp16": 121.0, "hbm_gb": 24, "bandwidth_gbs": 300},
}


@dataclass
class ProfileResult:
    profile: str = ""
    # Parameters
    total_params: int = 0
    active_params: int = 0
    # Per-component
    embed_params: int = 0
    attn_params: int = 0
    mlp_params: int = 0
    moe_gate_params: int = 0
    routed_expert_params: int = 0
    shared_expert_params: int = 0
    norm_params: int = 0
    n_layer: int = 0
    n_expert: int = 0
    n_expert_top: int = 0
    block_size: int = 0
    micro_batch: int = 0
    # FLOPs
    flops_per_token_forward: int = 0  # FLOPs for one token through one layer
    # Memory
    weights_mib_fp16: float = 0.0
    weights_mib_fp32: float = 0.0
    kv_cache_mib: float = 0.0
    activation_mib: float = 0.0
    # Throughput
    max_batch_training: int = 1
    max_batch_inference: int = 1
    tokens_per_sec_train: float = 0.0
    tokens_per_sec_infer: float = 0.0
    # Cost
    cost_per_1b_tokens_train: float = 0.0
    cost_per_1m_tokens_infer: float = 0.0


# ---------------------------------------------------------------------------
# Core analysis — all arithmetic, no model instantiation needed
# ---------------------------------------------------------------------------

def analyze_profile(profile: str) -> ProfileResult:
    """Compute resource profile from config module attributes.

    This function reads ONLY the config module (no torch, no model building).
    """
    result = ProfileResult(profile=profile)

    # --- Load config ---
    # We need the resolved config values, not the module defaults.
    # Import config with the profile env var set.
    old_profile = os.environ.get("CHATON_PROFILE")
    os.environ["CHATON_PROFILE"] = profile
    # Clear cached imports
    for mod in list(sys.modules.keys()):
        if mod.startswith("config") or mod == "config":
            del sys.modules[mod]
    import config as cfg  # noqa: F811
    # Restore
    if old_profile is None:
        del os.environ["CHATON_PROFILE"]
    else:
        os.environ["CHATON_PROFILE"] = old_profile

    p = result
    v = cfg.vocab_size
    d = cfg.n_embd
    nh = cfg.n_head
    nkv = cfg.n_kv_head
    hd = d // nh  # head_dim
    nl = cfg.n_layer
    ne = cfg.n_expert
    net = cfg.n_expert_top
    nse = getattr(cfg, "n_shared_expert", 0)
    bs = cfg.block_size
    mb = getattr(cfg, "micro_batch", 4)
    ga = getattr(cfg, "grad_accum", 4)
    use_moe = cfg.use_moe
    mlp_type = getattr(cfg, "mlp_type", "swiglu")

    # === Parameter counts ===
    # Embedding: wte, lm_head (tied, so count once)
    embed = v * d
    p.embed_params = embed

    # Attention: Q, K, V projections + output projection
    # Q: d * (nh * hd) = d²
    # KV: d * (2 * nkv * hd) = d * 2 * nkv * hd
    # O: d²
    q_params = d * d  # Q projects d → d (split into nh heads)
    kv_params = d * 2 * nkv * hd  # KV projects d → 2 * kv_dim
    o_params = d * d  # O projects d → d
    attn_per_layer = q_params + kv_params + o_params  # no bias
    p.attn_params = attn_per_layer * nl

    # MLP (dense or expert)
    if mlp_type == "swiglu":
        # SwiGLU: gate (d → d*4/3*2? No — SwiGLU has up, gate, down.
        # Standard: up/gate project to 8/3*d, down projects back.
        # Llama uses: ffn_dim = 8/3 * d, rounded to multiple of 256.
        ffn_dim = int((8 / 3) * d)
        # Round to nearest 256
        ffn_dim = ((ffn_dim + 255) // 256) * 256
        mlp_per_expert = 3 * d * ffn_dim  # up, gate, down
    else:
        # Standard GELU MLP: up (d → 4d), down (4d → d)
        mlp_per_expert = 2 * d * (4 * d)  # up + down

    p.mlp_params = mlp_per_expert  # per-expert (dense = 1 expert)

    if use_moe:
        # MoE gate
        gate_params = d * ne
        p.moe_gate_params = gate_params

        # Routed experts: ne experts per layer, each mlp_per_expert
        routed = mlp_per_expert * ne * nl
        p.routed_expert_params = routed

        # Shared expert(s): nse experts per layer, always active
        shared = mlp_per_expert * nse * nl
        p.shared_expert_params = shared
    else:
        # Dense MLP: one big MLP per layer
        p.mlp_params = mlp_per_expert * nl * nl

    # Normalization: 2 RMSNorm per layer (pre-attn, pre-mlp) + 1 final
    # RMSNorm has d params (weight)
    norm = (2 * nl + 1) * d
    p.norm_params = norm
    p.n_layer = nl
    p.n_expert = ne
    p.n_expert_top = net
    p.block_size = bs
    p.micro_batch = mb

    # Total params (not counting tied lm_head which shares wte weight)
    total = embed + p.attn_params + norm

    if use_moe:
        total += p.moe_gate_params + p.routed_expert_params + p.shared_expert_params
        # Active params per token (forwarded):
        # - Embed: always
        # - Attention: always (all layers)
        # - MoE gate: always
        # - Routed experts: net/ne fraction of routed params per layer
        # - Shared experts: always
        # - Norms: always
        active_per_token = embed + p.attn_params + norm
        active_per_token += p.moe_gate_params
        # Routed fraction: per layer, net/ne of routed expert params
        routed_active = routed * (net / ne)
        active_per_token += routed_active
        active_per_token += p.shared_expert_params
        p.active_params = int(active_per_token)
    else:
        active_per_token = total  # dense: all params are active
        p.active_params = total

    p.total_params = total

    # === FLOPs per token (forward pass) ===
    # Attention: QKV projections (2 * d² per layer), attention score (2*T*d per layer),
    # and output projection (d²).  For analytical simplicity:
    # FLOPs_attn_per_layer ≈ 4 * n_head * d * hd  +  2 * n_head * hd * T
    # But the dominant term for long sequences is the matmuls.
    #
    # Matmul FLOPs = 2 * M * N * K  for (M, K) @ (K, N)
    # QKV proj:  x (B, T, d) @ W_q (d, d)   → 2 * T * d²   per QKV (3× for QKV)
    flops_qkv = 3 * 2 * d * d  # 3 projections (Q, K, V), ×2 for mul+add
    # Attention: Q @ K^T: (head_dim, head_dim) per head per token
    flops_atten = 2 * nh * hd * hd  # self-attention for one token
    # Output proj: out (d,) @ W_o (d, d) → 2 * d²
    flops_out = 2 * d * d

    if use_moe:
        # MoE gate: x @ W_gate → 2 * d * ne
        flops_gate = 2 * d * ne
        # Expert MLP: only net/ne fraction of MLP compute
        # Per expert: 3 * 2 * d * ffn_dim (SwiGLU: up, gate, down)
        flops_expert_per = 3 * 2 * d * ffn_dim
        flops_routed = flops_expert_per * (net / ne)  # fraction per token
        # Shared expert(s)
        flops_shared = flops_expert_per * nse if nse > 0 else 0
        flops_ff = flops_gate + flops_routed + flops_shared
    else:
        flops_ff = 2 * 2 * d * (4 * d)  # dense GELU MLP: up + down

    flops_per_layer = flops_qkv + flops_atten + flops_out + flops_ff
    p.flops_per_token_forward = int(flops_per_layer * nl)

    # === Memory estimates ===
    # Weights in FP16: params * 2 bytes
    p.weights_mib_fp16 = total * 2 / (1024 * 1024)
    # Weights in FP32 (optimizer): params * 4 bytes (master copy)
    p.weights_mib_fp32 = total * 4 / (1024 * 1024)
    # AdamW optimizer states: 2 states per param (mom, var) = 8 bytes/param
    opt_mib = total * 8 / (1024 * 1024)

    # KV cache: 2 * n_layers * n_kv_heads * head_dim * 2 bytes * block_size
    # (2 for K and V, 2 bytes for FP16)
    kv_per_token_per_layer = 2 * nkv * hd * 2  # bytes for K+V for one token
    p.kv_cache_mib = kv_per_token_per_layer * nl * bs / (1024 * 1024)

    # Activations (rough estimate): ~4 * n_embd * block_size * batch_size * bytes
    # A more accurate estimate for transformer training:
    # Each layer stores: (hidden, attn_out, mlp_out) → ~3 * d * T * batch_size
    # Mixed precision: some in FP16, some in FP32
    act_bytes = 3 * d * bs * mb * 2  # FP16
    # Plus attention scores (n_head * T * T): dominant for long sequences
    act_score_bytes = nh * bs * bs  # per layer attention scores
    p.activation_mib = (act_bytes * nl + act_score_bytes) / (1024 * 1024)

    # === GPU memory needed ===
    # The checkpoint load: weights (FP16) + optimizer states (FP32) + grad (FP16)
    gpu_mem_needed_mib = (
        p.weights_mib_fp16  # model weights
        + p.weights_mib_fp32  # master copy
        + opt_mib  # optimizer states
        + p.weights_mib_fp16  # gradients (same size as weights)
        + p.kv_cache_mib * mb  # KV cache for batch
        + p.activation_mib  # activations
    )
    gpu_mem_needed_gb = gpu_mem_needed_mib / 1024

    # === Max batch size ===
    gpu_specs_to_check = [("A100", 80.0), ("RTX4090", 24.0), ("T4", 16.0)]

    for gpu_name, hbm_gb in gpu_specs_to_check:
        overhead_mib = 2 * 1024  # 2GB for framework overhead
        available = hbm_gb * 1024 - overhead_mib
        # Max batch for training = available / (per-sample memory)
        # Per-sample = (weights + opt + grad) / batch + KV_cache + activations/batch
        per_sample_train = (
            (p.weights_mib_fp16 + p.weights_mib_fp32 + opt_mib) / mb
            + p.kv_cache_mib
            + p.activation_mib / mb
        )
        max_b_train = max(1, int(available / max(1, per_sample_train)))
        # Max batch for inference = available / (weights + KV_cache)
        per_sample_infer = (
            p.weights_mib_fp16  # FP16 weights
            + p.kv_cache_mib  # KV cache per sequence
        )
        max_b_infer = max(1, int((available - p.weights_mib_fp16) / max(1, p.kv_cache_mib)))

        # Token throughput
        gpu_spec = GPU_SPECS.get(gpu_name, {"tflops_fp16": 312.0, "hbm_gb": 80, "bandwidth_gbs": 2000})
        tflops = gpu_spec["tflops_fp16"]
        # FLOPs per token forward+backward ≈ 3× forward (2× forward for backward + 1× forward from recomputation)
        # With activation checkpointing: forward ≈ 1.3×, backward ≈ 2×
        flops_per_token_train = p.flops_per_token_forward * 3.5  # fwd + bwd + overhead
        flops_per_token_infer = p.flops_per_token_forward  # fwd only

        # Theoretical max tokens/s = GPU FLOPs / FLOPs per token
        # Realistic: 40% MFU for training, 60% for inference
        tokens_s_train = (tflops * 1e12 * 0.4) / flops_per_token_train
        tokens_s_infer = (tflops * 1e12 * 0.6) / flops_per_token_infer

        # Also bound by memory bandwidth
        weights_bytes_per_token = total * 2 / tokens_s_train if tokens_s_train > 0 else float("inf")
        # Budget memory bandwidth bound

        # Cost
        # Typical GPU rental (per hour)
        gpu_hourly = {"A100": 2.00, "RTX4090": 0.50, "T4": 0.30}.get(gpu_name, 1.00)
        tokens_per_batch_train = max_b_train * bs
        tokens_per_hour_train = tokens_s_train * 3600
        cost_per_1b = (1e9 / tokens_per_hour_train) * gpu_hourly if tokens_per_hour_train > 0 else 0
        cost_per_1m_infer = (1e6 / tokens_s_infer / 3600) * gpu_hourly if tokens_s_infer > 0 else 0

        print(
            f"  {gpu_name:>10} ({hbm_gb:.0f}GB): "
            f"train batch={max_b_train}, "
            f"{tokens_s_train/1000:.0f}K tok/s, "
            f"${cost_per_1b:.2f}/B tok; "
            f"infer batch={max_b_infer}, "
            f"{tokens_s_infer/1000:.0f}K tok/s"
        )

    return result


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_profile(r: ProfileResult):
    """Pretty-print the resource profile."""
    def fmt(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        if n >= 1e6:
            return f"{n/1e6:.1f}M"
        if n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(n)

    def fmt_mib(n):
        if n >= 1024:
            return f"{n/1024:.1f} GB"
        return f"{n:.0f} MiB"

    def fmt_flops(n):
        if n >= 1e12:
            return f"{n/1e12:.1f} TFLOPs"
        if n >= 1e9:
            return f"{n/1e9:.2f} GFLOPs"
        return f"{n/1e6:.0f} MFLOPs"

    total_active_ratio = (r.active_params / max(1, r.total_params)) * 100

    print()
    print("  le fat chaton — resource profile")
    print("  ════════════════════════════════════════════════")
    print(f"  Profile:  {r.profile}")
    print(f"  Params:   {fmt(r.total_params)} total / {fmt(r.active_params)} active "
          f"({total_active_ratio:.1f}%)")
    print(f"  FLOPs:    {fmt_flops(r.flops_per_token_forward)}/token forward")
    print()
    print("  Parameter breakdown:")
    print(f"    Embedding:      {fmt(r.embed_params)}")
    print(f"    Attention:      {fmt(r.attn_params)}  ({r.n_layer} layers × QKV+O)")
    if r.moe_gate_params:
        print(f"    MoE gate:       {fmt(r.moe_gate_params)}")
        print(f"    Routed experts: {fmt(r.routed_expert_params)}  ({r.n_expert} experts, {r.n_expert_top}-top)")
        print(f"    Shared experts: {fmt(r.shared_expert_params)}")
    else:
        print(f"    MLP (dense):    {fmt(r.mlp_params)}")
    print(f"    Norms:          {fmt(r.norm_params)}  (RMSNorm)")
    print()
    print("  Memory (FP16 training):")
    print(f"    Model weights:  {fmt_mib(r.weights_mib_fp16)}")
    print(f"    Optimiser:      {fmt_mib(r.weights_mib_fp32 * 2)}  (FP32 master + Adam states)")
    print(f"    KV cache:       {fmt_mib(r.kv_cache_mib)}  (batch=1, block_size={r.block_size})")
    gpu_needed = (r.weights_mib_fp16 + r.weights_mib_fp32 + r.weights_mib_fp32 * 2
                  + r.kv_cache_mib + r.activation_mib) / 1024
    print(f"    Estimated GPU:  {gpu_needed:.1f} GB  (micro_batch={r.micro_batch})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FLOP and memory profiler for le fat chaton"
    )
    parser.add_argument("--profile", default=None,
                        help="Profile to analyze (default: current env or 'fat')")
    parser.add_argument("--gpu", default=None,
                        help="GPU for throughput estimate (e.g., A100, RTX4090, T4)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Micro batch size for memory estimate")
    parser.add_argument("--list-profiles", action="store_true",
                        help="Show available profiles and exit")

    args = parser.parse_args()

    if args.list_profiles:
        # Show built-in profiles from config.py
        profiles = ["dev", "smol-fat", "fat"]
        print("Available profiles:")
        for p in profiles:
            os.environ["CHATON_PROFILE"] = p
            # Clear cached config
            for mod in list(sys.modules.keys()):
                if mod.startswith("config") or mod == "config":
                    del sys.modules[mod]
            import config as cfg
            print(f"  {p:>10}: n_embd={cfg.n_embd}, n_layer={cfg.n_layer}, "
                  f"n_head={cfg.n_head}, n_kv_head={cfg.n_kv_head}, "
                  f"use_moe={cfg.use_moe}, "
                  f"n_expert={cfg.n_expert}/{cfg.n_expert_top} "
                  f"({'dense' if not cfg.use_moe else f'MoE {cfg.n_expert}x{cfg.n_expert_top}'})")
        return

    profile = args.profile or os.environ.get("CHATON_PROFILE", "fat")
    print(f"Analyzing profile: {profile}")
    r = analyze_profile(profile)
    print_profile(r)


if __name__ == "__main__":
    main()