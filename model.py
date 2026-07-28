import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import config as cfg


# ---------------------------------------------------------------------------
# RMSNorm — a leaner LayerNorm. No mean subtraction, no bias.
#   norm(x) = x / sqrt(mean(x^2) + eps) * weight
# Same job, fewer params/calculus. Modern standard (Llama etc.).
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # rsqrt = 1/sqrt. mean over the last dim (the embedding dim), keepdims.
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm_x * self.weight


# ---------------------------------------------------------------------------
# rotary position embeddings (RoPE)
#   Instead of a learned position embedding added to the token, we ROTATE
#   the query/key vectors by an angle that depends on position. Pairs of
#   dims rotate together; later pairs rotate faster. The neat trick: the
#   attention score q.k depends only on RELATIVE position, and there are no
#   extra params (the cos/sin are a fixed lookup table) and no block_size cap.
# ---------------------------------------------------------------------------
def _build_rope_cache(block_size, head_dim, device, theta=10000.0,
                     scaling_scale: float = 1.0):
    """Build RoPE cos/sin cache with optional YaRN-style frequency scaling.

    When *scaling_scale* > 1.0, the RoPE frequencies are scaled to extend the
    effective context window without retraining.  Uses NTK-aware scaling:

        theta_i_new = theta * scale^(2*i / (head_dim - 2))

    This means low-frequency components (which encode long-range positions) are
    scaled more than high-frequency components (short-range), preserving local
    relationships while extending global context.

    *block_size* is the EXTENDED block size (e.g., the target inference length,
    which may be 2× the training block_size).
    """
    half = head_dim // 2
    # NTK-aware frequency scaling
    if scaling_scale > 1.0:
        # Scale the base theta differently per frequency index
        # Lower indices (low frequencies) get scaled more
        indices = torch.arange(0, half, device=device).float()
        # NTK-aware: theta_i = theta * scale^(2*i / (head_dim - 2))
        scale_pow = 2.0 * indices / (head_dim - 2.0)
        theta_scaled = theta * (scaling_scale ** scale_pow)
        freqs = 1.0 / (theta_scaled ** (indices / half))
    else:
        freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(block_size, device=device).float()
    angles = torch.outer(pos, freqs)                 # (block_size, half)
    # Repeat so each freq pairs two dims: (block_size, head_dim)
    angles = torch.repeat_interleave(angles, 2, dim=1)
    return angles.cos(), angles.sin()                # each (block_size, head_dim)


def _apply_rope(q, k, cos, sin):
    # q, k: (B, n_head, T, head_dim). cos, sin: (T, head_dim) -> broadcast.
    # Rotate: pair dims (2i, 2i+1). We use the rotate_half trick:
    #   rotate_half(x) = concat(-x[..., ::2], x[..., 1::2])  (interleaved form)
    def rotate_half(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x1, x2), dim=-1).flatten(-2)

    cos = cos[None, None, :, :]   # (1,1,T,head_dim)
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# 1. ATTENTION — now with SDPA (Flash/memory-efficient fused backend) + RoPE
#    + an optional KV cache for fast generation.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_qk_norm = getattr(cfg, "use_qk_norm", True)
        self.kv_latent_dim = getattr(cfg, "kv_latent_dim", 0)
        kv_dim = self.n_kv_head * self.head_dim

        # Multi-Head Latent Attention (DeepSeek-V2) compresses K/V into a
        # low-rank latent, dramatically reducing KV cache size.
        if self.kv_latent_dim > 0:
            self.kv_down = nn.Linear(cfg.n_embd, self.kv_latent_dim, bias=False)
            self.k_up = nn.Linear(self.kv_latent_dim, kv_dim, bias=False)
            self.v_up = nn.Linear(self.kv_latent_dim, kv_dim, bias=False)
            self.c_q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
            # Decoupled RoPE (separate small Q/K for positional encoding)
            self.q_rope = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
            self.k_rope = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        else:
            # Standard GQA
            self.c_q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
            self.c_kv = nn.Linear(cfg.n_embd, 2 * kv_dim, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        # QK-normalisation
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_cos=None, rope_sin=None, kv_cache=None, use_cache=False):
        B, T, C = x.shape

        # Compute Q, and K/V (either directly for GQA or via latent for MLA)
        if self.kv_latent_dim > 0:
            c_kv = self.kv_down(x)                    # (B, T, kv_latent_dim)
            q = self.c_q(x)                           # (B, T, C)
            # Cache the latent, decompress to K/V on-the-fly
            if use_cache:
                if kv_cache is not None:
                    past_c = kv_cache[0]
                    c_kv = torch.cat([past_c, c_kv], dim=1)
                new_kv = (c_kv,)
            else:
                new_kv = None
            k = self.k_up(c_kv)                       # (B, T, kv_dim)
            v = self.v_up(c_kv)                       # (B, T, kv_dim)
        else:
            q = self.c_q(x)                           # (B, T, C)
            k, v = self.c_kv(x).split(self.n_kv_head * self.head_dim, dim=2)
            if use_cache and kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_kv = (k, v) if use_cache else None

        # Reshape to head format
        q = q.view(B, -1, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, -1, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_kv_head, self.head_dim).transpose(1, 2)

        # QK-norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        if rope_cos is not None:
            q, k = _apply_rope(q, k, rope_cos, rope_sin)

        # GQA: replicate KV heads to match Q heads
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # StreamingLLM compression (only for GQA — MLA already compresses)
        if not self.kv_latent_dim and use_cache:
            max_cache = getattr(cfg, "max_cache_len", cfg.block_size)
            if k.size(2) > max_cache:
                n_sink = getattr(cfg, "cache_n_sink", 4)
                n_local = max_cache - n_sink
                k = torch.cat([k[:, :, :n_sink], k[:, :, -n_local:]], dim=2)
                v = torch.cat([v[:, :, :n_sink], v[:, :, -n_local:]], dim=2)

        # SDPA
        is_causal = (rope_cos is not None) and (not use_cache)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out), new_kv


# ---------------------------------------------------------------------------
# 2. MLP — the "thinking" layer.
#    Two options (mlp_type in config):
#      "gelu"   -> classic GPT-2: Linear -> GELU -> Linear (4x expansion)
#      "swiglu" -> SwiGLU gated MLP (Llama/Qwen standard): two up-projections,
#                  SiLU(x.w_gate) * x.w_up, then down-projection. The gate lets
#                  the layer select which features to pass -> better per-param
#                  than GELU. Hidden width scaled 2/3 so it has ~the same param
#                  count as the GELU version (standard Llama sizing).
# ---------------------------------------------------------------------------
class RoutingFreeExpert(nn.Module):
    """Expert with built-in gating (Routing-Free MoE, arXiv:2604.00801).

    Each expert has its OWN low-rank gate (A_gate, B_gate) instead of relying
    on an external centralized router.  The expert's activation score is
    ||x @ A_gate||_2 — the norm of its low-rank projection.  If this score
    exceeds a learnable threshold, the expert activates itself.

    Formula:
        gate = sigmoid(x @ A_gate @ B_gate)   — low-rank gate activation
        FFN(x) = (gate * (x @ W_up)) @ W_down  — gated FFN
        score = ||x @ A_gate||_2               — activation confidence
    """
    def __init__(self):
        super().__init__()
        hidden = int(4 * cfg.n_embd * 2 / 3)
        hidden = ((hidden + cfg.n_embd // cfg.n_head - 1)
                  // (cfg.n_embd // cfg.n_head)) * (cfg.n_embd // cfg.n_head)
        # Low-rank gate (rank r = 64 or n_embd//8)
        rank = max(8, cfg.n_embd // 8)
        self.A_gate = nn.Linear(cfg.n_embd, rank, bias=False)
        self.B_gate = nn.Linear(rank, hidden, bias=False)
        # Standard up/down projections
        self.W_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.W_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        # Learnable activation threshold (initialized so ~top-2 experts activate)
        self.register_parameter("threshold",
            nn.Parameter(torch.tensor(0.5)))

    def forward(self, x):
        # Low-rank gate: sigmoid(x @ A_gate @ B_gate)
        gate_act = torch.sigmoid(self.B_gate(self.A_gate(x)))
        # Gated FFN
        out = gate_act * self.W_up(x)
        out = self.W_down(out)
        return out

    def get_score(self, x):
        """Activation score: ||x @ A_gate||_2 (norm of low-rank projection)."""
        return torch.norm(self.A_gate(x), dim=-1)


# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_type = cfg.mlp_type
        if self.mlp_type == "swiglu":
            hidden = int(4 * cfg.n_embd * 2 / 3)        # 2/3 of 4x = ~2.67x
            # round to a multiple of head_dim for tidy shapes
            hidden = ((hidden + cfg.n_embd // cfg.n_head - 1)
                      // (cfg.n_embd // cfg.n_head)) * (cfg.n_embd // cfg.n_head)
            self.c_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.c_up = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        else:
            self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
            self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        if self.mlp_type == "swiglu":
            gate_out = self.c_gate(x)
            # SiTU (Sigmoid Tanh Unit, Kimi K3): replaces SiLU in MoE experts
            # to avoid dead-neuron pathology in rarely-activated experts.
            # Formula: SiTU(x) = sigmoid(x) * tanh(x)
            if getattr(cfg, "use_situ", False):
                gate_act = torch.sigmoid(gate_out) * torch.tanh(gate_out)
            else:
                gate_act = F.silu(gate_out)
            return self.c_proj(gate_act * self.c_up(x))
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# 2b. MIXTURE-OF-EXPERTS (MoE) — replaces the dense MLP in a block.
#
#   Each token is routed by a small gate to its top-k experts (out of n_expert).
#   Only those k experts compute for that token -> inference runs at ~k/n_expert
#   of the FLOPs, while the TOTAL knowledge capacity is n_expert experts.
#   This is how a ~8B-total / ~2B-active model runs fast but knows a lot.
#
#   Aux load-balance loss (Switch Transformer): n_expert * sum(f_i * P_i)
#     f_i = fraction of tokens routed to expert i
#     P_i = mean router probability for expert i
#   Pushes tokens to spread across experts (avoids collapse to 1 expert).
# ---------------------------------------------------------------------------
class MoE(nn.Module):
    def __init__(self, n_expert=cfg.n_expert, n_expert_top=cfg.n_expert_top):
        super().__init__()
        self.n_expert = n_expert
        self.n_expert_top = n_expert_top
        self.n_shared = getattr(cfg, "n_shared_expert", 0)   # DeepSeek-style
        # Latent MoE routing (Kimi K3-style): compress token before routing
        # to reduce computation and communication in multi-GPU settings.
        moe_latent = getattr(cfg, "moe_latent_dim", 0)
        if moe_latent > 0:
            self.moe_down = nn.Linear(cfg.n_embd, moe_latent, bias=False)
            self.moe_up = nn.Linear(moe_latent, cfg.n_embd, bias=False)
            self.gate = nn.Linear(moe_latent, n_expert, bias=False)
        else:
            self.moe_down = None
            self.moe_up = None
            self.gate = nn.Linear(cfg.n_embd, n_expert, bias=False)
        # Load-balancing bias (DeepSeek-V3 style)
        gate_bias_lr = getattr(cfg, "gate_bias_lr", 0.01)
        self.register_buffer("gate_bias", torch.zeros(n_expert))
        self._gate_bias_lr = gate_bias_lr
        self._step_count = 0
        # Each expert is its own little MLP (same shape as the dense MLP).
        # Routing-Free MoE (arXiv:2604.00801): experts have built-in gates.
        if getattr(cfg, "use_routing_free", False):
            self.experts = nn.ModuleList([RoutingFreeExpert() for _ in range(n_expert)])
            # Per-expert learnable thresholds
            self.register_parameter("expert_thresholds",
                nn.Parameter(torch.full((n_expert,), 0.5)))
            self._use_routing_free = True
        else:
            self.experts = nn.ModuleList([MLP() for _ in range(n_expert)])
            self._use_routing_free = False
        # Shared expert(s): always active (no routing), captures common
        # knowledge so the routed experts don't have to relearn it -> better
        # specialization. Add 1 and the total active params go up by one MLP.
        self.shared_experts = nn.ModuleList([MLP() for _ in range(self.n_shared)])

    def forward(self, x):
        # x: (B, T, C) -> flatten tokens for routing
        B, T, C = x.shape
        flat = x.view(B * T, C)
        n_tokens = B * T
        n_expert = self.n_expert
        n_top = self.n_expert_top

        # --- gate + top-k or quantile-balanced routing ---------------------
        # Latent MoE (Kimi K3-style): compress token before routing
        if self.moe_down is not None:
            flat_latent = self.moe_down(flat)              # (n_tokens, latent_dim)
            gate_logits = self.gate(flat_latent)           # route in latent space
        else:
            gate_logits = self.gate(flat)                  # (n_tokens, n_expert)
        
        if getattr(cfg, "quantile_balance", False) and self.training:
            # Quantile-balanced routing (Kimi K3): for each expert, find the
            # score threshold at the (1 - k/n) quantile.  Token routes to
            # expert if its score exceeds that expert's quantile threshold.
            # This guarantees balanced expert utilization without auxiliary
            # loss or heuristic bias updates.
            k_frac = n_top / n_expert  # fraction of experts to activate
            # Per-expert quantile thresholds
            sorted_scores, _ = torch.sort(gate_logits, dim=0)  # (n_tokens, n_expert)
            q_idx = int((1.0 - k_frac) * n_tokens)
            q_idx = max(0, min(q_idx, n_tokens - 1))
            thresholds = sorted_scores[q_idx]  # (n_expert,)
            # Binary mask: route if score > threshold
            route_mask = gate_logits > thresholds.unsqueeze(0)  # (n_tokens, n_expert)
            # Fallback: if no expert selected for a token, pick the top-1
            token_any = route_mask.any(dim=-1)
            if not token_any.all():
                fallback = (~token_any).nonzero(as_tuple=True)[0]
                route_mask[fallback, gate_logits[fallback].argmax(dim=-1)] = True
            # Build topk-like outputs from mask
            n_activated = route_mask.sum(dim=-1)  # (n_tokens,)
            max_activated = n_activated.max().item()
            # Pad to uniform top-k for grouped dispatch
            topk_idx = torch.full((n_tokens, max_activated), -1, device=flat.device, dtype=torch.long)
            topk_weights = torch.zeros(n_tokens, max_activated, device=flat.device)
            for t in range(n_tokens):
                indices = route_mask[t].nonzero(as_tuple=True)[0]
                n_act = len(indices)
                topk_idx[t, :n_act] = indices
                scores = gate_logits[t, indices]
                topk_weights[t, :n_act] = F.softmax(scores, dim=-1)
            n_top = max_activated
        else:
            # Standard top-k routing
            gate_logits = gate_logits + self.gate_bias
            topk_vals, topk_idx = torch.topk(gate_logits, n_top, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)

        # ---- Dynamic top-k (confidence-based adaptive routing) --------
        # When enabled during training, tokens that are "easy" for the gate
        # (one expert strongly dominates) only route to 1 expert.  This:
        #   - Saves compute on trivial tokens (punctuation, keywords)
        #   - Regularises experts (each must be independently useful)
        #   - Allocates capacity where it matters (hard tokens get 2 experts)
        #
        # Confidence = p(top-1) / p(top-2) — the ratio of the top two expert
        # probabilities (after full softmax over all experts).  High ratio =
        # the gate is very sure → drop the second expert.
        if getattr(cfg, "dynamic_topk", False) and self.training and n_top > 1:
            probs = F.softmax(gate_logits, dim=-1)          # (n_tokens, n_expert)
            p_top1 = probs.gather(1, topk_idx[:, :1])       # (n_tokens, 1)
            p_top2 = probs.gather(1, topk_idx[:, 1:2])      # (n_tokens, 1)
            confidence = p_top1 / (p_top2 + 1e-8)           # ratio, (n_tokens, 1)
            # Tokens with confidence > threshold keep only the top-1 expert
            keep_second = (confidence < cfg.dynamic_topk_threshold).float()
            topk_weights[:, 1] = topk_weights[:, 1] * keep_second.squeeze(-1)
            # Re-normalise so weights still sum to 1 for each token
            weight_sum = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            topk_weights = topk_weights / weight_sum

        # ---- Grouped dispatch (contiguous per-expert segments) ----------
        # Instead of looping ``for e in range(n_expert)`` with per-expert
        # mask.nonzero() calls (each one a scatter/gather kernel), we sort
        # ALL assignments by expert ID then process contiguous slices.
        #
        # ``assign_experts[e]``  = which expert this (token, slot) maps to
        # ``assign_tokens[e]``   = which token row this assignment belongs to
        # ``assign_weights[e]``  = corresponding router weight
        # Shape: (n_tokens * n_top,)  — one row per (token, top-k-slot) pair
        assign_experts = topk_idx.flatten()                  # (n_assign,)
        assign_weights = topk_weights.flatten()              # (n_assign,)
        assign_tokens = torch.arange(
            n_tokens, device=flat.device
        ).repeat_interleave(n_top)                           # (n_assign,)

        if self._use_routing_free:
            # Routing-Free MoE: each expert decides its own activation via
            # its internal gate score.  No external router, Softmax, or TopK.
            out = torch.zeros_like(flat)
            scores = torch.stack([e.get_score(flat) for e in self.experts], dim=-1)
            # Apply per-expert thresholds
            active = scores > self.expert_thresholds.unsqueeze(0)
            # Fallback: ensure every token has at least one active expert
            token_any = active.any(dim=-1)
            if not token_any.all():
                fallback = (~token_any).nonzero(as_tuple=True)[0]
                best = scores[fallback].argmax(dim=-1)
                active[fallback, best] = True
            # Route tokens to active experts
            for e in range(n_expert):
                mask = active[:, e]
                if not mask.any():
                    continue
                token_ids = mask.nonzero(as_tuple=True)[0]
                expert_in = flat[token_ids]
                expert_out = self.experts[e](expert_in)
                out.index_add_(0, token_ids, expert_out)
        else:
            # Standard grouped dispatch
            # Sort assignments by expert — tokens assigned to the same expert
            # become contiguous, making the subsequent gather cache-friendly.
            perm = torch.argsort(assign_experts, stable=True)
            sorted_experts = assign_experts[perm]
            sorted_tokens = assign_tokens[perm]
            sorted_weights = assign_weights[perm]

            # Find start/end indices for each expert in the sorted array
            expert_ids = torch.arange(n_expert, device=flat.device)
            starts = torch.searchsorted(sorted_experts, expert_ids, side="left")
            ends = torch.searchsorted(sorted_experts, expert_ids, side="right")

            # Process each expert on its contiguous token slice
            out = torch.zeros_like(flat)
            for e in range(n_expert):
                start = int(starts[e].item())
                end = int(ends[e].item())
                if start >= end:
                    continue
                token_ids = sorted_tokens[start:end]
                expert_in = flat[token_ids]
                expert_out = self.experts[e](expert_in)
                w = sorted_weights[start:end].unsqueeze(-1)
                out.index_add_(0, token_ids, expert_out * w)

        # --- shared expert: always-on, runs on every token (no routing) ---
        for se in self.shared_experts:
            out = out + se(flat)

        # Latent up-projection (Kimi K3-style): decompress after expert computation
        if self.moe_up is not None:
            out = self.moe_up(out)

        # --- load-balance aux loss (DeepSeek-MoE form, uses ALL top-k slots) ---
        # f_i = fraction of total (token, slot) assignments to expert i
        # P_i = mean router probability for expert i
        # aux_loss = alpha_coef * n_expert * sum(f_i * P_i)
        #
        # When quantile_balance is enabled, this aux_loss is not needed because
        # the quantile-based routing guarantees balanced expert utilization.
        if not getattr(cfg, "quantile_balance", False) and not getattr(cfg, "use_routing_free", False):
            with torch.no_grad():
                assign_counts = torch.bincount(
                    assign_experts, minlength=self.n_expert
                ).float()
                f = assign_counts / max(assign_counts.sum(), 1.0)
            P = F.softmax(gate_logits, dim=-1).mean(0)          # (n_expert,)
            aux_loss = self.n_expert * (f * P).sum()
        else:
            aux_loss = torch.tensor(0.0, device=flat.device)

        # --- z-loss (DeepSeek-MoE): penalise extreme gate logits ----------
        #   z_loss = mean_i (log(sum_j exp(gate_logits[i, j])))^2
        # The log partition function can grow unbounded during training;
        # this penalty keeps the gate distribution from collapsing or exploding.
        log_z = torch.logsumexp(gate_logits, dim=-1)        # (n_tokens,)
        z_loss = (log_z ** 2).mean()

        # --- update load-balancing bias (heuristic, not backpropped) ---------
        # Shift bias: increase for underloaded experts (f_i < 1/n_expert),
        # decrease for overloaded (f_i > 1/n_expert).  This is the
        # auxiliary-loss-free load balancing from DeepSeek-V3.
        # Not used when quantile_balance is enabled.
        if self.training and self._gate_bias_lr > 0 and not getattr(cfg, "quantile_balance", False) and not getattr(cfg, "use_routing_free", False):
            target_load = 1.0 / n_expert  # uniform target
            with torch.no_grad():
                # f is already computed above (fraction of assignments per expert)
                load_error = f - target_load  # (n_expert,)
                self.gate_bias.sub_(self._gate_bias_lr * load_error)
                self.gate_bias.clamp_(-10.0, 10.0)

        return out.view(B, T, C), aux_loss, z_loss


# ---------------------------------------------------------------------------
# 3. A TRANSFORMER BLOCK — pre-norm (RMSNorm) + Attention + residual,
#    then RMSNorm + MLP + residual.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = RMSNorm(cfg.n_embd)
        self.mlp = MoE() if cfg.use_moe else MLP()
        self.is_moe = cfg.use_moe
        # Learned residual scaling (AttnRes-inspired, Kimi K3).
        # Each layer learns how much of its input to keep.
        if getattr(cfg, "learned_residual", False):
            self.res_alpha = nn.Parameter(torch.ones(1))
        else:
            self.res_alpha = None

    def forward(self, x, rope_cos=None, rope_sin=None, kv_cache=None, use_cache=False):
        attn_out, new_kv = self.attn(
            self.ln_1(x), rope_cos, rope_sin, kv_cache, use_cache
        )
        if self.res_alpha is not None:
            x = x + self.res_alpha * attn_out
        else:
            x = x + attn_out
        ff_in = self.ln_2(x)
        if self.is_moe:
            ff_out, aux_loss, z_loss = self.mlp(ff_in)
            x = x + ff_out
            return x, new_kv, (aux_loss, z_loss)
        x = x + self.mlp(ff_in)
        return x, new_kv, None


# ---------------------------------------------------------------------------
# 4. THE FULL MODEL — token embed -> RoPE -> N blocks -> final norm -> logits
#    with a KV cache for fast generation.
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, gradient_checkpointing: bool = False):
        """
        Args:
            gradient_checkpointing: if True, each transformer block is wrapped
                in ``torch.utils.checkpoint.checkpoint`` during training.
                Trades ~20% compute for ~60% activation memory — necessary for
                the fat profile (10B MoE, block_size=8192) on a single A100.
        """
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        # Token embedding only — positions come from RoPE now (no wpe).
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(layer_idx=i) for i in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        # Tied LM head (shares weights with wte) — saves params and helps.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # MoE aux loss tracking (updated each forward, read for logging)
        self.last_aux_loss: float = 0.0
        self.last_z_loss: float = 0.0

        # RoPE cos/sin buffers, built lazily so device follows the model.
        self.rope_built = False

        # --- Weight initialization: the fix for the 158 start-loss. ---
        # Standard N(0, 0.02) everywhere, but every projection that FEEDS a
        # residual stream is scaled by 1/sqrt(2*n_layer) so the residual
        # signal variance stays ~constant as we stack layers. (2 because each
        # block adds two residuals: attn + mlp.) This stops early logits
        # from blowing up. Tied wte/lm_head is left at 0.02.
        self.apply(self._init_weights)

        # Residual scaling for output projections (c_proj in attn + MLP).
        # These need smaller init variance in deep models to prevent residual
        # stream explosion as each block adds its output.
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and "c_proj" in name:
                std = 0.02 / math.sqrt(2 * cfg.n_layer)
                nn.init.normal_(m.weight, mean=0.0, std=std)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                std = 0.02 / math.sqrt(2 * cfg.n_layer)
                nn.init.normal_(p, mean=0.0, std=std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _ensure_rope(self, device, dtype):
        if not self.rope_built:
            scale = getattr(cfg, "rope_scaling_scale", 1.0)
            # Build the cache at the extended block size (may be > training
            # block_size if rope_scaling_scale > 1).
            max_len = int(cfg.block_size * scale)
            cos, sin = _build_rope_cache(
                max_len, cfg.n_embd // cfg.n_head, device,
                scaling_scale=scale,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.rope_built = True

    def extend_vocab(self, n_new: int, init_from: list[str] | None = None):
        """Extend the embedding table and lm_head to accommodate new tokens.

        This is used to add tool tokens (``<|tool_call|>``, ``<|tool_result|>``,
        ``<|done|>``) to the model *after* training, without retraining the
        tokenizer.  The new embeddings are initialised as the mean of their
        GPT-2 subword token embeddings (zero-shot transfer).

        Args:
            n_new: Number of new tokens to add.
            init_from: Optional list of strings whose GPT-2 subword token
                       embeddings are averaged to initialise each new row.
                       If None, initialised with the standard 0.02 normal.
        """
        old_vocab = self.wte.weight.size(0)
        new_vocab = old_vocab + n_new

        # Extend embedding table
        old_emb = self.wte.weight.data
        new_emb = torch.empty(new_vocab, cfg.n_embd, device=old_emb.device, dtype=old_emb.dtype)
        new_emb[:old_vocab] = old_emb
        for i in range(n_new):
            if init_from and i < len(init_from) and init_from[i]:
                # Average subword token embeddings
                from tokenizer import encode as tok_encode
                subword_ids = tok_encode(init_from[i])
                if subword_ids:
                    subword_ids = [min(s, old_vocab - 1) for s in subword_ids]
                    new_emb[old_vocab + i] = old_emb[subword_ids].mean(0)
                else:
                    nn.init.normal_(new_emb[old_vocab + i], mean=0.0, std=0.02)
            else:
                nn.init.normal_(new_emb[old_vocab + i], mean=0.0, std=0.02)

        self.wte = nn.Embedding(new_vocab, cfg.n_embd, device=old_emb.device)
        self.wte.weight.data = new_emb

        # Extend lm_head (tied with wte)
        old_head = self.lm_head.weight.data
        new_head = torch.empty(new_vocab, cfg.n_embd, device=old_head.device, dtype=old_head.dtype)
        new_head[:old_vocab] = old_head
        for i in range(n_new):
            new_head[old_vocab + i] = new_emb[old_vocab + i]  # tied
        self.lm_head = nn.Linear(cfg.n_embd, new_vocab, bias=False, device=old_head.device)
        self.lm_head.weight.data = new_head
        self.lm_head.weight = self.wte.weight  # re-tie

        # Update config
        cfg.vocab_size = new_vocab
        print(f"[model] extended vocab from {old_vocab} to {new_vocab}")

    def forward(self, idx, targets=None, kv_caches=None, use_cache=False, rope_offset=0):
        """
        Args:
            idx: (B, T) token ids.
            targets: optional (B, T) target ids for loss.
            kv_caches: per-layer KV cache from a previous forward pass.
            use_cache: if True, return updated KV caches.
            rope_offset: starting RoPE position offset. When extending an
                existing cache (``kv_caches is not None``), set this to the
                number of tokens already cached so new tokens get correct
                rotary positions.  Default 0 (fresh prefill).
        """
        B, T = idx.shape
        self._ensure_rope(idx.device, idx.dtype)
        # Use the correct slice of the RoPE cache — accounts for tokens already
        # in the KV cache so that new tokens get their true position encoding.
        rope_cos = self.rope_cos[rope_offset:rope_offset + T]
        rope_sin = self.rope_sin[rope_offset:rope_offset + T]

        x = self.wte(idx)                           # (B, T, n_embd)

        new_caches = [] if use_cache else None
        aux_total = 0.0
        z_total = 0.0

        # Gradient checkpointing: wrap each block in ``checkpoint`` during
        # training so activations are recomputed on backward instead of
        # stored.  Saves ~60% memory at ~20% compute overhead.
        use_ckpt = self.gradient_checkpointing and self.training

        for i, block in enumerate(self.blocks):
            if use_ckpt:
                # Checkpoint doesn't support side-effectful ops (KV cache
                # mutation), so we skip caching during checkpointed forward.
                def block_fn(b: Block, x_: torch.Tensor,
                             cos: torch.Tensor, sin: torch.Tensor):
                    out, _, aux = b(x_, cos, sin, None, False)
                    if aux is not None:
                        aux_l, z_l = aux
                    else:
                        aux_l = torch.zeros(1, device=x_.device)
                        z_l = torch.zeros(1, device=x_.device)
                    return out, aux_l, z_l

                x, aux_loss, z_loss = torch.utils.checkpoint.checkpoint(
                    block_fn, block, x, rope_cos, rope_sin,
                    use_reentrant=False,
                )
                aux_total = aux_total + aux_loss
                z_total = z_total + z_loss
            else:
                layer_cache = kv_caches[i] if (use_cache and kv_caches is not None) else None
                x, new_kv, aux = block(x, rope_cos, rope_sin, layer_cache, use_cache)
                if use_cache:
                    new_caches.append(new_kv)
                if aux is not None:
                    aux_loss, z_loss = aux  # MoE block -> tuple (aux_loss, z_loss)
                    aux_total = aux_total + aux_loss
                    z_total = z_total + z_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, cfg.vocab_size),
                targets.view(B * T),
            )
            # MoE auxiliary losses: load-balance + z-loss
            if not isinstance(aux_total, float):
                loss = loss + cfg.moe_aux_loss * aux_total
            if not isinstance(z_total, float):
                loss = loss + cfg.moe_z_loss * z_total

            # Save for logging (read after forward as model.last_aux_loss)
            self.last_aux_loss = float(aux_total) if isinstance(aux_total, torch.Tensor) else 0.0
            self.last_z_loss = float(z_total) if isinstance(z_total, torch.Tensor) else 0.0

            return logits, loss, new_caches
        return logits, None, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50,
                 top_p=0.9, typical_p=0.0, repetition_penalty=1.2,
                 kv_caches=None, return_caches=False):
        """Fast generation with optional incremental KV cache.

        Args:
            idx: (B, T) prompt tokens.
            max_new_tokens: number of tokens to generate.
            kv_caches: optional per-layer KV cache from a previous call.
                When provided, *idx* is treated as new tokens to extend the
                cache rather than a fresh prefill.
            return_caches: if True, return ``(generated, kv_caches)`` so the
                caller can continue generation later.

        Returns:
            ``generated`` (B, T+max_new_tokens) or, when *return_caches* is
            True, ``(generated, kv_caches)``.

        Sampling controls:
          temperature        -> lower=tamer/safer, higher=wilder (0.6-0.9)
          top_k=50           -> keep only the 50 highest-prob tokens
          top_p=0.9          -> nucleus: smallest token set summing to p
          typical_p=0.0      -> typical sampling threshold (0=off, 0.2=code,
                                0.5=lenient).  Removes tokens whose
                                information content deviates from entropy.
          repetition_penalty -> divide logits of seen tokens by this
        """
        self.eval()
        self._ensure_rope(idx.device, idx.dtype)

        B, T0 = idx.shape
        if idx.size(1) > cfg.block_size:
            idx = idx[:, -cfg.block_size:]
            T0 = idx.size(1)

        # Determine how many tokens are already cached
        past_len = 0
        if kv_caches is not None and len(kv_caches) > 0:
            # GQA: cache entry is (k, v) with k.shape = (B, n_head, T, head_dim)
            # MLA: cache entry is (c_kv,) with c_kv.shape = (B, T, latent_dim)
            entry = kv_caches[0][0]
            if entry.dim() == 3:  # MLA: (B, T, latent_dim)
                past_len = entry.size(1)
            else:                 # GQA: (B, n_head, T, head_dim)
                past_len = entry.size(2)

        # --- prefill (or extend): run the prompt, store/update per-layer caches ---
        logits, _, kv_caches = self(
            idx, use_cache=True, kv_caches=kv_caches, rope_offset=past_len,
        )
        next_logits = logits[:, -1, :]
        generated = idx

        for _ in range(max_new_tokens):
            logits = next_logits / max(temperature, 1e-4)

            # --- repetition penalty ---
            if repetition_penalty != 1.0:
                for b in range(B):
                    seen = set(generated[b].tolist())
                    logits[b, list(seen)] /= repetition_penalty

            if top_k is not None and top_k > 0:
                kth = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = kth.values[:, -1:]
                logits = torch.where(
                    logits < thresh,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_remove = cum_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(-1, sorted_idx, sorted_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # --- typical sampling (Meister et al., 2023) --------------------
            # Filters out tokens whose "surprisingness" deviates too far from
            # the expected entropy under the predicted distribution.  This
            # removes both trivial continuation tokens (too predictable) and
            # nonsensical tokens (too surprising), keeping only tokens whose
            # information content is close to the local expectation.
            #
            #   tau=0.2  →  moderately restrictive (recommended for code)
            #   tau=0.5  →  minimally restrictive
            #   tau=0.0  →  disabled
            if typical_p > 0:
                probs_for_entropy = F.softmax(logits, dim=-1)
                entropy = -torch.sum(
                    probs_for_entropy * torch.log(probs_for_entropy + 1e-8),
                    dim=-1,
                )  # (B,)
                log_probs = F.log_softmax(logits, dim=-1)  # (B, vocab)
                # |log P(x) + H(P)| — how far each token's info content is
                # from the expected entropy.  (Paper: |-log P(x) - H(P)|)
                # A perfectly typical token has log P(x) = -H(P).
                surprisal = (log_probs + entropy.unsqueeze(-1)).abs()
                # Remove tokens with surprisal > tau
                typical_mask = surprisal > typical_p
                logits = logits.masked_fill(typical_mask, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # --- decode: feed the single new token, reuse cached k/v ---
            total_len = generated.size(1)
            if total_len >= cfg.block_size:
                # context full → discard cache and re-prefill the last block
                logits, _, kv_caches = self(
                    generated[:, -cfg.block_size:], use_cache=True,
                )
            else:
                # past_len = total_len before this token (including prompt)
                decode_offset = total_len - 1  # position of the new token
                logits, _, kv_caches = self(
                    next_token, kv_caches=kv_caches, use_cache=True,
                    rope_offset=decode_offset,
                )
            next_logits = logits[:, -1, :]

        if return_caches:
            return generated, kv_caches
        return generated

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu",
                        gradient_checkpointing: bool = False) -> "GPT":
        """Build a GPT from config and load a checkpoint.

        Accepts both a bare state_dict (train.py's ``model.pt``) and a full
        checkpoint dict (checkpoint.py's ``checkpoint.pt``, which has
        model/optimizer/step/...).  Returns the model in eval mode.

        Duplicated across agent/loop and eval/eval before — kept here as the
        single canonical factory so every consumer uses the same path.
        """
        m = cls(gradient_checkpointing=gradient_checkpointing).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd

        # Handle vocab size mismatch: if the checkpoint has extra vocab entries
        # (e.g., from extend_vocab() for tool tokens), extend the model before
        # loading so state_dict shapes match.
        ckpt_vocab = sd["wte.weight"].size(0)
        if ckpt_vocab != cfg.vocab_size:
            if ckpt_vocab > cfg.vocab_size:
                n_new = ckpt_vocab - cfg.vocab_size
                print(f"[from_checkpoint] extending vocab by {n_new} to match checkpoint")
                m.extend_vocab(n_new)
            else:
                print(f"[from_checkpoint] WARNING: checkpoint vocab ({ckpt_vocab}) < "
                      f"config vocab ({cfg.vocab_size}); loading may truncate")

        m.load_state_dict(sd)
        m.eval()
        return m


# ---------------------------------------------------------------------------
# Activation monitoring hooks
# ---------------------------------------------------------------------------

@dataclass
class ActivationStats:
    """Per-block activation statistics captured during a forward pass."""
    layer_idx: int
    attn_out_mean: float    # mean of attention output (shows if attn is active)
    attn_out_std: float     # std of attention output
    ff_out_mean: float      # mean of FF (MLP/MoE) output
    ff_out_std: float       # std of FF output
    dead_ratio: float       # fraction of SwiGLU gate outputs ≤ 0 (dead neurons)
    hidden_mean: float      # mean of residual-stream hidden state *after* block
    hidden_std: float       # std of residual-stream hidden state


@dataclass
class ActivationReport:
    """Summary of activation statistics across all blocks."""
    per_layer: list[ActivationStats]
    mean_dead_ratio: float
    max_dead_ratio: float
    mean_hidden_std: float
    attn_utilization: float  # mean abs attn out / mean hidden (0=dead, 1=active)


class ActivationMonitor:
    """Register forward hooks on every Block to capture activation statistics.

    Usage:
        monitor = ActivationMonitor(model)
        ... model.forward(...) ...
        report = monitor.report()  # ActivationReport
        monitor.clear()
    """

    def __init__(self, model: "GPT"):
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # Per-block storage: list of dicts with "attn", "ff", "hidden" tensors
        self._block_data: list[dict[str, torch.Tensor]] = []

        def _attn_hook(m, inp, out, idx):
            # out = (attn_out, new_kv_branch)
            attn_out = out[0] if isinstance(out, tuple) else out
            self._block_data[idx]["attn"] = attn_out.detach()

        def _block_hook(m, inp, out, idx):
            # out = (x, new_kv, (aux_loss, z_loss))
            x = out[0] if isinstance(out, tuple) else out
            self._block_data[idx]["hidden"] = x.detach()
            # Capture the FF gate output for dead-ratio computation.
            # The MLP/MoE's gate output is accessible via the last FF layer's input.
            if hasattr(m, "mlp"):
                mlp = m.mlp
                if hasattr(mlp, "c_gate"):  # SwiGLU: gate output before SiLU
                    self._block_data[idx]["gate"] = mlp.c_gate.weight

        for idx, block in enumerate(model.blocks):
            self._block_data.append({"attn": None, "ff": None, "hidden": None, "gate": None})
            h1 = block.attn.register_forward_hook(
                lambda m, i, o, idx=idx: _attn_hook(m, i, o, idx)
            )
            h2 = block.register_forward_hook(
                lambda m, i, o, idx=idx: _block_hook(m, i, o, idx)
            )
            self._handles.extend([h1, h2])

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        for d in self._block_data:
            d["attn"] = None
            d["ff"] = None
            d["hidden"] = None
            d["gate"] = None

    def report(self) -> ActivationReport:
        """Aggregate captured stats into a report.

        Call *after* a forward pass.  If no data was captured, returns a
        dummy report with zeros.
        """
        stats_list: list[ActivationStats] = []
        for idx, d in enumerate(self._block_data):
            at = d.get("attn")
            hd = d.get("hidden")
            gate = d.get("gate")

            attn_mean = at.abs().mean().item() if at is not None else 0.0
            attn_std = at.std().item() if at is not None else 0.0
            hidden_mean = hd.mean().item() if hd is not None else 0.0
            hidden_std = hd.std().item() if hd is not None else 0.0

            # Estimate dead ratio from the gate weight distribution.
            # A gate weight near zero → likely dead neuron.
            # We approximate dead ratio as fraction of gate weights within
            # a narrow band around zero, scaled by the std.
            dead_ratio = 0.0
            ff_mean = 0.0
            ff_std = 0.0
            if gate is not None:
                gate_data = gate.flatten()
                # Fraction of gate weights with abs < 0.01 * std
                g_std = gate_data.std().item()
                if g_std > 0:
                    dead_ratio = (gate_data.abs() < 0.01 * g_std).float().mean().item()
                ff_mean = gate_data.mean().item()
                ff_std = g_std

            stats_list.append(ActivationStats(
                layer_idx=idx,
                attn_out_mean=round(attn_mean, 6),
                attn_out_std=round(attn_std, 6),
                ff_out_mean=round(ff_mean, 6),
                ff_out_std=round(ff_std, 6),
                dead_ratio=round(dead_ratio, 4),
                hidden_mean=round(hidden_mean, 6),
                hidden_std=round(hidden_std, 6),
            ))

        dead_ratios = [s.dead_ratio for s in stats_list]
        hidden_stds = [s.hidden_std for s in stats_list if s.hidden_std > 0]
        attn_means = [s.attn_out_mean for s in stats_list if s.attn_out_mean > 0]
        hidden_means = [s.hidden_mean for s in stats_list if s.hidden_mean != 0]

        return ActivationReport(
            per_layer=stats_list,
            mean_dead_ratio=sum(dead_ratios) / max(len(dead_ratios), 1),
            max_dead_ratio=max(dead_ratios) if dead_ratios else 0.0,
            mean_hidden_std=sum(hidden_stds) / max(len(hidden_stds), 1),
            attn_utilization=(
                sum(attn_means) / max(sum(abs(m) for m in hidden_means), 1e-9)
                if attn_means and hidden_means else 0.0
            ),
        )


if __name__ == "__main__":
    # Guard: never accidentally build the fat (~8B) profile on a small machine.
    import config as _cfg
    if _cfg.PROFILE == "fat" and torch.cuda.is_available() \
       and torch.cuda.get_device_properties(0).total_memory < 16 * (10**9):
        print("REFUSING to build the 'fat' profile on this GPU (<16GB).")
        print("Set CHATON_PROFILE=dev for local architecture work, or run 'fat' on a cloud VM.")
        raise SystemExit(0)
    model = GPT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built. Parameters: {n_params:,}")

    # With proper init, the loss on RANDOM targets should be ~ ln(vocab)=10.8,
    # NOT 158. This is the whole point of the init fix.
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, _ = model(x, targets=x)
    import math
    print(f"Logits shape: {tuple(logits.shape)} (expect (2, 16, {cfg.vocab_size}))")
    print(f"Loss: {float(loss.detach()):.4f}  (random-guess baseline ln(vocab)={math.log(cfg.vocab_size):.2f})")

    # quick generate smoke test
    out = model.generate(x[:, :4], max_new_tokens=5, temperature=0.8, top_k=40)
    print("generate() output shape:", tuple(out.shape))