import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
def _build_rope_cache(block_size, head_dim, device, theta=10000.0):
    # freq[i] = 1 / theta^(2i/head_dim) for i in [0, head_dim/2)
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(block_size, device=device).float()
    angles = torch.outer(pos, freqs)                 # (block_size, head_dim/2)
    # repeat_interleave so each freq pairs two dims: (block_size, head_dim)
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
        # Grouped-Query Attention: Q has all n_head, but K/V only have n_kv_head
        # (the Q heads are grouped and share a KV head). Fewer KV params + a
        # smaller KV cache -> cheaper long-context decode (agent harness). When
        # n_kv_head == n_head this is plain MHA (identical to the old behavior).
        kv_dim = self.n_kv_head * self.head_dim
        self.c_q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_kv = nn.Linear(cfg.n_embd, 2 * kv_dim, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, rope_cos=None, rope_sin=None, kv_cache=None, use_cache=False):
        B, T, C = x.shape

        q = self.c_q(x)                                  # (B, T, C)
        k, v = self.c_kv(x).split(self.n_kv_head * self.head_dim, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # apply rotary position embedding to q and k (NOT v)
        if rope_cos is not None:
            q, k = _apply_rope(q, k, rope_cos, rope_sin)

        # GQA: replicate each KV head across the Q heads that share it so the
        # matmul shapes line up for SDPA. n_head // n_kv_head = group size.
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # --- KV cache: on decode steps we only get the NEW token (T=1).
        #     Append its k/v to the cached past k/v and attend over all of it. ---
        if use_cache and kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)      # (B, n_head, T_past+1, head_dim)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v) if use_cache else None

        # SDPA: one fused call replaces scores/scale/mask/softmax/@v.
        # is_causal=True only when there's no cache (full prefill); when a
        # cache is present (T=1 querying against the whole past) it's already
        # causal and we must NOT mask, so is_causal=False there.
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
            return self.c_proj(F.silu(self.c_gate(x)) * self.c_up(x))
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
        self.gate = nn.Linear(cfg.n_embd, n_expert, bias=False)
        # Each expert is its own little MLP (same shape as the dense MLP).
        self.experts = nn.ModuleList([MLP() for _ in range(n_expert)])
        # Shared expert(s): always active (no routing), captures common
        # knowledge so the routed experts don't have to relearn it -> better
        # specialization. Add 1 and the total active params go up by one MLP.
        self.shared_experts = nn.ModuleList([MLP() for _ in range(self.n_shared)])

    def forward(self, x):
        # x: (B, T, C) -> flatten tokens for routing
        B, T, C = x.shape
        flat = x.view(B * T, C)

        # --- gate + top-k routing ---
        gate_logits = self.gate(flat)                       # (B*T, n_expert)
        topk_vals, topk_idx = torch.topk(gate_logits, self.n_expert_top, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)         # (B*T, n_expert_top)

        # dispatch: for each expert, gather the tokens routed to it, run the
        # expert MLP on just those, then scatter results back. This keeps
        # memory bounded to active params (the whole point of MoE).
        out = torch.zeros_like(flat)
        for e in range(self.n_expert):
            # which (token-row, topk-slot) pairs chose expert e?
            mask = (topk_idx == e)                          # (B*T, n_expert_top)
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            expert_in = flat[token_idx]                     # only the routed tokens
            expert_out = self.experts[e](expert_in)
            # weight each expert's output by its top-k router weight
            w = topk_weights[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_out * w)

        # --- shared expert: always-on, runs on every token (no routing) ---
        for se in self.shared_experts:
            out = out + se(flat)

        # --- load-balance aux loss (Switch Transformer form) ---
        with torch.no_grad():
            # f_i = fraction of tokens whose TOP-1 choice was expert i
            top1 = topk_idx[:, 0]
            counts = torch.bincount(top1, minlength=self.n_expert).float()
            f = counts / counts.sum()
        P = F.softmax(gate_logits, dim=-1).mean(0)          # mean router prob per expert
        aux_loss = self.n_expert * (f * P).sum()

        return out.view(B, T, C), aux_loss


# ---------------------------------------------------------------------------
# 3. A TRANSFORMER BLOCK — pre-norm (RMSNorm) + Attention + residual,
#    then RMSNorm + MLP + residual.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = RMSNorm(cfg.n_embd)
        # MoE layer if enabled, else the dense MLP. MoE returns (out, aux_loss).
        self.mlp = MoE() if cfg.use_moe else MLP()
        self.is_moe = cfg.use_moe

    def forward(self, x, rope_cos=None, rope_sin=None, kv_cache=None, use_cache=False):
        attn_out, new_kv = self.attn(
            self.ln_1(x), rope_cos, rope_sin, kv_cache, use_cache
        )
        x = x + attn_out
        ff_in = self.ln_2(x)
        if self.is_moe:
            ff_out, aux = self.mlp(ff_in)
            x = x + ff_out
            return x, new_kv, aux
        x = x + self.mlp(ff_in)
        return x, new_kv, None


# ---------------------------------------------------------------------------
# 4. THE FULL MODEL — token embed -> RoPE -> N blocks -> final norm -> logits
#    with a KV cache for fast generation.
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Token embedding only — positions come from RoPE now (no wpe).
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        # Tied LM head (shares weights with wte) — saves params and helps.
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # RoPE cos/sin buffers, built lazily so device follows the model.
        self.rope_built = False

        # --- Weight initialization: the fix for the 158 start-loss. ---
        # Standard N(0, 0.02) everywhere, but every projection that FEEDS a
        # residual stream is scaled by 1/sqrt(2*n_layer) so the residual
        # signal variance stays ~constant as we stack layers. (2 because each
        # block adds two residuals: attn + mlp.) This stops early logits
        # from blowing up. Tied wte/lm_head is left at 0.02.
        self.apply(self._init_weights)
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
            cos, sin = _build_rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head, device)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.rope_built = True

    def forward(self, idx, targets=None, kv_caches=None, use_cache=False):
        B, T = idx.shape
        self._ensure_rope(idx.device, idx.dtype)
        # crop the rope cache to the current query length T (works for T=1 decode)
        rope_cos = self.rope_cos[:T]
        rope_sin = self.rope_sin[:T]

        x = self.wte(idx)                           # (B, T, n_embd)

        new_caches = [] if use_cache else None
        aux_total = 0.0
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if (use_cache and kv_caches is not None) else None
            x, new_kv, aux = block(x, rope_cos, rope_sin, layer_cache, use_cache)
            if use_cache:
                new_caches.append(new_kv)
            if aux is not None:
                aux_total = aux_total + aux

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, cfg.vocab_size),
                targets.view(B * T),
            )
            # add the MoE load-balancing aux loss if any block is an MoE
            if not isinstance(aux_total, float):
                loss = loss + cfg.moe_aux_loss * aux_total
            return logits, loss, new_caches
        return logits, None, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.2):
        """Fast generation: prefill the whole prompt (building the KV cache),
        then decode ONE token at a time reusing the cache.

        Sampling controls:
          temperature        -> lower=tamer/safer, higher=wilder (0.6-0.9 sweet spot)
          top_k=50           -> keep only the 50 highest-prob tokens (kills the tail)
          top_p=0.9          -> nucleus: keep smallest token set whose prob sums to 0.9
          repetition_penalty -> divide logits of tokens already generated by this
                                (1.0 = off; 1.2-1.3 breaks the 'Singapore Singapore'
                                loops a 17M model falls into). THE key fix.
        """
        self.eval()
        self._ensure_rope(idx.device, idx.dtype)

        # --- prefill: run the full prompt, store per-layer (k,v) caches ---
        B, T0 = idx.shape
        if idx.size(1) > cfg.block_size:
            idx = idx[:, -cfg.block_size:]
            T0 = idx.size(1)
        logits, _, kv_caches = self(idx, use_cache=True)
        next_logits = logits[:, -1, :]

        generated = idx
        for _ in range(max_new_tokens):
            logits = next_logits / max(temperature, 1e-4)

            # --- repetition penalty: divide logits of already-seen tokens ---
            # (divide, not subtract, so a very confident repeat gets tamed but a
            #  weak one still survives; applied on the raw scaled logits)
            if repetition_penalty != 1.0:
                for b in range(B):
                    seen = set(generated[b].tolist())
                    logits[b, list(seen)] /= repetition_penalty

            if top_k is not None and top_k > 0:
                # keep only the top_k logits, set the rest to -inf
                kth = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = kth.values[:, -1:]
                logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

            if top_p is not None and 0 < top_p < 1.0:
                # nucleus: keep the smallest token set whose cumulative prob >= top_p.
                # Work in sorted order, then scatter-remove back to the original layout.
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                # tokens AFTER the first one that crosses top_p are removed
                sorted_remove = cum_probs > top_p
                # but always keep the top-1 token (shift the mask right by one)
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                # map the sorted-order removal mask back to the original vocab order
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(-1, sorted_idx, sorted_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B,1)
            generated = torch.cat([generated, next_token], dim=1)

            # --- decode: feed only the new token, reuse cached k/v ---
            if generated.size(1) - T0 >= cfg.block_size:
                # context would exceed block_size -> reset cache with a fresh prefill
                logits, _, kv_caches = self(generated[:, -cfg.block_size:], use_cache=True)
            else:
                logits, _, kv_caches = self(next_token, kv_caches=kv_caches, use_cache=True)
            next_logits = logits[:, -1, :]

        return generated


# Quick sanity check when you run: python model.py
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