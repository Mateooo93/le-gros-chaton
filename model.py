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
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

    def forward(self, x, rope_cos=None, rope_sin=None, kv_cache=None, use_cache=False):
        B, T, C = x.shape

        qkv = self.c_attn(x)                       # (B, T, 3*C)
        q, k, v = qkv.split(cfg.n_embd, dim=2)     # each (B, T, C)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # apply rotary position embedding to q and k (NOT v)
        if rope_cos is not None:
            q, k = _apply_rope(q, k, rope_cos, rope_sin)

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
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


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
        self.mlp = MLP()

    def forward(self, x, rope_cos=None, rope_sin=None, kv_cache=None, use_cache=False):
        attn_out, new_kv = self.attn(
            self.ln_1(x), rope_cos, rope_sin, kv_cache, use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


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
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if (use_cache and kv_caches is not None) else None
            x, new_kv = block(x, rope_cos, rope_sin, layer_cache, use_cache)
            if use_cache:
                new_caches.append(new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, cfg.vocab_size),
                targets.view(B * T),
            )
            return logits, loss, new_caches
        return logits, None, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        """Fast generation: prefill the whole prompt (building the KV cache),
        then decode ONE token at a time reusing the cache. top_k=50 keeps only
        the 50 most-likely next tokens before sampling -> kills tail garbage."""
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
            if top_k is not None and top_k > 0:
                # keep only the top_k logits, set the rest to -inf
                kth = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = kth.values[:, -1:]
                logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
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