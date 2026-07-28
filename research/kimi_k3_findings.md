# Kimi K3 Research — Key Findings for Le Gros Chaton

## Source
Moonshot AI, July 2026. Blog: https://www.kimi.com/blog/kimi-k3
Architecture breakdown: https://kenhuangus.substack.com/p/demystifying-kimi-k3-how-chinas-28t

## Key Innovations

### 1. ✅ Quantile Balancing (IMPLEMENTED)
Replaces auxiliary load-balancing loss with quantile-based routing.
- Token routes to expert if score is in top quantile per expert
- Deterministic, hyperparameter-free, guarantees even utilization
- Eliminates need for aux_loss and gate_bias heuristics
- **Config flag: `CHATON_QUANTILE_BALANCE=1`**

### 2. ❌ Kimi Delta Attention (KDA) — FUTURE WORK
Replaces scalar decay in DeltaNet with per-channel diagonal matrix.
- Each hidden dimension gets independent forgetting rate
- Enables 1M context with 6.3× faster decoding, 75% KV cache reduction
- Requires FlashKDA kernels (not implemented)
- Builds on Gated DeltaNet + Diagonal-Plus-Low-Rank transition

### 3. ❌ Attention Residuals (AttnRes) — FUTURE WORK
Each layer attends over earlier layers with learned weights.
- Block AttnRes: ordinary residuals inside block, attention over depth between blocks
- Drops cost from O(L·d) to O(N·d)
- Bought 7.5-point GPQA-Diamond jump on 48B model
- Fixes gradient dilution in deep stacks (100+ layers)

### 4. ❌ Stable LatentMoE — PARTIALLY COVERED
- 896 experts, 16 active (98.2% sparsity)
- Token compressed via down-projection before routing
- Per-Head Muon optimizer for training stability
- SiTU activation (Sigmoid Tanh Unit) avoids dead neurons

### 5. ❌ Quantization-Aware Training — FUTURE WORK
- MXFP4 weights + MXFP8 activations from SFT onward
- Model learns resilience to low-precision format

## Relevance to Our Project

| Innovation | Our Status | Impact | Priority |
|-----------|-----------|--------|----------|
| Quantile Balancing | ✅ Implemented (CHATON_QUANTILE_BALANCE=1) | Replaces aux_loss | High |
| SiTU Activation | ❌ Not implemented | Better for sparse MoE | Medium |
| Gated MLA | ❌ Not implemented | Combines MLA with gating | Medium |
| Attention Residuals | ❌ Not implemented | Benefit for deep models | Low (our model is shallow) |
| Quantization-Aware Training | ❌ Not implemented | Inference efficiency | Low (post-training) |

## Coding Agent-Specific Insights

Kimi K3 excels at coding benchmarks because:
1. **1M context window** (KDA) — entire codebase fits in context
2. **Attention Residuals** — synthesizing coherent architecture across many components
3. **50B+ active params** — headroom for reasoning beyond routine completion
4. **SWE-Marathon** — Kimi K3 handles 27M token average rollouts per task

## Action Items
- [x] Quantile Balancing implemented (this iteration)
- [ ] Research SiTU activation for MoE experts
- [ ] Monitor Kimi K3 technical report for implementation details
- [ ] Apply Quantile Balancing understanding to improve future MoE routers
