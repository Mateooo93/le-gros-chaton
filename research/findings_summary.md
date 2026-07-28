# Research Findings — July 2026

## Kimi K3 (Moonshot AI)
2.8T MoE, 896 experts/16 active, 1M context.
**Implemented:** Quantile Balancing, SiTU, Latent MoE, Learned Residual
**Future:** Kimi Delta Attention (needs FlashKDA CUDA kernels)

## DeepSWE (Together AI + Agentica, 2025)
Qwen3-32B RL-only → 42.2% SWE-Bench Verified.
**Applied:** Test-time scaling (--n-samples), proportional rewards

## Self-Play SWE-RL (SSR, ICML 2025)
Self-play bug injection/repair, +10.4 SWE-Bench, no human data.
**Applied:** self_play_data.py

## SERA (Soft-Verified Efficient Repository Agents, 2026)
Soft-verified generation with partial test results as training signal.
**Aligned with:** our proportional rewards approach

## Key Takeaway
The winning recipe: strong architecture (MoE+quantile balancing) + RL with
proportional rewards + test-time scaling + self-play data generation.


## Routing-Free Mixture-of-Experts (arXiv 2604.00801, April 2026)
Eliminates centralized routers, Softmax, TopK, and load balancing entirely.
Each expert independently decides activation via its own low-rank gate norm.
- **Architecture**: Expert has A_gate (D×r), B_gate (r×D_act). Gate = sigma(x·A_gate·B_gate)
- **Activation score**: ||x·A_gate||₂ (norm of low-rank projection)
- **No router, no Softmax, no TopK** — each expert decides independently
- **Threshold adaptation**: learned threshold controls sparsity
- **Load balancing**: unified loss for expert + token balancing
- **Status**: ⏳ Planning implementation
