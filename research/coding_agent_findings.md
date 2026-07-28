# Coding Agent Research — July 2026

## Papers Reviewed

### 1. Kimi K3 (Moonshot AI, July 2026)
2.8T param MoE, 896 experts/16 active. Key innovations:
- **Quantile Balancing** ✅ Implemented (CHATON_QUANTILE_BALANCE=1)
- **SiTU Activation** ✅ Implemented (CHATON_USE_SITU=1)
- Kimi Delta Attention (KDA) — linear attention with per-channel decay
- Attention Residuals (AttnRes) — learned cross-layer attention
- Stable LatentMoE — latent compression before expert routing

### 2. DeepSWE (Together AI + Agentica, July 2025)
Qwen3-32B trained with RL-only → 42.2% SWE-Bench Verified.
Key insights:
- **RL-only training** (no SFT) from base model works with 200 steps
- Test-time scaling pushes to 59%
- Open-source: https://github.com/agentica-project/rllm

### 3. Self-Play SWE-RL (ICML 2025)
Self-play paradigm: one agent injects bugs, another fixes them.
Key insights:
- **No human data needed** — only sandboxed repos
- Automated bug injection + test suite as reward
- +10.4 on SWE-Bench Verified, generalizes to natural language
- Paper: https://arxiv.org/abs/2512.18552

## Action Items for Our Project

### Implemented
- [x] **Quantile Balancing** — CHATON_QUANTILE_BALANCE=1
- [x] **SiTU Activation** — CHATON_USE_SITU=1
- [x] Research findings documented

### Future (when GPU available)
- [ ] Train smol-fat with quantile_balance + situ to validate
- [ ] Implement self-play data generation for coding agent RL
- [ ] Evaluate on SWE-Bench Lite (beyond HumanEval)
- [ ] Incorporate DeepSWE reward formulation into agent_rl.py
