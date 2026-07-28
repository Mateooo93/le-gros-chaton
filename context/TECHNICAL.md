# Le Gros Chaton — Technical Documentation

## Overview

Le Gros Chaton ("the fat kitten") is a from-scratch Mixture-of-Experts transformer
for agentic coding tasks, built in PyTorch.  The project targets a ~10.5B total /
~3.8B-active MoE model trained through a 4-stage RL pipeline:

0. **Base pretrain** — next-token prediction on code + prose (WSD schedule)
1. **RFT** — Rejection-sampling Fine-Tuning (sample → verify → SFT)
2. **RLVR** — GRPO with code-verifier reward
3. **PRM** — Process Reward Model (step-level scoring)
4. **Agent RL** — agent-loop-as-rollout RL
5. **Eval** — agentic evaluation on HumanEval / MBPP

All stages are orchestrated by `pipeline.py --profile <name>`.

## Profiles

| Profile    | Params (total) | Params (active) | Active % | n_layer | n_embd | n_head | n_expert | n_shared |
|------------|---------------:|----------------:|---------:|--------:|-------:|-------:|---------:|---------:|
| dev        | 14.4M          | 14.4M           | 100%     | 6       | 288    | 6      | 0        | 0        |
| smol-fat   | 290M           | 120M            | 41.4%    | 12      | 512    | 8      | 8        | 1        |
| fat        | 10.5B          | 3.83B           | 36.6%    | 32      | 2048   | 16     | 16       | 1        |

Set with `CHATON_PROFILE=smol-fat`.

## Architecture

### Core (all profiles)
- **SwiGLU MLP** — gated activation (SiLU × linear) used by Llama/Qwen/DeepSeek
- **Grouped-Query Attention (GQA)** — fewer KV heads than query heads; smaller KV cache
- **Rotary Position Embedding (RoPE)** — with optional **NTK-aware YaRN scaling** (`CHATON_ROPE_SCALE=2` extends context 2× without retraining)
- **QK-normalisation** — RMSNorm on Q and K *before* RoPE; prevents logit growth at long contexts. Enabled by default; disable with `CHATON_USE_QK_NORM=0`

### MoE (smol-fat, fat)
- **Top-2 routing** with learned linear gate + softmax over top-2 scores
- **Quantile Balancing** (Kimi K3-style, `CHATON_QUANTILE_BALANCE=1`) — deterministic,
  hyperparameter-free load balancing. Token routes to expert if score is in top
  quantile. Replaces aux_loss and gate_bias with guaranteed even utilization.
- **Latent MoE** (Kimi K3-style, `CHATON_MOE_LATENT_DIM=256`) — compresses token
  representation via down-projection before routing, up-projects after expert
  computation. Reduces routing compute and multi-GPU communication.
- **SiTU Activation** (Kimi K3-style, `CHATON_USE_SITU=1`) — Sigmoid Tanh Unit
  replaces SiLU in SwiGLU to prevent dead-neuron pathology in rarely-activated
  experts. Formula: `SiTU(x) = sigmoid(x) * tanh(x)`.
- **Learned Residual** (`CHATON_LEARNED_RESIDUAL=1`) — per-layer learnable scaling
  factor on the attention residual connection, inspired by Kimi K3's Attention
  Residuals. Each layer learns how much information to preserve.
- **Grouped dispatch** — replaces per-expert `mask.nonzero()` loops with `torch.argsort` + `torch.searchsorted` contiguous segments. Cache-friendly, fewer kernel launches
- **Z-loss** — `mean(log(sum(exp(gate_logits)))^2)` penalty prevents router logit explosion (DeepSeek-V2 coefficient 0.001)
- **Load-balancing aux loss** — `loss = n_expert * sum(f_i * P_i)` over ALL top-k slots (not just top-1), matching the DeepSeek-V2/V3 formulation
- **Gate bias** (DeepSeek-V3 style) — per-expert bias adjusted heuristically each forward: increased for underloaded experts, decreased for overloaded. Not backpropped. Bias LR controlled by `gate_bias_lr` config (default 0.01)
- **Dynamic top-k** — when `CHATON_DYNAMIC_TOPK=1`, tokens with high gate confidence (p1/p2 > threshold) drop the second expert, saving compute on easy tokens
- **Shared expert** — always-active expert (DeepSeek-style), controlled by `n_shared_expert` in config

### Memory / Compute Optimisations
- **Gradient checkpointing** — `CHATON_GRAD_CKPT=0` to disable; trades 20% compute for 60% activation memory
- **Selective weight decay** — 1D params (norms, biases) + embeddings get 0 decay; 2D params (projections) get `weight_decay` (default 0.1)
- **KV cache compression** (StreamingLLM) — when cache exceeds `max_cache_len`, keeps `cache_n_sink` (default 4) initial tokens + most recent, evicts the middle. Bounds cache during long agent rollouts
- **Tool tokens** — `<|tool_call|>`, `<|tool_result|>`, `<|done|>` at VOCAB_SIZE+{0,1,2}. Embeddings initialised from mean of GPT-2 subword tokens. `GPT.extend_vocab(n, init_from=...)` adds them at runtime

## Data Pipeline (`data_code.py`)

The pipeline builds a uint16 memmap from interleaved code + prose documents.

**Corpus selection:** `CHATON_CODE_CORPUS` (default `starcoderdata`). Code/prose blend ratio via `CHATON_PROSE_BLEND` (default 0.1 = 10% prose).

**Document-level processing** (in order):
1. **Syntax validation** — `compile(text, flags=ast.PyCF_ONLY_AST)` skips non-parsing files. Enabled with `CHATON_VALIDATE_SYNTAX=1`
2. **Educational quality filter** — 6-heuristic composite score (docstrings, comments, type hints, tests, identifier quality, length diversity). `CHATON_MIN_EDU_SCORE=0.5` to enable
3. **MinHash near-deduplication** — 128 universal hash functions, character 5-gram shingles, LSH with 16 bands × 8 hashes, Jaccard threshold 0.85. `CHATON_DEDUP_THRESHOLD=0` to disable

**Document interleaving:** shuffled code pool + shuffled prose pool, alternated (code/prose/code/prose…) so the model sees both domains without predicting cross-document transitions.

**Lazy init:** no downloads, tokenization, or GPU uploads happen at import time — deferred to first `get_batch()` call.

## Training (`train.py`)

### Learning Rate Schedule
- **WSD** (warmup-stable-decay) — linear warmup to `lr_max`, plateau at `lr_max`, linear cooldown to `lr_min`
- Set with `CHATON_LR_SCHEDULE=wsd` and `CHATON_COOLDOWN_ITERS=<N>`
- **Cosine** also supported (default for backward compatibility)

### Throughput Tracking
`ThroughputTracker` logs tokens/s, FLOPs/s, GPU memory, and ETA at each eval interval:

```
step  500  lr 1.00e-03  train loss 3.42  val loss 3.61  156,000 tok/s  mem 18.3 GB  eta 7320s  (42%)
```

All metrics written to the experiment logger (`runs/<run_name>/log.jsonl`).

### Activation Monitoring
`ActivationMonitor` registers forward hooks on every `Block` to capture per-layer statistics:
- **Dead ratio** — fraction of SwiGLU gate weights near zero
- **Hidden std** — std of residual stream activations (exploding → instability, collapsing → vanishing signal)
- **Attn utilization** — ratio of attention output to hidden state

## RL Pipeline

All stages are orchestrated by `pipeline.py`:

```
python pipeline.py --profile smol-fat --stages all          # full pipeline
python pipeline.py --profile dev --stages 0 1               # pretrain + RFT only
python pipeline.py --profile smol-fat --stages 3 --resume   # resume PRM training
```

**Stage 0: Base Pretrain** — `train.py` with code data. Output: `training/<profile>/base_pretrain/model.pt`

**Stage 1: RFT** (`rft.py`) — sample N solutions per problem, verify with sandbox, behaviourally deduplicate, SFT on passing solutions.
- `python rft.py collect --n-samples 32 --max-problems 100`
- `python rft.py train --data rft_data.json`

**Stage 2: RLVR** (`rlvr.py`) — GRPO (group-relative policy optimisation). No value model; group-normalised advantage from verifier reward. KL penalty to prevent collapse.
- `python rlvr.py --ckpt model.pt --group-size 8 --n-steps 200`

**Stage 3: PRM** (`prm.py`) — Process Reward Model as a 2-layer MLP (n_embd → n_embd/2 → 1) on frozen base model. Monte-Carlo backward labelling from verifier outcome. Pessimistic min-step scoring.
- `python prm.py collect --n-problems 200`
- `python prm.py train --data prm_data.json`
- `python prm.py score --solution <code>`

**Stage 4: Agent RL** (`agent_rl.py`) — treats multi-turn agent trajectories as RL rollouts. REINFORCE-with-baseline, group-normalised advantages, KL penalty.
- `python agent_rl.py --ckpt model.pt --problems humaneval --n-steps 50`

**Stage 5: Eval** (`eval/agent_eval.py`) — runs the full agent loop on each problem. Reports pass@1, avg steps, avg wall time.
- `python eval/agent_eval.py --ckpt model.pt --limit 20`

## Inference (`inference.py`)

Unified `InferenceEngine` class used by `chat.py` and programmatically:

```python
from inference import InferenceEngine
engine = InferenceEngine("model.pt")
reply = engine.chat([{"role": "user", "content": "Write fib"}])
```

Supports all sampling strategies:
- **top-k** — keep only the k highest-probability tokens
- **top-p (nucleus)** — smallest token set whose cumulative probability exceeds p
- **typical_p** (Meister et al., 2023) — filters tokens whose `|-log P(x) - H(P)|` exceeds threshold. Removes both trivial tokens (too predictable) and nonsensical tokens (too surprising). Recommended: `typical_p=0.2` for code

KV cache is maintained across chat turns for efficient multi-turn interaction. `engine.clear_cache()` resets for a new conversation.

## Agent Loop (`agent/loop.py`)

The agent loop turns the base LM into a terminal tool-user:
1. Receives a task description
2. Generates commands (via `<cmd>` XML tags or `<|tool_call|>` tokens)
3. Runs them in a sandboxed shell
4. Appends output to the conversation
5. Loops until `<done>` or `<|done|>`

**KV-cache-optimised:** the cache is maintained across turns; the conversation isn't re-encoded from scratch at each step (was O(n²) in context length).

**Token healing:** tool tokens (`<|tool_call|>`, `<|tool_result|>`, `<|done|>`) provide explicit protocol markers instead of fragile XML text patterns.

## Evaluation Harness (`eval/agent_eval.py`)

Runs the full agent loop on each problem and produces:
- **Pass@1** — fraction of problems solved
- **Avg steps** — average tool-call steps before completion
- **Avg commands** — average commands run
- **Wall time** — average time per problem

Supports `--compare` for side-by-side run comparison.

## Code Quality

- **pre-commit hooks**: ruff (linting + auto-fix), black (formatting), isort (import sorting), plus sanity checks
- **pytest suite**: 21+ tests for model, verifier, and data pipeline
- **Static analysis**: `profile_analyzer.py` computes FLOPs/memory/throughput from config values (no torch needed)
- **Experiment logger**: JSONL logs with fsync crash safety, git hash, config snapshot, env vars. CLI viewer: `python log.py runs/train/log.jsonl`

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Flat imports at project root | Works with `python train.py`, Colab/Modal notebooks, keeps structure simple |
| GRPO over PPO | No value model = less memory, fewer moving parts |
| WSD over Cosine | Robust to VM-hopping; plateau at full LR until cooldown |
| MinHash 5-grams | Captures code structure (indentation, brackets, keywords) |
| Grouped dispatch (argsort) | Contiguous memory access, fewer kernel launches vs `mask.nonzero()` |
| Typical sampling | Reduces boilerplate AND hallucination in code generation |
| StreamingLLM cache | Bounds KV cache during long agent rollouts |

## Training Cost Estimates

Estimated throughput and cost per 1B tokens trained on common GPUs
(from `profile_analyzer.py`):

| Profile | Params | GPU | Train tok/s | Cost/B tok | Batch |
|---------|--------|-----|------------|-----------|-------|
| dev | 14.4M | A100-80GB | 3,701K | $0.15 | 2,517 |
| dev | 14.4M | RTX 4090 | 1,957K | $0.07 | 710 |
| dev | 14.4M | T4 | 771K | $0.11 | 451 |
| smol-fat | 290M | A100-80GB | 368K | $1.51 | 136 |
| smol-fat | 290M | RTX 4090 | 195K | $0.71 | 38 |
| smol-fat | 290M | T4 | 77K | $1.09 | 24 |
| fat | 10.5B | A100-80GB | 9K | $60.14 | 4 |

Cost estimates assume FP16 training with gradient checkpointing and
`torch.compile` enabled.  Actual throughput varies by GPU, driver, and
system configuration. T4 and RTX 4090 cannot fit the fat profile (needs
A100-80GB minimum).

## References

Key papers informing architecture and training design:

1. **DeepSeek-V2**: DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv:2405.04434, 2024.
2. **DeepSeek-V3**: DeepSeek-AI. "DeepSeek-V3: A Computational Framework for Load Balancing." arXiv:2412.19437, 2024. — Auxiliary-loss-free load balancing via per-expert bias.
3. **Qwen2.5-Coder**: Hui et al. "Qwen2.5-Coder Technical Report." arXiv:2409.12186, 2024. — Code RLVR training recipe.
4. **Llama 2**: Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288, 2023. — RMSNorm, SwiGLU, residual scaling init.
5. **WSD Schedule**: Hu et al. "WSD Schedule: A Better Learning Rate Schedule for Pre-training." 2024. — Warmup-stable-decay LR schedule.
6. **Typical Sampling**: Meister et al. "Typical Decoding for Natural Language Generation." arXiv:2202.00666, 2022.
7. **StreamingLLM**: Xiao et al. "Efficient Streaming Language Models with Attention Sinks." arXiv:2309.17453, 2023.
8. **MinHash**: Broder. "On the Resemblance and Containment of Documents." SEQUENCES, 1997. — Near-deduplication.
9. **GRPO**: Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning." arXiv:2402.03300, 2024.
10. **Process Reward Models**: Lightman et al. "Let's Verify Step by Step." OpenAI, 2023. — PRM for step-level credit assignment.
11. **QLoRA**: Dettmers et al. "QLoRA: Efficient Finetuning of Quantized Language Models." arXiv:2305.14314, 2023.
12. **YaRN**: Peng et al. "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071, 2023. — NTK-aware RoPE scaling.
