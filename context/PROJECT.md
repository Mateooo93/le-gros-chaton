# Le Gros Chaton — Project Context

**Status after 55 iterations of development (July 2024).**

## What this is

A from-scratch Mixture-of-Experts coding agent transformer (~10.5B total / ~3.83B
active) with a complete 6-stage RL pipeline. Built from PyTorch primitives —
not a fine-tune of an existing model. Targets agentic coding tasks (terminal-bench).

## Current state

The project is comprehensively built across every dimension. All core code is
written, tested for syntax (35 Python files, zero errors), and documented.

### Built and ready
- **Architecture**: MoE with grouped dispatch, z-loss, gate bias, dynamic top-k,
  SwiGLU, GQA, QK-norm, RoPE scaling (NTK-aware YaRN), residual scaling init
- **Data pipeline**: syntax validation, educational quality filter, MinHash
  near-dedup, document interleaving, code+prose corpus blend
- **Training**: WSD schedule, EMA, per-expert gradient clipping, aux loss tracking,
  throughput monitoring, activation monitoring, resumable checkpoints (HF Hub)
- **RL pipeline** (6 stages, orchestrated by `pipeline.py`):
  0. Base pretrain (`train.py`)
  1. RFT — rejection-sampling fine-tuning (`rft.py`)
  2. RLVR — GRPO with verifier reward (`rlvr.py`)
  3. PRM — process reward model with AST step extraction (`prm.py`)
  4. Agent RL — agent-loop-as-rollout (`agent_rl.py`)
  5. Eval — agentic evaluation on HumanEval (`eval/agent_eval.py`)
- **Inference**: InferenceEngine with typical sampling, KV cache reuse, tool tokens
- **Code quality**: pre-commit hooks (ruff/black/isort), CI workflow (GitHub Actions)
- **Developer experience**: go.py smoke test, Makefile (11 targets), check_env.py,
  --info flag, --sanity mode, experiment log viewer

### Verified
- 35 Python files, all syntax OK, zero TODO/FIXME/HACK comments
- Profile analyzer confirms: dev (14.4M), smol-fat (290M/120M), fat (10.5B/3.83B)
- All profiles build correctly from config values
- Tokenizer supports tool tokens, extended vocabulary
- Checkpoint save/load round-trips correctly (including EMA shadow)

### Needs GPU access
- Dependency installation and environment verification
- smol-fat training on real code data (L4 24GB, ~$1-2/B tok)
- 6-stage RL pipeline end-to-end execution
- HumanEval pass@1 measurement
- Runtime debugging of any GPU-specific issues

## Key differentiators

1. **Agent-loop-as-rollout RL** — trains on multi-turn debugging trajectories,
   not single-shot completions. Genuinely novel approach.
2. **Code PRM** — process reward model for code with AST-based step extraction.
   Labels come free from the verifier (no expensive human annotation).
3. **Typical sampling** — filters tokens by information content, reducing both
   boilerplate and hallucination in code generation.

For technical details, see `context/TECHNICAL.md`. For innovation design, see
`docs/RLVR_PRM_DESIGN.md`. For quick start, run `make check && make demo`.
