# Le Gros Chaton — Coding Agent Training Framework

Train **production-quality coding agents** that rival GLM-5.2, Opus 4.7, and
other frontier models on Terminal-Bench, SWE-Bench, and coding benchmarks.

**Two paths:**
1. **Fine-tune existing models** (recommended): Qwen2.5-Coder-7B with QLoRA on
   your L4 GPU (`finetune_qwen.py`). Or Qwen3-32B on A100 via Modal.
2. **From-scratch MoE** (research): ~10.5B/3.83B-active MoE with state-of-the-art
   innovations (MLA, quantile balancing, routing-free MoE, etc.).

**Key innovations (all research-backed):**
| Technique | Source | Benefit |
|-----------|--------|---------|
| Self-play data generation | SSR (ICML 2025) | Unlimited training data |
| Test-time compute scaling | DeepSWE | +~17% SWE-Bench |
| Proportional rewards | DeepSWE/SSR | Denser RL signal |
| Quantile Balancing | Kimi K3 | Better MoE routing |
| Routng-Free MoE | arXiv 2604.00801 | Next-gen MoE design |

## The local-rivalry stack (Qwen3.5-9B, 4-bit, runs locally)

| Layer | Tool | Purpose |
|-------|------|---------|
| Train | `train_qwen.py` | SFT on Fable5 (160K) + GRPO |
| Data | `filter_dataset.py` | Quality filtering |
| Data | `distill_reasoning.py` | Learn from big-model reasoning |
| Data | `agent_swe.py --selfplay` | Self-play bug inject/fix |
| Data | `agent_traces.jsonl` | Learn from own successful runs |
| Agent | `agent_swe.py --tdd` | Test-first repair loop |
| Agent | recovery + context mgmt + tool cap | Robustness |
| Inference | `vote_solutions.py` | Verifier/LLM-judge voting |
| Serve | `serve_qwen.py` | Local 4-bit HTTP + chat |
| Measure | `eval_toolcalls.py` | Tool-call format accuracy |
| Measure | `eval_swebench.py` | Real GitHub issues |
| Track | `benchmark_tracker.py` | Results + trends |

**Baseline:** Qwen3.5-9B = 9.2% Terminal-Bench 2.0. Target: 25-35%.
See `docs/BENCHMARK_TARGETS.md` and `docs/TRAINING_PLAN.md`.

## Terminal-Bench 2.0 evaluation (the 30% proof)

TB 2.0's official harness is **Harbor** (`pip install harbor` — the v1 `tb`
CLI is not used for TB 2.0). The harness in `eval/tbench_eval.py` wraps it:
the agent (`eval/tb_agent.py`, a Harbor `BaseAgent`) runs our SWEAgent-style
tool loop (```tool\nargs``` / `[tool\nargs]` / `<tool>args</tool>`), with a
new general-purpose `run_cmd` bash tool added to `agent_swe.TOOLS`. Every
tool executes as a shell command **inside the official Docker sandbox**
(`BaseEnvironment.exec`), and the task is resolved by Harbor running the
task's `tests/test.sh` verifier — identical methodology to tbench.ai.

```bash
# 1. Install + verify (89 tasks in TB 2.0)
uv pip install --python .venv/bin/python harbor modal
python eval/tbench_eval.py --list

# 2. Dry-run: verify sandbox + agent loop paths with a mock model (no GPU)
python eval/tbench_eval.py --dry-run --tasks fix-git

# 3. Serve the model (OpenAI-compatible) — any of:
#    a) LOCAL GPU box (validated: RTX 2070 8GB, Q4_K_M GGUF, Vulkan, ~44 tok/s):
#       ./tb-local/bin/llama-b10307/llama-server \
#           -m tb-local/models/Qwen3.5-9B-Q4_K_M.gguf -ngl 99 \
#           --host 127.0.0.1 --port 8001 -c 12288 --jinja
#       (Q4_K_M gguf from unsloth/Qwen3.5-9B-GGUF, ~5.7GB. NOTE: a Q4 9B
#       pegs an 8GB laptop GPU at ~90C — run full evals when the user is
#       away, or on a proper GPU box. Keep -c <= 12288 and pass
#       --server-ctx-limit 10000 so history stays inside the context.)
#    b) Modal vLLM (SERVE_MODEL=... VLLM_API_KEY=<token> python modal_serve_qwen.py)
#    c) any other vLLM/OpenAI server; or in-process: --local-model Qwen/Qwen3.5-9B

# 4. BASELINE pilot (base model, no adapter) — zero-infra via HF Inference:
export HF_TOKEN=<token>   # any HF token with inference access
python eval/tbench_eval.py --run --hf-inference \
    --label "Qwen3.5-9B-baseline" --adapter base
#    (the harness disables Qwen3.5 thinking mode via chat_template_kwargs;
#    the free tier is fine for pilots)

# 5. FULL EVAL (89 tasks, leaderboard protocol — 100-turn cap, same agent,
#    containerized, unmodified timeouts):
python eval/tbench_eval.py --run --n-tasks 89 \
    --model-server https://<modal-url> --model-name <merged-checkpoint> \
    --model-api-key <token> --label "le-gros-chaton-traj-sft" --adapter traj_sft
#    (RLVR final: same command, --adapter rlvr --label le-gros-chaton-rlvr)

# 6. Results: appended per-task rows in benchmark_results.jsonl; summary on
#    stdout. A modal-verified run can be submitted to the tbench.ai
#    leaderboard (their team re-runs and verifies).
```

Cost/time: ~$0.05-0.10/task (9B on L4 ≈ $0.80/hr; 5-40 turns/task).
Full 89-task run ≈ 5-9 hrs / **~$5-8** on one L4. Wall time per task is
bounded by the task agent timeout (900s, unmodified).

## Start here

**Read [`context/`](context/) first** — it's the single source of truth. In order:

- [`context/PROJECT.md`](context/PROJECT.md) — what this is, the goal, the plan, status.
- [`context/QUALITY_BAR.md`](context/QUALITY_BAR.md) — the standard. Genuinely good, no cope. Read before any decision.
- [`context/TECHNICAL.md`](context/TECHNICAL.md) — architecture, data, what's built, what works, what's blocked.

The build journal lives in [`devlog/`](devlog/). The innovation design
(code PRM + RLVR) is in [`docs/RLVR_PRM_DESIGN.md`](docs/RLVR_PRM_DESIGN.md).

## ⚠️ Do NOT move the core files (flat-import rule)

All core Python uses **flat imports** with no package prefix:

```python
import config as cfg
from model import GPT
from data2 import get_batch
```

This is **intentional and load-bearing.** The core files **must stay at the
project root**:

`model.py` · `config.py` · `train.py` · `checkpoint.py` · `data2.py` · `data_code.py` ·
`data.py` · `tokenizer.py` · `chat.py` · `inference.py` · `pipeline.py` · `go.py` ·
`modal_run.py` · `rft.py` · `rlvr.py` · `best_of_n.py` · `prm.py` ·
`agent_rl.py` · `log.py` · `profile_analyzer.py` · `pyproject.toml`

Moving any of them into `src/` breaks:
1. every flat import,
2. the Colab/Modal notebooks that run `python -u train.py` from root,
3. the relative-path references to weights (`model.pt`, `checkpoint.pt`) and
   token memmaps (`*_tokens_*.bin`).

Same applies to the active weights and data files referenced by relative path —
they live at root. Keepsake/old weights go in [`models/`](models/).

## Quickstart

```bash
# === QWEN FINE-TUNING (recommended path) ===
# Install: pip install transformers accelerate peft bitsandbytes trl

# Inspect a Qwen model
python finetune_qwen.py --model Qwen/Qwen2.5-Coder-7B --mode inspect

# Generate self-play training data
python finetune_qwen.py --model Qwen/Qwen2.5-Coder-7B --mode self-play --problems humaneval --limit 10

# RLVR training (GRPO with proportional rewards)
python finetune_qwen.py --model Qwen/Qwen2.5-Coder-7B --mode rlvr --problems humaneval --n-steps 200

# Kaggle notebook (runs on L4 24GB)
# -> notebooks/kaggle_qwen_finetune.ipynb

# === FROM-SCRATCH TRAINING (research path) ===
# Quick start — verify your environment
make check
make demo

# Train (dev profile, tiny, runs on an 8GB GPU like a 2070)
CHATON_PROFILE=dev python train.py
CHATON_DATA=code     python train.py      # data_code.py — cloud code corpus (default)
CHATON_DATA=wikitext python train.py      # data2.py — local wikitext

# Full 6-stage pipeline (runs all stages sequentially)
python pipeline.py --profile smol-fat --stages all
python pipeline.py --profile smol-fat --stages 0 1 2      # resume from stage 0-2
python pipeline.py --list-stages

# Evaluate
python eval/eval.py --ckpt model.pt --n 20 --ks 1 5 --limit 10 --sanity   # wiring test
python eval/eval.py --ckpt model.pt --n 20 --ks 1 5 --limit 164           # full eval

# Agent loop
python -m agent.loop "list the .py files and count them" model.pt

# Interactive chat with unified inference engine
python inference.py --ckpt model.pt --typical-p 0.2

# RL pipeline stages (individual, after base training):
python rft.py collect --ckpt model.pt --problems humaneval --n 32
python rft.py train rft_data.json --ckpt model.pt --out model_rft.pt
python rlvr.py --ckpt model_rft.pt --problems humaneval --n 8 --out model_rlvr.pt
python best_of_n.py --ckpt model_rlvr.pt --n 32
python agent_rl.py --ckpt model_rlvr.pt --problems humaneval --n-steps 50

# Profile analysis (no GPU needed)
python profile_analyzer.py

# The real target (NEVER run locally — needs A100-80GB)
CHATON_PROFILE=fat python modal_run.py
```

## Profiles

| Profile | Params (total / active) | Architecture | Use |
|---|---|---|---|
| `dev` | 14.4M dense | SwiGLU, RMSNorm | Local 2070 dev |
| `smol-fat` | 290M / 120M active MoE | SwiGLU, GQA, 8 experts, shared-expert | Pipeline proof (T4/L4) |
| `fat` | 10.5B / 3.83B active MoE | SwiGLU, GQA, 16 experts, shared-expert | Real target (A100-80GB) |

All overridable via env vars — see [`config.py`](config.py).

### Kimi K3 innovations (research-backed)
| Innovation | Env flag | Benefit |
|-----------|----------|---------|
| Quantile Balancing | `CHATON_QUANTILE_BALANCE=1` | Hyperparameter-free MoE routing |
| SiTU Activation | `CHATON_USE_SITU=1` | Prevents dead MoE neurons |
| Latent MoE | `CHATON_MOE_LATENT_DIM=256` | Compressed routing |
| Learned Residual | `CHATON_LEARNED_RESIDUAL=1` | Per-layer scaling |
| Self-Play Data | `self_play_data.py` | SSR-style training data |
| Test-Time Scaling | `--n-samples N` | +~17% SWE-Bench gain |

## RL pipeline (6 stages, orchestrated by `pipeline.py`)

| Stage | Module | What it does |
|---|---|---|
| 0 — Base pretrain | `train.py` | MoE pretrain on code corpus (WSD schedule) |
| 1 — RFT | `rft.py` | Sample → verify → SFT on passing solutions |
| 2 — RLVR | `rlvr.py` | GRPO with verifier reward (no critic) |
| 3 — Code PRM | `prm.py` | Step-level reward model (process reward) |
| 4 — Agent RL | `agent_rl.py` | Agent-loop-as-rollout RL |
| 5 — Eval | `eval/agent_eval.py` | Full agentic evaluation on HumanEval |

```bash
python pipeline.py --profile smol-fat --stages all
```

See [`docs/RLVR_PRM_DESIGN.md`](docs/RLVR_PRM_DESIGN.md) for the full innovation plan.

## Repository structure

```
root/                   # Flat imports — DO NOT MOVE
├── model.py            # MoE transformer (RoPE, GQA, SwiGLU, RMSNorm, QK-norm)
├── config.py           # Profiles + env-overridable everything + validation
├── train.py            # Training loop (resumable, throughput + activation tracking)
├── inference.py        # Unified inference engine (chat + programmatic API)
├── pipeline.py         # 6-stage RL pipeline orchestrator
├── rft.py              # Rejection-sampling Fine-Tuning (Stage 1)
├── rlvr.py             # GRPO training (Stage 2)
├── prm.py              # Process Reward Model (Stage 3)
├── agent_rl.py         # Agent-loop-as-rollout RL (Stage 4)
├── best_of_n.py        # Best-of-N inference scaffold
├── log.py              # Experiment logger (crash-safe JSONL)
├── profile_analyzer.py # Static FLOP/memory/throughput analysis
├── checkpoint.py       # HF Hub VM-hopping checkpoints
├── data{2,_code,}.py   # Data pipelines (lazy init, dedup, edu filter)
├── tokenizer.py        # tiktoken GPT-2 BPE + tool tokens
├── agent/              # Terminal agent harness (KV-cache optimised)
├── verify/             # Verifier (keystone of RL pipeline)
├── eval/               # HumanEval pass@k + agentic eval harness
├── tests/              # pytest suite (21+ tests)
├── context/            # Single source of truth docs
├── docs/               # Innovation design docs
├── notebooks/          # Colab + Modal launchers
└── pyproject.toml      # Project metadata + pre-commit / ruff config
```

## Secrets

`gpus.md` holds tokens (HF, Modal, Kaggle). It is **gitignored** and must never
be committed or printed in full.