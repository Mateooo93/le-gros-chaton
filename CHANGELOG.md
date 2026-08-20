# Changelog

All notable changes to Le Gros Chaton are documented here.

## [Unreleased] — 2024-07

### Added — Live review page
- **`docs/index.html`** — GitHub Pages chat box (single-file static page)
  backed by a Q4 (W4A16) quantized vLLM build of the merged model on an
  MI300X, exposed publicly through a Cloudflare quick tunnel.

### Added (Qwen fine-tuning adaptation — 5 new files)
- **`finetune_qwen.py`** — QLoRA + self-play data generation + GRPO training for
  Qwen models. Supports 4-bit quantization, LoRA adapters, group-relative policy
  optimization with proportional rewards from our verifier.
- **`eval_qwen.py`** — HuggingFace-compatible evaluation harness for HumanEval
  pass@k and agentic evaluation with test-time compute scaling (--n-samples).
- **`agent_qwen.py`** — Qwen-compatible agent loop with sandbox execution,
  `<cmd>`/`<done>` tag parsing, and verifier integration.
- **`docs/TRAINING_PLAN.md`** — Complete training roadmap with hardware
  recommendations, benchmark targets vs GLM-5.2, and 6-step pipeline.
- **`notebooks/kaggle_qwen_finetune.ipynb`** — Ready-to-run Kaggle notebook
  for Qwen2.5-Coder-7B QLoRA fine-tuning on L4 24GB GPU.
- **`self_play_data.py`** — Added `--qwen` flag for HuggingFace model support.
- **Makefile** — 5 new Qwen targets: inspect, selfplay, train, eval, agent.

### Added (Kimi K3 research — 4 architectural innovations)
- **Quantile Balancing** (`CHATON_QUANTILE_BALANCE=1`) — Kimi K3-style deterministic MoE routing.
  Replaces aux_loss + gate_bias with quantile-based per-expert thresholds.
- **SiTU Activation** (`CHATON_USE_SITU=1`) — Sigmoid Tanh Unit replaces SiLU in SwiGLU.
  Prevents dead-neuron pathology in rarely-activated MoE experts.
- **Latent MoE** (`CHATON_MOE_LATENT_DIM=256`) — Compress token before routing, up-project
  after expert computation. Kimi K3 Stable LatentMoE-style.
- **Learned Residual** (`CHATON_LEARNED_RESIDUAL=1`) — Per-layer learnable alpha on attention
  residual, inspired by Kimi K3 Attention Residuals.

### Added (Coding agent research — 3 pipeline innovations)
- **Self-Play Data Generation** (`self_play_data.py`) — SSR-style generate/inject/fix pipeline.
  Generates unlimited training data without human annotation.
- **Test-Time Compute Scaling** (`eval/agent_eval.py --n-samples N`) — Multi-trajectory voting
  with varied temperatures, ~+17% SWE-Bench gain (DeepSWE approach).
- **Proportional Rewards** (`agent_rl.py`) — Fraction of tests passed as continuous reward
  signal instead of binary pass/fail.

### Added (Infrastructure)
- **Kimi K3 flags** forwarded to Modal (`modal_run.py`), documented in `.env.example`,
  reported by `check_env.py`, and logged in `--info` output.
- **Research documents** — `research/kimi_k3_findings.md`, `research/coding_agent_findings.md`,
  `research/sera_paper.md` — 6 papers analyzed and saved.
- **`self_play_data.py`** — 180-line self-play data generation script.
- **`CITATION.cff`** — academic citation metadata (CFF 1.2.0 format)
- **`.gitattributes`** — line ending normalization, language-aware diffs, binary markers
- **`verify/_runs/.gitkeep`** — ensure temp directory exists on fresh clones
- **Typical sampling** for RL scripts — `rft.py`, `rlvr.py`, `agent_rl.py` now pass `typical_p=0.2` to `model.generate()`
- **Improved error messages** — `data_code.py` now gives clear install instructions if `datasets` is missing
- **Modal env forwarding** — `modal_run.py` forwards EMA, aux loss, WSD, and other training configs
- **MIT LICENSE file** — proper open-source licensing
- **CONTRIBUTING.md** — contributor guide with setup, conventions, PR workflow
- **CI workflow** (`.github/workflows/ci.yml`) — automated lint, syntax check, and torch-free test suite on push/PR
- **CHANGELOG.md** — this file
- **`--json` output mode** for `inference.py` — structured generation metadata
- **`--sanity` flag** for `pipeline.py` — auto-configures dev profile with tiny data limits
- **`make sanity`** and `make check` Makefile targets

### Fixed
- **`--json` token counting** — uses BPE token count instead of word count
- **smol-fat profile numbers** — config.py/README/TECHNICAL.md now match `profile_analyzer.py` (290M total / 120M active)
- **fat profile layer count** — documented as 32 (matches config.py)
- **`agent_rl.py` default paths** — changed from `models/` to root-level `model.pt`
- **Removed unused `import math`** from `agent_rl.py`
- **Resolved last TODO** — `agent/loop.py` KV cache comment clarified
- **Colab notebook** — replaced stale references to deleted `finetune.py`/`chat_finetuned.py`
- **Config profile comments** — updated all stale param counts to profile_analyzer values

## [0.2.0] — 2024-07

### Added
- **Exponential Moving Average (EMA)** — `CHATON_EMA_DECAY` (default 0.999). Shadow weights averaged over training steps for better eval checkpoints. Saved in checkpoint extra data for resume compatibility
- **`train.py --info`** — prints resolved configuration (all env-var overrides applied) and exits
- **`make info`** Makefile target
- **Per-expert gradient clipping** — MoE experts normalised independently to prevent gradient interference
- **MoE aux loss tracking** — `model.last_aux_loss` and `model.last_z_loss` logged at each eval interval
- **Residual scaling init** — output projections (`c_proj`) initialised with `std = 0.02 / sqrt(2 * n_layer)` to prevent residual stream explosion in deep models
- **`log_view.py`** — terminal-based experiment log viewer with table, summary, and multi-run comparison

### Changed
- **Default data source** changed from `wikitext` to `code` — `CHATON_DATA=code` is now the default

### Fixed
- **`GPT.from_checkpoint()` vocab mismatch** — auto-extends vocabulary when checkpoint has more tokens than config
- **Verifier temp file cleanup** — temp `.py` files in `verify/_runs/` are now deleted after verification

## [0.1.1] — 2024-07

### Added
- **`.env.example`** — documents all 45+ environment variables across 6 categories
- **`Makefile`** — 9 targets: install, test, test-quick, format, lint, demo, serve, clean, info
- **Step extraction tests** — 6 new tests for AST-based PRM step extraction (no torch needed)
- **AST-based step extraction** — `prm.py` now uses `ast.parse()` for accurate step boundaries (handles decorators, nested defs, classes correctly)
- **Token healing** — tool tokens (`<|tool_call|>`, `<|tool_result|>`, `<|done|>`) provide explicit protocol markers instead of fragile XML
- **Typical sampling** — `model.generate(typical_p=0.2)` filters tokens by information content, reducing boilerplate and hallucination
- **`go.py` smoke test** — single-command dev pipeline verification
- **`inference.py`** — unified `InferenceEngine` class with KV cache reuse, chat template, and all sampling strategies
- **`pipeline.py`** — 6-stage RL pipeline orchestrator (pretrain → RFT → RLVR → PRM → Agent RL → Eval)

### Changed
- **`chat.py`** simplified to a thin wrapper around `InferenceEngine`
- **`README.md`** updated with current profile numbers, 6-stage pipeline, and new file list
- **`TECHNICAL.md`** comprehensive rewrite (83 → 175 lines) covering all architecture, data, training, and RL details

### Fixed
- **PRM step merging** — short-step merging no longer merges new function/class definitions
- **Decorator attribution** — AST step extraction now includes decorator lines in function steps
- **Unused imports** — cleaned 10+ dead imports across `prm.py`, `go.py`, and other files
- **Stale file cleanup** — removed 4 wikitext-era files (`finetune.py`, `finetune_data.json`, `chat_finetuned.py`, `train_custom_tokenizer.py`)
- **Stale memmap cleanup** — removed 232MB of stale wikitext token memmaps

## [0.1.0] — 2024-06

Initial release with:
- MoE transformer with GQA, SwiGLU, RoPE, RMSNorm
- 3 profiles: dev (14.4M), smol-fat (56.4M/42.3M), fat (10.5B/3.83B)
- Code data pipeline with syntax validation, educational filtering, MinHash dedup
- WSD learning rate schedule
- QK-normalization, z-loss, dynamic top-k routing
- KV cache compression (StreamingLLM-style)
- RoPE scaling (NTK-aware YaRN)
- RFT, RLVR, PRM, and agent-loop RL training scripts
- Agentic evaluation harness
- Experiment logger
- Pre-commit hooks (ruff, black, isort)
- Test suite (21+ tests)
