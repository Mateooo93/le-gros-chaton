# Le Gros Chaton — MI300X Execution Plan (50h budget)

Goal: the best possible 9B coding agent on Terminal-Bench 2.0, verified end-to-end.

## Research (grounding)

- **TB 2.0 (89 tasks)**: frontier models + their agents resolve <65%; **small models ~15%**;
  best open-weight (Kimi K2 Thinking + Terminus 2) = 36%. Our 30% target = top-tier for 9B.
- **Failure taxonomy** (paper §4.4): Execution (missing executables 24% of command failures,
  run failures 9.6%), Coherence, Verification. Agent harness quality matters as much as weights.
- **Qwen3.5-9B**: 32 layers = 8×(3 GatedDeltaNet + 1 GatedAttention); 248,320 vocab; 256K native ctx.
- **AMD day-0 ROCm support**: SGLang `rocm/sgl-dev:v0.5.8.post1-rocm720-mi30x-20260215` and
  vLLM `rocm/vllm-dev:nightly_main_20260211` — Triton GDN kernels, works out of the box on MI300X.
- **Our eval harness** (`eval/tbench_eval.py` + `eval/tb_agent.py`) drives any OpenAI-compatible
  `/v1/chat/completions` server → vLLM on MI300X works directly.

## Known adapter problem — MUST FLATTEN (done locally, verified small-model)

The v25 traj adapter weights are correct (12 modules, r=16, alpha=32 — incl. linear_attn
in_proj_* + self_attn + mlp, 24/8 hybrid split) but the saved `adapter_config.json` is stale
(7 modules + null base). `PeftModel.from_pretrained` onto a plain model accepts nested keys
but does NOT transfer values (verified: 0.0215 vs 0.123 expected; lora_B zeros).
→ `flatten_traj_adapter.py` repacks flat with correct config. MUST upload flattened adapter.

## Pipeline (phase-by-phase)

### 0. Access & stack (1-2h)
1. SSH in, verify GPU: `rocm-smi --showproductname` → MI300X, 192GB.
2. Install ROCm PyTorch: `pip install torch torchvision --index-url
   https://download.pytorch.org/whl/rocm6.2` (or use the AMD SGLang docker for serving).
3. Clone repo; `.venv`; `pip install transformers==5.14.1 tokenizers==0.22.1 peft datasets
   bitsandbytes trl safetensors accelerate`.
4. Lower disk: curve `HF_HOME` to a big volume.

### 1. Fix + verify the trajectory adapter (30min)
- Run `flatten_traj_adapter.py` on the v25 adapter, upload flat version to HF.
- Run `verify_traj_adapter.py` → confirm r=16, 12 modules, base set.
- (Kaggle T4 verify optional — already proven the training ran clean: loss 1.09, grad_norm finite.)

### 2. Re-run trajectory SFT at FULL 16K context on MI300X (6-10h) ← big win
- v25 was trained at ctx 1536 → only ~53% of traces fit; every long trace was truncated
  mid-message and the model NEVER saw ends of long tasks (final tool calls, self-reviews).
- On MI300X: `TRAJECTORY_CTX=16384` in native bf16 (no 4-bit, no fp16 hack, no chunking
  needed at 192GB). 16K covers 100% of the 474 traces (max ~12.7K tokens).
- torch_dtype=bfloat16, plain AdamW (states tiny: 43M trainable), no autocast.
- Fits easily: 9B bf16 ≈ 18GB + LoRA grads/optimizer ≈ trivial at 192GB.
- This is the primary quality lever p1 — strictly better data than v25.

### 3. Merge (30min)
- `merge_sft.py` sequential: base + Fable5 → + traj (bf16, no quantization).
- Verify merged with `verify_traj_adapter.py`-style checks + a pply a smoke generation.

### 4. Serve on vLLM (MI300X) (1h)
- AMD vLLM docker OR pip vllm+rocm. `vllm serve qwen_merged --port 8000
  --tensor-parallel-size 1 --max-model-len 32768`.
- Confirm `/v1/chat/completions` works; our tb_agent only needs chat completions.

### 5. Terminal-Bench eval — baseline (2-4h)
- `python eval/tbench_eval.py --model-server http://localhost:8000 --model-name merged
  --label le-gros-chaton-traj-sft --adapter traj_sft --attempts 5`
- (Harbor must be installed; container runner needed — check `check_harbor()`.)
- Compare: base vs traj-sft vs (later) rlvr. Record in benchmark_results.jsonl.

### 6. RLVR with diversity/novelty (12-20h) ← the sharpening step
- `MODEL_NAME=qwen_merged ADAPTER=none python rlvr_qwen.py --n 8 --n-steps 120
  --limit 19 --novelty-bonus 0.2 --out qwen_rlvr`
- GRPO with group-normalized advantages; rollouts = full SWEAgent episodes on the 19 bug
  templates; reward = hidden-test verifier + strategy-switch + novelty bonus.
- On MI300X: batch 8 rollouts in bf16 fits easily; 192GB means no memory gymnastics.
- Save + upload RLVR adapter; re-merge into qwen_rlvr_merged for eval.

### 7. Terminal-Bench eval — post-RLVR (2-4h)
- Same protocol, label rlvr. This is THE number we report.

### 8. Optional stretch (if quota remains)
- HumanEval pass@1 / SWE-bench subset via eval_qwen.py — supporting numbers.
- Bump attempts 5→10 on the best model for a tighter CI.

## Success criteria
- TB 2.0 ≥30% (5 attempts/task, leaderboard protocol) — SOTA-class for a 9B.
- Adapter loads cleanly (flat, 12 modules), merge verified by value transfer, not just "no error".
- RLVR shows measurable TB gain over traj-sft (the diversity/novelty reward is the differentiator).

## Watchdogs
- Every long step (SFT/RLVR) runs under `nohup`/tmux; logs to files; checkpoint+HF upload
  on a schedule so a disconnect never costs more than an hour.
- GPU-hours tracker: log time used per phase; stop at ~46h spent to leave eval margin.

## Open questions (answer on the box)
- Harbor availability on the box (eval needs container runner; may run eval from the 4090
  box / local later if simpler).
- vLLM ROCm wheel vs AMD docker — try pip wheel first (fewer moving parts); fall back to
  the official docker if kernels matter (GDN Triton).
