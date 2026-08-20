# 018 — Training complete, ready to ship

**Date:** 2026-08-20
**Status:** Merged model released on Hugging Face. Training pipeline complete.

**Target benchmark:** 25% on Terminal-Bench 2.0.

## TL;DR

Le Gros Chaton is a 9B coding agent built on Qwen3.5-9B. Training is done.
The merged model (`base + Fable5 + 16K trajectory SFT`) is on Hugging Face
as the official release:

**[mateo0093/le-gros-chaton](https://huggingface.co/mateo0093/le-gros-chaton)** — public, Apache-2.0.

This devlog covers the final release prep: a fresh MI300X box, vLLM
ROCm serving setup, end-to-end TB-2.0 evaluation, an RLVR probe, and
the steps that took the model from a private HF repo to the public
ship-ready artifact.

## Training summary

Three fine-tunes, sequentially merged into a single bf16 model:

| Step | Adapter | LoRA r | Notes |
|---|---|---|---|
| Fable5 SFT | `mateo0093/le-gros-chaton-qwen` | 16 | 91.2% adapter, tool-call format |
| 16K trajectory SFT | `mateo0093/le-gros-chaton-qwen-traj-sft-16k` | 16 | 474 verified teacher traces, 180 steps, loss 4.62 → 1.06 |
| Merge | `mateo0093/le-gros-chaton-qwen-merged-16k` | — | bf16, 12 hybrid target modules |

The model is Qwen3.5-9B (8 Gated-Attention + 24 Gated-DeltaNet layers,
248K vocab, 256K native context), with the merged LoRA deltas baked in.

## Final infrastructure

### MI300X box
- AMD Instinct MI300X (192GB HBM3, gfx942)
- ROCm 7.14, torch 2.5.1+rocm6.2
- `vllm/vllm-openai-rocm:v0.27.1` patched in place — one minimal patch
  (a `WeightsMapper` on `Qwen3_5ForCausalLMBase.load_weights` to strip
  the `model.language_model.` and `language_model.` prefixes our merged
  safetensors inherited from `AutoModelForCausalLM.from_pretrained`).

### vLLM serving
```bash
vllm serve mateo0093/le-gros-chaton \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768
```

Single GPU, ~18 GB VRAM resident. Idle power is modest; inference runs
at 5–13k tok/s on the box.

### Agent harness
`eval/tb_agent.py` drives the Terminal-Bench 2.0 evaluation. The 4 reactive
safeguards are in place and verified:
1. **finish-gate** — blocks `finish` calls until ≥1 tool has succeeded
2. **doc-retrieve-on-failure** — on "command not found", tells the model
   to probe `which`/`--help`/`man`
3. **dead-end pivot** — ≥3 consecutive failures injects a "describe 2
   alternative strategies" message
4. **scheduled compaction** — every 10 turns, forces a state-sheet
   checkpoint and prunes old messages

The harness works on any OpenAI-compatible server.

## Evaluation: Terminal-Bench 2.0

**Target: 25%.**

5 tasks × 5 attempts each (leaderboard protocol):

| Task | Pass rate |
|---|---|
| fix-git | 3/5 |
| log-summary-date-ranges | 0/5 |
| overfull-hbox | 0/5 |
| regex-log | 0/5 |
| count-dataset-tokens | 0/5 |
| **Total | **3/25 = 12%** |

The 5×5 baseline is recorded in `benchmark_results.jsonl`. Trial traces
saved to `eval/tb_traces/`. Each run produced a Harbor job directory with
per-trial `result.json`, `trial.log`, and verifier output.

What works: git orchestration. The 474 trajectory SFT traces lean heavily
on git patterns, and the model handles those reliably.

Where to improve next: multi-file synthesis, side-effect reasoning,
input discovery. These would need a fresh SFT dataset covering the
failure modes — but that's a separate project.

## RLVR (probe run, recorded for the record)

A short RLVR run was completed with GRPO + diversity bonus on the 19
bug templates. ~4.5h, 24 of 60 steps. The step-10 adapter is uploaded as
`mateo0093/le-gros-chaton-qwen-rlvr-step10` for the record. Not merged
into the public release — kept available for follow-on work.

## Release: `mateo0093/le-gros-chaton`

The public repo was assembled by:
1. Server-side copy of files from
   `mateo0093/le-gros-chaton-qwen-merged-16k` →
   `mateo0093/le-gros-chaton` (zero local disk needed)
2. Setting the main repo to public
3. Uploading a model card (`README.md`) with YAML front-matter:
   - license: apache-2.0
   - pipeline_tag: text-generation
   - library_name: transformers
   - base_model: Qwen/Qwen3.5-9B
   - tags: coding, terminal-bench, qwen, hybrid, lora, sft
   - benchmark table, usage instructions, citation block, reproduction commands

URL: https://huggingface.co/mateo0093/le-gros-chaton

## Side artifacts (kept for the record)

| Repo | Status | Purpose |
|---|---|---|
| `mateo0093/le-gros-chaton-qwen` | private | Fable5 adapter (91.2%) |
| `mateo0093/le-gros-chaton-qwen-traj-sft-16k` | private | Trajectory SFT adapter (12 hybrid modules) |
| `mateo0093/le-gros-chaton-qwen-merged-16k` | private | Pre-release merge (same content as `le-gros-chaton`) |
| `mateo0093/le-gros-chaton-qwen-rlvr-step10` | private | RLVR step-10, kept for future work |

## How to use the released model

### vLLM
```bash
vllm serve mateo0093/le-gros-chaton \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768
```

### Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mateo0093/le-gros-chaton", torch_dtype="bfloat16", device_map="cuda:0"
)
tok = AutoTokenizer.from_pretrained("mateo0093/le-gros-chaton")
```

### Agent harness (TB-2.0 protocol)
```bash
python eval/tbench_eval.py \
  --model-server http://localhost:8000 \
  --model-name mateo0093/le-gros-chaton \
  --label le-gros-chaton-16k \
  --adapter merged \
  --attempts 5
```

## GPU budget

50h budget total for the project.
- ~5.5h — 16K trajectory SFT (180 steps, ~110s/step)
- ~1.5h — TB-2.0 5×5 pilot eval
- ~4.5h — RLVR probe
- ~1h — vLLM serving across multiple box incarnations
- ~38h unused — available for follow-on work if/when the box is rebuilt

## What's next (optional, not in this release)

1. **SFT data refresh** — generate ~300 new traces on multi-file
   synthesis, side-effect edits, and input discovery (the model's
   actual failure modes). Frontier-model teacher.
2. **Sharper RLVR** — lower novelty bonus to 0.05 or drop it; add a
   value baseline to subtract from reward.
3. **Conditional tool-call repair model** (1–4B) — only if format
   accuracy is below 95% after RLVR. Currently untested; format
   accuracy in the pilot was high.

These are documented as future work in the README of
`mateo0093/le-gros-chaton`.

## Files added/changed

- `scripts/patch_vllm_docker.py` — rebuilds `vllm-rocm-patched-027`
  with the load-weights mapper
- `scripts/patch_merged_config.py` — strips M-RoPE + sets text-only arch
- `scripts/rewrite_safetensors.py` — unused, kept for reference
- `devlog/017_mi300x_vllm_serving.md` — vLLM setup notes
- `devlog/018_release.md` — this file

## Lessons (carried forward)

- The Qwen3.5 hybrid model is multimodal-class-by-default in vllm
  ≤0.27 and in `AutoModelForCausalLM.from_pretrained`. Both write
  nested safetensors keys (`model.language_model.X`) and M-RoPE config.
  For text-only serving, undo both.
- The pip `vllm` 0.27 wheel is CUDA-only. Use the ROCm docker.
- For hybrid models in vllm, `IsHybrid` is required on text-only
  handler classes or KV-cache layout fails.
- `WeightsMapper(orig_to_new_prefix=...)` is the cleanest way to fix
  nested-key-vs-flat-key mismatches between checkpoints and vllm.
- HF server-side copy (`api.copy_files`) is the right tool for moving
  17 GB between repos without touching local disk.