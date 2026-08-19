# 017 — MI300X vLLM serving: getting past the multimodal-class trap

**Date:** 2026-08-19
**Status:** Serving works; pilot run hit a model-quality loop on `fix-git`.

## TL;DR

Old box destroyed, new box `134.199.193.235` provisioned, ROCm stack intact. vLLM
ROCm docker needed five patches to serve our text-only merged Qwen3.5 hybrid
model. The merged model itself also needed two config edits (architecture name +
stripped M-RoPE). End-to-end TB eval pipeline is now functional: 1 task ran
in 151s on the box. The model itself is poor at loop recovery on its own
(harness corrections ignored); this is a model-quality issue, not an infra one.

## What I did

### Box setup
- New MI300X box: `134.199.193.235` (root, ed25519 key already in `~/.ssh/mi300x`)
- `apt install python3.12-venv python3-pip libgomp1`
- Clone repo to `/root/le-gros-chaton`
- Scp `.env` (chmod 600, HF_TOKEN)
- `bash setup_mi300x.sh` — pulled torch 2.5.1+rocm6.2, bf16 matmul sanity OK
- `pip install vllm` — installed vllm 0.27.1 (CUDA-only wheel — see below)
- `pip install amdsmi` — required for vllm 0.16rc2's ROCm platform detection

### Download merged model
- `snapshot_download("mateo0093/le-gros-chaton-qwen-merged-16k")` to `/root/cache/huggingface`
- 17.93 GB, ~1 min

### vLLM serving (this is the hard part)

Tried 3 paths in order; only #3 worked.

**Path 1 — pip `vllm` 0.27.1 (CUDA wheel).** Fails with
`RuntimeError: Failed to infer device type` because the pip wheel has no
`_rocm_C.abi3.so` and ROCm torch reports CUDA via HIP. Even after
`pip install amdsmi` (which vllm needs for ROCm platform detection), the
engine raises `AssertionError: DP adjusted local rank 0 is out of bounds
for 0 devices`.

**Path 2 — docker `rocm/vllm-dev:nightly_main_20260211`.** vllm 0.16rc2
with the right ROCm `_C`. Initially fails with
`TypeError: Invalid type of HuggingFace config. Expected type:
Qwen3_5Config` because the docker's qwen3_5.py only knows the multimodal
class (`Qwen3_5ForConditionalGeneration`) which demands `vision_config`.
Our merged model is text-only (`Qwen3_5TextConfig`, no vision).

Renaming the merged config to `Qwen3_5ForConditionalGeneration` gets past
that error but exposes the next one: `NotImplementedError: The page size
of the layer is not divisible`. The docker only registers the multimodal
class, which routes through the `language_model.X` weight mapper; our
safetensors is written by `AutoModelForCausalLM.from_pretrained` which
loads the multimodal class and produces nested keys like
`model.language_model.X`.

**Path 3 — patch the docker.** Apply 5 patches inside the docker and
commit a new image `vllm-rocm-patched:latest`:

1. **`registry.py`** — add a `"Qwen3_5ForCausalLM"` → `"Qwen3_5ForCausalLM"`
   entry so the architecture resolution routes to the text-only handler.
2. **`qwen3_5.py: typing imports`** — add `from typing import ClassVar, Literal`.
3. **`qwen3_5.py: Qwen3_5ProcessingInfo.get_hf_config`** — fall back to
   `Qwen3_5TextConfig` when the composite `Qwen3_5Config` isn't available.
4. **`qwen3_5.py: Qwen3_5ForCausalLMBase`** — make it inherit `IsHybrid`,
   add `is_hybrid: ClassVar[Literal[True]] = True`, and add
   `get_mamba_state_dtype_from_config`, `get_mamba_state_shape_from_config`,
   `get_mamba_state_copy_func` classmethods (copied from the multimodal
   handler). Without `IsHybrid`, vllm fails with the page-size error.
5. **`qwen3_5.py: Qwen3_5ForCausalLMBase.load_weights`** — add a
   `WeightsMapper` that strips both `model.language_model.` and
   `language_model.` prefixes, so the text-only handler can load the
   nested-keyed safetensors.

After commit, the patched image serves our merged model cleanly.

### Merged config edits
- `architectures`: `[Qwen3_5ForConditionalGeneration]` → `[Qwen3_5ForCausalLM]`
- `rope_parameters`: strip `mrope_interleaved` and `mrope_section` (the
  merge script saved them from the multimodal class; vllm asserts M-RoPE
  is not implemented for text-only)
- Drop `dtype` field (vllm uses `--dtype`)

These edits are scripted at `scripts/patch_merged_config.py` and
`scripts/patch_vllm_docker.py` — idempotent.

### Eval pilot

`.venv/bin/python eval/tbench_eval.py --model-server http://134.199.193.235:8000 \
   --model-name le-gros-chaton --label le-gros-chaton-16k --adapter merged \
   --tasks fix-git --attempts 1`

Result: 0/1 (FAIL in 125s, 100 turns, 42 tool calls).

The pipeline works (Harbor sandbox → tb_agent loop → vLLM completions → verifier
→ results JSONL). The agent just gets stuck on `git show` loops and the model's
recovery from harness corrections is poor. This is the kind of issue RLVR is
designed to address; it's not an infra bug.

## Numbers

| Step                    | Time     | Cost     |
|-------------------------|----------|----------|
| SSH key + box setup     | ~5 min   | 0 GPU    |
| ROCm stack              | ~2 min   | 0 GPU    |
| Model download (17.9GB) | ~1 min   | 0 GPU    |
| vLLM serve cold start   | ~90s     | GPU now  |
| Pilot (1 task)          | 151s     | GPU live |

GPU hours used: negligible so far. vLLM consumes about 18 GB VRAM
(`--max-model-len 32768`, single GPU). Idle power draw is modest.

## What's next

1. **Full pilot (5 tasks × 1 attempt)** — get the baseline TB-2.0 number
   from the merged model.
2. **5-attempt eval** — the headline number.
3. **RLVR** — diversity + novelty reward on the 19 bug templates.
4. **Conditional tool-call repair model** — only if `eval_toolcalls.py`
   accuracy post-RLVR ≤95%.

## Files added/changed

- `scripts/patch_vllm_docker.py` (new) — builds `vllm-rocm-patched:latest`
  from `rocm/vllm-dev:nightly_main_20260211` with the 5 patches applied.
- `scripts/patch_merged_config.py` (rewritten) — strips M-RoPE + sets
  `Qwen3_5ForCausalLM` architecture (previously renamed in the wrong
  direction).
- `~/.ssh/config` — updated `Host mi300x` HostName to `134.199.193.235`.

## Lessons

- The Qwen3.5 hybrid model is multimodal-class-by-default in vllm 0.16rc2
  and in `AutoModelForCausalLM.from_pretrained` (which loads the multimodal
  Qwen3.5 class). Either of these will write nested safetensors keys
  (`model.language_model.X`) and M-RoPE config. Both need to be undone for
  text-only serving.
- The pip `vllm` 0.27.1 wheel is CUDA-only — it has `_C_stable_libtorch`
  but no `_rocm_C`. Don't waste time on the pip wheel for ROCm.
- For hybrid models in vllm 0.16rc2, `IsHybrid` is required even on
  text-only handler classes or KV-cache layout fails.
- fail2ban on the box rate-limits SSH after a burst of probe calls. Wait
  30s between SSH commands when iterating.