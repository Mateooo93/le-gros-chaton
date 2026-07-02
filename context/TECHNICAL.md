# Technical state — what's built and how it fits

Reference doc. Read this to know what exists, what each file does, and the dependency rules (so you don't break imports).

## The flat-import rule (DO NOT BREAK)

All core Python uses **flat imports**:

```python
import config as cfg
from model import GPT
from data2 import get_batch
```

No package prefixes. This means the core files **must stay at the project root**. If you move them into `src/`, every import breaks, and the Colab/Modal notebooks that run `python -u train.py` from root break too.

Same applies to weights (`model.pt`, `*_finetuned.pt`) and data (`*_tokens_*.bin` memmaps). They're referenced by **relative path from cwd**, so they stay at root.

## Core files (root, flat imports)

| File | Role |
|---|---|
| `model.py` | The transformer. GPT-2 style + RoPE + RMSNorm + SDPA + KV cache. MoE branch (gate, top-k routing, load-balance aux loss). SwiGLU / GQA / shared expert toggles. Fat guard in `__main__` (refuses to build on <16GB GPU). |
| `config.py` | Profile system: `dev` / `smol-fat` / `fat`. Env-overridable everything. |
| `train.py` | AdamW + fp16 + GradScaler + gradient accumulation + cosine LR. Resume logic. Data-source switch (`CHATON_DATA`). |
| `checkpoint.py` | Resumable checkpoints (model + optimizer + step + scaler + config snapshot). HF Hub push/pull for VM-hopping. |
| `data2.py` | WikiText streaming memmap. Namespaced by `(corpus, vocab)` so switching corpus auto-rebuilds. |
| `data_code.py` | Code corpus streaming (starcoderdata / smollm-corpus / stack-v2). Bounded memmap. Cloud-only (downloads GBs). |
| `tokenizer.py` | GPT-2 BPE via tiktoken. `VOCAB_SIZE=50257`, `EOT_TOKEN=50256`. |
| `chat.py`, `chat_finetuned.py`, `finetune.py` | Instruct-tuning + chat entrypoints (the old 17M dense line). |

## Packages

| Dir | Role |
|---|---|
| `agent/` | Terminal agent harness. `sandbox.py` (exec + timeout + dangerous-pattern block), `loop.py` (generate -> parse `<cmd>` -> exec -> feedback -> repeat). |
| `verify/` | `verifier.py` — does a candidate solution pass the tests? Per-test granularity. **Keystone module**, every RL stage depends on it. |
| `eval/` | HumanEval pass@k harness. `humaneval_loader.py` + (coming) `eval.py`. |
| `docs/` | `RLVR_PRM_DESIGN.md` — the 4-stage innovation design. |
| `devlog/` | Student-voice project journal. |
| `context/` | This folder. The brief. |
| `notebooks/` | Colab + Modal launchers. |
| `models/` | Old weights (keepsakes). |

## Profiles (`CHATON_PROFILE`)

| Profile | Params | Use |
|---|---|---|
| `dev` | 17.6M dense | Local 2070 dev only. |
| `smol-fat` | 240M / 63M-active MoE | Pipeline proof + Phase 2 coder. Fits T4 (squeezed) / L4 (comfortable). |
| `fat` | 10.25B / 3.65B-active MoE | The real target. A100-80GB only. NEVER run locally. |

## The data plan

- **Phase 1 (now)**: wikitext-2. Throwaway. Proves pipeline.
- **Phase 2**: `starcoderdata` python. First real coder.
- **Phase 3**: `starcoderdata` python + ~15% `cosmopedia` prose. Fresh init. The real fat coder.
- **No warm-start** from prose models. General knowledge enters via corpus mix, not weights.

## Secrets

`gpus.md` holds tokens (HF, Modal, Kaggle). It is **gitignored (line 14)** and must never be committed or printed in full. The HF token was rotated once after a leak. Do not let it leak again.

## What's verified working

- MoE builds + backprops at dev/sm-fat/fat (fat on paper only).
- Checkpoint round-trip (push to HF Hub, pull on fresh VM, weights + optimizer + step all match).
- Verifier: correct solutions pass, broken ones fail with correct per-test detail.
- Agent sandbox: blocks `rm -rf /`, times out hung commands.
- Pipeline runs on Colab T4 (training now, silent until step 250).

## What's not yet built

- `eval.py` pass@k runner (loader exists, runner coming).
- `rft.py`, `rlvr.py`, `prm.py` (design done in `docs/RLVR_PRM_DESIGN.md`).
- Phase 2/3 actual training runs.
