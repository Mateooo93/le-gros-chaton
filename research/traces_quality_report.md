# Teacher-Trajectory Dataset Quality Report

**Owner:** TraceQuality — **Date:** 2026-08-07
**Scope:** `agent_traces_full.jsonl` (10 verified traces; local + HF `mateo0093/le-gros-chaton-traces`, byte-identical md5 `3198e499…`), the trajectory-SFT ingestion path (`train_qwen.py --trajectory-sft`, `colab/trajectory_sft.ipynb`), and the normalization tool `colab/normalize_traces.py` (new).

---

## 1. Dataset overview

| Metric | Value |
|---|---|
| Traces | 10 (all `verified=True`, all `n_pass == n_total`) |
| Unique bugs | 5 templates × 2 samples (binary_search, sum_evens, is_palindrome, reverse_list, max_subarray) |
| Messages | 222 total (avg 22.2/trace, min 18, max 26) |
| Roles present | `assistant` (160), `user` (62) — **no `system` message anywhere** |
| `[thinking]` blocks | 55 (separate assistant messages prefixed `[thinking]\n`) |
| Teacher field | `moonshotai/kimi-k3-free` on all 10 — but see §2.3 (provenance is suspect) |
| Empty assistant turns | 0 |
| Truncated content | 0 (no message cut mid-tool-call) |

**Token sizes** (real Qwen3.5-9B tokenizer, full ChatML text):

- min 1,308 (is_palindrome_2_0) · max **2,453** (binary_search_5_0) · mean ~1.9K.
- `TRAJECTORY_CTX = 16384` gives **6.7× headroom** on the longest trace. No truncation occurs with current data; the window is not a constraint until traces grow ~7×.

## 2. Tool-call format analysis

### 2.1 Distribution (per tool-call block; some messages contain several)

| Style | Example | Blocks | Notes |
|---|---|---|---|
| Backtick (canonical) | ` ```list_dir\n<path>\n``` ` | 59 | harness + teacher prompt style |
| Fat-cat bracket | `<\|open\|>call tool="read_file"\n<path><\|close\|>argument<\|sep\|>…message<\|sep\|>` | **11** | in 2 traces only |
| Plain bracket | `[tool\nargs]` | 0 | harness accepts it; absent from current data |
| Angle | `<tool>args</tool>` | 0 | harness fallback; absent |

Tool mix (blocks, all styles): `read_file` 25, `run_test` 13, `list_dir` 12, `write_file` 11, `finish` 9, `search_code` 0 — 70 total across 62 call-containing messages.

### 2.2 Why this matters

The 10 traces contain **two different tool-call syntaxes**. SFT on this mix teaches the model both dialects, and the harness only *accepts* backtick / `[tool]` / `<tool>` — the `<|open|>` fat-cat dialect is **not parseable by `agent_swe._parse_actions`**. Mixed formats measurably confuse the small model (it imitates the most-recently-seen style in a context). Fix: normalize to the harness-canonical backtick form before training (done, §5).

### 2.3 Fat-cat calls are unexecuted dead-ends (important)

`teacher_trajectories.parse_tool_calls` only parses backticks and `[tool\nargs]` — it does **not** recognize `<|open|>call tool=…`. So whenever the teacher emitted fat-cat syntax, the loop treated the turn as "no tool call", nudged, and never executed or returned a result. Consequences in the traces:

- **10 unexecuted call messages** in `reverse_list_8_0` (first 13 messages are consecutive assistants — 7 nudged turns before the teacher finally used backticks), 1 in `binary_search_0_0`. All other traces' "unexecuted" messages are just the `finish` terminator (followed by the `[finish]` echo — legitimate).
- After normalization these become backtick calls with **no following tool result** → SFT would train the pattern *"emit a tool call, then emit another call without seeing a result"*, which mismatches the real harness loop (one call → one result → next turn).

Two of the 10 traces are thus degraded training signal even after format normalization. **Recommendation (GenPipeline):** teach `parse_tool_calls` the `<|open|>` syntax so calls execute, or drop unexecuted call messages / regenerate such traces. TraceQuality did **not** drop data (normalizer is format-only by contract).

Provenance note: the `teacher` field says `moonshotai/kimi-k3-free` for all 10, yet 2 traces carry the Fable5-era `<|open|>` dialect — either the free-tier endpoint drifted or those traces came from a non-Kimi run. Worth auditing the current GenPipeline run's output.

## 3. Message-structure findings

1. **No system prompt, no issue message in traces.** The trace `messages` start at the model's first `[thinking]` response; the top-level `issue` field is the only record of the task. The harness always presents `system + "Issue: …"` before the model acts, so SFT was training the model to produce exploration/tool calls **from an empty prompt** — maximally misaligned with deployment. **Fixed in `train_qwen.py`** (§4) by prepending a reconstructed user message; the notebook's inline copy still needs the same (flagged to GPUPrep).
2. **`[thinking]` blocks** are plain-text-prefixed assistant messages (teacher-side convention), not Qwen native thinking tokens. They are assistant-role → **already trained** (verified: first trainable span is `<|im_start|>assistant\n[thinking]…`). Recommendation: **keep them** — they are the self-awareness/planning behavior the project wants baked in; stripping would remove the "think before acting" demonstration. (Optionally revisit native Qwen thinking tokens in RLVR, but don't mix formats.)
3. **Consecutive assistant messages** (thinking + response + call per turn) are fine for ChatML training; the model learns to emit `<|im_end|>` mid-stream, which is standard Qwen behavior.
4. **One nudge echo** (`reverse_list_8_0`): the teacher quoted the nudge text in an assistant message. Harmless, but a symptom of §2.3.
5. **`[finish]` echo** as final user message: masked in training, consistent with harness behavior. Fine.
6. **Self-review is absent everywhere.** `self_review` field is **empty in 10/10 traces**; no message content contains "SELF-REVIEW". Kimi K3 did not follow the teacher prompt's self-review instruction, and the extraction regex found nothing. The `gen_trajectories` path generates a model-produced `_self_review` but stores it **only as a top-level field, not as an assistant message** — so it is never trained either. **The user wants self-review baked as trainable assistant tokens; today it is not in the data at all.** Recommendation (GenPipeline): append the self-review as a final `{"role":"assistant","content":"SELF-REVIEW: …"}` message in the trace so the assistant-mask trains it, and/or make the teacher produce it reliably (it currently prompts for it, but doesn't enforce).

## 4. SFT ingestion audit (`train_qwen.py --trajectory-sft`, notebook)

### Bugs found and fixed (train_qwen.py; notebook flagged to GPUPrep)

| # | Bug | Before | After |
|---|---|---|---|
| 1 | **Crash: dict passed to `dataset.map`** | `dataset = {"messages": […]}` then `dataset.map(...)` → `AttributeError`; `--trajectory-sft` could never run end-to-end | `HFDataset.from_dict({"messages": […]})` |
| 2 | **Task never fed to the model** | First trained token was `<|im_start|>assistant` at position 0 — model learned to act from an empty prompt | Prepend a masked `Issue: …` user message — **conditionally**, only when `messages[0].role != "user"` (new traces carry the real user prompt as `trace[0]`, never double-prefixed) |
| 3 | **Truncation cuts mid-message** | `if len(merged_ids) >= max_length: break` then hard slice — a tool call could be trained half-written | Drop the overflowing message at a clean boundary; only the pathological first-message case is sliced |
| 4 | **Unknown roles trained as assistant** | `else: # assistant → train=True` — a future `role:"tool"` would be trained | Explicit `role == "assistant"` only; unknown roles are masked context |
| 5 | **Dead code** | Full-text `tokenizer(text)` computed then discarded | Removed (verified: per-chunk merging is byte-identical to full-text tokenization for this tokenizer — 0 divergent token positions across all 10 traces) |
| 6 | **Load path** | Always read raw `agent_traces_full.jsonl` | Prefers `agent_traces_normalized.jsonl`; dedupes by `instance_id` on resume |
| 7 | **Not a bug — verified-only ✓** | — | `load_agent_traces_full` already filters `verified`; all current traces also have `n_pass == n_total` |

### Audit answers (the questions asked)

- **Does `tokenize_trajectory_fn` handle system/[thinking]?** System messages would be masked correctly, but none exist in the data; `[thinking]` is plain text inside assistant messages → trained (desired). The notebook's copy uses `f"<|im_start|>{role}\n…"` directly (no per-role branch) — also correct for assistant masking, but lacks fixes #2–#4.
- **Is `TRAJECTORY_CTX=16384` enough?** Yes — longest trace 2.45K tokens ≈ 15% of the window. Keep 16K (5–6× headroom) for the growing corpus; no change needed.
- **Filter verified-only?** Already done. Consider additionally `n_pass == n_total` (currently redundant) and dropping traces with unexecuted calls (§2.3).
- **Is the trailing `"\n<|im_start|>assistant\n"` prompt sane?** Yes — it matches `format_chat`/harness convention. Note it is appended to the full-text string but is intentionally absent from the merged training ids (the final `…<|im_end|>` is the last trained token; nothing to predict after an empty prompt). No weird completions; every real boundary (user result → `<|im_start|>assistant`) is trained.
- **Is self-review baked as trainable assistant tokens?** **No** — it doesn't exist in the data (see §3.6). Nothing to mask until GenPipeline adds it.

## 5. Normalization tool — `colab/normalize_traces.py`

Converts **every tool call in every message** to the canonical harness form:

```
```tool
<args>
```
```

- Rewrites `<|open|>call tool="x"…<|close|>…<|sep|>` (fat-cat), `[tool\nargs]`, `<tool>args</tool>` — only when the block name is a known tool (`read_file`, `write_file`, `search_code`, `list_dir`, `run_test`, `finish`, `prune`), so prose code fences (```` ```python ````) and `[thinking]`/`[finish]` are never touched.
- **No data loss:** same trace count, same message count, same field order; every non-message field and every role byte-identical; malformed blocks are left untouched rather than mangled.
- **Idempotent:** second run converts 0 blocks; output is byte-identical (md5 verified).

Run on the current corpus:

| | before | after |
|---|---|---|
| backtick calls | 59 | **70** |
| fat-cat calls | 11 | **0** |
| bracket / angle | 0 | 0 |
| messages touched | — | 11 |

Output: `agent_traces_normalized.jsonl` (10 traces, 222 messages). Train on this file (train_qwen.py now prefers it automatically).

## 6. Recommendations

1. **Canonical format: backticks** (```tool\nargs```) — the harness's documented format, what the teacher prompt instructs, and what the SFT should see exclusively. ✔ Normalizer enforces this.
2. **Regenerate/filter the 2 fat-cat traces** (or make `parse_tool_calls` accept `<|open|>`) — unexecuted calls train dead-end behavior (§2.3).
3. **Fix task grounding in the notebook** — sync the issue-prepend + boundary truncation + role-mask fixes into `colab/trajectory_sft.ipynb` (GPUPrep), and have it pull `agent_traces_normalized.jsonl` from HF.
4. **Keep `[thinking]`** as trainable assistant text; do not strip. It is the reasoning/self-awareness signal the project targets.
5. **Bake self-review into the trace as a final assistant message** (GenPipeline) — currently 0% coverage; the field-based approach never trains it.
6. **Diversity:** 10 traces = 5 unique fixes (identical patch pairs per template — the novelty filter only compares within the same task index `i`, so repeated templates across `i` slip through). With ~10–60 traces, prefer **more bug templates over more samples**; every duplicate fix is a wasted slot. Raise `--n` template coverage before increasing `--samples`.
7. **Context:** keep `TRAJECTORY_CTX=16384`; no truncation risk at 10× current scale.
8. **Epochs:** with ≤60 traces, 3–5 epochs at eff-batch 8 is reasonable (≈30–180 steps); watch for memorization (identical-patch duplicates inflate it) and hold out 1–2 traces to sanity-check that test pass rate moves, not just train loss.
9. **Verify teacher provenance** on the new run — the `teacher` field is set from an env default, not from per-call API metadata; 2/10 traces' content contradicts it.

## 7. Update (same day) — generator fixes landed, corpus re-normalized

GenPipeline acted on §2.3/§3.6/§6 in `teacher_trajectories.py`:

1. **`<|open|>` dialect now executes** — `parse_open_tag_call()` added; the 2 polluted seed traces (`binary_search_0_0`, `reverse_list_8_0`) were dropped from the local file. Local corpus is now **8 clean traces / 174 messages / 49 backtick calls, 0 fat-cat** (the polluted traces were the *only* carriers of the fat-cat dialect — consistent with this report). `agent_traces_normalized.jsonl` regenerated from the current file; still fully normalized + idempotent.
2. **Self-review is now trainable** — if the finish message lacks a `SELF-REVIEW:` marker, the teacher makes one follow-up call and the review is appended as a final `{"role":"assistant","content":"SELF-REVIEW: …"}` trace message, so the assistant-mask trains it.
3. **Traces now start with the real user prompt** as `trace[0]` (`{"role":"user","content":"Fix this issue in the repo at <dir>:\n\n<issue>"}`).

Corresponding ingestion change in `train_qwen.py`: the task-grounding prepend is now **conditional** — a reconstructed `Issue: …` user message is only prepended when `messages[0].role != "user"` (legacy traces), never double-prefixing the new user-first traces. Verified on both shapes (8 old-style traces prepended; a synthetic new-style trace trusted verbatim; zero double-prefixes). GPUPrep's `trajectory_sft.py` should mirror this once GenPipeline's new traces reach HF. One note: the new traces embed the ephemeral temp repo path in the user message — harmless, since the harness also embeds the real repo path and the model only needs to echo it.

## 8. Verification performed

- Normalizer: conversion 11→0 fat-cat, idempotency (0 conversions on 2nd run, byte-identical md5), round-trip (222 messages, all fields/roles intact).
- Ingestion fix: end-to-end `load_agent_traces_full → Dataset.from_dict(issue-prepend) → map(tokenize)` on the real Qwen3.5 tokenizer — 10/10 rows tokenize, max 2,453 tokens, issue masked, first trainable span = `<|im_start|>assistant\n[thinking]`, zero non-backtick calls, no mid-message cuts.
- `train_qwen.py` compiles (`py_compile`); `--trajectory-sft` path now reaches the Trainer with a valid tokenized Dataset.
