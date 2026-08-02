# Research: Making a ≤9B Qwen3.5 Model a Powerful Long-Running Coding Agent

Research date: 2026-08-02. Synthesizes live web research (arXiv, HuggingFace,
ACM/ACL) against our pipeline (Qwen3.5-9B QLoRA SFT + GRPO-style RLVR).

## TL;DR

1. **The single biggest lever is DATA, not training algorithm.** The strongest
   small coding agents (OmniCoder-9B, SWE-Dev, SWE-Next-7B, TMax-9B) are all
   trained on **real agentic trajectories** (tool calls, terminal output,
   error recovery) — not chat-style instruction data. Our Fable5 SFT is
   chat-style; it teaches the model to *answer*, not to *act*.
2. **OmniCoder-9B is the closest reference point to us** — it is literally
   Qwen3.5-9B LoRA-SFT'd (r=64, α=32, all layers incl. MLP) on 425K real
   agentic trajectories from Claude Opus 4.6 / GPT-5.4 / GPT-5.3-Codex /
   Gemini 3.1 Pro, 65K-token sample packing, assistant-token-only loss,
   constant LR 2e-4. Terminal-Bench 2.0 ≈ 23.6-28.1%.
3. **Long-running tasks need long-context + context management training.**
   Long-Horizon-Terminal-Bench: agents average **9.8M tokens per task** —
   far beyond any window. Agents fail not on local steps but on *sustaining
   progress, verifying completion, and staying within budget* (context rot +
   action looping). Training mitigations exist (trajectory splitting, staged
   RL with extended timeouts, self-editing/context-management curricula).
4. **Naive GRPO is unstable for long-horizon agentic RL** — TMax explicitly
   switched to DPPO (masks tokens where inference/train logprobs diverge),
   FP32 LM head, large group size, outcome-only *graded* rewards.
5. **Small models need loop-discouragement.** SWE-Protégé (Qwen2.5-Coder-7B):
   SFT on expert-augmented trajectories + agentic GRPO with shaped rewards
   that penalize action looping → 42.4% SWE-bench Verified (+25.4 over prior
   SLM SOTA), with sparse expert guidance (~4 calls/task, 11% of tokens).

## Evidence per topic

### 1. Small coding-agent models (≤9B) — what works

| Model | Base | Method | Data | Key result |
|---|---|---|---|---|
| **OmniCoder-9B** | Qwen3.5-9B | LoRA SFT r=64 α=32, all layers+MLP | 425K real agentic trajectories (5 sources: Claude Opus 4.6, GPT-5.4, GPT-5.3-Codex, Gemini 3.1 Pro) | Terminal-Bench 2.0 ~23.6-28.1%; 262K native ctx |
| **TMax-9B** | (open recipe) | SFT warm-start + DPPO RL | 14.6K synthetic envs, graded rewards | Terminal-Bench 2.0 27.2% (vs Haiku 4.5 29.8%) |
| **SWE-Protégé** | Qwen2.5-Coder-7B | SFT + agentic GRPO w/ loop penalty | expert-augmented trajectories | 42.4% SWE-bench Verified |
| **SWE-agent-LM-7B** | Qwen2.5-Coder-7B | SFT | 5K trajectories (SWE-smith) | SWE-bench SOTA at release |
| **SWE-Dev-9B** | GLM-4-9B | trajectory SFT | SWE-Dev-train | repo-level agent |
| **SWE-Next-7B** | Qwen2.5-Coder-7B | full-param SFT | execution-grounded trajectories | repository-level agent |

Consensus recipe: **base model → SFT on real, execution-grounded agentic
trajectories → (optional) RL with graded/verifier rewards.**

### 2. Long-running task capability — the failure modes

From Long-Horizon-Terminal-Bench (46 tasks, dense graded rewards, 9.8M
tokens/task avg):
- Agents fail because they **cannot sustain progress, verify completion, or
  finish within budget** — not because individual steps are wrong.
- Two failure types: **timeout-driven incomplete progress** (context rot:
  memory of early context degrades as history grows) and **premature
  stopping / weak self-verification**.
- Action looping (repeatedly issuing the same tool call) is the signature
  failure of small models on long SWE tasks (SWE-Protégé).

### 3. Training for long-horizon capability

- **Trajectory-splitting SFT** (KLong, withdrawn but method stands): split
  long trajectories into overlapping sub-trajectories — preserve early
  context, progressively truncate later context, keep overlap — so the model
  sees full long tasks at train time. Train on 65K+ sequence lengths.
- **Progressive/staged RL** (KLong, TMax): train RL in stages with
  progressively extended timeouts/task length; increase difficulty via
  curriculum (TMax's 9 difficulty axes).
- **Context-management training** (Chroma Context-1, 20B): teach the agent a
  `prune` tool, show it its token budget, staged curriculum recall→precision;
  self-editing context beats raw accumulation. Same idea as ACM / ARC
  (reflection-driven context management), PRO-LONG (programmatic memory).
- **DPPO > GRPO for long-horizon** (TMax): GRPO collapses when rollout groups
  are mostly failures (CRPS paper confirms: sparse terminal rewards → weak
  advantages with group normalization). DPPO masks diverging logprobs, FP32
  LM head, large group size.

### 4. Reward design

- Outcome-only **graded** rewards (partial credit, continuously-valued)
  outperform binary pass/fail (TMax, SERA, our proportional rewards already
  align).
- Explicitly **penalize looping / reward progress** (SWE-Protégé shaped
  rewards, Long-Horizon-Terminal-Bench dense per-subtask grading).
- Self-verification: reward models for finishing correctly, penalize
  premature stopping.

## What this means for OUR pipeline (Qwen3.5-9B, LoRA, $60 Modal + 13h Kaggle)

### Gaps in our current approach
1. **Fable5 is chat data, not agentic trajectory data.** It teaches
   instruction-following, not tool-use/terminal/recovery behavior. This is
   the #1 change to make.
2. **max_length=512** (train_qwen.py default) is far too short for
   trajectories. OmniCoder used 65K-token packing; our 10k/16k runs trained
   on ~512-token chunks — the model never sees a real agent session.
3. **RLVR uses GRPO-style normalization** with group_size=4 on 10 problems —
   the exact regime TMax says is unstable for long-horizon. No loop penalty,
   no progress reward, no self-verification reward.
4. **No context-management training** — the agent harness truncates to last
   ~8 messages (hardcoded), but the model is never trained to prune/decide
   what to retain.

### Recommended changes (in priority order, budget-aware)
1. **Build agentic trajectory data** for SFT: run our own `agent_swe.py`
   harness on SWE-bench-lite / Terminal-Bench tasks with the verifier,
   recording (tool_call, result, error, recovery) traces. Even 5-20K high-
   quality successful traces beats 160K chat rows for agentic capability.
   (Free — our own harness + sandbox.)
2. **SFT on trajectories with long sequence length** (8-16K packed, LoRA
   r=64 α=32 all layers). Requires Modal (Kaggle T4 at 512 ctx was the 10k/16k
   run; long-ctx needs L4/A100).
3. **RL with graded rewards + loop penalty** (upgrade RLVR): proportional
   rewards (have it) + per-subtask dense grading where available + explicit
   action-looping penalty + self-verification term. Prefer DPPO-style masking
   over plain GRPO normalization.
4. **Context-management:** add a `prune`/`forget` tool + token budget to
   `agent_swe.py`, and include context-edit traces in SFT data.
5. **Eval on the right benchmarks:** Terminal-Bench 2.0, Long-Horizon-
   Terminal-Bench (dense graded = better signal than pass/fail), SWE-bench
   Verified — not just HumanEval.

### What NOT to change
- Base model Qwen3.5-9B (proven — OmniCoder used it; 262K native ctx helps
  long tasks; our infra already works).
- QLoRA SFT → RLVR overall flow (matches every successful recipe).
- Checkpoint/resume/HF plumbing (already built).

## Budget reality check
- Fable5 160K chat SFT (~$40-50 on L4 with fast kernels) makes the model
  *more knowledgeable*, not more *agentic*.
- **If budget is tight, prefer: 10-20K agentic trajectories SFT (long ctx,
  ~$10-20) + RLVR (~$5-8) over the full 160K chat SFT.** That maximizes
  Terminal-Bench/SWE-bench agentic score per dollar.
- Best use of the 13h Kaggle quota: generate trajectory data locally or in
  the loop, not chat SFT.

## Sources
- OmniCoder-9B / OmniCoder-2-9B: huggingface.co/Tesslate/OmniCoder-9B, /OmniCoder-2-9B
- TMax (arXiv 2606.23321): terminal-agent RL recipe, DPPO
- SWE-Protégé (arXiv 2602.22124): SLM long-horizon SWE, loop penalty
- Long-Horizon-Terminal-Bench (arXiv 2607.08964): dense graded rewards, failure modes
- KLong (arXiv 2602.17547, withdrawn — method only): trajectory-splitting SFT + progressive RL
- Chroma Context-1: self-editing context agent, staged curriculum
- SWE-Dev, SWE-Next, SWE-smith (SWE-agent-LM-7B), SWE-EVO, CRPS, ACM/ARC
