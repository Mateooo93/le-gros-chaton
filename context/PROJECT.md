# le fat chaton — PROJECT CONTEXT

> **Read this first.** This is the single source of truth for what this project is,
> what state it's in, and what the quality bar is. If you only read one file, read this.

## What this is

`le fat chaton` ("the fat cat") is a **from-scratch coding agent** built by a solo
student. Not a fine-tune of an existing model. Not a wrapper around Qwen-Coder.
A transformer written from PyTorch primitives, trained from random init, that
needs to become good enough at code to **mog terminal-bench** (do agentic coding
tasks: read a task, write code, run it in a shell, read the errors, fix it,
repeat until it works).

The small sibling `le gros chaton` (17.6M dense, wikitext, val loss 3.73) was the
learning artifact. `le fat chaton` is the real thing: a ~10B Mixture-of-Experts
coder, trained on code, that runs an agent loop in the terminal.

## The quality bar — THIS IS THE KEY POINT

**This model needs to be really, really good.** Not "good for a student project."
Not "it kind of works." Genuinely competitive with open coder models in its class.
The whole point is to prove a solo builder with free GPUs and $30 of credits can
produce something that punches above its weight through smart architecture,
good data, and a strong agent harness.

Concretely, "good" means:

- **It writes correct code.** Measured by `pass@1` / `pass@5` on HumanEval-style
  tasks, climbing toward the DeepSeek-Coder-V2-Lite range (HumanEval ~80+). Not
  val loss. Val loss is necessary but meaningless on its own.
- **It can do agentic loops.** Generate a command, run it, read the failure,
  fix it. That's where most of a terminal-bench score comes from, and it's what
  the `agent/` harness is for.
- **It's fast in the terminal.** Small *active* params (MoE), so it runs like a
  small model but knows like a big one. A coding agent that takes 60s per step
  is unusable.

**Do not cut corners to make it "kind of work."** The honest truth: $30 of Modal
credits cannot fully pretrain a 10B model. That's fine. What matters is that every
dollar goes toward real signal: filtered code data, a correct MoE, verified RL.
If it can't be genuinely good, we'd rather know that honestly than fake it.

## What's done (verified working)

- **Architecture** (`model.py` + `config.py`): MoE + SwiGLU + GQA + shared expert,
  all env-toggleable. Builds + backprops on the 2070 dev profile. Fat math:
  `~10.25B total / ~3.65B active`.
- **Resumable checkpoints** (`checkpoint.py`): model + optimizer + step + scaler.
  HF Hub push/pull verified end-to-end (weights, optimizer momentum, step all
  round-trip). This is the VM-hopping mechanism.
- **Data pipeline** (`data2.py` wikitext, `data_code.py` code): train.py switches
  via `CHATON_DATA=wikitext|code`. Code corpus = starcoderdata python + prose blend.
- **Agent harness** (`agent/`): sandbox (blocks dangerous cmds, times out) +
  tool-use loop (parse `<cmd>`, run, feed output back). Verified.
- **Verifier** (`verify/verifier.py`): runs a candidate solution against tests,
  per-test granularity. The keystone of the whole RL pipeline.
- **RLVR + code PRM design** (`docs/RLVR_PRM_DESIGN.md`): the innovation.
  Process reward models for code, labels free from execution.
- **Training notebooks**: `notebooks/chaton_smol-fat_train.ipynb` (Colab T4,
  pipeline proof), `notebooks/chaton_modal_train.ipynb` (Modal L4).

## What's not done

- **Pipeline proof not yet confirmed live.** The smol-fat run trains but the
  step-250 checkpoint reaching HF Hub is the actual proof it's durable.
- **`eval.py`** (HumanEval pass@k): in progress. This is the metric that turns
  "training" into "are we getting better at code."
- **`rft.py`** (rejection-sampling fine-tune): stage 1 of the RL plan.
- **The fat coder pretrain itself**: Phase 3, the big spend, not started.

## The plan in one line per phase

1. **Phase 1** — pipeline proof on wikitext (free Colab T4). Throwaway weights.
2. **Phase 2** — smol-fat real train on code (Modal L4, ~$3). First real coder.
3. **Phase 3** — fat 10.25B pretrain on code+prose (Modal A100-80GB, the $30).
4. **Phase 4** — the innovation: RFT + RLVR + code PRM + agent harness eval.

See `context/north_star.md` for the why, `context/state.md` for the technical how.
