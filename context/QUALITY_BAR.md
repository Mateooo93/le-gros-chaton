# The Quality Bar — read this first

## What we are building

A **coding agent** that mogs on terminal-bench. Not a toy. Not a demo. Not "it kinda works sometimes." A model you can hand a real task and it goes off, runs commands, reads errors, fixes its own bugs, and delivers working code.

The benchmark target is **terminal-bench** (agentic coding in a real shell). The model class we're chasing is **Qwen-Coder / DeepSeek-Coder**. Those are the floor, not the ceiling.

## The standard is HIGH

This is non-negotiable and stated up front so no one (future Claude session, or me) waters it down:

- **This model must be genuinely good at code.** Not "good for a student project." Good. HumanEval pass@1 in the 70s+. The ability to write working Python on a real prompt.
- **No cope.** If a number is bad, say it's bad. Do not rationalize a weak result as a success. `val loss went down` is meaningless if `pass@1` is near zero.
- **Every improvement is measured.** The eval harness (`eval.py`, HumanEval pass@k) is the source of truth, not training loss. If we can't measure it, we don't claim it.
- **Beat the obvious baseline.** A 0.5B Qwen2.5-Coder off the shelf gets HumanEval ~70+. Our model has to be in that conversation, or we change the plan honestly rather than pretending.

## Why this is hard (so we respect it)

- **Pretraining a great base from scratch solo is near-impossible.** Chinchilla says a 10B model needs ~200B tokens. That is cluster-scale compute and thousands of dollars. We have $30 + free tiers. We will NOT match Qwen's pretrain. Full stop. Own this.
- **So the wins have to come from elsewhere.** The architecture (MoE, so it runs fast), the data (filtered educational code, not raw dump), the RL (verified execution = free perfect labels), and the agent harness (test-driven self-repair = score multiplier). The recipe, not the brute compute.
- **The novelty is the code PRM.** No frontier lab ships a step-level reward model for code. Math PRMs exist (OmegaPRM, Math-Shepherd). Code PRMs are underexplored. Execution gives free per-step labels that math can't get. This is where a solo builder can actually contribute something new. See `docs/RLVR_PRM_DESIGN.md`.

## What "good" looks like, concretely

| Milestone | Metric | Target |
|---|---|---|
| smol-fat baseline | HumanEval pass@1 | > 0 (proves it learned code at all) |
| fat base pretrain | HumanEval pass@1 | 25-40% |
| after Stage1 RFT | HumanEval pass@1 | +5-10 pts |
| after Stage2 RLVR | HumanEval pass@1 | 50%+ |
| agent harness + best-of-N | terminal-bench tasks solved | meaningfully above raw completion |

If we land below these, we diagnose honestly, not declare victory.

## What we are NOT doing

- Not shipping a chatbot. General chat is a side effect, not the goal.
- Not warm-starting from the wikitext model. Different architecture, and warm-starting a coder from prose wastes compute unlearning prose. Fresh init on a mixed code+prose corpus.
- Not shrinking the vocab. Code has rare tokens. Full 50k GPT-2 BPE stays.
- Not pretending $30 pretrains a 10B. It doesn't. We get a real partial, and the RL/harness stages are where the real value compounds.

## The honest North Star

> Build a coding agent that, given a terminal task, actually solves it. Measure everything. Be honest when it's bad. Earn the wins through architecture + data + RL + harness, not cope.

— le fat chaton
