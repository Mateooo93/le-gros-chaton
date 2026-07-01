# RLVR + code PRM — the design for "le fat chaton" to beat Qwen-Coder-class

> This is the innovation, not the architecture. Anyone can pretrain a MoE.
> The win comes from *how you train on code you can verify* and *how you pick
> good steps without running to completion*. Read this as a plan, not code yet.

## Why this is the right bet (research, not vibes)

Frontier coding models all converged on the same two ideas in 2024-2026:

1. **RLVR — "hard to solve, easy to verify".** Reinforcement learning where the
   reward isn't a human-label or a bigger model's opinion (RLHF) — it's a
   *verifier*: does the code run? do the tests pass? You don't need a reward
   model at all, you need a sandbox. Qwen3-Coder trained this way ("hard to
   solve, easy to verify" Code RL + 20K-env Agent RL). DeepSeek-R1 proved pure
   RLVR (no SFT cold-start even) can produce reasoning.

2. **Process Reward Models (PRMs) for *step-level* credit assignment.** Instead
   of scoring only the final answer (outcome reward — sparse, slow to learn
   from), score EACH step of a solution. OpenAI's "Let's Verify Step by Step"
   showed PRMs beat outcome-only RM on math. **Nobody has shipped a PRM for
   code.** Math PRMs check "is this algebra step correct?". A *code* PRM checks
   "is this *reasoning/commando/edit* step moving toward a passing test?".

So: RLVR is the engine, the code PRM is how we get sample-efficient,
step-credited training. The novelty we claim is **PRM-over-code-steps**, applied
inside an agent loop (where a "step" = a tool call, not a chain-of-thought token).

## The four stages, in order

### Stage 0 — base pretrain (already designed)
Fat MoE (10.25B/3.65B-active) on code corpus (smollm-corpus → stack-v2 as budget
allows). Output: a model that writes *plausible* code. It will be wrong often.
This is the cold start. Nothing here is novel — it's table stakes.

### Stage 1 — SFT on verified solutions (RFT = Rejection-sampling Fine-Tuning)
Cheap, no RL yet. We need data with a ground-truth verifier signal:

  1. Take coding problems WITH hidden tests (HumanEval, MBPP, CodeContests,
     APCeval, and critically **SWE-bench-lite** for agentic tasks).
  2. Sample N solutions from the Stage-0 model at high temperature (N=16-64).
  3. Run each in the sandbox against the visible tests.
  4. Keep only the ones that PASS **and** are diverse (dedup by behavior, not
     text — two correct solutions that take the same path count once).
  5. SFT the base model on the kept (problem → passing solution) pairs.

This is RFT. It bootstraps the model to "reliably produces code that passes
visible tests" before RL touches it. RL on a model that can't ever pass is
useless (zero reward signal = no gradient).

> Why this beats "just pretrain more tokens": RFT tokens are *verifiably
> correct* — information density per token is far higher than raw GitHub. This
> is the Phi/distillation argument applied *with execution as the filter*.

### Stage 2 — RLVR (verified-execution RL)
Now RL. The reward = the verifier. Two flavors, do both:

  **A. GRPO / RLOO** (no value model needed — good, we don't want to train one):
  - For a problem, sample G generations from the *current* policy.
  - Each gets reward r ∈ {0,1} (tests pass) — or a shaped reward
    r = 0.5*(visible_tests_pass) + 0.5*(hidden_tests_pass) for denser signal.
  - GRPO advantage: A_i = (r_i − mean(r))/std(r). Push up the log-prob of
    above-average rollouts, pull down below-average. Group baseline removes the
    need for a critic.
  - This is "hard to solve, easy to verify" — the reward is REAL, not learned.

  **B. Agent-RL** (the terminal-bench-relevant part): the generation isn't a
  one-shot code block, it's an *agent rollout* — a sequence of tool calls in the
  harness (`agent/loop.py`). Reward at the end = does the final repo state pass
  the test suite? This trains the model *to use the loop*, not just to write
  code. Qwen3-Coder's "20K-env Agent RL" is exactly this. **Our agent harness is
  already built (task #4) — this stage plugs into it.**

### Stage 3 — the code PRM (the novelty)
GRPO only credits the *final* outcome. Long agent rollouts (20 tool calls to fix
one bug) have a credit-assignment problem: which of the 20 calls was the good
one? An outcome reward says "the whole trajectory: +1" — useless signal spread
thin. A **Process Reward Model** scores each *step*.

**What a code PRM scores (our definition):**
A "step" in the agent loop = one `<cmd>` + its `<output>`. The PRM inputs
  (task, full trajectory so far, this step, the step's output)
and outputs a scalar p ∈ [0,1] = "probability this step is on a path that will
eventually pass the tests". Trained on:

  - **Positive steps**: from Stage-1 verified-passing trajectories, label every
    step with 1 (it led to a pass) — but weight by *causal contribution*
    (later).
  - **Negative steps**: from failed trajectories, find the step where things
    went wrong (the first step after which no continuation ever passes —
    "monte-carlo rollout" labeling).
  - **Hard negatives via execution**: a step that *looks* right (clean diff,
    runs) but doesn't move toward passing tests. The PRM must learn to spot
    these — outcome-only RMs cannot. This is where code PRMs > math PRMs:
    we have a verifier (tests) to define "toward passing" objectively.

**How the PRM is used:**
  1. **Best-of-N at inference**: run N agent rollouts, score each *step* with
     the PRM, pick the rollout with highest mean (or min) step-score. No
     training — pure inference-time win. This alone lifts terminal-bench.
  2. **PRM-weighted RL** (the training-time win): replace GRPO's outcome-only
     advantage with step-level advantages from the PRM:
       A_step = PRM(step) − baseline_PRM(trajectory_prefix)
     Now the "good step" gets credit even if the trajectory ultimately failed,
     and the "bad step" gets blamed even if the trajectory got lucky and passed.
     This is *exactly* the credit-assignment fix long agentic rollouts need.

> **The claim:** math-PRM ("is this reasoning step sound?") is subjective and
> needs human labels. Code-PRM ("does this *executed* step move toward passing
> tests?") is defined by execution — we get labels *for free* from the sandbox.
> That's the novel angle and why it's tractable solo.

## What we actually need to BUILD (concrete modules)

| module | what it does | stage |
|---|---|---|
| `verifier.py` | run a candidate solution against a problem's test suite in the sandbox, return pass/fail + per-test detail | 1,2,3 |
| `rft.py` | sample N from model, verify, dedup, SFT on the survivors | 1 |
| `rlvr.py` | GRPO/RLOO loop: sample G, score via verifier, advantage = normalized reward, policy update | 2 |
| `agent_rl.py` | run the *agent loop* as the rollout, reward = final state passes | 2 |
| `prm.py` | the Process Reward Model itself (small transformer head on top of base, or separate small model) + training (step labels from rollouts) | 3 |
| `best_of_n.py` | inference-time: N rollouts, PRM-scored, pick best | 3 |
| `eval.py` | harness runs a task subset of SWE-bench-lite/Terminal-Bench, reports pass@1/pass@5 | all |

The verifier is the keystone — every stage depends on it. Build it first. It
reuses `agent/sandbox.py`.

## Solo-budget honest order of operations

1. **verifier.py + eval.py on a tiny held-out set** — even on the *current
   17M base*, to prove the harness+verifier loop works mechanically.
2. RFT on a free VM (Kaggle/Colab) — cheapest signal, most mileage per dollar.
3. RLVR (GRPO) — needs more GPU, Modal A100 chunks.
4. PRM — train the small reward head, plug into best-of-N *first* (no RL needed
   to see a bench lift), then PRM-weighted RL.

## What's NOT here (deferred)

- LoRA/QLoRA of the policy for memory (we may need it; vanilla grad on 10B is
  heavy). Decide when we hit the memory wall, not before.
- KL penalty to the reference policy for RL stability — standard, add when
  GRPO first diverges.
- Self-play / synthetic problem generation (generating *new* problems to train
  on) — high variance, defer.

> **Bottom line:** the architecture (MoE+SwiGLU+GQA+shared) makes it *fast* and
> *high-capacity*. RLVR + the code PRM makes it *good at agentic coding* in a
> way that's tractable for one person with a sandbox and a few free GPUs.
> That's the whole thesis of "le fat chaton".