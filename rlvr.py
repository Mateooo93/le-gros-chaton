"""RLVR — Reinforcement Learning from Verified Reward (GRPO).

Stage 2 of the innovation pipeline.  After RFT bootstraps the model to
*sometimes* pass tests, RLVR amplifies that behaviour into a reliable skill.

THE IDEA (GRPO — Group Relative Policy Optimisation)
-----------------------------------------------------
Instead of a value model (critic), GRPO samples G completions from the *current*
policy, scores them with the verifier, and computes the advantage as:

    A_i = (r_i - mean(r)) / std(r)

Each completion's log-prob is pushed toward the above-average ones and away
from below-average ones.  The group-normalised baseline removes the need for a
trained critic — perfect for a solo-budget project where training a value model
is expensive and brittle.

References
----------
- DeepSeek-R1: ``GRPO`` (Group Relative Policy Optimization)
- ``RLVR`` terminology from Qwen3-Coder technical report
- Our verifier (``verify/verifier.py``) provides the reward signal

USAGE
-----
  python rlvr.py --ckpt model_rft.pt --problems humaneval \
         --n 8 --limit 20 --out model_rlvr.pt

  # Resume from a partially-trained checkpoint:
  python rlvr.py --ckpt model_rlvr.pt --resume rlvr_checkpoint.pt

TRAINING LOOP
-------------
  for each problem in curriculum:
    1. sample G completions from current policy
    2. run each in sandbox; reward = 1 if hidden tests pass else 0
    3. compute group-normalised advantages
    4. one policy-gradient step (no critic, no KL penalty yet)
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
import torch.nn.functional as F
from model import GPT
import config as cfg
from tokenizer import encode, decode, EOT_TOKEN
from agent.sandbox import run_cmd


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RLVRStep:
    """One training step's data."""
    problem_id: str
    prompt: str
    completions: list[str]       # G generated solutions
    rewards: list[float]         # G verifier scores
    log_probs: list[float]       # G mean log-probs of the completion under current policy
    advantages: list[float] = field(default_factory=list)


def load_problems(source: str, limit: int | None = None) -> list:
    """Load problems (mirrors rft.load_problems)."""
    if source == "humaneval":
        from eval.humaneval_loader import load
        return load(limit=limit)
    elif source.endswith(".json"):
        with open(source) as f:
            raw = json.load(f)
        from verify.verifier import Problem
        out = []
        for r in raw[:limit] if limit else raw:
            out.append(Problem(
                id=r["id"], prompt=r["prompt"], tests=r["tests"],
                entry_point=r.get("entry_point"),
            ))
        return out
    else:
        raise ValueError(f"unknown problem source {source!r}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Diversity / novelty rewards (creativity objective)
# ---------------------------------------------------------------------------


def _n_gram_set(text: str, n: int = 3) -> set:
    """Token-level n-gram set for novelty scoring."""
    toks = text.split()
    if len(toks) < n:
        return set(toks)
    return set(zip(*[toks[i:] for i in range(n)]))


def novelty_score(candidate: str, others: list[str]) -> float:
    """How novel is *candidate* relative to *others*? 0 (identical to some
    other solution) .. 1 (shares nothing with any other solution).

    Uses max Jaccard n-gram overlap across all others, inverted.
    """
    if not others:
        return 1.0
    cand = _n_gram_set(candidate)
    if not cand:
        return 0.0
    max_overlap = max(len(cand & _n_gram_set(o)) / len(cand | _n_gram_set(o))
                      if _n_gram_set(o) else 0.0 for o in others)
    return 1.0 - max_overlap


def diversity_reward(completions: list[str], rewards: list[float],
                     novelty_bonus: float = 0.2) -> list[float]:
    """Add a novelty bonus to verifier rewards: a correct solution that is
    unlike the other sampled solutions gets extra reward (pushing the policy
    to explore new strategies instead of converging on one memorized style).

    Failed solutions are NOT given novelty credit (we don't want to reward
    creative garbage) — but they still participate as "others" for the
    novelty computation, so a correct-but-redundant solution is penalised
    relative to a correct-and-fresh one.
    """
    out = list(rewards)
    correct = [c for c, r in zip(completions, rewards) if r > 0]
    if not correct:
        return out
    for i, (comp, r) in enumerate(zip(completions, rewards)):
        if r > 0:
            others = [c for j, c in enumerate(completions) if j != i]
            out[i] = r + novelty_bonus * novelty_score(comp, others)
    return out


def strategy_switch_reward(rollout_steps: list[str]) -> float:
    """Reward trying a NEW approach after a failed attempt (0..1).

    Consumes a list of action descriptions from one agent rollout (e.g. the
    tool calls). Returns 1.0 if the agent attempted >=2 different strategies
    (measured by n-gram distance between consecutive attempts), else 0.0.
    This is the "learn from failure, don't repeat it" objective — the
    behavioural backbone of creativity.
    """
    if len(rollout_steps) < 2:
        return 0.0
    switches = 0
    for prev, cur in zip(rollout_steps, rollout_steps[1:]):
        if novelty_score(cur, [prev]) > 0.5:  # meaningfully different action
            switches += 1
    return min(1.0, switches / max(1, len(rollout_steps) - 1))


# Verifier-based reward
# ---------------------------------------------------------------------------

def compute_reward(problem, solution: str, timeout: float = 15.0) -> float:
    """Return 1.0 if *solution* passes *problem*'s tests, else 0.0.

    Returns a float so the interface extends to shaped rewards later
    (e.g. 0.5 * visible_tests + 0.5 * hidden_tests).
    """
    from verify.verifier import verify, _python_check_code
    import hashlib

    cwd = os.path.join(PROJ_ROOT, "verify", "_runs")
    os.makedirs(cwd, exist_ok=True)

    code = _python_check_code(problem, problem.prompt + solution) \
        if hasattr(problem, 'entry_point') else (
            f"{problem.prompt}{solution}\n{problem.tests}\nprint('OK')\n"
        )
    tag = hashlib.md5(problem.id.encode()).hexdigest()[:12]
    run_file = os.path.join(cwd, f"_rlvr_{tag}.py")
    with open(run_file, "w") as f:
        f.write(code)

    from agent.sandbox import run_cmd
    r = run_cmd(f"python {os.path.basename(run_file)}", timeout=timeout, cwd=cwd)
    return 1.0 if (r.rc == 0 and "OK" in r.stdout) else 0.0


# ---------------------------------------------------------------------------
# GRPO core
# ---------------------------------------------------------------------------

def _gather_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Extract the log-prob of the *tokens* under the *logits* distribution.

    Args:
        logits: (B, T, V) raw logits.
        tokens: (B, T) token ids.

    Returns:
        (B, T) per-token log-probabilities.
    """
    log_probs = F.log_softmax(logits, dim=-1)          # (B, T, V)
    return log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)


def grpo_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    problem,
    n_completions: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    epsilon: float,
    device: str,
    verbose: bool = False,
    diversity: bool = False,
    novelty_bonus: float = 0.2,
) -> RLVRStep | None:
    """Run one GRPO step on *problem*.

    1. Sample G completions.
    2. Compute rewards via the verifier.
    3. Compute group-normalised advantages.
    4. Take a policy-gradient step with clipped surrogate objective.

    Returns the step data (for logging) or None if no completion passed.
    """
    prompt_ids = torch.tensor([encode(problem.prompt)], dtype=torch.long, device=device)
    L = prompt_ids.size(1)

    # --- 1. Sample G completions ---
    batch = prompt_ids.expand(n_completions, -1).contiguous()
    with torch.no_grad():
        out = model.generate(
            batch, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=50, top_p=top_p,
            typical_p=0.2,
            repetition_penalty=1.0,
        )

    completions: list[str] = []
    completion_tokens: list[list[int]] = []
    rewards: list[float] = []

    for b in range(n_completions):
        comp_ids = out[b, L:].tolist()
        comp_text = decode(comp_ids)
        # Truncate at first top-level boundary
        for tok in ["\nclass ", "\ndef ", "\nif __name__", "\n\n\n"]:
            i = comp_text.find(tok)
            if i != -1:
                comp_text = comp_text[:i]
                comp_ids = encode(comp_text)
                break
        completions.append(comp_text)
        completion_tokens.append(comp_ids)

        # --- 2. Compute reward ---
        reward = compute_reward(problem, comp_text)
        rewards.append(reward)

    # If no completion passed, skip this step (no gradient signal)
    if max(rewards) == 0.0:
        return None

    # --- Creativity: shape rewards with a novelty bonus ---
    # A correct solution unlike the other sampled solutions gets extra reward,
    # pushing the policy to explore fresh strategies instead of converging on
    # one memorized style.
    if diversity:
        shaped = diversity_reward(completions, rewards, novelty_bonus=novelty_bonus)
        rewards = shaped

    # --- 3. Compute old log-probs and advantages ---
    with torch.no_grad():
        old_log_probs_list: list[float] = []

        for b in range(n_completions):
            comp_ids = completion_tokens[b]
            if not comp_ids:
                old_log_probs_list.append(0.0)
                continue
            # Run forward pass on the completion to get log-probs
            full_ids = torch.cat([prompt_ids[0], torch.tensor(comp_ids, device=device)])
            full_input = full_ids[:-1].unsqueeze(0)
            full_target = full_ids[1:].unsqueeze(0)
            logits, _, _ = model(full_input, use_cache=False)
            lp = _gather_log_probs(logits, full_target)
            # Mean log-prob over solution tokens (excluding prompt)
            sol_lp = lp[0, L - 1:]
            old_log_probs_list.append(sol_lp.mean().item())

        # Group-normalised advantages
        rewards_t = torch.tensor(rewards, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8
        advantages = ((rewards_t - mean_r) / std_r).tolist()

    # --- 4. Policy-gradient step ---
    optimizer.zero_grad(set_to_none=True)

    new_log_probs_list: list[float] = []
    total_loss = 0.0

    for b in range(n_completions):
        comp_ids = completion_tokens[b]
        if not comp_ids:
            new_log_probs_list.append(0.0)
            continue
        adv = advantages[b]

        full_ids = torch.cat([prompt_ids[0], torch.tensor(comp_ids, device=device)])
        full_input = full_ids[:-1].unsqueeze(0)
        full_target = full_ids[1:].unsqueeze(0)

        logits, _, _ = model(full_input, use_cache=False)
        new_lp = _gather_log_probs(logits, full_target)
        sol_lp = new_lp[0, L - 1:]
        mean_new_lp = sol_lp.mean()
        new_log_probs_list.append(mean_new_lp.item())

        # Clipped surrogate objective (PPO-style clip)
        ratio = torch.exp(mean_new_lp - old_log_probs_list[b])
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        loss_b = -torch.min(ratio * adv, clipped * adv)
        total_loss = total_loss + loss_b

    total_loss = total_loss / n_completions

    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return RLVRStep(
        problem_id=problem.id, prompt=problem.prompt,
        completions=completions, rewards=rewards,
        log_probs=new_log_probs_list, advantages=advantages,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    ckpt_path: str,
    problem_source: str = "humaneval",
    n_completions: int = 8,
    n_epochs: int = 3,
    lr: float = 1e-6,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    epsilon: float = 0.2,
    limit: int | None = None,
    device: str = "",
    out_path: str = "model_rlvr.pt",
    ckpt_interval: int = 50,
    verbose: bool = True,
    diversity: bool = False,
    novelty_bonus: float = 0.2,
):
    """Run the GRPO training loop.

    For each epoch, iterates over problems, samples G completions, computes
    advantages from the verifier, and takes a policy-gradient step.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    problems = load_problems(problem_source, limit=limit)

    if verbose:
        print(f"[rlvr] training on {len(problems)} problems, "
              f"G={n_completions}, epochs={n_epochs}, lr={lr}, device={device}")

    model = GPT.from_checkpoint(ckpt_path, device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    t0 = time.time()
    total_steps = 0
    total_passing = 0

    for epoch in range(n_epochs):
        epoch_passing = 0
        for pi, problem in enumerate(problems):
            step = grpo_step(
                model, optimizer, scaler, problem,
                n_completions=n_completions, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, epsilon=epsilon,
                device=device, verbose=(verbose and pi == 0),
                diversity=diversity, novelty_bonus=novelty_bonus,
            )

            if step is not None:
                total_steps += 1
                n_pass = sum(1 for r in step.rewards if r > 0)
                epoch_passing += n_pass
                total_passing += n_pass

                if verbose and (total_steps % 10 == 0 or pi == 0):
                    elapsed = time.time() - t0
                    print(
                        f"[rlvr] epoch {epoch + 1}/{n_epochs}  "
                        f"step {total_steps}  {step.problem_id}  "
                        f"pass {n_pass}/{n_completions}  "
                        f"(total {total_passing} passes)  "
                        f"{elapsed:.0f}s elapsed"
                    )

            # Periodic checkpoint
            if (pi + 1) % ckpt_interval == 0:
                _save_checkpoint(model, optimizer, scaler, total_steps, out_path)

        if verbose:
            print(f"[rlvr] epoch {epoch + 1} done — "
                  f"{epoch_passing} passes across {len(problems)} problems")

    # Final save
    model_to_save = getattr(model, "_orig_mod", model)
    torch.save(model_to_save.state_dict(), out_path)
    if verbose:
        print(f"[rlvr] saved to {out_path}")


def _save_checkpoint(model, optimizer, scaler, step, path):
    """Save a resumable checkpoint."""
    payload = {
        "model": getattr(model, "_orig_mod", model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, path)
    print(f"[rlvr] checkpoint saved at step {step} → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="RLVR — GRPO training for le fat chaton")
    ap.add_argument("--ckpt", default="model_rft.pt", help="Base checkpoint")
    ap.add_argument("--problems", default="humaneval", help="Problem source")
    ap.add_argument("--n", type=int, default=8, help="Completions per step (G)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--epsilon", type=float, default=0.2, help="Clip ratio")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="model_rlvr.pt")
    ap.add_argument("--ckpt-interval", type=int, default=50)
    ap.add_argument("--device", default="")
    ap.add_argument("--diversity", action="store_true",
                    help="Shape rewards with a novelty bonus (creativity objective): \
correct solutions unlike the group's others get extra reward")
    ap.add_argument("--novelty-bonus", type=float, default=0.2,
                    help="Max reward added for a novel correct solution (0-1)")

    args = ap.parse_args()
    train(
        ckpt_path=args.ckpt, problem_source=args.problems,
        n_completions=args.n, n_epochs=args.epochs,
        lr=args.lr, max_new_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p,
        epsilon=args.epsilon, limit=args.limit,
        device=args.device, out_path=args.out,
        ckpt_interval=args.ckpt_interval,
        diversity=args.diversity, novelty_bonus=args.novelty_bonus,
    )


if __name__ == "__main__":
    main()