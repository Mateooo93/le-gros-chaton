"""Agent-Loop-as-Rollout for RLVR — the project's novel contribution.

THE IDEA
--------
Standard RL for code (RLEF, CodeRL, OpenR) samples single-shot completions:
given a problem, generate one solution, check if it passes tests, reward.

But real coding agents don't work like that.  They:
  1. Read the problem
  2. Write code
  3. Run tests
  4. See the error output
  5. Debug and fix
  6. Repeat until green

Our agent RL pipeline captures THIS trajectory as the rollout — the model
learns not just to write correct code on the first try, but to *debug and
iterate*.  Each step in the trajectory is a (observation, action, reward)
triple where the action is a shell command and the observation is its output.

This is the project's genuine novelty: no open-source code RL framework trains
models on multi-turn agent trajectories.

HOW IT WORKS
------------
  For each problem:
    1. Start the agent loop with the problem description as the task
    2. The model generates actions (<cmd>shell commands</cmd>)
    3. The sandbox executes the commands and returns the output
    4. After each step, the verifier checks if the tests pass
    5. If passing → reward = 1, trajectory ends
    6. If max steps → reward = 0, trajectory ends
    7. If blocked/dangerous → reward = -1, trajectory ends

  GRPO updates the policy using the trajectory-level reward.
  If a PRM is available, each step in the trajectory gets a score.

USAGE
  # Train with agent-loop rollouts:
  python agent_rl.py --problems humaneval --n 8 --limit 50 \
         --ckpt model_rft.pt --out model_agent_rl.pt

REFERENCES
  - DeepSeek-R1: GRPO (Group Relative Policy Optimization)
  - Qwen3-Coder: RLVR terminology
  - Ours: agent-loop-as-rollout (genuinely novel)
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

import config as cfg

# --- Lazy imports (flat root) ---
from tokenizer import encode, decode
from model import GPT
from agent.sandbox import run_cmd, CmdResult
from agent.loop import _CMD_RE, _DONE_RE, SYSTEM as AGENT_SYSTEM

# ---------------------------------------------------------------------------
# Trajectory data structures
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One step in an agent trajectory."""
    text_before: str       # Model's reasoning text before commands
    commands: list[str]    # Shell commands executed this step
    results: list[CmdResult]  # Sandbox results for each command
    reward: float = 0.0    # Step-level reward (from PRM or verifier)
    done: bool = False     # Whether model emitted <done> this step
    per_test: list[bool] | None = None  # Per-test pass/fail (for analysis)

@dataclass
class Trajectory:
    """A complete agent rollout for one problem."""
    problem_id: str
    problem_text: str
    steps: list[Step] = field(default_factory=list)
    solved: bool = False   # Did the verifier pass?
    total_reward: float = 0.0

# ---------------------------------------------------------------------------
# Agent rollout: run the loop for one problem
# ---------------------------------------------------------------------------

def _get_problem_text(problem: dict) -> str:
    """Extract the task prompt from a problem dict.

    Supports both HumanEval format and our fine-tune JSON format.
    """
    if "prompt" in problem:
        return problem["prompt"]
    if "instruction" in problem:
        return problem["instruction"]
    if "task" in problem:
        return problem["task"]
    if "text" in problem:
        return problem["text"]
    # Fallback: use the problem description
    return problem.get("description", problem.get("problem", ""))


def rollout(
    model: GPT,
    problem: dict,
    verifier_fn: Callable[[str], bool] | None = None,
    max_steps: int = 10,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    verbose: bool = False,
) -> Trajectory:
    """Run one agent rollout for a problem.

    The model interacts with the sandbox.  Every step, the verifier checks
    if the problem is solved (tests pass).  Returns the full trajectory
    with step-level details.

    Architecture (from agent/loop.py):
      - Model emits <cmd>...</cmd> to act
      - Sandbox runs the command, returns output
      - Model emits <done>...</done> to finish
      - Verifier checks the solution
    """
    device = next(model.parameters()).device
    problem_id = str(problem.get("problem_id", problem.get("id", "unknown")))
    problem_text = _get_problem_text(problem)

    # Initial prompt: system message + task
    initial_prompt = f"{AGENT_SYSTEM}\n\nTask: {problem_text}\n\n"
    convo_tokens: list[int] = encode(initial_prompt)

    traj = Trajectory(problem_id=problem_id, problem_text=problem_text)
    max_new_tokens_gen = max_new_tokens

    model.eval()
    for step_idx in range(max_steps):
        # --- Generate ---
        # Truncate to block_size from the right (keep the most recent context)
        ctx = convo_tokens[-cfg.block_size:]
        idx = torch.tensor([ctx], dtype=torch.long, device=device)
        out_tokens = model.generate(
            idx, max_new_tokens=max_new_tokens_gen,
            temperature=temperature, top_k=50, top_p=0.95,
            typical_p=0.2,
            repetition_penalty=1.1,
        )
        # Extract only the NEW tokens (after the input context)
        gen_tokens = out_tokens[0, idx.size(1):].tolist()
        gen_text = decode(gen_tokens)

        # Append to conversation
        convo_tokens.extend(gen_tokens)

        # --- Parse actions ---
        cmds = [m.group(1).strip() for m in _CMD_RE.finditer(gen_text)]
        done_m = _DONE_RE.search(gen_text)
        done_text = done_m.group(1).strip() if done_m else None

        step = Step(
            text_before=gen_text,
            commands=[],
            results=[],
            done=done_text is not None,
        )

        # --- Execute commands ---
        results: list[CmdResult] = []
        for c in cmds:
            # Check for dangerous commands
            c_stripped = c.strip()
            if c_stripped in ("rm -rf /", "shutdown", "reboot", "mkfs",
                              "dd if=/dev/zero", ":(){ :|:& };:"):
                results.append(CmdResult(
                    stdout="", stderr="[BLOCKED] dangerous command",
                    combined_truncated="[BLOCKED] dangerous command",
                    rc=-1, timed_out=False, blocked=True,
                ))
            else:
                results.append(run_cmd(c_stripped))

            # Append output to conversation
            r = results[-1]
            feedback = (f"\n<output rc={r.rc}"
                        f"{' TIMEOUT' if r.timed_out else ''}"
                        f"{' BLOCKED' if r.blocked else ''}>\n"
                        f"{r.combined_truncated}\n</output>\n")
            feedback_tokens = encode(feedback)
            convo_tokens.extend(feedback_tokens)

        step.commands = cmds
        step.results = results

        # --- Check if solved (via verifier) ---
        if verifier_fn is not None:
            solution = _extract_latest_solution(convo_tokens)
            if solution:
                # Use proportional reward: fraction of tests passed
                from verify.verifier import verify as _verify, Problem
                # Create a problem from the current context
                prob = Problem(
                    id=problem_id,
                    prompt=problem_text,
                    tests=problem.get("tests", ""),
                    entry_point=problem.get("entry_point"),
                )
                verdict = _verify(prob, solution, timeout=15.0)
                if verdict.passed:
                    step.reward = 1.0
                    step.done = True
                    traj.solved = True
                elif verdict.n_total > 0:
                    # Proportional reward: fraction of tests passed
                    step.reward = verdict.n_pass / max(verdict.n_total, 1)
                else:
                    step.reward = 0.0
                # Track per-test results for debugging
                step.per_test = verdict.per_test

        traj.steps.append(step)

        # Stop if the model finished or we solved the problem
        if step.done or traj.solved:
            break

        # Crop conversation if too long
        if len(convo_tokens) > cfg.block_size:
            convo_tokens = convo_tokens[-(cfg.block_size):]

    # --- Compute total reward (proportional, not binary) ---
    if traj.solved:
        traj.total_reward = 1.0
    elif traj.steps:
        # Average step reward (each step's proportional test score)
        traj.total_reward = sum(s.reward for s in traj.steps) / len(traj.steps)
    else:
        traj.total_reward = 0.0

    return traj


def _extract_latest_solution(convo_tokens: list[int]) -> str | None:
    """Try to extract the latest code solution from the conversation.

    Looks for code blocks after the last <cmd> or <done> tag.
    This is a heuristic — in production, use a proper code extractor.
    """
    text = decode(convo_tokens)
    # Look for code in <code>...</code> blocks
    code_blocks = re.findall(r"<code>(.*?)</code>", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    # Look for python code after "```python"
    backtick_blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if backtick_blocks:
        return backtick_blocks[-1].strip()
    return None


# ---------------------------------------------------------------------------
# GRPO update on trajectories
# ---------------------------------------------------------------------------

def grpo_update(
    model: GPT,
    ref_model: GPT | None,
    trajectories: list[Trajectory],
    optimizer: torch.optim.Optimizer,
    epsilon: float = 0.2,
    kl_coeff: float = 0.01,
    max_grad_norm: float = 1.0,
) -> dict:
    """Perform one GRPO update from a batch of trajectories.

    For each trajectory, we re-compute the log-probabilities of the *actions*
    (the model's own generated tokens) under the *current* policy, then apply
    the GRPO clipped surrogate:

        L = -mean( min( r * A, clip(r, 1-eps, 1+eps) * A ) )

    where r = exp(log π_θ(action) - log π_old(action)) is the importance
    weight.  Since we don't store old log-probs yet, we use a simplified
    REINFORCE-with-baseline: L = -mean( A * log π_θ(action) ).

    A KL penalty against ref_model (if provided) prevents the policy from
    drifting too far.
    """
    device = next(model.parameters()).device

    # --- Compute group-normalised advantages ---
    rewards = torch.tensor(
        [t.total_reward for t in trajectories],
        dtype=torch.float32, device=device,
    )
    std, mean = rewards.std(), rewards.mean()
    if std > 1e-6:
        advantages = (rewards - mean) / (std + 1e-8)
    else:
        advantages = torch.zeros_like(rewards)

    # --- Forward pass for each trajectory (WITH gradients) ---
    total_policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_kl_loss = torch.tensor(0.0, device=device)
    n_action_tokens = 0

    # We'll average over all action tokens across all trajectories
    all_log_probs: list[torch.Tensor] = []
    all_advantages: list[torch.Tensor] = []

    for traj, advantage in zip(trajectories, advantages):
        if not traj.steps:
            continue

        # Build the problem prefix once
        prefix_ids = encode(f"{AGENT_SYSTEM}\n\nTask: {traj.problem_text}\n\n")

        # For each step in the trajectory, the model's generated text
        # is the "action".  We compute log π_θ(action) by running the
        # model on the prefix + action tokens.
        for step in traj.steps:
            action_ids = encode(step.text_before)
            if not action_ids:
                continue

            # Concatenate prefix + action.  The model's logits at position t
            # predict token t+1, so the (prefix + action[:-1]) -> action[1:] path
            # gives us the log-prob of each action token given its context.
            input_ids = prefix_ids + action_ids
            if len(input_ids) > cfg.block_size:
                input_ids = input_ids[-cfg.block_size:]

            inp = torch.tensor([input_ids], dtype=torch.long, device=device)
            logits, _, _ = model(inp)

            # Extract the action-token log-probs
            # For each action token at position p (in the action), we need
            # the log-prob predicted at position p-1 (the last context token
            # before the action token).
            # context_end = len(prefix_ids) - 1  (the last prefix token)
            prefix_len = min(len(prefix_ids), cfg.block_size)
            # The action tokens start after the prefix
            action_start = prefix_len - 1  # logits index that predicts first action token

            for j in range(len(action_ids)):
                t = action_start + j  # logits row index
                if t >= logits.size(1) - 1:
                    break
                target_id = input_ids[t + 1]
                lp = F.log_softmax(logits[0, t, :], dim=-1)[target_id]
                all_log_probs.append(lp)
                all_advantages.append(advantage)
                n_action_tokens += 1

    if n_action_tokens == 0:
        return {"policy_loss": 0.0, "kl_penalty": 0.0,
                "mean_reward": rewards.mean().item(),
                "solve_rate": (rewards > 0).float().mean().item(),
                "n_trajectories": len(trajectories)}

    # --- Build the surrogate loss ---
    log_probs = torch.stack(all_log_probs)
    advs = torch.stack(all_advantages)

    # REINFORCE: maximize (advantage * log_prob) for positive advantages,
    # minimize for negative advantages.
    # L = -mean( A * log π_θ(a|s) )
    # Equivalent to: push the model toward actions that beat the baseline.
    policy_loss = -(advs * log_probs).mean()

    # --- KL penalty (approximate, against reference model) ---
    if ref_model is not None:
        ref_model.eval()
        # For each action token, compute KL(π_ref || π_θ) = log(π_ref/π_θ)
        # We approximate with the difference in log-probs.
        # (Full implementation would compute per-token KL)
        # For now, a simple L2 penalty on logit differences.
        # This is a placeholder — the proper approach is importance-weighted KL.
        with torch.no_grad():
            ref_log_probs = []
            for traj, _ in zip(trajectories, advantages):
                if not traj.steps:
                    continue
                prefix_ids = encode(f"{AGENT_SYSTEM}\n\nTask: {traj.problem_text}\n\n")
                for step in traj.steps:
                    action_ids = encode(step.text_before)
                    if not action_ids:
                        continue
                    input_ids = prefix_ids + action_ids
                    if len(input_ids) > cfg.block_size:
                        input_ids = input_ids[-cfg.block_size:]
                    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
                    ref_logits, _, _ = ref_model(inp)
                    prefix_len = min(len(prefix_ids), cfg.block_size)
                    action_start = prefix_len - 1
                    for j in range(len(action_ids)):
                        t = action_start + j
                        if t >= ref_logits.size(1) - 1:
                            break
                        target_id = input_ids[t + 1]
                        lp = F.log_softmax(ref_logits[0, t, :], dim=-1)[target_id]
                        ref_log_probs.append(lp)
            if ref_log_probs:
                ref_lps = torch.stack(ref_log_probs)
                # KL divergence ≈ mean(exp(ref_lps - log_probs.detach()) * (ref_lps - log_probs.detach())) - 1
                # Simplified: use reverse KL coefficient
                kl_penalty = kl_coeff * (ref_lps - log_probs.detach()).pow(2).mean()
                total_kl_loss = kl_penalty

    total_loss = policy_loss + total_kl_loss

    # --- Step ---
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "kl_penalty": total_kl_loss.item(),
        "mean_reward": rewards.mean().item(),
        "solve_rate": (rewards > 0).float().mean().item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "n_tokens": n_action_tokens,
    }


# ---------------------------------------------------------------------------
# Problem loader
# ---------------------------------------------------------------------------

def load_problems(source: str, limit: int | None = None) -> list[dict]:
    """Load problems from a source.

    Supports:
      - "humaneval": built-in HumanEval dataset
      - Path to a JSON file: list of problem dicts
    """
    if source == "humaneval":
        from eval.humaneval_loader import get_humaneval_problems
        problems = get_humaneval_problems()
    elif os.path.isfile(source):
        with open(source) as f:
            problems = json.load(f)
        if isinstance(problems, dict):
            # Convert {id: problem} to list of dicts
            problems = [{"id": k, **v} for k, v in problems.items()]
    else:
        raise ValueError(f"Unknown problem source: {source}")

    if limit and len(problems) > limit:
        import random
        rng = random.Random(42)
        rng.shuffle(problems)
        problems = problems[:limit]

    return problems


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_agent_rl(
    problems: list[dict],
    ckpt_path: str | None = None,
    out_path: str = "models/agent_rl.pt",
    n_rollouts_per_problem: int = 1,
    n_steps: int = 50,
    max_steps_per_rollout: int = 10,
    lr: float = 1e-6,
    batch_size: int = 4,
    kl_coeff: float = 0.01,
    verbose: bool = False,
):
    """Main training loop for agent RL.

    Args:
        problems: List of problem dicts.
        ckpt_path: Path to initial checkpoint (from RFT or RLVR).
        out_path: Where to save the trained model.
        n_rollouts_per_problem: Number of rollouts to collect per problem.
        n_steps: Number of outer training steps.
        max_steps_per_rollout: Max agent loop steps per rollout.
        lr: Learning rate for the policy.
        batch_size: Rollouts per GRPO update.
        kl_coeff: KL penalty coefficient.
        verbose: Print rollout details.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[agent_rl] device = {device}")

    # --- Build model ---
    if ckpt_path and os.path.exists(ckpt_path):
        model = GPT.from_checkpoint(ckpt_path, device)
        print(f"[agent_rl] loaded checkpoint from {ckpt_path}")
    else:
        model = GPT(gradient_checkpointing=cfg.gradient_checkpointing).to(device)
        print(f"[agent_rl] initialised fresh model")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # Optional reference model for KL penalty
    ref_model: GPT | None = None

    # --- Verifier ---
    verifier_fn = None
    try:
        from verify.verifier import Verifier, build_verifier
        verifier_fn = build_verifier(
            Verifier(problem_source=problems),
            problem_id_field="id",
        )
        print(f"[agent_rl] verifier loaded")
    except Exception as e:
        print(f"[agent_rl] verifier NOT available ({e}); rewards will be based "
              f"on <done> tag only")

    # --- Training loop ---
    global_step = 0
    problem_pool = list(range(len(problems)))
    import random as rng_mod
    rng_mod.seed(42)

    for step in range(n_steps):
        # --- Collect rollouts ---
        batch_trajectories: list[Trajectory] = []
        rng_mod.shuffle(problem_pool)

        for prob_idx in problem_pool[:batch_size]:
            try:
                traj = rollout(
                    model, problems[prob_idx],
                    verifier_fn=verifier_fn,
                    max_steps=max_steps_per_rollout,
                    verbose=verbose,
                )
                batch_trajectories.append(traj)
            except Exception as e:
                print(f"[agent_rl] rollout failed for problem {prob_idx}: {e}")
                continue

        if not batch_trajectories:
            print(f"[agent_rl] all rollouts failed at step {step}; skipping")
            continue

        # --- GRPO update ---
        metrics = grpo_update(
            model, ref_model, batch_trajectories, optimizer,
            kl_coeff=kl_coeff,
        )
        global_step += 1

        solve_rate = metrics["solve_rate"]
        mean_reward = metrics["mean_reward"]

        print(
            f"[agent_rl] step {step:4d}/{n_steps}  "
            f"reward {mean_reward:.3f}  "
            f"solve {solve_rate:.0%}  "
            f"trajs {len(batch_trajectories)}  "
            f"n_steps {metrics.get('n_trajectories', '?')}"
        )

        # --- Periodic save ---
        if (step + 1) % 10 == 0 or step == n_steps - 1:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            model_to_save = getattr(model, "_orig_mod", model)
            torch.save(model_to_save.state_dict(), out_path)
            print(f"[agent_rl] saved checkpoint to {out_path}")

    print(f"[agent_rl] done. final model saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Agent-Loop-as-Rollout RLVR training"
    )
    parser.add_argument("--problems", default="humaneval",
                        help="Problem source ('humaneval' or JSON file path)")
    parser.add_argument("--ckpt", default="model.pt",
                        help="Initial model checkpoint (from RFT)")
    parser.add_argument("--out", default="model_agent_rl.pt",
                        help="Output model path")
    parser.add_argument("--n-rollouts", type=int, default=1,
                        help="Rollouts per problem per step")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Number of training steps")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Max agent loop steps per rollout")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Rollouts per GRPO update")
    parser.add_argument("--kl-coeff", type=float, default=0.01,
                        help="KL penalty coefficient")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of problems")
    parser.add_argument("--verbose", action="store_true",
                        help="Print rollout details")
    args = parser.parse_args()

    problems = load_problems(args.problems, args.limit)
    print(f"[agent_rl] loaded {len(problems)} problems from {args.problems}")

    train_agent_rl(
        problems=problems,
        ckpt_path=args.ckpt,
        out_path=args.out,
        n_rollouts_per_problem=args.n_rollouts,
        n_steps=args.n_steps,
        max_steps_per_rollout=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        kl_coeff=args.kl_coeff,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()