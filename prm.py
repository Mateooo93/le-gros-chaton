"""Process Reward Model for code - Stage 3 (the project's novelty).

THE IDEA
--------
GRPO (rlvr.py) only credits the *final* outcome.  In long agent rollouts
(10-20 tool calls to fix one bug), this is an extremely sparse signal.

A Process Reward Model scores each *step* of a solution (or agent rollout):
  PRM(task, trajectory_so_far, step, step_output) -> p ∈ [0, 1]
where p = "probability this step is on a path that will eventually pass tests".

This is novel for code because math PRMs need expensive human annotators
("is this algebra step correct?"), while code PRMs get labels *for free*
from the verifier - "does this executed step move toward passing tests?"
is defined objectively by whether any continuation starting from this step
eventually passes.

ARCHITECTURE
------------
A lightweight classification head (2-layer MLP) on top of the frozen base
model's final hidden state at each step boundary.  The base model is frozen;
only the PRM head is trained (fast, ~1M params).

USAGE
-----
  Collect training data:
      python prm.py collect --rollouts rft_data.json --out prm_train.json

  Train PRM head:
      python prm.py train --data prm_train.json --ckpt models/prm.pt

  PRM-guidance for Best-of-N:
      python best_of_n.py --ckpt model_rlvr.pt --n 32 --prm prm.pt
"""

import argparse
import ast
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Imports from the project (flat root, per the design choice).
# ---------------------------------------------------------------------------
import config as cfg
from model import GPT
from tokenizer import encode

# ---------------------------------------------------------------------------
# Step extraction: split a code solution into "steps" at logical boundaries.
# A step is a sequence of tokens that form a coherent unit: a function body,
# a block of statements, a comment block, etc.
# ---------------------------------------------------------------------------

# Patterns that indicate a step boundary in code.
_BOUNDARY_RE = None   # lazy-compiled


def _get_boundary_pattern() -> "re.Pattern":
    global _BOUNDARY_RE
    if _BOUNDARY_RE is not None:
        return _BOUNDARY_RE
    import re
    # Step boundaries: blank lines, top-level function/class defs, comment
    # blocks (3+ consecutive comment lines), return statements at top level,
    # and lines starting with specific keywords.
    _BOUNDARY_RE = re.compile(
        r"(?:^|\n)"
        r"(?:"
        r"  \s*\n \s* (?=\S)           "  # blank line followed by code
        r"| def \s+ \w+                 "  # function definition
        r"| class \s+ \w+               "  # class definition
        r"| [ \t]*# [ \t]*              "  # top-level comment
        r"| \s* return \b                "  # return statement
        r"| \s* (?:if|elif|else|for|while|try|except|finally|with) \b"
        r")",
        re.VERBOSE,
    )
    return _BOUNDARY_RE


def _extract_steps_ast(code: str, min_step_tokens: int = 8) -> list[str]:
    """AST-based step extraction - more accurate than regex for Python code.

    Walks the top-level statements in the AST and maps each back to its source
    lines.  This correctly handles nested definitions, decorators, multi-line
    strings containing keywords, and comment-attribution (comments before a
    function are part of that function's step).

    Falls back to regex-based extraction for non-Python code or parse errors.
    """
    import re
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return extract_steps_regex(code, min_step_tokens)

    min_chars = min_step_tokens * 4
    lines = code.splitlines(keepends=True)

    def _node_lines(node) -> str:
        """Extract source lines for an AST node."""
        start = getattr(node, "lineno", 1) - 1
        # Decorator lines come before the function def lineno
        if hasattr(node, "decorator_list") and node.decorator_list:
            start = min(start, node.decorator_list[0].lineno - 1)
        end = getattr(node, "end_lineno", start + 1)
        return "".join(lines[start:end])

    raw_steps = []
    for node in ast.iter_child_nodes(tree):
        seg = _node_lines(node).strip()
        if seg:
            raw_steps.append(seg)

    # If AST gave nothing (e.g. `pass`), fall back to regex
    if not raw_steps:
        return extract_steps_regex(code, min_step_tokens)

    # Merge short steps into the previous one
    steps: list[str] = []
    for seg in raw_steps:
        if steps and len(seg) < min_chars:
            # Don't merge new function/class/definition boundaries
            if not any(seg.startswith(kw) for kw in ("def ", "class ", "@")):
                steps[-1] = steps[-1] + "\n" + seg
                continue
        steps.append(seg)

    return steps if steps else [code.strip()]


def extract_steps_regex(code: str, min_step_tokens: int = 8) -> list[str]:
    """Regex-based step extraction (fallback for non-Python code).

    Same logic as the original ``extract_steps()``."""
    import re
    pattern = _get_boundary_pattern()
    min_chars = min_step_tokens * 4

    boundaries = [0]
    for m in pattern.finditer(code):
        pos = m.start()
        if pos > boundaries[-1]:
            boundaries.append(pos)
    boundaries.append(len(code))

    raw_steps = []
    for i in range(len(boundaries) - 1):
        seg = code[boundaries[i]:boundaries[i + 1]].strip()
        if seg:
            raw_steps.append(seg)

    steps: list[str] = []
    for seg in raw_steps:
        if steps and len(seg) < min_chars:
            if not any(seg.startswith(kw) for kw in ("def ", "class ", "@")):
                steps[-1] = steps[-1] + "\n" + seg
                continue
        steps.append(seg)

    return steps if steps else [code.strip()]


def extract_steps(code: str, min_step_tokens: int = 8) -> list[str]:
    """Split *code* into logical steps at natural boundaries.

    Uses AST parsing for Python code (more accurate) and falls back to
    regex-based extraction otherwise.

    Each step is a coherent unit (function body, statement block, etc.).
    Steps shorter than *min_step_tokens* are merged into the previous step
    to avoid trivial single-line steps.
    """
    return _extract_steps_ast(code, min_step_tokens)


# ---------------------------------------------------------------------------
# Monte-Carlo step labeling.
#
# For each step s_i in a solution:
#   - If the solution passes all tests -> label = 1 for ALL steps (trivially).
#   - If the solution fails -> find the first step s_j where NO continuation
#     (i.e. generating a different suffix from that step onward) passes.
#     Steps before j: label = 1 (they were on a fixable trajectory).
#     Steps j onward: label = 0.
#
# We approximate this by taking the original solution from step s_i onward,
# and checking if it passes.  In the full version, we'd sample multiple
# continuations from each step (expensive but more accurate).
# ---------------------------------------------------------------------------


def label_steps(steps: list[str], passes: bool,
                verifier_fn: Callable[[str], bool] | None = None,
                n_continuations: int = 3) -> list[int]:
    """Assign binary labels (0 or 1) to each step.

    Args:
        steps: The list of step strings for one solution.
        passes: Whether the *complete* solution passes all tests.
        verifier_fn: Optional callable to check partial continuations.
                     Takes a code string, returns True if it passes.
        n_continuations: Number of noisy continuations to try from each
                         step (Monte-Carlo estimate).  Only used when
                         passes=False and verifier_fn is provided.

    Returns:
        A list of length ``len(steps)`` with 0/1 labels.
    """
    if passes:
        return [1] * len(steps)

    if verifier_fn is None:
        # Without a verifier, we conservatively label all steps as failed
        # (the RFT data didn't capture intermediate trajectory info).
        return [0] * len(steps)

    # --- Monte-Carlo labeling ---
    # For each step, check if ANY continuation from that step forward passes.
    # If a continuation passes, all steps up to and including this one are
    # "on a fixable path" -> label 1.
    labels = [0] * len(steps)

    # Work backwards: the later steps are more likely to be unfixable.
    full_code = "\n".join(steps)

    for i in range(len(steps) - 1, -1, -1):
        # Build prefix up to and including step i
        prefix = "\n".join(steps[:i + 1])

        # Try n_continuations from this prefix (Monte-Carlo sampling of
        # continuations).  For now, we check the original continuation
        # (the true suffix) as the cheapest approximation.
        any_pass = False

        # Check the original suffix first (it's the most likely continuation)
        original_suffix = "\n".join(steps[i + 1:])
        combined = prefix + "\n" + original_suffix
        if verifier_fn(combined):
            any_pass = True

        # If the original didn't pass, try sampling noisy continuations
        # (in the full version, this would use the model to generate
        # multiple continuations from this step).
        if not any_pass and n_continuations > 1:
            for _ in range(n_continuations - 1):
                # Simple heuristic: drop or add random lines as noise
                # (placeholder - in production, use model.generate)
                noisy = _noisy_continuation(steps, i)
                if noisy and verifier_fn(noisy):
                    any_pass = True
                    break

        if any_pass:
            # Everything up to and including step i is on a fixable path
            labels[i] = 1
            # All preceding steps inherit the positive label
            for j in range(i):
                labels[j] = 1
            break
        # else: step i is unfixable -> stays 0, continue backward

    return labels


def label_steps_execution(steps: list[str]) -> list[int]:
    """Label steps using execution feedback (CodePRM-inspired).

    For each prefix up to step i, check if the code COMPILES (AST parse)
    and EXECUTES (runs without exception).  This is cheaper than Monte-Carlo
    sampling and provides a direct signal about code quality at each step.

    A step is labeled 1 if the prefix up to and including it is syntactically
    valid and runs without error.

    Returns:
        A list of length ``len(steps)`` with 0/1 labels.
    """
    labels = []
    for i in range(len(steps)):
        prefix = "\n".join(steps[:i + 1])
        good_step = False
        # Check 1: does it compile?
        try:
            compile(prefix, "<prm>", "exec", flags=ast.PyCF_ONLY_AST)
            # Check 2: does it execute without crashing?
            try:
                exec(prefix, {"__builtins__": __builtins__})
                good_step = True
            except Exception:
                good_step = False  # runtime error
        except SyntaxError:
            good_step = False  # doesn't parse
        labels.append(1 if good_step else 0)
    return labels


if __name__ == "__main__":
    # Quick self-test
    steps = extract_steps("def foo():\n    return 1\n")
    print("Steps:", steps)
    print("Labels:", label_steps_execution(steps))
    steps2 = extract_steps("def broken(\n    pass\n")
    print("Broken steps:", steps2)
    print("Broken labels:", label_steps_execution(steps2))


def _noisy_continuation(steps: list[str], cutoff: int) -> str | None:
    """Generate a noisy continuation from step *cutoff*."""
    # Simple noise: randomly remove or duplicate some steps
    suffix = steps[cutoff + 1:]
    if not suffix:
        return None
    rng = random.Random(42)
    if rng.random() < 0.3 and len(suffix) > 1:
        suffix = suffix[:-rng.randint(1, min(3, len(suffix)))]
    return "\n".join(steps[:cutoff + 1]) + "\n" + "\n".join(suffix)


# ---------------------------------------------------------------------------
# PRM head: a lightweight MLP that takes the base model's last hidden state
# at a step boundary and outputs a scalar p ∈ [0, 1].
# ---------------------------------------------------------------------------

class PRMHead(nn.Module):
    """Process Reward Model head - predicts p(step is on passing trajectory).

    Architecture: a 2-layer MLP on top of the base model's hidden state.
    The base model is frozen; only this head is trained (~1M params for a
    10B model with hidden_dim=4096).

    * ``hidden_dim``: the base model's embedding dimension (cfg.n_embd).
    * ``intermediate_dim``: hidden size of the PRM MLP (default: hidden_dim // 2).
    """

    def __init__(self, hidden_dim: int | None = None,
                 intermediate_dim: int | None = None,
                 exec_feature_dim: int = 0):
        """
        Args:
            hidden_dim: Base model embedding dimension (cfg.n_embd).
            intermediate_dim: PRM MLP hidden size (default: hidden_dim // 2).
            exec_feature_dim: Execution feedback features (0 = disabled).
        """
        super().__init__()
        hidden_dim = hidden_dim or cfg.n_embd
        input_dim = hidden_dim + exec_feature_dim
        inter = intermediate_dim or max(64, hidden_dim // 2)
        self.exec_feature_dim = exec_feature_dim
        self.fc1 = nn.Linear(input_dim, inter, bias=True)
        self.fc2 = nn.Linear(inter, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor,
                exec_features: torch.Tensor | None = None) -> torch.Tensor:
        """Score *hidden_states*, optionally conditioned on *exec_features*.

        Args:
            hidden_states: (B, hidden_dim) or (B, T, hidden_dim).
                If 3D, the last timestep is used.
            exec_features: (B, exec_feature_dim) or None.
                CodePRM-inspired execution feedback (compile status,
                runtime status, etc.) concatenated with hidden states.

        Returns:
            Scores as logits (B,) before sigmoid.
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        if exec_features is not None and self.exec_feature_dim > 0:
            hidden_states = torch.cat([hidden_states, exec_features], dim=-1)
        x = F.gelu(self.fc1(hidden_states))
        x = self.fc2(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Training data collection from RFT/RLVR rollouts.
# ---------------------------------------------------------------------------

def collect_training_data(
    rollouts_path: str,
    out_path: str,
    verifier_fn: Callable[[str], bool] | None = None,
    n_continuations: int = 3,
    label_mode: str = "mc",
) -> dict:
    """Convert RFT/RLVR rollout data into step-level PRM training examples.

    Args:
        rollouts_path: Path to RFT/RLVR rollout JSON.
        out_path: Output path for PRM training data.
        verifier_fn: Optional verifier for MC labeling.
        n_continuations: MC continuations per step (MC mode only).
        label_mode: "mc" (Monte-Carlo backward) or "exec" (execution feedback).

    Expected *rollouts_path* format (from rft.py ``rft_data.json``):
      [
        {
          "text": <full solution code>,
          "passes": true/false,
          "problem_id": "...",
          "stdout": "...",
          "stderr": "..."
        },
        ...
      ]

    Output format:
      {
        "examples": [
          {
            "problem_id": "...",
            "prefix_code": <code up to step i>,
            "step_code": <step i content>,
            "label": 0/1,
            "full_code": <the original full solution>
          },
          ...
        ],
        "stats": { "total_steps": ..., "positive": ..., "negative": ... }
      }

    Expected *rollouts_path* format (from rft.py ``rft_data.json``):
      [
        {
          "text": <full solution code>,
          "passes": true/false,
          "problem_id": "...",
          "stdout": "...",
          "stderr": "..."
        },
        ...
      ]

    Output format:
      {
        "examples": [
          {
            "problem_id": "...",
            "prefix_code": <code up to step i>,
            "step_code": <step i content>,
            "label": 0/1,
            "full_code": <the original full solution>
          },
          ...
        ],
        "stats": { "total_steps": ..., "positive": ..., "negative": ... }
      }
    """
    with open(rollouts_path) as f:
        rollouts = json.load(f)

    examples: list[dict] = []
    stats = {"total_steps": 0, "positive": 0, "negative": 0}

    for rollout in rollouts:
        code = rollout.get("text", rollout.get("code", ""))
        passes = rollout.get("passes", False)
        problem_id = rollout.get("problem_id", rollout.get("id", "unknown"))

        steps = extract_steps(code)
        if len(steps) <= 1:
            continue   # skip trivial single-step solutions

        if label_mode == "exec":
            labels = label_steps_execution(steps)
        else:
            labels = label_steps(steps, passes, verifier_fn, n_continuations)

        for i, (step_text, label) in enumerate(zip(steps, labels)):
            prefix = "\n".join(steps[:i])
            examples.append({
                "problem_id": problem_id,
                "prefix_code": prefix,
                "step_code": step_text,
                "label": label,
                "full_code": code,
            })

        stats["total_steps"] += len(steps)
        stats["positive"] += sum(labels)
        stats["negative"] += len(labels) - sum(labels)

    out = {"examples": examples, "stats": stats}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[prm] collected {len(examples)} step-level examples from "
          f"{len(rollouts)} rollouts ({stats['positive']} + / "
          f"{stats['negative']} -)")
    return out


# ---------------------------------------------------------------------------
# PRM head training.
# ---------------------------------------------------------------------------

def build_prm_dataset(
    data_path: str,
    model: GPT,
    max_examples: int = 50_000,
) -> list[dict]:
    """Build (hidden_state, label) pairs from collected step data.

    For each example, run the frozen base model on the prefix + step and
    extract the last hidden state.  This produces the actual training
    examples for the PRM head.

    Returns a list of dicts with keys "hidden" (torch.Tensor) and "label".
    """
    with open(data_path) as f:
        data = json.load(f)

    examples = data["examples"]
    if max_examples and len(examples) > max_examples:
        # Stratified sample: keep equal positive/negative ratio
        pos = [e for e in examples if e["label"] == 1]
        neg = [e for e in examples if e["label"] == 0]
        half = max_examples // 2
        rng = random.Random(42)
        rng.shuffle(pos)
        rng.shuffle(neg)
        examples = pos[:half] + neg[:half]
        rng.shuffle(examples)

    device = next(model.parameters()).device
    dataset: list[dict] = []
    model.eval()

    with torch.no_grad():
        for i, ex in enumerate(examples):
            if i > 0 and i % 500 == 0:
                print(f"[prm] encoded {i}/{len(examples)} examples...")

            # Build input text: prefix + step (but we want the hidden state
            # *after* the step, which requires a forward pass through the
            # step's tokens).
            input_code = ex["prefix_code"] + "\n" + ex["step_code"]
            input_ids = encode(input_code)
            if len(input_ids) > cfg.block_size:
                # Truncate from the left (keep the step and end of prefix)
                input_ids = input_ids[-(cfg.block_size):]

            inp = torch.tensor([input_ids], dtype=torch.long, device=device)
            # Forward pass - we need the hidden state before lm_head
            # The model returns (logits, loss, new_caches).  We need the
            # hidden state from the transformer body, not the logits.
            # We access it by using the model's forward and capturing the
            # output of ln_f (the final norm).
            # Since GPT doesn't expose hidden states directly, we do a
            # manual forward pass through the blocks.
            hidden = _extract_hidden(model, inp)

            # Take the last token's hidden state as the step representation
            step_hidden = hidden[0, -1, :].cpu()   # (hidden_dim,)
            dataset.append({
                "hidden": step_hidden,
                "label": ex["label"],
            })

    return dataset


def _extract_hidden(model: GPT, inp: torch.Tensor) -> torch.Tensor:
    """Run *model* on *inp* and return the transformer body output (after
    ln_f but before lm_head).

    This is a stripped-down forward pass that goes through every block
    and stops before the language model head.
    """
    B, T = inp.shape
    model._ensure_rope(inp.device, inp.dtype)
    rope_cos = model.rope_cos[:T]
    rope_sin = model.rope_sin[:T]

    x = model.wte(inp)
    for block in model.blocks:
        attn_out, new_kv = block.attn(
            block.ln_1(x), rope_cos, rope_sin, None, False
        )
        x = x + attn_out
        ff_in = block.ln_2(x)
        if block.is_moe:
            ff_out, _, _ = block.mlp(ff_in)
        else:
            ff_out = block.mlp(ff_in)
        x = x + ff_out
    x = model.ln_f(x)
    return x


def train_prm_head(
    dataset: list[dict],
    model: GPT,
    prm_head: PRMHead,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    out_path: str = "models/prm.pt",
):
    """Train the PRM head on extracted hidden states.

    The base model stays frozen - only the PRM head is updated.
    """
    device = next(model.parameters()).device
    prm_head = prm_head.to(device)
    optimizer = torch.optim.AdamW(prm_head.parameters(), lr=lr, weight_decay=0.01)

    # Stack all hidden states and labels
    all_hidden = torch.stack([d["hidden"] for d in dataset]).to(device)
    all_labels = torch.tensor([d["label"] for d in dataset],
                              dtype=torch.float32, device=device)

    n = len(dataset)
    best_loss = float("inf")

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n, device=device)
        all_hidden = all_hidden[perm]
        all_labels = all_labels[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch_hidden = all_hidden[i:i + batch_size]
            batch_labels = all_labels[i:i + batch_size]

            logits = prm_head(batch_hidden)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prm_head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[prm] epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            if out_path:
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                torch.save({
                    "prm_head_state_dict": prm_head.state_dict(),
                    "hidden_dim": cfg.n_embd,
                    "loss": best_loss,
                }, out_path)
                print(f"[prm] saved best checkpoint to {out_path}")


# ---------------------------------------------------------------------------
# Inference: using the PRM to score solutions during Best-of-N.
# ---------------------------------------------------------------------------

def score_solution(
    code: str,
    model: GPT,
    prm_head: PRMHead,
    device: torch.device | None = None,
) -> float:
    """Score a complete solution by its step-level PRM probabilities.

    Returns the *minimum* step score (worst-case step) - a pessimistic
    estimate that a single bad step kills the whole solution.
    """
    if device is None:
        device = next(model.parameters()).device

    steps = extract_steps(code)
    if not steps:
        return 0.0

    step_scores: list[float] = []
    prefix = ""

    model.eval()
    with torch.no_grad():
        for step in steps:
            prefix = (prefix + "\n" + step).strip()
            input_ids = encode(prefix)
            if len(input_ids) > cfg.block_size:
                input_ids = input_ids[-(cfg.block_size):]

            inp = torch.tensor([input_ids], dtype=torch.long, device=device)
            hidden = _extract_hidden(model, inp)
            step_hidden = hidden[0, -1:, :]   # (1, hidden_dim)
            logit = prm_head(step_hidden)       # (1,)
            prob = torch.sigmoid(logit).item()
            step_scores.append(prob)

    # Return minimum step score (pessimistic)
    return min(step_scores) if step_scores else 0.0


def improve_with_prm(
    code: str,
    model: GPT,
    prm_head: PRMHead,
    generate_fn: Callable[[str], str],
    device: torch.device | None = None,
    threshold: float = 0.3,
    max_iters: int = 5,
) -> str:
    """Iteratively refine a solution using PRM step-level feedback.

    CodePRM-style Generate-Verify-Refine (GVR) pipeline:
      1. Score each step with the PRM
      2. Identify the first step below *threshold*
      3. Crop the solution at that step and regenerate
      4. Repeat until all steps score above threshold or *max_iters*.

    Args:
        code: The candidate solution to refine.
        model: Base model (for hidden state extraction).
        prm_head: Trained PRM head.
        generate_fn: Function that takes a prefix string and returns
                     a continuation string.
        threshold: PRM score threshold (default 0.3).
        max_iters: Maximum refinement iterations.

    Returns:
        Refined code string (may be unchanged if no improvement found).
    """
    if device is None:
        device = next(model.parameters()).device

    current_code = code

    for iteration in range(max_iters):
        steps = extract_steps(current_code)
        if not steps or len(steps) <= 1:
            break  # single-step solutions can't be refined

        # Score each step prefix
        prefix = ""
        worst_idx = -1
        worst_score = 1.0

        model.eval()
        with torch.no_grad():
            for i, step in enumerate(steps):
                prefix = (prefix + "\n" + step).strip()
                input_ids = encode(prefix)
                if len(input_ids) > cfg.block_size:
                    input_ids = input_ids[-cfg.block_size:]

                inp = torch.tensor([input_ids], dtype=torch.long, device=device)
                hidden = _extract_hidden(model, inp)
                step_hidden = hidden[0, -1:, :]
                logit = prm_head(step_hidden)
                prob = torch.sigmoid(logit).item()

                if prob < worst_score:
                    worst_score = prob
                    worst_idx = i

        # If all steps score above threshold, we're done
        if worst_score >= threshold:
            break

        # Crop at the worst step and regenerate
        prefix = "\n".join(steps[:worst_idx])
        continuation = generate_fn(prefix)
        current_code = (prefix + "\n" + continuation).strip()

    return current_code


# ---------------------------------------------------------------------------
# CLI: ``python prm.py collect|train``
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process Reward Model for code"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- collect ---
    collect_p = sub.add_parser("collect", help="Collect step-level PRM data")
    collect_p.add_argument("--rollouts", required=True,
                           help="Path to RFT/RLVR rollout JSON")
    collect_p.add_argument("--out", default="prm_train_data.json",
                           help="Output path for collected examples")
    collect_p.add_argument("--n-continuations", type=int, default=3,
                           help="Monte-Carlo continuations per step")
    collect_p.add_argument("--label-mode", default="mc", choices=["mc", "exec"],
                           help="Labeling mode: 'mc' (Monte-Carlo) or 'exec' (execution feedback)")

    # --- train ---
    train_p = sub.add_parser("train", help="Train PRM head")
    train_p.add_argument("--data", required=True,
                         help="PRM training data JSON (from 'collect')")
    train_p.add_argument("--ckpt", default="models/prm.pt",
                         help="Output path for PRM head checkpoint")
    train_p.add_argument("--lr", type=float, default=1e-3,
                         help="PRM head learning rate")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--batch-size", type=int, default=64)

    # --- score ---
    score_p = sub.add_parser("score", help="Score a solution with the PRM")
    score_p.add_argument("--code", help="Code string to score")
    score_p.add_argument("--file", help="Path to code file to score")
    score_p.add_argument("--prm-ckpt", required=True,
                         help="PRM head checkpoint")
    score_p.add_argument("--model-ckpt", required=True,
                         help="Base model checkpoint")
    score_p.add_argument("--profile", default=None,
                         help="Model profile (if checkpoint doesn't have one)")

    args = parser.parse_args()

    if args.command == "collect":
        collect_training_data(
            rollouts_path=args.rollouts,
            out_path=args.out,
            n_continuations=args.n_continuations,
            label_mode=args.label_mode,
        )

    elif args.command == "train":
        print("[prm] loading base model...")
        # Build the full model to get the hidden dimension
        model = GPT(gradient_checkpointing=cfg.gradient_checkpointing)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        print(f"[prm] building dataset (extracting hidden states)...")
        dataset = build_prm_dataset(args.data, model)

        print(f"[prm] initializing PRM head (hidden_dim={cfg.n_embd})...")
        prm_head = PRMHead(hidden_dim=cfg.n_embd)

        train_prm_head(
            dataset=dataset,
            model=model,
            prm_head=prm_head,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_path=args.ckpt,
        )
        print(f"[prm] done. head saved to {args.ckpt}")

    elif args.command == "score":
        if args.code:
            code = args.code
        elif args.file:
            with open(args.file) as f:
                code = f.read()
        else:
            print("error: provide --code or --file", file=sys.stderr)
            sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[prm] loading base model from {args.model_ckpt}...")
        model = GPT.from_checkpoint(args.model_ckpt)
        model.to(device)
        model.eval()

        prm_state = torch.load(args.prm_ckpt, map_location=device, weights_only=True)
        prm_head = PRMHead(hidden_dim=prm_state.get("hidden_dim", cfg.n_embd))
        prm_head.load_state_dict(prm_state["prm_head_state_dict"])
        prm_head.to(device)
        prm_head.eval()

        score = score_solution(code, model, prm_head, device)
        print(f"[prm] score: {score:.4f}")


if __name__ == "__main__":
    main()