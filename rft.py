"""Rejection-sampling Fine-Tuning (RFT) — Stage 1 of the RL pipeline.

THE IDEA
--------
Before we can run RL (GRPO / RLVR), the model needs to be able to *sometimes*
produce a correct solution.  If the base model never passes a test, RL gets zero
reward signal on every rollout — no gradient.

RFT bootstraps the model:
  1. Take coding problems WITH hidden tests (HumanEval, MBPP, …).
  2. Sample N solutions from the current model at high temperature (N = 16-64).
  3. Run each in the sandbox against the visible tests.
  4. Keep only the solutions that PASS **and** are behaviourally diverse.
  5. SFT (standard supervised fine-tuning) on the kept (problem → solution) pairs.

The resulting model "knows how to pass tests" — not perfectly, but well enough
that RL can find signal.

USAGE
-----
  # After training a base model (model.pt):
  python rft.py --ckpt model.pt --n 32 --problems humaneval --out rft_data.json
  # Then SFT on the collected data:
  python rft.py --train rft_data.json --ckpt model.pt --out model_rft.pt
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

# Make flat imports work from the project root
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
from model import GPT
import config as cfg
from tokenizer import encode, decode, EOT_TOKEN
from agent.sandbox import run_cmd, CmdResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RFTExample:
    """A (problem, solution) pair that passed verification."""
    problem_id: str
    prompt: str           # the model input (signature + docstring)
    solution: str         # the model output (the completed body)
    tests: str            # the test suite that was used for filtering


# ---------------------------------------------------------------------------
# Problem sources
# ---------------------------------------------------------------------------

def load_problems(source: str, limit: int | None = None) -> list:
    """Load problems from a supported source.  Returns a list of objects with
    ``.id``, ``.prompt``, ``.tests``, and optionally ``.entry_point``."""
    if source == "humaneval":
        from eval.humaneval_loader import load
        return load(limit=limit)
    elif source.endswith(".json"):
        with open(source) as f:
            raw = json.load(f)
        # Expect a list of dicts with keys: id, prompt, tests, [entry_point]
        out = []
        for r in raw[:limit] if limit else raw:
            from verify.verifier import Problem
            out.append(Problem(
                id=r["id"], prompt=r["prompt"], tests=r["tests"],
                entry_point=r.get("entry_point"),
            ))
        return out
    else:
        raise ValueError(f"unknown problem source {source!r}")


# ---------------------------------------------------------------------------
# Solution sampling
# ---------------------------------------------------------------------------

def sample_solutions(
    model: GPT,
    problem,
    n: int,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_size: int = 8,
    device: str = "cpu",
) -> list[str]:
    """Sample *n* completions for *problem* from *model*.

    Returns a list of *n* completion strings (the model output after the
    prompt, truncated at the first top-level ``def``/``class`` boundary).
    """
    prompt_ids = torch.tensor([encode(problem.prompt)], dtype=torch.long, device=device)
    L = prompt_ids.size(1)
    out: list[str] = []

    # Stop-tokens for truncation (match eval/eval.py)
    _STOP = ["\nclass ", "\ndef ", "\nif __name__", "\n\n\n"]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            bs = min(batch_size, n - start)
            batch = prompt_ids.expand(bs, -1).contiguous()
            gen = model.generate(
                batch, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=50, top_p=top_p,
                typical_p=0.2,
                repetition_penalty=1.0,  # off for code
            )
            for b in range(bs):
                comp = decode(gen[b, L:].tolist())
                # Truncate at first top-level boundary
                cut = len(comp)
                for tok in _STOP:
                    i = comp.find(tok)
                    if i != -1 and i < cut:
                        cut = i
                out.append(comp[:cut])
    return out


# ---------------------------------------------------------------------------
# Verification and filtering
# ---------------------------------------------------------------------------

def verify_solution(problem, solution: str, timeout: float = 15.0) -> bool:
    """Return True if *solution* passes *problem*'s tests."""
    from verify.verifier import verify, _python_check_code
    # Assemble the runnable code
    cwd = os.path.join(PROJ_ROOT, "verify", "_runs")
    os.makedirs(cwd, exist_ok=True)
    code = _python_check_code(problem, problem.prompt + solution) \
        if hasattr(problem, 'entry_point') else (
            f"{problem.prompt}{solution}\n{problem.tests}\nprint('OK')\n"
        )
    # Write and run
    import hashlib
    tag = hashlib.md5(problem.id.encode()).hexdigest()[:12]
    run_file = os.path.join(cwd, f"_rft_{tag}.py")
    with open(run_file, "w") as f:
        f.write(code)
    r = run_cmd(f"python {os.path.basename(run_file)}", timeout=timeout, cwd=cwd)
    return r.rc == 0 and "OK" in r.stdout


def behaviourally_diverse(
    solutions: list[str], threshold: float = 0.85,
) -> list[str]:
    """Deduplicate solutions by behavioural similarity.

    Crude heuristic: two solutions that differ by fewer than *threshold*
    fraction of characters are treated as identical (likely the same algorithm
    with minor formatting).  Returns a subset of *solutions*.
    """
    if not solutions:
        return []
    # Sort by length so short solutions are compared first
    kept = [solutions[0]]
    for sol in solutions[1:]:
        # Check against all kept solutions
        is_new = True
        for k in kept:
            # Normalise: strip whitespace for comparison
            s_norm = "".join(sol.split())
            k_norm = "".join(k.split())
            if len(s_norm) == 0 or len(k_norm) == 0:
                continue
            # Jaccard-like character overlap
            shorter = min(len(s_norm), len(k_norm))
            if shorter == 0:
                continue
            matches = sum(1 for i in range(shorter) if s_norm[i] == k_norm[i])
            if matches / shorter >= threshold:
                is_new = False
                break
        if is_new:
            kept.append(sol)
    return kept


# ---------------------------------------------------------------------------
# Collect phase: sample → verify → filter
# ---------------------------------------------------------------------------

def collect(
    ckpt_path: str,
    problem_source: str,
    n: int = 32,
    limit: int | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    timeout: float = 15.0,
    batch_size: int = 8,
    dedup_threshold: float = 0.85,
    device: str = "",
    out_path: str = "rft_data.json",
    verbose: bool = True,
) -> list[RFTExample]:
    """Sample, verify, dedup, and collect passing solutions.

    Returns the kept RFTExamples and writes them to *out_path* as JSON.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    problems = load_problems(problem_source, limit=limit)

    if verbose:
        print(f"[rft] collecting from {len(problems)} problems, "
              f"n={n}, temp={temperature}, device={device}")

    model = GPT.from_checkpoint(ckpt_path, device) if ckpt_path else None

    examples: list[RFTExample] = []
    t0 = time.time()

    for pi, problem in enumerate(problems):
        # 1. Sample
        if model is not None:
            comps = sample_solutions(
                model, problem, n, max_new_tokens, temperature, top_p,
                batch_size, device,
            )
        else:
            # Sanity mode: use the problem's canonical solution
            comps = [getattr(problem, 'canonical_solution', '')] if hasattr(problem, 'canonical_solution') else []

        # 2. Verify each
        passing: list[str] = []
        for ci, sol in enumerate(comps):
            if verify_solution(problem, sol, timeout=timeout):
                passing.append(sol)

        # 3. Dedup
        unique = behaviourally_diverse(passing, threshold=dedup_threshold)

        for sol in unique:
            examples.append(RFTExample(
                problem_id=problem.id,
                prompt=problem.prompt,
                solution=sol,
                tests=problem.tests,
            ))

        if verbose:
            elapsed = time.time() - t0
            rate = (pi + 1) / max(elapsed, 1e-6)
            print(f"[rft] {pi + 1}/{len(problems)} {problem.id}: "
                  f"{len(passing)}/{n} pass → {len(unique)} kept "
                  f"({rate:.1f} prob/s, total {len(examples)} examples)")

    # Write out
    with open(out_path, "w") as f:
        json.dump([e.__dict__ for e in examples], f, indent=2)
    if verbose:
        print(f"\n[rft] wrote {len(examples)} examples to {out_path}")
        print(f"[rft] elapsed {time.time() - t0:.1f}s")

    return examples


# ---------------------------------------------------------------------------
# Train phase: SFT on the collected data
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    ckpt_path: str = "model.pt",
    out_path: str = "model_rft.pt",
    lr: float = 5e-6,
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum: int = 2,
    device: str = "",
    verbose: bool = True,
):
    """Supervised fine-tune on collected RFT data.

    Loads the base model from *ckpt_path*, trains on the (prompt → solution)
    pairs from *data_path*, and saves to *out_path*.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(data_path) as f:
        raw = json.load(f)
    examples = [RFTExample(**r) for r in raw]

    if verbose:
        print(f"[rft] training on {len(examples)} examples, {epochs} epochs, "
              f"lr={lr}, device={device}")

    model = GPT.from_checkpoint(ckpt_path, device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # Format: prompt + solution + EOT. Loss only on the solution tokens.
    total_steps = epochs * max(1, len(examples) // (batch_size * grad_accum))
    step = 0

    for epoch in range(epochs):
        # Shuffle
        indices = torch.randperm(len(examples)).tolist()
        for start in range(0, len(examples), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_ex = [examples[i] for i in batch_idx]
            bs = len(batch_ex)

            # Build (input, target) pairs with loss masking
            xs, ys = [], []
            for ex in batch_ex:
                prompt_ids = encode(ex.prompt)
                solution_ids = encode(ex.solution) + [EOT_TOKEN]
                full = (prompt_ids + solution_ids)[:cfg.block_size + 1]
                x = full[:-1]
                y = full[1:].copy()
                # Mask prompt tokens — model is scored only on the solution
                for i in range(len(prompt_ids) - 1):
                    if i < len(y):
                        y[i] = -100
                xs.append(x)
                ys.append(y)

            # Pad to uniform length
            max_len = max(len(x) for x in xs)
            for i in range(len(xs)):
                pad = max_len - len(xs[i])
                if pad > 0:
                    xs[i] = xs[i] + [EOT_TOKEN] * pad
                    ys[i] = ys[i] + [-100] * pad

            x_t = torch.tensor(xs, dtype=torch.long, device=device)
            y_t = torch.tensor(ys, dtype=torch.long, device=device)

            # Accumulated forward/backward
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for mb in range(grad_accum):
                mb_start = mb * bs // grad_accum
                mb_end = (mb + 1) * bs // grad_accum
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16):
                    _, loss, _ = model(x_t[mb_start:mb_end], targets=y_t[mb_start:mb_end])
                scaler.scale(loss / grad_accum).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if verbose and step % 20 == 0:
                print(f"[rft] epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                      f"loss {accum_loss / grad_accum:.4f}")

    model_to_save = getattr(model, "_orig_mod", model)
    torch.save(model_to_save.state_dict(), out_path)
    if verbose:
        print(f"[rft] saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="RFT — Rejection-sampling Fine-Tuning")
    sub = ap.add_subparsers(dest="command", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Sample, verify, dedup solutions")
    p_collect.add_argument("--ckpt", default="model.pt", help="Base checkpoint")
    p_collect.add_argument("--problems", default="humaneval", help="Source (humaneval | path/to/problems.json)")
    p_collect.add_argument("--n", type=int, default=32, help="Samples per problem")
    p_collect.add_argument("--limit", type=int, default=None, help="Problems to grade")
    p_collect.add_argument("--temperature", type=float, default=1.0)
    p_collect.add_argument("--max-tokens", type=int, default=512)
    p_collect.add_argument("--timeout", type=float, default=15.0)
    p_collect.add_argument("--batch-size", type=int, default=8)
    p_collect.add_argument("--dedup-threshold", type=float, default=0.85)
    p_collect.add_argument("--out", default="rft_data.json")
    p_collect.add_argument("--device", default="")

    # train
    p_train = sub.add_parser("train", help="SFT on collected RFT data")
    p_train.add_argument("data", help="Path to collected data JSON")
    p_train.add_argument("--ckpt", default="model.pt", help="Base checkpoint")
    p_train.add_argument("--out", default="model_rft.pt", help="Output checkpoint")
    p_train.add_argument("--lr", type=float, default=5e-6)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--grad-accum", type=int, default=2)
    p_train.add_argument("--device", default="")

    args = ap.parse_args()

    if args.command == "collect":
        collect(
            ckpt_path=args.ckpt, problem_source=args.problems, n=args.n,
            limit=args.limit, temperature=args.temperature,
            max_new_tokens=args.max_tokens, timeout=args.timeout,
            batch_size=args.batch_size, dedup_threshold=args.dedup_threshold,
            device=args.device, out_path=args.out,
        )
    elif args.command == "train":
        train(
            data_path=args.data, ckpt_path=args.ckpt, out_path=args.out,
            lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
            grad_accum=args.grad_accum, device=args.device,
        )


if __name__ == "__main__":
    main()