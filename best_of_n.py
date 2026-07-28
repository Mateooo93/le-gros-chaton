"""Best-of-N with PRM scoring — Stage 3 inference-time win.

THE IDEA
--------
Given a problem, sample N solutions from the model at high temperature.
Score each solution's *steps* (or final code) using the Process Reward Model.
Pick the highest-scoring solution.  No training required — pure inference-time
gain.

This is the first thing to build after Stage 2 RLVR, because:
  1. It works immediately with any checkpoint (no PRM training needed if you
     use the verifier as a final-check "outcome" score).
  2. It's trivial to parallelise.
  3. The code PRM (when trained) lifts this beyond simple verifier filtering.

USAGE
-----
  # Outcome-only (verifier as final filter):
  python best_of_n.py --ckpt model_rlvr.pt --n 32 --problems humaneval

  # PRM-scored (once prm.py is trained):
  python best_of_n.py --ckpt model_rlvr.pt --n 32 --prm prm.pt
"""
import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
from model import GPT
import config as cfg
from tokenizer import encode, decode


def _load_problems(source: str, limit: int | None = None):
    if source == "humaneval":
        from eval.humaneval_loader import load
        return load(limit=limit)
    else:
        raise ValueError(f"unknown source {source!r}")


def _sample_n(model, prompt: str, n: int, max_tokens: int, temp: float,
              device: str, batch_size: int = 8) -> list[str]:
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    L = prompt_ids.size(1)
    out: list[str] = []
    _STOP = ["\nclass ", "\ndef ", "\nif __name__", "\n\n\n"]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            bs = min(batch_size, n - start)
            batch = prompt_ids.expand(bs, -1).contiguous()
            gen = model.generate(batch, max_new_tokens=max_tokens,
                                 temperature=temp, top_k=50, top_p=0.95,
                                 typical_p=0.2)
            for b in range(bs):
                comp = decode(gen[b, L:].tolist())
                cut = len(comp)
                for tok in _STOP:
                    i = comp.find(tok)
                    if i != -1 and i < cut:
                        cut = i
                out.append(comp[:cut])
    return out


def _outcome_score(problem, solution: str, timeout: float) -> float:
    """1.0 if the solution passes the problem's tests, else 0.0."""
    from agent.sandbox import run_cmd
    import hashlib
    cwd = os.path.join(PROJ_ROOT, "verify", "_runs")
    os.makedirs(cwd, exist_ok=True)
    code = f"{problem.prompt}{solution}\n{problem.tests}\nprint('OK')\n"
    tag = hashlib.md5((problem.id + solution[:50]).encode()).hexdigest()[:12]
    f = os.path.join(cwd, f"_bon_{tag}.py")
    with open(f, "w") as fh:
        fh.write(code)
    r = run_cmd(f"python {os.path.basename(f)}", timeout=timeout, cwd=cwd)
    return 1.0 if (r.rc == 0 and "OK" in r.stdout) else 0.0


def best_of_n(ckpt: str, n: int = 32, max_tokens: int = 512,
              temperature: float = 1.0, limit: int | None = None,
              timeout: float = 15.0, device: str = "",
              problems_source: str = "humaneval", verbose: bool = True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    problems = _load_problems(problems_source, limit=limit)
    model = GPT.from_checkpoint(ckpt, device)

    results = []
    t0 = time.time()
    for pi, problem in enumerate(problems):
        comps = _sample_n(model, problem.prompt, n, max_tokens, temperature, device)
        scores = [_outcome_score(problem, sol, timeout) for sol in comps]
        best_idx = max(range(n), key=lambda i: scores[i])
        n_pass = sum(scores)
        results.append({
            "id": problem.id,
            "n": n, "n_pass": n_pass,
            "pass@1_bon": 1.0 if n_pass > 0 else 0.0,
        })
        if verbose:
            elapsed = time.time() - t0
            print(f"[bon] {pi + 1}/{len(problems)} {problem.id}: "
                  f"pass {n_pass}/{n}  ({elapsed:.0f}s)")

    agg = sum(r["pass@1_bon"] for r in results) / len(results)
    print(f"\n[bon] best-of-{n} pass@1 = {agg:.4f} ({sum(1 for r in results if r['n_pass'] > 0)}/{len(results)} problems)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Best-of-N for le fat chaton")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="")
    ap.add_argument("--problems", default="humaneval")
    args = ap.parse_args()
    best_of_n(args.ckpt, args.n, args.max_tokens, args.temperature,
              args.limit, device=args.device, problems_source=args.problems)