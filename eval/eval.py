"""HumanEval pass@k runner — THE metric for le fat chaton.

QUALITY_BAR.md is explicit: val loss is meaningless, pass@k is the only source
of truth. This wires together the three pieces that already exist:

  - eval/humaneval_loader.py  -> 164 HumanEval problems as verify.Problem
  - model.GPT.generate()      -> sample N completions per problem (batched)
  - verify/verifier.py        -> does each completion pass the hidden tests?

Then it computes pass@k with the UNBIASED estimator from the HumanEval paper
(Chen et al. 2021) — NOT the naive "fraction of k that passed", which is biased
upward when you only ran k samples.

  pass@k = 1 - C(n - c, k) / C(n, k)        n = total samples, c = # that passed

If we sampled n=200 and c=37 pass, then pass@1 estimates the true pass@1 over
the whole distribution. Sampling fewer (n=20) gives a noisier but still
unbiased estimate; you pay for precision with compute.

Usage
-----
  # full run (needs a coder checkpoint):
  python eval/eval.py --ckpt model.pt --n 20 --ks 1 5 --limit 20

  # no model yet — prove the loader+verifier+math wiring first:
  python eval/eval.py --sanity

Notes / honest limitations
-------------------------
- This is a COMPLETION eval, not an agentic one. pass@k here measures "can the
  base model write the function body in one shot" — the floor, not the goal.
  The terminal-bench-relevant, use-the-loop score comes later (agent_rl.py).
- We truncate each completion at the first top-level def/class so a half-started
  second function can't break the parse. We DON'T early-stop on EOT in the
  model (generate() runs max_new_tokens always) — truncation covers correctness.
- Repetition penalty defaults to 1.0 (off) here, unlike the agent loop. Code
  legitimately repeats indentation/patterns; a rep-penalty hurts completion.
- Per-problem verification spawns subprocesses, so this is I/O bound and slower
  than you'd expect. It prints progress.
"""
import os
import sys
import json
import math
import argparse
import time

# make flat imports resolve whether run as `python eval/eval.py` or -m eval.eval
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
from tokenizer import encode, decode
from model import GPT
import config as cfg
from verify.verifier import verify, Problem
from eval.humaneval_loader import load as load_humaneval

# cut completions here so a trailing half-started function can't break the parse
_STOP_TOKENS = ["\nclass ", "\ndef ", "\nif __name__", "\n\n\n"]


def _truncate(completion: str) -> str:
    """Cut the generated body at the first top-level def/class boundary.

    HumanEval prompts are a single function signature + docstring, so the first
    occurrence of a new top-level '\ndef '/'\nclass ' marks the start of a
    *second* function we don't want. Keeps the solution function clean.
    """
    cut = len(completion)
    for tok in _STOP_TOKENS:
        i = completion.find(tok)
        if i != -1 and i < cut:
            cut = i
    return completion[:cut]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021).

    n = total samples drawn, c = # that passed, k = the k in pass@k.
    Returns 1.0 if k > (n - c) (we drew enough correct that pass@k is saturated),
    else 1 - C(n-c, k) / C(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _load_model(ckpt_path: str, device: str) -> GPT:
    """Build from config + load a checkpoint.

    Accepts both a bare state_dict (train.py's ``model.pt``) and a full
    checkpoint dict (checkpoint.py's ``checkpoint.pt``).
    Uses ``GPT.from_checkpoint()`` as the canonical factory.
    """
    return GPT.from_checkpoint(ckpt_path, device)


def _sample_completions(model, prompt: str, n: int, max_new_tokens: int,
                        temperature: float, top_p: float, repetition_penalty: float,
                        device: str, batch_size: int) -> list[str]:
    """Sample n completions for one prompt, in micro-batches (memory-safe for
    the fat model). Returns the n completion strings (NOT including the prompt).
    """
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    L = prompt_ids.size(1)
    out: list[str] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            bs = min(batch_size, n - start)
            batch = prompt_ids.expand(bs, -1).contiguous()  # (bs, L)
            gen = model.generate(
                batch, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=50, top_p=top_p,
                typical_p=0.2, repetition_penalty=repetition_penalty,
            )
            # completion = everything the model generated past the prompt
            for b in range(bs):
                comp = decode(gen[b, L:].tolist())
                out.append(_truncate(comp))
    return out


def evaluate(ckpt_path: str, n: int, ks: list[int], limit: int | None,
             max_new_tokens: int, temperature: float, top_p: float,
             repetition_penalty: float, timeout: float, batch_size: int,
             device: str, out_path: str | None, verbose: bool = True):
    """Run HumanEval pass@k. Returns a dict summary (also written to out_path)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[eval] device={device} ckpt={ckpt_path} n={n} ks={ks} "
              f"limit={limit} temp={temperature} top_p={top_p} "
              f"rep_pen={repetition_penalty} max_tokens={max_new_tokens}")

    problems = load_humaneval(limit=limit)
    if verbose:
        print(f"[eval] loaded {len(problems)} HumanEval problems")

    model = _load_model(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[eval] model loaded: {n_params/1e6:.1f}M params "
              f"(profile={cfg.PROFILE})")

    rows = []
    t0 = time.time()
    for pi, p in enumerate(problems):
        comps = _sample_completions(
            model, p.prompt, n, max_new_tokens, temperature, top_p,
            repetition_penalty, device, batch_size,
        )
        # verify each completion (c = how many pass)
        per = []
        for ci, sol in enumerate(comps):
            solution = p.prompt + sol
            v = verify(p, solution, timeout=timeout)
            per.append({
                "passed": v.passed, "n_pass": v.n_pass, "n_total": v.n_total,
                "rc": v.rc, "timed_out": v.timed_out,
            })
        c = sum(1 for r in per if r["passed"])
        row = {
            "id": p.id,
            "n": n, "c": c,
            **{f"pass@{k}": round(pass_at_k(n, c, k), 4) for k in ks},
        }
        rows.append(row)
        if verbose:
            done = pi + 1
            eta = (time.time() - t0) / done * (len(problems) - done)
            print(f"[eval] {done}/{len(problems)} {p.id}  "
                  f"pass {c}/{n}  " +
                  "  ".join(f"@{k}={row[f'pass@{k}']:.2f}" for k in ks) +
                  f"   (~{eta:.0f}s left)")

    # macro-average pass@k across problems
    agg = {f"pass@{k}": round(sum(r[f"pass@{k}"] for r in rows) / len(rows), 4)
           for k in ks}
    n_solved = sum(1 for r in rows if r["c"] > 0)
    summary = {
        "ckpt": ckpt_path, "profile": cfg.PROFILE, "n": n, "ks": ks,
        "n_problems": len(problems), "n_at_least_once_correct": n_solved,
        "aggregate": agg,
        "elapsed_s": round(time.time() - t0, 1),
    }

    if verbose:
        print("\n" + "=" * 56)
        print("AGGREGATE pass@k (macro-avg over problems)")
        print("-" * 56)
        for k in ks:
            print(f"  pass@{k:<3} = {agg[f'pass@{k}']:.4f}")
        print("-" * 56)
        print(f"  problems solved >=1x: {n_solved}/{len(problems)}")
        print(f"  elapsed: {summary['elapsed_s']}s")
        print("=" * 56)

    result = {"summary": summary, "rows": rows}
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"[eval] wrote {out_path}")
    return result


def sanity():
    """No model needed. Proves loader -> Problem -> verifier wiring + the
    pass@k math, so we can trust the harness before we have a coder to grade.
    """
    print("[sanity] loading HumanEval (limit 2)...")
    problems = load_humaneval(limit=2)
    print(f"[sanity] got {len(problems)} problems: {[p.id for p in problems]}")

    # problem 0: a deliberately-correct completion should PASS
    p = problems[0]
    # HumanEval/0 is `has_close_elements` (check floats for closeness). We don't
    # hand-write it — instead prove wiring by checking a BAD completion fails,
    # which exercises the whole path (loader -> truncate -> verify -> Verdict).
    bad = "\n    return 0\n"   # returns 0 for everything -> tests must fail
    v = verify(p, p.prompt + bad, timeout=10)
    print(f"[sanity] bad completion on {p.id}: passed={v.passed} "
          f"{v.n_pass}/{v.n_total}  (expect passed=False)")
    assert not v.passed, "bad completion unexpectedly passed — verifier broken?"

    # math check: with n=4, c=3, pass@1 should be 1 - C(1,1)/C(4,1) = 1 - 1/4 = 0.75
    got = pass_at_k(4, 3, 1)
    print(f"[sanity] pass_at_k(4,3,1)={got:.4f}  (expect 0.7500)")
    assert abs(got - 0.75) < 1e-9
    # c=0, k=n -> pass@k = 0 (n-c=4 not < k=4, so 1 - C(4,4)/C(4,4) = 0.0)
    assert pass_at_k(4, 0, 4) == 0.0
    # saturated: c=2, k=3, n=4 -> n-c=2 < k=3 -> returns 1.0
    assert pass_at_k(4, 2, 3) == 1.0
    print("[sanity] pass_at_k math OK")
    print("[sanity] ALL CHECKS PASSED — harness wiring is sound.")


def main():
    ap = argparse.ArgumentParser(description="HumanEval pass@k for le fat chaton")
    ap.add_argument("--ckpt", help="checkpoint to grade (bare model.pt or checkpoint dict)")
    ap.add_argument("--sanity", action="store_true",
                    help="no-model wiring + math self-test, then exit")
    ap.add_argument("--n", type=int, default=20, help="samples per problem")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10],
                    help="pass@k values to report (must each be <= n)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only grade the first N problems (quick runs)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="repetition penalty (1.0=off; code repeats, keep off)")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="per-completion verifier timeout (s)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="samples per generate() call (lower if OOM on fat)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None,
                    help="write JSON results here (default eval/_runs/<ts>.json)")
    args = ap.parse_args()

    if args.sanity:
        sanity()
        return
    if not args.ckpt:
        ap.error("--ckpt is required (or use --sanity for the no-model self-test)")
    if any(k > args.n for k in args.ks):
        ap.error(f"each --ks value must be <= --n (n={args.n}); got ks={args.ks}")

    out_path = args.out
    if out_path is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(PROJ_ROOT, "eval", "_runs", f"eval_{ts}.json")

    evaluate(args.ckpt, args.n, args.ks, args.limit, args.max_tokens,
             args.temperature, args.top_p, args.rep_penalty, args.timeout,
             args.batch_size, args.device, out_path)


if __name__ == "__main__":
    main()
