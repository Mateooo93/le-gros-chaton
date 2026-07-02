"""HumanEval task loader — the standard code-generation benchmark.

HumanEval (Chen et al., 2021) = 164 hand-written Python function-completion
tasks. Each has:
  - task_id:      "HumanEval/0".."HumanEval/163"
  - prompt:       signature + docstring (the model COMPLETES the body)
  - canonical_solution: reference (we don't show this to the model)
  - test:         a `def check(candidate):` body of assert statements
  - entry_point:  the function name the solution must define

We convert each into a verify.Problem so the existing verifier runs them.
HumanEval's tests are wrapped in `def check(candidate): ...`, so we call
`check(entry_point)` after binding candidate -> entry_point. This matches how
the original eval harness works.

Dataset: HuggingFace `openai_humaneval` (164 rows, ~1MB). Gated (accept terms
once on the HF site) but free. We cache to a local JSON to avoid re-downloading
and to run fully offline after the first fetch.
"""
import json
import os

# verify/ is the current package; project root is one level up.
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ_ROOT, "eval_data", "humaneval.json")

# We import Problem lazily inside load() to avoid a circular import if verifier
# ever imports from this module (it doesn't today, but defensive).


def _fetch_humaneval():
    """Download HumanEval from HF once, return a list of dicts."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    out = []
    for row in ds:
        out.append({
            "id": row["task_id"].lower(),            # "humaneval/0"
            "prompt": row["prompt"],
            "test": row["test"],                     # `def check(candidate): ...`
            "entry_point": row["entry_point"],
            "canonical_solution": row.get("canonical_solution", ""),
        })
    return out


def load(limit=None, cache=True):
    """Return a list of verify.Problem for HumanEval.

    limit: only return the first N tasks (useful for quick evals).
    cache: if True, fetch once to JSON then read from disk thereafter.
    """
    from verify.verifier import Problem

    raw = None
    if cache and os.path.exists(CACHE):
        with open(CACHE) as f:
            raw = json.load(f)
    if raw is None:
        print("[humaneval] fetching from HuggingFace (one-time)...")
        raw = _fetch_humaneval()
        if cache:
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            with open(CACHE, "w") as f:
                json.dump(raw, f)
            print(f"[humaneval] cached {len(raw)} tasks -> {CACHE}")

    tasks = raw[:limit] if limit else raw
    problems = []
    for t in tasks:
        # HumanEval's test field is a `def check(candidate):` block. We keep it
        # whole — the verifier appends `candidate = <entry_point>` then this block,
        # so calling check() works because `candidate` is bound by then. But the
        # block defines check() rather than calling it, so we append an explicit
        # call at the end via the entry_point convention.
        tests = t["test"]
        if "def check(candidate):" in tests:
            tests = tests + f"\ncheck({t['entry_point']})\n"
        problems.append(Problem(
            id=t["id"],
            prompt=t["prompt"],
            tests=tests,
            entry_point=t["entry_point"],
        ))
    return problems


if __name__ == "__main__":
    # quick smoke: load a few tasks, print one, confirm shape
    tasks = load(limit=5)
    print(f"loaded {len(tasks)} tasks")
    p = tasks[0]
    print(f"\n--- {p.id} ---")
    print("PROMPT:")
    print(p.prompt)
    print("TESTS (first 8 lines):")
    print("\n".join(p.tests.splitlines()[:8]))
