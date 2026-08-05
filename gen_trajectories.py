"""Agentic trajectory generator — builds training data for the coding agent.

Research-backed (see research/agentic_9b_long_horizon.md):
- Small agents learn agency from REAL tool-use trajectories, not chat data.
- Self-play bug injection (SSR / SWE-smith style): create buggy repo -> agent
  explores + fixes -> verifier grades -> keep successful traces.

Generates `agent_traces_full.jsonl`, one JSON object per successful run:
  {
    "instance_id": str,
    "issue": str,                 # problem statement the agent saw
    "messages": [ {"role","content"}... ],   # FULL trajectory (system+user+assistant)
    "turns": int,
    "patch": str,
    "verified": bool,
    "n_pass": int, "n_total": int,
    "tool_calls": int
  }

Usage:
    python gen_trajectories.py --n 50 --out agent_traces_full.jsonl
    python gen_trajectories.py --n 20 --use-4bit --ckpt qwen_coding_agent
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from agent_swe import SWEAgent, TOOLS  # noqa: E402
from eval_qwen import load_qwen  # noqa: E402


# ---- Seed programs with an injected bug + hidden test that must pass ----
# Each: {"id","files": {path: source}, "bug": how to corrupt, "test": hidden test code}
# The bug is applied to produce the repo; the test is what verifies the fix.

def _buggy_versions():
    """Return a list of (files, buggy_file, good_line, buggy_line) templates."""
    return [
        # ---- Binary search off-by-one ----
        {
            "id": "binary_search",
            "files": {
                "search.py": '''def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
                "main.py": "from search import binary_search\n",
            },
            "bug": "search.py",
            "good": "            lo = mid + 1",
            "buggy": "            lo = mid",
            "test": '''from search import binary_search
assert binary_search([1,3,5,7,9], 9) == 4
assert binary_search([1,3,5,7,9], 1) == 0
assert binary_search([1,3,5,7,9], 6) == -1
assert binary_search(list(range(100)), 57) == 57
print("OK")
''',
            "issue": "binary_search returns wrong index for some inputs (hangs or wrong position). Investigate and fix.",
        },
        # ---- Off-by-one in loop bound ----
        {
            "id": "sum_evens",
            "files": {
                "stats.py": '''def sum_evens(nums):
    total = 0
    for i in range(len(nums)):
        if nums[i] % 2 == 0:
            total += nums[i]
    return total
''',
                "main.py": "from stats import sum_evens\n",
            },
            "bug": "stats.py",
            "good": "    for i in range(len(nums)):",
            "buggy": "    for i in range(len(nums) - 1):",
            "test": '''from stats import sum_evens
assert sum_evens([1,2,3,4]) == 6
assert sum_evens([2,4,6,8]) == 20
assert sum_evens([1,3,5]) == 0
assert sum_evens([]) == 0
print("OK")
''',
            "issue": "sum_evens misses the last element for some inputs. Fix it.",
        },
        # ---- Wrong comparison ----
        {
            "id": "is_palindrome",
            "files": {
                "text.py": '''def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]
''',
                "main.py": "from text import is_palindrome\n",
            },
            "bug": "text.py",
            "good": "    return s == s[::-1]",
            "buggy": "    return s != s[::-1]",
            "test": '''from text import is_palindrome
assert is_palindrome("racecar") is True
assert is_palindrome("A man a plan a canal Panama") is True
assert is_palindrome("hello") is False
assert is_palindrome("") is True
print("OK")
''',
            "issue": "is_palindrome returns the wrong result. Fix the logic.",
        },
        # ---- Swap in-place bug ----
        {
            "id": "reverse_list",
            "files": {
                "lists.py": '''def reverse_inplace(lst):
    for i in range(len(lst) // 2):
        tmp = lst[i]
        lst[i] = lst[len(lst) - 1 - i]
        lst[len(lst) - 1 - i] = tmp
    return lst
''',
                "main.py": "from lists import reverse_inplace\n",
            },
            "bug": "lists.py",
            "good": "        lst[len(lst) - 1 - i] = tmp",
            "buggy": "        lst[i] = tmp",
            "test": '''from lists import reverse_inplace
a = [1,2,3]
assert reverse_inplace(a) == [3,2,1]
b = [1,2,3,4]
assert reverse_inplace(b) == [4,3,2,1]
c = [7]
assert reverse_inplace(c) == [7]
print("OK")
''',
            "issue": "reverse_inplace corrupts the list. Find and fix the bug.",
        },
        # ---- Missing edge case: empty input ----
        {
            "id": "max_subarray",
            "files": {
                "dp.py": '''def max_subarray(nums):
    if not nums:
        return 0
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
''',
                "main.py": "from dp import max_subarray\n",
            },
            "bug": "dp.py",
            "good": "        cur = max(x, cur + x)",
            "buggy": "        cur = max(x, cur) + x",
            "test": '''from dp import max_subarray
assert max_subarray([1,2,3]) == 6
assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6
assert max_subarray([5]) == 5
assert max_subarray([-1,-2]) == -1
assert max_subarray([]) == 0
print("OK")
''',
            "issue": "max_subarray returns wrong max for some inputs. Fix it.",
        },
    ]


def make_repo(template, out_dir: str):
    """Write the buggy repo files + git init."""
    os.makedirs(out_dir, exist_ok=True)
    for path, content in template["files"].items():
        full = os.path.join(out_dir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    # apply the bug
    bug_file = os.path.join(out_dir, template["bug"])
    with open(bug_file) as f:
        src = f.read()
    src = src.replace(template["good"], template["buggy"])
    with open(bug_file, "w") as f:
        f.write(src)
    subprocess.run(["git", "init", "-q"], cwd=out_dir)
    subprocess.run(["git", "add", "-A"], cwd=out_dir)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=out_dir)
    return out_dir


def verify_repo(repo_dir: str, test_code: str) -> tuple[bool, int, int]:
    """Run the hidden test against the repo. Returns (passed, n_pass, n_total)."""
    import tempfile as _tf
    test_file = os.path.join(repo_dir, "_test_check.py")
    with open(test_file, "w") as f:
        f.write(test_code)
    try:
        # -B: don't write __pycache__ .pyc files. Without it, importing the
        # buggy version caches bytecode; if we then fix the file within the
        # same mtime-second, the subprocess re-imports the STALE buggy .pyc
        # and verify_repo wrongly reports the fixed code as broken.
        r = subprocess.run([sys.executable, "-B", "_test_check.py"], cwd=repo_dir,
                           capture_output=True, text=True, timeout=30)
        passed = r.returncode == 0 and "OK" in r.stdout
        n_asserts = test_code.count("assert ")
        return passed, (n_asserts if passed else 0), n_asserts
    except subprocess.TimeoutExpired:
        return False, 0, test_code.count("assert ")
    finally:
        try:
            os.remove(test_file)
        except OSError:
            pass
        # Remove any __pycache__ left over (paranoia — -B should prevent it)
        try:
            pyc_dir = os.path.join(repo_dir, "__pycache__")
            if os.path.isdir(pyc_dir):
                for f in os.listdir(pyc_dir):
                    os.remove(os.path.join(pyc_dir, f))
                os.rmdir(pyc_dir)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20,
                        help="Number of trajectories to generate")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None, help="Optional LoRA adapter")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--out", default="agent_traces_full.jsonl")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Also save failed runs (for negative mining)")
    parser.add_argument("--only-verified", action="store_true",
                        help="Only keep runs that pass the hidden test")
    parser.add_argument("--samples", type=int, default=1,
                        help="Generate N trajectories per task (diversity sampling)")
    parser.add_argument("--temp", type=float, default=0.9,
                        help="Sampling temperature for diverse solutions")
    parser.add_argument("--novelty-thresh", type=float, default=0.5,
                        help="Keep a solution if its max n-gram overlap with already-kept \
                              solutions for the same task is below this (0-1)")
    args = parser.parse_args()

    print("[gen] Loading model...")
    model, tokenizer = load_qwen(args.model, args.ckpt, use_4bit=args.use_4bit)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    templates = _buggy_versions()
    results = []
    n_success = 0
    work = tempfile.mkdtemp(prefix="chaton_repos_")

    for i in range(args.n):
        tpl = templates[i % len(templates)]
        repo_dir = os.path.join(work, f"task_{i}")
        make_repo(tpl, repo_dir)
        # Sanity: the buggy repo must FAIL the hidden test
        passed, np_, nt_ = verify_repo(repo_dir, tpl["test"])
        assert not passed, f"bug not actually injected for {tpl['id']}"

        # --- Diversity sampling: run the agent --samples times at high temp,
        # keep solutions that are verified AND novel vs already-kept ones. ---
        kept_for_task = []
        for s in range(args.samples):
            agent = SWEAgent(model, tokenizer, repo_dir, device=device, tdd=False,
                             temperature=args.temp)
            t0 = time.time()
            result = agent.run(tpl["issue"], instance_id=f"{tpl['id']}_{i}_{s}")
            dt = time.time() - t0

            verified, n_pass, n_total = verify_repo(repo_dir, tpl["test"])

            # Novelty filter: skip if the patch is too similar to one we already
            # kept for this task (n-gram overlap on the diff).
            patch = result.get("patch", "")
            if verified and kept_for_task:
                overlap = max(_patch_overlap(patch, kp["patch"])
                              for kp in kept_for_task)
                if overlap > args.novelty_thresh:
                    print(f"[gen]   {s+1}/{args.samples} | {tpl['id']} | "
                          f"verified but redundant (overlap {overlap:.2f}) — skip")
                    continue

            # Self-review: the model reflects on what it did and learned. This gets
            # baked into the weights via trajectory SFT — the model learns to
            # self-assess without any prompt asking it to.
            self_review = ""
            try:
                from agent_swe import _self_review
                self_review = _self_review(model, tokenizer, device, tpl["issue"], result)
            except Exception as e:
                print(f"[gen] self-review skipped: {e}")

            entry = {
                "instance_id": result["instance_id"],
                "issue": tpl["issue"],
                "messages": [{"role": m["role"], "content": m["content"]}
                             for m in result.get("trace", [])],
                "turns": result["turns"],
                "patch": patch,
                "verified": verified,
                "n_pass": n_pass,
                "n_total": n_total,
                "tool_calls": len([m for m in result.get("trace", [])
                                   if m["role"] == "assistant"]),
                "seconds": round(dt, 1),
                "self_review": self_review,
                "sample": s,
            }
            results.append(entry)
            if verified:
                n_success += 1
                kept_for_task.append(entry)
            print(f"[gen] {i+1}/{args.n} s{s+1}/{args.samples} | {tpl['id']} | "
                  f"verified={verified} ({n_pass}/{n_total}) | "
                  f"turns={entry['turns']} | {dt:.0f}s")

        # Cleanup AFTER all samples for this task (sample s=0 deleting the
        # repo while s>=1 still needs it caused FileNotFoundError).
        shutil.rmtree(repo_dir, ignore_errors=True)

    # Filter: keep verified runs (or all if --keep-failed)
    kept = results
    if args.only_verified:
        kept = [r for r in results if r["verified"]]
    if not args.keep_failed:
        kept = [r for r in results if r["verified"]]

    out_path = os.path.join(PROJ_ROOT, args.out)
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\n[gen] Done: {len(results)} runs, {n_success} verified, "
          f"{len(kept)} kept -> {out_path}")
    print(f"[gen] Verified rate: {n_success/max(1,len(results))*100:.0f}%")

    shutil.rmtree(work, ignore_errors=True)


def _patch_overlap(patch_a: str, patch_b: str) -> float:
    """N-gram overlap (Jaccard) between two patches, 0 (disjoint) to 1 (same).

    Used for novelty filtering in diversity sampling. Patches are compared on
    their added-line n-grams (size 2) so small formatting differences don't
    hide genuinely different strategies, and vice versa.
    """
    import re

    def lines(p):
        added = [l[1:].strip() for l in p.splitlines() if l.startswith("+")
                 and not l.startswith("+++")]
        return [w for l in added for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", l)]

    def ngrams(toks, n=2):
        return set(zip(*[toks[i:] for i in range(n)])) if len(toks) >= n else set(toks)

    a, b = ngrams(lines(patch_a)), ngrams(lines(patch_b))
    if not a and not b:
        return 1.0  # both empty => treat as identical (avoid keeping dupes)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


if __name__ == "__main__":
    main()
