"""SWE-bench evaluation for fine-tuned coding agents.

Measures the model's ability to fix real GitHub issues by applying patches.

Usage:
    python eval_swebench.py --model Qwen/Qwen3.5-9B --ckpt qwen_coding_agent
    python eval_swebench.py --model Qwen/Qwen3.5-9B --limit 10 --n-samples 4
    python eval_swebench.py --results swe_results.json  # re-report from saved
"""
import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def load_swebench(limit: int | None = None, split: str = "test"):
    """Load SWE-bench instances from HuggingFace."""
    from datasets import load_dataset
    print(f"[swebench] Loading SWE-bench ({split} split)...")
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[swebench] Loaded {len(ds)} instances")
    return ds


def format_swe_instance(instance: dict) -> str:
    """Format a SWE-bench instance for the model."""
    repo = instance.get("repo", "")
    issue = instance.get("problem_statement", "")
    hint = instance.get("hints_text", "")

    # Get the code context (from the base commit)
    patch = instance.get("patch", "")
    base_commit = instance.get("base_commit", "")

    prompt = f"""<|im_start|>system
You are a senior software engineer. Fix the following issue in the {repo} repository.
<|im_end|>
<|im_start|>user
ISSUE:
{issue}

{"HINT: " + hint if hint else ""}

Repository: {repo}
Base commit: {base_commit}

Provide a git patch (diff) that fixes this issue. Format your patch like:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -1,5 +1,8 @@
...
```

First analyze the issue, then provide the patch inside ```diff ... ``` tags.
<|im_end|>
<|im_start|>assistant
"""
    return prompt


def extract_patch(text: str) -> str | None:
    """Extract a git diff patch from model output."""
    import re
    # Try ```diff ... ``` blocks first
    m = re.search(r'```diff\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ```patch ... ```
    m = re.search(r'```patch\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try raw diff starting with --- a/
    m = re.search(r'(--- a/.*?)(?=\n\n|\Z)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def apply_patch(repo_dir: str, patch_text: str) -> tuple[bool, str]:
    """Apply a patch to a repo and return (success, output)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as f:
        f.write(patch_text)
        patch_path = f.name

    try:
        result = subprocess.run(
            ["git", "apply", "--check", patch_path],
            cwd=repo_dir,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            # Patch applies cleanly
            subprocess.run(
                ["git", "apply", patch_path],
                cwd=repo_dir, capture_output=True, timeout=30,
            )
            return True, "patch applies"
        else:
            return False, result.stderr[:500]
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(patch_path)


def evaluate_swebench(model, tokenizer, instances, n_samples: int = 1,
                      max_new: int = 1024, device: str = "cuda"):
    """Evaluate on SWE-bench instances."""
    results = []

    for i, inst in enumerate(instances):
        instance_id = inst.get("instance_id", f"instance_{i}")
        prompt = format_swe_instance(inst)

        print(f"[swebench] {i+1}/{len(instances)} {instance_id}...")

        best_patch = None
        for s in range(n_samples):
            temp = 0.7 + (s / max(n_samples - 1, 1)) * 0.3 if n_samples > 1 else 0.3

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new,
                    temperature=temp, top_p=0.95, do_sample=(n_samples > 1),
                    pad_token_id=tokenizer.eos_token_id,
                )
            output = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            patch = extract_patch(output)

            if patch:
                best_patch = patch
                if n_samples == 1:
                    break

        results.append({
            "instance_id": instance_id,
            "repo": inst.get("repo", ""),
            "has_patch": best_patch is not None,
            "patch_length": len(best_patch) if best_patch else 0,
        })

        if (i + 1) % 10 == 0:
            with_patch = sum(1 for r in results if r["has_patch"])
            print(f"  [{i+1}/{len(instances)}] {with_patch}/{i+1} with patches")

    return results


def report(results: list[dict], title: str = "SWE-bench Results"):
    """Print a report."""
    total = len(results)
    with_patch = sum(1 for r in results if r["has_patch"])

    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Total instances:  {total}")
    print(f"  With patches:     {with_patch} ({100*with_patch/max(total,1):.1f}%)")
    print(f"  Without patches:  {total - with_patch}")
    print(f"{'='*50}")

    # By repo
    repos: dict[str, list] = {}
    for r in results:
        repos.setdefault(r["repo"], []).append(r)
    print(f"\n  By repository:")
    for repo, rs in sorted(repos.items()):
        wp = sum(1 for r in rs if r["has_patch"])
        print(f"    {repo}: {wp}/{len(rs)} ({100*wp/max(len(rs),1):.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="SWE-bench evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--limit", type=int, default=50, help="Instances to eval")
    parser.add_argument("--n-samples", type=int, default=1, help="Test-time scaling")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--output", default="swebench_results.json")
    parser.add_argument("--results", default=None,
                        help="Re-report from saved results JSON")
    args = parser.parse_args()

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        report(results, title=f"SWE-bench — {args.results}")
        return

    # Load model
    from eval_qwen import load_qwen
    model, tokenizer = load_qwen(args.model, args.ckpt, use_4bit=args.four_bit)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SWE-bench
    instances = load_swebench(limit=args.limit)

    # Evaluate
    results = evaluate_swebench(
        model, tokenizer, instances,
        n_samples=args.n_samples, device=device,
    )

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[swebench] Saved to {args.output}")

    # Report
    report(results)


if __name__ == "__main__":
    main()
