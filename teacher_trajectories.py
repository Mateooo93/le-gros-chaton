"""Teacher-driven trajectory generation — Kimi K3 (via TokenRouter) as the
teacher model, running OUR agent loop on OUR bug repos.

This is the OmniCoder-9B recipe applied to Le Gros Chaton: a frontier-class
model generates agentic trajectories in OUR harness format (tool calls +
results + finish), we grade each with the hidden-test verifier, and keep only
VERIFIED + NOVEL traces. The trajectory SFT then bakes that behavior into the
9B weights — which is how the 9B learns the tool-use format it couldn't
self-distill (the 91% SFT checkpoint reads files forever but never writes).

The output format matches gen_trajectories.py exactly:
  agent_traces_full.jsonl  (one JSON object per verified run)

Usage:
    python teacher_trajectories.py --n 20 --samples 3
    python teacher_trajectories.py --n 50 --samples 2 --temp 0.9 --no-4bit
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# NOTE: no torch import — the teacher runs via API (CPU-only, free).

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from gen_trajectories import _buggy_versions, make_repo, verify_repo, _patch_overlap

# --- Teacher endpoint (env-overridable; TOKENROUTER_* kept as fallback) ---
API_URL = os.environ.get(
    "TEACHER_API_URL",
    os.environ.get("TOKENROUTER_API_URL",
                   "https://secondary3--ep-kimi-k3-server.us-west.modal.direct/v1/chat/completions"))
API_KEY = os.environ.get(
    "TEACHER_API_KEY",
    os.environ.get("TOKENROUTER_API_KEY",
                   "wk-0bRRK2Jamd4Q98gyEQBIHa.ws-E0nQW84LH5JmYmC7vdxVpH"))
MODEL = os.environ.get("TEACHER_MODEL", "moonshotai/Kimi-K3")

TOOLS_DESC = """\
Available tools (call EXACTLY one per turn, in this format):
```list_dir
<dirpath>
```
```read_file
<filepath>
```
```search_code
<pattern>
```
```write_file
<filepath>
<new content>
```
```run_test
<test command>
```
```finish
<explanation of your fix>
```
"""

SYSTEM_TEACHER = f"""\
You are an expert software engineer fixing a bug. You are being observed to
produce training data for a small model, so demonstrate excellent practice:
explore, diagnose, fix, verify.

{TOOLS_DESC}

Rules:
- You get a tool RESULT after every call. Use it.
- Do not repeat a tool call that already returned the same result.
- When you believe the bug is fixed, call ```run_test with the test command.
- Only call ```finish after you are confident the fix works.
- When you call ```finish, your explanation is the fix summary. Then, in the
  SAME message directly after the closing ``` of the finish block, write a
  brief SELF-REVIEW (under 100 words) on its own line starting with
  "SELF-REVIEW:" covering: what you did, what you learned, what you would do
  differently next time, and how confident you are the fix is correct.
"""


def teacher_complete(messages, max_tokens=1200, temperature=0.7,
                     retries=10):
    """Call Kimi K3 via TokenRouter (OpenAI-compatible). Retries on timeout/SSL
    with exponential backoff — the free tier is flaky and recovers."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API_URL, data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {API_KEY}",
                         "X-Webhook-Token": API_KEY,
                         "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            # Kimi K3 is a reasoning model — content may be None with reasoning_content.
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            usage = data.get("usage", {})
            return content, reasoning, usage
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                # Permanent client error — 30 min of backoff would be wasted.
                raise RuntimeError(
                    f"teacher_complete permanent HTTP {e.code}: {e.reason}")
            last_err = e
            wait = min(20, 5 * (2 ** attempt))  # 5,10,20... capped at 20s
            print(f"    [teacher] attempt {attempt+1}/{retries} failed: {e} "
                  f"— retrying in {wait}s...", flush=True)
            time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = min(20, 5 * (2 ** attempt))  # 5,10,20... capped at 20s
            print(f"    [teacher] attempt {attempt+1}/{retries} failed: {e} "
                  f"— retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"teacher_complete failed after {retries} attempts: {last_err}")


def parse_tool_calls(text: str):
    """Parse ```tool\nargs``` blocks (also [tool\nargs] fallback)."""
    actions = []
    for m in re.finditer(r'```(\w+)\s*\n(.*?)```', text, re.DOTALL):
        actions.append((m.group(1), m.group(2).strip()))
    if not actions:
        for m in re.finditer(r'\[(\w+)\s*\n(.*?)\]', text, re.DOTALL):
            actions.append((m.group(1), m.group(2).strip()))
    return actions


def parse_open_tag_call(text: str):
    """Parse Fable5-dialect <|open|>call tool="name" ... <|close|>call blocks.

    The teacher occasionally drifts into this XML-ish envelope (same tool
    set). Each call yields (tool_name, args_text) in the exact ```tool\\nargs```
    shape so the executor stays unchanged. Multi-arg calls (write_file) are
    joined the same way: path\\ncontent.
    """
    def clean_arg(seg: str) -> str:
        """Drop <|sep|>-separated attribute noise (index=, key=, type=, tool=)
        from an argument segment, keeping the payload."""
        for part in seg.split("<|sep|>"):
            part = part.strip()
            if not part:
                continue
            if re.fullmatch(r'(?:key|type|index|tool)="[^"]*"', part):
                continue
            return part
        return ""

    actions = []
    for m in re.finditer(r'<\|open\|>call tool="([a-z_]+)"([\s\S]*?)<\|close\|>call', text):
        name = m.group(1)
        body = m.group(2)
        args = []
        pieces = re.split(r'<\|open\|>argument[^<]*<\|sep\|>', body)
        args.append(clean_arg(pieces[0].split("<|close|>argument", 1)[0]))
        for p in pieces[1:]:
            args.append(clean_arg(p.split("<|close|>argument", 1)[0]))
        args = [a for a in args if a.strip()]
        actions.append((name, "\n".join(args)))
    return actions


def _ask_self_review(messages, temperature, retries):
    """One follow-up teacher call asking for the SELF-REVIEW, so every
    verified trace ends with trainable self-review tokens. Returns the review
    text ('' on failure)."""
    followup = [{"role": "user",
                 "content": "You are done with the task. Now write your "
                            "brief SELF-REVIEW (under 100 words), plain text "
                            "only (no tool calls, no code fences), on its own "
                            "line starting with \"SELF-REVIEW:\": what you "
                            "did, what you learned, what you would do "
                            "differently next time, and how confident you "
                            "are the fix is correct."}]
    for _ in range(2):
        try:
            # Kimi is a reasoning model: a tight budget burns out on thinking
            # and returns empty content. Leave room for reasoning + answer.
            reply, _, _ = teacher_complete(
                messages + followup, max_tokens=800,
                temperature=temperature, retries=retries)
            # The teacher may still drift into tool-call envelopes — strip
            # them so the review text survives on its own.
            clean = re.sub(r"```\w*\s*[\s\S]*?```", "", reply)
            clean = re.sub(r"<\|open\|>[\s\S]*", "", clean)
            m = re.search(r"SELF-REVIEW:\s*(.*)", clean, re.DOTALL)
            if m:
                return m.group(1).strip()[:500]
            if clean.strip():
                # No marker, but the reply is the review itself.
                return clean.strip()[:300]
            # Empty content: reasoning-only response — try once more.
        except Exception:
            pass
    return ""


def execute_tool(action, args_text, repo_dir):
    """Run a tool call against the repo. Mirrors SWEAgent._execute_tool."""
    args_text = args_text.strip()
    try:
        if action == "list_dir":
            path = os.path.join(repo_dir, args_text or ".")
            if not os.path.isdir(path):
                return f"Error: directory not found: {args_text}"
            dirs = sorted(f for f in os.listdir(path)
                          if os.path.isdir(os.path.join(path, f)))
            files_list = sorted(f for f in os.listdir(path)
                                if not os.path.isdir(os.path.join(path, f)))
            return f"Directory {args_text}:\n" + "\n".join(
                [f"  {d}/" for d in dirs] + [f"  {f}" for f in files_list])[:2000]
        elif action == "read_file":
            path = args_text
            full = path if os.path.isabs(path) else os.path.join(repo_dir, path)
            if not os.path.isfile(full):
                return f"Error: file not found: {path}"
            with open(full) as f:
                return f.read()[:3000]
        elif action == "search_code":
            pattern = args_text
            hits = []
            for root, _, files in os.walk(repo_dir):
                if ".git" in root:
                    continue
                for fn in files:
                    fp = os.path.join(root, fn)
                    try:
                        with open(fp) as f:
                            for ln, line in enumerate(f, 1):
                                if pattern in line:
                                    rel = os.path.relpath(fp, repo_dir)
                                    hits.append(f"{rel}:{ln}: {line.rstrip()[:100]}")
                    except Exception:
                        continue
            return "\n".join(hits[:20]) or f"No matches for '{pattern}'"
        elif action == "write_file":
            # args: <filepath>\n<content>
            parts = args_text.split("\n", 1)
            path = parts[0].strip()
            content = parts[1] if len(parts) > 1 else ""
            full = path if os.path.isabs(path) else os.path.join(repo_dir, path)
            os.makedirs(os.path.dirname(full), exist_ok=True) if "/" in full else None
            with open(full, "w") as f:
                f.write(content)
            return f"Wrote {path} ({len(content)} chars)"
        elif action == "run_test":
            r = subprocess.run(args_text, shell=True, cwd=repo_dir,
                               capture_output=True, text=True, timeout=60)
            return (r.stdout + r.stderr)[:2000] or "No output"
        elif action == "finish":
            return "FINISHED"
        else:
            return f"Unknown tool: {action}"
    except Exception as e:
        return f"Error: {e}"


def teacher_run(issue, repo_dir, max_turns=15, temperature=0.7, retries=30):
    """Run Kimi K3 through one agent loop. Returns the full trace."""
    user_prompt = f"Fix this issue in the repo at {repo_dir}:\n\n{issue}"
    messages = [
        {"role": "system", "content": SYSTEM_TEACHER},
        {"role": "user", "content": user_prompt},
    ]
    # Traces start with the REAL prompt the model saw (ground truth for SFT;
    # trainers used to reconstruct this message from the issue field).
    trace = [{"role": "user", "content": user_prompt}]
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    for turn in range(max_turns):
        content, reasoning, usage = teacher_complete(
            messages, max_tokens=1200, temperature=temperature, retries=retries)
        usage_total["prompt_tokens"] += usage.get("prompt_tokens") or 0
        usage_total["completion_tokens"] += usage.get("completion_tokens") or 0
        if reasoning:
            trace.append({"role": "assistant", "content": f"[thinking]\n{reasoning[:500]}"})
        if content:
            trace.append({"role": "assistant", "content": content})
        print(f"  [teacher] turn {turn+1}: {content[:120]!r}")

        actions = parse_tool_calls(content)
        if not actions and "<|open|>" in content:
            # Teacher drifted into the Fable5 dialect — parse it so the calls
            # EXECUTE and get results (never train on unexecuted calls).
            actions = parse_open_tag_call(content)
        if not actions:
            # Reasoning model often talks without calling — nudge once
            messages.append({"role": "assistant", "content": content or "[reasoning only]"})
            messages.append({"role": "user", "content":
                "Make a tool call now using the ```tool_name\\nargs``` format."})
            continue

        messages.append({"role": "assistant", "content": content})
        for action, args_text in actions:
            if action == "finish":
                trace.append({"role": "user", "content": f"[finish] {args_text}"})
                # Guarantee trainable self-review tokens: if the finish message
                # carries no explicit SELF-REVIEW, ask once and append it as a
                # final assistant message.
                if not re.search(r"SELF-REVIEW:\s*", content):
                    sr = _ask_self_review(messages, temperature, retries)
                    if sr:
                        trace.append({"role": "assistant",
                                      "content": f"SELF-REVIEW: {sr}"})
                print(f"  [teacher] run usage: {usage_total}", flush=True)
                # Return the FULL message, not just the fenced args, so a
                # SELF-REVIEW written after the closing fence is captured.
                return trace, content
            result = execute_tool(action, args_text, repo_dir)
            trace.append({"role": "user",
                          "content": f"Tool [{action}]({args_text}):\n{result[:2000]}"})
            messages.append({"role": "user",
                             "content": f"Tool [{action}]({args_text}):\n{result[:2000]}"})
    print(f"  [teacher] run usage: {usage_total}", flush=True)
    return trace, "(reached max turns)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Tasks to attempt")
    parser.add_argument("--samples", type=int, default=2,
                        help="Teacher runs per task (diversity)")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--novelty-thresh", type=float, default=0.5)
    parser.add_argument("--out", default="agent_traces_full.jsonl")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--retries", type=int, default=30,
                        help="API retries per call (free tier is flaky)")
    args = parser.parse_args()

    templates = _buggy_versions()
    out_path = os.path.join(PROJ_ROOT, args.out)

    # --- Resume support: load already-kept traces so a restart skips done work ---
    done_ids = set()
    existing = []
    # Verified traces grouped by template id, so the novelty filter still
    # compares against traces kept in EARLIER runs after a restart.
    kept_by_task = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                existing.append(e)
                done_ids.add(e["instance_id"])
                if e.get("verified"):
                    tpl_id = e["instance_id"].rsplit("_", 2)[0]
                    kept_by_task.setdefault(tpl_id, []).append(e)
    print(f"[teacher] resume: {len(existing)} traces already in {args.out}")

    def append_entry(entry):
        # Persist ONLY verified traces incrementally: the file is the verified
        # dataset at every point (wc -l == verified count, resume only skips
        # completed work, and a crash mid-run never leaves unverified noise).
        if not entry.get("verified"):
            return
        with open(out_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    results = list(existing)
    n_success = sum(1 for r in results if r.get("verified"))
    work = tempfile.mkdtemp(prefix="teacher_repos_")

    for i in range(args.n):
        tpl = templates[i % len(templates)]
        repo_dir = os.path.join(work, f"task_{i}")
        make_repo(tpl, repo_dir)
        passed, _, _ = verify_repo(repo_dir, tpl["test"])
        assert not passed, f"bug not actually injected for {tpl['id']}"

        real_files = ", ".join(sorted(tpl["files"].keys()))
        issue = (f"{tpl['issue']}\n\nHint: the bug is in {tpl['bug']}. "
                 f"Repo files: {real_files}.")

        # Novelty comparison pool: traces kept for THIS TASK (same index i),
        # including any kept before a crash+restart (loaded from the resume
        # file). Only within-task dedup: our templates have 1-line canonical
        # fixes, so cross-task patch overlap is ~1.0 for every correct
        # solution — comparing across tasks would keep one trace per template
        # and starve the dataset. Trajectory diversity comes from each task
        # being an independent run (fresh sampling), not from the patch.
        kept_for_task = [e for e in kept_by_task.get(tpl["id"], [])
                         if e["instance_id"].rsplit("_", 2)[1] == str(i)]
        for s in range(args.samples):
            inst_id = f"{tpl['id']}_{i}_{s}"
            if inst_id in done_ids:
                print(f"[teacher] {i+1}/{args.n} s{s+1} | {tpl['id']} | already done — skip")
                continue

            try:
                t0 = time.time()
                trace, finish_msg = teacher_run(issue, repo_dir, args.max_turns,
                                                args.temp, args.retries)
                dt = time.time() - t0
            except Exception as e:
                print(f"[teacher] {i+1}/{args.n} s{s+1} | {tpl['id']} | "
                      f"RUN FAILED ({e}) — skipping, continuing", flush=True)
                continue

            verified, n_pass, n_total = verify_repo(repo_dir, tpl["test"])
            patch = ""
            try:
                r = subprocess.run(["git", "diff"], cwd=repo_dir,
                                   capture_output=True, text=True)
                patch = r.stdout
            except Exception:
                pass

            # Extract the teacher's self-review. Priority: (1) explicit marker
            # in the full finish message (may sit after the closing fence),
            # (2) the dedicated SELF-REVIEW assistant message teacher_run
            # appended, (3) the fence-stripped finish explanation.
            self_review = ""
            if isinstance(finish_msg, str):
                sr = re.search(r"SELF-REVIEW:\s*(.*)", finish_msg, re.DOTALL)
                if sr:
                    self_review = sr.group(1).strip()[:500]
                else:
                    for m in reversed(trace):
                        if m["role"] == "assistant" and m["content"].startswith("SELF-REVIEW:"):
                            self_review = m["content"].split("SELF-REVIEW:", 1)[1].strip()[:500]
                            break
                    if not self_review and finish_msg != "(reached max turns)":
                        # Kimi often skips the marker: fall back to the finish
                        # explanation with the ```tool fences stripped.
                        text = re.sub(r"```\w*\s*", "", finish_msg).strip()
                        self_review = text[:300]

            if verified and kept_for_task:
                overlap = max(_patch_overlap(patch, kp["patch"])
                              for kp in kept_for_task)
                if overlap > args.novelty_thresh:
                    print(f"[teacher] {i+1}/{args.n} s{s+1} | {tpl['id']} | "
                          f"verified but redundant (overlap {overlap:.2f}) — skip")
                    # Consume the slot: it was verified but not novel, and is
                    # intentionally not persisted — don't re-run it on restart.
                    done_ids.add(inst_id)
                    continue

            entry = {
                "instance_id": inst_id,
                "issue": issue,
                "messages": trace,
                "turns": len([m for m in trace if m["role"] == "assistant"]),
                "patch": patch,
                "verified": verified,
                "n_pass": n_pass, "n_total": n_total,
                "tool_calls": len([m for m in trace
                                   if m["role"] == "user"
                                   and m["content"].startswith("Tool [")]),
                "seconds": round(dt, 1),
                "self_review": self_review,
                "teacher": MODEL,
            }
            results.append(entry)
            append_entry(entry)  # incremental: verified traces survive crashes
            done_ids.add(inst_id)
            if verified:
                n_success += 1
                kept_for_task.append(entry)
            print(f"[teacher] {i+1}/{args.n} s{s+1} | {tpl['id']} | "
                  f"verified={verified} ({n_pass}/{n_total}) | "
                  f"turns={entry['turns']} | {dt:.0f}s | "
                  f"self_review={'yes' if self_review else 'no'}", flush=True)
        shutil.rmtree(repo_dir, ignore_errors=True)

    kept = results if args.keep_failed else [r for r in results if r["verified"]]
    # Re-write canonical (only verified unless --keep-failed)
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\n[teacher] Done: {len(results)} runs, {n_success} verified, "
          f"{len(kept)} kept -> {out_path}")
    print(f"[teacher] Verified rate: {n_success/max(1,len(results))*100:.0f}%")
    print(f"[teacher] With self-review: {sum(1 for r in kept if r.get('self_review'))}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
