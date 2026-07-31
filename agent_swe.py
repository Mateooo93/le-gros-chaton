"""SWE-bench agent — navigates repos, edits files, creates patches.

Proper agentic coding loop for real software engineering tasks.
Not just "run a command" — explores codebases, understands issues,
makes targeted edits, and generates git diffs.

Usage:
    python agent_swe.py --model Qwen/Qwen3.5-9B --repo /path/to/repo --issue "fix the bug"
    python agent_swe.py --model Qwen/Qwen3.5-9B --instance swebench_instance.json
    python agent_swe.py --model Qwen/Qwen3.5-9B --ckpt qwen_coding_agent
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Tools the agent can use
TOOLS = {
    "read_file": {"desc": "Read a file from the repo", "args": "<filepath>"},
    "write_file": {"desc": "Write content to a file", "args": "<filepath>\\n<content>"},
    "search_code": {"desc": "Search for a pattern in the codebase", "args": "<pattern>"},
    "list_dir": {"desc": "List files in a directory", "args": "<dirpath>"},
    "run_test": {"desc": "Run a specific test", "args": "<test_command>"},
    "finish": {"desc": "Submit the patch and finish", "args": "<explanation>"},
}

SYSTEM_PROMPT = """You are a software engineer fixing bugs in a codebase. You have access to these tools:

{tools}

Think step by step. Use one tool at a time. When you're done, use `finish` to submit.

Rules:
- Read files before editing them
- Run tests after making changes
- Generate a proper git diff

EXAMPLE — you must emit tool calls EXACTLY like this:

```list_dir
src
```
```read_file
src/main.py
```
```run_test
pytest tests/test_main.py
```
```finish
Fixed the off-by-one error in src/main.py
```
"""

TDD_PROMPT = """You are a TDD software engineer fixing bugs in a codebase. You have access to these tools:

{tools}

Work in this exact order:
1. REPRODUCE: write a small test that exposes the bug, run it, confirm it FAILS
2. FIX: make the minimal change to the code
3. VERIFY: run the test again, confirm it PASSES
4. REGRESS: run the existing test suite, confirm nothing broke
5. `finish` with a summary and your git diff

This test-first loop catches mistakes early and proves your fix works.
"""


class SWEAgent:
    """Agentic loop for SWE-bench tasks."""

    def __init__(self, model, tokenizer, repo_dir: str, device: str = "cuda",
                 tdd: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.repo_dir = repo_dir
        self.device = device
        self.history = []
        self.max_turns = 20
        self.tdd = tdd

    def run(self, issue: str, instance_id: str = "unknown") -> dict:
        """Run the agent on a SWE task."""
        tools_str = "\n".join(f"  {k}: {v['desc']}" for k, v in TOOLS.items())
        system = (TDD_PROMPT if self.tdd else SYSTEM_PROMPT).format(tools=tools_str)

        # Project context: CLAUDE.md/AGENTS.md files are in 80% of orgs
        # (State of AI Coding 2026). Inject repo-level instructions if present.
        ctx = self._load_project_context()
        if ctx:
            system += f"\n\nPROJECT CONTEXT:\n{ctx[:2000]}" 

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Issue: {issue}\n\nRepo: {self.repo_dir}\n\nStart by exploring the codebase."},
        ]

        for turn in range(self.max_turns):
            # Context window management: keep the last ~8 messages to avoid
            # overflowing the model's context (critical for long SWE tasks).
            if len(messages) > 10:
                messages = messages[:1] + messages[-9:]

            prompt = self._format_messages(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=512,
                    temperature=0.3, top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            messages.append({"role": "assistant", "content": response})
            print(f"\n--- Turn {turn + 1} ---")
            print(response[:300])

            # Parse tool calls (support multiple per turn = parallel execution)
            actions = self._parse_actions(response)

            if not actions:
                actions = [("finish", "Task appears complete or unclear.")]

            finished = False
            for action, args_text in actions:
                if action == "finish":
                    print(f"\n✅ Agent finished: {args_text[:200]}")
                    finished = True
                    break

                result = self._execute_tool(action, args_text)
                # Recovery: detect failures and inject corrective feedback
                if self._is_failure(result):
                    correction = self._failure_hint(action, result)
                    messages.append({"role": "user", "content":
                        f"Result:\n{result[:1500]}\n\n{correction}"})
                else:
                    messages.append({"role": "user", "content": f"Result:\n{result[:2000]}"})

            if finished:
                return {
                    "instance_id": instance_id,
                    "turns": turn + 1,
                    "explanation": args_text,
                    "patch": self._get_patch(),
                    "success": True,
                }

        return {
            "instance_id": instance_id,
            "turns": self.max_turns,
            "patch": self._get_patch(),
            "success": False,
        }

    def _format_messages(self, messages):
        """Format chat messages for the model."""
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _parse_actions(self, text: str) -> list[tuple[str, str]]:
        """Parse ALL tool calls from model output (for parallel execution).

        Supports multiple ```tool\nargs``` blocks in one response.
        """
        actions = []
        for m in re.finditer(r'```(\w+)\s*\n(.*?)```', text, re.DOTALL):
            actions.append((m.group(1), m.group(2).strip()))

        # Fallback: <tool>args</tool> blocks
        if not actions:
            for m in re.finditer(r'<(\w+)>(.*?)</\w+>', text, re.DOTALL):
                actions.append((m.group(1), m.group(2).strip()))

        return actions

    def _parse_action(self, text: str):
        """Parse tool calls from model output."""
        # Look for ```tool_name\nargs\n``` pattern
        m = re.search(r'```(\w+)\s*\n(.*?)```', text, re.DOTALL)
        if m:
            return m.group(1), m.group(2).strip()

        # Look for <tool>args</tool> pattern
        m = re.search(r'<(\w+)>(.*?)</\w+>', text, re.DOTALL)
        if m:
            return m.group(1), m.group(2).strip()

        # Look for tool_name at start of line: read_file(path)
        m = re.search(r'^(\w+)[( ](.+?)[)]', text, re.MULTILINE)
        if m:
            return m.group(1), m.group(2).strip().strip("'\"")

        return "finish", "Task appears complete or unclear."

    def _load_project_context(self) -> str:
        """Load project context from CLAUDE.md / AGENTS.md / .chaton.md."""
        for name in ["CLAUDE.md", "AGENTS.md", "AGENT.md", ".chaton.md", ".cursorrules"]:
            path = os.path.join(self.repo_dir, name)
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        return f.read()
                except Exception:
                    pass
        return ""

    def _is_failure(self, result: str) -> bool:
        """Detect failure symptoms in tool output (Harness-Bench finding)."""
        if not result or result.startswith("Error:"):
            return True
        lower = result.lower()
        markers = [
            "not found", "no such file", "error", "failed", "traceback",
            "syntaxerror", "importerror", "keyerror", "filenotfound",
            "does not exist", "command not found", "returned non-zero",
            "permission denied",
        ]
        return any(m in lower for m in markers)

    def _failure_hint(self, action: str, result: str) -> str:
        """Generate corrective feedback (self-recovery)."""
        hints = {
            "read_file": "File may not exist or path wrong. Use list_dir to find it.",
            "search_code": "Pattern may not match. Try simpler pattern or check structure.",
            "run_test": "Test command failed. Check error; may be syntax or missing dep.",
            "list_dir": "Directory may not exist. Try repo root or check typos.",
            "write_file": "Could not write. Check parent directory exists.",
        }
        base = hints.get(action, "Tool call failed. Examine error and retry with corrected args.")
        return "[Recovery] " + base + " Error was: " + result[:300]

    def _execute_tool(self, action: str, args_text: str) -> str:
        """Execute a tool and return the result."""
        try:
            if action == "read_file":
                path = self._resolve_path(args_text)
                with open(path) as f:
                    content = f.read()
                return f"File {path} ({len(content)} chars):\n{content[:2000]}"

            elif action == "write_file":
                parts = args_text.split("\n", 1)
                path = self._resolve_path(parts[0].strip())
                content = parts[1] if len(parts) > 1 else ""
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                return f"Written {len(content)} chars to {path}"

            elif action == "search_code":
                result = subprocess.run(
                    ["grep", "-rn", args_text, "--include=*.py", "--include=*.js",
                     "--include=*.ts", "--include=*.rs", "--include=*.go",
                     "--include=*.java", "--include=*.c", "--include=*.cpp",
                     "--include=*.h", "--include=*.hpp"],
                    cwd=self.repo_dir, capture_output=True, text=True, timeout=15,
                )
                output = result.stdout[:2000] or result.stderr[:500]
                return f"Search results:\n{output}"

            elif action == "list_dir":
                path = self._resolve_path(args_text)
                files = os.listdir(path) if os.path.isdir(path) else []
                # Filter to show important files first
                dirs = sorted(f for f in files if os.path.isdir(os.path.join(path, f)))
                files_list = sorted(f for f in files if not os.path.isdir(os.path.join(path, f)))
                return f"Directory {path}:\n" + "\n".join(
                    [f"  {d}/" for d in dirs] + [f"  {f}" for f in files_list]
                )[:2000]

            elif action == "run_test":
                result = subprocess.run(
                    args_text, shell=True, cwd=self.repo_dir,
                    capture_output=True, text=True, timeout=60,
                )
                return (result.stdout + result.stderr)[:2000] or "No output"

            else:
                return f"Unknown tool: {action}. Available: {', '.join(TOOLS.keys())}"

        except Exception as e:
            return f"Error: {e}"

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative or absolute path."""
        path = path.strip().strip("'\"").strip("`")
        if os.path.isabs(path):
            return path
        return os.path.join(self.repo_dir, path)

    def _get_patch(self) -> str:
        """Get the git diff from the repo."""
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=self.repo_dir, capture_output=True, text=True, timeout=10,
            )
            return result.stdout
        except Exception:
            return ""


def collect_selfplay_data(model, tokenizer, task: str, repo_dir: str,
                          device: str = "cuda", max_pairs: int = 5):
    """Collect self-play training data: the model injects bugs then fixes them.

    SSR (Self-Play SWE-RL, ICML 2025) trains agents by generating
    bug-inject → bug-fix pairs from real codebases with zero human labels.
    Saves (task, buggy_code, fixed_code) pairs to selfplay_data.json.
    """
    import json as _json
    from agent_qwen import generate

    pairs = []
    for i in range(max_pairs):
        # Ask model to introduce a subtle bug into the repo
        inject_prompt = f"""In {repo_dir}, introduce a subtle bug into one source file.
Do NOT break syntax. Make the bug a logic error (wrong comparison, off-by-one,
reversed condition). Output only the modified code."""
        buggy = generate(model, tokenizer, inject_prompt, max_new=256, temperature=0.8)

        # Ask model to find and fix it
        fix_prompt = f"""There is a subtle logic bug in this code. Find and fix it.

CODE:
{buggy[:1500]}

Output only the fixed code."""
        fixed = generate(model, tokenizer, fix_prompt, max_new=256, temperature=0.3)

        pairs.append({
            "task": task,
            "buggy": buggy,
            "fixed": fixed,
        })
        print(f"[selfplay] pair {i+1}/{max_pairs} collected")

    with open("selfplay_data.json", "w") as f:
        _json.dump(pairs, f, indent=2)
    print(f"[selfplay] Saved {len(pairs)} pairs to selfplay_data.json")


def main():
    parser = argparse.ArgumentParser(description="SWE-bench agent")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--repo", default=None, help="Path to git repo")
    parser.add_argument("--issue", default=None, help="Issue description")
    parser.add_argument("--instance", default=None, help="SWE-bench instance JSON")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--tdd", action="store_true",
                        help="Test-Driven Development loop: test first, then fix, then verify")
    parser.add_argument("--selfplay", action="store_true",
                        help="Collect self-play training data (bug inject + fix pairs)")
    args = parser.parse_args()

    # Load model
    from eval_qwen import load_qwen
    model, tokenizer = load_qwen(args.model, args.ckpt, use_4bit=args.four_bit)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get issue + repo
    if args.instance:
        with open(args.instance) as f:
            inst = json.load(f)
        issue = inst.get("problem_statement", inst.get("issue", ""))
        repo = args.repo or inst.get("repo_path", ".")
    else:
        issue = args.issue or "Fix the bug"
        repo = args.repo or "."

    if args.selfplay:
        collect_selfplay_data(model, tokenizer, issue, repo, device)
        return

    agent = SWEAgent(model, tokenizer, repo, device=device, tdd=args.tdd)
    result = agent.run(issue)

    print(f"\n{'='*50}")
    print(f"Result: {result['success']}")
    print(f"Turns: {result['turns']}")
    if result["patch"]:
        print(f"Patch ({len(result['patch'])} chars):")
        print(result["patch"][:1000])


if __name__ == "__main__":
    main()
