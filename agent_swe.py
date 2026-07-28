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
"""


class SWEAgent:
    """Agentic loop for SWE-bench tasks."""

    def __init__(self, model, tokenizer, repo_dir: str, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.repo_dir = repo_dir
        self.device = device
        self.history = []
        self.max_turns = 20

    def run(self, issue: str, instance_id: str = "unknown") -> dict:
        """Run the agent on a SWE task."""
        tools_str = "\n".join(f"  {k}: {v['desc']}" for k, v in TOOLS.items())
        system = SYSTEM_PROMPT.format(tools=tools_str)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Issue: {issue}\n\nRepo: {self.repo_dir}\n\nStart by exploring the codebase."},
        ]

        for turn in range(self.max_turns):
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

            # Parse tool call
            action, args_text = self._parse_action(response)

            if action == "finish":
                print(f"\n✅ Agent finished: {args_text[:200]}")
                return {
                    "instance_id": instance_id,
                    "turns": turn + 1,
                    "explanation": args_text,
                    "patch": self._get_patch(),
                    "success": True,
                }

            result = self._execute_tool(action, args_text)
            messages.append({"role": "user", "content": f"Result:\n{result[:2000]}"})

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


def main():
    parser = argparse.ArgumentParser(description="SWE-bench agent")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--repo", default=None, help="Path to git repo")
    parser.add_argument("--issue", default=None, help="Issue description")
    parser.add_argument("--instance", default=None, help="SWE-bench instance JSON")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
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

    agent = SWEAgent(model, tokenizer, repo, device=device)
    result = agent.run(issue)

    print(f"\n{'='*50}")
    print(f"Result: {result['success']}")
    print(f"Turns: {result['turns']}")
    if result["patch"]:
        print(f"Patch ({len(result['patch'])} chars):")
        print(result["patch"][:1000])


if __name__ == "__main__":
    main()
