"""Terminal-Bench 2.0 agent for Le Gros Chaton (Harbor ``BaseAgent``).

Design decision
---------------
Terminal-Bench 2.0 is run by the official **Harbor** harness
(``pip install harbor``; ``harbor run -d terminal-bench@2.0 ...``) — the
successor of the v1 ``tb`` CLI. Harbor drives "external" agents through the
``BaseAgent`` interface: our loop runs on the host (or a GPU box) and executes
every tool as a shell command inside the task's Docker sandbox via
``BaseEnvironment.exec``. That is exactly the leaderboard setup:
``harbor run -d terminal-bench/terminal-bench-2 --agent-import-path
"path.to.agent:SomeAgent" -k 5``.

Tool loop
---------
The agent reuses the SWEAgent tool protocol (`` ```tool\\nargs``` `` /
``[tool\\nargs]`` / ``<tool>args</tool>``) and adds a general-purpose
``run_cmd`` bash tool to the toolset (see ``agent_swe.TOOLS``). All tools —
including read_file/write_file/search_code/list_dir/run_test — are executed
inside the sandbox, so the model gets a real terminal, which is what
Terminal-Bench tasks require.

Model backends
--------------
The model can be served three ways (constructor kwargs / env, passed via
``harbor run --ak key=value`` or ``--ae KEY=VALUE``):

- ``model_server_url`` (+ ``model_api_key``): any OpenAI-compatible
  ``/v1/chat/completions`` endpoint (Modal vLLM, vLLM on a GPU box, ...).
  This is the recommended path and the only one that works on boxes without
  enough VRAM/disk for the 9B weights.
- ``local_model`` / ``local_ckpt``: load transformers weights in-process
  (needs a GPU box with ~20GB disk for the fp16 weights, or 4-bit).
- ``mock``: scripted model responses — used by ``tbench_eval.py --dry-run``
  to verify the sandbox/harness paths without any model.

The official full-eval commands are documented in ``eval/tbench_eval.py``
docstring and the repo README.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import time
import uuid
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from agent_swe import TOOLS  # noqa: E402  (adds run_cmd to the shared toolset)

# --------------------------------------------------------------------------
# Model clients
# --------------------------------------------------------------------------


class ModelClient:
    """Synchronous chat interface used by the agent loop."""

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


class ServerModelClient(ModelClient):
    """OpenAI-compatible /v1/chat/completions client (vLLM, Modal, HF Hub...).

    ``hf_inference`` enables HF Inference Providers compatibility: the
    Qwen3.5-9B served there emits long ``reasoning`` tokens by default, so we
    pass ``chat_template_kwargs={"enable_thinking": false}`` to get plain
    content (this is a no-op for servers that ignore extra fields).
    """

    def __init__(self, base_url: str, api_key: str = "", model_name: str = "",
                 temperature: float = 0.3, max_new_tokens: int = 1500,
                 timeout: float = 300.0, hf_inference: bool = False,
                 max_retries: int = 4, disable_thinking: bool = True):
        import httpx
        self._httpx = httpx
        self.base_url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key
        self.model_name = model_name or "default"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.hf_inference = hf_inference
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }
        if self.disable_thinking:
            # Qwen3.5 burns its budget on verbose `reasoning` tokens; this
            # disables thinking on HF Inference, vLLM, and llama.cpp servers.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._httpx.post(
                    self.base_url, json=payload, headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    raise self._httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}",
                        request=resp.request, response=resp,
                    )
                data = resp.json()
                try:
                    msg = data["choices"][0]["message"]
                    content = msg.get("content")
                    # Servers differ: HF returns `reasoning`, llama.cpp
                    # `reasoning_content`, some return content=None when the
                    # budget went to thinking. Surface the thinking text so
                    # the loop can react instead of hanging.
                    if not content:
                        content = (msg.get("reasoning_content")
                                   or msg.get("reasoning") or "")
                    return content
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(
                        f"Unexpected chat-completions response: {data}") from exc
            except (self._httpx.HTTPStatusError, self._httpx.TransportError) as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                # Do not retry auth/not-found errors; retry everything else
                # (HF free tier cold-starts and rate limits are transient).
                if status in (401, 403, 404) or attempt == self.max_retries - 1:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"Model server request failed after {self.max_retries} attempt(s): "
            f"{last_exc}") from last_exc


class LocalModelClient(ModelClient):
    """In-process transformers generation (GPU box with the weights)."""

    def __init__(self, model_name: str, ckpt: str | None = None,
                 use_4bit: bool = False, temperature: float = 0.3,
                 max_new_tokens: int = 900):
        from eval_qwen import load_qwen
        self.model, self.tokenizer = load_qwen(model_name, ckpt, use_4bit=use_4bit)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict]) -> str:
        import torch
        prompt = format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                temperature=self.temperature, top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


class MockModelClient(ModelClient):
    """Scripted model for dry-runs: exercises the loop + sandbox, no model.

    ``script`` is a list of canned responses returned in order; the last
    response repeats if the model would have been called more times.
    """

    DEFAULT_SCRIPT = [
        "Let me look around the sandbox first.\n```run_cmd\npwd && ls -la\n```",
        "The sandbox is reachable and commands execute. Done (dry-run).\n"
        "```finish\nDry-run: verified sandbox shell access.\n```",
    ]

    def __init__(self, script: list[str] | None = None):
        self.script = script or self.DEFAULT_SCRIPT

    def chat(self, messages: list[dict]) -> str:
        # Count how many assistant turns so far to index the script.
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        return self.script[min(n_assistant, len(self.script) - 1)]

    def name(self) -> str:
        return "mock"


def make_client(kwargs: dict[str, Any]) -> ModelClient:
    """Build the model client from agent kwargs / environment."""
    server_url = kwargs.get("model_server_url") or os.environ.get("MODEL_SERVER_URL")
    if server_url:
        hf = (kwargs.get("hf_inference") or os.environ.get("HF_INFERENCE") == "1"
              or "huggingface.co" in server_url)
        return ServerModelClient(
            base_url=server_url,
            api_key=kwargs.get("model_api_key") or os.environ.get("MODEL_API_KEY", ""),
            model_name=kwargs.get("model_name") or os.environ.get("MODEL_NAME", ""),
            temperature=float(kwargs.get("temperature", 0.3)),
            max_new_tokens=int(kwargs.get("max_new_tokens", 1500)),
            hf_inference=hf,
        )
    if kwargs.get("mock") or os.environ.get("TB_MOCK") == "1":
        return MockModelClient()
    if kwargs.get("local_model") or os.environ.get("LOCAL_MODEL"):
        return LocalModelClient(
            model_name=kwargs.get("local_model") or os.environ.get("LOCAL_MODEL", ""),
            ckpt=kwargs.get("local_ckpt") or os.environ.get("LOCAL_CKPT"),
            use_4bit=bool(kwargs.get("four_bit")) or os.environ.get("LOCAL_4BIT") == "1",
            temperature=float(kwargs.get("temperature", 0.3)),
        )
    raise ValueError(
        "No model backend configured for the TB agent. Pass --ak "
        "model_server_url=<url> (OpenAI-compatible), --ak local_model=<hf-id>, "
        "or --ak mock=true (dry-run)."
    )


# --------------------------------------------------------------------------
# Tool parsing / message formatting (mirrors agent_swe helpers)
# --------------------------------------------------------------------------


def parse_tool_calls(text: str) -> list[tuple[str, str]]:
    """Parse ALL tool calls from model output (multiple syntaxes)."""
    actions = []
    for m in re.finditer(r'```(\w+)\s*\n(.*?)```', text, re.DOTALL):
        actions.append((m.group(1), m.group(2).strip()))
    if not actions:
        for m in re.finditer(r'\[(\w+)\s*\n(.*?)\]', text, re.DOTALL):
            actions.append((m.group(1), m.group(2).strip()))
    if not actions:
        for m in re.finditer(r'<(\w+)>(.*?)</\w+>', text, re.DOTALL):
            actions.append((m.group(1), m.group(2).strip()))
    return actions


def format_messages(messages: list[dict]) -> str:
    """ChatML formatting identical to SWEAgent._format_messages."""
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


# --------------------------------------------------------------------------
# The TB system prompt (terminal-agent; run_cmd is the primary tool)
# --------------------------------------------------------------------------

TB_SYSTEM_PROMPT = """You are an expert terminal operator. You have a shell inside an isolated Linux container and must complete the user's task by running commands and editing files.

Available tools (call EXACTLY ONE per turn, in this format):
{tool_blocks}

Rules:
- Think step by step. Inspect the environment first (pwd, ls) before acting.
- Use `run_cmd` for ANY shell command: installing packages, moving files, running scripts, curl, git, python, etc.
- `run_test` runs a command too and is fine for tests.
- When you finish, call `finish` with a short explanation of what you did.
- You get a tool RESULT after every call. Use it. Never repeat the same command in a loop.
"""


def _build_tool_blocks(tools: dict) -> str:
    lines = []
    for name, spec in tools.items():
        lines.append(f"```{name}\n{spec['args']}```  # {spec['desc']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Sandbox tool execution
# --------------------------------------------------------------------------


async def exec_in_sandbox(environment: BaseEnvironment, command: str,
                          timeout_sec: int = 300) -> str:
    """Run a bash command in the task sandbox and format the result."""
    try:
        result: ExecResult = await environment.exec(
            command, timeout_sec=timeout_sec,
        )
    except Exception as exc:  # harbor raises on timeout/exec errors
        return f"Error: command failed to execute: {exc}"
    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append("stderr:\n" + result.stderr.rstrip())
    if not parts:
        parts.append("(no output)")
    parts.append(f"[exit code {result.return_code}]")
    return "\n".join(parts)[:4000]


async def execute_tool(environment: BaseEnvironment, action: str,
                       args_text: str, timeout_sec: int = 300) -> str:
    """Execute a tool call inside the sandbox. Every tool is a shell command."""
    args_text = args_text.strip()
    try:
        if action == "run_cmd":
            return await exec_in_sandbox(environment, args_text, timeout_sec)
        elif action == "run_test":
            return await exec_in_sandbox(environment, args_text, timeout_sec)
        elif action == "read_file":
            return await exec_in_sandbox(
                environment, f"cat {_q(args_text)} 2>&1 | head -c 2000",
                timeout_sec)
        elif action == "list_dir":
            return await exec_in_sandbox(
                environment, f"ls -la {_q(args_text or '.')} 2>&1 | head -c 2000",
                timeout_sec)
        elif action == "search_code":
            return await exec_in_sandbox(
                environment,
                f"grep -rn --include='*.py' --include='*.js' --include='*.ts' "
                f"--include='*.rs' --include='*.go' --include='*.c' "
                f"--include='*.cpp' --include='*.h' --include='*.hpp' "
                f"{_q(args_text)} . 2>&1 | head -c 2000",
                timeout_sec)
        elif action == "write_file":
            parts = args_text.split("\n", 1)
            path = parts[0].strip()
            content = parts[1] if len(parts) > 1 else ""
            b64 = base64.b64encode(content.encode()).decode()
            cmd = (
                f"mkdir -p $(dirname {_q(path)}) && "
                f"echo {b64} | base64 -d > {_q(path)} && "
                f"wc -c < {_q(path)}"
            )
            return await exec_in_sandbox(environment, cmd, timeout_sec)
        elif action == "prune":
            # Context is managed automatically by the loop; no-op for the model.
            return "Context is managed automatically; no manual prune needed."
        else:
            return (f"Unknown tool: {action}. Available: "
                    f"{', '.join(TOOLS.keys())}")
    except Exception as exc:
        return f"Error: {exc}"


def _q(s: str) -> str:
    """Minimal shell-quoting for a single argument."""
    s = s.strip().strip("'\"").strip("`")
    return "'" + s.replace("'", "'\\''") + "'"


def _is_failure(result: str) -> bool:
    if not result or result.startswith("Error:"):
        return True
    lower = result.lower()
    markers = [
        "not found", "no such file", "error", "failed", "traceback",
        "syntaxerror", "importerror", "keyerror", "filenotfound",
        "does not exist", "command not found", "returned non-zero",
        "permission denied", "exit code 1", "exit code 2",
    ]
    return any(m in lower for m in markers)


def _failure_hint(action: str, result: str) -> str:
    hints = {
        "read_file": "File may not exist or path is wrong. Use list_dir/run_cmd (ls) to explore.",
        "run_cmd": "Command failed. Read the error and retry with a corrected command.",
        "run_test": "Test command failed. Check the error output.",
        "list_dir": "Directory may not exist. Try 'ls -la' or check typos.",
        "write_file": "Could not write. Check the parent directory exists.",
        "search_code": "Pattern may not match. Try a simpler pattern.",
    }
    base = hints.get(action, "Tool call failed. Examine the error and retry.")

    # doc-retrieve-on-failure: "command not found" (24% of TB command
    # failures per the TB 2.0 error analysis) is usually a knowledge gap,
    # not a fixable mistake. Point the model at the tool's own docs instead
    # of guessing again — check_path candidates come from the error text.
    lower = result.lower()
    if action in ("run_cmd", "run_test") and (
        "command not found" in lower or "not found" in lower
        or "no such file" in lower
    ):
        cand = re.search(r"([\w./+-]+): (?:command )?not found", result)
        cmd = cand.group(1) if cand else None
        doc = ""
        if cmd:
            doc = (
                f"Probe its interface first (STOP guessing): "
                f"`which {cmd}`, `{cmd} --help`, `man {cmd}` if present, and "
                f"`ls /usr/bin/{cmd}*` to see what is actually installed. "
                f"If it is NOT installed, install it (apt/pip as appropriate) "
                f"before relying on it."
            )
        else:
            doc = (
                "Resolve what is missing: `which <cmd>`, `--help`, or check "
                "the package list (`apt list --installed` / `pip list`). "
                "Do NOT retry the same command unchanged."
            )
        return "[Recovery] " + doc + " Error was: " + result[:250]

    return "[Recovery] " + base + " Error was: " + result[:300]


# --------------------------------------------------------------------------
# Harbor agent
# --------------------------------------------------------------------------


class LeGrosChatonTBAgent(BaseAgent):
    """External Harbor agent running the Le Gros Chaton tool loop in a sandbox."""

    @staticmethod
    def name() -> str:
        return "le-gros-chaton"

    def version(self) -> str:
        return os.environ.get("TB_AGENT_VERSION", "0.1.0")

    def __init__(self, logs_dir, model_name=None, logger=None, mcp_servers=None,
                 skills_dir=None, *args, extra_env=None, **kwargs):
        super().__init__(
            logs_dir=logs_dir, model_name=model_name, logger=logger,
            mcp_servers=mcp_servers, skills_dir=skills_dir,
            *args, extra_env=extra_env,
        )
        self._kwargs = dict(kwargs or {})
        # The `-m` flag reaches the constructor as `model_name`; make it
        # available to make_client (which otherwise only sees --ak kwargs).
        if "model_name" not in self._kwargs and self.model_name:
            self._kwargs["model_name"] = self.model_name
        self.client = None
        self.max_turns = int(self._kwargs.get("max_turns", 100))
        self.temperature = float(self._kwargs.get("temperature", 0.3))
        self.exec_timeout = int(self._kwargs.get("exec_timeout", 300))
        self.ctx_budget = int(self._kwargs.get("ctx_budget", 65536))
        # The serving backend's context window is usually the binding
        # constraint (e.g. llama-server -c 16384); prune to fit it.
        self.server_ctx_limit = int(self._kwargs.get("server_ctx_limit", 14000))
        # Scheduled compaction: force a state-sheet checkpoint every N turns
        # (0 disables). Counters for finish-gate + dead-end detection live in
        # the run loop, not here.
        self.compact_every = int(self._kwargs.get("compact_every", 10))

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the model runs on the host / remote server."""
        pass

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        t0 = time.time()
        if self.client is None:
            self.client = make_client(self._kwargs)
        self.logger.info("TB agent run with client %s", self.client.name())

        tools_str = _build_tool_blocks(TOOLS)
        system = TB_SYSTEM_PROMPT.format(tool_blocks=tools_str)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task:\n{instruction}"},
        ]

        recent_actions: list[tuple[str, str]] = []
        tool_calls_used = 0
        finished = False
        # Finish-gate + dead-end detection + scheduled compaction state.
        n_successful_tools = 0      # any tool result that was NOT a failure
        consecutive_failures = 0    # tool failures in a row (task-level)
        summary = {"turns": 0, "tool_calls": 0, "finish_note": "", "error": None}

        for turn in range(self.max_turns):
            summary["turns"] = turn + 1
            # Context pruning: drop oldest non-system messages while over budget.
            prompt = format_messages(messages)
            n_tok = self._count_tokens(prompt)
            effective_budget = min(self.ctx_budget, self.server_ctx_limit)
            while n_tok > effective_budget and len(messages) > 3:
                messages.pop(1)
                prompt = format_messages(messages)
                n_tok = self._count_tokens(prompt)

            try:
                response = await asyncio.to_thread(self.client.chat, messages)
            except Exception as exc:
                self.logger.error("Model call failed (turn %s): %s", turn + 1, exc)
                summary["consecutive_model_failures"] = \
                    summary.get("consecutive_model_failures", 0) + 1
                if summary["consecutive_model_failures"] >= 3:
                    summary["error"] = f"model call failed 3x: {exc}"
                    break
                messages.append({"role": "user", "content":
                    f"[system] The model call failed transiently ({exc}). "
                    "Retry with a tool call."})
                continue
            summary["consecutive_model_failures"] = 0
            messages.append({"role": "assistant", "content": response})

            actions = parse_tool_calls(response)
            if not actions:
                correction = (
                    "You did not make a tool call. You MUST call exactly one "
                    "tool using this format:\n"
                    "```tool_name\nargs```\n"
                    f"Available tools: {', '.join(TOOLS)}.\n"
                    "Take an action with a tool call."
                )
                messages.append({"role": "user", "content": correction})
                continue

            for action, args_text in actions:
                if action == "finish":
                    # FINISH-GATE: don't let the model declare victory without
                    # any observable progress. Terminal-Bench rewards END
                    # STATE — a bare finish after zero successful actions is
                    # almost always a hallucinated "done".
                    if n_successful_tools == 0:
                        gate = (
                            "You tried to finish, but nothing you ran has "
                            "succeeded yet (no verified action in this "
                            "episode). Finish will NOT be accepted. Take a "
                            "real action first — explore, edit, or run a "
                            "verification — then finish with proof."
                        )
                        messages.append({"role": "user", "content": gate})
                        self.logger.warning(
                            "FINISH-GATE: blocked finish with 0 successful tools")
                        continue
                    summary["finish_note"] = args_text[:200]
                    finished = True
                    break

                # DEAD-END DETECTOR: a run of consecutive failures means the
                # current strategy is stuck. Force a strategy break instead of
                # letting it grind the same approach (coherence failure class).
                if consecutive_failures >= 3:
                    self.logger.warning(
                        "DEAD-END: %d consecutive failures -> strategy pivot",
                        consecutive_failures)
                    pivot = (
                        f"You have failed {consecutive_failures} actions in a "
                        "row with the current strategy. STOP grinding. "
                        "Describe TWO genuinely different approaches to this "
                        "task, then pick ONE and try it."
                    )
                    messages.append({"role": "user", "content": pivot})
                    consecutive_failures = 0

                sig = (action, args_text[:60])
                dup = sum(1 for a in recent_actions[-8:] if a == sig)
                if dup >= 3:
                    correction = (
                        f"You have called {action}({args_text[:50]}) {dup} times "
                        "recently. This is a LOOP. STOP and try something "
                        "genuinely different, or call finish if you are done."
                    )
                    messages.append({"role": "user", "content": correction})
                    recent_actions = []
                    self.logger.warning("LOOP detected on %s", action)
                    continue

                recent_actions.append(sig)
                tool_calls_used += 1
                summary["tool_calls"] = tool_calls_used

                result = await execute_tool(
                    environment, action, args_text,
                    timeout_sec=self.exec_timeout,
                )
                self.logger.info("[tool %s] %s", action, args_text[:120])
                # (turns already tracked at the top of the turn loop)
                if _is_failure(result):
                    consecutive_failures += 1
                    correction = _failure_hint(action, result)
                    messages.append({"role": "user", "content":
                        f"Result:\n{result[:2000]}\n\n{correction}"})
                else:
                    consecutive_failures = 0
                    n_successful_tools += 1
                    messages.append({"role": "user", "content":
                        f"Result:\n{result[:2000]}"})

            # SCHEDULED COMPACTION: every COMPACT_EVERY turns, force the model
            # to write a compact state-sheet (goal/known/tried/failed/next) and
            # drop the oldest messages so the session stays focused (context
            # rot is a top coherence failure; TB found no success correlation
            # with raw token volume — managing context is what wins).
            if turn and turn % self.compact_every == 0 and not finished:
                checkpoint = (
                    "[CHECKPOINT] Before your next action, write a COMPACT "
                    "state-sheet on one line: "
                    "[STATE] goal=... | known=... | tried=... | failed=... | "
                    "next=... . Then continue with exactly one tool call. "
                    "Do not repeat commands you have already run."
                )
                messages.append({"role": "user", "content": checkpoint})
                # Drop old messages but keep system + task + the checkpoint.
                # (Mandatory context management beats reactive pruning.)
                if len(messages) > 12:
                    messages = (
                        messages[:3] + messages[-9:] if not finished
                        else messages
                    )
                self.logger.info("COMPACT at turn %d (messages -> %d)",
                                 turn, len(messages))

            if finished:
                break

        summary["seconds"] = round(time.time() - t0, 1)
        summary["max_turns_reached"] = not finished
        context.metadata = {
            "turns": summary["turns"],
            "tool_calls": summary["tool_calls"],
            "finish_note": summary["finish_note"],
            "seconds": summary["seconds"],
            "max_turns_reached": summary["max_turns_reached"],
            "client": self.client.name(),
        }
        # Persist the raw transcript for later RLVR/trajectory work.
        try:
            self._save_trace(messages, context.metadata)
        except Exception:
            pass

    def _count_tokens(self, prompt: str) -> int:
        tokenizer = getattr(self.client, "tokenizer", None)
        if tokenizer is not None:
            try:
                return len(tokenizer(prompt)["input_ids"])
            except Exception:
                pass
        # Conservative chars/token (code and URLs tokenize denser than 4/1);
        # under-estimating lets requests overflow the serving context.
        return len(prompt) // 3

    def _save_trace(self, messages: list[dict], meta: dict) -> None:
        out_dir = os.environ.get("TB_TRACES_DIR") or os.path.join(
            PROJ_ROOT, "eval", "tb_traces")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"tb_{int(time.time())}_{uuid.uuid4().hex[:8]}.jsonl")
        with open(path, "w") as f:
            for m in messages:
                f.write(json.dumps(m) + "\n")
            f.write(json.dumps({"meta": meta}) + "\n")
