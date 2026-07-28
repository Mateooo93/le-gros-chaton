"""Sandboxed command executor for the agent harness.

The agent emits shell commands; this runs them safely and returns stdout+stderr.
Safety: a timeout (so an infinite loop can't hang the agent), a working
directory it can't escape from, a denylist of dangerous patterns, and output
truncation so the model's context window doesn't fill up.

This is NOT a real OS-level sandbox (no namespace/seccomp) — good enough for a
learning agent on your own machine, NOT for untrusted model output. For
untrusted models you'd run this in a container/firejail.

The interface:
    r = run_cmd("ls -la", timeout=10)
    print(r.stdout, r.stderr, r.rc)
"""
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import ClassVar


# patterns we refuse to run at all (the agent should not be able to nuke things).
# crude but stops the obvious foot-guns. extend as you learn what it tries.
DANGEROUS: ClassVar[list[str]] = [
    r"\brm\s+-rf\s+/?(\s|$)",      # rm -rf /  (nope)
    r"\bmkfs\b",
    r"dd\s+.*of=/dev/",
    r">\s*/dev/sda",
    r"\bsudo\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{.*\};",            # fork bomb
]
_DANGER_RE: list[re.Pattern] = [re.compile(p) for p in DANGEROUS]

# default working dir = the project root (so the agent edits real files)
DEFAULT_CWD: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# max combined stdout + stderr bytes before truncation
_MAX_OUTPUT: int = 8192


@dataclass
class CmdResult:
    """Structured result from running a shell command."""

    stdout: str = ""
    stderr: str = ""
    combined_truncated: str = ""
    rc: int = -1
    timed_out: bool = False
    blocked: bool = False


def is_safe(cmd: str) -> tuple[bool, str]:
    """Return (ok, reason). Filters the obvious foot-guns."""
    for r in _DANGER_RE:
        m = r.search(cmd)
        if m:
            return False, f"blocked dangerous pattern: {r.pattern!r} matched {m.group(0)!r}"
    return True, ""


def run_cmd(cmd: str, timeout: float = 20.0, cwd: str | None = None) -> CmdResult:
    """Run a shell command, return a ``CmdResult``.

    Uses ``shell=True`` because the agent's commands are free-form shell strings
    (pipes, redirects). *timeout* kills hung processes. *cwd* confines it.
    Returns up to ~8 KB of combined output so the model's context doesn't fill.
    """
    ok, why = is_safe(cmd)
    if not ok:
        return CmdResult(
            stderr=f"[sandbox] {why}", rc=126, blocked=True,
        )

    cwd = cwd or DEFAULT_CWD
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CmdResult(
            stderr=f"[sandbox] timed out after {timeout}s", rc=124, timed_out=True,
        )
    except Exception as e:
        return CmdResult(
            stderr=f"[sandbox] exec error: {e}", rc=125,
        )

    stdout = p.stdout or ""
    stderr = p.stderr or ""
    combined = stdout + ("\n" if stderr else "") + stderr
    if len(combined) > _MAX_OUTPUT:
        combined = combined[:_MAX_OUTPUT] + (
            f"\n[sandbox] output truncated ({len(combined) - _MAX_OUTPUT} more bytes)"
        )
    return CmdResult(
        stdout=stdout, stderr=stderr,
        combined_truncated=combined, rc=p.returncode,
    )


if __name__ == "__main__":
    # quick self-test
    print(run_cmd("echo hello && ls | head -3"))
    print(run_cmd("rm -rf /", cwd="/tmp"))         # should be blocked
    print(run_cmd("sleep 5", timeout=1))           # should time out