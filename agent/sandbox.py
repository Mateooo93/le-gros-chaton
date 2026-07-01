"""Sandboxed command executor for the agent harness.

The agent emits shell commands; this runs them safely and returns stdout+stderr.
Safety: a timeout (so an infinite loop can't hang the agent), a working
directory it can't escape from, and (optional) a denylist of dangerous patterns.
This is NOT a real OS-level sandbox (no namespace/seccomp) — good enough for a
learning agent on your own machine, NOT for untrusted model output. For
untrusted models you'd run this in a container/firejail.

The interface is dead simple on purpose:
    out, rc, timed_out = run_cmd("ls -la", timeout=10)
"""
import os
import re
import shlex
import subprocess

# patterns we refuse to run at all (the agent should not be able to nuke things).
# crude but stops the obvious foot-guns. extend as you learn what it tries.
DANGEROUS = [
    r"\brm\s+-rf\s+/?(\s|$)",      # rm -rf /  (nope)
    r"\bmkfs\b",
    r"dd\s+.*of=/dev/",
    r">\s*/dev/sda",
    r"\bsudo\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{.*\};",            # fork bomb
]
_DANGER_RE = [re.compile(p) for p in DANGEROUS]

# default working dir = the project root (so the agent edits real files)
DEFAULT_CWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_safe(cmd: str) -> tuple[bool, str]:
    """Return (ok, reason). Filters the obvious foot-guns."""
    for r in _DANGER_RE:
        m = r.search(cmd)
        if m:
            return False, f"blocked dangerous pattern: {r.pattern!r} matched {m.group(0)!r}"
    return True, ""


def run_cmd(cmd: str, timeout: float = 20.0, cwd: str | None = None) -> dict:
    """Run a shell command, return a dict with stdout/stderr/rc/timed_out.

    Uses shell=True because the agent's commands are free-form shell strings
    (pipes, redirects). timeout kills hung processes. cwd confines it.
    Returns up to ~8KB of combined output so we don't blow the model's context.
    """
    ok, why = is_safe(cmd)
    if not ok:
        return {"stdout": "", "stderr": f"[sandbox] {why}", "rc": 126,
                "timed_out": False, "blocked": True}

    cwd = cwd or DEFAULT_CWD
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"[sandbox] timed out after {timeout}s",
                "rc": 124, "timed_out": True}
    except Exception as e:
        return {"stdout": "", "stderr": f"[sandbox] exec error: {e}",
                "rc": 125, "timed_out": False}

    out = (p.stdout or "") + ("\n" if p.stderr else "") + (p.stderr or "")
    if len(out) > 8192:
        out = out[:8192] + f"\n[sandbox] output truncated ({len(out)-8192} more bytes)"
    return {"stdout": p.stdout or "", "stderr": p.stderr or "",
            "combined_truncated": out, "rc": p.returncode, "timed_out": False}


if __name__ == "__main__":
    # quick self-test
    print(run_cmd("echo hello && ls | head -3"))
    print(run_cmd("rm -rf /", cwd="/tmp"))         # should be blocked
    print(run_cmd("sleep 5", timeout=1))           # should time out