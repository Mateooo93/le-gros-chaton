"""Verifier — does a candidate solution pass a problem's tests?

This is the keystone of the whole RLVR/RFT/PRM pipeline. Every stage depends on
it: RFT filters sampled solutions by it, RLVR uses it as the reward, the code
PRM gets "toward passing" labels from running it step-by-step. Build it first.

Design: a "Problem" carries (prompt, tests, language). `verify()` runs a
candidate solution in the agent sandbox, returns a structured Verdict with
pass/fail per test and the raw output. We reuse agent/sandbox.py so a real
sandbox (timeout + cwd + dangerous-pattern block) is already in place.

Initially this handles HumanEval-style problems:
  - prompt: the function signature + docstring (the model completes the body)
  - tests: either a list of `assert func(...)==...` lines OR a test file path
  - language: "python" for now (the verifier is code-agnostic by design — add
    a language pack later for go/js/etc)
"""
import os
import sys
import textwrap
from dataclasses import dataclass, field

# agent/ is a sibling package — add the project root so the import resolves
# whether we run as `python verify/verifier.py` or `python -m verify.verifier`.
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from agent.sandbox import run_cmd


@dataclass
class Problem:
    """A coding problem with a visible prompt and a hidden test suite."""
    id: str
    prompt: str               # what the model sees + must complete
    tests: str                # executable test code appended to the completion
    language: str = "python"
    # optional entry point name for HumanEval-style `check(candidate)` setups
    entry_point: str | None = None


@dataclass
class Verdict:
    """Result of verifying one candidate."""
    passed: bool
    n_pass: int
    n_total: int
    per_test: list[bool] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    rc: int = -1
    timed_out: bool = False
    error: str = ""           # if the verifier itself failed (not the solution)


def _python_check_code(problem: Problem, solution: str) -> str:
    """Assemble the runnable check file: solution + tests.

    Plain concatenation — no dedent. The solution's own indentation is
    authoritative; we just append the candidate binding + tests + OK marker.
    """
    if problem.entry_point:
        # HumanEval convention: tests call `check(candidate)`. Bind candidate
        # to the entry point so the asserts can find it even if the solution
        # didn't define it with that exact name (it should, HumanEval requires it).
        return f"{solution}\ncandidate = {problem.entry_point}\n{problem.tests}\nprint('OK')\n"
    # generic: just append the tests
    return f"{solution}\n{problem.tests}\nprint('OK')\n"


def verify(problem: Problem, solution: str, timeout: float = 15.0,
           cwd: str | None = None) -> Verdict:
    """Run `solution` against `problem.tests`, return a per-test Verdict."""
    cwd = cwd or os.path.join(PROJ_ROOT, "verify", "_runs")
    os.makedirs(cwd, exist_ok=True)

    code = _python_check_code(problem, solution)
    run_file = os.path.join(cwd, f"_sol_{abs(hash(problem.id)) & 0xffff}.py")
    with open(run_file, "w") as f:
        f.write(code)

    r = run_cmd(f"python {os.path.basename(run_file)}", timeout=timeout, cwd=cwd)
    out = (r.get("combined_truncated") or "")
    rc = r.get("rc", -1)

    # count asserts: a passing run prints OK and exits 0. A failing run exits
    # non-zero with an AssertionError. We parse per-test results from the tests
    # block by running them one at a time if the bulk run failed — cheap way to
    # get per-test detail without a test framework dependency.
    if rc == 0 and "OK" in (r.get("stdout") or ""):
        # whole suite passed — every test passed
        n_total = _count_asserts(problem.tests)
        return Verdict(passed=True, n_pass=n_total, n_total=n_total,
                       per_test=[True] * n_total, stdout=r.get("stdout", ""),
                       stderr=r.get("stderr", ""), rc=rc, timed_out=r.get("timed_out", False))

    if r.get("timed_out"):
        return Verdict(passed=False, n_pass=0, n_total=_count_asserts(problem.tests),
                       stdout=r.get("stdout", ""), stderr=r.get("stderr", ""),
                       rc=rc, timed_out=True)

    # bulk run failed -> run each assert individually to get per-test detail
    per_test = _per_test_results(problem, solution, timeout, cwd)
    n_total = len(per_test)
    n_pass = sum(per_test)
    return Verdict(passed=(n_pass == n_total and n_total > 0),
                   n_pass=n_pass, n_total=n_total, per_test=per_test,
                   stdout=r.get("stdout", ""), stderr=r.get("stderr", ""), rc=rc,
                   timed_out=False,
                   error=r.get("stderr", "")[:500] if rc != 0 else "")


def _count_asserts(tests: str) -> int:
    """Crude count of `assert` statements — used as the test count."""
    return max(1, len([l for l in tests.splitlines() if l.strip().startswith("assert")]))


def _per_test_results(problem: Problem, solution: str, timeout: float, cwd: str) -> list[bool]:
    """Run each assert separately so we know which ones pass."""
    asserts = [l.strip() for l in problem.tests.splitlines()
               if l.strip().startswith("assert")]
    results = []
    for a in asserts:
        code = f"{solution}\n{a}\nprint('OK')\n"
        f = os.path.join(cwd, f"_t_{abs(hash(a)) & 0xffff}.py")
        with open(f, "w") as fh:
            fh.write(code)
        r = run_cmd(f"python {os.path.basename(f)}", timeout=timeout, cwd=cwd)
        results.append(r.get("rc") == 0 and "OK" in (r.get("stdout") or ""))
    return results or [False]


# --- two real HumanEval-style problems for the self-test ---
_SELF_TEST_PROBLEMS = [
    Problem(
        id="humaneval/2",
        prompt="def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        tests="assert add(1,2)==3\nassert add(0,0)==0\nassert add(-1,1)==0\n",
        entry_point="add",
    ),
    Problem(
        id="humaneval/1",
        prompt="def fib(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        tests="assert fib(0)==0\nassert fib(1)==1\nassert fib(10)==55\n",
        entry_point="fib",
    ),
]


if __name__ == "__main__":
    # correct solutions -> should pass
    print("=== correct solutions (expect all pass) ===")
    solutions = {
        "humaneval/2": _SELF_TEST_PROBLEMS[0].prompt + "    return a + b\n",
        "humaneval/1": _SELF_TEST_PROBLEMS[1].prompt
                       + "    if n < 2: return n\n    a,b = 0,1\n"
                       + "    for _ in range(n-1): a,b = b,a+b\n    return b\n",
    }
    for p in _SELF_TEST_PROBLEMS:
        v = verify(p, solutions[p.id])
        print(f"{p.id}: pass={v.passed} {v.n_pass}/{v.n_total} rc={v.rc} timed_out={v.timed_out}")

    print("\n=== a broken solution (expect fail, 0 pass) ===")
    p = _SELF_TEST_PROBLEMS[0]
    sol = p.prompt + "    return a - b\n"   # WRONG (subtraction not addition)
    v = verify(p, sol)
    print(f"{p.id}: pass={v.passed} {v.n_pass}/{v.n_total} rc={v.rc}")
    print(f"  per_test={v.per_test}  stderr={v.stderr[:120]!r}")