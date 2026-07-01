"""The agent loop — turn a base LM into a terminal tool-user.

THE BIG IDEA
------------
A coding agent isn't a bigger model. It's a small model + a LOOP:
  1. give the model a task + a system prompt that says "to act, emit
     <cmd>shell command</cmd>; I will run it and give you the output"
  2. model generates text. We strip out any <cmd>...</cmd> blocks it emitted.
  3. run those commands in the sandbox, capture stdout/stderr.
  4. append the model's text AND the command output back into the conversation.
  5. go to 2. Repeat until the model emits <done>...</done> (a final answer /
     "I'm finished") or we hit the step cap.

Most of terminal-bench score comes from THIS loop, not from the weights: a
mediocre model that can run `pytest`, read the failure, edit the file, and
re-run will beat a great model that has to guess the answer in one shot.

EXTENSIONS (later, see agent/README or tasks):
  - test-driven self-repair: prepend the failing test output to the prompt so
    the model's next turn is "fix this specific error".
  - best-of-N: run the loop N times with temperature sampling, keep the run
    whose final state passes a verifier (unit tests green / diff applies).
  - PRM-step scoring: a Process Reward Model scores each step so we can pick
    the best branch without running to completion (the novelty for "le fat chaton").
"""
import re
import torch
from tokenizer import decode, encode
from model import GPT
import config as cfg
from agent.sandbox import run_cmd

# ---- the prompt that turns a base LM into an agent ----
# NOTE (for the student): this prompt IS the agent. A bad prompt = a bad agent,
# even with a perfect model. This one is deliberately simple + explicit about
# the two tags <cmd> and <done>. You'll want to tune it a lot.
SYSTEM = """\
You are a terminal coding agent. You solve tasks by running shell commands.

To act, emit a command inside <cmd> tags, e.g.:
  <cmd>ls -la</cmd>
The user will run it and append the output. You may emit multiple <cmd> blocks
before a single <cmd> must be a shell command on one line (no newlines inside).
Reason briefly between commands. When the task is complete, emit a final
answer inside <done> tags:
  <done>The fix was applied: changed X to Y in foo.py. Tests pass.</done>
Keep commands small and testable. Prefer `cat`, `grep`, `pytest`, `python -c`.
"""

# regex to pull <cmd>...</cmd> blocks out of the model's output (greedy=False so
# multiple blocks in one generation each get matched). DOTALL keeps newlines.
_CMD_RE = re.compile(r"<cmd>(.*?)</cmd>", re.DOTALL)
_DONE_RE = re.compile(r"<done>(.*?)</done>", re.DOTALL)


def parse_actions(text: str):
    """Return (cmds, done_text). done_text is None if the model didn't finish."""
    cmds = [m.group(1).strip() for m in _CMD_RE.finditer(text)]
    done_m = _DONE_RE.search(text)
    done = done_m.group(1).strip() if done_m else None
    return cmds, done


def _load_model(ckpt_path: str, device: str):
    """Build the model from config and load a checkpoint (weights only)."""
    m = GPT().to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    # checkpoints from train.py save a bare state_dict; ckpt.py saves a dict.
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd)
    m.eval()
    return m


def run(task: str, ckpt_path: str = "model.pt", max_steps: int = 10,
        max_new_tokens: int = 200, temperature: float = 0.7, verbose: bool = True):
    """Run the agent loop on a task. Returns the final <done> answer (or None)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(ckpt_path, device)

    # the conversation, as a single growing string (we re-encode the whole thing
    # each step — simple, fine for a learning harness; a real one uses the KV
    # cache to incrementally decode).
    convo = f"{SYSTEM}\n\nTask: {task}\n\n"

    for step in range(max_steps):
        idx = torch.tensor([encode(convo)], dtype=torch.long, device=device)
        # crop to the model's context window from the LEFT (keep the latest)
        if idx.size(1) > cfg.block_size:
            idx = idx[:, -cfg.block_size:]
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=50, top_p=0.9,
                             repetition_penalty=1.15)
        gen = decode(out[0].tolist())
        # the newly generated text is everything after the prompt
        gen = gen[len(decode(idx[0].tolist())):]
        if verbose:
            print(f"\n--- step {step} model output ---\n{gen}")

        cmds, done = parse_actions(gen)
        if done is not None:
            if verbose:
                print(f"\n[done] {done}")
            return done

        # append the model's reasoning to the convo so it remembers what it said
        convo += gen

        if not cmds:
            # model didn't emit a command and didn't say <done> -> nudge it
            convo += "\n(user: you must emit a <cmd>...</cmd> to act, or <done> to finish.)\n"
            continue

        # run each command, append output back into the conversation
        for c in cmds:
            if verbose:
                print(f"\n$ {c}")
            r = run_cmd(c)
            # format the feedback so the model can parse it
            feedback = f"\n<output rc={r['rc']}{' TIMEOUT' if r['timed_out'] else ''}>\n{r['combined_truncated']}\n</output>\n"
            convo += feedback
            if verbose:
                print(feedback.strip())

    if verbose:
        print("\n[agent] step cap reached without <done>")
    return None


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "list the files in the current dir and count how many .py files there are"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "model.pt"
    run(task, ckpt_path=ckpt, max_steps=6)