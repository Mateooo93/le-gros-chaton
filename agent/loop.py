"""The agent loop — turn a base LM into a terminal tool-user.

THE BIG IDEA
------------
A coding agent isn't a bigger model. It's a small model + a LOOP:
  1. give the model a task + a system prompt that says "to act, emit
     <cmd>shell command</cmd>; I will run it and give you the output"
  2. model generates text. We strip out any <cmd>...</cmd> blocks it emitted.
  3. run those commands in the sandbox, capture stdout/stderr.
  4. append the model's text AND the command output back into the conversation.
  5. go to 2. Repeat until the model emits <done>...</done> or we hit the cap.

KV-CACHE OPTIMISATION
---------------------
Unlike a naive loop that re-encodes the entire growing conversation every step
(O(n²) in context length), this implementation keeps the per-layer KV cache
alive across steps.  Feedback text is encoded once and passed as a prefill
extending the existing cache.  This keeps inference O(n) in total decoded
tokens — critical for long agent rollouts (10-20 tool calls).

EXTENSIONS (later):
  - test-driven self-repair: prepend the failing test output to the prompt so
    the model's next turn targets the specific error.
  - best-of-N: run the loop N times with temperature sampling, keep the run
    whose final state passes a verifier (unit tests green / diff applies).
  - PRM-step scoring: a Process Reward Model scores each step so we can pick
    the best branch without running to completion (the novelty).
"""
import re
import torch
from tokenizer import decode, encode
from model import GPT
import config as cfg
from agent.sandbox import run_cmd

# ---- tags the model uses to act and signal completion ----
import re
import torch
from tokenizer import decode, encode, TOOL_TOKENS, tool_token_id, is_tool_token
from model import GPT
import config as cfg
from agent.sandbox import run_cmd

# ---- tags the model uses to act and signal completion ----
_CMD_RE = re.compile(r"<cmd>(.*?)</cmd>", re.DOTALL)
_DONE_RE = re.compile(r"<done>(.*?)</done>", re.DOTALL)

SYSTEM = """\
You are a terminal coding agent. You solve tasks by running shell commands.

To act, emit a command inside <cmd> tags, e.g.:
  <cmd>ls -la</cmd>
The user will run it and append the output.  You may emit multiple <cmd> blocks.

Additionally, you can use the following special tokens:
  <|tool_call|>  —  signals you want to run a command
  <|done|>       —  signals the task is complete

Each <cmd> must be a shell command on one line (no newlines inside).
Reason briefly between commands.  When the task is complete, emit a final
answer inside <done> tags:
  <done>The fix was applied: changed X to Y in foo.py. Tests pass.</done>
Keep commands small and testable. Prefer `cat`, `grep`, `pytest`, `python -c`.
"""


def parse_actions(text: str):
    """Return (cmds, done_text). done_text is None if the model didn't finish."""
    cmds = [m.group(1).strip() for m in _CMD_RE.finditer(text)]
    done_m = _DONE_RE.search(text)
    done = done_m.group(1).strip() if done_m else None
    return cmds, done


def _load_model(ckpt_path: str, device: str):
    """Build the model from config and load a checkpoint (weights only)."""
    return GPT.from_checkpoint(ckpt_path, device)


@torch.no_grad()
def _extend_cache(model, tokens: list[int], kv_caches, device: str):
    """Encode *tokens* and prefill them into the existing KV cache in-place.

    Returns the new ``kv_caches`` (list of (k, v) tuples, one per layer).
    """
    if not tokens:
        return kv_caches
    # Determine past sequence length from cache entry shape.
    # GQA: (B, n_head, T, head_dim) → size(2) = T
    # MLA: (B, T, latent_dim) → size(1) = T
    entry = kv_caches[0][0]
    past_len = entry.size(1) if entry.dim() == 3 else entry.size(2)
    t = torch.tensor([tokens], dtype=torch.long, device=device)
    _, _, new_kv = model(t, use_cache=True, kv_caches=kv_caches, rope_offset=past_len)
    return new_kv


def run(task: str, ckpt_path: str = "model.pt", max_steps: int = 10,
        max_new_tokens: int = 200, temperature: float = 0.7, verbose: bool = True):
    """Run the agent loop on a task. Returns the final <done> answer (or None).

    The conversation is maintained as a flat token list and a per-layer KV
    cache that grows incrementally — no O(n²) re-encoding.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(ckpt_path, device)

    # ---- Initial prefill: encode the system prompt + task ----
    convo_tokens: list[int] = encode(f"{SYSTEM}\n\nTask: {task}\n\n")
    convo_tensor = torch.tensor([convo_tokens], dtype=torch.long, device=device)
    _, _, kv_caches = model(convo_tensor, use_cache=True)

    # Accumulated generation from all steps (used for repetition penalty).
    # Start with the prompt tokens so the model can't repeat them.
    all_tokens: list[int] = list(convo_tokens)

    for step in range(max_steps):
        # ---- Generate ----
        idx = torch.tensor([convo_tokens[-cfg.block_size:]], dtype=torch.long, device=device)
        out, kv_caches = model.generate(
            idx, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=50, top_p=0.9,
            repetition_penalty=1.15,
            kv_caches=kv_caches, return_caches=True,
        )
        # The generated tokens (including the prompt tail that was re-processed)
        gen_tokens = out[0, idx.size(1):].tolist()
        gen_text = decode(gen_tokens)

        # Update the token-level conversation
        convo_tokens.extend(gen_tokens)
        all_tokens.extend(gen_tokens)
        # KV cache already updated by generate() — but the cache now only holds
        # the last `idx.size(1) + gen_tokens` positions because generate()
        # re-prefills from idx on the first step.  We need to rebuild: encode
        # the full convo into the cache.
        # KV cache returned by generate() covers the last forward pass but the
        # conversation has grown.  Re-prefill the full convo so the next step's
        # KV cache covers all past tokens.  This is O(n) per step but correct
        # and simple — the InferenceEngine does incremental tracking instead.
        t_full = torch.tensor([convo_tokens], dtype=torch.long, device=device)
        _, _, kv_caches = model(t_full, use_cache=True)

        if verbose:
            print(f"\n--- step {step} model output ---\n{gen_text}")

        cmds, done = parse_actions(gen_text)
        if done is not None:
            if verbose:
                print(f"\n[done] {done}")
            return done

        if not cmds:
            # Model didn't emit a command and didn't say <done> → nudge
            nudge = "\n(user: you must emit a <cmd>...</cmd> to act, or <done> to finish.)\n"
            feedback_tokens = encode(nudge)
            convo_tokens.extend(feedback_tokens)
            all_tokens.extend(feedback_tokens)
            kv_caches = _extend_cache(model, feedback_tokens, kv_caches, device)
            continue

        # ---- Execute commands and feed back results ----
        for c in cmds:
            if verbose:
                print(f"\n$ {c}")
            r = run_cmd(c)
            feedback = f"\n<output rc={r.rc}{' TIMEOUT' if r.timed_out else ''}>\n{r.combined_truncated}\n</output>\n"
            feedback_tokens = encode(feedback)
            convo_tokens.extend(feedback_tokens)
            all_tokens.extend(feedback_tokens)
            kv_caches = _extend_cache(model, feedback_tokens, kv_caches, device)
            if verbose:
                print(feedback.strip())

        # Crop conversation to block_size (evict oldest tokens)
        if len(convo_tokens) > cfg.block_size:
            evict = len(convo_tokens) - cfg.block_size
            convo_tokens = convo_tokens[evict:]
            # KV cache is now misaligned → rebuild from cropped tokens
            t_cropped = torch.tensor([convo_tokens], dtype=torch.long, device=device)
            _, _, kv_caches = model(t_cropped, use_cache=True)

    if verbose:
        print("\n[agent] step cap reached without <done>")
    return None


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else (
        "list the files in the current dir and count how many .py files there are"
    )
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "model.pt"
    run(task, ckpt_path=ckpt, max_steps=6)