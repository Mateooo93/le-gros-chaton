# Chaton agent harness

Turn a base LM into a terminal coding agent. **The loop, not the weights, is
where most of a terminal-bench score comes from.**

## Run it

```bash
# task as first arg, checkpoint as second
python -m agent.loop "list the .py files in this dir and count them" model.pt
```

It loads the model, prints the conversation step by step, runs commands in the
sandbox, and ends when the model emits `<done>...</done>` or hits the step cap.

## How the loop works

```
 task + SYSTEM prompt
        │
        ▼
 ┌─→ generate text ──────────────┐
 │   │                          │
 │   parse <cmd> / <done>        │
 │   │            │              │
 │   done? ─yes→ return answer  │
 │   │                          │
 │   run <cmd> in sandbox ───────┤   ← stdout/stderr appended to convo
 │   │                          │
 └── append output, repeat ─────┘
        (stop at max_steps)
```

## Files

- `sandbox.py` — runs a command safely (timeout + cwd + dangerous-pattern block).
  Returns `{stdout, stderr, combined_truncated, rc, timed_out}`.
- `loop.py` — the loop: load model → generate → parse → exec → feed back.
  `run(task, ckpt_path, max_steps, ...)` is the entrypoint.
- `__init__.py` — makes `agent` a package so `python -m agent.loop` works.

## Why this design

- **Single growing string** as the conversation (re-encoded each step). Simple,
  correct, fine for learning. A production harness would reuse the KV cache to
  decode incrementally (that's what `model.generate(use_cache=True)` is for).
- **`<cmd>` / `<done>` tags** instead of JSON tool-calls: a tiny base model can
  learn to emit tags long before it can emit valid JSON. Start minimal.
- **Sandbox = pattern-block + timeout + cwd**, NOT a real OS sandbox. Good
  enough for a learning agent on your own machine. For untrusted model output,
  run commands inside a throwaway container (firejail/docker) instead.

## TODO after the base model can actually follow the prompt (needs scale)

1. **test-driven self-repair** — prepend the failing `pytest` output to the
   prompt so the model's next turn targets the specific error.
2. **best-of-N** — run the loop N times (temperature sampling), keep the run
   whose final state passes a verifier (`pytest` green). Cheap, big win.
3. **PRM branch selection** — a Process Reward Model scores each step; pick the
   best branch early instead of running all N to completion. This is the
   "le fat chaton" novelty (PRMs for code are underexplored vs math).
4. **KV-cache incremental decode** — swap the re-encode-each-step for
   `use_cache=True` so long conversations stay fast.
5. **eval** — wire to SWE-bench-lite / Terminal-Bench task sets.

## Honest note

A 17M-param Chaton will produce gibberish `<cmd>` tags — it hasn't been trained
on the agent format. This harness is the SHELL that the future "le fat chaton"
(once pretrained on code + fine-tuned on agent traces) will fill with real
ability. Build the harness first, then grow the model into it.