"""Qwen-compatible agent loop — drop-in replacement for agent/loop.py.

Uses a HuggingFace Qwen model instead of our custom GPT, with the same
sandbox execution, tool-call parsing, and verifier integration.

Usage:
    python -m agent_qwen --model Qwen/Qwen3.5-9B "list .py files"
    python -m agent_qwen --model Qwen/Qwen3.5-9B --ckpt ./lora_adapter
"""
import argparse
import os
import re
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from agent.sandbox import run_cmd

SYSTEM = """\
You are a terminal coding agent. You solve tasks by running shell commands.

To act, emit a command inside <cmd> tags, e.g.:
  <cmd>ls -la</cmd>
The user will run it and append the output.  You may emit multiple <cmd> blocks.

Each <cmd> must be a shell command on one line (no newlines inside).
Reason briefly between commands.  When the task is complete, emit a final
answer inside <done> tags:
  <done>The fix was applied: changed X to Y in foo.py. Tests pass.</done>
"""

_CMD_RE = re.compile(r"<cmd>(.*?)</cmd>", re.DOTALL)
_DONE_RE = re.compile(r"<done>(.*?)</done>", re.DOTALL)


def load_qwen(model_name: str, ckpt_path: str | None = None, use_4bit: bool = False):
    """Load Qwen model from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path or model_name,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, conversation: str, max_new: int = 200,
             temperature: float = 0.7) -> str:
    """Generate a response given the conversation history."""
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            temperature=temperature, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run(task: str, model_name: str = "Qwen/Qwen3.5-9B",
        ckpt_path: str | None = None, max_steps: int = 10,
        max_new_tokens: int = 200, temperature: float = 0.7,
        verbose: bool = True, use_4bit: bool = False,
        tdd: bool = False) -> str | None:
    """Run the agent loop with a Qwen model.

    Returns the final <done> message, or None if max steps reached.
    """
    model, tokenizer = load_qwen(model_name, ckpt_path, use_4bit)

    # Build conversation (TDD variant adds test-first discipline)
    system = SYSTEM
    if tdd:
        system += "\n\nWork test-first: write a failing test, fix code, verify it passes."
    conversation = f"{system}\n\nTask: {task}\n\n"
    timeout = 30.0

    for step in range(max_steps):
        # Generate
        gen_text = generate(model, tokenizer, conversation, max_new_tokens, temperature)

        # Check if the model generated a response after the prompt
        # (the generate function returns only NEW tokens)
        conversation += gen_text

        # Parse actions
        cmds = [m.group(1).strip() for m in _CMD_RE.finditer(gen_text)]
        done_m = _DONE_RE.search(gen_text)
        done_text = done_m.group(1).strip() if done_m else None

        if verbose:
            print(f"\n--- Step {step + 1} ---")
            print(f"Model: {gen_text[:300]}")

        # Execute commands
        for c in cmds:
            if c.strip() in ("rm -rf /", "shutdown", "reboot", "mkfs",
                             "dd if=/dev/zero", ":(){ :|:& };:"):
                result = "[BLOCKED] dangerous command"
            else:
                r = run_cmd(c, timeout=timeout)
                result = r.combined_truncated[:1000]

            if verbose:
                print(f"$ {c}")
                print(f"  {result[:200]}")

            conversation += f"\n$ {c}\n{result}\n"

        # Check if done
        if done_text:
            if verbose:
                print(f"\n✅ Task complete: {done_text}")
            return done_text

        # Truncate if conversation gets too long
        if len(conversation) > 16000:
            conversation = conversation[-12000:]

    if verbose:
        print(f"\n⚠ Max steps ({max_steps}) reached without completion.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Qwen agent loop")
    parser.add_argument("task", nargs="?", default="list the .py files and describe each",
                        help="Task description")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--tdd", action="store_true",
                        help="TDD loop: test first, fix, verify")
    args = parser.parse_args()

    result = run(
        task=args.task,
        model_name=args.model,
        ckpt_path=args.ckpt,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verbose=not args.quiet,
        tdd=args.tdd,
    )

    if not args.quiet:
        print(f"\n{'='*40}")
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
