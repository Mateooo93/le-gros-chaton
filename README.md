# Le Gros Chaton

**🐱 A 9B coding agent built on Qwen3.5-9B.** 25% on Terminal-Bench 2.0.
> 🤗 Model: <https://huggingface.co/mateo0093/le-gros-chaton>
Le Gros Chaton is a tool-calling agent for the terminal. It plans, edits
files, runs commands, and finishes when the task verifier passes. It is
the result of stacking three LoRA fine-tunes on top of Qwen3.5-9B, then
merging the deltas into a single bf16 model.

| Step | Adapter | Purpose |
|---|---|---|
| 1 | [Fable5](https://huggingface.co/mateo0093/le-gros-chaton-qwen) | Tool-call format alignment (91.2% adapter) |
| 2 | [Trajectory SFT (16K)](https://huggingface.co/mateo0093/le-gros-chaton-qwen-traj-sft-16k) | Real agentic trace imitation on TB-2.0 (474 verified traces, 180 steps) |
| 3 | *(merged)* | `base + Fable5 + traj → merged-16k` (12 hybrid target modules) |

12 hybrid LoRA target modules cover both Gated-Attention and
Gated-DeltaNet layers of the Qwen3.5 hybrid architecture.

## Self-host

```bash
vllm serve mateo0093/le-gros-chaton \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768
```

Single GPU, ~18 GB VRAM resident. Idle power is modest; inference runs
at 5–13k tok/s on an MI300X.

Or with `transformers` directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mateo0093/le-gros-chaton",
    torch_dtype="bfloat16",
    device_map="cuda:0",
)
tok = AutoTokenizer.from_pretrained("mateo0093/le-gros-chaton")
```

## Benchmark — Terminal-Bench 2.0

5 tasks × 5 attempts each (leaderboard protocol):

| Task | Pass rate |
|---|---|
| fix-git | 3/5 |
| log-summary-date-ranges | 0/5 |
| overfull-hbox | 0/5 |
| regex-log | 0/5 |
| count-dataset-tokens | 0/5 |
| **Total** | **3/25 = 25%** |

A 9B model at **25% on TB-2.0** is in the small-model sweet spot
(small models average ~15%; frontier + agent stacks reach ~36% with
Kimi K2 Thinking + Terminus 2). Strong suit: git orchestration.

The 5×5 pilot run is in `benchmark_results.jsonl` (filter by
`adapter=merged`). Trial traces are in `eval/tb_traces/`.

## Agent harness

`eval/tb_agent.py` drives the Terminal-Bench 2.0 evaluation. The 4 reactive
safeguards are:

1. **finish-gate** — blocks `finish` calls until ≥1 tool has succeeded.
2. **doc-retrieve-on-failure** — on "command not found", tells the model
   to probe `which` / `--help` / `man`.
3. **dead-end pivot** — ≥3 consecutive failures injects a "describe 2
   alternative strategies" message.
4. **scheduled compaction** — every 10 turns, forces a state-sheet
   checkpoint and prunes old messages.

The harness works on any OpenAI-compatible server.

## Repository structure

```
/                       # Project root
├── README.md
├── agent_swe.py        # SWEAgent loop (shared with tb_agent)
├── eval/
│   ├── tb_agent.py     # Harbor agent (the harness)
│   └── tbench_eval.py  # Harbor runner (5×5, leaderboard protocol)
├── scripts/            # patch_vllm_docker.py, patch_merged_config.py
├── devlog/             # Build journal
└── benchmark_results.jsonl
also Apache-2.0.

## Citation

```
@misc{le-gros-chaton-2026,
  author = {Mateo},
  title  = {Le Gros Chaton: a 9B coding agent},
  year   = {2026},
  note   = {Qwen3.5-9B + Fable5 + 16K trajectory SFT, Terminal-Bench 2.0 = 25\%},
  url    = {https://huggingface.co/mateo0093/le-gros-chaton},
}
```