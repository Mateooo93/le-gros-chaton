# Project Status — Le Gros Chaton

## Goal
A local (9B) coding agent that rivals big models on Terminal-Bench, SWE-bench,
and HumanEval. Baseline: Qwen3.5-9B = 9.2% Terminal-Bench 2.0. Target: 25-35%.

## Status (28 iterations)
All 55 Python files syntax-clean, 40/40 unit tests pass.
Full 12-layer stack built and committed to GitHub.

## Stack summary
- Train: `train_qwen.py` (SFT Fable5 + GRPO)
- Data: filter, dedup, reasoning distillation, self-play, agent traces
- Agent: `agent_swe.py --tdd` + recovery + context mgmt + tool cap + traces
- Inference: `vote_solutions.py` (verifier + LLM-judge), `serve_qwen.py`
- Eval: `eval_qwen.py --all-evals`, `eval_swebench.py`, `eval_toolcalls.py`
- Track: `benchmark_tracker.py` (JSONL + CSV + trends)

## Current activity
Kaggle training running (Qwen3.5-9B, Fable5 SFT). User's Modal budget: $30.
Research ongoing: SLM survey, Harness-Bench, GLM-5.2, SSR, DeepSWE, SERA.

## Commands
```bash
python train_qwen.py --sft-only --limit 10000     # train
python eval_qwen.py --ckpt qwen_coding_agent --all-evals  # measure
python benchmark_tracker.py --trend humaneval     # track
python serve_qwen.py --ckpt qwen_coding_agent --serve  # deploy
```

## Security notes
- Secrets live in gpus.md (gitignored, never committed)
- History scrubbed of leaked tokens (filter-branch + force push)
- Root-level notebook copies gitignored to prevent token re-leak
