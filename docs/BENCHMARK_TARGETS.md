# Terminal-Bench 2.0 — Baseline and Targets

## Baseline: Qwen3.5-9B

Published score: **9.2% ± 2.4** on Terminal-Bench 2.0
(Ranked #138 on public leaderboard, 2026-05-14)

Reference points from the July 2026 leaderboard:
| Model | Score |
|-------|-------|
| GPT-5.5 | 82.7% |
| GLM-5.2 | ~81% |
| DeepSeek-V4 (754B/40B) | 63.5% |
| Qwen3.6-35B-A3B | 24.6% |
| **Qwen3.5-9B (baseline)** | **9.2%** |

## Our Goal

Beat the 9.2% baseline using a **local 9B model + harness stack** —
no bigger model, no cloud API at inference time.

## How we expect to gain

| Lever | Mechanism | Expected gain |
|-------|-----------|---------------|
| SFT on Fable5 (160K rows) | Learn agentic tool patterns | 9.2% → 15-20% |
| GRPO with proportional rewards | Verifier-guided improvement | +3-5% |
| Reasoning distillation | Think like bigger models | +2-4% |
| Test-time scaling (vote_solutions) | Best-of-N | +30-50% relative |
| TDD agent loop | Verify-then-fix discipline | +2-3% |
| Tool-call format SFT | Less parsing failures | +1-2% |

## Realistic ceiling for 9B + harness

~25-35% on Terminal-Bench 2.0 (comparable to Qwen3.6-35B-A3B's 24.6%)
would be a huge win for a local 9B — beating a model 4× its size.

## Measurement

Track progress with:
```bash
python benchmark_tracker.py --trend humaneval
python eval_qwen.py --model Qwen/Qwen3.5-9B --ckpt qwen_coding_agent --record
python eval_swebench.py --model Qwen/Qwen3.5-9B --ckpt qwen_coding_agent --limit 50
```
