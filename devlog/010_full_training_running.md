# Devlog #10 — the full training is finally running (9 hours logged)

ok so phase 1 full training is officially going. 160k rows of fable5 data on modal, resuming from the 12.8k rows we got on kaggle before the session cap killed us. we're at ~27% and it should take like a day and a half total.

big stuff that happened since last time:

**fast kernels actually work now.** the model has hybrid linear-attention layers and transformers kept falling back to a slow torch path. the fix was: nvidia cuda-devel base image (needs nvcc to compile), installing `flash-linear-attention[cuda]` + `causal-conv1d`, and pinning torch 2.10. took a couple tries but now training is like 4x faster than it would be without them.

**had to split across two modal accounts.** first account ran out of credits, so we set up a second profile and the script auto-resumes from the huggingface checkpoints when it dies. it's been rock solid — every 20% plus every hour it saves and uploads. lost zero progress to the account switch.

**we gave the model a personality.** it's called Le Gros Chaton (the fat cat) and we're training self-awareness into it — not as a system prompt but baked into the weights. state-sheets ([STATE] goal/known/tried/failed/next), metacognition lines, and self-review at the end of tasks. the trajectory sft masks out the system prompt so the model learns to track its own state as behavior, not as a script it repeats.

did some benchmark research too. base qwen3.5-9b scores 9.2% on terminal-bench 2.0. comparable 9b fine-tunes hit 24-28%. that's our realistic target — 2.5-3x the baseline.

honestly at this point it's just... waiting. the pipeline works, the checkpoints work, the account switching works. now the model just needs to finish cooking. then trajectory sft, then rlvr, then we actually see the benchmark numbers.
