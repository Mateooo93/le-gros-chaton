# Devlog #5 — ok so I'm not building a 10B MoE anymore

So last time I had this whole plan. 10B MoE, from scratch, SwiGLU, GQA, all that. And I actually built it. Like 60 Python files, tests passing, RL pipeline, self-play data gen, the whole thing. It was kinda sick ngl.

But then I was like... wait. I'm building an ENGINE that TRAINS coding agents. Not the agent. And the engine is cool but training a 10B model from zero on $30 of Modal credits is not how you beat GLM-5.2.

You know what beats GLM-5.2? Fine-tuning a model that already knows how to code.

## the pivot (again)

Qwen3.5-9B came out. 9B params. Fits on a Kaggle T4 with 4-bit QLoRA. And Kaggle gives me 30 HOURS of free GPU. 2× T4. That's literally free training.

So I had to adapt the entire codebase to work with HuggingFace models instead of my custom GPT. Made like 5 new files. All the verifier/agent/reward stuff I built for the MoE? Now works with Qwen too. Didn't have to rewrite everything — just slapped a Qwen adapter on top.

## the dataset

Found this dataset on HF: `Nexlab/fable5-agentic-coding-sft`. 160K rows of Claude Fable 5 output. Like actual coding traces — building games, fixing bugs, writing shell scripts. Not just "write a function that passes a test." Real agentic stuff.

Training right now actually. Qwen3.5-9B on 10K rows. Should take like 8 hours on 2× T4. We'll see if it doesn't OOM again lol.

## research rabbit hole (couldn't help it)

While waiting for training I read papers:
- Routing-Free MoE — what if experts just decide for themselves when to activate? No router, no top-k. Already implemented it in model.py.
- GLM-5.2 has this IndexShare thing that reuses computation across layers. 2.9× cheaper at 1M context. Maybe later.
- DeepSWE trained Qwen3-32B with RL only and got 42% SWE-Bench. So our approach is valid.

## current vibe

- Training running on Kaggle (hopefully not crashing)
- SWE-bench evaluator ready to go
- Weights auto-upload to HF when done
- If it works: cool. If it OOMs again: we fight

The from-scratch MoE is still there. It's like a research playground now. The real path is just fine-tuning Qwen and seeing how good we can get it.

one command to train. one to eval. one to run the agent. that's the dream.
