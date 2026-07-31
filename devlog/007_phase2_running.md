# Devlog #7 — phase 2 running

phase 1 (sft) is done and phase 2 (rlvr) is running now on kaggle. this is the part where the model actually gets trained on solving coding problems with our verifier giving rewards, not just imitating fable data.

honestly this is the part i care about more. sft just makes it mimic. rlvr is where it learns to actually fix code. grpo with proportional rewards = the model gets partial credit for getting some tests passing, which is way better than all-or-nothing.

while it runs im just waiting tbh. not much to build. the pipeline works, the eval works, the agent works.

next after this: full benchmark run. humaneval, swebench, agentic eval. we'll see if all this actually made the model good at coding or if i just spent a week on a worse qwen lol.
