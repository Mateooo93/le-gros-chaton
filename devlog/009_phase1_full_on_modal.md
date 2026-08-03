# Devlog #9 — finally training, and now with checkpoints

so last time i was stuck debugging phase 1/2 on kaggle. well, still debugging lol, but we actually got somewhere.

the first issue was transformers not knowing what the qwen3.5 model was — had to pin transformers 5.14.1. then tokenizers kept fighting with it (0.22.0 wants old huggingface-hub, 0.22.1 is fine). then torch on kaggle kept giving a p100 which modern pytorch just refuses to run on, so we had to tell it to give us a t4 instead.

then the fun one: out of memory on the very first step. batch 2 was too big for the gpu they gave us, so batch 1 it is.

also i learned the hard way that kaggle kills your session after 12 hours. the first long run got cancelled at 80% done and i had nothing saved. so now the training script saves a checkpoint every 20% + every hour and uploads it to huggingface, so if it dies we just resume. it actually saved us — the run got cut and we picked right back up from the checkpoint.

right now the full 160k sft is running on modal (paying for gpus now, kaggle time ran out). it's at like 9% and should take a couple days. we'll see how it goes, and then rlvr after that.

the end of the errors... might actually be in sight this time.
