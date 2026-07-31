# Devlog #6 — training done (phase 1)

finally got the qwen finetune to actually run without ooming. 625 steps, took like 8 hours on kaggle. it finished.

havent run full benchmarks yet cause i was busy building other stuff while it trained. but it trained. thats the main thing.

built a swebench agent that can actually navigate repos and edit files and make patches. not just the basic agent loop that runs commands. this one actually works for real se tasks.

also added swebench eval so we can measure if the model is actually good at fixing bugs or just pretending.

next: run the benchmarks on the trained model and see if its actually any good lol. if sft was enough or if we need the rlvr phase too.

43 python files now. project keeps growing.
