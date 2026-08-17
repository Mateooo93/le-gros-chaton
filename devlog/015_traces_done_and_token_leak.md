# Devlog #15 — traces are done, and we almost leaked the token (16 hours logged)

big week. the trace generator finally hit the target and then some. 474 verified trajectories, zero malformed, zero duplicates, 19 templates, 95% pass self-review. we were aiming for 450 and overshot. the dataset is synced to huggingface in two flavors (raw + normalized) and that grind is officially over. it was the longest stretch of this whole project and honestly the model is only going to be as good as these traces, so... worth it.

**the disk almost killed the generator.** our home drive hit 100% full — huggingface had cached the 4-bit model (8GB) plus the fable5 dataset and it just ate everything. generators need scratch space to write, so this was genuinely blocking. deleted both caches, verified nothing was using them (checked lsof/pgrep, they were just sitting there), disk went from 129MB free to 8.7GB. phew.

**and then we found a real leak.** the huggingface token was hardcoded in a shell script and it got committed. like, actually committed. if we'd pushed that, anyone could have yanked it and burned our HF account. caught it before it ever reached github (verified the remote was clean, zero occurrences), then rewrote the unpushed history with filter-branch, expired the reflogs, gc'd, and re-verified zero occurrences everywhere. token now lives in a gitignored .env with tight permissions. never again.

**the kimi endpoint situation.** the old teacher account's endpoint started 503'ing mid-generation and died permanently. had to switch to a second kimi account — which wanted an extra header we didn't have. took a minute to figure out, but the generator resumed without losing progress.

**gpu timeline.** the 2070 can't really train the 9B (we knew this), kaggle hours are back (30 of them, t4), and the 4090 from my dad lands next week. the plan: kaggle t4 for trajectory sft now, 4090 for merging + terminal-bench eval + rlvr.

status: 474/474 traces done, disk clean, token safe, kaggle phase wired up and pushing. the llama is being fed.
