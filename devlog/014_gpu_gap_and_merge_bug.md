# Devlog #14 — worked a lot, still not done

we worked a lot this week. honestly the main takeaway is that the fable5 sft was good but its traces alone weren't enough — the model needs way more verified agent trajectories, so we've been grinding those out (target is 450).

we also ran out of credits mid-generation (the modal endpoint started failing with 503s) and had to switch to a second kimi account. took a minute to get the auth right — the new endpoint wanted an extra header we didn't have. after that the generator resumed fine and it's been adding verified traces since (~340/450 now).

biggest scare: we found out peft 0.20 silently drops the second adapter when you merge two at once. tested it and yeah, the outputs were identical with one or two adapters — so our trajectory sft would have been lost in the merge. fixed it by merging sequentially (base first, then the second adapter). checked against the live model and it matches, so we're good.

my 2070 couldn't fit the 9b model for real training, so we're waiting on a 4090 (my dad's) next week. all the scripts are ready — sft, merge, rlvr, eval, even a setup script for the gpu box. just need the hardware.

status: 340/450 traces, everything ready, waiting on the 4090. we've worked a lot, but the real run hasn't happened yet :/.