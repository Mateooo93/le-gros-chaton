# Devlog #12 — training kept dying (10 hours logged)

for the last 10 hours the training kept crashing. not the model, not the gpu — the modal client. every hour or so it would lose connection and take the whole run down with it. we'd restart, it'd run an hour, crash again. three times.

the fix was simple: `modal run -d` runs detached, so the training lives on modal's servers and doesn't care if our local process dies. killed the client on purpose to test — training kept going. that's it, that's the fix.

also worried the loss wasn't moving but it was fine. lr had wound down on the crashed run, looked scary, re-warmed fine after restart. loss is 0.668 now and dropping.

status: 72% through the fable5 sft. checkpoints safe on hf. after this: trajectory sft with the self-awareness stuff baked in, then rlvr, then real numbers.
