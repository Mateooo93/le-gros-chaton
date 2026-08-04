# Devlog #12 — the crash saga, and the fix that ended it

the last 24 hours were a lesson in distributed training reliability. the modal training kept dying. not because of the model, not because of the gpu — because of the client.

**what kept happening:** `modal run` ties the training app's life to the local client process. our local network is flaky, so every ~hour the client would lose its heartbeat (`Deadline exceeded`) or crash with a grpclib bug (`'Connection' object has no attribute '_transport'`), and each time it took the remote training down with it. we'd relaunch, it'd run for 45-60 minutes, crash again. three times in a row. super frustrating because the resume-from-hf-checkpoint part always worked perfectly — the process was the problem, not the pipeline.

**the fix:** `modal run -d` (detached mode). the app runs server-side and keeps going even if the local process dies. we killed the local client on purpose to test it — step kept advancing 3211 → 3220 with zero local process alive. training is now immortal as long as modal's servers are up. should have done this day one.

**loss question that came up:** someone (me) was worried loss wasn't moving. turns out the numbers were misleading — the crashed run's tail showed lr 1.8e-05 (scheduler had wound down), and after the detached relaunch the schedule re-warmed properly. current readings: loss 0.668 and dropping, lr 0.00015, grad_norm 0.29 — healthy.

**state of the fat cat:**
- ~72% through the 160k fable5 sft (step ~4900/6799 of the resumed run)
- checkpoints uploading to hf every ~7h, detached so nothing can kill it
- identity + self-awareness prompt built and committed (state-sheets, metacognition, self-review)
- trajectory sft pipeline ready to go when this finishes
- benchmark research done: base qwen3.5-9b is 9.2% terminal-bench, comparable 9b fine-tunes hit 24-28%

next: let the sft finish (~8h?), then trajectory sft with self-awareness baked into the weights, then rlvr, then we see actual numbers. the fat cat grows fat.
