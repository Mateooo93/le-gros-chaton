# Devlog #14 — the GPU gap, a silent merge bug, and a 4090 on the horizon

big week for logistics. we ran out of money on the teacher account mid-generation (the modal endpoint just started 503-ing every request), and santa came through — my dad's 4090 (24GB!) is ours next week. that changes everything: the local 2070 proved too small for the 9B (more on that), but 24GB runs the whole remaining pipeline comfortably.

### the trace grind, account edition

so we were at ~335 verified traces of 450 when the og kimi account died. the generator kept retrying into the void, supervisor restarted it in a loop, and the count froze. easy diagnose: test the endpoint with curl, 503, account's toast.

new account landed with 30 fresh dollars. dropped the new key in and... 401. a `webhook token not found` that made zero sense at first. turns out the new modal endpoint wants an extra header (`X-Webhook-Token`) alongside authorization — or at least the first requests need it. once we sent both, 200, kimi talking again.

the supervisor picked up where it left off (crash-safe since day one, thank god), and the count is climbing again: 339/450 as i write this.

### the merge bug that would've been a disaster

the scariest find of the week: `merge_and_unload(adapter_names=["fable", "traj"])` in peft 0.20 silently merges ONLY THE FIRST ADAPTER. i proved it on a tiny model — `merge[fable]` and `merge[fable, traj]` output the EXACT same logits (0.000e+00 diff). the trajectory SFT trained a fresh LoRA on top of the fable adapter, and the "merge everything" step would have just... dropped the entire trajectory SFT. shipped fable-only, called it a trajectory model. no error, no warning — just quietly garbage.

the fix is sequential merge: base+fable → merged1, then merged1+traj → final. validated with the exact same tiny model: sequential merge output vs the live stacked model is 1.6e-7 apart (that's float noise). exact match. and it's committed so the 4090 never sees the broken path.

### the 8GB experiment (and why we're done with it)

the local 2070 smoke test on the 9B was a whole saga: device_map=auto fails at 8GB, fp16 lm_head OOMs at 3.79GB, quantized lm_head asserts in bitsandbytes, meta-tensors from the cached config. we got it loading and even ran forward through 22/24 layers before hitting the hardware ceiling. conclusion: 8GB genuinely can't train 9B+LoRA, full stop. so the real training always needed the 4090. now we have it.

### what we built while waiting

- **rlvr_qwen.py** — RLVR ported off the toy nanoGPT to the real Qwen 9B: agent rollouts via the swe harness, hidden-test verifier, diversity + novelty + strategy-switch rewards (creativity stays). dry-run verified: problems load, rewards compute.
- **merge_sft.py** — with the sequential fix above, one command to fuse base+fable+traj into a single full checkpoint. uploads it too.
- **run_sft_pipeline.sh** — the whole thing: sync traces → trajectory SFT → merge → TB 2.0 eval → RLVR. env-driven, one command on any GPU box.
- **setup_4090.sh** — bootstraps a fresh ubuntu+gpu box: cuda stack, docker/harbor (TB2 eval sandboxes need it), venv, env file, gpu check.
- supervised-gen now auto-syncs raw + normalized traces to HF on completion. traces normalized to 100% canonical backtick format (11 legacy `fat-cat` style messages converted).
- **audit sweep**: found and fixed the merge bug + an exported-env gap (`OUT_DIR` wasn't exported to the SFT consumer). everything else shipped clean.

### the plan

4090 arrives next week → `bash setup_4090.sh --run` → ~12-18h total: SFT (~2h), merge (~20min), TB 2.0 eval (2-4h), RLVR with diversity (6-12h). then we finally get the real number: is the fat cat ≥30% on terminal-bench? the harness is ready, 79 tasks, pilot verified.

**status**: 339/450 traces, generating on the new $30 hotness; all training code done, committed, and blocker-free except the 4090. 😼