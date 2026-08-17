# Devlog #16 — the t4 memory saga: 22 versions and a nan (16 hours logged)

this devlog is just me fighting a 16GB card. we pushed the trajectory sft to kaggle and then... watched it burn. version after version. it was the longest bug hunt of the project and it's mostly over now.

**the first mistake: the pre-quantized checkpoint.** `techwithsergiu/Qwen3.5-9B-bnb-4bit` has bf16 compute baked into the weights. t4 has no bf16 tensor cores, so it emulates in fp32 — 668 seconds per step. six hundred sixty eight. that's 11 minutes a step. we caught it when a run was "training" at that speed and realized the whole thing was fp32 emulation. swapped to loading the full `Qwen/Qwen3.5-9B` and quantizing on the fly with fp16 compute → 147s/step. 4.5x faster.

**the oom ladder.** every fix revealed the next wall:
- 2048 context: forward alone hit 14.38/14.56 GiB — no room for backward
- the stock loss materializes [seq, 248K] logits in fp32 (~5.7 GiB on this model) — oom
- fp16=True makes accelerate wrap everything in convert_outputs_to_fp32 — oom again
- so we wrote an `AssistantTokenTrainer` that only upcasts the ~700 assistant-token positions, chunked 256 at a time, never materializing the full vocab. that part finally fit.

**then the nan.** the run started fine and then loss printed 5652 with grad_norm nan. it wasn't oom anymore — the gated-delta-net's exp/softmax was overflowing in pure fp16, poisoned the grads, and it would burn the entire 9h session producing garbage. the fix: manual `torch.autocast` around just the forward, promoting the exp/softmax to fp32 while keeping matmuls fp16. no accelerate wrapper, no full-logits upcast. pushed it as v23.

**and there was a session limit trap.** kaggle said "maximum batch gpu session count of 2 reached" and blocked the push — the nan run was still holding a slot even though it was doomed. api has no documented cancel, but we dug through the sdk, found the cancel-session endpoint, extracted the session id via the search api, and killed it. reserved gpu time dropped from 41k seconds to 0.

**v23 worked.** loss 4.03, grad_norm 3.89. finite. first healthy numbers we'd seen in this whole saga. and then, one step later... `cuda error: illegal memory access` inside bitsandbytes' paged_adamw_8bit. the *optimizer* was crashing now. whatever, those are easy: the lora is 43m params, optimizer states are ~350mb, plain adamw_torch is fine. we don't need cpu paging for something that small anyway.

status: v23 proved the forward pass is fixed (finite loss, no nan), v24 with plain adamw is pushed. buy more ram, llama. we'll see you after the merge.
