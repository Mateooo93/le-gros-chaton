# Devlog #16 — the t4 memory saga ends, enter the MI300X (16 hours logged)

this devlog is mostly me fighting a 16GB card, version after version. it was the longest bug hunt of the project and it's finally over — because we got an **AMD Instinct MI300X** (192GB) and the whole memory war just... stopped mattering.

**the slow start.** the pre-quantized checkpoint (`techwithsergiu/Qwen3.5-9B-bnb-4bit`) has bf16 compute baked in; t4 has no bf16 tensor cores so it emulates in fp32 — 668s/step, 11 minutes a step. swapping to the full base with on-the-fly fp16 quant → 147s/step. 4.5x faster.

**the oom ladder.** ctx 2048 blew the forward, the stock loss materializes [seq, 248K] in fp32 (~5.7GB), fp16=True wraps everything in fp32 upcasts. the fix was a custom `AssistantTokenTrainer` that upcasts only the ~700 assistant positions, chunked 256 at a time.

**then the nan.** loss printed 5652 with grad_norm nan — the gated-delta-net's exp/softmax was overflowing in pure fp16. manual `torch.autocast` around just the forward fixed it.

**the session-limit trap.** kaggle blocked new pushes with the nan run holding a slot. undocumented, but we found the cancel-session endpoint in the sdk, pulled the session id via the search api, and killed it.

**v23 worked** (loss 4.03, finite) and then the bitsandbytes paged optimizer crashed with illegal memory access. whatever: 43m trainable params means ~350mb optimizer states — plain `adamw_torch` is fine.

**and then we got the amd gpu.** 192GB of bf16 on the MI300X. no quantization, no autocast, no fp16 hacks — and the real kicker: we can train at **16K context** now, which was impossible on the t4. the kaggle run only fit 1.5K tokens so most long traces got chopped mid-message; the model never saw how tasks END. on the mi300x every trace fits whole.

status: 16K trajectory sft is running on the amd gpu right now (native bf16, ~5h). after that: merge, eval on terminal-bench, rlvr. the memory war is over — now we fight for benchmarks.
