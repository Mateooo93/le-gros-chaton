# Devlog #16 — the t4 war, and then the mi300x (16 hours logged)

the t4 (16GB) fought us the whole way: bf16 baked into a pre-quantized checkpoint made steps 668s (fp32 emulation), 2K context oom'd, the [seq, 248K] logits oom'd, fp16 overflowed into nan, kaggle's session cap locked us out, and the paged optimizer crashed last. every fix was another wall. we won narrowly with fp16 compute + a chunked assistant-token loss + autocast + plain adamw.

then we got the **AMD MI300X** (192GB). no quantization, no autocast, no hacks — and 16K context, which the t4 could never fit. every trace now trains whole instead of chopped mid-message.

status: 16K bf16 trajectory SFT running on the mi300x. next: merge, terminal-bench eval, rlvr.
