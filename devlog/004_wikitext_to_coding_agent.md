# Devlog 4

## wait, why am I training on Wikipedia?

**Date:** 2026-07-02

Last week I got le gros chaton (my from-scratch transformer) to actually train. Ran wikitext-103 on a free Colab T4, 12k steps, val loss hit **3.73**. Felt insane for something I wrote myself. I was hyped.

Then I looked at what it generated. Fine at English-ish. Knew about Singapore (wiki articles). But code? Total garbage. Fair enough — *it had never seen a single line of code in its life.*

## the pivot

I thought about what I actually want this thing to **do**: a coding agent. Something that mogs on terminal-bench, that you give a task and it runs commands, fixes its own bugs, gets it done. Qwen-coder / deepseek-coder territory.

Wikitext is not getting me there. You can't learn Python from Wikipedia articles about *python the snake*.

So the project became **le fat chaton**, the bigger sibling. And it's going to be a *coder*, not a chatbot.

## the architecture rabbit hole

I read what frontier coding models actually do. They're basically all **Mixture-of-Experts (MoE)** now:

- Qwen3-coder: 480B params, 35B active per token
- Deepseek-coder-v2-lite: 16B, 2.4B active

The trick: **big total params (knows a lot), small active params per token (runs fast and cheap).** Runs like a small model, knows like a big one. Exactly what I want for a snappy terminal agent.

So I rewrote the model:

- `MoE` class — gate routes each token to its top-2 of 8 experts, plus a load-balance loss so it doesn't pick the same expert every time (Switch Transformer trick)
- **SwiGLU** — gated MLP, basically free quality, every Llama uses it
- **Grouped-query attention** — smaller KV cache = faster long-context decoding (the agent harness needs this)
- **Shared expert** — one always-on expert for common knowledge, rest specialize (Deepseek style)

Full fat config on paper: **~10.25B total, ~3.65B active.** Not building it locally though. It would crash my PC again. (I tried once. It was bad.)

## the data question

For a coder I need, you know, **code.** Options I looked at:

- `smollm-corpus python-edu` — 7.6M educational Python files, best quality, but slow to pull from Software Heritage's S3
- `starcoderdata python` — ~50GB Python, streams fast from HF, what Code Llama lineage trained on. **The workhorse.**
- `the-stack-v2` — 900B tokens. Overkill for my budget lol

Going with starcoderdata Python, blended with ~15-20% cosmopedia (prose) so the model has general knowledge and doesn't become a weirdo that only knows code. That blend is literally the Qwen-coder recipe. General knowledge goes in via the *corpus mix*, not via warm-starting from a prose model.

Speaking of which — **no, I'm not warm-starting from the wikitext model.** Different architecture (17M dense vs 10B MoE, the weights don't fit into each other). And even if they did, warm-starting a coder from prose wastes compute *unlearning* the prose. Fresh random init on the mixed corpus. Clean slate.

## where the wikitext run fits

So the wikitext runs (this one and the smol-fat proof) are **throwaway.** Not the real model. Their job is to prove the *pipeline*:

1. Does the MoE train without crashing?
2. Does the checkpoint reach HuggingFace Hub?
3. If the VM dies (it WILL, free Colab dies after ~90min), can I pull the checkpoint back and resume from the exact step, optimizer momentum and all?

That's what I'm actually testing. Once it's proven I throw the weights away and start the real fat coder fresh. No real progress lost because there was no real progress to lose. It was a stress test.

## honestly

This went from "lets make a little LM that talks" to "lets build a coding agent that mogs the big labs" real fast. Ambitious for a solo student with a 2070 and $30 of Modal credits? Absolutely. Doing it anyway? Yeah.

Next: the eval harness, so I can measure if any of this is actually making the model better at code (pass@1, pass@5 on HumanEval). Val loss going down means nothing if the model still can't write a working function.

— mateo
