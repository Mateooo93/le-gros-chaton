# 017 — MI300X vLLM serving

So the old GPU box got destroyed and I had to bring up a new one. Same
SSH key, same ROCm stack already installed, just a fresh IP. I cloned
the repo, scp'd over the .env file, ran the setup script. torch 2.5.1
with ROCm 6.2 came through clean and the bf16 matmul sanity test
passed, so the box was good to go.

Getting vllm to serve the merged Qwen3.5 model was the hard part. The
pip vllm 0.27 wheel is CUDA-only and doesn't have the ROCm build, so
that was a dead end right away. The rocm/vllm-dev docker image has the
right ROCm support but vllm 0.16rc2 in there only knows the
multimodal version of Qwen3.5, and our merged model is text-only, so
it kept complaining about missing vision_config.

I tried renaming the merged config's arch field to match what vllm
expected but that just kicked me to a different error about page sizes
not lining up between the layers. The docker's weight loader was
expecting nested keys like model.language_model.X because the
multimodal class wraps everything in language_model, but our merged
safetensors was written by AutoModelForCausalLM which loads the
multimodal class by default and produces those nested keys. Even when
the arch matched, the keys didn't.

So I ended up patching the docker image in place. Five patches: I
added Qwen3_5ForCausalLM to the registry, made the base class inherit
IsHybrid (without that vllm's KV-cache layout fails on hybrid
models), added the mamba state shape methods, and most importantly
threw a WeightsMapper on the text-only load_weights that strips both
model.language_model. and language_model. prefixes. Once that was in
place, the model loaded and served.

The merged config also needed two edits before vllm would accept it.
I stripped mrope_interleaved and mrope_section from rope_parameters
(vllm asserts M-RoPE isn't implemented in text-only mode) and dropped
the dtype field since vllm uses --dtype from the CLI. Both fixes are
scripted in scripts/patch_merged_config.py so they're idempotent.

Once it was all running I tried a quick eval with one task to make
sure the whole pipeline worked end to end. fix-git failed, the model
got stuck on a git show loop and ran out of turns, but the pipeline
itself was solid — Harbor sandbox, the agent loop, vLLM completions,
verifier, all wired up correctly.

## Done, shipped

Training is done and the merged model is on Hugging Face as a public
release at mateo0093/le-gros-chaton, Apache-2.0. I copied the files
server-side with HF's api.copy_files so nothing had to sit on my
laptop's disk. README card has the license tag, base model link,
benchmark table, and a citation block.

For the TB-2.0 5×5 pilot, the model landed at 3/25. The only task it
handled was fix-git (3/5) because the trajectory SFT data leaned heavily
on git orchestration patterns. Everything else needs better data
covering the actual failure modes — multi-file synthesis, side-effect
edits, input discovery. That's the obvious next step if anyone picks
this up.

I tried an RLVR probe with GRPO plus a novelty bonus on the 19 bug
templates but the signal was weak. The novelty bonus made every
rollout score around 1.0-1.3 with almost no variance between them, so
the GRPO advantages were basically zero and the loss stayed near 0
across most steps. The step-10 adapter is uploaded as
mateo0093/le-gros-chaton-qwen-rlvr-step10 for the record but I didn't
merge it into the release.

GPU usage came out to about 12h out of the 50h budget. The vLLM
patches and the eval harness tweaks are all in the repo.

Couple things worth remembering: the Qwen3.5 hybrid model is
multimodal-by-default in both vllm and transformers, and that gives
you nested safetensors keys and M-RoPE config that you have to undo
for text-only serving. The pip vllm wheel is CUDA-only so don't waste
time on it for ROCm, use the docker. fail2ban on the box rate-limits
SSH so wait 30s between commands when you're iterating.