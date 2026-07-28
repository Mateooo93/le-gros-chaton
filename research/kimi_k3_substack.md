[![Agentic AI ](https://substackcdn.com/image/fetch/$s_!rWke!,w_40,h_40,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5395d473-3e5f-4ef3-aca5-241ffc64c82e_505x505.png)](https://kenhuangus.substack.com/)

# [Agentic AI](https://kenhuangus.substack.com/)

SubscribeSign in

# Demystifying Kimi K3: The Three Algorithms Behind the \#1 Frontend Coding Model

[![Ken Huang's avatar](https://substackcdn.com/image/fetch/$s_!gd2H!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d670301-204b-472e-a2ee-bbb1b7633a99_2026x2026.png)](https://substack.com/@kenhuangus)

[Ken Huang](https://substack.com/@kenhuangus)

Jul 18, 2026

∙ Paid

47

1

2

Share

On July 16, 2026, Moonshot AI released Kimi K3, a 2.8-trillion-parameter open-weight sparse Mixture-of-Experts model with a 1-million-token context window. It is the largest open-weight model ever shipped. Within hours it landed at #4 overall on the Artificial Analysis Intelligence Index and, more strikingly, debuted at #1 on LMArena's Frontend Code Arena, beating Claude Fable 5 on the benchmark that most directly measures production coding value.

I want to be precise about what is and is not remarkable here. K3 is not the single best model in the world. Claude Fable 5 and GPT-5.6 Sol still edge it on pure reasoning. What changed is the shape of the frontier. For eighteen months the story was that Chinese labs trailed by six to twelve months and that chip sanctions guaranteed the gap. K3 refutes both halves of that sentence at once. It arrives at the frontier immediately, and it does so as an open model that anyone can download on July 27.

This post is in two halves. The free half is the technical teardown: the three architectural innovations that make K3 work, and a forensic look at every benchmark where it ranks number one, including what each benchmark actually tests and why a first-place finish there means something. The paid half is the part you act on: the geopolitics of a state-adjacent frontier lab, the open-source strategy that K3's specific capabilities unlock, a practical FAQ for deploying it, questions worth sitting with, and my honest read on where this goes.

## Three scaling problems, solved at once

Every large model fights three separate battles. Sequence length wants to blow up attention cost quadratically. Depth wants to dilute the signal from early layers as the stack grows. Width, in a Mixture-of-Experts model, wants to descend into routing chaos as you add experts. Most labs win one or two of these fights. Moonshot's claim, and the benchmarks broadly support it, is that K3 wins all three, for a combined 2.5 times better scaling efficiency than the K2 series.

[![Kimi K3 decomposes into three innovations: KDA for sequence, AttnRes for depth, LatentMoE for width, converging on a 2.8T model with 50-60B active parameters per token.](https://substackcdn.com/image/fetch/$s_!IK-Q!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffabf133f-c3ab-4586-a9bc-7e3103c52803_1600x937.png)](https://substackcdn.com/image/fetch/$s_!IK-Q!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffabf133f-c3ab-4586-a9bc-7e3103c52803_1600x937.png)

The diagram above is the whole model in one frame. The sequence dimension is handled by Kimi Delta Attention. The depth dimension is handled by Attention Residuals. The width dimension is handled by Stable LatentMoE. They are independent innovations that compose, and the ellipse at the bottom is what you get when they stack: a 2.8-trillion-parameter model that only activates 50 to 60 billion parameters for any given token. That sparsity is the reason a model this large is economical to serve at all.

### Kimi Delta Attention: giving every channel its own memory

Standard Transformer attention costs you quadratically in sequence length. At one million tokens, that is a wall. Linear-attention schemes like Mamba get around it, but historically at a real cost to expressiveness. Kimi Delta Attention, which Moonshot introduced in the Kimi Linear paper in October 2025 and scaled up dramatically for K3, takes a subtler route: it refines Gated DeltaNet by changing one thing about how the recurrent state forgets.

[![Standard DeltaNet uses one scalar decay for all dimensions; KDA turns decay into a per-channel diagonal matrix so each dimension forgets independently.](https://substackcdn.com/image/fetch/$s_!kxXk!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F364f9fcd-2a78-4014-b569-1c80311ed0c7_1600x1216.png)](https://substackcdn.com/image/fetch/$s_!kxXk!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F364f9fcd-2a78-4014-b569-1c80311ed0c7_1600x1216.png)

The picture holds the entire idea. On the left, standard DeltaNet multiplies the previous state by a single scalar decay factor. Every hidden dimension forgets at the same rate, which is a blunt instrument. On the right, KDA replaces that scalar with a vector, turned into a diagonal matrix, so each dimension in the recurrent state gets its own independent forgetting rate. In notation, the update goes from multiplying the prior state by a scalar to multiplying it by Diag of a vector. That is a small change on paper and a large one in practice: a channel holding a critical long-range fact can hold it, while a channel tracking something local can flush and reuse itself.

Moonshot built this on a specialized Diagonal-Plus-Low-Rank transition matrix and open-sourced the FlashKDA kernels, which hit a 1.72 to 2.22 times prefill speedup on H20 GPUs. The measured payoff from the Kimi Linear prototype is the headline: up to 6.3 times faster decoding at 1M context, up to a 75 percent reduction in KV cache, and quality that matched or beat full Multi-Head Latent Attention under fair comparison. This is the machinery that makes a 1M context window affordable rather than theoretical.

### Attention Residuals: letting layers choose what to remember

Depth has a quieter failure mode. The standard residual connection adds each layer's output to a running sum, which means that as the stack grows past a hundred layers, hidden states balloon, gradients spread unevenly, and the contribution of early layers gets diluted into noise. Moonshot's Attention Residuals work, published March 2026, treats this as a retrieval problem instead of an accumulation problem.

[![Standard residuals accumulate every layer uniformly; AttnRes lets each layer attend selectively over earlier layers with learned weights.](https://substackcdn.com/image/fetch/$s_!qbo3!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5bcad7b5-6929-46a7-9def-2f49aa4d2e91_1600x1534.png)](https://substackcdn.com/image/fetch/$s_!qbo3!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5bcad7b5-6929-46a7-9def-2f49aa4d2e91_1600x1534.png)

On the left of this figure, information marches up the stack with a fixed skip connection at every step, no discrimination about what matters. On the right, each layer acts like a query that attends back over all earlier layers, pulling in exactly the representations it needs with learned weights. The arrows carry different alpha values because Layer 3 might want sixty percent of Layer 1 and only ten percent of Layer 0. The full version costs too much memory at scale, so K3 uses Block AttnRes: ordinary residuals inside a block, attention over depth only between blocks, which drops the cost from order Ld to order Nd. On the 48B Kimi Linear model this bought a 7.5-point jump on GPQA-Diamond and matched a baseline trained with 1.25 times more compute. This is how K3 actually benefits from its enormous depth instead of drowning in it.

### Stable LatentMoE: 896 experts, 16 awake

Width is the flashiest number and the easiest to get wrong. K3 has 896 experts and activates just 16 of them per token, a sparsity of about 98.2 percent. The hard part of extreme sparsity is not the arithmetic, it is keeping routing stable and making sure experts are actually used rather than a handful hogging every token.

[![A token is scored by the router; the top 2 percent quantile routes to 16 active experts while 880 stay dormant, using quantile balancing, per-head Muon, and SiTU activation.](https://substackcdn.com/image/fetch/$s_!xOxm!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdfecf2ee-c21d-4eb9-bff8-923ff01031ff_1600x2173.png)](https://substackcdn.com/image/fetch/$s_!xOxm!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdfecf2ee-c21d-4eb9-bff8-923ff01031ff_1600x2173.png)

The routing diagram shows Moonshot's answer. Instead of the brittle heuristic load-balancing losses that plague many MoE models, K3 uses Quantile Balancing: a token routes to an expert if its router score lands in the top quantile, which is deterministic, hyperparameter-free, and guarantees even utilization with zero dead experts. Two supporting tricks matter. Per-Head Muon applies the Muon optimizer independently to each attention head for training stability, and the Sigmoid Tanh Unit activation is chosen specifically to avoid the dead-neuron pathology that rarely-activated experts are prone to. The result, the two green boxes in the figure, is roughly 50 to 60 billion active parameters per token out of 2.8 trillion total.

One more piece makes serving possible: quantization-aware training. K3 trains with MXFP4 weights and MXFP8 activations from the supervised fine-tuning stage onward, so the model learns to be resilient to the low-precision format it will be served in, rather than eating the usual 10 to 15 percent quality hit from quantizing a giant model after the fact.

## Where K3 finishes first, and what that actually measures

The #4 overall ranking is the headline that traveled. The more interesting story is the handful of benchmarks where K3 finishes first outright, because those are the ones tied to real engineering value rather than composite trivia.

[![K3 ranks #1 on Frontend Code Arena at 1679 Elo, top on Program Bench, top 3 on SWE-Marathon, with 57 Intelligence and 76.24 Coding indices.](https://substackcdn.com/image/fetch/$s_!Q1Zv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15a3f89a-2f2d-427e-86d3-d494bf0120f9_1600x2800.png)](https://substackcdn.com/image/fetch/$s_!Q1Zv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15a3f89a-2f2d-427e-86d3-d494bf0120f9_1600x2800.png)

This summary board is the map for the rest of this section. Frontend Code Arena is the clean number-one. Program Bench and SWE-Marathon are the deep-engineering wins. The overall ratings underneath, a 57 on the Intelligence Index and a 76.24 on the Coding Index, place K3 fourth in the world overall and first among open models on web engineering. Now the detail on each.

### Frontend Code Arena, and why a 17-place jump matters

Frontend Code Arena is a live human-preference benchmark on LMArena where models generate real HTML, CSS, and JavaScript from a prompt and humans vote on head-to-head pairs. What makes it hard, and meaningful, is that a frontend answer cannot be graded by "does it compile." A model can emit syntactically perfect HTML that renders with a broken layout, wrong colors, or buttons that do nothing. Serious frontend evaluation harnesses like FrontendBench run the generated code in a sandboxed Node.js plus Puppeteer plus Jest environment to check element presence, functional correctness, and interactive logic all three. The arena captures the same three axes through human judgment at scale.

[![Frontend Code Arena grades visual fidelity, functional correctness, and interactive logic across six domains; K3 wins five and places second only on gaming.](https://substackcdn.com/image/fetch/$s_!Y02h!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbccbf8f6-f4da-463f-8288-00fa101e00db_1600x3062.png)](https://substackcdn.com/image/fetch/$s_!Y02h!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbccbf8f6-f4da-463f-8288-00fa101e00db_1600x3062.png)

The breakdown figure is the substance of the win. K3 took first place in six of seven frontend domains, e-commerce, dashboards, forms, landing pages, and web apps among them, losing only the gaming category to Fable 5. It debuted at 1679 Elo across 1,757 votes, a 17-place leap from Kimi K2.6, which sat at number eighteen, and its raw accuracy climbed from 33 to 46 percent generation over generation. The significance is not the Elo number itself, which still carries a plus or minus 17-point confidence band this early. It is that this is the first time an open-weight model has ranked ahead of every proprietary model on a comprehensive web-engineering board. Frontend is the most visible surface of AI-assisted development, the thing a user sees first, and an open model just took the top of it.

### Program Bench: rebuild the program from the binary

Program Bench is a different and arguably harder animal. The agent is handed a compiled binary and its documentation and asked to reconstruct the complete source codebase that reproduces the program's behavior. This is not "write a function that passes a unit test." It forces interface discovery, architecture-level reasoning, and behavioral fidelity across more than one code path. A model can pass HumanEval by emitting one short correct function; Program Bench refuses to be gamed that way because the target is an entire working program, verified across multiple paths.

K3 is a top performer here, and the reasons line up exactly with its architecture. Reconstructing a whole codebase demands holding the entire thing in context, which the 1M window and KDA make affordable. Synthesizing a coherent architecture across many layers is precisely what Attention Residuals were built to help with. And the fifty-plus billion active parameters give it far more headroom than routine code completion needs, leaving capacity for the reasoning. This benchmark matters because it mirrors the real work of refactoring legacy systems, migrating codebases, and reverse-engineering software nobody has documentation for.

### SWE-Marathon: can it work for days without cheating

SWE-Marathon is the closest thing we have to measuring a real engineering job. Tasks are ultra-long-horizon, building a compiler, optimizing a kernel, standing up a production service, the kind of work measured in hours to days rather than minutes. The numbers around it are staggering: mean rollout usage is 27.2 million tokens per task, with a right tail reaching 877 million, and individual long-horizon trials can cost hundreds of dollars to evaluate. A full sweep runs into the tens of thousands.

The subtle challenge SWE-Marathon exposes is benchmark integrity. Over a long enough horizon, agents start reward hacking, exploiting the scoring system instead of genuinely solving the task, which is why the benchmark ships full agent trajectories for scrutiny. K3 lands in the top three here. Its 1M context lets it hold entire project state without forgetting what it did an hour ago, and KDA's decoding speed is what makes rollouts this long computationally survivable in the first place. If Frontend Code Arena proves K3 can produce polished output, SWE-Marathon proves it can sustain the grind, and the grind is where most real software gets built.

The paid section below picks up from here: the geopolitics of a frontier model funded partly by a Chinese state-owned enterprise, the specific open-source strategy K3's capabilities make possible, a deployment FAQ with real cost math, questions I think are worth sitting with, and where I believe this goes next. The free analysis above stands on its own; the paid half is the part you can act on.

![User's avatar](https://substackcdn.com/image/fetch/$s_!gd2H!,w_64,h_64,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d670301-204b-472e-a2ee-bbb1b7633a99_2026x2026.png)

## Continue reading this post for free, courtesy of Ken Huang.

Claim my free post

[Or purchase a paid subscription.](https://kenhuangus.substack.com/subscribe?simple=true&next=https%3A%2F%2Fkenhuangus.substack.com%2Fp%2Fdemystifying-kimi-k3-how-chinas-28t&utm_source=paywall&utm_medium=web&utm_content=207513451&just_signed_up=falsesimple=true&utm_source=paywall&utm_medium=email&utm_content=207513451&next=https://kenhuangus.substack.com/p/demystifying-kimi-k3-how-chinas-28t)

PreviousNext

© 2026 ken · [Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start your Substack](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer) [Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com/) is the home for great culture