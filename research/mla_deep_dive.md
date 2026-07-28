[Lior Sinai](https://liorsinai.github.io/)
Menu


- [Home](https://liorsinai.github.io/)
- [About](https://liorsinai.github.io/about)
- [Posts](https://liorsinai.github.io/posts)

# DeepSeek's Multi-Head Latent Attention

Posted on 22 February, 2025

Generated with OpenAI's DALL-E

_A deep dive into DeepSeek’s Multi-Head Latent Attention, including the mathematics and implementation details. The layer is recreated in Julia using Flux.jl._

See also previous posts on transformers:

- [Transformers from first principles in Julia](https://liorsinai.github.io/machine-learning/2022/05/18/transformers).
- [Generative transformer from first principles in Julia](https://liorsinai.github.io/machine-learning/2024/03/23/transformers-gpt).

All code available at [github.com/LiorSinai/TransformersLite.jl/tree/feature/mla](https://github.com/LiorSinai/TransformersLite.jl/tree/feature/mla).

### Table of Contents

1. [Introduction](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#introduction)
2. [KV Caching](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#kv-caching)
1. [Theory](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#kv-caching-theory)
2. [Code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#kv-caching-code)
3. [Multi-Head Latent Attention](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#multi-head-latent-attention)
1. [C cache](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-cache)
2. [Code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-code)
3. [Absorption](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-absorption)
4. [Broadcasted batched multiplication](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#broadcasted-batched-mul)
5. [Code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-absorption-code)
6. [Decoupled RoPE](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-rope)
7. [Code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#rope-code)
4. [Conclusion](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#conclusion)

## 1 Introduction

In January 2025, [DeepSeek](https://www.deepseek.com/) unveiled their new DeepSeek-V3 and DeepSeek R1 models.
It took the [world](https://edition.cnn.com/2025/01/27/tech/deepseek-ai-explainer/index.html) [by](https://www.technologyreview.com/2025/01/31/1110740/how-deepseek-ripped-up-the-ai-playbook-and-why-everyones-going-to-follow-it/) [storm](https://hn.algolia.com/?q=deepseek).
Users were impressed with its abilities on top of their claims that it was up to 50× more efficient to train and run than their competitors.

They also released multiple papers
( [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-VL2), [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1))
with an impressive array of new techniques across the whole machine learning pipeline, from high level theory to intricate implementation details.
Most of it built on existing ideas in innovative ways. They include:

- Theory
  - Multi-Head Latent Attention (MLA): compress vectors during attention, which reduces the cache size during inference.
  - DeepSeekMoE: segmented and isolated mixture of experts.
  - Multi-token prediction.
  - Reinforcement learning with Group Relative Policy Optimization but without supervised data.
  - Improved chain-of-thought reasoning.
- Implementation
  - DualPipe: accelerate training by overlapping forward and backward computation communication phases.
  - Exponential moving average on the CPU.
  - Mixed precision floating point numbers during training. FP8 and FP32 are used.
  - Low-precision storage and communication.

The aim of this post is to explore the first of these ideas, namely Multi-Head Latent Attention (MLA).
The idea behind MLA is simple: it compresses the input matrix into a single matrix for caching during inference, as opposed to standard KV caching which caches two intermediate matrices.
DeepSeek created MLA by combining three older ideas:

1. Attention and multi-head attention from [Attention is all you need (2017)](https://arxiv.org/abs/1706.03762).
2. KV Caching (see [below](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#kv-caching)).
3. Low-Rank Adaption matrices (LoRA) from [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685).

MLA has side effects which DeepSeek barely address.
The compression also results in a performance boost but also comes with a qualitative penalty.
That these are not discussed is a notable omission in their [paper](https://github.com/deepseek-ai/DeepSeek-VL2).
The performance boost is addressed in this article but not the impact on quality.

DeepSeek also adds two unwieldy enhancements to MLA. While they only speak good of these, they complicate the mathematics and require specialised optimised code to see performance gains.
These enhancements are discussed in this article for completeness.
However my recommendation is to not use them.

1. Weight absorption.
2. Decoupled rotary position embeddings (RoPE), a modification of [RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864).

The original source code is written in Python with the PyTorch machine learning framework.
However, my favourite language is Julia and continuing with my previous posts on transformers, all the code here is written in Julia using the Flux.jl machine learning framework.

Julia uses column major format whereas Python uses row major format. In Julia sample vectors are columns while in Python they are rows.
Equations between the two formats will look backwards to each other.
They need to be transposed and definitions also need to be transposed.
E.g. KTQ→(KTcQc)T=QTcKc=QrKTr

The mathematics therefore follows Julia’s columnar major format and not Python’s row major format.
In this format each column represents a single sample. The second dimension is sequence length and all remaining dimensions represent batches.

## 2 KV Caching

### 2.1 Theory

There is an inefficiency in using transformers for generation.
In classification [use cases](https://liorsinai.github.io/machine-learning/2022/05/18/transformers#use-case-amazon-reviews), the model only needs to calculate attention over the input sentence once (per layer) before making its prediction.
But in text generation, it needs to recalculate it over the entire sentence.
One analogy I read is that this is like having to reread the entire sentence so far in order to produce the next word and then rereading again to produce the next word, and so on.

In mathematical terms, this is computing the attention between the first token and itself, between the first two tokens, between the first 3 tokens and so on until it’s between all the tokens, and repeating this whole process each time for every new token.
This was the case for the [generator](https://liorsinai.github.io/machine-learning/2024/03/23/transformers-gpt) I made based on Andrej Karpathy’s [Zero to Hero course](https://karpathy.ai/zero-to-hero.html).

It would be better to have a “memory” of what has been generated so far.
This is where the KV cache comes in, for Key-Value cache.
We can store the previous keys and values and use them to calculate the new attention value.
This will give the exact same value as recalculating the entire attention.

Here is a mathematical proof.
For a visual interpretation, please see João Lages’ [excellent article](https://medium.com/@joaolages/kv-caching-explained-276520203249).

The attention equation is

A=Vsoftmax(1√dhKTQ)

where

K=WKXQ=WQXV=WVX

where X∈Rd×n×B and WK,WQ,WV∈RdhH×d.

Focusing on one head in one batch with dimension dh, the input matrix X is split into the first n−1 columns and the nth column.
The first multiplication can then be written as (dimensional analysis on right):[1](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#fn:forgive)

S=KTQ=\[K1:n−1Kn\]T\[Q1:n−1Qn\];\[dh×(n−1)dh×1\]T\[dh×(n−1)dh×1\]=\[KT1:n−1KTn\]\[Q1:n−1Qn\];\[(n−1)×dh1×dh\]\[dh×(n−1)dh×1\]=\[KT1:n−1Q1:n−1KT1:n−1QnKTnQ1:n−1KTnQn\];\[(n−1)×(n−1)(n−1)×11×(n−1)1×1\]

Looking at the final line, the first (n−1) columns of the query can be safely dropped without affecting the nth column.
(We would do this anyway in [generation](https://liorsinai.github.io/machine-learning/2024/03/23/transformers-gpt#generation).)
The nth column only depends on K1:n−1, Kn and Qn.
Of these, K1:n−1 will come from the cache and the other two will be calculated from Xn.

It is important to note that dropping the first (n−1) columns is also valid because almost all other layers in a transformer are independent of position.
For example for the dense layers Y=WX, permuting the columns of X will result in a corresponding permutation of the columns of Y.
E.g. if X has two columns and \[Y1Y2\]=W\[X1X2\] then \[Y2Y1\]=W\[X2X1\]
The exception is the position embedding layers which will require a new parameter to be passed through the whole transformer to indicate the position.


Similarly for the next multiplication we have (Z=softmax(S)):

A=VZ=\[V1:n−1Vn\]\[Z1:n−1,1:n−1Z1:n−1,nZn,1:n−1Zn,n\]=\[V1:n−1Z1:n−1,1:n−1+VnZn,1:n−1V1:n−1Z1:n−1,n+VnZn,n\]

However, we said we are dropping the first (n−1) columns.
Without them only the nth column is calculated.
Hence we have:

An=\[V1:n−1Vn\]\[Z1:n−1,nZn,n\]=V1:n−1Z1:n−1,n+VnZn,n

which depends on V1:n−1, which will come from the cache, and Vn, which will be calculated from Xn.

There will be two caches each with size dhH×N×B for H heads, a maximum sequence length of N and a maximum batch size of B.
The total cache size can grow very large for large transformers with many multi-head attention layers.
The primary aim of MLA is to reduce the size of this cache. That will be covered in the next section.

### 2.2 Code

Building on my code in [TransformersLite.jl](https://github.com/LiorSinai/TransformersLite.jl), it is straightforward to create a new `MultiHeadAttentionKVCache` layer with two caches:

```julia
struct MultiHeadAttentionKVCache{
    Q<:Dense, K<:Dense, V<:Dense, O<:Dense, C<:Array{T, 3}  where T
    }
    nhead::Int
    denseQ::Q
    denseK::K
    denseV::V
    denseO::O
    cache_k::C
    cache_v::C
end

Flux.@layer trainable=(denseQ, denseK, denseV, denseO)
```

The forward pass calculates the current `q`, `k` and `v` values from the input and gets the rest from the cache.
It then continues without any additional modifications from the original code:

```julia
function (mha::MultiHeadAttentionKVCache)(
    query::A3, key::A3, value::A3
    ; start_pos::Int=1, use_cache::Bool=true, kwargs...
    ) where {T, A3 <: AbstractArray{T, 3}}
    q = mha.denseQ(query) # size(q) == (dh, 1, B)
    k = mha.denseK(key)
    v = mha.denseV(value) # size(k) == size(v) == (dh, 1, B)
    if use_cache
        dim, seq_length, batch_dim = size(query)
        end_pos = start_pos + seq_length - 1
        mha.cache_k[:, start_pos:end_pos, 1:batch_dim] = k
        mha.cache_v[:, start_pos:end_pos, 1:batch_dim] = v
        K = mha.cache_k[:, 1:end_pos, 1:batch_dim]
        V = mha.cache_v[:, 1:end_pos, 1:batch_dim]
    else
        K = k
        V = v
    end
    A, scores = multi_head_scaled_dot_attention(mha.nhead, q, K, V; kwargs...)
    mha.denseO(A), scores
end
```

Here is a small example of it in action. (For the full code, see [test/MultiHeadAttention.jl](https://github.com/LiorSinai/TransformersLite.jl/blob/feature/mla/test/MultiHeadAttention.jl).)

Create the layer and inputs:

```julia
using TransformersLite
using TransformersLite: MultiHeadAttention, MultiHeadAttentionKVCache
using TransformersLite: make_causal_mask, clone_add_kv_cache
nhead, dim_model, dim_out = 4, 32, 13
mha = MultiHeadAttention(nhead, dim_model, dim_out)
mha = clone_add_kv_cache(mha, 64, 8)
X = randn(Float32, 32, 10, 5)
```

Fill the cache:

```julia
mask = make_causal_mask(ones(10, 10))
A, scores = mha(X, X, X; mask=mask, start_pos=1, use_cache=true)
size(A) # (13, 10, 5)
size(scores) # (10, 10, 4, 5)
```

Use the cache with a new vector:

```julia
x = randn(Float32, 32, 1, 5)
mask = repeat([true], inner=(11, 1))
Ax, scoresx = mha(x, x, x; mask=mask, start_pos=11, use_cache=true)
size(Ax) # (13, 1, 5)
size(scoresx) # (11, 1, 4, 5)
```

Compare without the cache:

```julia
Xx = cat(X, x, dims=2)
mask = make_causal_mask(ones(11, 11))
AXx, scoresXx = mha(Xx, Xx, Xx; mask=mask, start_pos=1, use_cache=false)
isapprox(AXx[:, end, :], Ax[:, end, :]) # true
```

## 3 Multi-Head Latent Attention

### 3.1 C cache

We’ve seen that the KV cache has size 2dhH×N×B elements per multi-head attention layer. (Each element is 1-4 bytes depending if FP8, FP16 or FP32 is used.)
The aim of MLA is to reduce this, specifically to dc×N×B elements per multi-head attention layer.
Therefore we will choose dc<2dhH.

![Illustration of KV caching methods](https://liorsinai.github.io/assets/posts/transformers/deepseek_mla.png)Different KV caching techniques.

[DeepSeek](https://github.com/deepseek-ai/DeepSeek-VL2)’s innovation is to introduce a weight matrix WDKV∈Rdc×d to compress the input X∈Rd×n to a lower rank matrix CKV∈Rdc×n.
This CKV matrix is then stored in the cache.
Then two other weight matrices WUK and WUV∈RdhH×dc uncompress the same CKV matrix to the key K and value V respectively. The above [figure](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#fig-mla) shows this visually.

cKVn=WDKVxnK=WUKCKV1:nV=WUVCKV1:n

The KV cache is now replaced with a CKV cache of size dc×N×B.
DeepSeek theorises that this compression also results in a regularization effect that improves performance.
This is supported by other LoRA research.
However, the lossy compression might instead adversely affect quality.
DeepSeek provides no evidence towards either claim.

The compression also results in a significant performance boost which DeepSeek strangely does not mention in their [paper](https://github.com/deepseek-ai/DeepSeek-VL2).
Note that in MLA there are three matrix multiplications to perform to create K and V
instead of two matrix multiplications in MHA.
However the three multiplications comprise of less scalar operations:[2](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#fn:matrix_complexity)\# MLA ops\# MHA ops=2(2dhH+d)dcnB2(2dhH)dnB=2dhHd+12dhHdc=32r
with the standard d=dhH and a compression ratio r=dhHdc.
This requires r>1.5 for performance gains.
The only performance penalty is the memory required for the nth cKVn vector before it is transferred to the cache, which is dc×1×B.

DeepSeek-V3 uses dh=128, H=128 and dc=4dh=512 which means it has a compression ratio of 32 and a 20× speed up!

To reduce the activation memory during training, DeepSeek also applies the same strategy to the query:

CQ=WDQXQ=WUQCQ

In total, five matrix multiplications are needed to create Q, K and V instead of three.
The overall ratio of scalar operations is 53r, which requires r>1.67 for performance gains.

DeepSeek give further enhancements to this which will be described shortly.
They also apply layer normalisation to CQ and CKV which I will ignore in this article.
For now, let’s see this basic version of MLA in action.

### 3.2 Code

First create a struct similar to the `MultiHeadAttentionKVCache` layer.

```julia
struct MultiHeadLatentAttention{D1<:Dense, D2<:Dense, A<:AbstractArray{T, 3} where T}
    nhead::Int
    denseDQ::D1
    denseUQ::D1
    denseDKV::D1
    denseUK::D1
    denseUV::D1
    denseO::D2
    cache_ckv::A
end

Flux.@layer MultiHeadLatentAttention trainable=(denseDQ, denseUQ, denseDKV, denseUK, denseUV, denseO)
```

Here is a convenience constructor to construct it from the various input dimensions:

```julia
function MultiHeadLatentAttention(;
    nhead::Int, dim_in::Int, dim_head::Int, dim_lora, dim_out::Int,
    max_seq_length::Int, max_batch_size::Int
    )
    denseDQ = Dense(dim_in => dim_lora; bias=false)
    denseUQ = Dense(dim_lora => dim_head * nhead; bias=false)
    denseDKV = Dense(dim_in => dim_lora; bias=false)
    denseUK = Dense(dim_lora => dim_head*nhead; bias=false)
    denseUV = Dense(dim_lora => dim_head*nhead; bias=false)
    denseO = Dense(dim_head*nhead => dim_out; bias=false)
    cache_ckv = Array{Float32, 3}(undef, dim_lora, max_seq_length, max_batch_size)
    MultiHeadLatentAttention(
        nhead,
        denseDQ, denseUQ,
        denseDKV, denseUK, denseUV,
        denseO,
        cache_ckv
    )
end
```

The forward pass is:

```julia
function (mla::MultiHeadLatentAttention)(query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=true, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    dm, seq_length, batch_dim = size(key)
    cq = mla.denseDQ(query) # size(cq) == (dc, dq, B)
    ckv = mla.denseDKV(key) # size(ckv) == (dc, dkv, B)
    if use_cache
        end_pos = start_pos + seq_length - 1
        mla.cache_ckv[:, start_pos:end_pos, 1:batch_dim] = ckv
        ckv = mla.cache_ckv[:, 1:end_pos, 1:batch_dim]
    end
    K = mla.denseUK(ckv) # size(k) == (dh*nhead, dkv, B)
    V = mla.denseUV(ckv) # size(v) == (dh*nhead, dkv, B)
    Q = mla.denseUQ(cq)  # size(q) == (dh*nhead, dq, B)
    A, scores = multi_head_scaled_dot_attention(mla.nhead, Q, K, V; mask=mask)
    A = mla.denseO(A)
    A, scores
end
```

Create a test layer with compression ratio r=dhHdc=4:

```julia
nhead, dim_head, dim_lora, dim_out = 8, 64, 128, 8*64
dim_model = nhead * dim_head
N, max_seq_length, batch_dim = 20, 32, 8
mla = MultiHeadLatentAttention(
    nhead=nhead, dim_in=dim_model, dim_head=div(dim_model, nhead),
    dim_lora=dim_lora, dim_out=dim_out,
    max_seq_length=max_seq_length, max_batch_size=batch_dim
    )
X0 = randn(Float32, dim_model, N, batch_dim)
```

Fill the cache:

```julia
mask = make_causal_mask(ones(N, N));
A, scores = mla(X0, X0; mask=mask, use_cache=true);
size(A) # (512, 20, 8)
size(scores) # (20, 20, 8, 8)
```

Use the cache with a new vector:

```julia
x = randn(Float32, dim_model, 1, batch_dim)
mask = repeat([true], inner=(N + 1, 1))
Ax, scoresx = mla(x, x; mask=mask, start_pos=N+1, use_cache=true)
size(Ax) # (512, 1, 8)
size(scoresx) # (21, 1, 8, 8)
```

### 3.3 Absorption

DeepSeek suggests a way to further decrease the computational cost by absorbing weight matrices into each other.
To quote [DeepSeek](https://github.com/deepseek-ai/DeepSeek-VL2) directly:

> In addition, during inference, since WUK can be absorbed into WQ , and WUV can be absorbed
> into WO, we even do not need to compute keys and values out for attention.

What they mean is that the weight matrices can be multiplied to produce a single weight matrix.
This can only be done during inference because during training they need to be kept separate so that the gradients flow properly backwards through each matrix.

This technique can be used independently of MLA.

To show why this works, rewrite the attention equation 2.1.1 as follows:

S=KTQ=(WUKCKV)T(WUQCQ)=(CKV)T(WUK)TWUQCQ=(CKV)TWKQCQ;WKQ=(WUK)TWUQ

This looks straightforward but there are further complications with the dimensions.
Here is a dimensional analysis of the above equation following the two rules of batch matrix multiplication:

1. The inner matrix dimensions must match. That is, the second dimension of the first matrix must match the first dimension of the second.
2. All the batch dimensions (dimensions 3 and greater) must be equal.

(dc×n×B)T(dhH×dc)T(dhH×dc)(dc×n×B)=(n×dc×B)(dc×dh×H)(dh×dc×H)(dc×n×B)=(n×dc×B)(dc×dc×H)(dc×n×B)=(n×dc×1×B)(dc×dc×H×1)(dc×n×1×B)=n×n×H×B

where

- Line 2 reshapes the weight matrices from dhH×dc to dh×dc×H. This is necessary because the non-linear softmax function must be applied independently over each head dimension.
- Line 3 shows that WKQ∈Rdc×dc×H.
- Line 4 adds extra broadcast dimensions to make the batch dimensions match.

Broadcasting is a technique where the smaller array is replicated along all dimensions of size 1 to match the size of the larger array.
For broadcasted batched multiplication this only needs to be done for the 3rd and higher dimensions.
Pseudo-code for this is:

> **Broadcasted batched multiplication**
>
> inputs: A∈RI×R×LA×KA, B∈RR×J×LB×KB
>
> **for** l in 1:max(LA,LB)
>
> lA← 1 **if** LA=1 **else** l
>
> lB← 1 **if** LB=1 **else** l
>
> **for** k in 1:max(KA,KB)
>
> kA← 1 **if** KA=1 **else** k
>
> kB← 1 **if** KB=1 **else** k
>
> Cl,b=AlA,kABlB,kB

This same absorption technique can be applied to the value and output matrices:
Y=WOVZ=WO(WUVCKVZ)=WOVCKVZ

The dimensional analysis here is similar:

(do×dhH)(dhH×dc)(dc×n×B)(n×n×H×B)=(do×dh×H)(dh×dc×H)(dc×n×1×B)(n×n×H×B)=(do×dh×H)(dh×dc×H)(dc×n×H×B)=(do×dc×H)(dc×n×H×B)=(do×dcH)(dcH×n×B)=do×n×B

This shows that the CKVZ multiplication is a broadcasted batched multiplication.
However where WKQ∈Rdc×dc×H, WOV∈Rdo×dcH is a typical 2D matrix.
Therefore the usual matrix multiplication can be applied by reshaping the CKVZ result from a 3D dcH×n×B array to a dcH×nB matrix.

### 3.4 Broadcasted batched multiplication

I have written an implementation in Julia directly based on the [pseudo code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#pseudo-broadcast-batch-mul).
It can be seen here: [broadcasted\_batched\_mul.jl](https://github.com/LiorSinai/TransformersLite.jl/blob/feature/mla/src/broadcasted_batched_mul.jl).
However, it is inefficient and uses scalar indexing which is extremely slow on a GPU.

My solution instead is to physically replicate the broadcasted dimensions.
This is of course inefficient compared to virtual replication but it makes the function viable on a GPU.
The downside is it can be up to 4× slower than the naive code on a CPU.

```julia
using Flux: batched_mul
function broadcasted_batched_mul(x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    batch_dims_x = Tuple(size(x, idx) == 1 ? size(y, idx) : 1 for idx in 3:N)
    dims_x = (1, 1, batch_dims_x...)
    batch_dims_y = Tuple(size(y, idx) == 1 ? size(x, idx) : 1 for idx in 3:N)
    dims_y = (1, 1, batch_dims_y...)
    xb = repeat(x; outer=dims_x)
    yb = repeat(y; outer=dims_y)
    batched_mul(xb, yb)
end
```

The DeepSeek [source code](https://github.com/deepseek-ai/DeepSeek-V3) meanwhile uses [torch.einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html) and carries out the multiplications right to left instead of creating a new matrix.[3](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#fn:tensors)
Here is the relevant code.
As far as I know, this has the same drawbacks with scalar indexing with none of the advantages of absorption as described in their paper.

```python
q = self.wq_b(self.q_norm(self.wq_a(x)))
q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
kv = self.wkv_a(x)
wkv_b = self.wkv_b.weight
wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
q = torch.einsum("bshd,hdc->bshc", q, wkv_b)
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
scores = torch.einsum("bshc,btc->bsht", q, self.kv_cache[:bsz, :end_pos])
```

Presumably fast MLA implementations for GPU kernels would make use of virtual replication.
(It would be great if someone can clarify what [FlashMLA](https://github.com/deepseek-ai/FlashMLA) does.)

### 3.5 Code

I will detail some of the code here. For the full code, see [MultiHeadLatentAttentionV2.jl](https://github.com/LiorSinai/TransformersLite.jl/blob/feature/mla/src/layers/MultiHeadLatentAttentionV2.jl).

The first step is to create the WKQ and WOV matrices.

First, WKR=(WUK)TWUQ while reshaping from dhH×dc to dh×dc×H.

```julia
function _absorb_WUK_WUQ(nhead::Int, W_UK::AbstractMatrix, W_UQ::AbstractMatrix)
    dh = div(size(W_UK, 1), nhead)
    dim_lora = size(W_UK, 2)
    W_UQ = permutedims(reshape(W_UQ, dh, nhead, dim_lora), (1, 3, 2))
    W_UK = permutedims(reshape(W_UK, dh, nhead, dim_lora), (1, 3, 2))
    W_UKT = permutedims(W_UK, (2, 1, 3)) # (dh, dc, nhead)^T => (dc, dh, nhead)
    batched_mul(W_UKT, W_UQ)
end
W_KQ = _absorb_WUK_WUQ(nhead, denseUK.weight, denseUQ.weight)
```

Then WOV=WOWUV while preserving the head dimension through reshaping:

```julia
function _absorb_WO_WUV(nhead::Int, W_O::AbstractMatrix, W_UV::AbstractMatrix)
    dh = div(size(W_UV, 1), nhead)
    dim_lora = size(W_UV, 2)
    dout = size(W_O, 1)
    W_UVh = permutedims(reshape(W_UV, dh, nhead, dim_lora), (1, 3, 2)) # (dh*nhead, dc) => (dh, dc, nhead)
    W_Oh = reshape(W_O, dout, dh, nhead) # (dout, dh*nhead) => (dout, dh, nhead)
    W_OVh = batched_mul(W_Oh, W_UVh) # (dout, dh, nhead) * (dh, dc, nhead)
    reshape(W_OVh, dout, dim_lora*nhead) # (dout, dc, nhead) => (dout, dc*nhead)
end
W_OV = _absorb_WO_WUV(nhead, denseO.weight, denseUV.weight)
```

The forward pass is the same as in the original [code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-code) until the end of caching:

```julia
function mla_absorb(
    mla::MultiHeadLatentAttention, query::A3, key::A3
    ; start_pos::Int=1, use_cache::Bool=true, mask::Union{Nothing, M}=nothing
    ) where {T, A3 <: AbstractArray{T, 3}, M <: AbstractArray{Bool}}
    dm, seq_length, batch_dim = size(key)
    dh = div(dm, mla.nhead)
    cq = mla.norm_cq(mla.denseDQ(query))  # size(cq) == (dc, dq, B)
    ckv = mla.norm_ckv(mla.denseDKV(key)) # size(ckv) == (dc, dkv, B)
    if use_cache
        end_pos = start_pos + seq_length - 1
        mla.cache_ckv[:, start_pos:end_pos, 1:batch_dim] = ckv
        ckv = mla.cache_ckv[:, 1:end_pos, 1:batch_dim]
    end
```

Then add the broadcast dimensions:

```julia
    ckv_ = Flux.unsqueeze(ckv, dims=3)
    keyT = permutedims(ckv_, (2, 1, 3, 4)) # (dkv, dc, B) => (dkv, dc, 1, B)
    cq_ = Flux.unsqueeze(cq, dims=3) # (dkv, dc, B) => (dkv, dc, 1, B)
    W_KQ = Flux.unsqueeze(mla.W_KQ, dims=4); # (dc, dc, nhead) => (dc, dc, nhead, 1)
```

Then apply the equations as before except using `broadcasted_batched_mul` instead of `batched_mul`:

```julia
    atten_base = broadcasted_batched_mul(keyT, broadcasted_batched_mul(W_KQ, cq_))
    atten = one(T)/convert(T, sqrt(dh)) .* (atten_base)
    atten = apply_mask(atten, mask)
    scores = softmax(atten; dims=1)
    A = broadcasted_batched_mul(ckv_, scores) # (dc, dq, nhead, B)
    # (dc, dq, nhead, B) => (dc*nhead, dq, B)
    A = permutedims(A, [1, 3, 2, 4])
    A = reshape(A, :, size(A, 3), size(A, 4))
    mla.denseOV(A), scores
end
```

Test that these give the same result:

```julia
X0 = randn(Float32, dim_model, N, batch_dim)
mask = make_causal_mask(ones(N, N));
A_naive, scores_naive = mla_naive(mla, X0, X0; mask=mask, use_cache=true);
A_absorb, scores_absorb = mla_absorb(mla, X0, X0; mask=mask, use_cache=true);
isapprox(A_absorb, A_naive) # true
```

### 3.6 Decoupled RoPE

The last enhancement DeepSeek adds is [RoPE](https://arxiv.org/abs/2104.09864).
One issue with RoPE however is that it breaks the absorption property described above.
To prove this, RoPE can be represented as a series of matrix multiplications on each column in an input matrix X:

RoPE(X)=\[R1X1R2X2...RnXn\]

Applying this to the scores equation:
S=RoPE(K)TRoPE(Q)=RoPE(WUKCKV)TRoPE(WUQCQ)=\[R1WUK1CKV1...RnWUKnCKVn\]T=x\[R1WUQ1CQ1...RnWUQnCQn\]⟹Sij=(CKVi)T(WUKi)TRTiRjWUQjCQj

which shows that the rotation matrices will appear right in the middle of the product.

DeepSeek’s solution is to concatenate another matrix to the bottom of the key K and query Q respectively and only apply RoPE to these matrices.
Furthermore, because the new matrix KR will also need to be cached, they share it across all heads.
To put it another way, KR will be broadcasted across the head dimension during multiplication.
So for each head h:

Kh=\[WUKhCKVRoPE(WKRX)\]Qh=\[WUQhCQRoPE(WQRhCQ)\]

where WKR∈RdR×d and WQR∈RdRH×dc. This means that K,Q∈R(dh+dR)×n×H×B.
The cache will consist of both CKV and KR for a total size of (dc+dR)×N×B elements per layer.

Very conveniently, this results in an addition between the original and embedded scores:

Sh=KTQ=\[(K0)T(KR)T\]\[Q0QR\]=(K0)TQ0+(KR)TQR

which means that these results can be calculated separately. Note that Sh must now be scaled by 1/√dh+dR instead of 1/√dh.

### 3.7 Code

I will detail some of the code here. For the full code, see [MultiHeadLatentAttentionV2.jl](https://github.com/LiorSinai/TransformersLite.jl/blob/feature/mla/src/layers/MultiHeadLatentAttentionV2.jl).

Here is a Julia implementation of RoPE:

```julia
struct RoPE{T}
    base::Int
    dim::Int
    seq_length::Int
    freqs_complex::Matrix{Complex{T}}
end

RoPE(dim::Int, max_seq_length::Int; base::Int=10_000) = RoPE(Float32, dim, max_seq_length; base=base)

function RoPE(T::DataType, dim::Int, max_seq_length::Int; base::Int=10_000)
    @assert dim % 2 == 0 "Require even dim"
    θ = 1 ./ (base .^ ((0:2:(dim - 2)) / dim))
    angles = θ * transpose(0:(max_seq_length-1))
    freqs = map(x -> reverse(sincos(x)), angles)
    freqs_complex = map(cs -> Complex(cs...), freqs)
    RoPE{T}(base, dim, max_seq_length, freqs_complex)
end
```

The forward pass can be calculated with matrices, but the RoPE authors gave a more efficient implementation with complex numbers:

```julia
(r::RoPE)(x::AbstractArray) = apply_rope(x, r.freqs_complex[:, 1:size(x, 2)])
(r::RoPE)(x::AbstractArray, indices) = apply_rope(x, r.freqs_complex[:, indices])

function apply_rope(x::AbstractArray{T}, freqs_complex::AbstractMatrix{<:Complex{T}}) where T
    x_complex = reinterpret(Complex{T}, x)
    rx_complex = freqs_complex .* x_complex
    T.(reinterpret(T, rx_complex))
end
```

Then add the `embedding`, `denseQR` and `denseKR` layers to the `MultiHeadLatentAttention` struct.

The embeddings are applied as follows:

```julia
function _apply_embeddings(mla::MultiHeadLatentAttention, key::A3, cq::A3, idx::UnitRange{Int}) where {T, A3 <: AbstractArray{T, 3}}
    dim_lora, dq, batch_dim = size(cq)
    kr = mla.denseKR(key)
    qr = mla.denseQR(cq)
    kr = mla.embedding(kr, idx) # size(kr) == (dr, dkv, B)
    qr = permutedims(reshape(qr, :, mla.nhead, dq, batch_dim), (1, 3, 2, 4)) # (dr*nhead, dq, B) => (dr, dq, nhead, B)
    qr = mla.embedding(qr, idx)
    kr, qr
end
kr, qr = _apply_embeddings(mla, key, cq, start_pos:end_pos)
```

Note that embedding is done per head, hence the reshaping of `qr`.

For the naive method, concatenate along the head dimension. This requires reshaping for `qr` and repeating `kr`.

```julia
    Q, K = _cat_decoupled_embedding(mla.nhead, Q, qr, K, kr)
```

where:

```julia
function _cat_decoupled_embedding(
    nhead::Int, Qin::A3, Qr::A4, Kin::A3, kr::A3
    ) where {T, A3 <: AbstractArray{T, 3}, A4 <: AbstractArray{T, 4}}
    dhq, dq, B = size(Qin)
    dhk, dkv, B = size(Kin)
    Q = reshape(
        cat(reshape(Qin, :, nhead, dq, B), permutedims(Qr, (1, 3, 2, 4)), dims=1),
        : , dq, B)
    Kr = repeat(Flux.unsqueeze(kr, dims=2), outer=(1, 2, 1, 1))
    K = reshape(
        cat(reshape(Kin, :, nhead, dkv, B), reshape(Kr, :, nhead, dkv, B), dims=1),
        :, dkv, B)
    Q, K
end
```

Then continue as before.

With absorption, broadcast batched multiply `kr` and `qr` and add to the original attention:

```julia
    krT = Flux.unsqueeze(permutedims(kr, (2, 1, 3)), dims=3) # (dr, dkv, B) => (dkv, dr, 1, 1B)
    atten_base = broadcasted_batched_mul(keyT, broadcasted_batched_mul(W_KQ, cq_))
    atten_embed = broadcasted_batched_mul(krT, qr)
    atten = one(T)/convert(T, sqrt(dh + dr)) .* (atten_base + atten_embed)
```

## Conclusion

Overall, I think MLA is a smart and useful idea.
However, after having explored it in depth, I am more critical of their enhancements.

The basic premise of Multi-Head Latent Attention is simple.
It compresses the input matrix so that a single smaller CKV matrix can be stored instead of the key K and value V matrices.
It then uncompresses this matrix into K and V with two additional weight matrices.
This also results in a significant performance increase which scales with the compression ratio dhHdc by a factor of 23 \- so the compression needs to be greater than a modest 1.5 for gains to be realised - and requires no further modifications to existing MHA code.
However, it is unclear what the qualitative effects of the compression are, and it is strange that DeepSeek did not discuss the performance benefits.

To this DeepSeek adds weight absorption and decoupled RoPE.
We have seen this complicates the mathematics and requires careful dimensional analysis.
True performance gains only come with an optimised `broadcasted_batched_mul` function.
Their own open source code does not even have such optimisations.
Personally, I see no benefit to this and would recommend the naive method with RoPE applied normally.
That is, apply each of the weight matrices to their inputs individually and then apply RoPE to the entire K and Q matrices.

While I am impressed with the ingenuity behind MLA, DeepSeek’s omissions coupled with extra, unwieldy enhancements make me more skeptical of their methodology.
If I examine their other techniques, I will do so with more caution.

* * *

1. Indices for dimensions are shown when they are relevant and left out when they’re not. For example, the row index (along the embedding dimension) is generally ignored so Xj is the jth column/token of X. But then later for the scores I’ll use Xi,j because both indices represent a position in the token sequence. But then later I use Xh to indicate the hth head of X which is along the 3rd dimension. Please forgive me for this and other abuses of matrix notation in this post. [↩](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#fnref:forgive)

2. This is the naive matrix multiplication algorithm. For sizes n×d and d×m, for each of the nm output elements there are d multiplications and d−1 additions (no addition for the first element), so there are nm(2d−1) operations in total. [↩](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#fnref:matrix_complexity)

3. In general multiplication is not defined for higher order arrays. But there is a set of multidimensional algebraic objects called [tensors](https://en.wikipedia.org/wiki/Tensor) where it is, and Einstein notation was designed for this use case.
Confusingly, Google named their machine learning framework TensorFlow and calls higher order arrays tensors.
So one should differentiate between machine learning tensors and geometric tensors.
They are not the same.
To give a simple explanation: one can think of geometric tensors as higher order arrays with severe constraints on their entries and operations because they represent geometric objects. These constraints make it harder - not easier - to code higher order arrays as geometric tensors. [↩](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#fnref:tensors)


* * *

[← Previous\\
Post](https://liorsinai.github.io/mathematics/2025/01/11/bentley-ottman.html "Notes on the Martinez-Rueda Polygon Clipping algorithm") [Next\\
Post →](https://liorsinai.github.io/coding/2026/06/21/names.html "Names in the USA (1880-2025)")

[1 Introduction](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#introduction) [2 KV Caching](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#kv-caching) [-2.1 Theory](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#kv-caching-theory) [-2.2 Code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#kv-caching-code) [3 Multi-Head Latent Attention](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#multi-head-latent-attention) [-3.1 C cache](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-cache) [-3.2 Code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-code) [-3.3 Absorption](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-absorption) [-3.4 Broadcasted batched multiplication](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#broadcasted-batched-mul) [-3.5 Code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-absorption-code) [-3.6 Decoupled RoPE](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#mla-rope) [-3.7 Code](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#rope-code) [Conclusion](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#conclusion)

* * *

Adapted from Clean Blog theme by [Start Bootstrap](https://startbootstrap.com/themes/clean-blog-jekyll/)

Copyright © Lior Sinai 2026