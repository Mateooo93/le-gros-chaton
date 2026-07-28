##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHubSubmit in GitHub

![](https://arxiv.org/static/base/1.0.1/images/icons/smileybones-small.svg)arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

[![arXiv logo](https://arxiv.org/static/base/1.0.1/images/arxiv-logo-primary-light.svg)Back to arXiv](https://arxiv.org/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) [Report Issue](https://arxiv.org/html/2604.00801v1# "Report an Issue") [Back to Abstract](https://arxiv.org/abs/2604.00801v1 "Back to abstract page") [Download PDF](https://arxiv.org/pdf/2604.00801v1 "Download PDF")

01. [Abstract](https://arxiv.org/html/2604.00801v1#abstract1 "In Routing-Free Mixture-of-Experts")
02. [1 Introduction](https://arxiv.org/html/2604.00801v1#S1 "In Routing-Free Mixture-of-Experts")
03. [2 Preliminaries](https://arxiv.org/html/2604.00801v1#S2 "In Routing-Free Mixture-of-Experts")
04. [3 Methodology](https://arxiv.org/html/2604.00801v1#S3 "In Routing-Free Mixture-of-Experts")    1. [3.1 Architecture](https://arxiv.org/html/2604.00801v1#S3.SS1 "In 3 Methodology ‣ Routing-Free Mixture-of-Experts")
    2. [3.2 Training](https://arxiv.org/html/2604.00801v1#S3.SS2 "In 3 Methodology ‣ Routing-Free Mixture-of-Experts")
05. [4 Experiments](https://arxiv.org/html/2604.00801v1#S4 "In Routing-Free Mixture-of-Experts")    1. [4.1 Experimental Setup](https://arxiv.org/html/2604.00801v1#S4.SS1 "In 4 Experiments ‣ Routing-Free Mixture-of-Experts")
    2. [4.2 Main Results](https://arxiv.org/html/2604.00801v1#S4.SS2 "In 4 Experiments ‣ Routing-Free Mixture-of-Experts")
    3. [4.3 Architectural Comparison](https://arxiv.org/html/2604.00801v1#S4.SS3 "In 4 Experiments ‣ Routing-Free Mixture-of-Experts")
    4. [4.4 Training Dynamics](https://arxiv.org/html/2604.00801v1#S4.SS4 "In 4 Experiments ‣ Routing-Free Mixture-of-Experts")
06. [5 Discussion](https://arxiv.org/html/2604.00801v1#S5 "In Routing-Free Mixture-of-Experts")    1. [5.1 Per-Layer and Global Density](https://arxiv.org/html/2604.00801v1#S5.SS1 "In 5 Discussion ‣ Routing-Free Mixture-of-Experts")
    2. [5.2 Token and Expert Balancing](https://arxiv.org/html/2604.00801v1#S5.SS2 "In 5 Discussion ‣ Routing-Free Mixture-of-Experts")
07. [6 Related Work](https://arxiv.org/html/2604.00801v1#S6 "In Routing-Free Mixture-of-Experts")    1. [6.1 MoE Foundations](https://arxiv.org/html/2604.00801v1#S6.SS1 "In 6 Related Work ‣ Routing-Free Mixture-of-Experts")
    2. [6.2 Routing Mechanisms](https://arxiv.org/html/2604.00801v1#S6.SS2 "In 6 Related Work ‣ Routing-Free Mixture-of-Experts")
    3. [6.3 Load Balancing and Training](https://arxiv.org/html/2604.00801v1#S6.SS3 "In 6 Related Work ‣ Routing-Free Mixture-of-Experts")
08. [7 Conclusion](https://arxiv.org/html/2604.00801v1#S7 "In Routing-Free Mixture-of-Experts")
09. [References](https://arxiv.org/html/2604.00801v1#bib "In Routing-Free Mixture-of-Experts")
10. [A Load-Balancing Losses](https://arxiv.org/html/2604.00801v1#A1 "In Routing-Free Mixture-of-Experts")
11. [B Routing-Free MoE at Deployment](https://arxiv.org/html/2604.00801v1#A2 "In Routing-Free Mixture-of-Experts")    1. [B.1 Expert Parallelism](https://arxiv.org/html/2604.00801v1#A2.SS1 "In Appendix B Routing-Free MoE at Deployment ‣ Routing-Free Mixture-of-Experts")
    2. [B.2 Threshold Adaptation](https://arxiv.org/html/2604.00801v1#A2.SS2 "In Appendix B Routing-Free MoE at Deployment ‣ Routing-Free Mixture-of-Experts")
12. [C Statistical Significance Analysis](https://arxiv.org/html/2604.00801v1#A3 "In Routing-Free Mixture-of-Experts")
13. [D Additional Discussion](https://arxiv.org/html/2604.00801v1#A4 "In Routing-Free Mixture-of-Experts")    1. [D.1 Load-Balancing](https://arxiv.org/html/2604.00801v1#A4.SS1 "In Appendix D Additional Discussion ‣ Routing-Free Mixture-of-Experts")
    2. [D.2 Per-Layer and Global Density](https://arxiv.org/html/2604.00801v1#A4.SS2 "In Appendix D Additional Discussion ‣ Routing-Free Mixture-of-Experts")
14. [E Additional Experiment Results](https://arxiv.org/html/2604.00801v1#A5 "In Routing-Free Mixture-of-Experts")

[License: CC BY 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2604.00801v1 \[cs.LG\] 01 Apr 2026

# Routing-Free Mixture-of-Experts

Yilun Liu∗,†,1,3, Jinru Han∗,2, Sikuan Yan1, Volker Tresp1,3, Yunpu Ma†,1,3

1 Ludwig Maximilian University of Munich  2 University of California, Los Angeles

3 Munich Center for Machine Learning  ∗Equal contribution.

†yilun.liu@tum.decognitive.yunpu@gmail.com

###### Abstract

Standard Mixture-of-Experts (MoE) models rely on centralized routing mechanisms that introduce rigid inductive biases.
We propose Routing-Free MoE which eliminates any hard-coded centralized designs including external routers, Softmax, TopK and load balancing, instead encapsulating all activation functionalities within individual experts and are directly optimized through continuous gradient flow, enabling each expert to determine its activation entirely on its own.
We introduce a unified adaptive load-balancing framework to simultaneously optimize both expert-balancing and token-balancing objectives through a configurable interpolation, allowing flexible and customizable resource allocation.
Extensive experiments show that Routing-Free MoE can consistently outperform baselines with better scalability and robustness.
We analyze its behavior in detail and offer insights that may facilitate future MoE design and optimization.
Code is available at [https://github.com/liuyilun2000/RoutingFreeMoE/tree/release](https://github.com/liuyilun2000/RoutingFreeMoE/tree/release "").

![Refer to caption](https://arxiv.org/html/2604.00801v1/x1.png)Figure 1: Standard MoE relies on routing to orchestrate expert activations. Routing-Free MoE let each expert purely-independently determine its own activation. Green indicates activated components; red for inactive components; yellow for trainable components.

## 1 Introduction

The scalability of transformer-based Large Language Models (LLMs) (Vaswani et al., [2017](https://arxiv.org/html/2604.00801v1#bib.bib49 "")) is constrained by the substantial computational resources required (Kaplan et al., [2020](https://arxiv.org/html/2604.00801v1#bib.bib32 "")).
To efficiently expand model capacity without proportional computation cost growth, Mixture-of-Experts (MoE) designs focus on activating subsets of parameters for each input (Shazeer et al., [2017](https://arxiv.org/html/2604.00801v1#bib.bib47 ""); Fedus et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 "")).
This approach presents a fundamental challenge: how to optimally distribute inputs to experts while satisfying sparsity and balancing considerations.

Existing MoE designs are hindered by structural limitations across multiple dimensions.
Standard MoE relies on small, external routers that lack sufficient capacity for storing expert capabilities to determine expert preference for each input, forcing them to learn their prediction via indirect trial-and-error optimization, which inevitably leads to suboptimal routing and unstable early training dynamics (Lv et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib38 "")).
For computational efficiency, conventional MoE enforces rigid, global constraints on expert activation that ignore input-specific dynamics.
The fixed TopK selection imposes uniform sparsity, regardless of varying input complexity and expertise of experts, overlooking potential gains from fewer or more activations (Zhou et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib61 "")).
The Softmax operation forces a competitive probability distribution that sacrifices the absolute magnitude information of expert activations (Wang et al., [2024b](https://arxiv.org/html/2604.00801v1#bib.bib52 "")).

For load-balancing, existing strategies employ different balancing targets that force predetermined activation patterns, which are often mutually exclusive yet exhibit varying performance under different configurations (Fedus et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 ""); Zhou et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib61 ""); Muennighoff et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib40 "")).
Rigidly adhering to either may constrain the model’s ability to adaptively discover potentially better resource allocation patterns (Do et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib9 "")).

To bridge these gaps, we introduce Routing-Free MoE, a bottom-up architecture without any centralized routers, Softmax, TopK, or rigid load-balancing designs, where each expert individually and directly determines its own activation.
Our design enables each expert to activate itself purely when its internal confidence score surpasses a configurable threshold.
To satisfy efficiency requirements, we design a dynamic, configurable framework to adaptively achieve the sparsity and load-balancing objectives during training, allowing the optimal activation pattern to emerge spontaneously.
We utilize an auxiliary loss function that seamlessly integrates both token and expert balancing, providing flexibility to adaptively exploit both depending on training dynamics and workload requirements.

We validate Routing-Free MoE across three scales up to 0.8B with extensive experiments in comparison against standard MoE and strong baselines.
Across all settings, Routing-Free MoE consistently achieves better language modeling quality and downstream performance averaged across 9 evaluation benchmarks.
It also demonstrates notably improved scalability and robustness.
We analyze its training behavior in detail and further document intriguing phenomena for density and load-balancing, offering insights that may guide future improvements in MoE design and optimization.

![Refer to caption](https://arxiv.org/html/2604.00801v1/x2.png)Figure 2: Routing-Free MoE consistently outperforms standard MoE, AoE (Lv et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib38 "")), and ReMoE (Wang et al., [2024b](https://arxiv.org/html/2604.00801v1#bib.bib52 "")) in language modeling. All models are trained on OpenWebText (Gokaslan et al., [2019](https://arxiv.org/html/2604.00801v1#bib.bib19 "")) under identical environment conditions and best-performing configurations, as described in Section [4.1](https://arxiv.org/html/2604.00801v1#S4.SS1 "4.1 Experimental Setup ‣ 4 Experiments ‣ Routing-Free Mixture-of-Experts"). FLOPs are estimated for one epoch.

In summary, our key contributions are:

- •


A Routing-Free MoE architecture that eliminates routers, Softmax, TopK, and hard-coded load balancing mechanisms.

- •


A unified, adaptive load-balancing framework that jointly optimizes token-balancing and expert-balancing, allowing customizable application.

- •


Experiments and analyses demonstrating the improvements of Routing-Free MoE over baselines.


## 2 Preliminaries

Consider a standard MoE LLM(Fedus et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 ""); Jiang et al., [2024](https://arxiv.org/html/2604.00801v1#bib.bib29 "")) comprising LL transformer blocks.
For layer ℓ∈{1,⋯,L}\\ell\\in\\{1,\\cdots,L\\} and token sequence of length TT, the forward pass111LayerNorms and dropout are omitted for clarity. can be formulated as:

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝐱1:Tℓ\\displaystyle\\mathbf{x}^{\\ell}\_{1:T} | =SelfAttn​(𝐡1:Tℓ−1)+𝐡1:Tℓ−1,\\displaystyle=\\mathrm{SelfAttn}(\\mathbf{h}^{\\ell-1}\_{1:T})+\\mathbf{h}^{\\ell-1}\_{1:T}, |  | (1) |
|  | 𝐡tℓ\\displaystyle\\mathbf{h}^{\\ell}\_{t} | =MoE​(𝐱tℓ)+𝐱tℓ,\\displaystyle=\\mathrm{MoE}(\\mathbf{x}^{\\ell}\_{t})+\\mathbf{x}^{\\ell}\_{t}, |  | (2) |

where 𝐱1:Tℓ\\mathbf{x}^{\\ell}\_{1:T} denotes the attention module output with residual connection added. The MoE block performs a token-wise mapping, yielding output 𝐡tℓ\\mathbf{h}^{\\ell}\_{t} at token t∈{1,⋯,T}t\\in\\{1,\\cdots,T\\} with residual added.

The mainstream form of MoE LLMs incorporates multiple structurally identical FFN experts Ei​(⋅),i∈{1,⋯,N}E\_{i}(\\cdot),i\\in\\{1,\\cdots,N\\} within the MoE layer.
This is implemented via a token-wise routing mechanism among all NN experts at layer ℓ\\ell, with a gating network G​(⋅)G(\\cdot) assigning each token to a designated number KK of top-activated experts:

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | 𝐡\\displaystyle\\mathbf{h} | =∑i=1N(G​(𝐱)i​Ei​(𝐱))+𝐱,\\displaystyle=\\sum\\nolimits^{N}\_{i=1}\\left(G\\left(\\mathbf{x}\\right)\_{i}E\_{i}\\left(\\mathbf{x}\\right)\\right)+\\mathbf{x}, |  | (3) |
|  | G​(𝐱)\\displaystyle G\\left(\\mathbf{x}\\right) | =Softmax​(TopK​(𝐱𝐆,K)),\\displaystyle=\\mathrm{Softmax}\\left(\\mathrm{TopK}\\left(\\mathbf{x}\\mathbf{G},K\\right)\\right), |  | (4) |

where G​(⋅):ℝD↦ℝNG(\\cdot):\\mathbb{R}^{D}\\mapsto\\mathbb{R}^{N} denotes the routing mechanism, whose output serves as the weights of the weighted sum for outputs among all NN experts, and only KK of NN experts receive nonzero values.
The router learns its weight matrix 𝐆∈ℝD×N\\mathbf{G}\\in\\mathbb{R}^{D\\times N} that can be interpreted as a set of NN individual DD-dimensional expert vectors 𝐠i\\mathbf{g}\_{i}, each responding to a characteristic hidden state 𝐡i\\mathbf{h}\_{i} that should activate the corresponding expert EiE\_{i}, with 𝐬i=𝐱𝐠i∈ℝN\\mathbf{s}\_{i}=\\mathbf{xg}\_{i}\\in\\mathbb{R}^{N} denoting the token-to-expert affinity scores (Zhou et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib61 ""); Liu et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib36 "")).
TopK​(𝐬,K)\\mathrm{TopK}(\\mathbf{s},K) retains the top-KK scores and masks the rest to −∞-\\infty.

Each individual expert EiE\_{i} is implemented as a Feed-Forward Network (FFN).
Modern FFN typically takes the form of a Gated Linear Unit (GLU, Dauphin et al. ( [2017](https://arxiv.org/html/2604.00801v1#bib.bib7 "")); Shazeer et al. ( [2017](https://arxiv.org/html/2604.00801v1#bib.bib47 ""))):

|     |     |     |     |
| --- | --- | --- | --- |
|  | FFN​(𝐱)=\[σ​(𝐱𝐖up)⊙(𝐱𝐖gate)\]​𝐖down,\\mathrm{FFN}(\\mathbf{x})=\[\\sigma(\\mathbf{x}\\mathbf{W}\_{\\mathrm{up}})\\odot(\\mathbf{x}\\mathbf{W}\_{\\mathrm{gate}})\]\\mathbf{W}\_{\\mathrm{down}}, |  | (5) |

where σ\\sigma is the activation function and ⊙\\odot denotes element-wise multiplication.
In MoE LLMs, this design enables a second level of input-dependent information filtering, leading to potentially more effective representations.

The routing in standard MoE relies on several strong inductive biases.
With merely NN expert vectors 𝐠i∈ℝD\\mathbf{g}\_{i}\\in\\mathbb{R}^{D}, the router’s capacity is orders of magnitude smaller than the experts themselves, yet must compress the knowledge-intensive activation preferences of all NN experts into single dot-product scores without any direct signal about expert capabilities, improving only through indirect trial-and-error (Lv et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib38 "")).
TopK hard-codes a fixed sparsity ratio K/NK/N regardless of input complexity, preventing input-adaptive activation patterns (Zhou et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib61 "")).
Softmax discards absolute activation magnitudes by forcing a competitive probability distribution, suppressing the residual contribution of highly suitable experts when others also happen to score higher (Wang et al., [2024b](https://arxiv.org/html/2604.00801v1#bib.bib52 "")).

## 3 Methodology

### 3.1 Architecture

To alleviate the router bottleneck, Lv et al. ( [2025](https://arxiv.org/html/2604.00801v1#bib.bib38 "")) introduce Autonomy-of-Experts (AoE), with

|     |     |     |     |
| --- | --- | --- | --- |
|  | FFN​(𝐱)=\[σ​(𝐱𝐀gate​𝐁gate)⊙(𝐱𝐖up)\]​𝐖down,\\mathrm{FFN}(\\mathbf{x})=\[\\sigma(\\mathbf{x}\\mathbf{A}\_{\\mathrm{gate}}\\mathbf{B}\_{\\mathrm{gate}})\\odot(\\mathbf{x}\\mathbf{W}\_{\\mathrm{up}})\]\\mathbf{W}\_{\\mathrm{down}}, |  | (6) |

where 𝐀gate∈ℝD×r\\mathbf{A}\_{\\mathrm{gate}}\\in\\mathbb{R}^{D\\times r} projects the input hidden state 𝐱\\mathbf{x} from hidden dimension DD to a lower-dimensional rank r≪Dr\\ll D, and 𝐁gate∈ℝr×Dact\\mathbf{B}\_{\\mathrm{gate}}\\in\\mathbb{R}^{r\\times D\_{\\mathrm{act}}} projects it back.
This low-rank representation provides an alternative indicator of expert suitability that originates from within the expert itself rather than from an external router.
Each expert can therefore directly produce its own scalar activation score by applying a norm to its own internal representation ‖𝐱𝐀gate,i‖2\\\|\\mathbf{x}\\mathbf{A}\_{\\mathrm{gate},i}\\\|\_{2}.
However, AoE feeds the internal scores ‖𝐱𝐀gate‖2\\\|\\mathbf{xA}\_{\\mathrm{gate}}\\\|\_{2} back to the standard centralized TopK and Softmax routing pipeline:

|     |     |     |     |
| --- | --- | --- | --- |
|  | G​(𝐱)=Softmax​(TopK​(‖𝐱𝐀gate‖2,K)),\\displaystyle G(\\mathbf{x})=\\mathrm{Softmax}(\\mathrm{TopK}(\\\|\\mathbf{xA}\_{\\mathrm{gate}}\\\|\_{2},K)), |  | (7) |

thereby retaining the structural constraints and inductive biases of conventional routing.

Meanwhile, addressing the constraints of TopK and Softmax, Wang et al. ( [2024b](https://arxiv.org/html/2604.00801v1#bib.bib52 "")) propose ReMoE, which replaces TopK and Softmax with a single ReLU function applied directly to router’s output:

|     |     |     |     |
| --- | --- | --- | --- |
|  | G​(𝐱)=ReLU​(𝐱𝐆),G(\\mathbf{x})=\\mathrm{ReLU}(\\mathbf{x}\\mathbf{G}), |  | (8) |

The sparse activation arises naturally from ReLU without any explicit TopK selection or comparative normalization.
The absolute magnitude of router scores is preserved, allowing each expert’s residual contribution to be linearly weighted by router’s prediction, rather than a normalized relative preference.
Nevertheless, ReMoE still retains a centralized external router, preserving the information bottleneck and indirect optimization dynamics.

Building on these insights, our Routing-Free MoE seeks to eliminate all constraints of routing mechanisms.
We adopt AoE’s FFN design (Equation [6](https://arxiv.org/html/2604.00801v1#S3.E6 "In 3.1 Architecture ‣ 3 Methodology ‣ Routing-Free Mixture-of-Experts")) using each expert’s internal norm ‖𝐱𝐀gate‖2\\\|\\mathbf{xA}\_{\\mathrm{gate}}\\\|\_{2} as the initial expert preference score, grounding the activation decision in the expert’s own response to the input.
Since ‖𝐱𝐀gate‖2\\\|\\mathbf{xA}\_{\\mathrm{gate}}\\\|\_{2} is strictly non-negative unlike router’s 𝐱𝐆\\mathbf{xG}, we introduce a learnable per-expert bias term before activation upon ReMoE’s design, yielding

|     |     |     |     |
| --- | --- | --- | --- |
|  | Gi​(𝐱)=ReLU​(‖𝐱𝐀gate,i‖2−bi).G\_{i}(\\mathbf{x})=\\mathrm{ReLU}(\\\|\\mathbf{xA}\_{\\mathrm{gate},i}\\\|\_{2}-b\_{i}). |  | (9) |

With the per-expert bias term introduced, experts whose weighted norm falls below their own bias threshold contribute zero and are effectively deactivated, allowing each expert to jointly adapt both its 𝐀gate,i\\mathbf{A}\_{\\mathrm{gate},i} matrix and the bib\_{i} parameter to effectively adjust its own activation ratio.
We further introduce a global post-activation threshold θ\\theta as a configurable hyperparameter for external control over the overall sparsity level, giving the final binary activation decision of each expert:

|     |     |     |     |
| --- | --- | --- | --- |
|  | fi​(𝐱)=𝟙​{Gi​(𝐱)−θ≥0}.f\_{i}(\\mathbf{x})=\\mathds{1}\\left\\{G\_{i}(\\mathbf{x})-\\theta\\geq 0\\right\\}. |  | (10) |

The result is a fully decentralized architecture without any external router, TopK or Softmax, making each expert independently determine its own activation from within, allowing the global activation pattern to emerge bottom-up from collective self-adjustment of all experts.
Figure [1](https://arxiv.org/html/2604.00801v1#S0.F1 "Figure 1 ‣ Routing-Free Mixture-of-Experts") visualizes the architecture of Routing-Free MoE and its experts.

![Refer to caption](https://arxiv.org/html/2604.00801v1/x3.png)((a))Expert choice (EC) ensures hard expert-balancing and optimizes token-balancing via training.

![Refer to caption](https://arxiv.org/html/2604.00801v1/x4.png)((b))Token choice ensures hard token-balancing and optimizes expert-balancing via training.

![Refer to caption](https://arxiv.org/html/2604.00801v1/x5.png)((c))Routing-Free MoE adaptively optimizes both token-balancing and expert-balancing via training.

Figure 3: Load-balancing for tokens and experts. Routing-Free MoE introduces a unified load-balancing framework that simultaneously optimizes both expert-balancing and token-balancing through a configurable interpolation.

### 3.2 Training

Training MoE models requires simultaneously maintaining the activation ratio and balanced expert and token distribution.
Standard practice hard-codes TopK for the activation ratio, and addresses balancing via either _Token-Choice_ (TC) (Shazeer et al., [2017](https://arxiv.org/html/2604.00801v1#bib.bib47 ""); Fedus et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 "")) that guarantees per-token compute but not expert balance, or _Expert-Choice_ (EC) (Zhou et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib61 "")) that hard-codes uniform expert utilization but not per-token compute, as illustrated in Figure [3](https://arxiv.org/html/2604.00801v1#S3.F3 "Figure 3 ‣ 3.1 Architecture ‣ 3 Methodology ‣ Routing-Free Mixture-of-Experts").
Both enforce one balanced dimension as a hard constraint and optimizes the other via soft auxiliary loss.

As Routing-Free MoE has eliminated all centralized routing mechanisms, standard activation ratio and load-balancing strategies that rely on hard-coded TopK (Do et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib9 "")) no longer apply. We introduce a unified load-balancing framework by extending the auxiliary loss of Fedus et al. ( [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 "")) to jointly encourage both balanced token distribution across experts and balanced expert activation per token, without requiring any centralized mechanisms.

As the binary activation decision fif\_{i} in Equation [10](https://arxiv.org/html/2604.00801v1#S3.E10 "In 3.1 Architecture ‣ 3 Methodology ‣ Routing-Free Mixture-of-Experts") is non-differentiable, we directly use the pre-threshold activation weight Gi​(𝐱)G\_{i}(\\mathbf{x}) as a differentiable activation proxy of expert EiE\_{i}.
We define the mean activation density over a set of experts ℰ\\mathcal{E} and token batch ℬ\\mathcal{B} and its differentiable proxy as:

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | ρ​(ℰ,ℬ)\\displaystyle\\rho(\\mathcal{E},\\mathcal{B}) | =1\|ℰ\|​\|ℬ\|​∑ei∈ℰ∑𝐱∈ℬfi​(𝐱),\\displaystyle=\\frac{1}{\|\\mathcal{E}\|\|\\mathcal{B}\|}\\sum\_{e\_{i}\\in\\mathcal{E}}\\sum\_{\\mathbf{x}\\in\\mathcal{B}}f\_{i}(\\mathbf{x}), |  | (11) |
|  | ρ~​(ℰ,ℬ)\\displaystyle\\tilde{\\rho}(\\mathcal{E},\\mathcal{B}) | =1\|ℰ\|​\|ℬ\|​∑ei∈ℰ∑𝐱∈ℬGi​(𝐱).\\displaystyle=\\frac{1}{\|\\mathcal{E}\|\|\\mathcal{B}\|}\\sum\_{e\_{i}\\in\\mathcal{E}}\\sum\_{\\mathbf{x}\\in\\mathcal{B}}G\_{i}(\\mathbf{x}). |  | (12) |

Both quantities equal the target activation density ρ∞\\rho\_{\\infty} under perfectly uniform load.

We decompose the load-balancing objective into two complementary terms222We avoid using the terms expert choice and token choice in Routing-Free MoE because the notion of “choice” implies a centralized comparison and selection process, which contradicts our principle that all decisions should emerge locally and independently at individual experts and tokens..
The _expert-balancing_ loss ℒEB\\mathcal{L}\_{\\mathrm{EB}} encourages uniform distribution of tokens across experts by penalizing experts that consistently receive more or fewer tokens than average:

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒEB=1\|ℰ\|​∑ei∈ℰ(1\|ℬ\|​∑𝐱∈ℬfi​(𝐱))​(1\|ℬ\|​∑𝐱∈ℬGi​(𝐱)).\\mathcal{L}\_{\\mathrm{EB}}=\\frac{1}{\|\\mathcal{E}\|}\\sum\_{e\_{i}\\in\\mathcal{E}}\\left(\\frac{1}{\|\\mathcal{B}\|}\\sum\_{\\mathbf{x}\\in\\mathcal{B}}f\_{i}(\\mathbf{x})\\right)\\left(\\frac{1}{\|\\mathcal{B}\|}\\sum\_{\\mathbf{x}\\in\\mathcal{B}}G\_{i}(\\mathbf{x})\\right). |  | (13) |

The _token-balancing_ loss ℒTB\\mathcal{L}\_{\\mathrm{TB}} encourages uniform distribution
of activated experts per token by penalizing tokens that consistently activate more or fewer experts than average:

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒTB=1\|ℬ\|​∑𝐱∈ℬ(1\|ℰ\|​∑ei∈ℰfi​(𝐱))​(1\|ℰ\|​∑ej∈ℰGj​(𝐱)).\\mathcal{L}\_{\\mathrm{TB}}=\\frac{1}{\|\\mathcal{B}\|}\\sum\_{\\mathbf{x}\\in\\mathcal{B}}\\left(\\frac{1}{\|\\mathcal{E}\|}\\sum\_{e\_{i}\\in\\mathcal{E}}f\_{i}(\\mathbf{x})\\right)\\left(\\frac{1}{\|\\mathcal{E}\|}\\sum\_{e\_{j}\\in\\mathcal{E}}G\_{j}(\\mathbf{x})\\right). |  | (14) |

Each loss is a dot product between a binary non-differentiable term and a differentiable proxy, and is minimized when both factors equal ρ∞\\rho\_{\\infty} uniformly333See Appendix [A](https://arxiv.org/html/2604.00801v1#A1 "Appendix A Load-Balancing Losses ‣ Routing-Free Mixture-of-Experts")..
The two objectives are combined via a configurable interpolation parameter μ∈\[0,1\]\\mu\\in\[0,1\]:

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒLB=μ​ℒEB+(1−μ)​ℒTB.\\mathcal{L}\_{\\mathrm{LB}}=\\mu\\,\\mathcal{L}\_{\\mathrm{EB}}+(1-\\mu)\\,\\mathcal{L}\_{\\mathrm{TB}}. |  | (15) |

Setting μ=1\\mu=1 recovers a pure expert-balancing, and μ=0\\mu=0
recovers pure token-balancing.
This provides a single unified framework that interpolates both routing paradigms, allowing load-balancing to be tailored to specific deployment needs.

Rather than fixing the auxiliary loss coefficient λ\\lambda as a static hyperparameter, we follow Wang et al. ( [2024b](https://arxiv.org/html/2604.00801v1#bib.bib52 "")) and
adopt an adaptive schedule that drives the empirical activation ratio toward the target ρ∞\\rho\_{\\infty} throughout training. The total training
objective is

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒ=ℒLM+λt​ℒLB,\\mathcal{L}=\\mathcal{L}\_{\\mathrm{LM}}+\\lambda\_{t}\\,\\mathcal{L}\_{\\mathrm{LB}}, |  | (16) |

with λt\\lambda\_{t} updated at each training step tt as:

|     |     |     |     |
| --- | --- | --- | --- |
|  | λt+1=λt⋅(1+η)sign​(ρt​(ℰ,ℬ)−ρ∞).\\lambda\_{t+1}=\\lambda\_{t}\\cdot\\left(1+\\eta\\right)^{\\mathrm{sign}\\left(\\rho\_{t}(\\mathcal{E},\\mathcal{B})-\\rho\_{\\infty}\\right)}. |  | (17) |

When the current density ρt\\rho\_{t} exceeds the target ρ∞\\rho\_{\\infty}, λt\\lambda\_{t}
increases to exert stronger load-balancing pressure; when below, it decreases to allow more expert activations.
The step size η\\eta controls the responsiveness of this feedback loop. This formulation avoids the need to manually tune λ\\lambda while ensuring the model converges to the desired computational budget, and is compatible with both training from scratch and adaptation of pretrained MoE models into Routing-Free MoE.

To encourage all experts to participate during the early warm-up phase of training, each expert’s bias is initialized as 1​e−61\\mathrm{e}^{-6}, allowing all experts to become activated, jointly exploring the representation space and establishing initial specialization before sparsity is enforced.
As training progresses and
λ\\lambda increases, the sparsity regularization gradually strengthens, driving the activation density toward target ρ∞\\rho\_{\\infty}.
By this stage, experts have already developed distinct roles, avoiding expert collapse and ensuring a stable and effective phase transition.

Table 1: Evaluation results for standard MoE baseline and Routing-Free MoE across scales and benchmarks. Each model is trained on OpenWebText (Gokaslan et al., [2019](https://arxiv.org/html/2604.00801v1#bib.bib19 "")) from scratch for one epoch. Details in Appendix [E](https://arxiv.org/html/2604.00801v1#A5 "Appendix E Additional Experiment Results ‣ Routing-Free Mixture-of-Experts").

Arch.SizeFLOPsPPL↓\\downarrowPIQAHellaSWinoGARCeARCcOBQAQQPQNLISST-2Avg.SMoE92.44M90.93M31.2257.5627.1951.2233.4221.3324.6036.8249.4649.0838.96AoE93.85M88.57M30.0056.0927.0150.2033.8421.9325.0036.8249.4649.0838.82ReMoE92.44M90.93M29.6056.5826.6951.6233.3822.2726.0036.8349.4849.0839.10RFMoE95.32M91.08M27.4258.4927.0950.5935.7721.5924.4036.9849.4453.5639.77MMoE289.9M248.0M25.0058.3228.2652.3336.2021.4224.6036.8549.4649.3139.64RFMoE307.3M249.2M22.0858.9227.8549.7235.2721.5026.4037.0649.5157.3440.40LMoE808.4M608.4M24.5859.1929.3750.8336.4123.2925.0037.5749.2849.0840.00RFMoE870.6M613.2M19.9758.8728.2750.5137.4621.6726.6039.9349.8653.6740.76

## 4 Experiments

We conduct comprehensive experiments to validate the performance of Routing-Free MoE.

### 4.1 Experimental Setup

We implement Routing-Free MoE upon the HuggingFace implementation444 [https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models/mixtral](https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models/mixtral "") for the Mixtral architecture (Jiang et al., [2024](https://arxiv.org/html/2604.00801v1#bib.bib29 "")), which includes attention designed as Grouped Query Attention (GQA, Ainslie et al. ( [2023](https://arxiv.org/html/2604.00801v1#bib.bib1 ""))) with Rotary Position Embeddings (RoPE, Su et al. ( [2024](https://arxiv.org/html/2604.00801v1#bib.bib48 ""))), FFN using SwiGLU activation (Shazeer, [2020](https://arxiv.org/html/2604.00801v1#bib.bib46 "")), and Root Mean Square Layer Normalization (RMSNorm, Zhang and Sennrich ( [2019](https://arxiv.org/html/2604.00801v1#bib.bib59 ""))) applied to the residual stream prior to each attention and MoE FFN (Xiong et al., [2020](https://arxiv.org/html/2604.00801v1#bib.bib55 "")).

We conduct experiments across three scales (S/M/L) up to 0.8B, and compare against a standard MoE baseline with identical structure except for the routing mechanism.
AoE (Lv et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib38 "")) and ReMoE (Wang et al., [2024b](https://arxiv.org/html/2604.00801v1#bib.bib52 "")) are also examined as both additional baselines and ablated variants for comparison.
Detailed hyperparameters and training configurations are in Table [8](https://arxiv.org/html/2604.00801v1#A5.T8 "Table 8 ‣ Appendix E Additional Experiment Results ‣ Routing-Free Mixture-of-Experts") in Appendix.

Following standard practices, all models are trained on OpenWebText (Gokaslan et al., [2019](https://arxiv.org/html/2604.00801v1#bib.bib19 "")) for one epoch.
Zero-shot evaluation is conducted across 9 benchmarks (Gao et al., [2024b](https://arxiv.org/html/2604.00801v1#bib.bib17 "")).
We include PIQA (Bisk et al., [2020](https://arxiv.org/html/2604.00801v1#bib.bib2 "")), HellaSwag (Zellers et al., [2019](https://arxiv.org/html/2604.00801v1#bib.bib58 "")), WinoGrande (Sakaguchi et al., [2021](https://arxiv.org/html/2604.00801v1#bib.bib45 "")), ARC-Easy and ARC-Challenge (Clark et al., [2018](https://arxiv.org/html/2604.00801v1#bib.bib3 "")), OpenBookQA (Mihaylov et al., [2018](https://arxiv.org/html/2604.00801v1#bib.bib39 "")), and QQP, QNLI, and SST-2 from GLUE (Wang et al., [2018](https://arxiv.org/html/2604.00801v1#bib.bib50 "")). Accuracy is reported on WinoGrande, QQP, QNLI, and SST-2, and normalized accuracy on others.

### 4.2 Main Results

Table [1](https://arxiv.org/html/2604.00801v1#S3.T1 "Table 1 ‣ 3.2 Training ‣ 3 Methodology ‣ Routing-Free Mixture-of-Experts") presents language modeling perplexity and downstream benchmark evaluation results across all three scales under an iso-compute comparison, with FLOPs matched within ∼\\sim1% between MoE and Routing-Free MoE at each scale.
Figure [2](https://arxiv.org/html/2604.00801v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Routing-Free Mixture-of-Experts") shows the detailed validation perplexity evolution with total FLOPs throughout training.
Routing-Free MoE achieves consistently better validation perplexity than standard MoE across all scales.
On downstream benchmarks, Routing-Free MoE also consistently improves the average performance, with a detailed statistical analysis (Appendix [C](https://arxiv.org/html/2604.00801v1#A3 "Appendix C Statistical Significance Analysis ‣ Routing-Free Mixture-of-Experts")) supporting that Routing-Free MoE can produce a statistically significant improvement over the standard MoE baseline.
Notably, the gain of Routing-Free MoE does not diminish with scale, suggesting that the benefits of our approach can continue to provide gains as models scale up.

![Refer to caption](https://arxiv.org/html/2604.00801v1/x6.png)((a))Density ρ\\rho, density target ρ∞\\rho\_{\\infty}, and density proxy ρ~\\tilde{\\rho}

![Refer to caption](https://arxiv.org/html/2604.00801v1/x7.png)((b))Adaptive coefficient λ\\lambda and load-balancing loss ℒLB\\mathcal{L}\_{\\mathrm{LB}}

![Refer to caption](https://arxiv.org/html/2604.00801v1/x8.png)((c))Training ℒ\\mathcal{L}, language modeling ℒLM\\mathcal{L}\_{\\mathrm{LM}}, and regularization λ​ℒLB\\lambda\\mathcal{L}\_{\\mathrm{LB}}

Figure 4: Training dynamics of Routing-Free MoE at scale S, with r=16r=16, λ0=1​e−10\\lambda\_{0}=1\\mathrm{e}^{-10}, η=0.02\\eta=0.02, and α=1​e−3\\alpha=1\\mathrm{e}^{-3}.Table 2: Ablation performed at scale S. Detailed configurations for each run are provided in Table [E](https://arxiv.org/html/2604.00801v1#A5 "Appendix E Additional Experiment Results ‣ Routing-Free Mixture-of-Experts").

| Config. | Size | FLOPs | PPL↓\\downarrow |
| --- | --- | --- | --- |
| Standard MoE | 92.44M | 90.93M | 31.22 |
| w/o router (AoE, rr=16) | 93.85M | 88.57M | 30.00 |
| w/o router (AoE, rr=32) | 95.32M | 91.08M | 30.31 |
| w/o TopK&Softmax (ReMoE) | 92.44M | 90.93M | 29.60 |
| Routing-Free MoE (rr=16) | 93.85M | 88.57M | 28.73 |
| Routing-Free MoE (rr=32) | 95.32M | 91.08M | 28.33 |

### 4.3 Architectural Comparison

We experiment at scale S, decomposing the contribution of each architectural change over the standard MoE via incremental ablation, with main results shown in Table [2](https://arxiv.org/html/2604.00801v1#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Routing-Free Mixture-of-Experts").

Routing-Free MoE outperforms either by a substantial margin under matching size and FLOPs.
A visualization of perplexity curves during training for AoE and ReMoE can also be found in Figure [2](https://arxiv.org/html/2604.00801v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Routing-Free Mixture-of-Experts").

Table 3: Effect of low-rank projection dimension at scale S, with learning rate α=1​e−3\\alpha=1\\mathrm{e}^{-3}.

| 𝒓\\bm{r} | Size | FLOPs | PPL↓\\downarrow |
| --- | --- | --- | --- |
| 8 | 93.11M | 87.32M | 29.16 |
| 16 | 93.85M | 88.57M | 28.74 |
| 32 | 95.32M | 91.08M | 28.34 |
| 64 | 98.27M | 96.09M | 28.24 |

Table [3](https://arxiv.org/html/2604.00801v1#S4.T3 "Table 3 ‣ 4.3 Architectural Comparison ‣ 4 Experiments ‣ Routing-Free Mixture-of-Experts") reports the effect of rr. Increasing rr consistently improves perplexity, yet with diminishing returns. Therefore, we set 32 as the default rr and scale it proportionally with the hidden dimension DD across experiments on other scales for the optimal efficiency–quality trade-off.

A detailed analysis about Routing-Free MoE at deployment is provided in Appendix [B](https://arxiv.org/html/2604.00801v1#A2 "Appendix B Routing-Free MoE at Deployment ‣ Routing-Free Mixture-of-Experts"), including efficiency improvements by expert parallelism and the effects of adapting threshold θ\\theta at inference.

### 4.4 Training Dynamics

Figure [4](https://arxiv.org/html/2604.00801v1#S4.F4 "Figure 4 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Routing-Free Mixture-of-Experts") visualizes the training dynamics of Routing-Free MoE.
The empirical activation density ρ\\rho begins near 1 at
initialization, drops sharply to the target ρ∞\\rho\_{\\infty} as λ\\lambda grows, and remains close to the target thereafter.
The differentiable proxy ρ~\\tilde{\\rho} also settles slightly above ρ∞\\rho\_{\\infty} at convergence.
λ\\lambda rises steeply during the initial warm-up phase, plateaus as ρ\\rho converges to ρ∞\\rho\_{\\infty}, and collapses sharply near the end as density falls marginally below the target.
The regularization term λ​ℒLB\\lambda\\mathcal{L}\_{\\mathrm{LB}} exhibits a transient spike caused by the exponential growth of λ\\lambda, with its magnitude rising above 10−110^{-1}, making it a non-negligible contributor to the total loss ℒ\\mathcal{L} alongside ℒLM\\mathcal{L}\_{\\mathrm{LM}}, and steering gradient descent toward directions that also reduce ℒLB\\mathcal{L}\_{\\mathrm{LB}}.
As training progresses, λ​ℒLB\\lambda\\mathcal{L}\_{\\mathrm{LB}} decays to a negligible level, demonstrating how load‑balancing pressure is applied precisely when needed without persistently distorting the language‑modeling optimization objectives.

Figure [5](https://arxiv.org/html/2604.00801v1#S4.F5 "Figure 5 ‣ 4.4 Training Dynamics ‣ 4 Experiments ‣ Routing-Free Mixture-of-Experts") further examines training stability by sweeping the learning rate α\\alpha. Under conservative α\\alpha, both models converge smoothly with Routing-Free MoE consistently achieving lower perplexity. As α\\alpha increases, the standard MoE baseline begins to collapse much earlier than Routing-Free MoE. This is evident at scale S, where MoE fails at α=2​e−3\\alpha=2\\mathrm{e}^{-3} while Routing-Free MoE remains stable throughout training.
In addition, tuning α\\alpha for the standard MoE at scale L yields only marginal gains over its scale-M performance (as in Figure [2](https://arxiv.org/html/2604.00801v1#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Routing-Free Mixture-of-Experts")); whereas Routing-Free MoE continues to improve as scale increases, highlighting its enhanced scalability.
Routing-Free MoE not only outperforms the baseline, but also exhibits better training stability and less sensitivity to learning rates, tolerating a wider range of hyperparameter choices at scale.

![Refer to caption](https://arxiv.org/html/2604.00801v1/x9.png)((a))Training curves with different α\\alpha at scale S

![Refer to caption](https://arxiv.org/html/2604.00801v1/x10.png)((b))Training curves with different α\\alpha at scale L

Figure 5: Training dynamics of Routing-Free MoE by α\\alpha, with r=16r=16, λ0=1​e−10\\lambda\_{0}=1\\mathrm{e}^{-10}, and η=0.02\\eta=0.02.![Refer to caption](https://arxiv.org/html/2604.00801v1/x11.png)Figure 6: Evolution of expert activation across layers during training on OpenWebText under global density target.
Left panel shows the per‑layer mean activation as lines, along with IQR as shaded band regions using 1,000‑step moving average smoothing.
Right panel shows activation distribution at the final training step. Model at scale S.

## 5 Discussion

Beyond its superior performance compared with the baselines, here we analyze some intriguing properties of Routing-Free MoE.

### 5.1 Per-Layer and Global Density

With Top-K at each layer eliminated, a natural question is whether the target activation density ρ∞\\rho\_{\\infty} should still be enforced within each layer, or only globally across all experts.
Per-layer enforcement computes ℒL​B\\mathcal{L}\_{LB} separately at each layer, continuing to impose the inductive bias for a uniform sparsity at every depth.
A global ρ∞\\rho\_{\\infty}, in contrast, relaxes the constraint and allows individual layers to deviate from ρ∞\\rho\_{\\infty} so long as aggregate activation matches the target and improves the overall loss ℒ\\mathcal{L}.
Our experiment on scale S finds that relaxing this inductive bias leads to a significant improvement with perplexity drops from 39.44 to 28.74.
In Figure [6](https://arxiv.org/html/2604.00801v1#S4.F6 "Figure 6 ‣ 4.4 Training Dynamics ‣ 4 Experiments ‣ Routing-Free Mixture-of-Experts") we illustrate the emergent expert activation pattern that Routing-Free MoE develops by itself during training when this per-layer bias is removed.
A detailed analysis on this matter is provided in Appendix [D.2](https://arxiv.org/html/2604.00801v1#A4.SS2 "D.2 Per-Layer and Global Density ‣ Appendix D Additional Discussion ‣ Routing-Free Mixture-of-Experts").
Enforcing identical sparsity at every depth suppresses the compute‑hungry layers that naturally benefit from activating more experts, while simultaneously forcing unnecessary activations in layers where sparse representations suffice. Once this bias is lifted, the model is free to self‑organize into a more effective, functionally aligned activation structure.

Table 4: Effect of load balancing interpolation μ\\mu. Throughput is measured as samples per second.

|  | 𝝁\\bm{\\mu} | PPL↓\\downarrow | Eval. Throughput ↑\\uparrow |
| --- | --- | --- | --- |
| only TB | 0.0 | 28.41 | 645.7 |
|  | 0.2 | 28.35 | 648.3 |
| balanced | 0.5 | 28.34 | 662.3 |
|  | 0.8 | 28.38 | 643.9 |
| only EB | 1.0 | 28.43 | 648.8 |

![Refer to caption](https://arxiv.org/html/2604.00801v1/x12.png)((a))μ=0.0\\mu=0.0

![Refer to caption](https://arxiv.org/html/2604.00801v1/x13.png)((b))μ=0.5\\mu=0.5

![Refer to caption](https://arxiv.org/html/2604.00801v1/x14.png)((c))μ=1.0\\mu=1.0

Figure 7: Expert activation heatmaps on different input for representative layers at scale S. Each row shows the average expert activation ratios on a given benchmark. Darker colors indicate more frequent activation.

### 5.2 Token and Expert Balancing

A key contribution of this work is demonstrating empirically that token- and expert-balancing strategies can be combined in a complementary manner via soft interpolation rather than being treated as mutually exclusive.
Prior work (Zhou et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib61 ""); Muennighoff et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib40 "")) documented that EC and TC perform better under different configurations, but our unified framework provides a mechanism to adaptively balance both objectives.

Table [4](https://arxiv.org/html/2604.00801v1#S5.T4 "Table 4 ‣ 5.1 Per-Layer and Global Density ‣ 5 Discussion ‣ Routing-Free Mixture-of-Experts") shows that μ=0.5\\mu=0.5 achieves the lowest perplexity and highest throughput, which degrades toward either direction.
Token-balancing enables flexible compute allocation based on input characteristics, while expert-balancing ensures uniform expert utilization and encourages specialization; the two objectives address orthogonal failure modes and jointly produce better outcomes than either alone.
This complementarity is illustrated in Figure [7](https://arxiv.org/html/2604.00801v1#S5.F7 "Figure 7 ‣ 5.1 Per-Layer and Global Density ‣ 5 Discussion ‣ Routing-Free Mixture-of-Experts"). Under token-only balancing (Figure [7(a)](https://arxiv.org/html/2604.00801v1#S5.F7.sf1 "In Figure 7 ‣ 5.1 Per-Layer and Global Density ‣ 5 Discussion ‣ Routing-Free Mixture-of-Experts")), the absence of expert balancing leads to larger activation probability differences across the expert axis, with a few experts activated far more frequently than others. Conversely, under expert-only balancing (Figure [7(c)](https://arxiv.org/html/2604.00801v1#S5.F7.sf3 "In Figure 7 ‣ 5.1 Per-Layer and Global Density ‣ 5 Discussion ‣ Routing-Free Mixture-of-Experts")), expert loads are more horizontally uniform, but activation patterns vary notably across benchmarks, indicating that without token-level regularization, expert activation becomes overly sensitive to the input domain distribution. When both objectives are active (μ=0.5\\mu=0.5, Figure [7(b)](https://arxiv.org/html/2604.00801v1#S5.F7.sf2 "In Figure 7 ‣ 5.1 Per-Layer and Global Density ‣ 5 Discussion ‣ Routing-Free Mixture-of-Experts")), the activation pattern is more uniform along both axes and more consistent across inputs.

## 6 Related Work

### 6.1 MoE Foundations

Mixture-of-Experts was introduced as adaptive mixtures of local experts (Jacobs et al., [1991](https://arxiv.org/html/2604.00801v1#bib.bib28 ""); Jordan and Jacobs, [1994](https://arxiv.org/html/2604.00801v1#bib.bib31 "")) and later scaled to deep networks (Eigen et al., [2013](https://arxiv.org/html/2604.00801v1#bib.bib13 ""); Shazeer et al., [2017](https://arxiv.org/html/2604.00801v1#bib.bib47 "")). Interpretability studies reveal that FFNs in transformers capture knowledge with sparse activations (Geva et al., [2021](https://arxiv.org/html/2604.00801v1#bib.bib18 ""); Dai et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib5 ""); Dalvi et al., [2019](https://arxiv.org/html/2604.00801v1#bib.bib6 ""); Durrani et al., [2020](https://arxiv.org/html/2604.00801v1#bib.bib12 ""); Gurnee et al., [2023](https://arxiv.org/html/2604.00801v1#bib.bib24 "")), motivating MoE designs that activate only a subset of parameters (Liu et al., [2023](https://arxiv.org/html/2604.00801v1#bib.bib37 "")). Modern MoE architectures have since been deployed at scale (Lepikhin et al., [2020](https://arxiv.org/html/2604.00801v1#bib.bib34 ""); Fedus et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 ""); Zoph, [2022](https://arxiv.org/html/2604.00801v1#bib.bib62 ""); Komatsuzaki et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib33 ""); Rajbhandari et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib44 ""); Du et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib11 "")), with frontier models with over billions of parameters (Jiang et al., [2024](https://arxiv.org/html/2604.00801v1#bib.bib29 ""); Dai et al., [2024](https://arxiv.org/html/2604.00801v1#bib.bib4 ""); Grok, [2024](https://arxiv.org/html/2604.00801v1#bib.bib21 "")). Structural enhancements such as shared experts (Gou et al., [2023](https://arxiv.org/html/2604.00801v1#bib.bib20 ""); Dai et al., [2024](https://arxiv.org/html/2604.00801v1#bib.bib4 "")) further help reduce parameter redundancy.

### 6.2 Routing Mechanisms

Traditional MoE relies on centralized routers, learned linear projections followed by TopK selection (Lepikhin et al., [2020](https://arxiv.org/html/2604.00801v1#bib.bib34 ""); Fedus et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 "")).
Recent work has moved toward relaxing the routing mechanisms. Lv et al. ( [2025](https://arxiv.org/html/2604.00801v1#bib.bib38 "")) replaces the router with expert-internal scoring; Wang et al. ( [2024b](https://arxiv.org/html/2604.00801v1#bib.bib52 "")) replaces Softmax and TopK with ReLU gating.
Other approaches include using vector quantization for expert assignment (Do et al., [2024](https://arxiv.org/html/2604.00801v1#bib.bib10 "")), virtual shared experts ( [Wu et al.,](https://arxiv.org/html/2604.00801v1#bib.bib54 "")), using pretrained language models as routers (Liu and Lo, [2025](https://arxiv.org/html/2604.00801v1#bib.bib35 "")), and with ternary expert expansion (Yan et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib56 "")).
Huang et al. ( [2024](https://arxiv.org/html/2604.00801v1#bib.bib27 "")) propose adjusting expert counts based on input difficulty.
Do et al. ( [2025](https://arxiv.org/html/2604.00801v1#bib.bib9 "")) utilize global TopK selection over a combined similarity score to unify token- and expert-choice routing.

### 6.3 Load Balancing and Training

Token Choice (Shazeer et al., [2017](https://arxiv.org/html/2604.00801v1#bib.bib47 ""); Fedus et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 "")) guarantees per-token compute but not expert balance.
Expert Choice (Zhou et al., [2022](https://arxiv.org/html/2604.00801v1#bib.bib61 "")) guarantees expert load balance, but may potentially yield suboptimal matching.
Studies has shown that Token Choice and Expert Choice balancing are complementary rather than mutually exclusive (Muennighoff et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib40 "")).
Recent advances include auxiliary-loss-free balancing via dynamic bias (Wang et al., [2024a](https://arxiv.org/html/2604.00801v1#bib.bib51 ""); DeepSeek-AI et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib8 "")), similarity-preserving routers (Omi et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib41 "")), expert specialization through orthogonality losses (Guo et al., [2025b](https://arxiv.org/html/2604.00801v1#bib.bib23 ""); Feng et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib15 "")), and infrastructure-level scheduling (Zhao et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib60 ""); Yu et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib57 "")). For training stability, TopK selection’s discontinuity prevents gradient flow to non-selected experts; Wang et al. ( [2024b](https://arxiv.org/html/2604.00801v1#bib.bib52 "")) address this with differentiable ReLU gating and adaptive regularization. Qiu et al. ( [2025](https://arxiv.org/html/2604.00801v1#bib.bib43 "")) provide practical guidance on auxiliary losses, while He et al. ( [2024](https://arxiv.org/html/2604.00801v1#bib.bib25 "")) and Pan et al. ( [2024](https://arxiv.org/html/2604.00801v1#bib.bib42 "")) address auxiliary losses at inference.
Do et al. ( [2025](https://arxiv.org/html/2604.00801v1#bib.bib9 "")) propose a unified scoring function with global TopK selection that linearly combines TC and EC similarity scores, which addresses representation collapse and token dropping simultaneously, but is not applicable without routing mechanisms.
Our approach, instead, seeks unifying token-balancing and expert-balancing under fully decentralized expert self-activation.

## 7 Conclusion

We present Routing-Free Mixture-of-Experts, an MoE architecture that entirely eliminates centralized routing mechanisms, along with a unified adaptive load-balancing framework that jointly optimizes token- and expert-balancing objectives during training.
Experiments across 3 scales and 9 benchmarks validate that Routing-Free MoE consistently outperforms baselines, with improved scalability and robustness.
We further analyze the load-balancing behavior throughout training and the expert activation patterns.
Routing-Free MoE opens a path toward more flexible and efficient architectures.
We hope this work encourages future exploration of MoE design and optimization.

## Limitations

Our experiments are conducted exclusively on models trained from scratch on OpenWebText at scales up to 0.8B parameters.
While results are consistent across three scales, it remains an open question whether the findings generalize to even larger-scale pretraining regimes, which lie beyond our current resource limits.
Similarly, we only evaluate on 9 English benchmarks for commonsense reasoning and natural language inference which are still limited in scope and may not capture the full range of applications present in real-world applications.
Given that our method follows standard training practices common among a broad body of peer-reviewed research in this field, it is highly plausible that the observed trends will carry over to broader settings.

Another practical direction we have not explored is converting an existing pretrained standard MoE model into a Routing-Free MoE via continued pretraining or training-free adaptation, rather than training from scratch.
Such a conversion could possibly further reduce the compute cost of adopting our approach, making it a promising avenue for future work; yet it introduces a vast set of distinct challenges that lies beyond the scope of this work.

## Ethics Consideration

Routing‑Free MoE is proposed as a general purpose language model architecture and does not introduce domain‑specific risks beyond those inherent to language model pretraining. However, as with any method that improves model efficiency and scalability, it may lower the barrier to training more capable models, with corresponding societal implications that merit careful consideration in downstream applications.

The OpenWebText dataset used in our experiments may also reflect demographic or geographic biases present in large‑scale web corpora. We did not conduct dedicated bias or safety audits, and the absence of such analyses means that potential fairness, privacy, or safety issues may persist.
It is therefore incumbent upon downstream users and deployers to perform appropriate task‑specific fairness, privacy, and safety evaluations before any real‑world deployment. We disclaim responsibility for unintended consequences arising from downstream use that involve applying models trained using our approach.

## Acknowledgments

The authors gratefully acknowledge the scientific support and HPC resources provided by the Karlsruhe Institute of Technology National High Performance Computing Center (NHR@KIT) under the NHR projects 22189, 22560, and 24767.
The HoreKa supercomputer at NHR@KIT is funded by the Ministry of Science, Research and the Arts Baden-Württemberg and by the Federal Ministry of Education and Research of Germany.
This work is also partially supported by funding from Munich Center for Machine Learning (MCML).
We also acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO under project EHPC-AI-2024A06-060, hosted by CINECA (Italy) and the LEONARDO consortium through a EuroHPC Regular Access call.
In addition, the authors thank Xu He and Alinur Kozhanov from Technical University of Munich for their contributions during the early stages of this work.

## References

- Ainslie et al. (2023)
Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023.

[Gqa: Training generalized multi-query transformer models from multi-head checkpoints](https://aclanthology.org/2023.emnlp-main.298/ "").

In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, pages 4895–4901.

- Bisk et al. (2020)
Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. 2020.

[Piqa: Reasoning about physical commonsense in natural language](https://ojs.aaai.org/index.php/AAAI/article/view/6239 "").

In _Proceedings of the AAAI conference on artificial intelligence_, volume 34, pages 7432–7439.

- Clark et al. (2018)
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018.

[Think you have solved question answering? try arc, the ai2 reasoning challenge](https://api.semanticscholar.org/CorpusID:3922816 "").

_ArXiv_, abs/1803.05457.

- Dai et al. (2024)
Damai Dai, Chengqi Deng, Chenggang Zhao, R.x. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y.k. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024.

[DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models](https://doi.org/10.18653/v1/2024.acl-long.70 "").

In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 1280–1297, Bangkok, Thailand. Association for Computational Linguistics.

- Dai et al. (2022)
Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022.

[Knowledge neurons in pretrained transformers](https://aclanthology.org/2022.acl-long.581/ "").

In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 8493–8502.

- Dalvi et al. (2019)
Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Yonatan Belinkov, Anthony Bau, and James Glass. 2019.

[What is one grain of sand in the desert? analyzing individual neurons in deep nlp models](https://ojs.aaai.org/index.php/AAAI/article/view/4592 "").

In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 33, pages 6309–6317.

- Dauphin et al. (2017)
Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017.

[Language modeling with gated convolutional networks](http://proceedings.mlr.press/v70/dauphin17a.html "").

In _International conference on machine learning_, pages 933–941. PMLR.

- DeepSeek-AI et al. (2025)
DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, and 181 others. 2025.

[Deepseek-v3 technical report](https://arxiv.org/abs/2412.19437 "").

- Do et al. (2025)
Giang Do, Hung Le, and Truyen Tran. 2025.

[Unified sparse mixture of experts](https://arxiv.org/abs/2503.22996 "").

_arXiv preprint arXiv:2503.22996_.

- Do et al. (2024)
Giang Do, Kha Pham, Hung Le, and Truyen Tran. 2024.

[On the role of discrete representation in sparse mixture of experts](https://arxiv.org/abs/2411.19402 "").

_arXiv preprint arXiv:2411.19402_.

- Du et al. (2022)
Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P Bosma, Zongwei Zhou, Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, and 8 others. 2022.

[GLaM: Efficient scaling of language models with mixture-of-experts](https://proceedings.mlr.press/v162/du22c.html "").

In _Proceedings of the 39th International Conference on Machine Learning_, volume 162 of _Proceedings of Machine Learning Research_, pages 5547–5569. PMLR.

- Durrani et al. (2020)
Nadir Durrani, Hassan Sajjad, Fahim Dalvi, and Yonatan Belinkov. 2020.

[Analyzing individual neurons in pre-trained language models](https://aclanthology.org/2020.emnlp-main.395/ "").

In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_, pages 4865–4880.

- Eigen et al. (2013)
David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever. 2013.

[Learning factored representations in a deep mixture of experts](https://arxiv.org/abs/1312.4314 "").

_arXiv preprint arXiv:1312.4314_.

- Fedus et al. (2022)
William Fedus, Barret Zoph, and Noam Shazeer. 2022.

[Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity](https://www.jmlr.org/papers/v23/21-0998.html "").

_Journal of Machine Learning Research_, 23(120):1–39.

- Feng et al. (2025)
Yuchen Feng, Bowen Shen, Naibin Gu, Jiaxuan Zhao, Peng Fu, Zheng Lin, and Weiping Wang. 2025.

[Dive into moe: Diversity-enhanced reconstruction of large language models from dense into mixture-of-experts](https://aclanthology.org/2025.acl-long.951/ "").

In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 19375–19394.

- Gao et al. (2024a)
Chongyang Gao, Kezhen Chen, Jinmeng Rao, Baochen Sun, Ruibo Liu, Daiyi Peng, Yawen Zhang, Xiaoyuan Guo, Jie Yang, and VS Subrahmanian. 2024a.

[Higher layers need more lora experts](https://arxiv.org/abs/2402.08562 "").

_arXiv preprint arXiv:2402.08562_.

- Gao et al. (2024b)
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024b.

[The language model evaluation harness](https://doi.org/10.5281/zenodo.12608602 "").

- Geva et al. (2021)
Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021.

[Transformer feed-forward layers are key-value memories](https://aclanthology.org/2021.emnlp-main.446/ "").

In _Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing_, pages 5484–5495.

- Gokaslan et al. (2019)
Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. 2019.

Openwebtext corpus.

[http://Skylion007.github.io/OpenWebTextCorpus](http://skylion007.github.io/OpenWebTextCorpus "").

- Gou et al. (2023)
Yunhao Gou, Zhili Liu, Kai Chen, Lanqing Hong, Hang Xu, Aoxue Li, Dit-Yan Yeung, James T Kwok, and Yu Zhang. 2023.

[Mixture of cluster-conditional lora experts for vision-language instruction tuning](https://arxiv.org/abs/2312.12379 "").

_arXiv preprint arXiv:2312.12379_.

- Grok (2024)
Grok. 2024.

[Open release of grok-1](https://x.ai/blog/grok-os "").

- Guo et al. (2025a)
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, ZF Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025a.

[Deepseek-r1 incentivizes reasoning in llms through reinforcement learning](https://www.nature.com/articles/s41586-025-09422-z "").

_Nature_, 645(8081):633–638.

- Guo et al. (2025b)
Hongcan Guo, Haolang Lu, Guoshun Nan, Bolun Chu, Jialin Zhuang, Yuan Yang, Wenhao Che, Xinye Cao, Sicong Leng, Qimei Cui, and Xudong Jiang. 2025b.

[Advancing expert specialization for better moe](https://openreview.net/forum?id=iydmH9boLb "").

In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_.

- Gurnee et al. (2023)
Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. 2023.

[Finding neurons in a haystack: Case studies with sparse probing](https://arxiv.org/abs/2305.01610 "").

_arXiv preprint arXiv:2305.01610_.

- He et al. (2024)
Xin He, Shunkang Zhang, Yuxin Wang, Haiyan Yin, Zihao Zeng, Shaohuai Shi, Zhenheng Tang, Xiaowen Chu, Ivor Tsang, and Ong Yew Soon. 2024.

[Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference](https://arxiv.org/abs/2410.17954 "").

_arXiv preprint arXiv:2410.17954_.

- Hockney (1994)
Roger W. Hockney. 1994.

[The communication challenge for mpp: Intel paragon and meiko cs-2](https://doi.org/10.1016/S0167-8191(06)80021-9 "").

_Parallel Comput._, 20(3):389–398.

- Huang et al. (2024)
Quzhe Huang, Zhenwei An, Nan Zhuang, Mingxu Tao, Chen Zhang, Yang Jin, Kun Xu, Liwei Chen, Songfang Huang, and Yansong Feng. 2024.

[Harder task needs more experts: Dynamic routing in moe models](https://aclanthology.org/2024.acl-long.696/ "").

In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 12883–12895.

- Jacobs et al. (1991)
Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991.

[Adaptive mixtures of local experts](https://ieeexplore.ieee.org/abstract/document/6797059 "").

_Neural computation_, 3(1):79–87.

- Jiang et al. (2024)
Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, and 7 others. 2024.

[Mixtral of experts](https://arxiv.org/abs/2401.04088 "").

_arXiv preprint arXiv:2401.04088_.

- Jiao et al. (2024)
Difan Jiao, Yilun Liu, Zhenwei Tang, Daniel Matter, Jürgen Pfeffer, and Ashton Anderson. 2024.

[Spin: Sparsifying and integrating internal neurons in large language models for text classification](https://aclanthology.org/2024.findings-acl.277 "").

In _Findings of the Association for Computational Linguistics: ACL 2024_, pages 4666–4682.

- Jordan and Jacobs (1994)
Michael I Jordan and Robert A Jacobs. 1994.

[Hierarchical mixtures of experts and the em algorithm](https://ieeexplore.ieee.org/abstract/document/6796382 "").

_Neural computation_, 6(2):181–214.

- Kaplan et al. (2020)
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.

[Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361/1000 "").

_arXiv preprint arXiv:2001.08361_.

- Komatsuzaki et al. (2022)
Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. 2022.

[Sparse upcycling: Training mixture-of-experts from dense checkpoints](https://arxiv.org/abs/2212.05055 "").

_arXiv preprint arXiv:2212.05055_.

- Lepikhin et al. (2020)
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020.

[Gshard: Scaling giant models with conditional computation and automatic sharding](https://arxiv.org/abs/2006.16668 "").

_arXiv preprint arXiv:2006.16668_.

- Liu and Lo (2025)
Kuan-Ming Liu and Ming-Chih Lo. 2025.

[Llm-based routing in mixture of experts: A novel framework for trading](https://arxiv.org/abs/2501.09636 "").

_arXiv preprint arXiv:2501.09636_.

- Liu et al. (2025)
Yilun Liu, Yunpu Ma, Yuetian Lu, Shuo Chen, Zifeng Ding, and Volker Tresp. 2025.

[Parameter-efficient routed fine-tuning: Mixture-of-experts demands mixture of adaptation modules](https://arxiv.org/abs/2508.02587 "").

_arXiv preprint arXiv:2508.02587_.

- Liu et al. (2023)
Zeyu Liu, Tim Dettmers, Xi Lin, Veselin Stoyanov, and Xian Li. 2023.

[Towards a unified view of sparse feed-forward network in pretraining large language model](https://aclanthology.org/2023.emnlp-main.930/ "").

In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_, pages 15038–15061.

- Lv et al. (2025)
Ang Lv, Ruobing Xie, Yining Qian, Songhao Wu, Xingwu Sun, Zhanhui Kang, Di Wang, and Rui Yan. 2025.

[Autonomy-of-experts models](https://arxiv.org/abs/2501.13074 "").

_arXiv preprint arXiv:2501.13074_.

- Mihaylov et al. (2018)
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018.

[Can a suit of armor conduct electricity? a new dataset for open book question answering](https://aclanthology.org/D18-1260/ "").

In _Proceedings of the 2018 conference on empirical methods in natural language processing_, pages 2381–2391.

- Muennighoff et al. (2025)
Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Evan Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, and 5 others. 2025.

[OLMoe: Open mixture-of-experts language models](https://openreview.net/forum?id=xXTkbTBmqq "").

In _The Thirteenth International Conference on Learning Representations_.

- Omi et al. (2025)
Nabil Omi, Siddhartha Sen, and Ali Farhadi. 2025.

[Load balancing mixture of experts with similarity preserving routers](https://arxiv.org/abs/2506.14038 "").

_arXiv preprint arXiv:2506.14038_.

- Pan et al. (2024)
Bowen Pan, Yikang Shen, Haokun Liu, Mayank Mishra, Gaoyuan Zhang, Aude Oliva, Colin Raffel, and Rameswar Panda. 2024.

[Dense training, sparse inference: Rethinking training of mixture-of-experts language models](https://arxiv.org/abs/2404.05567 "").

_arXiv preprint arXiv:2404.05567_.

- Qiu et al. (2025)
Zihan Qiu, Zeyu Huang, Bo Zheng, Kaiyue Wen, Zekun Wang, Rui Men, Ivan Titov, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025.

[Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models](https://aclanthology.org/2025.acl-long.249/ "").

In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 5005–5018.

- Rajbhandari et al. (2022)
Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022.

[Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale](https://proceedings.mlr.press/v162/rajbhandari22a.html?ref=https://githubhelp.com "").

In _International conference on machine learning_, pages 18332–18346. PMLR.

- Sakaguchi et al. (2021)
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021.

[Winogrande: An adversarial winograd schema challenge at scale](https://dl.acm.org/doi/10.1145/3474381 "").

_Communications of the ACM_, 64(9):99–106.

- Shazeer (2020)
Noam Shazeer. 2020.

[Glu variants improve transformer](https://arxiv.org/abs/2002.05202 "").

_arXiv preprint arXiv:2002.05202_.

- Shazeer et al. (2017)
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017.

[Outrageously large neural networks: The sparsely-gated mixture-of-experts layer](https://openreview.net/forum?id=B1ckMDqlg "").

In _International Conference on Learning Representations_.

- Su et al. (2024)
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024.

[Roformer: Enhanced transformer with rotary position embedding](https://dl.acm.org/doi/10.1016/j.neucom.2023.127063 "").

_Neurocomputing_, 568:127063.

- Vaswani et al. (2017)
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.

[Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html "").

_Advances in neural information processing systems_, 30.

- Wang et al. (2018)
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018.

[Glue: A multi-task benchmark and analysis platform for natural language understanding](https://aclanthology.org/W18-5446/ "").

In _Proceedings of the 2018 EMNLP workshop BlackboxNLP: Analyzing and interpreting neural networks for NLP_, pages 353–355.

- Wang et al. (2024a)
Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. 2024a.

[Auxiliary-loss-free load balancing strategy for mixture-of-experts](https://arxiv.org/abs/2408.15664 "").

_arXiv preprint arXiv:2408.15664_.

- Wang et al. (2024b)
Ziteng Wang, Jun Zhu, and Jianfei Chen. 2024b.

[Remoe: Fully differentiable mixture-of-experts with relu routing](https://openreview.net/forum?id=4D0f16Vwc3 "").

In _The Thirteenth International Conference on Learning Representations_.

- Wendler et al. (2024)
Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024.

[Do llamas work in english? on the latent language of multilingual transformers](https://aclanthology.org/2024.acl-long.820/ "").

In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 15366–15394.

- (54)
Songhao Wu, Ang Lv, Ruobing Xie, Xingwu Sun, Di Wang, Rui Yan, and Yankai Lin.

[Union-of-experts: Experts in mixture-of-experts are secretly routers](https://openreview.net/forum?id=Ksgiup7ZNZ "").

- Xiong et al. (2020)
Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. 2020.

[On layer normalization in the transformer architecture](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf "").

In _International conference on machine learning_, pages 10524–10533. PMLR.

- Yan et al. (2025)
Shen Yan, Xingyan Bin, Sijun Zhang, Yisen Wang, and Zhouchen Lin. 2025.

[Tc-moe: Augmenting mixture of experts with ternary expert choice](https://openreview.net/forum?id=dsP91M4hDL "").

In _The Thirteenth International Conference on Learning Representations_.

- Yu et al. (2025)
Yanpeng Yu, Haiyue Ma, Krish Agarwal, Nicolai Oswald, Qijing Huang, Hugo Linsenmaier, Chunhui Mei, Ritchie Zhao, Ritika Borkar, Bita Darvish Rouhani, David Nellans, Ronny Krashinsky, and Anurag Khandelwal. 2025.

[Efficient moe serving in the memory-bound regime: Balance activated experts, not tokens](https://arxiv.org/abs/2512.09277 "").

_arXiv preprint arXiv:2512.09277_.

- Zellers et al. (2019)
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019.

[Hellaswag: Can a machine really finish your sentence?](https://aclanthology.org/P19-1472.pdf "")In _Proceedings of the 57th annual meeting of the association for computational linguistics_, pages 4791–4800.

- Zhang and Sennrich (2019)
Biao Zhang and Rico Sennrich. 2019.

[Root mean square layer normalization](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html "").

_Advances in neural information processing systems_, 32.

- Zhao et al. (2025)
Chenqi Zhao, Wenfei Wu, Linhai Song, and Yuchen Xu. 2025.

[Micromoe: Fine-grained load balancing for mixture-of-experts with token scheduling](https://arxiv.org/abs/2511.16947 "").

_arXiv preprint arXiv:2511.16947_.

- Zhou et al. (2022)
Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Y Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, and James Laudon. 2022.

[Mixture-of-experts with expert choice routing](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html "").

_Advances in Neural Information Processing Systems_, 35:7103–7114.

- Zoph (2022)
Barret Zoph. 2022.

[Designing effective sparse expert models](https://ieeexplore.ieee.org/abstract/document/9835248 "").

In _2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)_, pages 1044–1044. IEEE.


## Appendix A Load-Balancing Losses

Fedus et al. ( [2022](https://arxiv.org/html/2604.00801v1#bib.bib14 "")) first introduced a differentiable auxiliary load-balancing loss to MoE, defined as the scaled dot-product

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒLB=α⋅N⋅∑i=1Nfi⋅Pi,\\displaystyle\\mathcal{L}\_{\\mathrm{LB}}=\\alpha\\cdot N\\cdot\\sum\_{i=1}^{N}f\_{i}\\cdot P\_{i}, |  | (18) |

where fif\_{i} is the fraction of tokens dispatched to expert ii and PiP\_{i} is the
fraction of router probability allocated to expert ii. Since fif\_{i} is non-differentiable,
the gradient flows through PiP\_{i} alone, while fif\_{i} serves as a fixed per-step weight.
The loss is minimized under uniform routing, as ∑ifi​Pi=1/N\\sum\_{i}f\_{i}P\_{i}=1/N when both
vectors are uniform. Our ℒEB\\mathcal{L}\_{\\text{EB}} and ℒTB\\mathcal{L}\_{\\text{TB}} follow
the same product-of-averages structure and gradient strategy, extending it to
simultaneously encourage uniformity along both the expert and token axes without
any centralized routing mechanism.
We show that ℒEB\\mathcal{L}\_{\\text{EB}} and ℒTB\\mathcal{L}\_{\\text{TB}} are minimized
towards the desired uniform activation state at ρ∞\\rho\_{\\infty}. Let

|     |     |     |     |
| --- | --- | --- | --- |
|  | fi=1\|ℬ\|​∑𝐱∈ℬfi​(𝐱),g~i=1\|ℬ\|​∑𝐱∈ℬGi​(𝐱)\\displaystyle f\_{i}=\\frac{1}{\|\\mathcal{B}\|}\\sum\_{\\mathbf{x}\\in\\mathcal{B}}f\_{i}(\\mathbf{x}),\\quad\\tilde{g}\_{i}=\\frac{1}{\|\\mathcal{B}\|}\\sum\_{\\mathbf{x}\\in\\mathcal{B}}G\_{i}(\\mathbf{x}) |  | (19) |

denote the per-expert mean binary activation and its differentiable proxy. Under the
constraint that the adaptive coefficient λt\\lambda\_{t} pins the mean density
1\|ℰ\|​∑ifi=ρ∞\\frac{1}{\|\\mathcal{E}\|}\\sum\_{i}f\_{i}=\\rho\_{\\infty}, the sum ∑ifi​g~i\\sum\_{i}f\_{i}\\tilde{g}\_{i}
is minimized when all terms are equal by the rearrangement inequality, achieved
precisely at fi=g~i=ρ∞f\_{i}=\\tilde{g}\_{i}=\\rho\_{\\infty} for all ii. While the losses are
not written as explicit variance terms, any imbalance where over-activated experts
co-occur with larger proxy values increases ∑ifi​g~i\\sum\_{i}f\_{i}\\tilde{g}\_{i} above the
baseline ρ∞2\\rho\_{\\infty}^{2} attained under perfect balance, since fif\_{i} and g~i\\tilde{g}\_{i}
share a common dependence on Gi​(𝐱)G\_{i}(\\mathbf{x}). The excess above ρ∞2\\rho\_{\\infty}^{2}
is therefore a monotone function of the covariance between binary activations and
their proxies, providing implicit variance penalization through a fully differentiable
surrogate without requiring additional normalization terms. An identical argument
applies to ℒTB\\mathcal{L}\_{\\text{TB}} by symmetry over the token axis.

## Appendix B Routing-Free MoE at Deployment

### B.1 Expert Parallelism

Deploying MoE at scale often benefits from partitioning experts across multiple devices under _expert parallelism_ (EP) (Lepikhin et al., [2020](https://arxiv.org/html/2604.00801v1#bib.bib34 "")).
In this setting, we trace the full critical path of both architectures and analyze per-regime efficiency.

Consider a single MoE layer distributed to MM devices, each hosting N/MN/M experts, and processing a batch of TT tokens
with hidden dimension DD and bb bytes per element.
We adopt the standard α\\alpha-β\\beta model (Hockney, [1994](https://arxiv.org/html/2604.00801v1#bib.bib26 "")) that transferring nn bytes over one hop incurs latency α+n/B\\alpha+n/B, where
α\\alpha is the startup per-hop latency and BB is the per-link bandwidth.
In standard MoE each token is routed to KK experts, and Routing-Free MoE
activates an expected Keff=ρ∞⋅NK\_{\\mathrm{eff}}=\\rho\_{\\infty}\\cdot N experts per token.

For a standard MoE layer, the router matmul, Softmax, and TopK are strictly
sequential and centralized, with costs

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | trouting\\displaystyle t\_{\\mathrm{routing}} | =trouter+tSoftmax+tTopK\\displaystyle=t\_{\\mathrm{router}}+t\_{\\mathrm{Softmax}}+t\_{\\mathrm{TopK}} |  |
|  |  | ∝T⋅(D+2)⋅N.\\displaystyle\\propto T\\cdot(D+2)\\cdot N. |  | (20) |

Softmax requires all NN logits before any normalization can proceed; TopK requires a full selection pass over NN values. Neither can be parallelized across devices, and dispatch cannot begin until both complete.
Afterwards it produces a token-to-expert index tensor as dispatch assignment used to pack token buffers, which are sent to multiple target devices via blocking All-to-All:

|     |     |     |     |
| --- | --- | --- | --- |
|  | tA2A=(M−1)​α+K⋅T⋅D⋅bM⋅B.t\_{\\mathrm{A2A}}=(M{-}1)\\,\\alpha+\\frac{K\\cdot T\\cdot D\\cdot b}{M\\cdot B}. |  | (21) |

Expert FFNs run on the received K⋅T/NK\\cdot T/N tokens per expert with no inter-device communication, with the parallelized per-device cost

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | texpert\\displaystyle t\_{\\mathrm{expert}} | ∝3​K⋅TM⋅D⋅Dact.\\displaystyle\\propto 3K\\cdot\\frac{T}{M}\\cdot D\\cdot D\_{\\mathrm{act}}. |  | (22) |

Their outputs are then scattered back to originating devices with the same buffer geometry and All-to-All tA2At\_{\\mathrm{A2A}},
and the receiving device uses its locally-stored assignment mask to scatter outputs into the correct positions, which is a cheap local memory operation. Total per-layer cost is

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | TMoE\\displaystyle T\_{\\mathrm{MoE}} | =trouting+texpert\\displaystyle=t\_{\\mathrm{routing}}+t\_{\\mathrm{expert}} |  |
|  |  | +2​(M−1)​α+2​K⋅T⋅D⋅bM⋅B.\\displaystyle+2(M{-}1)\\,\\alpha+2\\;\\frac{K\\cdot T\\cdot D\\cdot b}{M\\cdot B}. |  | (23) |

Routing-Free MoE starts from a non-blocking All-Gather to broadcast the full token batch:

|     |     |     |     |
| --- | --- | --- | --- |
|  | tAG=(M−1)​α+(M−1)⋅T⋅D⋅bM⋅B.t\_{\\mathrm{AG}}=(M{-}1)\\,\\alpha+\\frac{(M{-}1)\\cdot T\\cdot D\\cdot b}{M\\cdot B}. |  | (24) |

Each device can immediately begin scoring its N/MN/M local experts on incoming chunks, pipelining scoring with communication.
The total scoring cost per device, including applying 𝐱𝐀gate\\mathbf{xA}\_{\\mathrm{gate}}, norm, bias, activation and thresholding, is

|     |     |     |     |
| --- | --- | --- | --- |
|  | tscoring∝T⋅D⋅r⋅NM,t\_{\\mathrm{scoring}}\\propto T\\cdot D\\cdot r\\cdot\\frac{N}{M}, |  | (25) |

For activated token–expert pairs, 𝐱𝐀gate,i\\mathbf{x}\\mathbf{A}\_{\\mathrm{gate},i}
computed during scoring is directly reused in the FFN forward pass, incurring
zero marginal cost. For non-activated tokens computation terminates after the
rank-rr projection, costing only T⋅D⋅rT\\cdot D\\cdot r rather than a full FFN
pass. The remaining per-device FFN cost for activated pairs is

|     |     |     |     |
| --- | --- | --- | --- |
|  | texpert∗∝Keff⋅TM⋅(r+2​D)⋅Dact.t\_{\\mathrm{expert}}^{\*}\\propto K\_{\\mathrm{eff}}\\cdot\\frac{T}{M}\\cdot(r+2D)\\cdot D\_{\\mathrm{act}}. |  | (26) |

Activated outputs are immediately returned as asynchronous point-to-point messages upon completion, with no barrier required:

|     |     |     |     |
| --- | --- | --- | --- |
|  | tcombine=α+Keff⋅T⋅D⋅bM⋅B.t\_{\\mathrm{combine}}=\\alpha+\\frac{K\_{\\mathrm{eff}}\\cdot T\\cdot D\\cdot b}{M\\cdot B}. |  | (27) |

Receiving devices accumulate partial sums as results arrive. Total per-layer cost is

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | TRFMoE\\displaystyle T\_{\\mathrm{RFMoE}} | =tscoring+texpert∗+M​α\\displaystyle=t\_{\\mathrm{scoring}}+t\_{\\mathrm{expert}}^{\*}+M\\alpha |  |
|  |  | +(M−1+Keff)⋅T⋅D⋅bM⋅B.\\displaystyle+\\frac{(M-1+K\_{\\mathrm{eff}})\\cdot T\\cdot D\\cdot b}{M\\cdot B}. |  | (28) |

Setting K=KeffK=K\_{\\mathrm{eff}} for an iso-compute comparison, the ratio of computation cost for routing and expert processing becomes

|     |     |     |     |
| --- | --- | --- | --- |
|  | tscoring+texpert∗trouting+texpert=r​D+KN​(r+2​D)​Dact(D+2)​M+KN​(3​D)​Dact.\\frac{t\_{\\mathrm{scoring}}+t\_{\\mathrm{expert}}^{\*}}{t\_{\\mathrm{routing}}+t\_{\\mathrm{expert}}}=\\frac{rD+\\frac{K}{N}(r+2D)D\_{\\mathrm{act}}}{(D+2)M+\\frac{K}{N}(3D)D\_{\\mathrm{act}}}. |  | (29) |

Since r≪Dr\\ll D and M≪DactM\\ll D\_{\\textrm{act}}, this ratio is strictly less than 11, and gets smaller when MM grows.
Therefore, under expert parallelism, Routing-Free MoE always requires less computation per layer than standard MoE.

For communication, the barrier synchronization saving Δ​α=(M−2)​α\\Delta\\alpha=(M{-}2)\\,\\alpha is
strictly positive for all M≥3M\\geq 3, independent of input size or activation ratio.
The bandwidth terms differ as

|     |     |     |     |
| --- | --- | --- | --- |
|  | ΔB=(K+1−M)⋅T⋅D⋅bM⋅B,\\Delta\_{B}=\\frac{(K+1-M)\\cdot T\\cdot D\\cdot b}{M\\cdot B}, |  | (30) |

which favors Routing-Free MoE when M<K+1M<K+1.
In the prefill stage where TT is large, the ΔB\\Delta\_{B} term dominates the communication overhead, and Routing-Free MoE is advantageous when the number of devices MM is no greater than the number of activated experts KK for each layer, which typically holds in practice for advanced large-scale MoE models (Guo et al., [2025a](https://arxiv.org/html/2604.00801v1#bib.bib22 "")) under expert-parallel setting at inference.
In the decode stage where T=1T{=}1, the bandwidth term becomes negligible and the critical path is instead dominated by α\\alpha and sequential computational bottlenecks, and Routing-Free MoE becomes exceptionally well-suited for such high-throughput, low-latency streaming inference scenarios.

To corroborate the theoretical analysis, we evaluate Routing-Free MoE against standard MoE under expert-parallel deployment across varying sequence lengths TT, device counts MM, and expert activations KK for our trained model at scale S in production-representative settings.
We isolate the prefill and the decode stages. For each configuration, we measured end-to-end processing time, breaking down contributions from computation, communication, and synchronization overhead.

Table 5: Prefill and autoregressive decode tokens-per-second throughput per device under expert parallelism.
Prefill process TT=1,024 tokens in a single forward pass; decode generates 128 tokens autoregressively with TT=1 per step. KK=3 denotes TopK for standard MoE and KeffK\_{\\text{eff}}=3 for Routing-Free MoE.
All runs use batch size 1 with input tokens sampled randomly from OpenWebText, bfloat16 precision, and report the mean over 10 repetitions after 3 warmup ones.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| Setting | 𝑻\\bm{T} | 𝑴\\bm{M} | 𝑲\\bm{K} | Throughput ↑\\uparrow |
| Prefill Stage |
| MoE | 1,024 | 1 | 3 | 22,346.66 |
| MoE | 1,024 | 2 | 3 | 18,034.84 |
| MoE | 1,024 | 3 | 3 | 18,558.08 |
| MoE | 1,024 | 4 | 3 | 18,355.43 |
| RFMoE | 1,024 | 1 | 3 | 22,277.77 |
| RFMoE | 1,024 | 2 | 3 | 22,268.24 |
| RFMoE | 1,024 | 3 | 3 | 21,822.29 |
| RFMoE | 1,024 | 4 | 3 | 21,784.77 |
| Decode Stage |
| MoE | 1 | 1 | 3 | 57.90 |
| MoE | 1 | 2 | 3 | 45.65 |
| MoE | 1 | 3 | 3 | 45.61 |
| MoE | 1 | 4 | 3 | 44.67 |
| RFMoE | 1 | 1 | 3 | 52.14 |
| RFMoE | 1 | 2 | 3 | 52.05 |
| RFMoE | 1 | 3 | 3 | 50.11 |
| RFMoE | 1 | 4 | 3 | 49.84 |

Table [5](https://arxiv.org/html/2604.00801v1#A2.T5 "Table 5 ‣ B.1 Expert Parallelism ‣ Appendix B Routing-Free MoE at Deployment ‣ Routing-Free Mixture-of-Experts") reports throughput under expert parallelism for models at scale S.
Standard MoE throughput drops sharply when moving from M=1M{=}1 to M=2M{=}2, with Prefill falls by ∼19%{\\sim}19\\%, and decode
falls by ∼21%{\\sim}21\\%, with little recovery at M=3M{=}3 or M=4M{=}4. This reflects the blocking All-to-All dispatches and the sequential, non-parallelizable routing pipeline that must complete before any expert computation begins.
In contrast, Routing-Free MoE retains ≥97.8%{\\geq}97.8\\% of its single-device prefill throughput and ≥95.6%{\\geq}95.6\\% of its decode throughput at M=4M{=}4, consistent with the asynchronous scoring and point-to-point communication pattern described above.
Although Routing-Free MoE is slightly slower than standard MoE on a single device due to the rank-rr gating projection applied to all NN experts rather than a single D×ND{\\times}N router matmul, the crossover occurs already at M=2M{=}2, where Routing-Free MoE surpasses MoE by 23%23\\% in prefill and 14%14\\% in decode throughput. The advantage grows with MM. At M=4M{=}4, Routing-Free MoE delivers 1.19×1.19{\\times} higher prefill throughput and 1.12×1.12{\\times} higher decode throughput. This confirms the theoretical prediction that the communication and synchronization savings of Routing-Free MoE compound with device count, while its computational cost ratio remains strictly below unity and decreases monotonically in MM.
The decode stage exhibits particularly graceful scaling because the per-step bandwidth term T⋅D⋅b/(M⋅B)T\\cdot D\\cdot b/\\left(M\\cdot B\\right) is negligible at
T=1T{=}1, leaving the critical path dominated by startup latency α\\alpha and sequential compute.
Here, Routing-Free MoE’s elimination of the centralized routing barrier translates almost entirely into wall-clock savings, highlighting its performance for latency-sensitive autoregressive serving.

### B.2 Threshold Adaptation

The design of global post-activation threshold θ\\theta provides a lightweight, inference-time mechanism for trading computation against model quality, beyond its role in controlling overall sparsity during training.
The routing-free training dynamics naturally encourage each expert to commit decisively to its activation state, which pushes its internal score either well above or well below θ\\theta rather
than hovering near the boundary.
As a result, moderate perturbations to θ\\theta at inference
time displace only low-confidence, marginally contributing activations, conferring inherent robustness to threshold miscalibration.

Table 6: Effect of the global threshold θ\\theta on downstream benchmark
average (Avg.) and estimated FLOPs at scale S.
ρeff\\rho\_{\\text{eff}} denotes the empirical mean activation density at θ\\theta.
†Averages at θ≥1.5\\theta\\geq 1.5 are driven by the anomalous SST-2 and OBQA spikes (see Figure [8](https://arxiv.org/html/2604.00801v1#A2.F8 "Figure 8 ‣ B.2 Threshold Adaptation ‣ Appendix B Routing-Free MoE at Deployment ‣ Routing-Free Mixture-of-Experts")).

| 𝜽\\bm{\\theta} | 𝝆eff\\bm{\\rho}\_{\\text{eff}}(%) | FLOPs | Avg.↑\\uparrow |
| --- | --- | --- | --- |
| 0.1 | 100.0 | 120.46M | 39.41 |
| 0.5 | 94.8 | 118.41M | 39.12 |
| 0.8 | 61.4 | 105.40M | 39.55 |
| 0.9 | 48.9 | 100.32M | 39.73 |
| 1.0 | 37.8 | 96.21M | 39.80 |
| 1.1 | 29.3 | 92.97M | 39.98 |
| 1.2 | 22.9 | 90.37M | 39.77 |
| 1.5 | 11.1 | 85.80M | 40.77† |
| 1.9 | 4.3 | 83.14M | 40.23† |

![Refer to caption](https://arxiv.org/html/2604.00801v1/x15.png)Figure 8: Per-benchmark accuracy across θ\\theta at
scale S. HellaSwag, QQP, QNLI are nearly invariant to the threshold, while PIQA, ARC-easy, ARC-challenge and Winogrande achieve their best performance around ρeff≈ρ∞\\rho\_{\\text{eff}}\\approx\\rho\_{\\infty}.
SST-2 and OBQA spikes sharply at larger θ\\theta,
driving the elevated averages.

Table [6](https://arxiv.org/html/2604.00801v1#A2.T6 "Table 6 ‣ B.2 Threshold Adaptation ‣ Appendix B Routing-Free MoE at Deployment ‣ Routing-Free Mixture-of-Experts") reports the downstream benchmark average across a sweep of θ\\theta at scale S. The most striking observation is
the stability of performance despite changes in activation density. Reducing ρeff\\rho\_{\\text{eff}} from 100% to
4.3%, a more than 20×\\times reduction in expert
activations and a 31% drop in FLOPs, degrades the average score by less than two absolute points.
This robustness is a direct consequence of the decisive activation patterns learned during routing-free training.
Because most expert scores lie far from the decision boundary,
sweeping θ\\theta over a wide range affects only a small number of uncertain, low-impact activations.
Notably, even at extremely low thresholds where nearly all experts are activated, as seen at ρeff≈100%\\rho\_{\\text{eff}}\\approx 100\\% with θ=0.1\\theta{=}0.1, the model does not suffer from gradient explosion or output instability.
Since the activation scores Gi​(𝐱)G\_{i}(\\mathbf{x}) directly scale each expert’s output before aggregation, experts that pass the threshold with low scores contribute proportionally little to the final representation.
Lowering θ\\theta therefore admits only marginal, low-weight activations rather than introducing equally weighted noise from all experts.

The per-benchmark breakdown in Figure [8](https://arxiv.org/html/2604.00801v1#A2.F8 "Figure 8 ‣ B.2 Threshold Adaptation ‣ Appendix B Routing-Free MoE at Deployment ‣ Routing-Free Mixture-of-Experts") reveals
that this aggregate stability is not an artifact of compensating trends.
The majority of benchmarks, including HellaSwag, QQP, and QNLI are effectively invariant to θ\\theta, confirming that
the core representations remain intact even under aggressive
sparsification.
A small number of tasks exhibit larger sensitivity.
SST-2 in particular spikes at θ=1.5\\theta{=}1.5, which accounts for most of the elevated aggregate averages at high thresholds.
We attribute this to the interaction between task-specific input distributions and the activation patterns of individual experts, rather than a systematic benefit of sparser computation.
These results highlight a practical advantage of the
routing-free design. The threshold θ\\theta serves as a single, transparent knob that allows practitioners to balance accuracy and efficiency at deployment time without retraining, with assurance that moderate adjustments will not disrupt the expert specialization patterns learned during training.

## Appendix C Statistical Significance Analysis

To support the claim that Routing-Free MoE consistently outperforms the standard MoE baseline, we conduct a formal statistical analysis over the benchmark results reported in Table [1](https://arxiv.org/html/2604.00801v1#S3.T1 "Table 1 ‣ 3.2 Training ‣ 3 Methodology ‣ Routing-Free Mixture-of-Experts").
Each model variant is evaluated on nine benchmarks across three scales, yielding 9×3=279\\times 3=27 paired observations, where each pair consists of the accuracy score of Routing-Free MoE and MoE on the same benchmark at the same scale.

We treat each (scale, benchmark) pair as a matched observation.
For each pair we compute the signed improvement Δi=RFMoEi−MoEi\\Delta\_{i}=\\text{RFMoE}\_{i}-\\text{MoE}\_{i} in percentage points (pp).
We test the one-sided null hypothesis H0:μΔ≤0H\_{0}\\colon\\mu\_{\\Delta}\\leq 0 against H1:μΔ>0H\_{1}\\colon\\mu\_{\\Delta}>0.
We apply a one-sided paired tt-test, which evaluates whether the mean improvement is significantly positive under an approximate normality assumption.
Effect size is quantified via Cohen’s dd computed on the paired differences, and win rate is reported as a descriptive statistic.

Table 7: Paired statistical tests comparing Routing-Free MoE against the standard MoE baseline across 9 benchmarks. pp-values are one-sided.

| Scale | W/L | Δ\\Delta Avg. | 𝒕\\bm{t}-stat | 𝒑𝒕\\bm{p\_{t}} | Cohen’s d\\bm{d} |
| --- | --- | --- | --- | --- | --- |
| S | 5 / 4 | +0.80 | 1.481 | 0.089 | 0.494 |
| M | 6 / 3 | +0.76 | 0.764 | 0.233 | 0.255 |
| L | 5 / 4 | +0.76 | 1.184 | 0.135 | 0.395 |
| Overall | 16 / 11 | +0.77 | 1.858 | 0.037 | 0.358 |

Table [7](https://arxiv.org/html/2604.00801v1#A3.T7 "Table 7 ‣ Appendix C Statistical Significance Analysis ‣ Routing-Free Mixture-of-Experts") reports per-scale statistics.
The mean improvement is positive at every scale (+0.80 pp at S, +0.76 pp at both M and L), and Cohen’s dd ranges from 0.26 to 0.49, indicating a consistent small-to-medium effect.
When observations are pooled across all three scales (n=27n=27), the one-sided paired tt-test yields t=1.858t=1.858, p=0.037p=0.037, rejecting H0H\_{0} at the α=0.05\\alpha=0.05 level.
Routing-Free MoE achieves a positive improvement in 16 out of 27 pairs, with a mean gain of +0.77+0.77 pp and Cohen’s d=0.36d=0.36.
Beyond downstream task accuracy, Routing-Free MoE also achieves consistently lower perplexity than the MoE baseline at every scale, with −12.2%-12.2\\% at S, −11.7%-11.7\\% at M, and −18.8%-18.8\\% at L, suggesting that the quality of language modeling improves consistently with scale under Routing-Free MoE’s design, independent of the downstream accuracy results.

Taken together, the statistical analysis supports the conclusion that Routing-Free MoE produces a reliable improvement over the standard MoE baseline. The effect is consistent in direction across all three scales and nine benchmarks, reaches statistical significance when observations are pooled with paired tt-test p=0.037p=0.037, and is accompanied by a moderate effect size with Cohen’s d=0.36d=0.36, along with substantial perplexity reduction at every scale.

## Appendix D Additional Discussion

### D.1 Load-Balancing

A recent line of work explored auxiliary-loss-free load balancing (DeepSeek-AI et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib8 ""); Guo et al., [2025a](https://arxiv.org/html/2604.00801v1#bib.bib22 "")).
Our baselines and Routing-Free MoE all employ auxiliary losses following standard practice, ensuring a controlled comparison across routing architectures.
Since auxiliary-loss-free balancing is orthogonal to the routing mechanism itself, any such technique could in principle be applied to standard MoE, AoE, ReMoE, and Routing-Free MoE alike.
Introducing it for one method but not others would confound the comparison, while applying it uniformly across all baselines would constitute a substantial engineering effort that lies beyond the scope of this work.
We therefore focus on the architectural distinction between routing-based and routing-free expert selection, which is the central contribution of this paper, and leave the integration of auxiliary-loss-free balancing as beyond the scope of this work.

Similarly, since all architectures compared in this work operate under the Token Choice load balancing paradigm, which is the dominant setting in the MoE literature (Muennighoff et al., [2025](https://arxiv.org/html/2604.00801v1#bib.bib40 "")), we choose not to include Expert Choice routing as a direct baseline.
Instead, our configurable μ\\mu-interpolation between token-balancing and expert-balancing losses allows Routing-Free MoE to smoothly interpolate the spectrum between token-balancing and expert-balancing behavior within a unified framework.
This design choice enables a thorough internal comparison across balancing strategies, as demonstrated in our ablation studies, without introducing the confound of an entirely different assignment paradigm.

### D.2 Per-Layer and Global Density

In Figure [6](https://arxiv.org/html/2604.00801v1#S4.F6 "Figure 6 ‣ 4.4 Training Dynamics ‣ 4 Experiments ‣ Routing-Free Mixture-of-Experts"), without any explicit supervision, the model develops a striking three-stage structure, with early layers showing high but rapidly decreasing activation with large variance, middle layers exhibiting a slow monotonic climb with low variance, and late layers displaying a sharp rise in both activation level and variance.
This emergent pattern coincides closely with interpretability findings on the heterogeneous functional roles that layers naturally develop in LLMs (Geva et al., [2021](https://arxiv.org/html/2604.00801v1#bib.bib18 ""); Gao et al., [2024a](https://arxiv.org/html/2604.00801v1#bib.bib16 ""); Wendler et al., [2024](https://arxiv.org/html/2604.00801v1#bib.bib53 ""); Jiao et al., [2024](https://arxiv.org/html/2604.00801v1#bib.bib30 "")), with early layers converting input from token space to concept space, middle layers processing them, and late layers preparing input-dependent outputs back to token space.
We argue that the performance gain from global density constraints stems precisely from removing the per‑layer inductive bias.
Enforcing identical sparsity at every depth suppresses the compute‑hungry layers that naturally benefit from activating more experts, while simultaneously forcing unnecessary activations in layers where sparse representations suffice. Once this bias is lifted, the model is free to self‑organize into a more effective, functionally aligned activation structure.

## Appendix E Additional Experiment Results

Table 8: Performance of standard MoE, AoE, ReMoE, and Routing-Free MoE across configurations. Each model was trained on OpenWebText (Gokaslan et al., [2019](https://arxiv.org/html/2604.00801v1#bib.bib19 "")).
Entries marked as n/a indicate fields not applicable to the respective architectural design.
Results underlined denote experiments with gradient explosion.

Arch.Config.rrLLDDDactD\_{\\mathrm{act}}SizeFLOPsλ\\lambdaη\\etaμ\\muα\\alphaLossVal. LossPPLMoETop3/12n/a1251212892.44M90.93Mn/an/an/a5e-43.8043.66739.13MoETop3/12n/a1251212892.44M90.93Mn/an/an/a1e-33.5953.44131.22MoETop3/12n/a1251212892.44M90.93Mn/an/an/a2e-35.5845.366214.0AoETop3/12161251212893.85M88.57Mn/an/an/a1e-33.4403.40130.00AoETop3/12321251212895.32M91.08Mn/an/an/a1e-33.4503.41130.31ReMoETop3/12n/a1251212892.44M90.93M1e-80.2n/a1e-33.5213.38929.60RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.0050.55e-43.5793.62037.34
RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.050.55e-43.5523.60036.60RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.080.55e-43.5373.64438.24RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.10.55e-43.5713.61237.04RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.020.51e-33.3093.35828.73RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.020.52e-33.2613.31127.42RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.020.55e-37.0976.9711065RFMoEρ∞\\rho\_{\\infty}=1/481251212893.11M87.32M1e-100.020.51e-33.3323.37329.16RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.020.51e-33.3093.35828.74RFMoEρ∞\\rho\_{\\infty}=1/4321251212895.32M91.08M1e-100.020.51e-33.2903.34428.34RFMoEρ∞\\rho\_{\\infty}=1/4641251212898.27M96.09M1e-100.020.51e-33.2903.34128.24RFMoEρ∞\\rho\_{\\infty}=1/481251212893.11M87.32M1e-100.020.52e-33.7173.69940.41RFMoEρ∞\\rho\_{\\infty}=1/4161251212893.85M88.57M1e-100.020.52e-33.3113.26126.08RFMoEρ∞\\rho\_{\\infty}=1/4321251212895.32M91.08M1e-100.020.52e-36.4356.564709.1RFMoEρ∞\\rho\_{\\infty}=1/4641251212898.27M96.09M1e-100.020.52e-33.5483.51633.65RFMoEρ∞\\rho\_{\\infty}=1/4321251212895.32M91.08M1e-100.020.01e-33.4813.34728.41RFMoEρ∞\\rho\_{\\infty}=1/4321251212895.32M91.08M1e-100.020.21e-33.4833.34528.35RFMoEρ∞\\rho\_{\\infty}=1/4321251212895.32M91.08M1e-100.020.81e-33.4883.34628.38RFMoEρ∞\\rho\_{\\infty}=1/4321251212895.32M91.08M1e-100.021.01e-33.4853.34728.43MoETop4/16n/a24768192289.92M247.95Mn/an/an/a5e-43.8083.30027.11MoETop4/16n/a24768192289.92M247.95Mn/an/an/a1e-33.7263.21925.00RFMoEρ∞\\rho\_{\\infty}=1/44824768192307.30M249.17M1e-100.020.55e-43.3603.29226.90RFMoEρ∞\\rho\_{\\infty}=1/44824768192307.30M249.17M1e-100.020.51e-33.1663.09522.08MoETop6/24n/a321024256808.42M608.38Mn/an/an/a5e-43.9473.20224.58MoETop6/24n/a321024256808.42M608.38Mn/an/an/a8e-44.4933.76843.29MoETop6/24n/a321024256808.42M608.38Mn/an/an/a1e-38.5328.1043309RFMoEρ∞\\rho\_{\\infty}=1/464321024256870.60M613.19M1e-100.020.55e-43.1793.10722.36RFMoEρ∞\\rho\_{\\infty}=1/464321024256870.60M613.19M1e-100.020.58e-43.0762.99419.97RFMoEρ∞\\rho\_{\\infty}=1/464321024256870.60M613.19M1e-100.020.51e-35.4483.54834.73

Experimental support, please
[view the build logs](https://arxiv.org/html/2604.00801v1/__stdout.txt)
for errors. Generated by
[L\\
A\\
T\\
Exml![[LOGO]](<Base64-Image-Removed>)](https://math.nist.gov/~BMiller/LaTeXML/).


## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
methods listed below:

- Click the "Report Issue" () button, located in the page header.

**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we
may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
should not be a barrier to accessing research. Thank you for your continued support in championing open access for
all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from
our **major funders**,
[**member institutions**](https://info.arxiv.org/about/ourmembers.html), ,
and all contributors.


[About](https://info.arxiv.org/about)· [Help](https://info.arxiv.org/help)· [Contact](https://info.arxiv.org/help/contact.html)· [Subscribe](https://info.arxiv.org/help/subscribe)· [Copyright](https://info.arxiv.org/help/license/index.html)· [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html)· [Accessibility](https://info.arxiv.org/help/web_accessibility.html)· [Operational Status (opens in new tab)](https://status.arxiv.org/)

Major funding support from

[![Simons Foundation](https://arxiv.org/static/base/1.0.1/images/funders/simons-foundation.png)](https://www.simonsfoundation.org/)[![Simons Foundation International](https://arxiv.org/static/base/1.0.1/images/funders/simons-foundation-international.png)](https://www.sfi.org.bm/)[![Schmidt Sciences](https://arxiv.org/static/base/1.0.1/images/funders/schmidt-sciences.png)](https://www.schmidtsciences.org/)