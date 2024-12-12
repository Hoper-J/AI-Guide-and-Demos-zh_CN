【占位文章，虽然时间线上 GPT 更早一点，但还是决定先讲 BERT，GPT 留待日后整理上传】

# GPT

**Improving Language Understanding by Generative Pre-Training**
Alec Radford et al. | [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | OpenAI | 2018.06

> **学习 & 参考资料**
>
> 如果有闲暇的话，不妨先了解一下它的“前身”解读：《[Transformer 论文精读](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/PaperNotes/Transformer%20论文精读.md)》。
>
> **论文逐段精读**
>
> —— 沐神的论文精读合集
>
> - [GPT，GPT-2，GPT-3 论文精读【论文精读】](https://www.bilibili.com/video/BV1AF411b7xQ/?share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390)
>
> **3Blue1Brown**
>
> —— 顶级的动画解释
>
> - [【官方双语】GPT是什么？直观解释Transformer | 【深度学习第5章】]( https://www.bilibili.com/video/BV13z421U7cs/?share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390)
> - [【官方双语】直观解释注意力机制，Transformer的核心 | 【深度学习第6章】](https://www.bilibili.com/video/BV1TZ421j7Ke/?share_source=copy_web)
>
> **可视化工具**
>
> - [TRANSFORMER EXPLAINER](https://poloclub.github.io/transformer-explainer/)
>
>   观察 Self-Attention 的中间过程，并调节右上角的温度（Temperature）查看对概率的影响。
>
>   网页端演示的是 GPT-2（Decoder-Only）。

## 前言

在深入研究之前，了解相关领域重要论文的时间线[^1]是一个很好的习惯：

![时间线](/Users/home/Downloads/agent/LLM-API-Guide-and-Demos/PaperNotes/assets/%E6%97%B6%E9%97%B4%E7%BA%BF.png)

**Google 的 Transformer（Attention is All You Need）** 于 2017 年 6 月发表，一年后，OpenAI 的团队发表了 **GPT** ，又过了两个月，Google 的另一个团队发表了 **BERT**。

[^1]: 按照沐神论文精读的课件设计进行展示，时间跨度为 3 年。

GPT 是一种自回归（Auto-Regressive，AR）模型，那么在进一步了解 GPT 之前，有必要先了解自回归和非自回归的概念[^2]：

> ![AR vs NAR](./assets/image-20241023203706721.png)

### 自回归（Auto-Regressive）

**自回归生成**是指序列生成过程中，**每个新生成的 token 依赖于之前生成的 token**。这意味着生成过程是**串行的**，每一步的输入由**前面已生成的 token 组成的上下文序列**构成。例如：

- 假设要生成一个长度为 $T$ 的句子 $y = (y_1, y_2, \dots, y_T)$，在生成句子 $y$ 的过程中，首先生成 $y_1$，然后在生成 $y_2$ 时需要考虑 $y_1$；在生成 $y_3$ 时，需要考虑 $(y_1, y_2)$，以此类推，直到生成结束符号（`<end>`）。

这种设计确保了生成过程中的连贯性和逻辑一致性。

### 非自回归（Non-Autoregressive）

**非自回归生成**是一种**并行生成**的方式，**一次性生成多个甚至全部的 token**，从而显著提高生成速度，但也会**牺牲一定的生成质量**。

> **拓展**
>
> 现在也有工作使用非自回归模型作为“预言家”来指导自回归模型并行生成，从而在生成质量不变的情况下大幅度提高生成速度，以“空间”换时间。
>
> ![预言家](./assets/%E9%A2%84%E8%A8%80%E5%AE%B6.png)
>
> **相关论文**：
>
> - Fast Inference from Transformers via Speculative Decoding: [arXiv 2211.17192](https://arxiv.org/pdf/2211.17192)
> - Accelerating Large Language Model Decoding with Speculative Sampling: [arXiv 2302.01318](https://arxiv.org/pdf/2302.01318)

[^2]: 摘自《[Transformer 论文精读 QA 部分的 Q2](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/PaperNotes/Transformer%20论文精读.md#q2-什么是自回归与非自回归)》。

除了名称上的区别，你可能还常听到以下概念：Encoder-Decoder（Transformer）、Decoder-Only（GPT）、Encoder-Only（BERT）。

...

还记得 Transformer 是针对机器翻译所提出的，GPT 正是如今的...



## GPT-2

**Language Models are Unsupervised Multitask Learners**
Alec Radford et al. | [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | OpenAI | 2019.02



## GPT-3

**Language Models are Few-Shot Learners**
Tom B. Brown et al. | [PDF](https://arxiv.org/pdf/2005.14165) | OpenAI | 2020.05



## GPT-4

**GPT-4 Technical Report**
[PDF](https://arxiv.org/pdf/2303.08774) | OpenAI | 2023.03



Q：Transformer被称为Encoder-Decoder架构，BERT被称为Encoder-Only架构，GPT被称为Decoder-Only架构，那么两两之间的区别是什么？
