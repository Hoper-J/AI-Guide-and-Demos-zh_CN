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

在深入研究之前，了解相关领域重要论文的时间线是一个很好的习惯（按照沐神论文精读的课件设计进行展示，时间跨度为 3 年）：

![时间线](/Users/home/Downloads/agent/LLM-API-Guide-and-Demos/PaperNotes/assets/%E6%97%B6%E9%97%B4%E7%BA%BF.png)

**Google 的 Transformer（Attention is All You Need）** 于 2017 年 6 月发表，一年后，OpenAI 的发表了 **GPT** ，又过了两个月，Google 的另一个团队发表了 **BERT**。除了名称上的区别，你可能还常听到以下概念：Encoder-Decoder（Transformer）、Decoder-Only（GPT 系列）、Encoder-Only（BERT）。

我们先来简要解读一下，本文暂时不涉及 BERT 的内容。

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
