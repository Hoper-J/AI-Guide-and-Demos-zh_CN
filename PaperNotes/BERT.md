# BERT

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
Jacob Devlin et al. | [arXiv 1810.04805](https://arxiv.org/pdf/1810.04805) | [Code - 官方 Tensorflow](https://github.com/google-research/bert) | NAACL 2019 | Google AI Language

> **学习 & 参考资料**
>
> - **论文逐段精读**
>
>   —— 沐神的论文精读合集
>
>   - [BERT 论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1PL411M7eQ/?share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390)
>
> - **3Blue1Brown**
>
>   —— 顶级的动画解释
>
>   - [【官方双语】GPT是什么？直观解释Transformer | 【深度学习第5章】]( https://www.bilibili.com/video/BV13z421U7cs/?share_source=copy_web&vd_source=e46571d631061853c8f9eead71bdb390)
>   - [【官方双语】直观解释注意力机制，Transformer的核心 | 【深度学习第6章】](https://www.bilibili.com/video/BV1TZ421j7Ke/?share_source=copy_web)
>
> - **代码**
>
>   —— 哈佛 NLP 团队公开的 Transformer 注释版本，基于 PyTorch 实现。
>
>   - [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
>
> - **可视化工具**
>
>   - [TRANSFORMER EXPLAINER](https://poloclub.github.io/transformer-explainer/)
>
>     观察 Self-Attention 的中间过程。
>
>     需要注意的是网页端演示的不是传统的 Transformer 架构，而是 GPT-2（Decoder-Only），不过 BERT 的架构中也包含 Self-Attention，通过 GPT-2 理解相同的部分是完全足够的。

## 前言

在计算机视觉（Computer Vision, CV）领域，很早就可以通过卷积神经网络（Convolutional Neural Network, CNN）在大型数据集上进行预训练（Pre-training），然后迁移到其他任务中以提升性能。但在自然语言处理（Natural Language Processing, NLP）领域，长期以来并没有类似的通用深度神经网络模型，这时候很多研究都是“各自为战”，为特定任务训练专属的模型，导致计算资源重复利用，研究成果难以共享，整体效率较低，重复的“造轮子”。

Transformer 架构的提出为这一状况带来了曙光，BERT 的出现更是彻底改变了 NLP 的研究格局。BERT 将 Transformer 架构从翻译任务推广到了其他的 NLP 任务中，刷新了 11 项任务的 SOTA（State of the Art），证明了其架构的通用性和有效性，也开启了预训练语言模型（Pre-trained Language Models, PLMs）研究的新时代。

> [!tip]
>
> BERT 的名字来源于美国经典儿童节目《芝麻街》（Sesame Street）的角色，论文中对应的全称为 **B**idirectional **E**ncoder **R**epresentations from **T**ransformer，即“基于 Transformer 架构的双向编码器表示”。是的，“硬凑名字”，类似地，BERT 的“前辈” **ELMo**（Embeddings from Language Models）[^1]也是如此。“学术严肃与幽默并存” :)

[^1]: [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365).

## 贡献

BERT 的主要贡献如下：

 - **双向上下文建模（Bidirectional Contextual Representation）**

   过去的语言模型（如 GPT[^2]）大多采用单向建模，即只利用从左到右的上下文信息来预测下一个词（token），无法充分利用句子的全局信息。BERT 引入掩码语言模型（Masked Language Model，MLM），随机遮掩输入序列中的部分词，迫使模型基于上下文来预测它（类似于完形填空），实现了深度的双向表征学习。

   在此之前也有研究（如 ELMo）将从左到右和从右到左两个单向模型的表示拼接在一起，以达到双向的目的，不过 BERT 对双向信息的利用更好。

 - **预训练与微调框架（Pre-training & Fine-tuning）**

   BERT 是**第一个**使用预训练与微调范式在一系列 NLP 任务（句子层面和词元层面）都达到 **SOTA** 的模型，可以说是全面验证了该方法的有效性。
   
   尽管这种思想并非由 BERT 首次提出，但却是因为 BERT 才广为人知，毕竟谁不喜欢架构简单、模型开源、结果还 SOTA 的研究呢？截止 2024.11，BERT 的引用量已经超过 118K[^3]，大量的研究摸着 BERT 过河。
   
[^2]: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

[^3]: 引用次数数据来源于 Google Scholar。

### Q1: 什么是预训练（Pre-training）？什么是微调（Fine-tuning）？

- **预训练**：利用大量无标注的数据，通过自监督学习（Self-supervised Learning）训练模型，使其学习通用的表示。
- **微调**：在特定的下游任务（如文本分类、问答、命名实体识别）中，使用少量标注数据，对预训练好的模型进行进一步训练，使其适应特定任务的需求。

### Q2: 什么是自监督学习？

自监督学习更像是介于无监督学习和有监督学习之间的概念，从数据上看，自监督学习的数据没有经过标注，从训练上来看，自监督学习又有着监督信号（伪标签）作为指导。它的核心思想是设计一个“代理任务”（Pretext Task），或者说人为的定义一些规则，从数据中自动生成“伪标签”来训练模型。

**举一些实际的例子**：

- **自然语言处理（NLP）**
  - **GPT 的自回归模型**：通过给定前面的词序列，预测下一个词（单向）。
  - **BERT 的 Masked Language Model（MLM）**：在输入文本中随机遮掩某个词，模型根据上下文（双向）进行预测。
- **计算机视觉（CV）**
  - **图像遮掩预测**：遮掩图像的一部分，模型根据周围的像素信息预测被遮挡的区域内容。
  - **早期的对比学习（Contrastive Learning）**：将同一图像的不同增强版本视为“正样本对”，不同图像视为“负样本对”。模型通过拉近正样本对的表示、拉远负样本对的表示进行学习。

### Q3: 论文提到将预训练表示应用到下游任务有两种策略：基于特征（feature-based）和微调（fine-tuning），二者有什么区别？

> 摘自论文 Introduction 部分：
>
> There are two existing strategies for applying pre-trained language representations to downstream tasks: ***feature-based*** and ***fine-tuning***. 
> The feature-based approach, such as ELMo (Peters et al., 2018a), uses task-specific architectures that include the pre-trained representations as additional features. 
> The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning ***all*** pre-trained parameters. 
> The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

**基于特征（feature-based）**方法：

- **预训练模型参数保持不变**，将其视为**固定的特征提取器**。
- **预训练模型的输出作为下游任务的额外输入**，可以直接与原始输入进行拼接。
- 下游任务可能设计了特定的模型。

**微调（fine-tuning）**方法：

- **预训练模型的参数在下游任务中继续更新**，对预训练模型进行**端到端的训练**，论文中提到的是全量微调，不过现在实际应用中微调也分很多种。
- 下游任务中，只需在预训练模型顶部添加少量的任务特定参数（如分类层）。

