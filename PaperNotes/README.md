# 论文随笔

没有论文作为项目的支撑总感觉缺了点意思，计划将逐步上传相关领域论文（对比学习->多模态/Transformer/Bert/GPT架构系列）的随笔。

> 随笔内容由论文+视频+源码+个人见解构成。
>
> 另外，每篇论文将给出相关链接，并掰碎讲解视频供想深入该论文的同学进行学习，可以将随笔部分当作知识导航，祝大家科研和工作顺利～

| Notes                                                        | Tag      | Describe                                                     | File                                                         | Online                                                       |
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [对比学习论文随笔 1：正负样本](./对比学习论文随笔%201：正负样本.md) | 对比学习 | 涉及使用正负样本思想且优化目标一致的基础论文：<br />- [Inst Disc CVPR 2018](./对比学习论文随笔%201：正负样本.md#inst-disc)<br />- [InvaSpread CVPR 2019](./对比学习论文随笔%201：正负样本.md#invaspread)<br />- [MoCo CVPR 2020](./对比学习论文随笔%201：正负样本.md#moco)<br />- [SimCLR ICML 2020](./对比学习论文随笔%201：正负样本.md#simclr)<br />- [MoCo v2 arXiv 2020](./对比学习论文随笔%201：正负样本.md#moco-v2) |                                                              |                                                              |
| [Transformer 论文精读](./Transformer%20论文精读.md)          | NLP      | Attention Is All You Need<br />NeurIPS 2017<br /><br />从零开始复现 Transformer（PyTorch），具体路径如下：<br />1. 缩放点积注意力->单头->掩码->自注意力->交叉注意力->多头->对齐论文<br/>2. 位置前馈网络（Position-wise Feed-Forward Networks）<br/>3. 残差连接（Residual Connection）和层归一化（Layer Normalization, LayerNorm），对应于 Add & Norm<br/>4. 输入嵌入（Embeddings）<br/>5. Softmax<br/>6. 位置编码（Positional Encoding）<br/>7. 编码器输入处理和解码器输入处理<br/>8. 掩码实现（填充掩码和未来掩码）<br/>9. 编码器层（Encoder Layer）和解码器层（Decoder Layer）<br/>10. 编码器（Encoder）和解码器（Decoder）<br/>11. 完整模型（Transformer）<br />将介绍模型架构中的所有组件，并解答可能的困惑（访问[速览疑问](./Transformer%20论文精读.md#速览疑问)进行快速跳转） | [Code](./Demos/动手实现%20Transformer.ipynb)                 | [Kaggle](https://www.kaggle.com/code/aidemos/transformer)<br />[Colab](https://colab.research.google.com/drive/1BtYPNjEHw3dudw5KKFe9dBEsUsgkm1Vt?usp=sharing) |
| [BERT 论文精读](./BERT%20论文精读.md)                        | NLP      | Pre-training of Deep Bidirectional Transformers for Language Understanding<br />NAACL 2019<br /><br />基于 Transformer 架构｜Encoder-Only<br />文章概览：<br />1. 预训练任务 MLM 和 NSP<br />2. BERT 模型的输入和输出，以及一些与 Transformer 不同的地方<br />3. 以 $\text{BERT}_\text{BASE}$ 为例，计算模型的总参数量<br /> | [作业 - BERT 微调抽取式问答](../Guide/22.%20作业%20-%20Bert%20微调抽取式问答.md) |                                                              |

> [!note]
>
> 因为图片加载的原因，靠后论文的跳转可能会偏上，可以选择下滑或者目录跳转
