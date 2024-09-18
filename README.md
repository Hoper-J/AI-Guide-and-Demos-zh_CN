# 这是一个中文的 LLM 入门项目

> 回顾过去的学习历程，发现在深度学习这条道路上，吴恩达和李宏毅老师的视频分享为我提供了非常巨大的帮助，幽默风趣的讲课方式，直观简单的理论阐述，使得课程生动有趣。但总会有学弟学妹们最初烦恼于怎么去获取国外大模型的 API，虽然总会解决，但我还是想将这个槛给拿掉，毕竟第一次总有畏难情绪。
>
> 这里不会提供🪜的教程，也不会使用大模型平台自定义的接口，而是使用 OpenAI SDK，期望能够让你学到更通用的知识。我会以阿里云大模型平台所提供的 API 为例，带你从 API 走进 LLM。
> 非常建议搭配课程【生成式人工智能导论】进行学习。
>
> - [课程主页](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php)
> - [视频 - Bilibili](https://www.bilibili.com/video/BV1BJ4m1e7g8)
>

## 快速访问

如果你只是加深【生成式人工智能导论】这门课的理解，可以从下面的链接快速访问：

- HW1，2不涉及代码相关知识，你可以通过访问对应的作业PDF来了解其中的内容：[HW1](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/GenAI%20作业%20PDF/HW1.pdf) | [HW2](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/GenAI%20作业%20PDF/HW2.pdf)。
- HW3: [引导文章](https://github.com/Hoper-J/LLM-Guide-and-Demos/blob/master/2.%20简单入门：通过API与Gradio构建AI应用.md) | [代码中文镜像](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/Demos/1.%20LLM%20API%20使用演示——从环境配置到多轮对话.ipynb) | [Colab](https://colab.research.google.com/drive/15jh4v_TBPsTyIBhi0Fz46gEkjvhzGaBR?usp=sharing)
- HW4: [引导文章](https://github.com/Hoper-J/LLM-Guide-and-Demos/blob/master/3.%20进阶指南：自定义%20Prompt%20提升大模型解题能力.md) | [代码中文镜像](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/Demos/3.%20自定义%20Prompt%20提升大模型解题能力——Gradio%20与%20ipywidgets版.ipynb) | [Colab](https://colab.research.google.com/drive/16JzVN_Mu4mJfyHQpQEuDx1q6jI-cAnEl?hl=zh-tw#scrollTo=RI0hC7SFT3Sr&uniqifier=1)
- HW5: [引导文章](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/8.%20尝试微调LLM：让它会写唐诗.md) | [代码中文镜像](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/Demos/6.%20尝试微调LLM：让它会写唐诗.ipynb) | [Colab](https://colab.research.google.com/drive/1nB3jwRJVKXSDDNO-pbURrao0N2MpqHl8?usp=sharing#scrollTo=uh5rwbr4q5Nw)
- 

**P.S. 中文镜像将完全实现作业代码的所有功能，Colab 链接对应于原作业，选择其中一个完成学习即可。**

## 目录

简要介绍目录完整的内容：

- **Demos**

  所有的代码文件都将存放在其中。

- **assets**

  不需要关注这个文件夹，这里是 .md 文件用到的图片。

- [0. 阿里大模型API获取步骤](https://github.com/Hoper-J/LLM-Guide-and-Demos/blob/master/0.%20阿里大模型API获取步骤.md)

  将带你一步步的获取 API，如果是第一次注册，你需要进行一次身份验证（人脸识别）。

- [1. 初识LLM API：环境配置与多轮对话演示](https://github.com/Hoper-J/LLM-Guide-and-Demos/blob/master/1.%20初识LLM%20API：环境配置与多轮对话演示.md)

  这是一段入门的配置和演示，对话代码摘自阿里开发文档。

- [2. 简单入门：通过API与Gradio构建AI应用](https://github.com/Hoper-J/LLM-Guide-and-Demos/blob/master/2.%20简单入门：通过API与Gradio构建AI应用.md)

  指导如何去使用 Gradio 搭建一个简单的 AI 应用。

  对应代码：[HW3：以 AI 搭建自己的應用 --  Colab](https://colab.research.google.com/drive/15jh4v_TBPsTyIBhi0Fz46gEkjvhzGaBR?usp=sharing) | [中文镜像](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/Demos/1.%20LLM%20API%20使用演示——从环境配置到多轮对话.ipynb)。

- [3. 进阶指南：自定义 Prompt 提升大模型解题能力](https://github.com/Hoper-J/LLM-Guide-and-Demos/blob/master/3.%20进阶指南：自定义%20Prompt%20提升大模型解题能力.md)

  你将学习自定义一个 Prompt 来提升大模型解数学题的能力，其中一样会提供 Gradio 和非 Gradio 两个版本，并展示代码细节。

  对应代码：[HW4：Become an AI Hypnosis Master -- Colab](https://colab.research.google.com/drive/16JzVN_Mu4mJfyHQpQEuDx1q6jI-cAnEl?hl=zh-tw#scrollTo=RI0hC7SFT3Sr&uniqifier=1) | [中文镜像](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/Demos/3.%20自定义%20Prompt%20提升大模型解题能力——Gradio%20与%20ipywidgets版.ipynb)。

- [4. 认识 LoRA：从线性层到注意力机制](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/4.%20认识%20LoRA：从线性层到注意力机制.md)

  在正式进入实践之前，你需要知道LoRA的基础概念，这篇文章会带你从线性层的LoRA实现到注意力机制。

- [5. 理解 Hugging Face 的 `AutoModel` 系列：不同任务的自动模型加载类](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/5.%20理解%20Hugging%20Face%20的%20%60AutoModel%60%20系列：不同任务的自动模型加载类.md)

  我们即将用到的模块是 Hugging Face 中的 AutoModel，这篇文章一样是一个前置知识。（当然你可以跳过，等后续产生疑惑时再看）

- [6. 开始实践：部署你的第一个LLM大语言模型](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/6.%20开始实践：部署你的第一个LLM大语言模型.md)

  在这里会带你实现LLM大语言模型的部署，项目到现在为止都不会有GPU的硬性要求，你可以继续学习。

- [7. 探究模型参数与显存的关系以及不同精度造成的影响](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/7.%20探究模型参数与显存的关系以及不同精度造成的影响.md)
  
  了解模型参数和显存的对应关系并掌握不同精度的导入方式会使得你对模型的选择更加称心。
  
- [8. 尝试微调LLM：让它会写唐诗](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/8.%20尝试微调LLM：让它会写唐诗.md)
  
  这篇文章与[3. 进阶指南：自定义 Prompt 提升大模型解题能力](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/3.%20进阶指南：自定义%20Prompt%20提升大模型解题能力.md)一样，本质上是专注于“用”而非“写”，你可以像之前一样，对整体的流程有了一个了解，尝试调整超参数部分来查看对微调的影响。
  对应代码：[HW5: LLM Fine-tuning -- Colab](https://colab.research.google.com/drive/1nB3jwRJVKXSDDNO-pbURrao0N2MpqHl8?usp=sharing#scrollTo=uh5rwbr4q5Nw) | [中文镜像](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/Demos/6.%20尝试微调LLM：让它会写唐诗.ipynb)。
  
- [9. 深入理解 Beam Search：原理, 示例与代码实现](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/9.%20深入理解%20Beam%20Search：原理%2C%20示例与代码实现.md)
  
  将从示例到代码演示，并讲解 Beam Search 的数学原理，这应该能解决一些之前阅读的困惑，最终提供一个简单的使用 Hugging Face Transformers 库的示例（如果跳过了之前的文章的话可以尝试这个示例）。
  
- [10. Top-K vs Top-P：生成式模型中的采样策略与 Temperature 的影响 ](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/10.%20Top-K%20vs%20Top-P：生成式模型中的采样策略与%20Temperature%20的影响.md)

  进一步向你展示其他的生成策略。
  
- **可选拓展阅读：**

  - [a. 使用 HFD 加快 Hugging Face 模型和数据集的下载](https://github.com/Hoper-J/LLM-Guide-and-Demos-zh_CN/blob/master/a.%20使用%20HFD%20加快%20Hugging%20Face%20模型和数据集的下载.md)
  
    如果你觉得模型下载实在是太慢了，可以参考这篇文章进行配置。
  
  

## 后续规划

1. 将优先完全复现【生成式人工智能导论】这门课程的所有代码以供学习，具体：

   - 将其中的行为使用 OpenAI 库进行替换
   - 使用 ipywidgets 模拟 Colab 的交互
   - 以中文进行作业引导

   下一节预告：...

2. ...



