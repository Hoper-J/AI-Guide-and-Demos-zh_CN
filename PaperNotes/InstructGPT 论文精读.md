# InstructGPT

**Training language models to follow instructions with human feedback**
Long Ouyang et al. | [PDF](https://arxiv.org/pdf/2203.02155) | [精简版](https://openai.com/index/instruction-following/) | OpenAI | 2022.03

> 这是一篇重要的文章，在 [ChatGPT](https://openai.com/index/chatgpt/) 的 Method 中有提及：
>
> “We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as [InstructGPT⁠](https://openai.com/index/instruction-following/), but with slight differences in the data collection setup. ”
>

## 数据集

### 数据来源

- **标注人员编写的提示**

  在初始阶段，为了训练第一个 InstructGPT 模型，研究团队专门聘请了标注人员来编写指令式（instruction-like）提示，因为这种类型的提示在常规 GPT-3 模型的 API 数据中并不常见。

  要求标注人员编写以下三种类型的提示：

  - **普通（Plain）**：编写任意任务，需要确保任务多样性

  - **少样本（Few-shot）**：提供一条指令以及多个问/答（query/response）示例对。

  - **基于用户（User-based）**：根据 OpenAI API 等候列表用户的应用需求，设计相关任务提示。

    > [OpenAI ChatGPT API Waitlist](https://share.hsforms.com/1u4goaXwDRKC9-x9IvKno0A4sk30) 中有一个必填的栏目 “Are there specific ideas you're excited to build with the ChatGPT API?” ，即“你有什么特别想用 ChatGPT API 实现的想法吗？”

- **API 收集的提示**

  > *“Our prompt dataset consists primarily of text prompts submitted to the OpenAI API, specifically those using an earlier version of the InstructGPT models (trained via supervised learning on a subset of our demonstration data) on the Playground interface.”*

  早期的 InstructGPT 模型被部署到 [Playground](https://beta.openai.com/playground) 中（新用户注册后会赠送一些 API 额度），当时用户和 InstructGPT 模型交互时会收到通知，告知其数据可能被用于模型的进一步训练（不包含来自生产环境的 API 用户数据）。

  收集到的数据做以下处理：

  1. **去重**：通过启发式方法检测并删除长前缀相同的提示。
  2. **限制提示数量**：每个用户的提示最多 200 条。
  3. **数据集划分**：根据用户 ID 划分为训练、验证和测试数据集，确保验证集与测试集中不会出现训练集用户的数据。
  4. **过滤敏感信息**：过滤并去除个人身份信息（personally identifiable information，PII）。

### 数据构建

研究团队基于上述提示数据构建了三种数据集，用于模型的不同训练阶段：

- **SFT 数据集（监督微调）**：包含 12,725 个训练提示，其中来自标注员的提示为 11,295 条，来自 API 的提示为 1,430 条。
- **RM 数据集（奖励模型）**：包含 33,207 个训练提示，其中来自标注员的提示为 6,623 条，来自 API 的提示为 26,584 条。
- **PPO 数据集（RLHF 强化学习）**：包含 31,144 条训练提示，完全来自于 API。

### 数据分布和示例

> **表 1**
>
> ![表 1](/Users/home/Downloads/agent/LLM-API-Guide-and-Demos/PaperNotes/assets/image-20250118122308319.png)

上表展示了 API 数据集中提示的使用类别分布，其中生成任务占比最高，为 45.6%。论文在附录 A.2.1 提供了一些提示示例，摘选部分进行理解：

| 用例类型                        | 样例                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| Generation<br />**生成**        | Write a short story where a brown bear to the beach, makes friends with a seal, and then return home.<br />写一篇关于一只棕熊去海滩与海豹交朋友然后回家的短篇故事。 |
| Open QA<br />**开放式问答**     | Who built the statue of liberty?<br />自由女神像是由谁建造的？ |
| Brainstorming<br />**头脑风暴** | What are 10 science fiction books I should read next?<br />我接下来应该阅读的 10 本科幻小说有哪些？ |
| Chat<br />**聊天**              | The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.<br /><br />Human: Hello, who are you?<br/> AI: I am an AI created by OpenAI. How can I help you today? Human: I’d like to cancel my subscription.<br/> AI:<br />一段对话 |
| Rewrite<br />**重写**           | Translate this sentence to Spanish:<br /><br /><English sentence><br />将以下句子翻译成西班牙语：<br /><br /><英文句子> |
| Summarization<br />**摘要**     | Summarize this for a second-grade student:<br /><br />{text}<br />为二年级学生总结以下内容：<br /><br />{文本} |
| Classification<br />**分类**    | {java code}<br /><br />What language is the code above written in?<br />上面的代码是用什么语言编写的？ |
| Other<br />**其他**             | start with where<br />从哪里开始                             |
| Closed QA<br />**封闭式问答**   | Answer the following question:<br />What shape is the earth?<br /><br />A) A circle <br />B) A sphere <br />C) An ellipse <br />D) A plane<br />一个选择题 |
| Extract<br />**抽取**           | Extract all place names from the article below:<br /><br />{news article}<br />从下面的文章中提取所有的地名：<br /><br />{新闻文章} |
