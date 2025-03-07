# DeepSeek API 输出解析 - OpenAI SDK

> **代码文件下载**：[Code](../Demos/deepseek-api-guide-2.ipynb)
>
> **在线链接**：[Kaggle](https://www.kaggle.com/code/aidemos/deepseek-api-guide-2) | [Colab](https://colab.research.google.com/drive/1WT0jpeIzWewoN5cT12Uwi92d5_tNff2J?usp=sharing)
>
> **前置文章**：[DeepSeek API 的获取与对话示例](./DeepSeek%20API%20的获取与对话示例.md)

## 目录

- [如何切换平台](#如何切换平台)
- [认识输出](#认识输出)
  - [DeepSeek-Chat](#deepseek-chat)
  - [DeepSeek-Reasoner](#deepseek-reasoner)
- [附录](#附录)

## 如何切换平台

> 本文不引入环境变量，如果对其感兴趣可以阅读《[初识 LLM API：环境配置与多轮对话演示](./01.%20初识%20LLM%20API：环境配置与多轮对话演示.md#环境变量配置)》的「环境变量配置」部分。
>
> [代码文件](../Demos/deepseek-api-guide-2.ipynb)已包含文章中所有平台的正确配置。

以 DeepSeek 单轮对话的代码样例进行讲解：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="your-api-key", # 1：替换成对应的 API_Key，可以使用环境变量而非明文填写，即 api_key=os.getenv("DEEPSEEK_API_KEY")
    base_url="https://api.deepseek.com", # 2：每个平台的 base_url 不同
)

# 单轮对话示例
completion = client.chat.completions.create(
    model="deepseek-chat", # 3：模型标识（model_id）可能存在差异
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ]
)
```

当前大模型平台都适配了 OpenAI SDK，所以只需要对应修改三个参数（分别对应于代码中的注释）：

1. api_key：《[DeepSeek API 的获取与对话示例](./DeepSeek%20API%20的获取与对话示例.md)》。
2. base_url。

3. model。

**不同平台参数对照表**[^1]：

|              | base_url                                            | chat_model_id             | reasoner_model_id         |
| ------------ | --------------------------------------------------- | ------------------------- | ------------------------- |
| DeepSeek     | "https://api.deepseek.com"                          | "deepseek-chat"           | "deepseek-reasoner"       |
| 硅基流动     | "https://api.siliconflow.cn/v1"                     | "deepseek-ai/DeepSeek-V3" | "deepseek-ai/DeepSeek-R1" |
| 阿里云百炼   | "https://dashscope.aliyuncs.com/compatible-mode/v1" | "deepseek-v3"             | "deepseek-r1"             |
| 百度智能云   | "https://qianfan.baidubce.com/v2"                   | "deepseek-v3"             | "deepseek-r1"             |
| 字节火山引擎 | https://ark.cn-beijing.volces.com/api/v3            | "deepseek-v3-241226"      | "deepseek-r1-250120"      |

以硅基流动平台为例，使用 chat 模型，修改如下：

```diff
- client = OpenAI(
-     api_key="your-api-key", #1
-     base_url="https://api.deepseek.com", # 2
- )
- completion = client.chat.completions.create(
-     model="deepseek-chat", # 3

+ client = OpenAI(
+     api_key="your-api-key", #1
+     base_url="https://api.siliconflow.cn/v1", # 2
+ )
+ completion = client.chat.completions.create(
+     model="deepseek-ai/DeepSeek-V3", # 3
```

最终：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="your-api-key", # 1：替换成对应的 API_Key
    base_url="https://api.siliconflow.cn/v1", # 2
)

# 单轮对话示例
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3", # 3
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ]
)
```

[^1]: [DeepSeek API 的获取与对话示例](./DeepSeek%20API%20的获取与对话示例.md).

## 认识输出

`print(completion.model_dump())` 的输出并不适合阅读，使用 `Pretty Print` 进行打印：

```python
# 结构化打印
from pprint import pprint
pprint(completion.model_dump())
    
# 下方代码作用和 pprint 一样
# import json
# print(json.dumps(completion.model_dump(), indent=4, ensure_ascii=False))
```

下面以 DeepSeek API 的聊天模型和推理模型为例进行解读，涉及的知识对于使用了 OpenAI SDK 的平台是通用的：

### DeepSeek-Chat

**输出**：

```yaml
{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'logprobs': None,
              'message': {'content': '您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-V3。如您有任何任何问题，我会尽我所能为您提供帮助。',
                          'function_call': None,
                          'refusal': None,
                          'role': 'assistant',
                          'tool_calls': None}}],
 'created': 1739002836,
 'id': '897844a1-65d9-4e74-bdd3-4d966c8c1710',
 'model': 'deepseek-chat',
 'object': 'chat.completion',
 'service_tier': None,
 'system_fingerprint': 'fp_3a5770e1b4',
 'usage': {'completion_tokens': 37,
           'prompt_cache_hit_tokens': 0,
           'prompt_cache_miss_tokens': 11,
           'prompt_tokens': 11,
           'prompt_tokens_details': {'cached_tokens': 0},
           'total_tokens': 48}}
```

**通过输出的字段我们能做些什么？**（详细说明见[附录](#附录)）：

- **获取模型回复（choices）**

  ```python
  print(completion.choices[0].message.content)
  ```

  **输出**：

  ```
  您好！我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-V3。如您有任何任何问题，我会尽我所能为您提供帮助。
  ```

- **获取用量信息（usage）**

  ```python
  def print_chat_usage(completion):
      stats = completion.usage
      hit = stats.prompt_cache_hit_tokens
      miss = stats.prompt_cache_miss_tokens
      
      print(f"===== TOKEN 消耗明细 =====")
      print(f"输入: {stats.prompt_tokens} tokens [缓存命中: {hit} | 未命中: {miss}]")
      print(f"输出: {stats.completion_tokens} tokens")
      print(f"总消耗: {stats.total_tokens} tokens")
      
      # 按 DeepSeek 定价计算成本（单位：元）
      # - 输入: 2元/百万 Tokens（缓存命中 0.5元/百万 Tokens）
      # - 输出: 8元/百万 Tokens
      # 官方价格文档：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
      input_cost = (hit * 0.5 + miss * 2) / 1_000_000
      output_cost = stats.completion_tokens * 8 / 1_000_000
      total_cost = input_cost + output_cost
      
      print(f"\n===== 成本明细 =====")
      print(f"输入成本: ￥{input_cost:.4f} 元")
      print(f"输出成本: ￥{output_cost:.4f} 元")
      print(f"预估总成本: ￥{total_cost:.4f} 元")
  
  print_chat_usage(completion)
  ```
  
  **输出**：
  
  ```
  ===== TOKEN 消耗明细 =====
  输入: 11 tokens [缓存命中: 0 | 未命中: 11]
  输出: 37 tokens
  总消耗: 48 tokens
  
  ===== 成本明细 =====
  输入成本: ￥0.0000 元
  输出成本: ￥0.0003 元
  预估总成本: ￥0.0003 元
  ```
  
  > [!important]
  >
  > 非 DeepSeek 官方平台不存在一些特殊字段（比如：`usage.prompt_cache_hit_tokens`），一个更兼容的版本：
  >
  > ```python
  > def print_chat_usage(completion, input_cost=2.0, output_cost=8.0, cache_hit_cost=0.5):
  >      """
  >      参数：
  >     - input_cost: 输入价格（元/百万 Tokens）
  >     - output_cost: 输出价格（元/百万 Tokens）
  >     - cache_hit_cost: 缓存命中价格（当平台不支持时自动退化到全价模式）
  > 
  >     按 DeepSeek 聊天模型定价设定默认成本（单位：元）：
  >     - 输入: 2元/百万 Tokens（缓存命中 0.5元/百万 Tokens）
  >     - 输出: 8元/百万 Tokens
  >     官方价格文档：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
  >     """
  >     stats = completion.usage
  > 
  >     # 尝试获取字段（兼容其他平台）
  >     hit = getattr(stats, 'prompt_cache_hit_tokens', 0)
  >     miss = getattr(stats, 'prompt_cache_miss_tokens', 
  >                   stats.prompt_tokens - hit if hasattr(stats, 'prompt_tokens') else 0)
  > 
  >     print(f"===== TOKEN 消耗明细 =====")
  >     # 仅在存在缓存机制时显示细节
  >     if hit + miss > 0:
  >         print(f"输入: {stats.prompt_tokens} tokens [缓存命中: {hit} | 未命中: {miss}]")
  >     else:
  >         print(f"输入: {stats.prompt_tokens} tokens")
  > 
  >     print(f"输出: {stats.completion_tokens} tokens")
  >     print(f"总消耗: {stats.total_tokens} tokens")
  > 
  >     # 动态成本计算
  >     input_cost = (hit * cache_hit_cost + miss * input_cost) / 1_000_000
  >     output_cost = stats.completion_tokens * output_cost / 1_000_000
  >     total_cost = input_cost + output_cost
  > 
  >     print(f"\n===== 成本明细 =====")
  >     print(f"输入成本: ￥{input_cost:.4f} 元")
  >     print(f"输出成本: ￥{output_cost:.4f} 元")
  >     print(f"预估总成本: ￥{total_cost:.4f} 元")
  > 
  > print_chat_usage(completion)
  > ```

### DeepSeek-Reasoner

修改代码中的 `model` 参数即可切换模型（以 DeepSeek 官方平台为例）：

```diff
- completion = client.chat.completions.create(
-     model="deepseek-chat", # 3

+ completion = client.chat.completions.create(
+     model="deepseek-reasoner", # 3
```

> 其他平台参考下表[^1]，对应 `reasoner_model_id` 列：
>
> |              | base_url                                            | chat_model_id                                                | reasoner_model_id                                            |
> | ------------ | --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | DeepSeek     | "https://api.deepseek.com"                          | "deepseek-chat"                                              | "deepseek-reasoner"                                          |
> | 硅基流动     | "https://api.siliconflow.cn/v1"                     | "deepseek-ai/DeepSeek-V3"                                    | "deepseek-ai/DeepSeek-R1"                                    |
> | 阿里云百炼   | "https://dashscope.aliyuncs.com/compatible-mode/v1" | "deepseek-v3"                                                | "deepseek-r1"                                                |
> | 百度智能云   | "https://qianfan.baidubce.com/v2"                   | "deepseek-v3"                                                | "deepseek-r1"                                                |
> | 字节火山引擎 | https://ark.cn-beijing.volces.com/api/v3/           | 访问[推理点](https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint?config=%7B%7D)获取 | 访问[推理点](https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint?config=%7B%7D)获取 |

修改后运行代码：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="your-api-key", # 1：替换成对应的 API_Key
    base_url="https://api.siliconflow.cn/v1", # 2
)

# 单轮对话示例
completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1", # 3：换成推理模型
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ]
)

from pprint import pprint
pprint(completion.model_dump())
```

**输出**：

```yaml
{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'logprobs': None,
              'message': {'content': '您好！我是DeepSeek-R1，一个由深度求索（DeepSeek）公司开发的智能助手，擅长通过思考来帮您解答复杂的数学，代码和逻辑推理等理工类问题。我会始终保持专业和诚实，如果您有任何问题，我会尽力为您提供帮助。',
                          'function_call': None,
                          'reasoning_content': '嗯，用户问“你是谁？”，我需要用中文回答。首先，我要明确我的身份是一个AI助手，由中国的深度求索公司开发，名字叫DeepSeek-R1。然后，要说明我的功能是帮助用户解答问题、提供信息。可能还需要提到我擅长多个领域，比如科技、科学、教育等等，以及使用场景，比如学习、工作、生活。还要保持友好和简洁，避免技术术语，让用户容易理解。\n'
                                               '\n'
                                               '接下来，我需要检查是否符合公司的指导方针，有没有需要强调的部分，比如安全性或隐私保护。可能还要提到持续学习优化，但不需要太详细。确保回答结构清晰，先自我介绍，再讲功能，最后表达愿意帮助的态度。要避免任何格式错误，用自然的口语化中文，不用markdown。然后组织语言，确保流畅自然，没有生硬的部分。最后通读一遍，确认准确性和友好性。',
                          'refusal': None,
                          'role': 'assistant',
                          'tool_calls': None}}],
 'created': 1739030062,
 'id': 'f8dfbdb0-6884-40db-bcd2-d411da96e1a7',
 'model': 'deepseek-reasoner',
 'object': 'chat.completion',
 'service_tier': None,
 'system_fingerprint': 'fp_7e73fd9a08',
 'usage': {'completion_tokens': 248,
           'completion_tokens_details': {'reasoning_tokens': 187},
           'prompt_cache_hit_tokens': 0,
           'prompt_cache_miss_tokens': 13,
           'prompt_tokens': 13,
           'prompt_tokens_details': {'cached_tokens': 0},
           'total_tokens': 261}}
```

- **获取模型回复（choices）**

  ```python
  # 获取推理思考过程（Reasoner特有字段）
  reasoning_content = completion.choices[0].message.reasoning_content
  print(f"===== 模型推理过程 =====\n{reasoning_content}")
  
  # 获取模型回复内容（与之前相同）
  content = completion.choices[0].message.content
  print(f"===== 模型回复 =====\n{content}")
  ```
  
  **输出**：
  
  ```
  ===== 模型推理过程 =====
  嗯，用户问“你是谁？”，我需要用中文回答。首先，我要明确我的身份是一个AI助手，由中国的深度求索公司开发，名字叫DeepSeek-R1。然后，要说明我的功能是帮助用户解答问题、提供信息。可能还需要提到我擅长多个领域，比如科技、科学、教育等等，以及使用场景，比如学习、工作、生活。还要保持友好和简洁，避免技术术语，让用户容易理解。
  
  接下来，我需要检查是否符合公司的指导方针，有没有需要强调的部分，比如安全性或隐私保护。可能还要提到持续学习优化，但不需要太详细。确保回答结构清晰，先自我介绍，再讲功能，最后表达愿意帮助的态度。要避免任何格式错误，用自然的口语化中文，不用markdown。然后组织语言，确保流畅自然，没有生硬的部分。最后通读一遍，确认准确性和友好性。
  ===== 模型回复 =====
  您好！我是DeepSeek-R1，一个由深度求索（DeepSeek）公司开发的智能助手，擅长通过思考来帮您解答复杂的数学，代码和逻辑推理等理工类问题。我会始终保持专业和诚实，如果您有任何问题，我会尽力为您提供帮助。
  ```
  
  > [!important]
  >
  > 部分部署了 DeepSeek-R1 的平台并没有解析 `<think>` 标签，此时访问 `message.reasoning_content` 会报错 `AttributeError`，这里手动进行处理：
  >
  > ```python
  > import re
  > 
  > def parse_reasoner_response(completion):
  >     """
  >     参数：
  >     - completion (object): API 返回的对象
  >     
  >     返回：
  >     - (reasoning_content, reply_content)
  >     
  >     处理两种平台格式：
  >     1. 有独立 reasoning_content 字段的平台：DeepSeek 官方，硅基流动，百度智能云...
  >     2. 可能需要从 content 解析 <think> 标签的平台：阿里云百炼（偶尔会没有 reasoning_content）...
  >     """
  >     message = completion.choices[0].message
  >     
  >     # 尝试直接获取 reasoning_content 字段
  >     reasoning = getattr(message, 'reasoning_content', None)
  >     
  >     # 有 reasoning_content 时直接获取最终回复
  >     if reasoning:
  >         final_content = getattr(message, 'content', '')
  >     else:
  >         # 如果没有，则尝试从 content 解析
  >         content = getattr(message, 'content', '')
  >         
  >         # 使用非贪婪模式匹配 <think> 标签
  >         reasoning_match = re.search(
  >             r'<think>(.*?)</think>', 
  >             content, 
  >             re.DOTALL  # 允许跨行匹配
  >         )
  >         
  >         if reasoning_match:
  >             reasoning = reasoning_match.group(1).strip()
  >             # 从原始内容移除推理部分
  >             final_content = re.sub(
  >                 r'<think>.*?</think>', 
  >                 '', 
  >                 content, 
  >                 flags=re.DOTALL
  >             ).strip()
  >         else:
  >             reasoning = ''
  >             final_content = content
  >     
  >     return reasoning, final_content
  > 
  > reasoning_content, content = parse_reasoner_response(completion)
  > 
  > print(f"===== 模型推理过程 =====\n{reasoning_content}")
  > print(f"\n===== 最终回复 =====\n{content}")
  > ```
  
- **获取用量信息（usage）**

  ```python
  def print_reasoner_usage(completion):
      stats = completion.usage
      hit = stats.prompt_cache_hit_tokens
      miss = stats.prompt_cache_miss_tokens
      
      print(f"===== TOKEN 消耗明细 =====")
      print(f"输入: {stats.prompt_tokens} tokens [缓存命中: {hit} | 未命中: {miss}]")
      print(f"输出: {stats.completion_tokens} tokens")
  
      # 推理模型的token分解
      if details := stats.completion_tokens_details:
          reasoning = details['reasoning_tokens']
          final = stats.completion_tokens - reasoning
          print(f"├─ 推理过程: {reasoning} tokens")
          print(f"└─ 最终回答: {final} tokens")
      
      print(f"总消耗: {stats.total_tokens} tokens")
      
      # 按 DeepSeek 定价计算成本（单位：元）
      # - 输入Token: 4元/百万Tokens（未命中缓存 1元/百万Tokens）
      # - 输出Token: 16元/百万Tokens
      # 官方价格文档：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
      input_cost = (hit * 1 + miss * 4) / 1_000_000
      output_cost = stats.completion_tokens * 16 / 1_000_000
      total_cost = input_cost + output_cost
      
      print(f"\n===== 成本明细 =====")
      print(f"输入成本: ￥{input_cost:.4f} 元")
      print(f"输出成本: ￥{output_cost:.4f} 元")
      print(f"预估总成本: ￥{total_cost:.4f} 元")
  
  print_reasoner_usage(completion)
  ```
  
  **输出**：
  
  ```
  ===== TOKEN 消耗明细 =====
  输入: 13 tokens [缓存命中: 0 | 未命中: 13]
  输出: 248 tokens
  ├─ 推理过程: 187 tokens
  └─ 最终回答: 61 tokens
  总消耗: 261 tokens
  
  ===== 成本明细 =====
  输入成本: ￥0.0001 元
  输出成本: ￥0.0040 元
  预估总成本: ￥0.0040 元
  ```
  
  > [!important]
  >
  > 非 DeepSeek 官方的部分平台（但百度智能云存在）不存在一些特殊字段（比如：`reasoning_tokens`），一个更兼容的版本：
  >
  > ```python
  > def print_reasoner_usage(completion, input_cost=4.0, output_cost=16.0, cache_hit_cost=1.0):
  >     """
  >     参数：
  >     - input_cost: 输入价格（元/百万 Tokens）
  >     - output_cost: 输出价格（元/百万 Tokens）
  >     - cache_hit_cost: 缓存命中价格（当平台不支持时自动退化到全价模式）
  > 
  >     按 DeepSeek 推理模型定价设定默认成本（单位：元）：
  >     - 输入: 4元/百万 Tokens（缓存命中 1元/百万 Tokens）
  >     - 输出: 16元/百万 Tokens
  >     官方价格文档：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
  >     """
  >     stats = completion.usage
  >     
  >     # 尝试获取字段（兼容其他平台）
  >     hit = getattr(stats, 'prompt_cache_hit_tokens', 0)
  >     miss = getattr(stats, 'prompt_cache_miss_tokens', 
  >                   stats.prompt_tokens - hit if hasattr(stats, 'prompt_tokens') else 0)
  >     
  >     print(f"===== TOKEN 消耗明细 =====")
  >     # 仅在存在缓存机制时显示细节
  >     if hit + miss > 0:
  >         print(f"输入: {stats.prompt_tokens} tokens [缓存命中: {hit} | 未命中: {miss}]")
  >     else:
  >         print(f"输入: {stats.prompt_tokens} tokens")
  >     
  >     print(f"输出: {stats.completion_tokens} tokens")
  > 
  >     # 尝试获取推理过程详情
  >     details = getattr(stats, 'completion_tokens_details', None)
  >     reasoning = 0
  >     if details:
  >         if not isinstance(details, dict):
  >             details = getattr(details, 'dict', lambda: {})()
  >         # 尝试获取 reasoning_tokens
  >         reasoning = details.get('reasoning_tokens', 0)
  >         
  >         # 仅在存在推理tokens数量字段时处理
  >         if reasoning > 0:
  >             final = stats.completion_tokens - reasoning
  >             print(f"├─ 推理过程: {reasoning} tokens")
  >             print(f"└─ 最终回答: {final} tokens")
  >     
  >     print(f"总消耗: {stats.total_tokens} tokens")
  >     
  >     # 动态成本计算
  >     input_cost_total = (hit * cache_hit_cost + miss * input_cost) / 1_000_000
  >     output_cost_total = stats.completion_tokens * output_cost / 1_000_000
  >     total_cost = input_cost_total + output_cost_total
  >     
  >     print(f"\n===== 成本明细 =====")
  >     print(f"输入成本: ￥{input_cost_total:.4f} 元")
  >     print(f"输出成本: ￥{output_cost_total:.4f} 元")
  >     print(f"预估总成本: ￥{total_cost:.4f} 元")
  > 
  > print_reasoner_usage(completion)
  > ```
  

> [!note]
>
> 补充几个 DeepSeek API 字段的官方说明[^2]：
>
> - **message.reasoning_content**：仅适用于 deepseek-reasoner 模型。内容为 assistant 消息中在最终答案之前的推理内容。
> - **usage.prompt_cache_hit_tokens**：用户 prompt 中，命中上下文缓存的 token 数。
> - **usage.prompt_cache_miss_tokens**：用户 prompt 中，未命中上下文缓存的 token 数。
>
> 注意，字段 `usage.prompt_cache_hit_tokens` 和 `usage.prompt_cache_miss_tokens` 仅存在于 DeepSeek API，故代码部分做了特殊处理。

[^2]: [对话补全 - DeepSeek API 文档](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion)

**下一章**：[DeepSeek API 流式输出解析 - OpenAI SDK](./DeepSeek%20API%20流式输出解析%20-%20OpenAI%20SDK.md)

## 附录

> 参考链接：[The chat completion object - OpenAI 官方文档](https://platform.openai.com/docs/api-reference/chat/object). 

对话补全（chat completion）的响应字段解释如下：

- **id** (string)
  
  对话补全的唯一标识符。
  
- **choices** (array)
  
  对话补全选项的列表。如果请求参数 `n` 大于 1，则会返回多个选项。
  
  每个选项包含以下字段：
  
  - **finish_reason** (string)
  
    模型停止生成 tokens 的原因。可能的取值包括：
  
    - `stop`：模型自然结束生成或匹配到提供的 `stop` 序列。
    - `length`：达到请求中指定的最大 tokens 数量。
    - `content_filter`：被内容过滤器标记而省略部分内容时。
    - `tool_calls`：当模型调用了工具时。
    - `function_call` (Deprecated)：当模型调用了函数时。
  
  > [!note]
  >
  > “选项”（choice）指的是模型生成的每个可能的对话回复。
  >
  > 当我们在请求中设置参数 `n` 大于 1 时，模型会生成多个备选回复，然后这些回复会以列表（array）的形式返回，字段 **index** 表示该回复在列表中的位置（从 0 开始计数），当 `n` 为 1 的时候“选项”就是指当前“对话补全”，此时获取 `message` 的代码就是常见的 `choices[0].message.content`。
  
  - **index** (integer)
  
    该选项在选项列表中的索引位置。
  
  - **message** (object)
    
    模型生成的对话补全消息，包含以下字段：
    
    - **content** (string or null)
    
      消息的文本内容。
    
    - **refusal** (string or null)
      
      模型生成的拒绝消息。
      
    - **tool_calls** (array)
      
      模型生成的工具调用记录，例如函数调用。
      
    - **role** (string)
      
      生成该消息的角色，也就是消息的发送者身份。比如：
      
      - `system`：系统消息。
      - `user`：用户输入。
      - `assistant`：由模型生成的回复。
      
    - **function_call** (Deprecated, object)
      
      已废弃，已由 `tool_calls` 替代。原用于描述模型生成的函数调用的名称及参数。
      
    - **audio** (object or null)
      
      当请求音频输出模式时，此对象包含模型音频响应的数据。
    
  - **logprobs** (object or null)
    
    关于该选项的 log probability 信息。
  
- **created** (integer)
  
  对话补全生成时的 Unix 时间戳（单位：秒）。
  
- **model** (string)

  用于生成对话补全的模型名称。

- **service_tier** (string or null)

  处理该请求所使用的服务层级。

- **system_fingerprint** (string)

  表示模型运行时后端配置的指纹。

  可与请求参数 `seed` 配合使用，用于判断后端配置更改是否可能影响结果的确定性。

- **object** (string)
  
  对象类型，总是 `chat.completion`。
  
- **usage** (object)
  
  本次补全请求的使用统计信息，包含以下字段：
  
  - **completion_tokens** (integer)
  
    生成的补全文本中的 tokens 数量。
  
  - **prompt_tokens** (integer)
  
    提示（prompt）中的 tokens 数量。
  
  - **total_tokens** (integer)
  
    请求中使用的 tokens 总数（prompt + completion），即用户的提示 + 模型的生成。
  
  - **completion_tokens_details** (object)
  
    对补全中使用 tokens 的详细拆分统计。
  
  - **prompt_tokens_details** (object)
  
    对提示中使用 tokens 的详细拆分统计。

> [!note]
>
> 如果有字段未在 [OpenAI 官方文档](https://platform.openai.com/docs/api-reference/chat/object)中出现，需要查阅模型对应平台的 API 文档。