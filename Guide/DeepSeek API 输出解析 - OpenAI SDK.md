# DeepSeek API 输出解析 - OpenAI SDK

> **代码文件下载**：[Code](../Demos/deepseek-api-guide-2.ipynb)
>
> **在线链接**：[Kaggle](https://www.kaggle.com/code/aidemos/deepseek-api-guide-2) | [Colab](https://colab.research.google.com/drive/1WT0jpeIzWewoN5cT12Uwi92d5_tNff2J?usp=sharing)
>
> **前置文章**：[DeepSeek API 的获取与对话示例](./DeepSeek%20API%20的获取与对话示例.md)

## 目录

- [如何切换平台](#如何切换平台)
- [认识输出](#认识输出)
  - [非思考模式](#非思考模式)
  - [思考模式](#思考模式)
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
    model="deepseek-v4-flash", # 3：模型标识（model_id）可能存在差异
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ],
    extra_body={"thinking": {"type": "disabled"}},  # 关闭思考
)
```

当前大模型平台都适配了 OpenAI SDK，所以只需要对应修改三个参数（分别对应于代码中的注释）：

1. api_key：《[DeepSeek API 的获取与对话示例](./DeepSeek%20API%20的获取与对话示例.md)》。
2. base_url。

3. model。

**不同平台参数对照表**[^1]：

|              | base_url                                            | model_id                  |
| ------------ | --------------------------------------------------- | ------------------------- |
| DeepSeek     | "https://api.deepseek.com"                          | "deepseek-v4-flash"           |
| 硅基流动     | "https://api.siliconflow.cn/v1"                     | "deepseek-ai/DeepSeek-V4-Flash" |
| 阿里云百炼   | "https://dashscope.aliyuncs.com/compatible-mode/v1" | "deepseek-v4-flash"             |
| 百度智能云   | "https://qianfan.baidubce.com/v2"                   | "deepseek-v4-flash"             |
| 字节火山引擎 | https://ark.cn-beijing.volces.com/api/v3            | "deepseek-v4-flash-260425" |

以硅基流动平台为例，使用 chat 模型，修改如下：

```diff
- client = OpenAI(
-     api_key="your-api-key", #1
-     base_url="https://api.deepseek.com", # 2
- )
- completion = client.chat.completions.create(
-     model="deepseek-v4-flash", # 3

+ client = OpenAI(
+     api_key="your-api-key", #1
+     base_url="https://api.siliconflow.cn/v1", # 2
+ )
+ completion = client.chat.completions.create(
+     model="deepseek-ai/DeepSeek-V4-Flash", # 3
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
    model="deepseek-ai/DeepSeek-V4-Flash", # 3
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ],
    extra_body={"thinking": {"type": "disabled"}},  # 关闭思考
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

下面以 DeepSeek API 的非思考和思考模式为例进行解读，涉及的知识对于使用了 OpenAI SDK 的平台是通用的：

> 现在 V4 默认开启思考，下文以非思考开头是因为早期 DeepSeek 分为对话模型 V3 和思考模型 R1，所以这里保留顺序演示。

### 非思考模式

**输出**：

```yaml
{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'logprobs': None,
              'message': {'annotations': None,
                          'audio': None,
                          'content': '你好！我是DeepSeek，由深度求索公司创造的AI助手。我是一个纯文本模型，能够帮你解答问题、提供信息、进行对话等。我支持中文、英文等多种语言，并且可以处理长文本（上下文高达1M）。虽然我不支持多模态识别，但你可以上传图片、PDF、Word等文件，我会从中读取文字信息来帮助你。\n'
                                     '\n'
                                     '我的知识截止日期是2025年5月，目前是免费使用的。有什么我可以帮你的吗？😊',
                          'function_call': None,
                          'refusal': None,
                          'role': 'assistant',
                          'tool_calls': None}}],
 'created': 1783351586,
 'id': '019f3809ad3d1bd505e366096e788369',
 'model': 'deepseek-ai/DeepSeek-V4-Flash',
 'object': 'chat.completion',
 'service_tier': None,
 'system_fingerprint': '',
 'usage': {'completion_tokens': 102,
           'completion_tokens_details': {'accepted_prediction_tokens': None,
                                         'audio_tokens': None,
                                         'reasoning_tokens': 0,
                                         'rejected_prediction_tokens': None},
           'prompt_cache_hit_tokens': 0,
           'prompt_cache_miss_tokens': 12,
           'prompt_tokens': 12,
           'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0},
           'total_tokens': 114}}
```

**通过输出的字段我们能做些什么？**（详细说明见[附录](#附录)）：

- **获取模型回复（choices）**

  ```python
  print(completion.choices[0].message.content)
  ```

  **输出**：

  ```
  你好！我是DeepSeek，由深度求索公司创造的AI助手。我是一个纯文本模型，能够帮你解答问题、提供信息、进行对话等。我支持中文、英文等多种语言，并且可以处理长文本（上下文高达1M）。虽然我不支持多模态识别，但你可以上传图片、PDF、Word等文件，我会从中读取文字信息来帮助你。

  我的知识截止日期是2025年5月，目前是免费使用的。有什么我可以帮你的吗？😊
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
      
      # 按 DeepSeek v4-flash 定价计算成本（单位：元，思考/非思考同价）
      # - 输入: 1元/百万 tokens（缓存命中 0.02元/百万 tokens）
      # - 输出: 2元/百万 tokens
      # 官方价格文档：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
      input_cost = (hit * 0.02 + miss * 1) / 1_000_000
      output_cost = stats.completion_tokens * 2 / 1_000_000
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
  输入: 12 tokens [缓存命中: 0 | 未命中: 12]
  输出: 102 tokens
  总消耗: 114 tokens

  ===== 成本明细 =====
  输入成本: ￥0.0000 元
  输出成本: ￥0.0002 元
  预估总成本: ￥0.0002 元
  ```
  
  > [!important]
  >
  > 非 DeepSeek 官方平台不存在一些特殊字段（比如：`usage.prompt_cache_hit_tokens`），一个更兼容的版本：
  >
  > ```python
  > def print_chat_usage(completion, input_cost=1.0, output_cost=2.0, cache_hit_cost=0.02):
  >     """
  >      参数:
  >     - input_cost: 输入价格（元/百万 tokens）
  >     - output_cost: 输出价格（元/百万 tokens）
  >     - cache_hit_cost: 缓存命中价格（当平台不支持时自动退化到全价模式）
  > 
  >     按 DeepSeek 聊天模型定价设定默认成本（单位：元）：
  >     - 输入: 1元/百万 tokens（缓存命中 0.02元/百万 tokens）
  >     - 输出: 2元/百万 tokens
  >     官方价格文档：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
  >     """
  >     stats = completion.usage
  > 
  >     # 尝试获取字段（兼容其他平台）
  >     hit = getattr(stats, 'prompt_cache_hit_tokens', None)  # 无缓存机制的平台没有该字段
  >     has_cache = hit is not None  # 注意：命中 0 个时字段仍存在，值为 0
  >     if not has_cache:
  >         hit = 0  # 后续统一按数字处理
  >     miss = getattr(stats, 'prompt_cache_miss_tokens', 
  >                   stats.prompt_tokens - hit if hasattr(stats, 'prompt_tokens') else 0)
  > 
  >     print(f"===== TOKEN 消耗明细 =====")
  >     # 仅在平台提供缓存字段时显示细节
  >     if has_cache:
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

### 思考模式

V4 起思考与非思考共用同一模型，删除关闭思考的 `extra_body` 参数即可切换到思考模式（以 DeepSeek 官方平台为例）：

```diff
  completion = client.chat.completions.create(
      model="deepseek-v4-flash", # 3
-     extra_body={"thinking": {"type": "disabled"}},  # 关闭思考
```

> 其他平台参考下表[^1]，对应 `model_id` 列：
>
> |              | base_url                                            | model_id                                                     |
> | ------------ | --------------------------------------------------- | ------------------------------------------------------------ |
> | DeepSeek     | "https://api.deepseek.com"                          | "deepseek-v4-flash"                                              |
> | 硅基流动     | "https://api.siliconflow.cn/v1"                     | "deepseek-ai/DeepSeek-V4-Flash"                                    |
> | 阿里云百炼   | "https://dashscope.aliyuncs.com/compatible-mode/v1" | "deepseek-v4-flash"                                                |
> | 百度智能云   | "https://qianfan.baidubce.com/v2"                   | "deepseek-v4-flash"                                                |
> | 字节火山引擎 | https://ark.cn-beijing.volces.com/api/v3/           | "deepseek-v4-flash-260425" |

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
    model="deepseek-ai/DeepSeek-V4-Flash", # 3：无需更换模型
    extra_body={"thinking": {"type": "enabled"}},  # 硅基流动上偶尔不默认思考，所以这里显式开启
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
              'message': {'annotations': None,
                          'audio': None,
                          'content': '你好！我是DeepSeek，由深度求索公司创造的AI助手。我可以帮你解答问题、提供信息、进行对话交流等。我的知识截止于2025年5月，支持处理文本、阅读链接和上传的文件（如图片、PDF、Word、Excel等）。目前我是免费使用的，欢迎随时向我提问！有什么我可以帮你的吗？😊',
                          'function_call': None,
                          'reasoning_content': '嗯，用户问“你是谁”，这是一个简单的自我介绍问题。用户可能是初次接触，需要了解我的身份和功能。我可以直接、清晰地说明我是DeepSeek，由深度求索公司创造，并简要介绍我的核心能力，比如知识范围、文件处理、长上下文和免费使用。最后用友好的语气询问是否需要帮助，这样既回答了问题，也开启了进一步对话。',
                          'refusal': None,
                          'role': 'assistant',
                          'tool_calls': None}}],
 'created': 1783351630,
 'id': '019f380a56932f25b566a2f9ff08862a',
 'model': 'deepseek-ai/DeepSeek-V4-Flash',
 'object': 'chat.completion',
 'service_tier': None,
 'system_fingerprint': '',
 'usage': {'completion_tokens': 157,
           'completion_tokens_details': {'accepted_prediction_tokens': None,
                                         'audio_tokens': None,
                                         'reasoning_tokens': 80,
                                         'rejected_prediction_tokens': None},
           'prompt_cache_hit_tokens': 0,
           'prompt_cache_miss_tokens': 12,
           'prompt_tokens': 12,
           'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0},
           'total_tokens': 169}}
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
  嗯，用户问“你是谁”，这是一个简单的自我介绍问题。用户可能是初次接触，需要了解我的身份和功能。我可以直接、清晰地说明我是DeepSeek，由深度求索公司创造，并简要介绍我的核心能力，比如知识范围、文件处理、长上下文和免费使用。最后用友好的语气询问是否需要帮助，这样既回答了问题，也开启了进一步对话。
  ===== 模型回复 =====
  你好！我是DeepSeek，由深度求索公司创造的AI助手。我可以帮你解答问题、提供信息、进行对话交流等。我的知识截止于2025年5月，支持处理文本、阅读链接和上传的文件（如图片、PDF、Word、Excel等）。目前我是免费使用的，欢迎随时向我提问！有什么我可以帮你的吗？😊
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
  >     参数:
  >     - completion (object): API 返回的对象
  >     
  >     返回:
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
          reasoning = details.reasoning_tokens
          final = stats.completion_tokens - reasoning
          print(f"├─ 推理过程: {reasoning} tokens")
          print(f"└─ 最终回答: {final} tokens")
      
      print(f"总消耗: {stats.total_tokens} tokens")
      
      # 按 DeepSeek v4-flash 定价计算成本（单位：元，思考/非思考同价）
      # - 输入: 1元/百万 tokens（缓存命中 0.02元/百万 tokens）
      # - 输出: 2元/百万 tokens
      # 官方价格文档：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
      input_cost = (hit * 0.02 + miss * 1) / 1_000_000
      output_cost = stats.completion_tokens * 2 / 1_000_000
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
  输入: 12 tokens [缓存命中: 0 | 未命中: 12]
  输出: 157 tokens
  ├─ 推理过程: 80 tokens
  └─ 最终回答: 77 tokens
  总消耗: 169 tokens

  ===== 成本明细 =====
  输入成本: ￥0.0000 元
  输出成本: ￥0.0003 元
  预估总成本: ￥0.0003 元
  ```
  
  > [!important]
  >
  > 非 DeepSeek 官方的部分平台（但百度智能云存在）不存在一些特殊字段（比如：`reasoning_tokens`），一个更兼容的版本：
  >
  > ```python
  > def print_reasoner_usage(completion, input_cost=1.0, output_cost=2.0, cache_hit_cost=0.02):
  >     """
  >     参数：
  >     - input_cost: 输入价格（元/百万 tokens）
  >     - output_cost: 输出价格（元/百万 tokens）
  >     - cache_hit_cost: 缓存命中价格（当平台不支持时自动退化到全价模式）
  > 
  >     按 DeepSeek 推理模型定价设定默认成本（单位：元）：
  >     - 输入: 1元/百万 tokens（缓存命中 0.02元/百万 tokens）
  >     - 输出: 2元/百万 tokens
  >     官方价格文档：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
  >     """
  >     stats = completion.usage
  >     
  >     # 尝试获取字段（兼容其他平台）
  >     hit = getattr(stats, 'prompt_cache_hit_tokens', None)  # 无缓存机制的平台没有该字段
  >     has_cache = hit is not None  # 注意：命中 0 个时字段仍存在，值为 0
  >     if not has_cache:
  >         hit = 0  # 后续统一按数字处理
  >     miss = getattr(stats, 'prompt_cache_miss_tokens', 
  >                   stats.prompt_tokens - hit if hasattr(stats, 'prompt_tokens') else 0)
  >     
  >     print(f"===== TOKEN 消耗明细 =====")
  >     # 仅在平台提供缓存字段时显示细节
  >     if has_cache:
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
> - **message.reasoning_content**：仅在思考模式下返回（V3/R1 时代对应 deepseek-reasoner 模型）。内容为 assistant 消息中在最终答案之前的推理内容。
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
      
    - **annotations** (array or null)
      
      消息的注释列表，配合内置工具（如网页搜索）使用时用于标注引用来源。
      
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
  
    对补全中使用 tokens 的详细拆分统计，其中 `reasoning_tokens` 为思考过程消耗的 tokens（正文用量代码中用到）。
  
  - **prompt_tokens_details** (object)
  
    对提示中使用 tokens 的详细拆分统计，其中 `cached_tokens` 为命中缓存的 tokens。

> [!note]
>
> 如果有字段未在 [OpenAI 官方文档](https://platform.openai.com/docs/api-reference/chat/object)中出现，需要查阅模型对应平台的 API 文档。
