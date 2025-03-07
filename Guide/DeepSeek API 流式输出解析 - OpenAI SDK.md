# DeepSeek API 流式输出解析 - OpenAI SDK

> 大模型服务平台通常采取流式输出方案，允许用户实时查看模型生成的内容，而不是等模型生成完所有文本后一次性查看。本文将以 DeepSeek API 为例，对流式输出返回的数据块（chunk）进行解析，并介绍如何实时“拼接”和展示回复内容。
>
> DeepSeek 官方的 API 偶尔会异常，参考 [API 服务状态](https://status.deepseek.com)，出现异常时可以切换到其他平台进行学习。
>
> **代码文件下载**：[Code](../Demos/deepseek-api-guide-3.ipynb)
>
> **在线链接**：[Kaggle](https://www.kaggle.com/code/aidemos/deepseek-api-guide-3) | [Colab](https://colab.research.google.com/drive/1Hfm7qU75GSvU8cO6RL108ZcmwaugXemo?usp=sharing)
>
> **前置文章**：[DeepSeek API 的获取与对话示例](./DeepSeek%20API%20的获取与对话示例.md)

## 目录

- [认识流式输出](#认识流式输出)
  - [DeepSeek-Chat](#deepseek-chat)
  - [DeepSeek-Reasoner](#deepseek-reasoner)
- [📝 作业](#-作业)
- [附录](#附录)

## 认识流式输出

在调用 API 时，设置参数 `stream=True` 即可开启流式输出，以 DeepSeek 单轮对话的代码样例为例：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="sk-1aad218fdac64263bb4196cf2282f2c2",
    base_url="https://api.deepseek.com/v1",
)

# 单轮对话示例
completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ],
    stream=True, # 开启流式输出
)
```

**结构化打印**：

```python
# 结构化打印每个数据块
from pprint import pprint
for chunk in completion:
    pprint(chunk.model_dump())
```

> [!note]
>
> `completion` 遍历一次后就会被消耗掉，如果需要对同一份流式输出数据进行多次操作，可以在循环中同时处理。
>
> 在[代码文件](../Demos/deepseek-api-guide-3.ipynb)中，每个示例都会启动一次新的对话。

下面以 DeepSeek API 的聊天模型和推理模型为例进行解读，涉及的知识对于使用了 OpenAI SDK 的平台是通用的：

### DeepSeek-Chat

**输出**：

```yaml
# ===== 第一个数据块 =====
{'choices': [{'delta': {'content': '',
                        'function_call': None,
                        'refusal': None,
                        'role': 'assistant',
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1739189255,
 'id': '97a2a86e-e55d-4543-8f58-c6e2399f99db',
 'model': 'deepseek-chat',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'fp_3a5770e1b4',
 'usage': None}
# ===== 第二个数据块 =====
{'choices': [{'delta': {'content': '您好',
                        'function_call': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1739189255,
 'id': '97a2a86e-e55d-4543-8f58-c6e2399f99db',
 'model': 'deepseek-chat',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'fp_3a5770e1b4',
 'usage': None}

# ===== ...（中间数据块）=====

# ===== 最后一个数据块 =====
{'choices': [{'delta': {'content': '',
                        'function_call': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': 'stop',
              'index': 0,
              'logprobs': None}],
 'created': 1739189255,
 'id': '97a2a86e-e55d-4543-8f58-c6e2399f99db',
 'model': 'deepseek-chat',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'fp_3a5770e1b4',
 'usage': {'completion_tokens': 37,
           'prompt_cache_hit_tokens': 0,
           'prompt_cache_miss_tokens': 11,
           'prompt_tokens': 11,
           'prompt_tokens_details': {'cached_tokens': 0},
           'total_tokens': 48}}
```

观察数据块的 `delta`, `usage` 和 `finish_reason` 字段：

- `delta`

  这是数据块中最关键的字段，它记录了本次新增的消息内容，是流式输出时的增量更新对象。

  - `role`

    在第一个数据块中，`delta` 中的 `role` 被设置为 `"assistant"`，用于标识这条消息是由模型（或者说 AI 助手）生成，后续的数据块中该字段为 `None`。

  - `content`

     `delta.content` 表示本次数据块中新增加的消息内容。

    - 第一个数据块中，`content` 为空，但标记了角色信息。
    - 随后的数据块中，`content` 分别返回 `"您好"` 和 `"！"`，这些数据块的 `content` 拼接起来就构成了完整的回复 “您好！...”。
  
  > 在没有启动流式输出时（`stream=False`），`delta` 对应的字段是 `message`，此时完整回复位于 `message.content`。
  
- `usage`

  仅在最后一个数据块返回 Token 的使用统计信息，其它数据块中 `usage` 均为 `None`。

- `finish_reason`

  - 当 `finish_reason` 为 `None` 时，表示当前数据块仅为回复的一部分，模型仍在生成后续内容。
  - 当 `finish_reason` 不为 `None`（例如 `"stop"`、`"length"` 等）时，表示回复已生成完毕。`finish_reason: "stop"` 表示正常结束。

> 前文《[DeepSeek API 输出解析 - OpenAI SDK](./DeepSeek%20API%20输出解析%20-%20OpenAI%20SDK.md#附录) 》的附录部分详细说明了这些字段的含义。

这些数据块输出可以简单按位置分为三种类型：

| 特征              | 首块        | 中间块                  | 尾块                    |
| ----------------- | ----------- | ----------------------- | ----------------------- |
| **delta.role**    | `assistant` | `None`                  | `None`                  |
| **delta.content** | `""`        | 增量文本（如 `"您好"`） | `""`                    |
| **finish_reason** | `None`      | `None`                  | `"stop"` / 其它结束标记 |
| **usage**         | `None`      | `None`                  | Token 统计              |

> [!tip]
>
> 可以将整个回复看作一个“帧序列”，每一帧（数据块）只包含本次更新的部分内容，最终完整的回复由帧中的 `delta.content` 按顺序拼接得到。

**实时打印回复示例**：

```python
for chunk in completion:
    # 判断回复内容是否非空
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')  # 设置 end='' 实现不换行，视觉上拼接输出
```

**拼接回复示例**：

```python
content = ""
for chunk in completion:
    if chunk.choices[0].delta.content:
    	content += chunk.choices[0].delta.content
print(content)
```

### DeepSeek-Reasoner

修改代码中的 `model` 参数即可切换模型（以 DeepSeek 官方平台为例）：

```diff
- completion = client.chat.completions.create(
-     model="deepseek-chat",

+ completion = client.chat.completions.create(
+     model="deepseek-reasoner",
```

> 其他平台参考下表[^1]，对应 `reasoner_model_id` 列：
>
> |              | base_url                                            | chat_model_id             | reasoner_model_id         |
> | ------------ | --------------------------------------------------- | ------------------------- | ------------------------- |
> | DeepSeek     | "https://api.deepseek.com"                          | "deepseek-chat"           | "deepseek-reasoner"       |
> | 硅基流动     | "https://api.siliconflow.cn/v1"                     | "deepseek-ai/DeepSeek-V3" | "deepseek-ai/DeepSeek-R1" |
> | 阿里云百炼   | "https://dashscope.aliyuncs.com/compatible-mode/v1" | "deepseek-v3"             | "deepseek-r1"             |
> | 百度智能云   | "https://qianfan.baidubce.com/v2"                   | "deepseek-v3"             | "deepseek-r1"             |
> | 字节火山引擎 | https://ark.cn-beijing.volces.com/api/v3/           | "deepseek-v3-241226"      | "deepseek-r1-250120"      |
> 
> [^1]: [DeepSeek API 的获取与对话示例](./Deepseek%20API%20的获取与对话示例.md).

修改后运行代码：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="sk-1aad218fdac64263bb4196cf2282f2c2",
    base_url="https://api.deepseek.com/v1",
)

# 单轮对话示例
completion = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
    ],
    stream=True, # 开启流式输出
)

# 结构化打印每个数据块
from pprint import pprint
for chunk in completion:
    pprint(chunk.model_dump())
```

**输出**：

```yaml
# ===== 第一个数据块 =====
{'choices': [{'delta': {'content': None,
                        'function_call': None,
                        'reasoning_content': '',
                        'refusal': None,
                        'role': 'assistant',
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1739243087,
 'id': 'd7e87628-4f15-41bf-9b03-e688c1a0e8d6',
 'model': 'deepseek-reasoner',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'fp_7e73fd9a08',
 'usage': None}
# ===== 第二个数据块 =====
{'choices': [{'delta': {'content': None,
                        'function_call': None,
                        'reasoning_content': '好的',
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1739243087,
 'id': 'd7e87628-4f15-41bf-9b03-e688c1a0e8d6',
 'model': 'deepseek-reasoner',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'fp_7e73fd9a08',
 'usage': None}
 
...（中间数据块 - 思考部分）

# ===== “最后”一个数据块 - 思考部分 =====
{'choices': [{'delta': {'content': None,
                        'function_call': None,
                        'reasoning_content': '。',
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1739243087,
 'id': 'd7e87628-4f15-41bf-9b03-e688c1a0e8d6',
 'model': 'deepseek-reasoner',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'fp_7e73fd9a08',
 'usage': None}
# ===== “第一个”数据块 - 内容部分 =====
 {'choices': [{'delta': {'content': '您好',
                        'function_call': None,
                        'reasoning_content': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1739243087,
 'id': 'd7e87628-4f15-41bf-9b03-e688c1a0e8d6',
 'model': 'deepseek-reasoner',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'fp_7e73fd9a08',
 'usage': None}
 
...（中间数据块 - 内容部分）

# ===== 最后一个数据块 =====
{'choices': [{'delta': {'content': '',
                        'function_call': None,
                        'reasoning_content': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': 'stop',
              'index': 0,
              'logprobs': None}],
 'created': 1739243087,
 'id': 'd7e87628-4f15-41bf-9b03-e688c1a0e8d6',
 'model': 'deepseek-reasoner',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'fp_7e73fd9a08',
 'usage': {'completion_tokens': 196,
           'completion_tokens_details': {'reasoning_tokens': 135},
           'prompt_cache_hit_tokens': 0,
           'prompt_cache_miss_tokens': 13,
           'prompt_tokens': 13,
           'prompt_tokens_details': {'cached_tokens': 0},
           'total_tokens': 209}}
```

从聊天模型切换到推理模型后，数据块中除 `delta.content` 外，还会出现 `delta.reasoning_content` 字段，用于记录模型在生成回复前的“思考”过程文本。整个阶段回复的生成可以分为两个部分：

1. **思考/推理部分**
   当数据块中 `delta.reasoning_content` 不为 `None`（此时 `delta.content` 为 `None`）时，说明该数据块记录的是模型的思考/推理过程。
2. **回复部分**
   当思考结束后，数据块中的 `delta.content` 开始返回实际回答文本，此时 `delta.reasoning_content` 为 `None`。

可以据此来打印和拼接回复。

**实时打印回复示例**：

```python
for chunk in completion:
    # 如果 chunk 中的 reasoning_content 不为 None，说明这是思考部分
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end='')  # 设置 end=''：不换行输出
    # reasoning_content 为 None 则说明到了回复部分，只要 content 不是 None，就打印
    # 这一判断是因为首个 chunk 的 reasoning_content 可能为 ""，直接使用 else 会打印出 content 的值 None
    elif chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

> [!important]
>
> [DeepSeek 官方文档](https://api-docs.deepseek.com/zh-cn/guides/reasoning_model)给出的流式示例为：
>
> ```python
> if chunk.choices[0].delta.reasoning_content:
>  ...
> else:
>  ...
> ```
>
> 这在其他平台使用时可能会导致打印预期外输出 `None` 和拼接报错：`TypeError: can only concatenate str (not "NoneType") to str`。
>
> 正确用法是：
>
> ```python
> if chunk.choices[0].delta.reasoning_content:
>  ...
> elif chunk.choices[0].delta.content:
>  ...
> ```
>
> 因为第一个 chunk 中的 reasoning_content 可能为`""`。
>
> **注意**，在调用**字节火山引擎**的 DeepSeek-R1 API 时，返回的数据块在完成思考过程后会移除 `reasoning_content` 字段。直接访问 `chunk.choices[0].delta.reasoning_content` 会导致错误：
>
> ```
> AttributeError: 'ChoiceDelta' object has no attribute 'reasoning_content'
> ```
>
> 为避免此问题，进一步修改代码为：
>
> ```python
> if getattr(chunk.choices[0].delta, 'reasoning_content', None):
>  ...
> elif chunk.choices[0].delta.content:
>  ...
> ```
>
> 可以将此修改应用至其他平台。

**拼接回复示例**：

```python
reasoning_content = ""
content = ""
for chunk in completion:
    # 如果 chunk 中的 reasoning_content 不为 None，说明这是思考部分
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content
    # reasoning_content 为 None 则说明到了回复部分，只要 content 不是 None，就打印
    # 这一判断是因为首个 chunk 的 reasoning_content 可能为 ""，不做限制会报错：TypeError: can only concatenate str (not "NoneType") to str
    elif chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content
print(f"===== 模型推理过程 =====\n{reasoning_content}")
print(f"===== 模型回复 =====\n{content}")
```

## 📝 作业

1. 思考如何获取流式输出时的 TOKEN 消耗明细？（提示：观察 `usage` 字段）

**下一章**：[DeepSeek API 多轮对话 - OpenAI SDK](./DeepSeek%20API%20多轮对话%20-%20OpenAI%20SDK.md)

## 附录

> 参考链接：[The message delta object - OpenAI 官方文档](https://platform.openai.com/docs/api-reference/assistants-streaming/message-delta-object).

开启流式输出（`stream=True`）后 `delta` 字段的解释如下：

- **id** (string)

  消息的标识符，可在 API 端点中引用。

- **object** (string)

  对象类型，总是 `thread.message.delta`。

- **delta** (object)

  包含消息中发生变化字段的增量对象。

  - **role** (string)

    生成该消息的实体，可能的取值为 `user` 或 `assistant`。

  - **content** (array)

    消息的内容，以文本和/或图片构成的数组形式呈现。

    - **Image file** (object)

      引用消息内容中的图片文件。

    - **Text** (object)

      消息中包含的文本内容。

    - **Refusal** (object)

      消息中包含的拒绝内容。

    - **Image URL** (object)

      引用消息内容中的图片 URL。

> [!note]
>
> 需要注意的是，实际返回的字段可能并非严格遵循以上结构，例如本文中的 `refusal` 字段直接位于 `delta` 下而不是嵌套在 `delta.content` 内。
