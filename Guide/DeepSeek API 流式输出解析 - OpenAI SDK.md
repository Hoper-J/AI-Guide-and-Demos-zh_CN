# DeepSeek API 流式输出解析 - OpenAI SDK

> 大模型服务平台通常采取流式输出方案，允许用户实时查看模型生成的内容，而不是等模型生成完所有文本后一次性查看。本文将以 DeepSeek API 为例，对流式输出返回的数据块（chunk）进行解析，并介绍如何实时“拼接”和展示回复内容。
>
> **代码文件下载**：[Code](../Demos/deepseek-api-guide-3.ipynb)
>
> **在线链接**：[Kaggle](https://www.kaggle.com/code/aidemos/deepseek-api-guide-3) | [Colab](https://colab.research.google.com/drive/1qZ1Ihc2XjD91tocjjrCp5cHVq4_WDBVd?usp=sharing)
>
> **前置文章**：[DeepSeek API 的获取与对话示例](./DeepSeek%20API%20的获取与对话示例.md)

## 目录

- [认识流式输出](#认识流式输出)
  - [非思考模式](#非思考模式)
  - [思考模式](#思考模式)
- [📝 作业](#-作业)
- [附录](#附录)

## 认识流式输出

在调用 API 时，设置参数 `stream=True` 即可开启流式输出，以 DeepSeek 单轮对话的代码样例为例：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com",
)

# 单轮对话示例
completion = client.chat.completions.create(
    model="deepseek-flash",
    extra_body={"thinking": {"type": "disabled"}},  # 关闭思考
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '请用一句话介绍你自己'}
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

下面以 DeepSeek API 的非思考和思考模式为例进行解读，涉及的知识对于使用了 OpenAI SDK 的平台是通用的：

### 非思考模式

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
 'created': 1792916248,
 'id': 'f9ee4597-6ead-4d32-85a5-612d99ad7948',
 'model': 'deepseek-flash',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'e72d3c7f663d3e1383b00b3884cb4501',
 'usage': None}

# ===== 第二个数据块 =====
{'choices': [{'delta': {'content': '我是',
                        'function_call': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1792916248,
 'id': 'f9ee4597-6ead-4d32-85a5-612d99ad7948',
 'model': 'deepseek-flash',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'e72d3c7f663d3e1383b00b3884cb4501',
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
 'created': 1792916248,
 'id': 'f9ee4597-6ead-4d32-85a5-612d99ad7948',
 'model': 'deepseek-flash',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'e72d3c7f663d3e1383b00b3884cb4501',
 'usage': {'completion_tokens': 22,
           'completion_tokens_details': None,
           'prompt_cache_hit_tokens': 0,
           'prompt_cache_miss_tokens': 16,
           'prompt_tokens': 16,
           'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0},
           'total_tokens': 38}}
```

观察数据块的 `delta`, `usage` 和 `finish_reason` 字段：

- `delta`

  这是数据块中最关键的字段，它记录了本次新增的消息内容，是流式输出时的增量更新对象。

  - `role`

    在第一个数据块中，`delta` 中的 `role` 被设置为 `"assistant"`，用于标识这条消息是由模型（或者说 AI 助手）生成，后续的数据块中该字段为 `None`。

  - `content`

     `delta.content` 表示本次数据块中新增加的消息内容。

    - 第一个数据块中，`content` 为空，但标记了角色信息。
    - 后续数据块逐段返回文本，按接收顺序拼接各块的 `content` 就能得到完整回复。
  
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
| **delta.content** | `""`        | 增量文本 | `""`                    |
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

**输出**：

```
我是DeepSeek，由深度求索公司创造的AI助手，致力于为你提供热情、细腻的帮助！
```

### 思考模式

V4 起思考与非思考共用同一模型，删除关闭思考的 `extra_body` 参数即可切换到思考模式（以 DeepSeek 官方平台为例）：

```diff
  completion = client.chat.completions.create(
      model="deepseek-flash",
-     extra_body={"thinking": {"type": "disabled"}},  # 关闭思考
```

> 其他平台参考下表[^1]，对应 `model_id` 列：
>
> |              | base_url                                            | model_id                  |
> | ------------ | --------------------------------------------------- | ------------------------- |
> | DeepSeek     | "https://api.deepseek.com"                          | "deepseek-flash"              |
> | 硅基流动     | "https://api.siliconflow.cn/v1"                     | "deepseek-ai/DeepSeek-V4.1-Flash" |
> | 阿里云百炼   | "https://dashscope.aliyuncs.com/compatible-mode/v1" | "deepseek-v4.1-flash"             |
> | 百度智能云   | "https://qianfan.baidubce.com/v2"                   | "deepseek-v4.1-flash"             |
> | 字节火山引擎 | https://ark.cn-beijing.volces.com/api/v3/           | "deepseek-v4-1-flash-261017"      |
> 
> [^1]: [DeepSeek API 的获取与对话示例](./DeepSeek%20API%20的获取与对话示例.md).

修改后运行代码：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com",
)

# 单轮对话示例
completion = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '请用一句话介绍你自己'}
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
 'created': 1792916250,
 'id': '8a25b09b-741b-47af-ae47-6ec9805aa1a5',
 'model': 'deepseek-flash',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'e72d3c7f663d3e1383b00b3884cb4501',
 'usage': None}

# ===== 第二个数据块 =====
{'choices': [{'delta': {'content': None,
                        'function_call': None,
                        'reasoning_content': 'The',
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1792916250,
 'id': '8a25b09b-741b-47af-ae47-6ec9805aa1a5',
 'model': 'deepseek-flash',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'e72d3c7f663d3e1383b00b3884cb4501',
 'usage': None}

# ===== ...（中间数据块）=====

# ===== 最后一个含思考文本的数据块 =====
{'choices': [{'delta': {'content': None,
                        'function_call': None,
                        'reasoning_content': '.',
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1792916250,
 'id': '8a25b09b-741b-47af-ae47-6ec9805aa1a5',
 'model': 'deepseek-flash',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'e72d3c7f663d3e1383b00b3884cb4501',
 'usage': None}

# ===== 第一个含回复文本的数据块 =====
{'choices': [{'delta': {'content': '我是',
                        'function_call': None,
                        'reasoning_content': None,
                        'refusal': None,
                        'role': None,
                        'tool_calls': None},
              'finish_reason': None,
              'index': 0,
              'logprobs': None}],
 'created': 1792916250,
 'id': '8a25b09b-741b-47af-ae47-6ec9805aa1a5',
 'model': 'deepseek-flash',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'e72d3c7f663d3e1383b00b3884cb4501',
 'usage': None}

# ===== ...（中间数据块）=====

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
 'created': 1792916250,
 'id': '8a25b09b-741b-47af-ae47-6ec9805aa1a5',
 'model': 'deepseek-flash',
 'object': 'chat.completion.chunk',
 'service_tier': None,
 'system_fingerprint': 'e72d3c7f663d3e1383b00b3884cb4501',
 'usage': {'completion_tokens': 50,
           'completion_tokens_details': {'accepted_prediction_tokens': None,
                                         'audio_tokens': None,
                                         'reasoning_tokens': 26,
                                         'rejected_prediction_tokens': None},
           'prompt_cache_hit_tokens': 0,
           'prompt_cache_miss_tokens': 41,
           'prompt_tokens': 41,
           'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0},
           'total_tokens': 91}}
```

从非思考模式切换到思考模式后，数据块中除 `delta.content` 外，还会出现 `delta.reasoning_content` 字段。读取时需要分别处理这两种内容：

1. **思考/推理部分**
   当 `delta.reasoning_content` 有文本时，将它追加到推理过程。
2. **回复部分**
   当 `delta.content` 有文本时，将它追加到最终回复。

通常先收到推理内容，再收到正文，但同一个数据块也可能同时包含推理末尾和正文开头。因此两个字段需要独立判断，不能用 `if/elif` 将它们当成互斥选项，没有文本的字段则跳过。

> [!note]
>
> 以阿里云百炼的 `deepseek-v4.1-flash` 为例，同样的代码多次运行，有时会收到下面这样的数据块，推理的最后几个字符和正文的开头落在了同一块里：
>
> ```yaml
> {'choices': [{'delta': {'content': '我是由',
>                         'function_call': None,
>                         'reasoning_content': ' Chinese.',
>                         'refusal': None,
>                         'role': None,
>                         'tool_calls': None},
>               'finish_reason': None,
>               'index': 0,
>               'logprobs': None}],
>  'created': 1793025279,
>  'id': 'chatcmpl-6c2e409d-f575-9d15-b23b-f217891bc772',
>  'model': 'deepseek-v4.1-flash',
>  'object': 'chat.completion.chunk',
>  'service_tier': None,
>  'system_fingerprint': None,
>  'usage': None}
> ```
>
> 如果沿用 `if/elif` 的写法，这一块的 `content` 会被跳过，拼出来的回复就少了开头的几个字。

**实时打印回复示例**：

```python
for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta is None:
        continue
    reasoning = getattr(delta, 'reasoning_content', None)

    # 两个字段独立处理，同一块可能同时含有推理和正文
    if reasoning:
        print(reasoning, end='')
    if delta.content:
        print(delta.content, end='')
```

> [!important]
>
> - `reasoning_content` 是扩展字段，部分平台会在正文阶段省略它，使用 `getattr(delta, 'reasoning_content', None)` 读取。
> - 两个字段都可能是 `None` 或空字符串，先判断是否有文本，再打印或拼接。
> - 同一块可能同时包含两种文本，使用两个独立的 `if`，避免漏掉正文。
> - 没有 `choices` 或 `delta` 的数据块不参与文本拼接，例如某些平台单独返回的用量统计块。

**拼接回复示例**：

```python
reasoning_content = ""
content = ""
for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta is None:
        continue
    reasoning = getattr(delta, 'reasoning_content', None)

    # 两个字段独立处理，同一块可能同时含有推理和正文
    if reasoning:
        reasoning_content += reasoning
    if delta.content:
        content += delta.content
print(f"===== 模型推理过程 =====\n{reasoning_content}")
print(f"===== 模型回复 =====\n{content}")
```

**输出**：

```
===== 模型推理过程 =====
我们需要回答中文一句话介绍自己。需要简洁。用户问“请用一句话介绍你自己”。我们作为助手，一句话即可。要符合身份。例如：“我是由深度求索公司开发的AI助手，乐于为你提供信息与帮助。” 注意不要声称？ 我是 DeepSeek? 系统没给具体身份，但通常可以答AI助手。一句话。保持自然。用中文。
===== 模型回复 =====
我是一个乐于助人的AI助手，愿意为你解答问题、提供信息与帮助。
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
