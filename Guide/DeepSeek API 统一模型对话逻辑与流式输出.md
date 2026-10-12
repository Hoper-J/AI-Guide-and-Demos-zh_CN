# DeepSeek API 统一模型对话逻辑与流式输出

> **代码文件下载**：[Code](../Demos/deepseek-api-guide-5.ipynb)
>
> **在线链接**：[Kaggle](https://www.kaggle.com/code/aidemos/deepseek-api-guide-5) | [Colab](https://colab.research.google.com/drive/1N-twn0YVMhvEQ_6hmDWI1giP6ipkzeQs?usp=sharing)

## 目录

- [进一步](#进一步)
  - [使用示例](#使用示例)
- [流式输出](#流式输出)
  - [核心逻辑](#核心逻辑)
  - [引入生成器](#引入生成器)
  - [代码实现](#代码实现)
    - [使用示例](#使用示例-1)
- [📝 作业](#-作业)

在[上一篇文章](./DeepSeek%20API%20多轮对话%20-%20OpenAI%20SDK.md)中，我们分别使用了 `ChatSession` 和 `ReasonerSession` 两个类来处理非思考与思考两种模式，这种划分来自 V3/R1 时代（2024 年底至 2025 年初），当时对话模型（V3）和推理模型（R1）是两个独立模型。现在 V4 出来后，思考变成了每次请求可以开关的选项，模型族也不再区分对话和推理，同一个模型两种返回都会遇到[^1]：

- **非思考模式**

  ```yaml
  {'choices': [{'finish_reason': 'stop',
                'index': 0,
                'logprobs': None,
                'message': {'content': '...',  # 模型回复
                            'function_call': None,
                            'refusal': None,
                            'role': 'assistant',
                            'tool_calls': None}}],
   ...}
  ```

- **思考模式**

  ```yaml
  {'choices': [{'finish_reason': 'stop',
                'index': 0,
                'logprobs': None,
                'message': {'content': '...',  # 模型回复
                            'function_call': None,
                            'reasoning_content': '...',  # 推理思考过程
                            'refusal': None,
                            'role': 'assistant',
                            'tool_calls': None}}],
   ...}
  ```

[^1]: [DeepSeek API 输出解析 - OpenAI SDK](./DeepSeek%20API%20输出解析%20-%20OpenAI%20SDK.md#思考模式).

可以观察到：思考模式的 `message` 部分比非思考模式**仅**多了一个 `reasoning_content` 字段，用于记录模型的推理思考过程。

**那么，能不能用一个 `ChatSession` 类同时处理两种模式呢？**

当然可以，只需要额外取一下 `reasoning_content` 字段：

```python
from openai import OpenAI
import os

class ChatSession:
    def __init__(self, client, model="deepseek-flash", system_message="You are a helpful assistant.", thinking=None):
        """
        参数:
        - client (openai.OpenAI): OpenAI 客户端实例
        - model (str): 模型名称，默认为 'deepseek-flash'（其他官方模型如 'deepseek-v4-pro' 用法相同）
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
        - thinking (bool|None): 思考模式开关，None 使用平台默认（当前默认开启），True/False 显式开/关
        """
        self.client = client
        self.model = model
        self.thinking = thinking  # None=平台默认（当前默认开启），True/False=显式开/关思考（V4 生效）
        self.messages = [{'role': 'system', 'content': system_message}]

    def append_message(self, role, content):
        """
        添加一条对话消息
        参数:
        - role (str): 消息角色，为 'user' 或 'assistant'。
        - content (str): 消息内容。
        """
        self.messages.append({'role': role, 'content': content})

    def get_response(self, user_input):
        """
        添加用户消息，调用 API 获取回复，并返回推理过程和回复内容
        参数:
        - user_input (str): 用户输入的消息
        返回：
        - (reasoning_content, content) (tuple): 推理过程（仅思考模式有）和回复内容
        """
        # 记录用户输入
        self.append_message('user', user_input)
        
        # 调用 API
        completion = self.client.chat.completions.create(
            model=self.model,
            **({"extra_body": {"thinking": {"type": "enabled" if self.thinking else "disabled"}}} if self.thinking is not None else {}),
            messages=self.messages
        )
        
        # 获取回复内容和推理过程（如果有）
        content = completion.choices[0].message.content
        reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', None)
        
        # 记录模型回复
        self.append_message('assistant', content)
        
        return reasoning_content, content
```

## 进一步

如果不想每次手动创建 `client`，可以让 `ChatSession` 在初始化时直接完成 `OpenAI` 客户端的实例化，只需要传入 `api_key` 和 `base_url` 参数：

```python
from openai import OpenAI
import os

class ChatSession:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-flash", system_message="You are a helpful assistant.", thinking=None):
        """
        参数:
        - api_key (str): 平台的 API Key，默认从环境变量 `DEEPSEEK_API_KEY` 读取
        - base_url (str): API 请求地址，默认为 DeepSeek 官方平台
        - model (str): 模型名称，默认为 'deepseek-flash'（其他官方模型如 'deepseek-v4-pro' 用法相同）
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
        - thinking (bool|None): 思考模式开关，None 使用平台默认（当前默认开启），True/False 显式开/关
        """
        # 处理 API Key 优先级：显式传入 > 环境变量
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API Key 未提供，请通过参数传入或设置环境变量 DEEPSEEK_API_KEY")
        self.base_url = base_url
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        self.model = model
        self.thinking = thinking  # None=平台默认（当前默认开启），True/False=显式开/关思考（V4 生效）
        self.messages = [{'role': 'system', 'content': system_message}]

    def append_message(self, role, content):
        """
        添加一条对话消息
        参数:
        - role (str): 消息角色，为 'user' 或 'assistant'
        - content (str): 消息内容
        """
        self.messages.append({'role': role, 'content': content})

    def get_response(self, user_input):
        """
        添加用户消息，调用 API 获取回复，并返回推理过程和回复内容
        参数：
        - user_input (str): 用户输入的消息
        返回:
        - (reasoning_content, content) (tuple): 推理过程（仅思考模式有）和回复内容
        """
        # 记录用户输入
        self.append_message('user', user_input)
        
        # 调用 API
        completion = self.client.chat.completions.create(
            model=self.model,
            **({"extra_body": {"thinking": {"type": "enabled" if self.thinking else "disabled"}}} if self.thinking is not None else {}),
            messages=self.messages
        )
        
        # 获取回复内容和推理过程（如果有）
        content = completion.choices[0].message.content
        reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', None)
        
        # 记录模型回复
        self.append_message('assistant', content)
        
        return reasoning_content, content
```

### 使用示例

**参数设置**：

```python
config = {
    "api_key": "your-api-key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-flash",
    "thinking": False,  # 关闭思考，True 开启，None 使用平台默认（当前默认开启）
    "system_message": "You are a helpful assistant."
}

session = ChatSession(**config)
```

> [!note]
>
> `**` 是 Python 中的解包操作符，它将字典中的键值对解包为函数的关键字参数。在这里，`**config` 将字典中的参数逐一传递给 `ChatSession()`， `ChatSession(**config)` 等价于：
>
> ```python
> ChatSession(
>        api_key="your-api-key",
>        base_url="https://api.deepseek.com",
>        model="deepseek-flash",
>        thinking=False,  # 关闭思考
>        system_message="You are a helpful assistant."
> )
> ```
>

**对话**：

```python
user_input = "请用一句话介绍你自己"
reasoning, reply = session.get_response(user_input)

# 非思考模式的 reasoning 为 None
if reasoning:
    print(f"===== 推理过程 =====\n{reasoning}\n")
print(f"===== 模型回复 =====\nAI: {reply}\n")
```

## 流式输出

在文章《[DeepSeek API 流式输出解析 - OpenAI SDK](./DeepSeek%20API%20流式输出解析%20-%20OpenAI%20SDK.md)》中，我们介绍了 API 的流式输出：允许实时展示模型的生成过程，无需等待完整回复。现在，让我们把这个功能集成到 `ChatSession` 类中。

### 核心逻辑

重新回顾一下数据流的结构，每个数据块（chunk）中可能包含两种类型的内容：推理过程（`reasoning_content`）和回复内容（`content`）。关键在于：

1. 同一个数据块可能同时包含推理内容和正文，需要分别检查这两个字段。
2. 需要记录完整的回复内容，因为这将被添加到对话历史中。
3. 先处理完当前数据块中的文本，再检查 `finish_reason` 是否表示生成完毕。

所以核心的处理逻辑如下：

```python
def _process_stream(self, completion):
    content = ""

    for chunk in completion:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta is None:
            continue
        reasoning = getattr(delta, 'reasoning_content', None)
        reply = delta.content

        # 先保留这一块已收到的正文，提前关闭时也不会漏掉
        if reply:
            content += reply

        # 两个字段独立处理，同一块可以分别输出推理和正文
        if reasoning:
            print(reasoning, end='')
        if reply:
            print(reply, end='')

        # 处理完当前块中的文本，再判断是否结束
        if choice.finish_reason is not None:
            break

    # 生成完毕，记录完整的模型回复
    self.append_message('assistant', content)
```

### 引入生成器

上面的 `_process_stream` 在收到数据块后直接 `print`，调用方拿不到内容。回顾一下非流式的 `get_response`：它把 `(reasoning_content, content)` 返回给调用方，怎么使用由调用方决定。

**那么，流式输出能不能也把内容“返回”出去，而不是在函数里打印呢？**

可以，但不能用 `return`。因为流式的内容是逐块到达的，而 `return` 只能返回一次。这时候我们需要用 `yield`，把 `_process_stream` 变成生成器（generator）。

> [!tip]
>
> **什么是生成器？**
>
> 举个简单的例子：
>
> ```python
> def count_up():
>     yield 1
>     yield 2
>     yield 3
>
> for n in count_up():
>     print(n)  # 依次打印 1、2、3
> ```
>
> 可以将 `yield` 理解为“交出一个值，然后停在这一行”：外部调用 `count_up()` 并不会执行函数体，只会得到一个生成器对象。`for` 循环每向它要一个值，函数就从上次停下的地方继续执行，遇到下一个 `yield` 再交出值并停下，等函数体执行完毕，循环结束。

把 `print` 换成 `yield`，并采用和非流式输出一致的返回格式 `(reasoning_content, content)`：

```python
def _process_stream(self, completion):
    content = ""  # 用于存储完整回复

    for chunk in completion:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta is None:
            continue
        reasoning = getattr(delta, 'reasoning_content', None)
        reply = delta.content

        # 先保留这一块已收到的正文，提前关闭时也不会漏掉
        if reply:
            content += reply

        # 两个字段独立处理，同一块可以分别输出推理和正文
        if reasoning:
            yield reasoning, None
        if reply:
            yield None, reply

        # 处理完当前块中的文本，再判断是否结束
        if choice.finish_reason is not None:
            break

    # 生成完毕，记录完整的模型回复
    self.append_message('assistant', content)
```

同一个数据块可以产生两次 `yield`，但每次返回的元组仍只携带一种内容。因此调用方可以继续用 `if/else` 区分这两种元组。正文在第一次 `yield` 之前就已累计，为后面的提前关闭处理保留这一块已经收到的内容。

**调用示例**：

```python
for reasoning, reply in chunks:  # chunks 是 _process_stream(...) 返回的生成器
    if reasoning:
        print(reasoning, end='')
    else:
        print(reply, end='')
```

这里演示打印的写法，效果和先前直接 `print` 一样。需要注意的是，调用方现在可以随时 `break` 退出循环（比如看到第一句就发现跑偏了）。此时 `_process_stream` 会停在 `yield`，`append_message` 没有执行，所以这一轮的回复不会写入对话历史。并且，只要调用方还持有这个生成器对象，它就会一直停在那里，直到被关闭——显式调用 `close()`，或者生成器对象被垃圾回收时由 Python 自动调用 `close()`。关闭时，Python 会在生成器停住的 `yield` 处抛出一个 `GeneratorExit`，下文的代码实现会用到它。

如果希望提前退出时也能保留已经收到的回复，可以把 `append_message` 放进 `finally`。`try/finally` 的逻辑是：不管 `try` 中的代码是正常执行完、中途抛出异常，还是生成器在 `yield` 处被关闭，`finally` 中的代码都会执行：

```python
def _process_stream(self, completion):
    content = ""  # 用于存储完整回复

    try:
        for chunk in completion:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is None:
                continue
            reasoning = getattr(delta, 'reasoning_content', None)
            reply = delta.content

            # 先保留这一块已收到的正文，提前关闭时也不会漏掉
            if reply:
                content += reply

            # 两个字段独立处理，同一块可以分别输出推理和正文
            if reasoning:
                yield reasoning, None
            if reply:
                yield None, reply

            # 处理完当前块中的文本，再判断是否结束
            if choice.finish_reason is not None:
                break
    finally:
        # 生成器结束或被关闭时执行，保留已收到的回复
        self.append_message('assistant', content)
```

这样，写入历史的时机就从“循环结束”变成了“生成器关闭”，由调用方决定何时收尾，用完后关闭即可。最简单的做法是使用 `contextlib.closing`，退出 `with` 时会自动调用生成器的 `close()`，具体写法见下文。

### 代码实现

还剩两个问题：

1. **读取流时出错**

   `finally` 同样会把不完整的回复写入历史，失败的一轮也被保留了下来。

2. **user 消息提前写入**

   前面的版本都是先 `append` 用户消息再发请求，如果请求失败，这条孤立的 user 消息会留在历史里，重试时就会出现两条相邻的 user 消息。

下面的版本做了三处调整：

- 用户输入先只放进本次请求，不提前修改 `self.messages`，两种模式都在 `get_response` 中发请求。
- 新增 `_commit`：有正文时才把本轮问答写入历史，两种模式共用这一规则。
- 流式输出改为在生成器正常结束或被调用方关闭时写入。关闭会在 `yield` 处抛出 `GeneratorExit`，用 `except GeneratorExit` 捕获后写入，再 `raise` 继续抛出。
  - 读取流时出现的其他异常不经过这个分支，直接穿过 `finally`，不写入。

```python
class ChatSession:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-flash", system_message="You are a helpful assistant.", thinking=None):
        """
        参数：
        - api_key (str): 平台的 API Key，默认从环境变量 `DEEPSEEK_API_KEY` 读取
        - base_url (str): API 请求地址，默认为 DeepSeek 官方平台
        - model (str): 模型名称，默认为 'deepseek-flash'（其他官方模型如 'deepseek-v4-pro' 用法相同）
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
        - thinking (bool|None): 思考模式开关，None 使用平台默认（当前默认开启），True/False 显式开/关
        """
        # 处理 API Key 优先级：显式传入 > 环境变量
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API Key 未提供，请通过参数传入或设置环境变量 DEEPSEEK_API_KEY")
        self.base_url = base_url

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.model = model
        self.thinking = thinking  # None=平台默认（当前默认开启），True/False=显式开/关思考（V4 生效）
        self.messages = [{'role': 'system', 'content': system_message}]

    def append_message(self, role, content):
        """
        添加一条对话消息

        参数:
        - role (str): 消息角色，为 'user' 或 'assistant'
        - content (str): 消息内容
        """
        self.messages.append({'role': role, 'content': content})

    def _create(self, messages, stream):
        """调用 API，思考开关按 self.thinking 决定是否传入"""
        return self.client.chat.completions.create(
            model=self.model,
            **({"extra_body": {"thinking": {"type": "enabled" if self.thinking else "disabled"}}} if self.thinking is not None else {}),
            messages=messages,
            stream=stream
        )

    def _commit(self, user_input, content):
        """有正文时才把本轮问答写入历史，两种模式共用"""
        if content:
            self.append_message('user', user_input)
            self.append_message('assistant', content)

    def get_response(self, user_input, stream=False):
        """
        调用 API 获取回复，有正文时再把本轮问答写入历史
        流式读取抛出异常时不写入，生成器提前关闭时保留已收到的正文

        参数：
        - user_input (str): 用户输入的消息
        - stream (bool): 是否启用流式输出，默认为 False

        返回：
        if stream=False:
            tuple: (reasoning_content, content)
            - reasoning_content (str|None): 推理过程，仅思考模式返回，非思考模式为 None
            - content (str): 模型的回复内容

        if stream=True:
            generator: 生成一系列 (reasoning_content, content) 元组
            - 对于推理过程: (reasoning_content, None)
            - 对于回复内容: (None, content)
            每个元组只携带一种非空文本，同一数据块可以产生两个元组
        """
        # 用户消息先只加入本次请求，两种模式都在这里发请求
        messages = self.messages + [{'role': 'user', 'content': user_input}]
        completion = self._create(messages, stream=stream)

        if stream:
            # 流式输出，返回生成器
            return self._process_stream(completion, user_input)

        content = completion.choices[0].message.content
        reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', None)
        self._commit(user_input, content)
        return reasoning_content, content

    def _process_stream(self, completion, user_input):
        """
        处理流式输出的生成器

        参数：
        - completion: 流式响应对象
        - user_input (str): 用户输入的消息，用于写入历史

        返回：
        generator: 生成器对象，每次返回 (reasoning_content, content) 元组
        - 当收到推理过程时: yield (reasoning_content, None)
        - 当收到回复内容时: yield (None, content)
        """
        content = ""  # 用于存储完整回复

        try:
            for chunk in completion:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is None:
                    continue
                reasoning = getattr(delta, 'reasoning_content', None)
                reply = delta.content

                # 先保留这一块已收到的正文，提前关闭时也不会漏掉
                if reply:
                    content += reply

                # 两个字段独立处理，同一块可以分别输出推理和正文
                if reasoning:
                    yield reasoning, None
                if reply:
                    yield None, reply

                # 处理完当前块中的文本，再判断是否结束
                if choice.finish_reason is not None:
                    break
            self._commit(user_input, content)  # 正常结束
        except GeneratorExit:
            self._commit(user_input, content)  # 调用方提前关闭，保留已收到的部分
            raise
        finally:
            completion.close()  # 释放连接，读取流出错时只走到这里，不写历史
```

> [!note]
>
> - `GeneratorExit` 继承自 `BaseException` 而不是 `Exception`，所以 `except Exception` 捕获不到它，这正是能用它区分“调用方关闭”和“读取出错”的原因。另外，捕获 `GeneratorExit` 之后不能再 `yield`，必须 `raise` 继续抛出或直接返回，否则 `close()` 会报 `RuntimeError: generator ignored GeneratorExit`。
> - `append_message` 目前只保存 `role` 和 `content`，对纯对话来说已经足够。但随着工具调用的兴起和发展，assistant 消息还需要保留 `reasoning_content` 和 `tool_calls` 字段，这也是后续要解决的问题。

#### 使用示例

- **非流式输出**

  ```python
  user_input = "请用一句话介绍你自己"
  stream = False  # 非流式输出
  
  config = {
      "api_key": "your-api-key",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-flash",
      "thinking": False,  # 关闭思考
      "system_message": "You are a helpful assistant."
  }
  session = ChatSession(**config)
  
  # 获取回复
  reasoning, reply = session.get_response(user_input, stream=stream)
  if reasoning:
      print(f"===== 推理过程 =====\n{reasoning}\n")
  print(f"===== 模型回复 =====\nAI: {reply}\n")
  ```

  **输出**：

  ```
  ===== 模型回复 =====
  AI: 我是一个由人工智能驱动的语言助手，能够帮你解答问题、提供信息、协助创作并完成各类文本任务。
  ```

- **流式输出**

  ```python
  from contextlib import closing

  user_input = "请用一句话介绍你自己"
  stream = True  # 流式输出
  
  config = {
      "api_key": "your-api-key",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-flash",
      "thinking": True,  # 开启思考以观察推理过程
      "system_message": "You are a helpful assistant."
  }
  session = ChatSession(**config)
  
  # 实时打印模型回复的增量内容，print() 设置 end='' 实现不换行输出
  with closing(session.get_response(user_input, stream=stream)) as chunks:
      for reasoning, reply in chunks:
          if reasoning:
              print(reasoning, end='')
          else:
              print(reply, end='')
  ```

  **输出**：

  ```
  我们需要回答中文一句话介绍自己。需要简洁。作为助手。不要过度。一句话。可以：我是DeepSeek，由深度求索公司创造的AI助手，能帮你回答问题、处理信息和完成多种任务。这算一句话。注意用户说“请用一句话介绍你自己”。直接一句话即可。我是 DeepSeek，由深度求索公司创造的 AI 助手，可以帮你解答问题、处理信息并完成多种任务。
  ```

  `closing()` 会在退出 `with` 时调用生成器的 `close()`。即使循环提前 `break`，也能在离开 `with` 时完成清理。

## 📝 作业

1. 增加获取 TOKEN 消耗明细的函数。
