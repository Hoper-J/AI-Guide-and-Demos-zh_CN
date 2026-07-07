# DeepSeek API 统一模型对话逻辑与流式输出

> **代码文件下载**：[Code](../Demos/deepseek-api-guide-5.ipynb)
>
> **在线链接**：[Kaggle](https://www.kaggle.com/code/aidemos/deepseek-api-guide-5) | [Colab](https://colab.research.google.com/drive/14u47q-lGfH7l1ehkBuTU0kgIsNarap9J?usp=sharing)

## 目录

- [进一步](#进一步)
  - [使用示例](#使用示例)
- [流式输出](#流式输出)
  - [核心逻辑](#核心逻辑)
  - [引入生成器](#引入生成器)
  - [代码实现](#代码实现)
    - [使用示例](#使用示例-1)
- [📝 作业](#-作业)

在[上一篇文章](./DeepSeek%20API%20多轮对话%20-%20OpenAI%20SDK.md)中，我们分别使用了 `ChatSession` 和 `ReasonerSession` 两个类来处理思考与非思考两种模式的对话逻辑（V3/R1 时代它们对应 `DeepSeek-Chat` 和 `DeepSeek-Reasoner` 两个独立模型）。回顾它们的 API 返回结果[^1]：

- **非思考模式（DeepSeek-Chat）**

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

- **思考模式（DeepSeek-Reasoner）**

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

**那么，能不能统一使用 `ChatSession` 类来兼容两种模式的对话逻辑呢？**

当然可以，只需要额外取一下 `reasoning_content` 字段：

```python
from openai import OpenAI
import os

class ChatSession:
    def __init__(self, client, model="deepseek-v4-flash", system_message="You are a helpful assistant.", thinking=None):
        """
        参数:
        - client (openai.OpenAI): OpenAI 客户端实例
        - model (str): 模型名称，默认为 'deepseek-v4-flash'（更强的 'deepseek-v4-pro' 用法相同）
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
        - thinking (bool|None): 思考模式开关，None 使用平台默认，True/False 显式开/关
        """
        self.client = client
        self.model = model
        self.thinking = thinking  # None=平台默认；True/False=显式开/关思考（V4 生效）
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
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-v4-flash", system_message="You are a helpful assistant.", thinking=None):
        """
        参数:
        - api_key (str): 平台的 API Key，默认从环境变量 `DEEPSEEK_API_KEY` 读取
        - base_url (str): API 请求地址，默认为 DeepSeek 官方平台
        - model (str): 模型名称，默认为 'deepseek-v4-flash'（更强的 'deepseek-v4-pro' 用法相同）
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
        - thinking (bool|None): 思考模式开关，None 使用平台默认，True/False 显式开/关
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
        self.thinking = thinking  # None=平台默认；True/False=显式开/关思考（V4 生效）
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
    "model": "deepseek-v4-flash",
    "thinking": False,  # 关闭思考；True 开启，None 使用平台默认
    "system_message": "You are a helpful assistant."
}

session = ChatSession(**config)
```

> [!note]
>
> `**` 是 Python 中的解包操作符，它将字典中的键值对解包为函数的关键字参数。在这里，`**config` 将字典中的参数逐一传递给 `ChatSession()` ， `ChatSession(**config)`等价于：
>
> ```python
> ChatSession(
>        api_key="your-api-key",
>        base_url="https://api.deepseek.com",
>        model="deepseek-v4-flash",
>        thinking=False,  # 关闭思考（None 则用平台默认）
>        system_message="You are a helpful assistant."
> )
> ```
>

**对话**：

```python
user_input = "Hello"
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

1. 在同一个数据块中，只会有其中一种类型的内容不为 `None`。
2. 需要记录完整的回复内容，因为这将被添加到对话历史中。
3. 当 `finish_reason` 不为 `None` 时代表生成完毕。

所以核心的处理逻辑如下：

```python
def _process_stream(self, completion):
    content = ""

    for chunk in completion:
        delta = chunk.choices[0].delta
        # 处理推理过程（仅思考模式有）
        if getattr(delta, 'reasoning_content', None):
            print(delta.reasoning_content, end='')
        # 处理回复内容
        elif delta.content:
            content += delta.content  # 需要记录 content 维护对话历史
            print(delta.content, end='')

        # 如果是最后一个数据块（finish_reason 不为 None）
        if chunk.choices[0].finish_reason is not None:
            # 记录完整的模型回复 content
            self.append_message('assistant', content)
            break
```

### 引入生成器

为了让流式输出的使用更加灵活，而非固定的打印输出，我们可以将其实现为一个生成器（generator），并采用和非流式输出一致的返回格式：

```python
def _process_stream(self, completion):
    content = ""  # 用于存储完整回复

    for chunk in completion:
        delta = chunk.choices[0].delta
        # 处理推理过程（仅思考模式有）
        if getattr(delta, 'reasoning_content', None):
            yield delta.reasoning_content, None
        # 处理回复内容
        elif delta.content:
            content += delta.content  # 需要记录 content 维护对话历史
            yield None, delta.content

        # 如果是最后一个数据块（finish_reason 不为 None）
        if chunk.choices[0].finish_reason is not None:
            # 记录完整的模型回复 content
            self.append_message('assistant', content)
            break
```

> [!tip]
>
> 这里的 `yield` 语句返回一个生成器对象，可以直接使用 `for` 循环处理（见下文示例）。

### 代码实现

```python
class ChatSession:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-v4-flash", system_message="You are a helpful assistant.", thinking=None):
        """
        参数：
        - api_key (str): 平台的 API Key，默认从环境变量 `DEEPSEEK_API_KEY` 读取
        - base_url (str): API 请求地址，默认为 DeepSeek 官方平台
        - model (str): 模型名称，默认为 'deepseek-v4-flash'（更强的 'deepseek-v4-pro' 用法相同）
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
        - thinking (bool|None): 思考模式开关，None 使用平台默认，True/False 显式开/关
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
        self.thinking = thinking  # None=平台默认；True/False=显式开/关思考（V4 生效）
        self.messages = [{'role': 'system', 'content': system_message}]

    def append_message(self, role, content):
        """
        添加一条对话消息

        参数:
        - role (str): 消息角色，为 'user' 或 'assistant'
        - content (str): 消息内容
        """
        self.messages.append({'role': role, 'content': content})

    def get_response(self, user_input, stream=False):
        """
        添加用户消息，调用 API 获取回复，并返回推理过程和回复内容

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
            其中必定有一个值为 None，另一个包含当前数据块的实际内容
        """
        # 记录用户输入
        self.append_message('user', user_input)
        
        # 调用 API
        completion = self.client.chat.completions.create(
            model=self.model,
            **({"extra_body": {"thinking": {"type": "enabled" if self.thinking else "disabled"}}} if self.thinking is not None else {}),
            messages=self.messages,
            stream=stream
        )
        
        if not stream:
            # 非流式输出
            content = completion.choices[0].message.content
            reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', None)
            
            # 记录模型回复
            self.append_message('assistant', content)
            
            return reasoning_content, content
        else:
            # 流式输出，返回生成器
            return self._process_stream(completion)
    
    def _process_stream(self, completion):
        """
        处理流式输出的数据块

        参数：
        - completion: API 返回的流式输出对象

        返回：
        generator: 生成器对象，每次返回 (reasoning_content, content) 元组
        - 当收到推理过程时: yield (reasoning_content, None)
        - 当收到回复内容时: yield (None, content)
        """
        content = ""  # 用于存储完整回复
        
        for chunk in completion:
            delta = chunk.choices[0].delta
            # 处理推理过程（仅思考模式有）
            if getattr(delta, 'reasoning_content', None):
                yield delta.reasoning_content, None
            # 处理回复内容
            elif delta.content:
                content += delta.content  # 需要记录 content 维护对话历史
                yield None, delta.content
                
            # 如果是最后一个数据块（finish_reason 不为 None）
            if chunk.choices[0].finish_reason is not None:
                # 记录完整的模型回复 content
                self.append_message('assistant', content)
                break
```

#### 使用示例

- **非流式输出**

  ```python
  user_input = "Hello"
  stream = False  # 非流式输出
  
  config = {
      "api_key": "your-api-key",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-v4-flash",
      "thinking": False,  # 关闭思考；True 开启，None 使用平台默认
      "system_message": "You are a helpful assistant."
  }
  session = ChatSession(**config)
  
  # 获取回复
  reasoning, reply = session.get_response(user_input, stream=stream)
  if reasoning:
      print(f"===== 推理过程 =====\n{reasoning}\n")
  print(f"===== 模型回复 =====\nAI: {reply}\n")
  ```

- **流式输出**

  ```python
  user_input = "Hello"
  stream = True  # 流式输出
  
  config = {
      "api_key": "your-api-key",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-v4-flash",
      "thinking": True,  # 开启思考以观察推理过程
      "system_message": "You are a helpful assistant."
  }
  session = ChatSession(**config)
  
  # 实时打印模型回复的增量内容，print() 设置 end=''实现不换行输出
  for reasoning, reply in session.get_response(user_input, stream=stream):
      if reasoning:
          print(reasoning, end='')  
      else:
          print(reply, end='')
  ```

## 📝 作业

1. 增加获取 TOKEN 消耗明细的函数。
