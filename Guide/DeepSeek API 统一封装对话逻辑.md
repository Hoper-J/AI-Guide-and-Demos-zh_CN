# DeepSeek API 统一封装对话逻辑

在[上一篇文章](./DeepSeek%20API%20多轮对话%20-%20OpenAI%20SDK.md)中，我们分别使用了 `ChatSession` 和 `ReasonerSession` 两个类来处理聊天模型（`DeepSeek-Chat`）和推理模型（`DeepSeek-Reasoner`）的对话逻辑。回顾它们的 API 返回结果[^1]：

- **DeepSeek-Chat**

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

- **DeepSeek-Reasoner**

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

[^1]: [DeepSeek API 输出解析 - OpenAI SDK](./DeepSeek%20API%20输出解析%20-%20OpenAI%20SDK.md#deepseek-reasoner).

可以观察到：推理模型（`DeepSeek-Reasoner`）的 `message` 部分比聊天模型（`DeepSeek-Chat`）**仅**多了一个 `reasoning_content` 字段，用于记录模型的推理思考过程。

**那么，能不能统一使用 `ChatSession` 类来兼容两种模型的对话逻辑呢？**

当然可以，只需要额外取一下 `reasoning_content` 字段：

```python
from openai import OpenAI
import os

class ChatSession:
    def __init__(self, client, model="deepseek-chat", system_message="You are a helpful assistant."):
        """
        参数：
        - client (openai.OpenAI): OpenAI 客户端实例
        - model (str): 模型名称（如 'deepseek-chat' 或 'deepseek-reasoner'），默认为 'deepseek-chat'
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
        """
        self.client = client
        self.model = model
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
        参数：
        - user_input (str): 用户输入的消息
        返回：
        - (reasoning_content, content) (tuple): 推理过程（仅推理模型有）和回复内容
        """
        # 记录用户输入
        self.append_message('user', user_input)
        
        # 调用 API
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        
        # 获取回复内容和推理过程（如果有）
        content = completion.choices[0].message.content
        reasoning_content = getattr(completion.choices[0].message, 'reasoning_content', None)
        
        # 记录模型回复
        self.append_message('assistant', content)
        
        return reasoning_content, content
```

### 进一步

如果不想每次手动创建 `client`，可以让 `ChatSession` 在初始化时直接完成 `OpenAI` 客户端的实例化，只需要传入 `api_key` 和 `base_url` 参数：

```python
from openai import OpenAI
import os

class ChatSession:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat", system_message="You are a helpful assistant."):
        """
        参数：
        - api_key (str): 平台的 API Key，默认从环境变量 `DEEPSEEK_API_KEY` 读取
        - base_url (str): API 请求地址，默认为 DeepSeek 官方平台
        - model (str): 模型名称（如 'deepseek-chat' 或 'deepseek-reasoner'），默认为 'deepseek-chat'
        - system_message (str): 系统消息，用于设定对话背景，默认为 'You are a helpful assistant.'
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
        返回：
        - (reasoning_content, content) (tuple): 推理过程（仅推理模型有）和回复内容
        """
        # 记录用户输入
        self.append_message('user', user_input)
        
        # 调用 API
        completion = self.client.chat.completions.create(
            model=self.model,
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
    "model": "deepseek-chat",  # 可以修改为推理模型，比如 "deepseek-reasoner"
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
>        model="deepseek-chat",
>        system_message="You are a helpful assistant."
> )
> ```
>

**对话**：

```python
user_input = "Hello"
reasoning, reply = session.get_response(user_input)

# 聊天模型（deepseek-chat）的 reasoning 为 None
if reasoning:
    print(f"===== 推理过程 =====\n{reasoning}\n")
print(f"===== 模型回复 =====\nAI: {reply}\n")
```