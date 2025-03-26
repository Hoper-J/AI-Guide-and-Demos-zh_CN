import openai

openai.api_base = "https://bljj.org/v1"
openai.api_key = 'sk-RNEGpMprccHAcc7RcHF1gaDWBJcP0r41PfpXw6BbkbKp44H9'


def chat(message: str, history: list[dict], model="gemini-2.0-flash-exp", temperature=0.7, max_tokens=10 * 1000):
    # 构建完整的消息历史
    messages = history + [{'role': 'user', 'content': message}]

    # 创建流式响应
    response = openai.Completion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature
    )

    # 用于累积完整的响应
    full_response = ""

    # 实时打印模型回复的增量内容
    for chunk in response:
        delta = chunk.choices[0].delta
        if 'content' in delta:
            content = delta.content
            print(content, end='', flush=True)
            full_response += content

    # 将完整的响应添加到历史记录中
    history.append({'role': 'user', 'content': message})
    history.append({'role': 'assistant', 'content': full_response})
    return full_response, history
