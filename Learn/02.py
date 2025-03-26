import gradio as gr
from chat import chat
import time
from typing import List, Tuple, Optional

# 全局变量存储对话历史
chat_history: List[dict] = []


def user(user_input: str, display_history: List[List[str]]) -> Tuple[str, str]:
    """
    处理用户输入
    Args:
        user_input: 用户输入的文本
        display_history: 当前的对话历史
    Returns:
        Tuple[str, str]: (清空输入框的值, 用户消息)
    """
    return "", user_input


def bot(user_message: str, display_history: List[List[str]]) -> List[List[str]]:
    """
    处理机器人响应
    Args:
        user_message: 用户消息
        display_history: 当前的对话历史
    Returns:
        List[List[str]]: 更新后的对话历史
    """
    global chat_history
    try:
        # 调用chat函数获取回复
        response, chat_history = chat(user_message, chat_history)
        return display_history + [[user_message, response]]
    except Exception as e:
        return display_history + [[user_message, f"发生错误: {str(e)}"]]


def clear_all() -> Optional[List[List[str]]]:
    """
    清除所有对话历史
    Returns:
        Optional[List[List[str]]]: 清空后的对话历史
    """
    global chat_history
    chat_history = []
    return None


# 创建Gradio界面
with gr.Blocks(css="footer {display: none !important}") as demo:
    # 标题
    title: gr.Markdown = gr.Markdown("# AI 对话助手")

    # 对话显示区域
    chat_display: gr.Chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(
            None, "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"),
        height=500
    )

    # 输入区域
    with gr.Row():
        # 文本输入框
        input_textbox: gr.Textbox = gr.Textbox(
            show_label=False,
            placeholder="请输入您的问题...",
            container=False
        )
        # 发送按钮
        send_button: gr.Button = gr.Button("发送", variant="primary")
        # 清除按钮
        clear_button: gr.Button = gr.Button("清除对话")

    # 绑定事件
    # 回车发送事件
    input_textbox.submit(
        user,
        [input_textbox, chat_display],
        [input_textbox, chat_display]
    ).then(
        bot,
        [chat_display, chat_display],
        chat_display
    )

    # 点击发送按钮事件
    send_button.click(
        user,
        [input_textbox, chat_display],
        [input_textbox, chat_display]
    ).then(
        bot,
        [chat_display, chat_display],
        chat_display
    )

    # 清除对话事件
    clear_button.click(
        clear_all,
        None,
        chat_display
    )

if __name__ == "__main__":
    demo.launch(share=True)
