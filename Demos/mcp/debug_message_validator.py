#!/usr/bin/env python3
"""
message_validator 调试工具，可以保存为 debug_message_validator.py 执行

演示 FastMCP 中 message_validator.validate_python 的实际作用，
展示如何将字典转换为 Message 对象，以及 Pydantic Union 类型的选择行为。
"""

from typing import Any, Literal
from pydantic import BaseModel, TypeAdapter
from mcp.types import ContentBlock, TextContent


class Message(BaseModel):
    """基础消息类 - MCP 协议中所有消息的基类"""
    role: Literal["user", "assistant"]
    content: ContentBlock

    def __init__(self, content: str | ContentBlock, **kwargs: Any):
        # 如果内容是字符串，自动包装为 TextContent
        if isinstance(content, str):
            content = TextContent(type="text", text=content)
        super().__init__(content=content, **kwargs)


class UserMessage(Message):
    """
    来自用户的消息

    注意: role 字段允许 "user" 或 "assistant"，默认为 "user"
    """
    role: Literal["user", "assistant"] = "user"

    def __init__(self, content: str | ContentBlock, **kwargs: Any):
        super().__init__(content=content, **kwargs)


class AssistantMessage(Message):
    """
    来自助手的消息

    注意: role 字段允许 "user" 或 "assistant"，默认为 "assistant"
    """
    role: Literal["user", "assistant"] = "assistant"

    def __init__(self, content: str | ContentBlock, **kwargs: Any):
        super().__init__(content=content, **kwargs)


# FastMCP 中的 message_validator 定义
# TypeAdapter 用于验证和转换数据为 Union 类型
message_validator = TypeAdapter[UserMessage | AssistantMessage](UserMessage | AssistantMessage)


def demo_message_validator():
    """
    调试版本的 message_validator 演示
    展示 FastMCP 中字典如何转换为 Message 对象
    """
    print(f"\n{'='*60}")
    print(f"🔍 开始调试 message_validator")
    print(f"   展示字典 → Message 对象的转换过程")
    print(f"{'='*60}")

    # 步骤1: 分析类型定义
    print(f"\n📋 类型定义分析:")

    # 正确获取 Pydantic 字段默认值
    user_role_field = UserMessage.model_fields.get('role')
    assistant_role_field = AssistantMessage.model_fields.get('role')

    user_default = user_role_field.default if user_role_field else "无字段"
    assistant_default = assistant_role_field.default if assistant_role_field else "无字段"

    print(f"   UserMessage 默认 role: {user_default}")
    print(f"   AssistantMessage 默认 role: {assistant_default}")
    print(f"   Union 类型顺序: UserMessage | AssistantMessage")

    # 验证实际行为
    print(f"\n🔧 实例化验证:")
    user_instance = UserMessage(content="测试")
    assistant_instance = AssistantMessage(content="测试")
    print(f"   UserMessage() 实际 role: {user_instance.role}")
    print(f"   AssistantMessage() 实际 role: {assistant_instance.role}")

    # 准备测试用例
    test_cases = [
        {
            "name": "用户消息字典",
            "data": {
                "role": "user",
                "content": "简单的文本消息"
            }
        },
        {
            "name": "助手消息字典",
            "data": {
                "role": "assistant",
                "content": "我是助手的回复"
            }
        }
    ]

    print(f"\n🔧 转换测试详情:")
    print(f"   测试数量: {len(test_cases)}")

    # 步骤2: 执行转换测试
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n   [{idx}] 测试名称: {test_case['name']}")
        print(f"        输入数据: {test_case['data']}")
        print(f"        字段分析:")
        print(f"            role = '{test_case['data']['role']}'")
        print(f"            content = '{test_case['data']['content']}'")

        try:
            # 调用 message_validator.validate_python
            result = message_validator.validate_python(test_case['data'])

            print(f"        ✅ 转换成功!")
            print(f"        🔄 转换结果:")
            print(f"            类型: {type(result).__name__}")
            print(f"            角色: {result.role}")
            print(f"            内容类型: {type(result.content).__name__}")

            if hasattr(result.content, 'text'):
                print(f"            内容文本: {result.content.text}")

            # 分析异常情况
            if test_case['data']['role'] == 'assistant' and isinstance(result, UserMessage):
                print(f"        ⚠️  异常发现: role='assistant' 但返回了 UserMessage")
                print(f"            -> 原因: Pydantic Union 按顺序验证")
                print(f"            -> UserMessage 也接受 role='assistant'")
                print(f"            -> 第一个成功验证的类型被选择")

        except Exception as e:
            print(f"        ❌ 转换失败:")
            print(f"            错误类型: {type(e).__name__}")
            print(f"            错误信息: {e}")

    # 步骤3: 测试错误处理
    print(f"\n📊 错误处理测试:")
    print(f"   测试 message_validator 的错误处理能力")

    # 准备错误测试用例
    error_cases = [
        {
            "name": "缺少 role 字段",
            "data": {
                "content": "没有角色信息"
            },
            "expected": "应该使用默认角色或报错"
        },
        {
            "name": "错误的 role 值",
            "data": {
                "role": "system",  # 不支持的角色
                "content": "系统消息"
            },
            "expected": "应该验证失败"
        },
        {
            "name": "缺少 content 字段",
            "data": {
                "role": "user"
            },
            "expected": "必需字段缺失"
        }
    ]

    for idx, test_case in enumerate(error_cases, 1):
        print(f"\n   [{idx}] 错误场景: {test_case['name']}")
        print(f"        输入数据: {test_case['data']}")
        print(f"        预期行为: {test_case['expected']}")

        try:
            result = message_validator.validate_python(test_case['data'])
            print(f"        ✅ 意外成功!")
            print(f"            结果: {result}")
            print(f"            类型: {type(result).__name__}")
        except Exception as e:
            print(f"        ❌ 预期的错误:")
            print(f"            错误类型: {type(e).__name__}")
            print(f"            错误信息: {str(e)}")


def demo_prompt_render_simulation():
    """
    模拟 Prompt.render() 中的消息转换过程
    展示用户函数返回值如何被处理成标准 MCP 消息
    """
    print(f"\n{'='*60}")
    print(f"🔄 模拟 Prompt.render() 消息转换")
    print(f"   展示用户函数返回值 → 标准 MCP 消息的过程")
    print(f"{'='*60}")

    # 准备用户函数可能返回的各种类型
    user_returns = [
        "简单字符串",
        {
            "role": "user",
            "content": "字典格式的用户消息"
        },
        {
            "role": "assistant",
            "content": "字典格式的助手消息"
        },
        UserMessage(content="直接的 UserMessage 对象"),
        AssistantMessage(content="直接的 AssistantMessage 对象"),
        ["多个", "字符串"],
        [
            "混合类型",
            {"role": "user", "content": "字典消息"},
            AssistantMessage(content="对象消息")
        ]
    ]

    print(f"\n📋 模拟场景分析:")
    print(f"   返回类型数量: {len(user_returns)}")
    print(f"   覆盖场景: 字符串、字典、对象、列表、混合类型")

    def simulate_render_conversion(result):
        """
        模拟 render() 方法中的转换逻辑
        按照 FastMCP 的实际处理顺序进行转换
        """
        print(f"        🔄 开始转换处理:")
        print(f"            原始类型: {type(result).__name__}")

        # 步骤1: 规范化为列表
        if not isinstance(result, list | tuple):
            result = [result]
            print(f"            -> 单项转为列表: [1项]")
        else:
            print(f"            -> 已是列表: [{len(result)}项]")

        # 步骤2: 逐项转换为消息
        messages = []
        for idx, msg in enumerate(result, 1):
            print(f"            项目{idx}: {type(msg).__name__}")

            try:
                if isinstance(msg, Message):
                    # Message 对象直接使用
                    messages.append(msg)
                    print(f"                ✅ 直接使用: {type(msg).__name__}({msg.role})")
                    print(f"                   内容: {str(msg.content)}")

                elif isinstance(msg, dict):
                    # 字典通过 message_validator 转换
                    converted = message_validator.validate_python(msg)
                    messages.append(converted)
                    print(f"                🔄 字典转换: {msg}")
                    print(f"                   结果: {type(converted).__name__}({converted.role})")

                elif isinstance(msg, str):
                    # 字符串包装为 UserMessage
                    content = TextContent(type="text", text=msg)
                    user_msg = UserMessage(content=content)
                    messages.append(user_msg)
                    print(f"                📝 字符串转换: '{msg}'")
                    print(f"                   结果: UserMessage(user)")

                else:
                    # 其他类型序列化为 JSON
                    import json
                    content_str = json.dumps(msg, ensure_ascii=False, indent=2)
                    user_msg = UserMessage(content=content_str)
                    messages.append(user_msg)
                    print(f"                📦 JSON转换: {type(msg).__name__}")
                    print(f"                   结果: UserMessage(user)")

            except Exception as e:
                print(f"                ❌ 转换失败: {str(e)}")

        return messages

    # 步骤3: 执行场景测试
    print(f"\n🔧 场景测试详情:")

    for idx, user_return in enumerate(user_returns, 1):
        print(f"\n   [{idx}] 场景名称: 用户返回 {type(user_return).__name__}")
        print(f"        原始数据: {str(user_return)}")

        messages = simulate_render_conversion(user_return)

        print(f"        📊 转换总结:")
        print(f"            生成消息数: {len(messages)}")

        for msg_idx, msg in enumerate(messages, 1):
            print(f"            消息{msg_idx}: {type(msg).__name__}({msg.role})")
            print(f"                     内容: {str(msg.content)}")


def debug_pydantic_union_behavior():
    """
    深入分析 Pydantic Union 类型选择行为
    解释为什么 role='assistant' 时返回 UserMessage 而不是 AssistantMessage
    """
    print(f"\n{'='*60}")
    print(f"🔍 深入分析 Pydantic Union 类型选择")
    print(f"   解释 Union 类型的验证顺序和选择逻辑")
    print(f"{'='*60}")

    # 准备测试数据
    test_data = {
        "role": "assistant",
        "content": "助手消息"
    }

    print(f"\n📋 测试数据分析:")
    print(f"   输入数据: {test_data}")
    print(f"   预期类型: AssistantMessage（因为 role='assistant'）")

    print(f"\n🔧 验证步骤:")

    # 步骤1: 直接构造 UserMessage
    print(f"\n   [1] 直接构造 UserMessage:")
    try:
        user_msg = UserMessage(**test_data)
        print(f"        ✅ 构造成功!")
        print(f"            结果类型: {type(user_msg).__name__}")
        print(f"            角色字段: {user_msg.role}")
        print(f"            ℹ️  说明: UserMessage 接受 role='assistant'")
    except Exception as e:
        print(f"        ❌ 构造失败: {e}")

    # 步骤2: 直接构造 AssistantMessage
    print(f"\n   [2] 直接构造 AssistantMessage:")
    try:
        assistant_msg = AssistantMessage(**test_data)
        print(f"        ✅ 构造成功!")
        print(f"            结果类型: {type(assistant_msg).__name__}")
        print(f"            角色字段: {assistant_msg.role}")
        print(f"            ℹ️  说明: AssistantMessage 也接受 role='assistant'")
    except Exception as e:
        print(f"        ❌ 构造失败: {e}")

    # 步骤3: TypeAdapter 选择
    print(f"\n   [3] TypeAdapter Union 选择:")
    try:
        adapter_result = message_validator.validate_python(test_data)
        print(f"        📊 最终结果:")
        print(f"            选择类型: {type(adapter_result).__name__}")
        print(f"            角色字段: {adapter_result.role}")
    except Exception as e:
        print(f"        ❌ TypeAdapter 转换失败: {e}")

    print(f"\n✨ 结论总结:")
    print(f"   📝 核心原理: Pydantic Union 按顺序验证")
    print(f"       1. Union[UserMessage, AssistantMessage] 先验证 UserMessage")
    print(f"       2. UserMessage.role 允许 'user' | 'assistant'")
    print(f"       3. role='assistant' 通过 UserMessage 验证")
    print(f"       4. 验证成功，返回 UserMessage 实例")
    print(f"       5. 不再尝试 AssistantMessage")
    print(f"   🎯 实际影响:")
    print(f"       - role='assistant' 总是返回 UserMessage")
    print(f"       - 只有明确指定类型才能获得 AssistantMessage")
    print(f"       - 或许是 bug，但本来二者的定义就一样，只是类名不同，不再继续深究")



def test():
    """测试各种类型的 message_validator 行为"""
    print("\n\n📌 测试1: 基础转换行为")
    demo_message_validator()

    print("\n\n📌 测试2: Prompt.render() 模拟")
    demo_prompt_render_simulation()

    print("\n\n📌 测试3: Pydantic Union 选择分析")
    debug_pydantic_union_behavior()


if __name__ == "__main__":
    test()
