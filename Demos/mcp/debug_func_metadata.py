#!/usr/bin/env python3
"""
func_metadata 调试工具，可以保存为 debug_func_metadata.py 执行
"""

import inspect
import json
from typing import Any, Callable, Sequence, Optional, List, Dict
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from typing_extensions import Annotated

from mcp.server.fastmcp.utilities.func_metadata import (
    FuncMetadata, 
    ArgModelBase,
    _get_typed_signature,
    _get_typed_annotation,
    _try_create_model_and_schema,
    InvalidSignature,
    PydanticUndefined,
    WithJsonSchema
)


def print_json(data, title="JSON数据"):
    """打印JSON数据"""
    try:
        print(f"\n📄 {title}:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ JSON打印失败: {e}")
        print(f"原始数据类型: {type(data)}")
        print(f"原始数据: {data}")


def debug_func_metadata(
    func: Callable[..., Any],
    skip_names: Sequence[str] = (),
    structured_output: bool | None = None,
) -> Any:
    """
    调试版本的 func_metadata 实现，会用到一些 from_function 中的逻辑，比如：func.__name__，func.__doc__ ...
    """
    print(f"\n{'='*60}")
    print(f"🔍 开始解析函数: {func.__name__}")
    print(f"   文档字符串: {func.__doc__}")
    print(f"{'='*60}")

    try:
        # 从这里开始 func_metadata()
        # 步骤1: 获取函数签名
        sig = _get_typed_signature(func)
        params = sig.parameters

        print(f"\n📋 函数签名分析:")
        print(f"   完整签名: {sig}")
        print(f"   返回类型: {sig.return_annotation}")
        print(f"   参数数量: {len(params)}")

        # 准备构建动态 Pydantic 模型的参数字典
        dynamic_pydantic_model_params: dict[str, Any] = {}
        globalns = getattr(func, "__globals__", {})

        print(f"\n🔧 参数处理详情:")
        print(f"   跳过的参数名: {list(skip_names)}")

        # 步骤2: 遍历每个参数
        processed_count = 0
        for idx, param in enumerate(params.values()):
            print(f"\n   [{idx+1}] 参数名: {param.name}")
            print(f"        原始注解: {param.annotation}")
            print(f"        参数种类: {param.kind}")
            print(f"        默认值: {param.default}")

            # 验证参数名
            if param.name.startswith("_"):
                print(f"        ❌ 错误: 参数名不能以 '_' 开头")
                raise InvalidSignature(f"{func.__name__} 的参数 {param.name} 不能以 '_' 开头")

            if param.name in skip_names:
                print(f"        ⏭️  跳过此参数")
                continue

            processed_count += 1
            annotation = param.annotation

            # 处理 `x: None` 或 `x: None = None` 的情况
            if annotation is None:
                print(f"        📝 处理: 类型为 None，添加默认值字段")
                annotation = Annotated[
                    None,
                    Field(default=param.default if param.default is not inspect.Parameter.empty else PydanticUndefined),
                ]

            if annotation is inspect.Parameter.empty:
                print(f"        ⚠️  处理: 无类型注解，默认为 Any")
                annotation = Annotated[
                    Any,
                    Field(),
                    # 🤷 默认将无类型参数视为字符串
                    WithJsonSchema({"title": param.name, "type": "string"}),
                ]

            # 获取类型化注解
            typed_annotation = _get_typed_annotation(annotation, globalns)
            print(f"        🔄 类型化注解: {typed_annotation}")

            # 创建字段信息
            field_info = FieldInfo.from_annotated_attribute(
                typed_annotation,
                param.default if param.default is not inspect.Parameter.empty else PydanticUndefined,
            )

            print(f"        ✅ 字段信息: annotation={field_info.annotation}, default={field_info.default}")

            # 处理参数名与 BaseModel 内置方法冲突的情况，这是必要的，因为 Pydantic 会因此发出警告
            # 例如：'dict' 或 'json' 等
            if hasattr(BaseModel, param.name) and callable(getattr(BaseModel, param.name)):
                print(f"        ⚠️  冲突处理: 参数名 '{param.name}' 与 BaseModel 方法冲突")
                # 使用别名机制避免警告
                field_info.alias = param.name
                field_info.validation_alias = param.name
                field_info.serialization_alias = param.name
                # 内部使用带前缀的参数名
                internal_name = f"field_{param.name}"
                dynamic_pydantic_model_params[internal_name] = (field_info.annotation, field_info)
                print(f"            -> 使用内部名称: {internal_name}")
            else:
                dynamic_pydantic_model_params[param.name] = (field_info.annotation, field_info)

        print(f"\n📊 参数处理总结:")
        print(f"   总参数数: {len(params)}")
        print(f"   处理参数数: {processed_count}")
        print(f"   模型字段: {list(dynamic_pydantic_model_params.keys())}")

        # 步骤3: 动态创建一个 Pydantic 模型来表示函数参数
        arguments_model_name = f"{func.__name__}Arguments"
        print(f"\n🏗️  创建 Pydantic 模型:")
        print(f"   模型名称: {arguments_model_name}")
        print(f"   基类: {ArgModelBase}")

        arguments_model = create_model(
            arguments_model_name,
            **dynamic_pydantic_model_params,
            __base__=ArgModelBase,
        )

        print(f"   ✅ 模型创建成功: {arguments_model}")

        # 生成并打印 JSON Schema
        try:
            # 这部分对应于 func_metadata() 之后的那行代码，提前进行查看
            schema = arguments_model.model_json_schema(by_alias=True)
            print_json(schema, f"{arguments_model_name} JSON Schema")
        except Exception as e:
       [I     print(f"❌ Schema 生成失败: {e}")

        # 步骤4: 处理返回值（完全按照原版本逻辑）
        print(f"\n🎯 返回值处理:")
        print(f"   structured_output 参数: {structured_output}")
        print(f"   返回注解: {sig.return_annotation}")

        if structured_output is False:
            print(f"   🔚 明确不需要结构化输出")
            result = FuncMetadata(arg_model=arguments_model)
            print(f"   ✅ 返回元数据: {result}")
            return result

        # 基于返回类型注释设置结构化输出支持
        if sig.return_annotation is inspect.Parameter.empty and structured_output is True:
            print(f"   ❌ 错误: 要求结构化输出但无返回注解")
            raise InvalidSignature(f"函数 {func.__name__}: 结构化输出需要返回注释")

        output_info = FieldInfo.from_annotation(_get_typed_annotation(sig.return_annotation, globalns))
        annotation = output_info.annotation

        print(f"      经过_get_typed_annotation处理后的类型: {annotation}")

        output_model, output_schema, wrap_output = _try_create_model_and_schema(
            annotation, func.__name__, output_info
        )

        if output_model:
            print(f"      ✅ 输出模型创建成功: {output_model}")
            if output_schema:
                print_json(output_schema, "返回值 JSON Schema")
        else:
            print(f"      ℹ️  未创建输出模型")

        print(f"      wrap_output: {wrap_output}")

        # 模型创建失败或产生警告 - 无结构化输出
        if output_model is None and structured_output is True:
            print(f"      ❌ 结构化输出失败: 返回类型不可序列化")
            raise InvalidSignature(
                f"函数 {func.__name__}: 返回类型 {annotation} 不支持结构化输出"
            )

        # 创建最终结果
        result = FuncMetadata(
            arg_model=arguments_model,
            output_schema=output_schema,
            output_model=output_model,
            wrap_output=wrap_output,
        )

        print(f"\n✨ func_metadata 处理完成!")
        print(f"   最终结果: {result}")
        print(f"{'='*60}\n")

        return result

    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return None


def test():
    """测试各种类型的函数"""
    # 混合注解
    print("\n\n📌 测试1: 混合类型注解")
    def func(
        data,  # 无类型注解
        format: str = "json",  # 有注解+默认值
        count: Optional[int] = None,  # 复杂类型+默认值
        validate: bool = True  # 基础类型+默认值
    ):  # 无返回类型注解
        """展示各种注解情况"""
        return data

    debug_func_metadata(func, skip_names="count")

    # 前缀参数测试
    print("\n\n📌 测试2: 前缀参数冲突")
    def prefix_func(_private: str, field_test: int) -> str:
        """前缀参数"""
        return "test"
    debug_func_metadata(prefix_func)

    print("\n\n📌 测试3: 结构化输出对比")
    def add(a: int, b: int) -> str:
        return a + b
    print("📌 无结构化")
    debug_func_metadata(add, structured_output=False)
    print("\n\n📌 结构化")
    debug_func_metadata(add, structured_output=True)


if __name__ == "__main__":
    test()
