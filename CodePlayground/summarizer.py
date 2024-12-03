# ===============================================================
# 这里是一个 Toy Summarizer，祝你玩的开心。
# 它会保留中间转换的文件，因为这或许对你有用。
# 如果遇到错误，欢迎通过 Issues 或 Discussions 提交反馈。为了更快解决问题，请尽可能附上运行环境和可复现的命令。

# 对应文章：《15. 用 API 实现 AI 视频摘要：动手制作属于你的 AI 视频小助手》
# https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/15.%20用%20API%20实现%20AI%20视频摘要：动手制作属于你的%20AI%20视频助手.md

# 使用方法：
#   python summarizer.py <file_path> [可选参数]
# 示例：
#   python summarizer.py ./examples/summarizer.mp4

# 查看完整帮助：使用 -h 或 --help
# ===============================================================

import os
import sys
import subprocess
import datetime
import argparse
import ssl
import urllib.error

import certifi
import librosa
import numpy as np
import srt
import whisper
import yaml

from openai import OpenAI
from utils.config_manager import load_config, save_config, get_api_key


def get_file_type(file_path):
    """
    根据文件扩展名判断文件类型。

    参数:
    - file_path (str): 文件的路径。

    返回:
    - str: 文件类型，返回 'video', 'audio', 'subtitle' 或 'unknown'。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        return 'video'
    elif ext in ['.mp3', '.wav', '.flac', '.aac', '.m4a']:
        return 'audio'
    elif ext in ['.srt', '.vtt', '.ass']:
        return 'subtitle'
    else:
        return 'unknown'

def video_to_audio(video_path, audio_path=None):
    """
    使用 ffmpeg 将视频文件转换为音频文件（WAV 格式）。

    参数:
    - video_path (str): 视频文件的路径。
    - audio_path (str): 保存音频文件的路径。如果未指定，将使用视频文件名并保存为同名的 WAV 文件。

    返回:
    - str: 转换后的音频文件路径。
    - None: 如果转换失败，返回 None。
    """
    try:
        # 如果未指定 audio_path，则使用视频文件名生成音频文件名
        if audio_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = f"{base_name}.wav"
        
        # 使用 ffmpeg 提取音频，保存为 WAV 格式
        command = f"ffmpeg -i \"{video_path}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{audio_path}\" -y"
        subprocess.run(command, shell=True, check=True)
        print(f"音频文件已生成：{audio_path}")
        return audio_path
    except Exception as e:
        print(f"视频转音频时出错: {e}")
        return None

def transcribe_audio(audio_path, subtitle_path, decode_options, model_name="medium"):
    """
    使用 Whisper 模型转录音频，并保存为字幕文件（SRT 格式）。

    参数:
    - audio_path (str): 音频文件的路径。
    - subtitle_path (str): 要保存的字幕文件的路径。
    - decode_options (dict): Whisper 模型的解码选项，如语言、初始提示、温度等。
    - model_name (str): Whisper 模型的名称（默认为 "medium"）。

    返回:
    - str: 转录的字幕文件路径。
    - None: 如果转录失败，返回 None。
    """
    try:
        # 设置 SSL 证书路径
        os.environ['SSL_CERT_FILE'] = certifi.where()

        # 加载 Whisper 模型
        model = whisper.load_model(model_name)
        
        # 转录音频，启用进度显示
        transcription = model.transcribe(
            audio_path, 
            language=decode_options["language"], 
            verbose=False,  # 启用进度条
            initial_prompt=decode_options["initial_prompt"],
            temperature=decode_options["whisper_temperature"]
        )
        
        # 创建字幕列表
        subtitles = []
        for i, segment in enumerate(transcription["segments"]):
            start_time = datetime.timedelta(seconds=segment["start"])
            end_time = datetime.timedelta(seconds=segment["end"])
            text = segment["text"].strip()
            subtitles.append(srt.Subtitle(index=i+1, start=start_time, end=end_time, content=text))
        
        # 生成 SRT 内容
        srt_content = srt.compose(subtitles)
        
        # 保存字幕文件
        with open(subtitle_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        print(f"字幕文件已保存：{subtitle_path}")
        return subtitle_path
    except urllib.error.URLError as e:
        print(f"网络错误: {e}. 可能是网络连接或代理问题，请检查网络设置。")
    except ssl.SSLError as e:
        print(f"SSL 错误: {e}. 可能是网络连接问题或证书问题，请检查网络设置和 SSL 证书。")
    except Exception as e:
        print("音频转录时出错")
        raise e
    return None

def format_timestamp(timestamp):
    """
    格式化时间戳，去掉小数点后部分。

    参数:
    - timestamp (datetime.timedelta): Whisper 模型生成的时间戳。

    返回:
    - str: 去掉小数点后部分的时间戳，格式为 'HH:MM:SS'。
    """
    # 去掉小数点后部分的时间
    formatted_timestamp = str(timestamp).split('.')[0]  # 只保留秒之前的部分
    return formatted_timestamp

def read_subtitle(file_path, timestamped=False):
    """
    读取字幕文件内容，并根据参数决定是否包含时间信息。
    
    参数:
    - file_path (str): 字幕文件的路径。
    - timestamped (bool): 如果为 True，返回包含时间戳的字幕文本，默认为 False。

    返回:
    - list 或 str: 如果 timestamped=True，返回带有时间信息的字幕列表；否则返回纯文本字符串。
    - None: 如果读取失败，返回 None。
    """
    try:
        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用 srt 解析字幕文件
        subtitles = list(srt.parse(content))

        if timestamped:
            # 返回包含格式化时间戳的字幕
            text_with_timestamps = [
                f"{format_timestamp(sub.start)} --> {format_timestamp(sub.end)}\n{sub.content}" 
                for sub in subtitles
            ]
            return text_with_timestamps
        else:
            # 返回纯文本字幕
            text = '\n'.join([sub.content for sub in subtitles])
            return text

    except FileNotFoundError as e:
        print(f"文件未找到: {e}. 请确认文件路径是否正确。")
    except UnicodeDecodeError:
        print(f"错误：无法解码文件 '{file_path}'，请检查文件编码格式。")
    except Exception as e:
        print(f"未知错误，请检查输入和配置。")
        raise e
    return None
    
def summarize_text(text, client, timestamped=False, model="qwen-vl-max-0809", llm_temperature=0.2, max_tokens=1000):
    """
    注意，我们没有对字幕文件做更多的处理，没有滑动窗口，没有文本清理，没有语义分割。
    只是使用 Prompt 指导模型生成摘要。即（美名其曰）：In-context Learning.

    参数:
    - text (str): 要生成摘要的字幕或文本内容。
    - client (OpenAI): OpenAI 客户端实例。
    - timestamped (bool): 是否基于时间戳生成摘要，默认为 False。
    - model (str): 使用的 OpenAI 模型名称，默认为 'qwen-vl-max-0809'。
    - llm_temperature (float): 生成摘要时的温度，默认为 0.2。
    - max_tokens (int): 生成摘要时的最大 token 数，默认为 1000。

    返回:
    - str: 生成的摘要文本。
    - None: 如果生成摘要失败，返回 None。
    """
    try:
        if timestamped:
            # 对含时间戳的文本生成摘要
            prompt = f"""
                我需要你根据以下字幕生成一个视频摘要。摘要格式应包括视频的总体总结和每个时间段的简短总结。请遵循以下格式（"[","]"包裹的是需要被对应替换的内容）：
                
                [整个视频的简短总结]
                1. [00:00:00]-[时间段1结束的时间戳]: 对应的摘要
                2. [时间段2开始的时间戳]-[时间段2结束的时间戳]: 对应的摘要
                3. ...
                
                请确保每个时间段的摘要是简明扼要的，且整个视频的总结应简洁并涵盖关键内容。
                
                以下是视频字幕内容：
                {text}
            """
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=llm_temperature,
                max_tokens=max_tokens
            )
            summary = response.choices[0].message.content.strip()
            print("摘要生成成功。")
            return summary
        else:
            # 对不含时间戳的文本生成摘要
            prompt = f"请为以下文本生成摘要：\n{text}"
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=llm_temperature,
                max_tokens=max_tokens
            )
            summary = response.choices[0].message.content.strip()
            print("摘要生成成功。")
            return summary
    except urllib.error.URLError as e:
        print(f"网络错误: {e}. 可能是网络连接或代理问题，请检查网络设置。")
    except ssl.SSLError as e:
        print(f"SSL 错误: {e}. 可能是网络连接问题或证书问题，请检查网络设置和 SSL 证书。")
    except AttributeError as e:
        print(f"属性错误: {e}. 请检查响应对象的结构。")
    except Exception as e:
        raise e
    return None

def process_file(file_path, client, output_dir=None, timestamped=False, model_name="medium", language="zh", whisper_temperature=0.2, llm_temperature=0.2, max_tokens=1000):
    """
    根据文件类型处理文件并生成摘要。处理包括视频文件转换为音频、音频转录为字幕等。

    参数:
    - file_path (str): 要处理的文件路径。
    - client (OpenAI): OpenAI 客户端实例。
    - output_dir (str): 保存生成文件的目录。如果为 None，保存到当前文件夹。
    - timestamped (bool): 如果为 True，生成包含时间戳的摘要，默认为 False。
    - model_name (str): Whisper 模型的名称，默认为 "medium"。
    - language (str): Whisper 模型的转录语言，默认为 "zh"。
    - whisper_temperature (float): 音频转字幕时的温度，默认为 0.2。
    - llm_temperature (float): 生成摘要时的温度，默认为 0.2。
    - max_tokens (int): 生成摘要时的最大 token 数，默认为 1000。

    返回:
    - str: 生成的摘要文本或错误消息。
    """
    if output_dir is None:
        output_dir = os.getcwd()  # 默认当前目录

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    file_type = get_file_type(file_path)
    print(f"检测到的文件类型：{file_type}")

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    subtitle_path = os.path.join(output_dir, f"{base_name}.srt")
    summary_path = os.path.join(output_dir, f"{base_name}.summary.txt")

    if file_type in ['video', 'audio']:
        if os.path.exists(subtitle_path):
            # 对于视频和音频文件，检查是否已有字幕文件
            print(f"检测到现有字幕文件：{subtitle_path}，跳过转录步骤。")
            text = read_subtitle(subtitle_path, timestamped=timestamped)
        else:
            if file_type == 'video':
                # 视频文件：提取音频，自动生成与视频同名的音频文件
                audio_file = video_to_audio(file_path, os.path.join(output_dir, f"{base_name}.wav"))
                if audio_file is None:
                    return "音频提取失败"
            else:
                # 音频文件：直接使用
                audio_file = file_path

            # 转录音频并保存字幕文件
            subtitle_file = transcribe_audio(audio_file, subtitle_path, decode_options={
                "language": language,
                "initial_prompt": "",
                "whisper_temperature": whisper_temperature
            }, model_name=model_name)
            if subtitle_file is None:
                return "音频转录失败"

            # 从字幕文件读取文本
            text = read_subtitle(subtitle_path, timestamped=timestamped)
            
    elif file_type == 'subtitle':
        # 字幕文件：读取内容
        text = read_subtitle(file_path, timestamped=timestamped)
    else:
        return "不支持的文件格式"

    if text is None:
        return "文本提取失败"

    # 生成摘要
    summary = summarize_text(text, client, timestamped=timestamped, llm_temperature=llm_temperature, max_tokens=max_tokens)
    if summary is None:
        return "摘要生成失败"

    # 保存摘要到文件
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"摘要文件已保存：{summary_path}")
    except Exception as e:
        print(f"保存摘要文件时出错")
        raise e

    return summary

def main():
    # 加载默认配置
    config = load_config(script_name='summarizer')
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(
        description="视频/音频/字幕文件生成摘要工具。该工具可以处理视频、音频文件，将它们转换为字幕，并生成摘要。",
        epilog="示例: python summarizer.py ./examples/summarizer.mp4"
    )
    parser.add_argument("file_path", type=str, help="要处理的文件路径")
    parser.add_argument("--api_key", type=str, help="OpenAI API 密钥")
    parser.add_argument("--output_dir", type=str, default=config.get('output_dir', './output'), help="生成文件的保存目录")
    parser.add_argument("--model_name", type=str, choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                        default=config.get('model_name', 'medium'), help="指定使用的 Whisper 模型名称")
    parser.add_argument("--language", type=str, default=config.get('language', 'zh'), help="音频转录的语言")
    parser.add_argument("--whisper_temperature", type=float, default=config.get('whisper_temperature', 0.2), help="Whisper 模型的温度")
    parser.add_argument("--llm_temperature", type=float, default=config.get('llm_temperature', 0.2), help="大语言模型的温度")
    parser.add_argument("--timestamped", action="store_true", default=config.get('timestamped', False), help="保留转录的时间戳")
    parser.add_argument("--max_tokens", type=int, default=config.get('max_tokens', 1000), help="生成摘要时的最大 token 数")

    try:
        args = parser.parse_args()
    except SystemExit:
        print("\n错误: 必须提供 'file_path' 参数。请指定要处理的视频、音频或字幕文件的路径。\n")
        print("使用 -h 或 --help 以获取命令行参数的详细说明。\n")
        sys.exit(1)

    # 获取 API 密钥并验证：从命令行或配置文件中获取
    api_key = args.api_key if args.api_key else get_api_key(config)
    api_base_url = config.get('api_base_url', "https://dashscope.aliyuncs.com/compatible-mode/v1")
    # 构建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url
    )

    # 打印配置
    print("\n========== 当前使用的配置 ==========\n")
    
    # 文件与路径相关配置
    print(">> 文件与路径相关配置 <<")
    print(f"  - 文件路径: {args.file_path if args.file_path else '未指定'}")
    print(f"  - 输出目录: {args.output_dir if args.output_dir else '未指定'}")
    
    # API 配置
    print("\n>> OpenAI 相关配置 <<")
    print(f"  - API Key: {'已提供' if api_key else '未提供'}")
    print(f"  - API Base URL: {api_base_url if api_base_url else '使用默认URL'}")
    
    # 模型相关配置
    print("\n>> Whisper 相关配置 <<")
    print(f"  - Whisper 模型: {args.model_name if args.model_name else '使用默认模型'}")
    print(f"  - 语言: {args.language if args.language else '未指定语言'}")
    print(f"  - Whisper Temperature: {args.whisper_temperature}")
    
    print("\n>> 大语言模型 相关配置 <<")
    print(f"  - LLM Temperature: {args.llm_temperature}")
    
    # 其他配置
    print("\n>> 其他配置 <<")
    print(f"  - 是否保留时间戳: {'是' if args.timestamped else '否'}")
    
    print("\n=====================================\n")
    
    # 生成摘要
    summary = process_file(
        file_path=args.file_path,
        client=client,
        output_dir=args.output_dir,
        timestamped=args.timestamped,
        model_name=args.model_name,
        language=args.language,
        whisper_temperature=args.whisper_temperature,
        llm_temperature=args.llm_temperature,
        max_tokens=args.max_tokens
    )
    print(f"\n生成的摘要：\n{summary}")

main()
