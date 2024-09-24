import os
import yaml


def load_config(config_file='config.yaml', script_name='summarizer') -> dict:
    """
    从 YAML 配置文件中加载指定脚本的默认配置。
    
    参数:
    - config_file (str): 配置文件的路径，默认为 'config.yaml'。
    - script_name (str): 当前脚本的名称，用于区分配置项。

    返回:
    - dict: 当前脚本的配置。
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config.get(script_name, {})
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_file}' 未找到。使用默认参数。")
        return {}
    except yaml.YAMLError as e:
        print(f"解析 YAML 配置文件时出错: {e}")
        return {}


def save_config(config, config_file='config.yaml', script_name='summarizer'):
    """
    将配置保存到 YAML 文件中。如果文件已存在，则更新其中的内容。
    
    参数:
    - config (dict): 要保存的配置。
    - config_file (str): 配置文件的路径，默认为 'config.yaml'。
    - script_name (str): 当前脚本的名称，用于区分配置项。
    """
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as file:
                current_config = yaml.safe_load(file)
        else:
            current_config = {}

        current_config[script_name] = config
        with open(config_file, 'w', encoding='utf-8') as file:
            yaml.dump(current_config, file, allow_unicode=True)
        print(f"配置文件 '{config_file}' 已成功更新。")
    except Exception as e:
        print(f"保存配置文件时出错: {e}")


def get_api_key(config, config_file='config.yaml', script_name='summarizer') -> str:
    """
    从配置文件中获取 API 密钥。如果配置中没有 API 密钥，则提示用户输入并保存到配置文件中。

    返回:
    - str: 获取到的 API 密钥。
    """
    api_key = config.get('api_key')
    if not api_key:
        api_key = input("请输入你的 OpenAI API 密钥: ").strip()
        config['api_key'] = api_key
        save_config(config, config_file=config_file, script_name=script_name)
    return api_key