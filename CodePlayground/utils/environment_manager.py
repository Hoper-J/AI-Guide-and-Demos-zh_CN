import os
import sys
import yaml

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EnvironmentManager:
    def __init__(self, config_file='environment.yaml'):
        """
        初始化环境管理器，并加载配置文件。
        
        参数:
        - config_file (str): 配置文件路径，默认为 'environment.yaml'。
        """
        self.config = self.load_config(config_file)
        print("环境管理器已初始化。")

    def load_config(self, config_file):
        """
        加载 YAML 配置文件。
        
        参数:
        - config_file (str): 配置文件路径。
        
        返回:
        - dict: 配置字典。
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"配置文件 '{config_file}' 未找到，使用默认设置。")
            return {}
        except yaml.YAMLError as e:
            print(f"解析配置文件时出错: {e}")
            return {}

    def check_installation(self, module_name, install_instructions, required=True):
        """
        检查是否已安装模块并提供安装提示。
        
        参数:
        - module_name (str): 模块名称。
        - install_instructions (str): 如果未安装模块时，显示的安装指引。
        - required (bool): 是否强制要求安装，如果是，则在缺失时退出。
        """
        try:
            __import__(module_name)
            print(f"✅ {module_name} 已安装。")
        except ImportError:
            if required:
                print(f"❌ 缺少必要的依赖: {module_name}。请按以下指引进行安装：")
                print(install_instructions)
                sys.exit(1)
            else:
                print(f"⚠️ 可选模块 {module_name} 未检测到。")
                print(install_instructions)
                self.missing_optional_modules.append(module_name)

    def check_pytorch_transformers(self):
        """
        检查 PyTorch 和 Transformers 的安装情况。
        """
        self.check_installation(
            "torch",
            self.config.get('pytorch', {}).get('install_instructions', ''),
            required=True
        )
        self.check_installation(
            "transformers",
            self.config.get('transformers', {}).get('install_instructions', ''),
            required=True
        )
        if DEVICE == 'cuda':
            print(f"检测到 CUDA 可用。当前 CUDA 版本: {torch.version.cuda}")
        else:
            print("未检测到可用的 CUDA，将使用 CPU 进行推理。")

    def check_autogptq(self):
        """
        检查 AutoGPTQ 的安装情况，并提供 CUDA 支持建议。
        """
        self.check_installation(
            "auto_gptq",
            self.config.get('autogptq', {}).get('install_instructions', '')
        )
        # 检查 CUDA 支持
        if DEVICE == 'cuda':
            gptq_version_info = os.popen("pip list | grep auto-gptq").read()
            if "cu" not in gptq_version_info:
                print(self.config.get('autogptq', {}).get('cuda_warning', ''))
            else:
                print("GPTQ 模型文件已支持 CUDA 推理。")

    def check_autoawq(self):
        """
        检查 AutoAWQ 的安装情况。
        """
        self.check_installation(
            "awq",
            self.config.get('autoawq', {}).get('install_instructions', '')
        )

    def check_llama_cpp(self):
        """
        检查 Llama-cpp-python 的安装情况。
        """
        self.check_installation(
            "llama_cpp",
            self.config.get('llama_cpp', {}).get('install_instructions', ''),
            required=True
        )

    def setup_chat(self):
        """
        根据配置文件逐一检查并设置依赖环境。
        """
        print("⚠️ 注意，当前是严格检查，你需要安装所有环境才能执行 chat.py，你也可以注释掉脚本文件中的 EnvironmentManager.setup_chat() 代码取消这个行为。")
        self.check_pytorch_transformers()
        self.check_autogptq()
        self.check_autoawq()
        self.check_llama_cpp()