import os
import sys
import yaml


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

        import torch
    
        if torch.cuda.is_available():
            print(f"检测到 CUDA 可用。当前 CUDA 版本: {torch.version.cuda}")
            self.device = 'cuda'
        else:
            print("未检测到可用的 CUDA，将使用 CPU 进行推理。")
            self.device = 'cpu'

    def check_autogptq(self):
        """
        检查 AutoGPTQ 的安装情况，并提供 CUDA 支持建议。
        """
        self.check_installation(
            "auto_gptq",
            self.config.get('autogptq', {}).get('install_instructions', '')
        )
        # 检查 CUDA 支持
        if self.device == 'cuda':
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

    def setup_chat_strict(self):
        """
        检查 torch, transformers, autogptq, autoawq 和 llama-cpp-python 是否安装。
        """
        print("⚠️ 注意，当前是严格检查，需要安装所有环境才能继续执行。")
        self.check_pytorch_transformers()
        self.check_autogptq()
        self.check_autoawq()
        self.check_llama_cpp()

    def setup_chat_by_model(self, model_name_or_path):
        """
        根据传入的模型路径来判断需要检查哪些依赖。
        例如:
          - GGUF: 只需要 llama-cpp-python
          - GPTQ: 需要 auto_gptq
          - AWQ:  需要 autoawq
          - 其他普通模型: 只要安装了 transformers/pytorch 即可
        """
        print("⚙️ 根据模型类型检查依赖...")
        self.check_pytorch_transformers()

        # 如果模型路径以 .gguf 结尾，检查 llama-cpp (GGUF 格式)
        if model_name_or_path.lower().endswith('.gguf'):
            print("检测到 GGUF 格式模型，检查 llama_cpp 依赖...")
            self.check_llama_cpp()
            return

        # 如果模型路径包含 gptq，检查 auto_gptq
        if "gptq" in model_name_or_path.lower():
            print("检测到 GPTQ 模型，检查 auto_gptq 依赖...")
            self.check_autogptq()

        # 如果模型路径包含 awq，检查 autoawq
        if "awq" in model_name_or_path.lower():
            print("检测到 AWQ 模型，检查 autoawq 依赖...")
            self.check_autoawq()

        # 如果以上都不符合，就默认只检查 pytorch/transformers，够用即可。
        print("✅ 依赖检查结束。")

    def get_device(self):
        return self.device