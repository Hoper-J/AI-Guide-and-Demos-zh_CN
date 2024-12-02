# ========== 导入库 ==========
import os
import glob
import argparse

import torch
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor

from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline

from peft import PeftModel

from PIL import Image
import numpy as np
import cv2

from utils.config_manager import load_config

# 设置全局设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 当前使用的设备: {DEVICE}")


def load_validation_prompts(validation_prompt_path):
    """
    加载验证提示文本。
    
    参数:
    - validation_prompt_path: str, 验证提示文件的路径，每一行就是一个 prompt
    
    返回:
    - validation_prompt: list, prompt 列表
    """
    with open(validation_prompt_path, "r", encoding="utf-8") as f:
        validation_prompt = [line.strip() for line in f.readlines()]
    return validation_prompt

def generate_images(pipeline, prompts, num_inference_steps=50, guidance_scale=7.5, save_folder="inference", generator=None):
    """
    使用 DiffusionPipeline 生成图像，保存到指定文件夹并返回生成的图像列表。

    参数:
    - pipeline: DiffusionPipeline, 已加载并配置好的 Pipeline
    - prompts: list, 文本提示列表
    - num_inference_steps: int, 推理步骤数
    - guidance_scale: float, 指导尺度
    - save_folder: str, 保存生成图像的文件夹路径
    - generator: torch.Generator, 控制生成随机数的种子

    返回:
    - generated_images: list, 生成的 PIL 图像对象列表
    """
    print("🎨 正在生成图像...")
    os.makedirs(save_folder, exist_ok=True)
    generated_images = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="生成图像中")):
        # 使用 pipeline 生成图像
        image = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
        
        # 保存图像到指定文件夹
        save_file = os.path.join(save_folder, f"generated_{i+1}.png")
        image.save(save_file)
        
        # 将图像保存到列表中，稍后返回
        generated_images.append(image)
    
    print(f"✅ 已生成并保存 {len(prompts)} 张图像到 {save_folder}")
    
    return generated_images

def prepare_lora_model(pretrained_model_name_or_path, model_path, weight_dtype):
    """
    加载完整的 Stable Diffusion 模型，包括 LoRA 层，并合并权重。
    
    参数:
    - pretrained_model_name_or_path: str, 预训练模型名称或路径
    - model_path: str, 微调模型检查点路径
    - weight_dtype: torch.dtype, 模型权重的数据类型
    
    返回:
    - tokenizer: CLIPTokenizer
    - unet: UNet2DConditionModel
    - text_encoder: CLIPTextModel
    """
    # 加载 Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer"
    )

    # 加载 CLIP 文本编码器
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        subfolder="text_encoder"
    )

    # 加载 UNet 模型
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        subfolder="unet"
    )
    
    # 检查模型路径是否存在
    if model_path is None or not os.path.exists(model_path):
        raise ValueError("必须提供有效的 model_path")
    
    # 使用 PEFT 的 from_pretrained 方法加载 LoRA 模型
    text_encoder = PeftModel.from_pretrained(text_encoder, os.path.join(model_path, "text_encoder"))
    unet = PeftModel.from_pretrained(unet, os.path.join(model_path, "unet"))

    # 合并 LoRA 权重到基础模型
    text_encoder = text_encoder.merge_and_unload()
    unet = unet.merge_and_unload()

    # 切换为评估模式
    text_encoder.eval()
    unet.eval()

    # 将模型移动到设备上并设置权重的数据类型
    unet.to(DEVICE, dtype=weight_dtype)
    text_encoder.to(DEVICE, dtype=weight_dtype)
    
    return tokenizer, unet, text_encoder


def parse_args(config):
    """
    解析命令行参数，并使用配置文件中的默认值。

    参数:
    - config: dict, 配置文件内容

    返回:
    - args: argparse.Namespace, 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(
        description="使用微调后的模型生成图像",
        epilog="示例: python generate_images.py -i ./prompts/validation_prompt.txt"
    )
    
    # 简短选项和长选项
    parser.add_argument("-i", "--prompts_path", type=str, default=config.get('prompts_path'),
                        help="验证提示文件路径，默认为 config.yaml 中的 'generate.prompts_path'")
    parser.add_argument("-r", "--root", type=str, default=config.get('root', './SD'),
                        help="根路径，默认为 './SD'")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="模型检查点路径，如果为空则根据 root 和 train.dataset_name 构造")
    parser.add_argument("-s", "--save_folder", type=str, default=config.get('save_folder'),
                        help="保存生成图像的文件夹路径，默认为 root + train.dataset_name + '/inference'")
    parser.add_argument("-p", "--pretrained_model_name_or_path", type=str, default=config.get('pretrained_model_name_or_path'),
                        help="预训练的 Stable Diffusion 模型名称或路径，默认为 config.yaml 中的 'generate.pretrained_model_name_or_path'")
    parser.add_argument("-n", "--num_inference_steps", type=int, default=config.get('num_inference_steps', 50),
                        help="推理步骤数，默认为 config.yaml 中的 'generate.num_inference_steps'")
    parser.add_argument("-g", "--guidance_scale", type=float, default=config.get('guidance_scale', 7.5),
                        help="指导尺度，默认为 config.yaml 中的 'generate.guidance_scale'")
    parser.add_argument("-w", "--weight_dtype", type=str, default=config.get('weight_dtype'),
                        help="权重数据类型，如 'torch.bfloat16' 或 'torch.float32'，默认为 config.yaml 中的 'generate.weight_dtype'")
    parser.add_argument("-e", "--seed", type=int, default=config.get('seed', 1126),
                        help="随机数种子，默认为 config.yaml 中的 'generate.seed' 或 1126")
    
    args = parser.parse_args()

    # 权重数据类型检查
    if args.weight_dtype:
        try:
            args.weight_dtype = getattr(torch, args.weight_dtype.split('.')[-1])
        except AttributeError:
            print(f"⚠️ 无效的 weight_dtype '{args.weight_dtype}'，使用默认 'torch.float32'")
            args.weight_dtype = torch.float32

    # 有需要的话加载 train_config 进行默认配置
    if args.model_path is None or args.save_folder is None:
        train_config = load_config(script_name='train')
    
        # 自动设置 model_path，如果未指定
        if args.model_path is None:
            # 尝试从 config.yaml 中获取 dataset_path
            dataset_path = train_config.get('dataset_path')
            if dataset_path is None:
                raise ValueError("model_path 为空且无法从 config.yaml 中获取 dataset_path，请指定 model_path 或在 config.yaml 中提供 dataset_path")
            dataset_name = os.path.basename(os.path.abspath(dataset_path))
            args.model_path = os.path.join(args.root, dataset_name, "logs", "checkpoint-last")
    
        # 自动设置 save_folder，如果未指定
        if args.save_folder is None:
            dataset_name = os.path.basename(os.path.abspath(dataset_path))
            args.save_folder = os.path.join(args.root, dataset_name, "inference")
    
    return args


def main():
    # 加载配置文件，指定脚本名称为 'generate'
    config= load_config(script_name='generate')
    
    # 解析命令行参数
    args = parse_args(config)
    
    # 设置随机数种子
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 准备 LoRA 模型
    print("✅ 准备 LoRA 模型...")
    tokenizer, unet, text_encoder = prepare_lora_model(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        model_path=args.model_path,
        weight_dtype=args.weight_dtype,
    )
    
    # 创建 DiffusionPipeline 并更新其组件
    print("🔄 创建 DiffusionPipeline...")
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        torch_dtype=args.weight_dtype,
        safety_checker=None,
    )
    pipeline = pipeline.to(DEVICE)
    
    # 设置随机数种子
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    
    # 加载验证提示
    print("📂 加载验证提示...")
    validation_prompts = load_validation_prompts(args.prompts_path)
    
    # 生成图像
    generate_images(
        pipeline=pipeline,
        prompts=validation_prompts,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        save_folder=args.save_folder,
        generator=generator
    )

if __name__ == "__main__":
    main()