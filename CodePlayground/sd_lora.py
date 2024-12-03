# ===============================================================
# 这是一个 Toy AI 绘画工具，用于在命令行使用 LoRA 微调 Stable Diffusion 并生成图像
# 如果遇到错误，欢迎通过 Issues 或 Discussions 提交反馈。为了更快解决问题，请尽可能附上运行环境和可复现的命令。

# 对应文章：《16. 用 LoRA 微调 Stable Diffusion：拆开炼丹炉，动手实现你的第一次 AI 绘画）》
# https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/16.%20用%20LoRA%20微调%20Stable%20Diffusion：拆开炼丹炉，动手实现你的第一次%20AI%20绘画.md

# 使用方法：
#   python sd_lora.py [参数可选]
# 示例：
#   python sd_lora.py -d ./Datasets/Brad -gp ./Datasets/prompts/validation_prompt.txt

# 查看完整帮助：使用 -h 或 --help
# ===============================================================

import os
import math
import glob
import argparse
import yaml

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr

from peft import LoraConfig, get_peft_model, PeftModel

from utils.config_manager import load_config

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 当前使用的设备: {DEVICE}")

# 图片后缀列表
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

# 指定训练图像的分辨率，所有图像都会 resize 处理
resolution = 512


class Text2ImageDataset(torch.utils.data.Dataset):
    """
    用于构建文本到图像模型的微调数据集。
    
    你可以根据需求定制这个类，例如适配特定格式的数据集或更改数据增强方法，保持返回形式一致就可以直接用于训练。

    参数:
    - images_folder: str, 图像文件夹路径
    - captions_folder: str, 标注文件夹路径
    - transform: function, 将原始图像转换为 torch.Tensor
    - tokenizer: CLIPTokenizer, 将文本标注转为 token ids

    返回:
    - (image_tensor, input_ids): 一个包含图像 Tensor 和对应文本 token ids 的元组。
    """
    def __init__(self, images_folder, captions_folder, transform, tokenizer):
        # 获取所有图像文件路径
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
        self.image_paths = sorted(self.image_paths)

        # 加载对应的文本标注
        caption_paths = sorted(glob.glob(os.path.join(captions_folder, "*.txt")))
        captions = []
        for p in caption_paths:
            with open(p, "r", encoding="utf-8") as f:
                captions.append(f.readline().strip())

        # 确保图像和文本标注数量一致
        if len(captions) != len(self.image_paths):
            raise ValueError("图像数量与文本标注数量不一致，请检查数据集。")

        # 使用 tokenizer 将文本标注转换为 tokens
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.input_ids = inputs.input_ids
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        input_id = self.input_ids[idx]
        try:
            # 加载图像并转换为 RGB 模式，然后应用数据增强
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image)
        except Exception as e:
            print(f"⚠️ 无法加载图像路径: {img_path}, 错误: {e}")
            # 返回一个全零的张量和空的输入 ID 以避免崩溃
            tensor = torch.zeros((3, resolution, resolution))
            input_id = torch.zeros_like(input_id)
        
        return tensor, input_id

    def __len__(self):
        return len(self.image_paths)

def prepare_lora_model(lora_config, pretrained_model_name_or_path, model_path, weight_dtype, resume=False):
    """
    加载完整的 Stable Diffusion 模型，包括 LoRA 层。

    参数:
    - lora_config: LoraConfig, LoRA 的配置对象
    - pretrained_model_name_or_path: str, Hugging Face 上的模型名称或路径
    - model_path: str, 预训练模型的路径
    - weight_dtype: torch.dtype, 模型权重的数据类型
    - resume: bool, 是否从上一次训练中恢复

    返回:
    - tokenizer: CLIPTokenizer
    - noise_scheduler: DDPMScheduler
    - unet: UNet2DConditionModel
    - vae: AutoencoderKL
    - text_encoder: CLIPTextModel
    """
    # 加载噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

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

    # 加载 VAE 模型
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae"
    )

    # 加载 UNet 模型
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        subfolder="unet"
    )
    
    # 如果设置为继续训练，则加载上一次的模型权重
    if resume:
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("当 resume 设置为 True 时，必须提供有效的 model_path")
        # 使用 PEFT 的 from_pretrained 方法加载 LoRA 模型
        text_encoder = PeftModel.from_pretrained(text_encoder, os.path.join(model_path, "text_encoder"))
        unet = PeftModel.from_pretrained(unet, os.path.join(model_path, "unet"))

        # 确保 UNet 和文本编码器的可训练参数的 requires_grad 为 True
        for param in unet.parameters():
            if not param.requires_grad:
                param.requires_grad = True
        for param in text_encoder.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                
        print(f"✅ 已从 {model_path} 恢复模型权重")

    else:
        # 将 LoRA 配置应用到 text_encoder 和 unet
        text_encoder = get_peft_model(text_encoder, lora_config)
        unet = get_peft_model(unet, lora_config)

        # 打印可训练参数数量
        print("📊 Text Encoder 可训练参数:")
        text_encoder.print_trainable_parameters()
        print("📊 UNet 可训练参数:")
        unet.print_trainable_parameters()
    
    # 冻结 VAE 参数
    vae.requires_grad_(False)

    # 将模型移动到设备上并设置权重的数据类型
    unet.to(DEVICE, dtype=weight_dtype)
    vae.to(DEVICE, dtype=weight_dtype)
    text_encoder.to(DEVICE, dtype=weight_dtype)
    
    return tokenizer, noise_scheduler, unet, vae, text_encoder

def prepare_optimizer(unet, text_encoder, unet_learning_rate=5e-4, text_encoder_learning_rate=1e-4):
    """
    为 UNet 和文本编码器的可训练参数分别设置优化器，并指定不同的学习率。

    参数:
    - unet: UNet2DConditionModel, Hugging Face 的 UNet 模型
    - text_encoder: CLIPTextModel, Hugging Face 的文本编码器
    - unet_learning_rate: float, UNet 的学习率
    - text_encoder_learning_rate: float, 文本编码器的学习率

    返回:
    - 优化器 Optimizer
    """
    # 筛选出 UNet 中需要训练的 LoRA 层参数
    unet_lora_layers = [p for p in unet.parameters() if p.requires_grad]
    
    # 筛选出文本编码器中需要训练的 LoRA 层参数
    text_encoder_lora_layers = [p for p in text_encoder.parameters() if p.requires_grad]
    
    # 将需要训练的参数分组并设置不同的学习率
    trainable_params = [
        {"params": unet_lora_layers, "lr": unet_learning_rate},
        {"params": text_encoder_lora_layers, "lr": text_encoder_learning_rate}
    ]
    
    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(trainable_params)
    
    return optimizer

def collate_fn(examples):
    pixel_values = []
    input_ids = []
    
    for tensor, input_id in examples:
        pixel_values.append(tensor)
        input_ids.append(input_id)
    
    pixel_values = torch.stack(pixel_values, dim=0).float()
    input_ids = torch.stack(input_ids, dim=0)
    
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def load_validation_prompts(validation_prompt_path):
    """
    加载验证提示文本。
    
    参数:
    - validation_prompt_path: str, 验证提示文件的路径，每一行对应一个 prompt，参考示例文件
    
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

# ========== 参数设置 ==========
def parse_args(train_config, generate_config):
    """
    解析命令行参数，用于训练和生成流程。
    
    默认值取自 config.yaml 的 'train' 和 'generate' 配置。所有参数均为可选项。
    """
    parser = argparse.ArgumentParser(
        description=(
            "使用 LoRA 微调 Stable Diffusion 模型，并在训练后生成图像。\n"
            "如果不传入对应参数，则默认在相应阶段使用 config.yaml 中的配置。"
        ),
        epilog="示例: python sd_lora.py -d ./Datasets/Brad -gp ./Datasets/prompts/validation_prompt.txt"
    )

    # 训练相关参数
    parser.add_argument("-d", "--dataset_path", type=str, default=train_config.get('dataset_path'),
                        help="数据集路径")
    parser.add_argument("-c", "--captions_folder", type=str, default=None,
                        help="文本标注文件夹路径（默认与数据集路径相同）")
    parser.add_argument("-r", "--root", type=str, default=train_config.get('root', './SD'),
                        help="根路径")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="模型检查点路径，默认为 root + dataset_name + 'logs/checkpoint-last'")
    parser.add_argument("-p", "--pretrained_model_name_or_path", type=str, default=train_config.get('pretrained_model_name_or_path'),
                        help="预训练的 Stable Diffusion 模型名称或路径")
    parser.add_argument("-s", "--seed", type=int, default=train_config.get('seed', 1126),
                        help="随机数种子")
    parser.add_argument("-w", "--weight_dtype", type=str, default=train_config.get('weight_dtype', 'torch.bfloat16'),
                        help="权重数据类型")
    parser.add_argument("-b", "--batch_size", type=int, default=train_config.get('batch_size', 2),
                        help="训练批次大小")
    parser.add_argument("-e", "--max_train_steps", type=int, default=train_config.get('max_train_steps', 200),
                        help="总训练步数")
    parser.add_argument("-u", "--unet_learning_rate", type=float, default=train_config.get('unet_learning_rate', 1e-4),
                        help="UNet 的学习率")
    parser.add_argument("-t", "--text_encoder_learning_rate", type=float, default=train_config.get('text_encoder_learning_rate', 1e-4),
                        help="文本编码器的学习率")
    parser.add_argument("-g", "--snr_gamma", type=float, default=train_config.get('snr_gamma', 5),
                        help="SNR 参数")
    parser.add_argument("-l", "--lr_scheduler_name", type=str, default=train_config.get('lr_scheduler_name', "cosine_with_restarts"),
                        help="学习率调度器名称")
    parser.add_argument("-warmup", "--lr_warmup_steps", type=int, default=train_config.get('lr_warmup_steps', 100),
                        help="学习率预热步数")
    parser.add_argument("-cycle", "--num_cycles", type=int, default=train_config.get('num_cycles', 3),
                        help="学习率调度器的周期数量")
    parser.add_argument("--resume", action="store_true", default=train_config.get('resume', False),
                        help="是否从上一次训练中恢复")
    parser.add_argument("--no-train", action="store_true", default=False,
                        help="是否跳过训练过程，直接进行图像生成，默认训练")
    parser.add_argument("--no-generate", action="store_true", default=False,
                        help="训练完成后是否跳过图像生成，默认生成")
    
    # 生成相关参数，前缀 g 代指 generate，和之前的短参数做区分
    parser.add_argument("-gp", "--prompts_path", type=str, default=generate_config.get('prompts_path'),
                        help="验证提示文件路径")
    parser.add_argument("-gs", "--save_folder", type=str, default=generate_config.get('save_folder'),
                        help="保存生成图像的文件夹路径，默认为 root + dataset_name + '/inference'")
    parser.add_argument("-gn", "--num_inference_steps", type=int, default=generate_config.get('num_inference_steps', 50),
                        help="推理步骤数")
    parser.add_argument("-gg", "--guidance_scale", type=float, default=generate_config.get('guidance_scale', 7.5),
                        help="指导尺度")
    
    args = parser.parse_args()

    # 如果 captions_folder 未指定，则默认为 dataset_path，意思是图片和标注在同一个文件夹下          
    if args.captions_folder is None:
        args.captions_folder = args.dataset_path

    # 权重数据类型检查
    if args.weight_dtype:
        try:
            args.weight_dtype = getattr(torch, args.weight_dtype.split('.')[-1])
        except AttributeError:
            print(f"⚠️ 无效的 weight_dtype '{args.weight_dtype}'，使用默认 'torch.float32'")
            args.weight_dtype = torch.float32

    # 自动设置 model_path，如果未指定
    if args.model_path is None:
        dataset_name = os.path.basename(os.path.abspath(args.dataset_path))
        args.model_path = os.path.join(args.root, dataset_name, "logs", "checkpoint-last")

    # 自动设置 save_folder，如果未指定
    if args.save_folder is None:
        dataset_name = os.path.basename(os.path.abspath(args.dataset_path))
        args.save_folder = os.path.join(args.root, dataset_name, "inference")

    return args

# ========== 主函数 ==========
def main():
    # 加载配置文件，获取 train 和 generate 部分的配置
    train_config, generate_config = load_config(script_name='train'), load_config(script_name='generate')

    # 解析命令行参数
    args = parse_args(train_config, generate_config)

    # 设置随机数种子
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 自动设置项目名称（数据集文件夹名称）
    dataset_name = os.path.basename(os.path.abspath(args.dataset_path))

    # 输出文件夹
    output_folder = os.path.dirname(args.model_path)
    os.makedirs(output_folder, exist_ok=True)

    if not args.no_train:
        # 数据增强操作
        train_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),  # 调整图像大小
                transforms.CenterCrop(resolution),  # 中心裁剪图像
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),  # 将图像转换为张量
            ]
        )
    
        # LoRA 配置
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "out_proj",
                "to_k", "to_q", "to_v", "to_out.0"
            ],
            lora_dropout=0
        )
    
        # ========== 微调前的准备 ==========
    
        # 初始化 tokenizer，用于加载数据集
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer"
        )
    
        # 准备数据集
        dataset = Text2ImageDataset(
            images_folder=args.dataset_path,
            captions_folder=args.captions_folder,
            transform=train_transform,
            tokenizer=tokenizer,
        )
    
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            num_workers=4,
        )
    
        print("✅ 数据集准备完成！")
    
        # 准备模型
        tokenizer, noise_scheduler, unet, vae, text_encoder = prepare_lora_model(
            lora_config,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            model_path=args.model_path,
            weight_dtype = args.weight_dtype,
            resume=args.resume
        )
    
        # 准备优化器
        optimizer = prepare_optimizer(
            unet, 
            text_encoder, 
            unet_learning_rate=args.unet_learning_rate, 
            text_encoder_learning_rate=args.text_encoder_learning_rate
        )
    
        # 设置学习率调度器
        lr_scheduler = get_scheduler(
            args.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.num_cycles
        )
    
        print("✅ 模型和优化器准备完成！可以开始训练。")
    
        # ========== 开始微调 ==========
    
        # 禁用并行化，避免警告
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
        # 初始化
        global_step = 0
    
        # 进度条显示训练进度
        progress_bar = tqdm(
            range(args.max_train_steps),
            desc="训练步骤",
        )
    
        # 训练循环
        for epoch in range(math.ceil(args.max_train_steps / len(train_dataloader))):
            unet.train()
            text_encoder.train()
            
            for step, batch in enumerate(train_dataloader):
                if global_step >= args.max_train_steps:
                    break
                
                # 编码图像为潜在表示（latent）
                latents = vae.encode(batch["pixel_values"].to(DEVICE, dtype=args.weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # 根据 VAE 的缩放因子调整潜在空间
    
                # 为潜在表示添加噪声，生成带噪声的图像
                noise = torch.randn_like(latents)  # 生成与潜在表示相同形状的随机噪声
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=DEVICE).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
                # 获取文本的嵌入表示
                encoder_hidden_states = text_encoder(batch["input_ids"].to(DEVICE))[0]
    
                # 计算目标值
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise  # 预测噪声
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)  # 预测速度向量
    
                # UNet 模型预测
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states)[0]
    
                # 计算损失
                if not args.snr_gamma:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # 计算信噪比 (SNR) 并根据 SNR 加权 MSE 损失
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    
                    # 计算加权的 MSE 损失
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
    
                # 反向传播
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_step += 1
    
                # 打印训练损失
                if global_step % 50 == 0 or global_step == args.max_train_steps:
                    print(f"🔥 步骤 {global_step}, 损失: {loss.item():.4f}")
    
                # 保存中间检查点，每 100 步保存一次
                if global_step % 100 == 0:
                    save_path = os.path.join(output_folder, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
    
                    # 使用 save_pretrained 保存 PeftModel
                    unet.save_pretrained(os.path.join(save_path, "unet"))
                    text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
                    print(f"💾 已保存中间模型到 {save_path}")
    
        # 保存最终模型到 checkpoint-last
        save_path = args.model_path
        os.makedirs(save_path, exist_ok=True)
        unet.save_pretrained(os.path.join(save_path, "unet"))
        text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
        print(f"💾 已保存最终模型到 {save_path}")
    
        print("🎉 微调完成！")
    else:
        print("🚫 已跳过训练过程。")

    # ========== 训练完成后生成图像 ==========
    if not args.no_generate:
        print("🖼 开始生成图像...")
        generate_after_training(args)
    else:
        print("🚫 已跳过图像生成。")

def generate_after_training(args):
    """
    训练完成后生成图像的函数
    """
    # 设置设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置参数
    prompts_path = args.prompts_path
    save_folder = args.save_folder
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    weight_dtype = args.weight_dtype
    seed = args.seed
    root = args.root

    # 设置随机数种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 准备 LoRA 模型
    print("✅ 准备 LoRA 模型用于生成图像...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype
    )

    # 加载微调后的权重
    model_path = args.model_path
    text_encoder = PeftModel.from_pretrained(text_encoder, os.path.join(model_path, "text_encoder"))
    unet = PeftModel.from_pretrained(unet, os.path.join(model_path, "unet"))

    # 合并 LoRA 权重
    text_encoder = text_encoder.merge_and_unload()
    unet = unet.merge_and_unload()

    # 切换为评估模式
    text_encoder.eval()
    unet.eval()

    # 将模型移动到设备
    text_encoder.to(DEVICE, dtype=weight_dtype)
    unet.to(DEVICE, dtype=weight_dtype)

    # 创建管道
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline = pipeline.to(DEVICE)

    # 设置随机数生成器
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    # 加载提示
    validation_prompts = load_validation_prompts(prompts_path)

    # 生成图像
    generate_images(
        pipeline=pipeline,
        prompts=validation_prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        save_folder=save_folder,
        generator=generator
    )

if __name__ == "__main__":
    main()