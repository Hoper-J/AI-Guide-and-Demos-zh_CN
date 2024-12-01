# ========== 导入库 ==========
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

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr

from peft import LoraConfig, get_peft_model, PeftModel

from utils.config_manager import load_config


# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 当前使用的设备: {DEVICE}")

# 图片后缀列表
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

class Text2ImageDataset(torch.utils.data.Dataset):
    """
    用于构建文本到图像模型的微调数据集

    参数:
    - images_folder: str, 图像文件夹路径
    - captions_folder: str, 标注文件夹路径
    - transform: function, 将原始图像转换为 torch.Tensor
    - tokenizer: CLIPTokenizer, 将文本标注转为 token ids

    返回:
    - 
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

# ========== 参数设置 ==========

def parse_args(config):
    parser = argparse.ArgumentParser(
        description="使用 LoRA 微调 Stable Diffusion 模型",
        epilog="示例: python train_lora.py -d ./data/Brad -c ./data/Brad/captions"
    )

    # 简短选项和长选项
    parser.add_argument("-d", "--dataset_path", type=str, default=config.get('dataset_path'),
                        help="数据集路径，默认为 config.yaml 中的 'train.dataset_path'")
    parser.add_argument("-c", "--captions_folder", type=str, default=None,
                        help="文本标注文件夹路径，默认为 dataset_path")
    parser.add_argument("-r", "--root", type=str, default=config.get('root', './SD'),
                        help="根路径，默认为 './SD'")
    parser.add_argument("-m", "--model_path", type=str, default=None,
                        help="模型检查点路径，默认为 root + dataset_name + 'logs/checkpoint-last'")
    parser.add_argument("-p", "--pretrained_model_name_or_path", type=str, default=config.get('pretrained_model_name_or_path'),
                        help="预训练的 Stable Diffusion 模型名称或路径，默认为 config.yaml 中的 'train.pretrained_model_name_or_path'")
    parser.add_argument("-s", "--seed", type=int, default=config.get('seed', 1126),
                        help="随机数种子，默认为 config.yaml 中的 'train.seed' 或 1126")
    parser.add_argument("-w", "--weight_dtype", type=str, default=config.get('weight_dtype', 'torch.bfloat16'),
                        help="权重数据类型，默认为 config.yaml 中的 'train.weight_dtype'")
    parser.add_argument("-b", "--batch_size", type=int, default=config.get('batch_size', 2),
                        help="训练批次大小，默认为 config.yaml 中的 'train.batch_size'")
    parser.add_argument("-e", "--max_train_steps", type=int, default=config.get('max_train_steps', 200),
                        help="总训练步数，默认为 config.yaml 中的 'train.max_train_steps'")
    parser.add_argument("-u", "--unet_learning_rate", type=float, default=config.get('unet_learning_rate', 1e-4),
                        help="UNet 的学习率，默认为 config.yaml 中的 'train.unet_learning_rate'")
    parser.add_argument("-t", "--text_encoder_learning_rate", type=float, default=config.get('text_encoder_learning_rate', 1e-4),
                        help="文本编码器的学习率，默认为 config.yaml 中的 'train.text_encoder_learning_rate'")
    parser.add_argument("-g", "--snr_gamma", type=float, default=config.get('snr_gamma', 5),
                        help="SNR 参数，默认为 config.yaml 中的 'train.snr_gamma'")
    parser.add_argument("-l", "--lr_scheduler_name", type=str, default=config.get('lr_scheduler_name', "cosine_with_restarts"),
                        help="学习率调度器名称，默认为 config.yaml 中的 'train.lr_scheduler_name'")
    parser.add_argument("-warmup", "--lr_warmup_steps", type=int, default=config.get('lr_warmup_steps', 100),
                        help="学习率预热步数，默认为 config.yaml 中的 'train.lr_warmup_steps'")
    parser.add_argument("-cycle", "--num_cycles", type=int, default=config.get('num_cycles', 3),
                        help="学习率调度器的周期数量，默认为 config.yaml 中的 'train.num_cycles'")
    parser.add_argument("--resume", action="store_true", default=config.get('resume', False),
                        help="是否从上一次训练中恢复，默认为 False")
    
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

    return args


def main():
    # 加载配置文件，指定脚本名称为 'train'
    config = load_config('config.yaml', script_name='train')
    
    # 解析命令行参数
    args = parse_args(config)
    
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
    
    # 训练图像的分辨率
    global resolution
    resolution = 512
    
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

if __name__ == "__main__":
    main()