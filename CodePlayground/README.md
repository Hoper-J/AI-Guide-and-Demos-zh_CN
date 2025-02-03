# CodePlayground

欢迎来到 **CodePlayground** 🎡，这是一个课程相关的脚本游乐场。你可以在这里玩转各种工具和脚本，享受学习和实验的乐趣。

​	注意⚠️，所有的脚本都是一个 Toy 的版本。

## 搭建场地

1. 克隆仓库（如果之前克隆过可以跳过）：

   ```bash
   git clone https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN.git
   cd AI-Guide-and-Demos-zh_CN/CodePlayground
   ```

2. 创建并激活虚拟环境（可选）：

   ```bash
   conda create -n playground python=3.9
   conda activate playground
   ```

3. 安装依赖

   ### PyTorch 依赖

   选择以下两种方式之一安装 PyTorch：

   ```python
   # pip
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # conda
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

   ### AI Summarizer 依赖

   a. **ffmpeg**（用于视频转音频）

   ```bash
   # Linux
   sudo apt-get update
   sudo apt-get install ffmpeg
   
   # Mac
   brew install ffmpeg
   ```
   
   b. **Python 库**
   
   ```python
   pip install openai-whisper openai pyyaml librosa srt certifi
   pip install numpy==1.26.4  # >= 2.0.0 会无法正常执行 summarizer.py
   ```
   
   ### SD LoRA 依赖
   
   ```bash
   pip install transformers diffusers peft tqdm numpy pyyaml pillow
   ```
   
   ### AI Chat 依赖
   
   根据模型文件对应配置。
   
   a. **GPTQ 模型文件**
   
   ```bash
   pip install optimum
   git clone https://github.com/PanQiWei/AutoGPTQ.git && %cd AutoGPTQ
   pip install -vvv --no-build-isolation -e .
   ```
   
   b. **AWQ 模型文件**
   
   ```bash
   pip install autoawq autoawq-kernels
   ```
   
   c. **GGUF 模型文件**
   
   ```bash
   CUDA_HOME="$(find /usr/local -name "cuda" -exec readlink -f {} \; \
             | awk '{print length($0), $0}' \
             | sort -n \
                | head -n1 \
                | cut -d ' ' -f 2)" && \
   CMAKE_ARGS="-DGGML_CUDA=on \
            -DCUDA_PATH=${CUDA_HOME} \
            -DCUDAToolkit_ROOT=${CUDA_HOME} \
            -DCUDAToolkit_INCLUDE_DIR=${CUDA_HOME} \
            -DCUDAToolkit_LIBRARY_DIR=${CUDA_HOME}/lib64 \
               -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc" \
   FORCE_CMAKE=1 \
   pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir --verbose
   ```
   


## 当前的玩具

<details> <summary> <strong>1. AI Summarizer</strong> </summary>

> [15. 用 API 实现 AI 视频摘要：动手制作属于你的 AI 视频助手](../Guide/15.%20用%20API%20实现%20AI%20视频摘要：动手制作属于你的%20AI%20视频助手.md)

**[summarizer.py](./summarizer.py)** 是一个 AI 摘要工具，用于从视频或音频文件中提取字幕并生成视频摘要，也可以直接处理现有的字幕文件。它集成了 Whisper 模型和 OpenAI API 来自动化这些过程。

#### 功能

- **视频转音频**：使用 FFmpeg 将视频文件转换为 WAV 格式的音频文件。
- **音频转录**：使用 Whisper 模型将音频转录为文本字幕。
- **字幕生成**：生成 SRT 格式的字幕文件。
- **视频摘要**：使用 OpenAI 的模型生成视频内容的摘要。

#### 快速使用

```bash
python summarizer.py examples/summarizer.mp4
```

仓库提供了一个样例视频供你运行，以防止可能存在的选择困难症 :)

#### 使用方法

你可以通过命令行运行 `summarizer.py`，并指定要处理的文件路径：

   ```bash
python summarizer.py file_path [--api_key YOUR_API_KEY] [--output_dir OUTPUT_DIR] [其他可选参数]
   ```

   - `file_path`：替换为要处理的文件路径，可以是视频、音频或字幕文件。
   - `--api_key`：可选参数，指定 OpenAI API 密钥。如果配置文件中已有密钥，则可以省略此参数。当不传入时，会要求输入，验证后会自动更新 config.yaml。
   - `--output_dir`：可选参数，指定生成文件保存的目录，默认为 `./output/` 文件夹。
   - 其他参数见[配置管理](#配置管理)或使用 `--help` 进行查看

   以上命令会从样例视频中提取音频，生成字幕并自动生成摘要。

   生成的文件默认会保存在 `./output` 文件夹下，包括：
   - **对应的音频文件**（MP3格式）
   - **转录生成的字幕文件**（SRT 格式）
   - **视频摘要文件**（TXT 格式）

#### 配置管理

脚本支持从 `config.yaml` 文件中读取默认配置，你可以通过编辑该文件来自定义参数，避免每次运行脚本时手动指定。

[config.yaml](./config.yaml#L1) 示例：

   ```yaml
summarizer:
  model_name: "medium"
  language: "zh"
  whisper_temperature: 0.2
  llm_temperature: 0.2
  timestamped: False
  max_tokens: 1000
  output_dir: "./output"
  api_key:
  api_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
   ```

**配置说明**

- `model_name`: Whisper 模型名称（如 `tiny`, `base`, `small`, `medium`, `large-v3`）。
- `language`: 转录语言，默认设置为 `zh`（中文）。
- `whisper_temperature`: Whisper 模型音频转字幕时的温度，范围为 0 到 1。
- `llm_temperature`: 大模型生成文本时的温度，范围为 0 到 1。
- `timestamped`: 是否保留转录文本的时间戳，布尔值。
- `max_tokens:` 摘要生成时的最大 token 数量。
- `output_dir`: 生成文件的默认保存目录。
- `api_key`: 你的 OpenAI API 密钥，可以通过命令行参数或配置文件指定。
- `api_base_url`: 默认使用阿里云大模型平台。


#### 注意事项

- **中间文件保留**：默认情况下，summarizer.py 会保留所有中间转换文件，如音频和字幕文件。如果你需要删除这些中间文件，可以在脚本中进行相应修改。
- **模型选择**：在 `model_name` 中选择 Whisper 模型时注意，模型越大对显存的占用越高，建议在显存充足的环境下使用。

</details>

<details> <summary> <strong>2. SD LoRA</strong> </summary>

> [16. 用 LoRA 微调 Stable Diffusion：拆开炼丹炉，动手实现你的第一次 AI 绘画](../Guide/16.%20用%20LoRA%20微调%20Stable%20Diffusion：拆开炼丹炉，动手实现你的第一次%20AI%20绘画.md)

**[sd_lora.py](./sd_lora.py)** 是一个 AI 绘画工具，对于指定数据集和 Stable Diffusion 模型，自动应用 LoRA 微调并生成图像。

### 功能

- **模型微调**：使用 LoRA 对预训练的 Stable Diffusion 模型进行简单的微调，适应特定的数据集或风格。
- **图像生成**：在训练完成后，使用微调后的模型根据文本提示生成图像。

### 使用方法

你可以通过命令行运行 `sd_lora.py`，并根据需要指定参数：

```bash
python sd_lora.py [可选参数]
```

默认使用 `config.yaml` 中的配置进行训练和图像生成。

### 示例

1. **准备样例数据集[^1]**：

   ```bash
   wget https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/raw/refs/heads/master/Demos/data/14/Datasets.zip
   unzip Datasets.zip
   ```

2. **使用指定的数据集和提示文件**：

   ```bash
   # 因为已经在 config.yaml 中配置，所以可以不指定参数
   python sd_lora.py
   # python sd_lora.py -d ./Datasets/Brad -gp ./Datasets/prompts/validation_prompt.txt
   ```

   - `-d` 或 `--dataset_path`：数据集路径。
   - `-gp` 或 `--prompts_path`：生成图像时使用的文本提示文件路径。

3. **跳过训练，仅生成图像**，使用 `--no-train` 参数：

   ```bash
   python sd_lora.py --no-train
   ```

   请确保在 `args.model_path` 指定的路径下存在已微调的模型权重。

4. **跳过图像生成，仅进行训练**，使用 `--no-generate` 参数：

   ```bash
   python sd_lora.py --no-generate
   ```

5. **指定其他参数**：

   ```bash
   python sd_lora.py -e 500 -b 4 -u 1e-4 -t 1e-5
   ```

   - `-e` 或 `--max_train_steps`：总训练步数。
   - `-b` 或 `--batch_size`：训练批次大小。
   - `-u` 或 `--unet_learning_rate`：UNet 的学习率。
   - `-t` 或 `--text_encoder_learning_rate`：文本编码器的学习率。
   - 其他参数使用 `--help` 进行查看。

### 配置管理

脚本支持从 `config.yaml` 文件中读取默认配置，避免每次运行时手动指定所有参数。

[config.yaml](./config.yaml#L12) 示例：

```yaml
train:
  root: "./SD"
  dataset_path: "./Datasets/Brad"
  captions_folder: # 存放文本标注的路径，默认和 dataset_path 一致
  model_path: # checkpoint-last 路径默认为 root + dataset_name + 'logs/checkpoint-last'，如果使用了 --no-train，需要确保 model_path 路径存在
  pretrained_model_name_or_path: "digiplay/AnalogMadness-realistic-model-v7"
  resume: False
  batch_size: 2
  max_train_steps: 200
  unet_learning_rate: 1e-4
  text_encoder_learning_rate: 1e-4
  seed: 1126
  weight_dtype: "torch.bfloat16"
  snr_gamma: 5
  lr_scheduler_name: "cosine_with_restarts"
  lr_warmup_steps: 100
  num_cycles: 3
generate:
  save_folder: # 图像保存路径默认为 root + train.dataset_name + '/inference'
  prompts_path: "./Datasets/prompts/validation_prompt.txt"
  num_inference_steps: 50
  guidance_scale: 7.5
```

**配置说明**

- **train**
  - `root`：项目的根路径，用于组织模型和输出文件。
  - `dataset_path`：数据集路径，包含图像和对应的文本描述。
  - `captions_folder`: 存放文本标注的路径，默认和 `dataset_path` 一致。
  - `model_path`：模型检查点路径，默认根据 `root` 和 `dataset_name` 自动生成。如果使用 `--no-train`，需要确保该路径存在已微调的模型。
  - `pretrained_model_name_or_path`：预训练的 Stable Diffusion 模型名称或本地路径。
  - `resume`: 是否从上一次训练中恢复，默认为否。
  - `batch_size`：训练批次大小。
  - `max_train_steps`：总训练步数。
  - `unet_learning_rate`：UNet 的学习率。
  - `text_encoder_learning_rate`：文本编码器的学习率。
  - `seed`：随机数种子，确保结果可复现。
  - `weight_dtype`：模型权重的数据类型，如 `"torch.bfloat16"`、`"torch.float32"` 等。
  - `snr_gamma`：信噪比 (SNR) 参数，用于调整训练过程中的损失计算。
  - `lr_scheduler_name`：学习率调度器的名称。
  - `lr_warmup_steps`：学习率预热步数。
  - `num_cycles`：学习率调度器的周期数量。
- **generate**
  - `save_folder`：生成的图像保存路径，默认为 `root + dataset_name + '/inference'`。
  - `prompts_path`：文本提示文件路径，每行一个提示。
  - `num_inference_steps`：生成图像时的推理步骤数。
  - `guidance_scale`：生成图像时的指导尺度。

### 注意事项

- **显存需求**：微调和生成过程对显存有一定要求。
- **数据集准备**：确保数据集中图像和对应的文本描述数量一致，且文件名对应，可以选择修改 `Text2ImageDataset` 类来适配特定格式的数据。

### 目录结构

在样例数据集上运行脚本后：

```
CodePlayground/
│
├── Datasets/                   # 数据集文件夹
│   ├── Brad/                   # 示例数据集文件夹（样例数据集中，文本描述与图片在同一个文件夹下）
│   │   ├── image_001.jpg       # 示例图片
│   │   ├── image_001.txt       # 示例图片的文本描述
│   │   ├── image_002.jpg
│   │   ├── image_002.txt
│   │   └── ...
│   └── prompts/                # 文本提示文件夹
│       ├── validation_prompt.txt # 生成图像时使用的提示
│
├── SD/                         # 默认输出路径
│   ├── Brad/                   # 使用的数据集名称，自动生成
│   │   ├── logs/               # 模型训练检查点
│   │   │   ├── checkpoint-last/ # 最后保存的微调模型
│   │   │   │   ├── unet/       # 微调后的 UNet 模型
│   │   │   │   ├── text_encoder/ # 微调后的文本编码器
│   │   │   ├── checkpoint-100/  # 中间检查点（步数命名）
│   │   │   │   ├── unet/
│   │   │   │   ├── text_encoder/
│   │   │   └── ...
│   │   ├── inference/          # 生成的图像文件夹
│   │   │   ├── generated_1.png # 示例生成图像
│   │   │   ├── generated_2.png
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
├── sd_lora.py                  # 微调和生成图像的主脚本
└── config.yaml                 # 配置文件
```


[^1]: [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset/data).

</details>

</details> <details> <summary> <strong>3. AI Chat</strong> </summary>

> [19a. 从加载到对话：使用 Transformers 本地运行量化 LLM 大模型（GPTQ & AWQ）](../Guide/19a.%20从加载到对话：使用%20Transformers%20本地运行量化%20LLM%20大模型（GPTQ%20%26%20AWQ）.md)
>
> [19b. 从加载到对话：使用 Llama-cpp-python 本地运行量化 LLM 大模型（GGUF）](../Guide/19b.%20从加载到对话：使用%20Llama-cpp-python%20本地运行量化%20LLM%20大模型（GGUF）.md)
>
> 根据 [AI Chat 依赖](#ai-chat-依赖)进行环境配置。

**[chat.py](./chat.py)** 是一个 LLM 对话工具，用于与量化的大模型（LLM）进行对话。支持 GPTQ、AWQ 和 GGUF 格式的模型加载与推理。

#### 功能

- **与 LLM 对话**：支持从模型路径加载不同格式的大语言模型，并根据配置与之进行交互。
- **配置管理**：现在支持初步的环境检测是否符合脚本运行条件。
- **聊天历史保存**：自动保存聊天记录并支持从历史记录中加载。

#### 快速使用

```bash
python chat.py <model_path>
```

替换 `<model_path>` 为 GPTQ、AWQ 或 GGUF 格式模型的路径，即可开始与模型进行交互。

以 [DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF) 为例，加载 Q5_K_L 量化版本：

```bash
python chat.py 'bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/*Q5_K_L.gguf' --remote
```

**注意，暂时仅支持拥有 `tokenizer.chat_template` 属性的模型进行正常对话，对于其他模型，需要自定义 [config.yaml](./config.yaml#L38) 中的 `custom_template` 参数。**

#### 使用方法

可以通过命令行运行 `chat.py`，并指定要加载的模型路径：

```bash
python chat.py <model_path> [--no_stream] [--max_length 512] [--io history.json] [其他可选参数]
```

- `model_path`：模型的名称或本地路径，可以是 GPTQ、AWQ 或 GGUF 格式的模型。
- `--no_stream`：禁用流式输出，模型会在生成完毕后一次性返回全部内容（不建议启用，默认流式输出）。
- `--max_length`：可选参数，生成文本的最大长度。
- `--io`：同时指定对话历史的输入和输出路径，避免重复配置。
- `--remote`：**仅适用于 GGUF 模型文件**，从 `<model_path>` 解析出 `repo_id` 和 `model_name` 进行远程模型文件的加载。
- 其他参数使用 `--help` 进行查看。

[config.yaml](./config.yaml#L35) 示例：

```yaml
chat:
  max_length: 512
  no_stream: False
  custom_template: |
    {{ bos_token }}
    {% for message in messages %}
        {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
            {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        {% endif %}
        
        {% if message['role'] == 'user' %}
            {{ '[INST] ' + message['content'] + ' [/INST]' }}
        {% elif message['role'] == 'assistant' %}
            {{ message['content'] + eos_token}}
        {% else %}
            {{ raise_exception('Only user and assistant roles are supported!') }}
        {% endif %}
    {% endfor %}
```

</details>


---

欢迎你随时在这个游乐场中探索更多脚本！