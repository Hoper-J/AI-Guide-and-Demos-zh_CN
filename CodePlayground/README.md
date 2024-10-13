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

## 当前的玩具

<details> <summary> <strong>1. AI Summarizer</strong> </summary>

> [15. 0 基础也能轻松实现 AI 视频摘要](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/15.%200%20基础也能轻松实现%20AI%20视频摘要.md)

**Summarizer** 是一个 AI 摘要工具，用于从视频或音频文件中提取字幕并生成视频摘要，也可以直接处理现有的字幕文件。它集成了 Whisper 模型和 OpenAI API 来自动化这些过程。

#### 功能

- **视频转音频**：使用 FFmpeg 将视频文件转换为 WAV 格式的音频文件。
- **音频转录**：使用 Whisper 模型将音频转录为文本字幕。
- **字幕生成**：生成 SRT 格式的字幕文件。
- **视频摘要**：使用 OpenAI 的模型生成视频内容的摘要。
- **配置管理**：支持从 `config.yaml` 文件中读取和保存配置。

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
   - 其他参数见[配置文件](#配置管理)或使用 `--help` 进行查看

   以上命令会从样例视频中提取音频，生成字幕并自动生成摘要。

   生成的文件默认会保存在 `./output` 文件夹下，包括：
   - **对应的音频文件**（MP3格式）
   - **转录生成的字幕文件**（SRT 格式）
   - **视频摘要文件**（TXT 格式）

#### 配置管理

脚本支持从 `config.yaml` 文件中读取默认配置，你可以通过编辑该文件来自定义参数，避免每次运行脚本时手动指定。

[配置文件](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/CodePlayground/config.yaml)示例：

   ```yaml
summarizer:
  model_name: "medium"
  language: "zh"
  whisper_temperature: 0.2
  llm_temperature: 0.2
  timestamped: false
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

</details> <details> <summary> <strong>2. AI Chat</strong> </summary>

> [19a. 从加载到对话：使用 Transformers 本地运行量化 LLM 大模型（GPTQ & AWQ）](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/19a.%20从加载到对话：使用%20Transformers%20本地运行量化%20LLM%20大模型（GPTQ%20%26%20AWQ）.md)
>
> [19b. 从加载到对话：使用 Llama-cpp-python 本地运行量化 LLM 大模型（GGUF）](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Guide/19b.%20从加载到对话：使用%20Llama-cpp-python%20本地运行量化%20LLM%20大模型（GGUF）.md)
>
> 建议阅读文章进行配置。

**Chat** 是一个 LLM 对话工具，用于与量化的大模型（LLM）进行对话。支持 GPTQ、AWQ 和 GGUF 格式的模型加载与推理。

#### 功能

- **与 LLM 对话**：支持从模型路径加载不同格式的大语言模型，并根据配置与之进行交互。
- **配置管理**：现在支持初步的环境检测是否符合脚本运行条件（待进一步测试）。
- **聊天历史保存**：自动保存聊天记录并支持从历史记录中加载。

#### 快速使用

```bash
python chat.py <model_path>
```

替换 `<model_path>` 为 GPTQ、AWQ 或 GGUF 格式模型的路径，即可开始与模型进行交互。

**注意，暂时仅支持拥有 `tokenizer.chat_template` 属性的模型进行正常对话，对于其他模型，需要自定义 [config.yaml](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/c29e7dc522fc34a897e4e9cff88fc6e0c1110139/CodePlayground/config.yaml#L14) 中的 `custom_template` 参数。**

运行脚本会严格检查所有的环境并给出安装指引，你可以注释 [setup_chat()](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/1f23368f5a3eaab865ccf9343445516a3d9ce671/CodePlayground/chat.py#L13) 对应的行来跳过这个行为（如果不需要加载 GPTQ 和 AWQ 的模型文件）。

#### 使用方法

你可以通过命令行运行 `chat.py`，并指定要加载的模型路径：

```bash
python chat.py <model_path> [--no_stream] [--max_length 512] [--io history.json] [其他可选参数]
```

- `model_path`：模型的名称或本地路径，可以是 GPTQ、AWQ 或 GGUF 格式的模型。
- `--no_stream`：禁用流式输出，模型会在生成完毕后一次性返回全部内容（不建议启用，默认流式输出）。
- `--max_length`：可选参数，生成文本的最大长度。
- `--io`：同时指定对话历史的输入和输出路径，避免重复配置。
- `--remote`：**仅适用于 GGUF 模型文件**，从 `<model_path>` 解析出 `repo_id` 和 `model_name` 进行远程模型文件的加载。
- 其他参数使用 `--help` 进行查看。

[配置文件](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/CodePlayground/config.yaml)示例：

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