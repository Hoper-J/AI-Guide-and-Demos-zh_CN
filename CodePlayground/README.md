# CodePlayground

欢迎来到 **CodePlayground** 🎡，这是一个课程相关的脚本游乐场。你可以在这里玩转各种工具和脚本，享受学习和实验的乐趣。

​	注意⚠️，所有的脚本都是一个 Toy 的版本。

## 搭建场地

1. 克隆仓库：

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
   pip install torch torchvision torchaudio
   
   # conda
   conda install pytorch::pytorch torchvision torchaudio -c pytorch
   ```

   ### AI Summarizer 依赖

   a. **ffmpeg**（用于视频转音频）

   ```bash
   # Linux
   sudo apt-get install ffmpeg
   
   # Mac
   brew install ffmpeg
   ```

   b. **Python 库**

   ```python
   pip install openai-whisper openai pyyaml numpy librosa srt certifi
   ```

## 当前的玩具

### 1. AI Summarizer

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
   - 其他参数包括 `--model_name`、`--language`、`--temperature` 和 `--timestamped`，可根据需要调整。

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
- `output_dir`: 生成文件的默认保存目录。
- `api_key`: 你的 OpenAI API 密钥，可以通过命令行参数或配置文件指定。
- `api_base_url`: 默认使用阿里云大模型平台。


#### 注意事项

- **中间文件保留**：默认情况下，summarizer.py 会保留所有中间转换文件，如音频和字幕文件。如果你需要删除这些中间文件，可以在脚本中进行相应修改。
- **模型选择**：在 `model_name` 中选择 Whisper 模型时注意，模型越大对显存的占用越高，建议在显存充足的环境下使用。

---

欢迎你随时在这个游乐场中探索更多脚本！