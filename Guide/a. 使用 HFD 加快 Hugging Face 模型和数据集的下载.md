# 使用 HFD 加快 Hugging Face 模型和数据集的下载

> Hugging Face 提供了丰富的预训练模型和数据集，而且使用 Hugging Face 提供的 `from_pretrained()` 方法可以轻松加载它们，但是，模型和数据集文件通常体积庞大，用默认方法下载起来非常花时间。
>
> 本文将指导你如何使用 **HFD（Hugging Face Downloader）** 来高效地下载 Hugging Face 上的模型和数据集。HFD 是一个轻量级的下载工具，支持多线程下载和镜像加速。
>
> 如果你遇到了代理相关的 443 报错，可以滑到章末查看。

## 目录

- [准备工作](#准备工作)
  - [所需工具安装](#所需工具安装)
    - [安装 Git](#安装-git)
    - [安装 Wget 或 Curl](#安装-wget-或-curl)
    - [安装 Aria2c](#安装-aria2c)
  - [安装 Git LFS](#安装-git-lfs)
    - [Linux](#linux)
    - [macOS](#macos)
    - [Windows](#windows)
  - [安装 HFD](#安装-hfd)
    - [下载 HFD](#下载-hfd)
    - [执行权限](#执行权限)
- [配置环境变量](#配置环境变量)
  - [Linux](#linux-1)
  - [Windows PowerShell](#windows-powershell)
- [使用 HFD 下载模型](#使用-hfd-下载模型)
  - [下载 GPT-2 模型](#下载-gpt-2-模型)
    - [参数说明](#参数说明)
    - [导入模型](#导入模型)
- [使用 HFD 下载数据集](#使用-hfd-下载数据集)
  - [下载 WikiText 数据集](#下载-wikitext-数据集)
  - [参数说明](#参数说明-1)
- [可能存在的问题（443 和 git clone failed）](#可能存在的问题443-和-git-clone-failed)
  - [取消代理](#取消代理)
  - [重新设置代理](#重新设置代理)

- [参考链接](#参考链接)

## 准备工作

在开始之前，请确保你的系统已经安装了以下工具（如果安装可以跳过下面的安装命令）：

- **Git**：版本控制系统，用于管理代码和大文件。
- **Wget** 或 **Curl**：用于下载脚本和文件。
- **Aria2c**（可选）：一个支持多线程下载的下载工具，可以进一步提升下载速度。

### 所需工具安装

#### 1. **安装 Git**

首先，你需要安装 Git 版本控制系统。如果你的系统还没有安装 Git，可以通过以下命令进行安装：

- **Linux (Ubuntu)**：

  ```bash
  sudo apt-get update
  sudo apt-get install git
  ```

- **macOS**：

  ```bash
  brew install git
  ```

- **Windows**：

  从 [Git for Windows](https://gitforwindows.org/) 下载并安装。

#### 2. **安装 Wget 或 Curl**

`HFD` 脚本依赖于 `wget` 或 `curl` 来下载资源，确保你至少安装了其中之一：

- **Linux (Ubuntu)**：

  ```bash
  sudo apt-get install wget curl
  ```

- **macOS**：

  ```bash
  brew install wget curl
  ```

- **Windows**：

  从 [Wget for Windows](https://eternallybored.org/misc/wget/) 或 [Curl 官方网站](https://curl.se/windows/) 下载并安装。

#### 3. **安装 Aria2c**

为了使用多线程下载提升速度，推荐安装 `aria2c` 下载工具：

- **Linux (Ubuntu)**：

  ```bash
  sudo apt-get install aria2
  ```

- **macOS**：

  ```bash
  brew install aria2
  ```

- **Windows**：

  从 [Aria2 官方网站](https://aria2.github.io/) 下载并安装。

### 安装 Git LFS

Git LFS 用于处理和管理大文件，确保你能够顺利下载 Hugging Face 上的模型和数据集。

#### Linux

安装 Git LFS，这里以 Ubuntu 为例：

```bash
sudo apt-get update
sudo apt-get install git-lfs
```

安装完成后，初始化 Git LFS：

```bash
git lfs install
```

#### macOS

使用 Homebrew 安装 Git LFS：

```bash
brew install git-lfs
git lfs install
```

#### Windows

1. 下载并安装 [Git for Windows](https://gitforwindows.org/)。
2. 下载 Git LFS 安装程序：[Git LFS 官方下载页面](https://git-lfs.github.com/)。
3. 运行安装程序并初始化 Git LFS：

```powershell
git lfs install
```

### 安装 HFD

HFD 是一个用于加速 Hugging Face 资源下载的脚本工具。以下是安装和配置步骤。

#### 下载 HFD

使用 `wget` 下载 HFD 脚本：

```bash
wget https://hf-mirror.com/hfd/hfd.sh
```

如果你使用的是 `curl`，可以使用以下命令：

```bash
curl -O https://hf-mirror.com/hfd/hfd.sh
```

#### 执行权限

下载完成后，给脚本增加执行权限：

```bash
chmod a+x hfd.sh
```

## 配置环境变量

为了让 HFD 能够正确地使用镜像加速下载，你需要设置 `HF_ENDPOINT` 环境变量。根据你使用的操作系统，设置方法有所不同。

### Linux

在终端中运行以下命令：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

为了让环境变量在每次启动终端时自动生效，可以将上述命令添加到 `~/.bashrc` 或 `~/.zshrc` 文件中：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### Windows PowerShell

在 PowerShell 中运行以下命令：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

为了让环境变量在每次启动 PowerShell 时自动生效，可以将上述命令添加到 PowerShell 配置文件中（通常位于 `Documents\PowerShell\Microsoft.PowerShell_profile.ps1`）：

```powershell
Add-Content -Path $PROFILE -Value '$env:HF_ENDPOINT = "https://hf-mirror.com"'
```

## 使用 HFD 下载模型

HFD 提供了一种简便的方法来下载 Hugging Face 上的预训练模型。以下是下载 `gpt2` 模型的步骤。

### 下载 GPT-2 模型

在终端中运行以下命令：

```bash
./hfd.sh gpt2 --tool aria2c -x 4
```

#### 参数说明

- `gpt2`：要下载的模型名称，对应替换为你自己想下载的。
- `--tool aria2c`：指定使用 `aria2c` 作为下载工具，以支持多线程下载。
- `-x 4`：设置 `aria2c` 的最大连接数为 4，以加快下载速度，你可以设置得更高。

**运行（-x 16）：**

![image-20240918220106023](./assets/image-20240918220106023.png)

### 导入模型

假设下载完之后保存在当前目录的`gpt2`文件夹下，可以使用以下命令直接导入，注意 `AutoModelForCausalLM` 仅用于当前模型，你需要根据实际情况进行替换：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置模型和分词器的本地路径
model_path = "./gpt2"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 测试模型加载是否成功
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# 使用模型生成文本
outputs = model.generate(**inputs)

# 解码输出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
```

**如果下载的是量化模型**

如果你使用了 4-bit 或 8-bit 量化技术（如在 HFD 中使用 quantization_config），那么需要额外配置量化参数，可以使用 Hugging Face 的 bitsandbytes 库加载量化后的模型。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

model_path = "./gpt2"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 或 torch.bfloat16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'  # 使用的量化类型
)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 测试量化模型
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
```

## 使用 HFD 下载数据集

类似于下载模型，HFD 也支持下载 Hugging Face 上的各种数据集。以下是下载 `wikitext` 数据集的步骤。

### 下载 WikiText 数据集

在终端中运行以下命令：

```bash
./hfd.sh wikitext --dataset --tool aria2c -x 4
```

#### 参数说明

- `wikitext`：要下载的数据集名称，对应替换为你自己想下载的。
- `--dataset`：指定下载数据集。
- `--tool aria2c` 和 `-x 4`：同上，使用 `aria2c` 进行多线程下载。

## 可能存在的问题（443 和 git clone failed）

### 取消代理

443 报错一般是因为之前配置了代理，然后现在过期不可用了。

在命令行查看是否设置代理：

```bash
env | grep -i proxy
```

可能的输出：

```bash
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
all_proxy=socks5://127.0.0.1:7891
```

使用以下命令取消：

```bash
unset http_proxy                                 
unset https_proxy
unset all_proxy
```

取消代理之后仍然可能报对应端口的错误，然后`Git clone failed.`这有可能是因为你的 Git 之前配置了代理。

查看配置（如果是当前项目配置，去掉 --global）：

```bash
git config --global --list
```

可能的输出：

```bash
http.proxy=http://127.0.0.1:7890
https.proxy=http://127.0.0.1:7890
```

如果存在代理，对应取消：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

现在应该可以正常下载。

### 重新设置代理

如果你想重新设置代理，下面也给出对应的命令，假设 HTTP/HTTPS 端口号为 7890， SOCKS5 为 7891。

- 终端代理：

    ```bash
    export http_proxy=http://127.0.0.1:7890
    export https_proxy=http://127.0.0.1:7890
    export all_proxy=socks5://127.0.0.1:7891
    ```
    
- Git 代理：
    ```bash
    git config --global http.proxy http://127.0.0.1:7890
    git config --global https.proxy http://127.0.0.1:7890
    ```

## 参考链接

[HF-Mirror](https://hf-mirror.com)