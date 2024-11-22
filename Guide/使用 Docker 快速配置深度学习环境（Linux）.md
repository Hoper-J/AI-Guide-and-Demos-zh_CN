# 使用 Docker 快速配置深度学习环境（Linux）

> 深度学习环境的配置过于繁琐，所以我制作了两个基础的镜像，希望可以帮助你节省时间，根据需要选择其中一种进行安装，版本说明：
>
> - **base** 版本基于 `pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel`，默认 `python` 版本为 3.11.10，可以通过 `conda install python==版本号` 直接修改版本。
> - **dl** 版本在 **base** 基础上，额外安装了深度学习框架和常用工具，具体查看[安装清单](#安装)。
>
> 如果你已经配置好了Docker，只需两行命令即可完成深度学习环境的搭建。对于没有 Docker 的同学，也不用担心，本文将提供详细的安装指引，帮助你一步步完成环境配置（仅介绍命令行的配置和安装，图形界面的逻辑是一致的）。
>
> P.S. 命令在 Ubuntu 18.04/20.04/22.04 下可以顺利执行，其余系统可通过文内链接跳转安装。

## 目录

- [镜像介绍](#镜像介绍)
  - [安装清单](#安装清单)

- [快速配置环境（两行命令）](#快速配置环境两行命令)
  - [1. 获取镜像（三选一）](#1-获取镜像三选一)
    - [国内镜像版](#国内镜像版)
    - [🪜科学上网版（直连）](#科学上网版直连)
    - [本地（网盘下载）](#本地网盘下载)
  - [2. 运行容器](#2-运行容器)
- [安装 Docker Engine](#安装-docker-engine)
  - [卸载旧版本](#卸载旧版本)
  - [使用 apt 仓库安装](#使用-apt-仓库安装)
- [GPU 驱动安装](#gpu-驱动安装)
- [安装 NVIDIA Container Toolkit](#安装-nvidia-container-toolkit)
- [拉取并运行 PyTorch Docker 镜像](#拉取并运行-pytorch-docker-镜像)

## 镜像介绍

所有版本都预装了 `sudo`、`pip`、`conda`、`wget`、`curl` 和 `vim` 等常用工具，且已经配置好 `pip` 和 `conda` 的国内镜像源。同时，集成了 `zsh` 和一些实用的命令行插件（命令自动补全、语法高亮、以及目录跳转工具 `z`）。此外，已预装 `jupyter notebook` 和 `jupyter lab`，设置了其中的默认终端为 `zsh`，方便进行深度学习开发，并优化了容器内的中文显示，避免出现乱码问题。其中还预配置了 Hugging Face 的国内镜像地址。

**链接**：

- [quickstart](https://hub.docker.com/repository/docker/hoperj/quickstart/general)，位于 Docker Hub，对应于下方的 pull 命令。
- [百度云盘](https://pan.baidu.com/s/1RJDfc5ouTDeBFhOdbIAHNg?pwd=bdka)，直接下载对应的版本，跳过科学版的命令进行配置。

### 安装清单

<details> <summary> <strong>base</strong> </summary>
**基础环境**：

- python 3.11.10
- torch 2.5.1 + cuda 11.8 + cudnn 9

**Apt 安装**：

- `wget`、`curl`：命令行下载工具
- `vim`、`nano`：文本编辑器
- `git`：版本控制工具
- `git-lfs`：Git LFS（大文件存储）
- `zip`、`unzip`：文件压缩和解压工具
- `htop`：系统监控工具
- `tmux`、`screen`：会话管理工具
- `build-essential`：编译工具（如 `gcc`、`g++`）
- `iputils-ping`、`iproute2`、`net-tools`：网络工具（提供 `ping`、`ip`、`ifconfig`、`netstat` 等命令）
- `ssh`：远程连接工具
- `rsync`：文件同步工具
- `tree`：显示文件和目录树
- `lsof`：查看当前系统打开的文件
- `aria2`：多线程下载工具
- `libssl-dev`：OpenSSL 开发库

**pip 安装**：

- `jupyter notebook`、`jupyter lab`：交互式开发环境
- `virtualenv`：Python 虚拟环境管理工具，可以直接用 conda
- `tensorboard`：深度学习训练可视化工具
- `ipywidgets`：Jupyter 小部件库，用以正确显示进度条

**插件**：

- `zsh-autosuggestions`：命令自动补全
- `zsh-syntax-highlighting`：语法高亮
- `z`：快速跳转目录

</details>

<details> <summary> <strong>dl</strong> </summary>

**dl**（Deep Learning）版本在 **base** 基础上，额外安装了深度学习可能用到的基础工具和库：

**Apt 安装**：

- `ffmpeg`：音视频处理工具
- `libgl1-mesa-glx`：图形库依赖（解决一些深度学习框架图形相关问题）

**pip 安装**：

- **数据科学库**：
  - `numpy`、`scipy`：数值计算和科学计算
  - `pandas`：数据分析
  - `matplotlib`、`seaborn`：数据可视化
  - `scikit-learn`：机器学习工具
- **深度学习框架**：
  - `tensorflow`、`tensorflow-addons`：另一种流行的深度学习框架
  - `tf-keras`：Keras 接口的 TensorFlow 实现
- **NLP 相关库**：
  - `transformers`、`datasets`：Hugging Face 提供的 NLP 工具
  - `nltk`、`spacy`：自然语言处理工具

如果需要额外的库，可以通过以下命令手动安装：

```bash
pip install --timeout 120 <替换成库名>
```

这里 `--timeout 120` 设置了 120 秒的超时时间，确保在网络不佳的情况下仍然有足够的时间进行安装。如果不进行设置，在国内的环境下可能会遇到安装包因下载超时而失败的情况。

</details>

## 快速配置环境（两行命令）

> 如果遇到报错，查阅《[Docker 基础命令介绍和常见报错解决](../Guide/Docker%20基础命令介绍和常见报错解决.md#解决常见报错)》。

### 1. 获取镜像（三选一）

假设你已经安装并配置好了 Docker，那么只需两行命令即可完成深度学习的环境配置，以 **dl** 镜像为例，拉取：

#### 国内镜像版

```bash
sudo docker pull dockerpull.org/hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

#### 🪜科学上网版（直连）

```bash
sudo docker pull hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
```

> [!note]
>
> 如果镜像有更新版本，可通过 `docker pull` 拉取最新镜像。

#### 本地（网盘下载）

> 通过[百度云盘](https://pan.baidu.com/s/1RJDfc5ouTDeBFhOdbIAHNg?pwd=bdka)下载文件（阿里云盘不支持分享大的压缩文件）。
>
> 同名文件内容相同，`.tar.gz` 为压缩版本，下载后通过以下命令解压：
>
> ```bash
> gzip -d dl.tar.gz
> ```

假设 `dl.tar` 被下载到了 `~/Downloads` 中，那么切换至对应目录：

```bash
cd ~/Downloads
```

然后加载镜像：

```bash
sudo docker load -i dl.tar
```

### 2. 运行容器

```bash
sudo docker run --gpus all -it --name ai hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel /bin/zsh
```

如果需要使用 Jupyter，可以使用以下命令：

```bash
sudo docker run --gpus all -it --name ai -p 8888:8888 hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel /bin/zsh
```

> [!tip]
>
> **常用操作提前看**：
>
> - **启动容器**：`docker start <容器名>`
> - **运行容器**：`docker exec -it <容器名> /bin/zsh`
>   - **容器内退出**：`Ctrl + D` 或 `exit`。
> - **停止容器**：`docker stop <容器名>`
> - **删除容器**：`docker rm <容器名>`
> 
---

**如果还没有安装 Docker，继续阅读，可以根据实际情况通过目录快速跳转。**

## 安装 Docker Engine

> 对于图形界面来说，可以跳过下面的命令直接安装 Desktop 版本（其中会提供 Docker Engine），这是最简单的方法。根据系统访问：
>
> - [Linux](https://docs.docker.com/desktop/setup/install/linux/)
> - [Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
> - [Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
>
> 以下是命令行的安装命令，在 Ubuntu 上运行，其余系统参考[官方文档](https://docs.docker.com/engine/install)。

### 卸载旧版本

在安装 Docker Engine 之前，需要卸载所有有冲突的包，运行以下命令：

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

`apt-get` 可能会报告没有安装这些包，忽略即可。

注意，卸载 Docker 的时候，存储在 /var/lib/docker/ 中的镜像、容器、卷和网络不会被自动删除。如果你想从头开始全新安装，请阅读 [Uninstall Docker Engine 部分](https://docs.docker.com/engine/install/ubuntu/#uninstall-docker-engine)。

### 使用 `apt` 仓库安装

首次安装 Docker Engine 之前，需要设置 Docker 的 `apt` 仓库。

1. **设置 Docker 的 `apt` 仓库**

   ```bash
   # 添加 Docker 的官方 GPG 密钥：
   sudo apt-get update
   sudo apt-get install ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc
   
   # 将仓库添加到 Apt 源：
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   ```

   > [!note]
   >
   > 如果你使用的是 Ubuntu 的衍生发行版，例如 Linux Mint，可能需要使用 `UBUNTU_CODENAME` 而不是 `VERSION_CODENAME`。
   >
   > 如果 `sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc` 执行失败，可以尝试以下命令：
   >
   > ```bash
   >sudo wget -qO- https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc
   > ```


2. **安装 Docker 包**

   ```console
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

3. **通过运行 `hello-world` 镜像来验证安装是否成功**

   ```console
   sudo docker run hello-world
   ```

   这个命令会下载测试镜像并运行，如果你看到以下输出，那么恭喜你安装成功：
   
   ![image-20241113173220588](./assets/image-20241113173220588.png)

## GPU 驱动安装

如果需要使用 GPU 的话，先安装适用于你的系统的 NVIDIA GPU 驱动程序，访问任一链接进行：

- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Official Drivers](https://www.nvidia.com/en-us/drivers/)

这部分配置文章很多，偷个懒 :) 就不开新环境演示了，下面讲点可能不同的。

## 安装 NVIDIA Container Toolkit

> 为了在 Docker 容器中使用 GPU，需要安装 NVIDIA Container Toolkit。
>
> 注意，我们现在不再需要安装 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker?tab=readme-ov-file)，官方在 2023.10.20 指出其已被 [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) 所取代，过去的配置命令可能已不再适用。

以下命令使用 Apt 完成，Yum 等其他命令访问参考链接：[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)。

1. **设置仓库和 GPG 密钥**

   设置 NVIDIA 的软件源仓库和 GPG 密钥，确保我们可以从官方源安装 NVIDIA Container Toolkit。

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list。
   ```

2. **安装 NVIDIA Container Toolkit**

   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

3. **配置 Docker**

   使用 `nvidia-ctk` 工具将 NVIDIA 容器运行时配置为 Docker 的默认运行时。

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

4. **重启 Docker**

   ```bash
   sudo systemctl restart docker
   ```

## 拉取并运行深度学习 Docker 镜像

> 现在可以拉取深度学习（[dl](https://hub.docker.com/repository/docker/hoperj/quickstart/general)）镜像，命令和之前一致。

1. **拉取镜像**

   ```bash
   sudo docker pull hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
   ```

   ![image-20241115163216096](./assets/image-20241115163216096.png)

2. **运行镜像**

   ```bash
   sudo docker run --gpus all -it hoperj/quickstart:dl-torch2.5.1-cuda11.8-cudnn9-devel
   ```

3. **检查 GPU**

   在容器内运行：

   ```bash
   nvidia-smi
   ```

   如果正确显示代表成功。不过对于实际使用来说，还需要了解基础命令和报错的解决方法。使用 `Ctrl + D` 或者命令行输入 `exit` 并回车退出容器，继续阅读《[Docker 基础命令介绍和常见报错解决](../Guide/Docker%20基础命令介绍和常见报错解决.md)》。

