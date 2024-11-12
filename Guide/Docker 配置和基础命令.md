

## Docker 容器基本命令

> 快速介绍项目运行过程中可能用到的命令

### 镜像管理

#### 查看本地镜像

```bash
docker images
```

列出本地所有的 Docker 镜像，包括仓库名、标签、镜像 ID、创建时间和大小。

![image-20241112223609346](/Users/home/Downloads/agent/LLM-API-Guide-and-Demos/Guide/assets/image-20241112223609346.png)

#### 拉取镜像

```bash
docker pull <image_name>:<tag>
```

例如，拉取...：

```bash
docker pull pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
```

#### 删除镜像

```bash
docker rmi <image_id_or_name>
```

**注意：** 删除镜像前，确保没有容器正在使用它。

### 创建容器

以当前使用的命令为例：

```bash
docker run --gpus all -it pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
```

先来解释一下 `--gpus all` 和 `-it` 的作用：

- `--gpus all`：允许容器使用主机的所有 GPU 资源，对于 AI 项目，这基本是必选项。
- `-it`：这是两个参数的组合，`-i` 表示“交互式”（interactive），`-t` 表示为容器分配一个伪终端（pseudo-TTY）。**`-it` 组合使用**可以获得完整的交互式终端体验。

> [!tip]
>
> 使用 `docker run --help` 可以查看更多参数的用法。
>
> 如果在执行 Docker 命令时遇到权限问题，可以在命令前加上 `sudo`。

#### 挂载

如果你希望在容器中访问主机的文件，可以使用 `-v` 参数。

1. **卷挂载**

   ```bash
   docker run --gpus all -it -v my_volume:container_path pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
   ```

   - `my_volume`：Docker 卷的名称。
   - `container_path`：容器中的路径。

   这样，保存在该路径的数据在容器删除后仍会保存在 `my_volume` 中。

2. **绑定主机目录到容器中**

   ```bash
   docker run --gpus all -it -v host_path:container_path pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel
   ```

   将主机系统的目录  `host_path` 目录挂载到容器内的 `container_path`，所有更改都会直接反映在主机目录上。

请将 `host_path` 和 `container_path` 替换为实际的路径。

#### 在容器中启动 Jupyter Notebook

以 8888 号端口为例，绑定：

```bash
docker run --gpus all -it -p 8888:8888 pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
```

现在你可以在容器中启动 Jupyter Notebook，并且可以在本机访问。

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 停止容器

#### 在容器终端内

- 使用 `Ctrl+D` 或输入 `exit`：退出并**停止**容器。
- 使用 `Ctrl+P` 然后 `Ctrl+Q`：退出容器的终端（detach），但让容器继续运行。

#### 从主机停止容器

如果你想从主机停止正在运行的容器，可以使用：

```bash
docker stop <container_id_or_name>
```

替换 `<container_id_or_name>` 为容器的 ID 或名称。

### 复制文件

#### 从主机复制文件到容器

```bash
docker cp <host_path> <container_id_or_name>:<container_path>
```

#### 从容器复制文件到主机

```bash
docker cp <container_id_or_name>:<container_path> <host_path>
```

### 重新连接到已存在的容器

在使用一段时间后，你可能会发现每次使用 `docker run` 去“运行”容器时，之前所做的改变都没有保存。

**这是因为每次运行 `docker run` 创建了新的容器。**

要找回在容器中的更改，需要重新连接到之前创建的容器。

#### 查看所有容器

```bash
docker ps -a
```

- `docker ps`：默认只显示正在运行的容器。
- `-a`：显示所有容器，包括已停止的。

#### 启动已停止的容器

如果你的容器已停止，可以使用以下命令启动它：

```bash
docker start <container_id_or_name>
```

替换 `<container_id_or_name>` 为容器的 ID 或名称。

#### 重新连接到运行中的容器

要在运行中的容器中执行命令或获得交互式终端，可以使用 `docker exec`：

```bash
docker exec -it <container_id_or_name> /bin/bash
```

- `/bin/bash`：在容器内启动一个 Bash Shell。
- 在 `docker run` 命令末尾也可添加 `/bin/bash`。

### 命名容器

有没有什么方法可以指定名称呢？每次通过 `docker ps -a` 复制 `id` 太不优雅了。

#### 使用 `--name` 参数

在创建容器时，可以使用 `--name` 参数为容器指定一个名称。例如：

```bash
docker run --gpus all -it --name ai pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
```

容器被命名为 `ai`，以后可通过该名称管理容器，不需要记住容器的 ID。

运行 `docker ps -a`：

![image-20241112215358397](/Users/home/Downloads/agent/LLM-API-Guide-and-Demos/Guide/assets/image-20241112215358397.png)

#### 使用容器名称的命令示例

- **启动容器：**

  ```bash
  docker start ai
  ```

- **停止容器：**

  ```bash
  docker stop ai
  ```

- **重新连接到容器：**

  ```bash
  docker exec -it ai /bin/bash
  ```

### 删除容器

#### 删除指定的容器

如果想删除一个容器，可以使用 `docker rm` 命令：

```bash
docker rm <container_id_or_name>
```

例如，删除名为 `ai` 的容器：

```bash
docker rm ai
```

**注意：** 需要先停止容器才能删除。

#### 删除所有未使用的容器

我们可以使用以下命令来删除所有处于“已退出”状态的容器：

```bash
docker container prune
```

这将删除所有已停止的容器（请谨慎使用，因为删除后无法恢复，适用于刚安装 Docker “不小心”创建了一堆容器）。



## 参考链接



[How to Fix Docker’s No Space Left on Device Error](https://www.baeldung.com/linux/docker-fix-no-space-error)
