# 命令行基础指令速查（Linux & Mac适用）

> 本文旨在帮助你快速了解和查阅常用的命令行指令，尤其是项目中可能用到的命令。
>
> 建议感到困惑时再查看，这里是一份速查表而非详细的入门教程。

## 目录

- [命令行操作指令](#命令行操作指令)
   - [查看命令手册](#查看命令手册)
   - [导航与目录操作](#导航与目录操作)
   - [文件操作](#文件操作)
   - [查看文件内容](#查看文件内容)
   - [文件权限与所有权](#文件权限与所有权)
   - [环境变量](#环境变量)
   - [系统信息](#系统信息)
   - [进程管理](#进程管理)
   - [网络操作](#网络操作)
   - [其他常用命令](#其他常用命令)
- [Apt-get 包管理工具](#apt-get-包管理工具)
   - [更新软件包列表](#更新软件包列表)
   - [升级已安装的软件包](#升级已安装的软件包)
   - [安装软件包](#安装软件包)
   - [移除软件包](#移除软件包)
   - [清理系统](#清理系统)
- [Conda](#conda)
   - [管理环境](#管理环境)
   - [管理包](#管理包)
- [Pip](#pip)
   - [管理包](#管理包-1)
- [Git](#git)
   - [仓库状态与更新](#仓库状态与更新)
   - [常用 Git 命令](#常用-git-命令)
   - [配置](#配置)

---

## 命令行操作指令

> 如果你想进一步学习 Linux，不用烦恼到底哪一本书入门合适，看这一本：《Linux命令行大全 2nd Edition 肖特斯》。

### 查看命令手册

几乎所有命令都可以通过 `man` 或 `--help` 查看帮助信息：

- `man ls`：查看 `ls` 命令的手册页。
- `ls --help`：显示 `ls` 命令的简易帮助信息。

如果你总是忘记指令，也可以使用 `man` 来查看对应的英文描述，例如：

- `ls`：list directory contents。

通过多次查看和练习，你将逐渐记住这些指令。

### 导航与目录操作

- **`pwd`**：显示当前工作目录的完整路径。

- **`cd`**：更改当前工作目录。
  - `cd /path/to/directory`：进入指定目录。
  - `cd ..`：返回上一级目录。
  - `cd ~`：返回用户主目录。
  - `cd -`：切换到上一个工作目录，适用于在两个目录间来回切换。

- **`ls`**：列出目录内容。
  - `ls`：列出当前目录的文件和文件夹。
  - `ls -l`：以长格式显示详细信息。常用别名：`ll`。
  - `ls -a`：显示所有文件，包括隐藏文件（以`.`开头）。
  - `ls -lh`：以人类可读格式显示文件大小（单位如 KB、MB）。

- **`mkdir`**：创建新目录。
  - `mkdir new_directory`：创建名为`new_directory`的目录。
  - `mkdir -p /path/to/directory`：递归创建目录。

### 文件操作

对文件夹的操作通常需要递归选项 `-r`。

- **`touch`**：创建新文件或更新文件的时间戳。
  - `touch new_file`：创建一个名为`new_file`的空文件。

- **`cp`**：复制文件/目录。
  - `cp source destination`：复制文件到目标路径。
  - `cp -r source_directory destination_directory`：递归复制目录及其内容。

- **`mv`**：移动或重命名文件/目录。
  - `mv source destination`：将文件从源路径移动到目标路径。
  - `mv old_name new_name`：重命名文件或目录。

- **`rm`**：删除文件/目录。
  - `rm file`：删除文件。
  - `rm -r directory`：递归删除目录及其内容。
  - `rm -rf directory`：强制递归删除目录及其内容，不提示确认。
  - **注意**：`rm`操作不可逆，不会将文件移至回收站，使用时需谨慎。

### 查看文件内容

- **`cat`**：连接并显示文件内容。
  - `cat file`：显示`file`的内容。
- **`less`**：分页查看文件内容，其中的指令同 vim。
  - `less file`：逐页查看`file`的内容。
- **`head`** 和 **`tail`**：查看文件的开头和结尾部分，`-n` 指定行数。
  - `head -n 10 file`：显示文件的前 10 行。
  - `tail -n 10 file`：显示文件的后 10 行。

### 文件权限与所有权

- **`chmod`**：更改文件权限。
  - `chmod 755 file`：将`file`的权限设置为755。
  - `chmod +x script.sh`：为`script.sh`添加可执行权限。

- **`chown`**：更改文件所有者，个人并不常用。
  - `chown user:group file`：将`file`的所有者更改为`user`，组更改为`group`。

### 环境变量

- **`export`**：设置环境变量，使用完只在当前的终端有效，重开就没了。
  
  - `export VAR=value`：设置环境变量`VAR`。
  - `echo $VAR`：查看环境变量`VAR`的值。
  
- **`env`**：显示当前所有环境变量。
  - `env`：列出所有环境变量。

- **永久设置环境变量**：编辑`~/.bashrc`或`~/.bash_profile`文件，如果你用的是zsh的话，则是 `~/.zshrc`。

  - 在文件末尾添加：`export VAR=value`。

  - 保存后，运行`source ~/.bashrc`使更改生效，或者重启命令行。

  - 如果不想开文件的话，在命令行执行：
    ```bash
    echo 'export VAR=value' >> ~/.bashrc
    source ~/.bashrc
    ```

### 系统信息

- **`uname`**：显示系统信息。
  - `uname -a`：显示所有系统信息。

- **`df`**：查看磁盘空间使用情况。
  - `df -h`：以人类可读的格式显示磁盘使用情况，在项目进行到后期的时候，你可能会遇到磁盘空间不足。

- **`du`**：查看文件或目录的大小。
  - `du -sh file_or_directory`：显示指定文件或目录的大小。
  - `du -h --max-depth=1 .`：显示当前目录下各文件和子目录的大小。

### 进程管理

- **`ps`**：显示当前进程列表。
  - `ps aux`：显示所有进程的详细信息。

- **`top`**：实时显示系统资源使用情况。

- **`kill`**：终止进程。
  - `kill PID`：发送`SIGTERM`信号终止进程。
  - `kill -9 PID`：发送`SIGKILL`信号强制终止进程。

- **`killall`**：终止指定名称的所有进程。
  - `killall process_name`：终止所有名为`process_name`的进程。

### 网络操作

- **`ssh`**：通过SSH登录远程主机。

  - `ssh user@hostname`：连接到远程主机。

- **`scp`**：安全复制文件。

  - `scp file user@hostname:/path`：将文件复制到远程主机。
  - `scp user@hostname:/path/file ./`：从远程主机复制文件到本地。

- **`curl`**：命令行下的HTTP请求工具，适合与 API 交互、发送表单等。

    - `curl http://example.com`：获取网页内容并输出到终端。
    - `curl -O http://example.com/file`：下载文件并保存到本地
- **`wget`**: 和 curl 功能类似，经常用来下载。
  - `wget http://example.com/file`：下载并保存文件。

- **`ping`**：测试网络连通性
    - `ping example.com`：检查到 `example.com` 的连通性。


### 其他常用命令

- **`grep`**：在文件中搜索特定文本。

  - `grep 'text' file`：在`file`中搜索包含`text`的行。
  - `grep -r 'text' directory`：在目录中递归搜索文本。

- **`find`**：查找文件或目录。

  - `find /path -name "filename"`：在指定路径下查找名为`filename`的文件。

- **`alias`**：为命令创建别名。

  - `alias ll='ls -alF'`：将`ll`设置为`ls -alF`。

- **`history`**：显示命令历史记录。

- **`sudo`**：以超级用户权限执行命令。

  - `sudo command`：以 root 权限执行`command`。

  **注意**：使用 `sudo` 需谨慎。

- **重定向和管道**（非常有用）

  - `command > file`：将命令的输出重定向到文件（覆盖）。
  - `command >> file`：将命令的输出追加到文件。
  - `command1 | command2`：将`command1`的输出作为`command2`的输入。

- **`whoami`**：显示当前用户。

- **`date`**：显示当前日期和时间。

---

## Apt-get 包管理工具

### 更新软件包列表

- **`sudo apt-get update`**：从软件源更新可用软件包的列表，确保系统使用最新的软件包信息。

### 升级已安装的软件包

- **`sudo apt-get upgrade`**：升级系统中所有已安装的软件包到新版本，但不会移除或添加包。

- **`sudo apt-get dist-upgrade`**：在执行 `upgrade` 的同时，处理依赖关系，允许安装或移除包以进行更全面的系统升级。

### 安装软件包

- **`sudo apt-get install package_name`**：安装指定的软件包 `package_name`。
  - 例如：`sudo apt-get install vim` 安装 Vim 文本编辑器。

### 移除软件包

- **`sudo apt-get remove package_name`**：卸载软件包，但保留配置文件。

- **`sudo apt-get purge package_name`**：卸载软件包并删除其配置文件，彻底移除包的所有痕迹。

### 清理系统

- **`sudo apt-get autoremove`**：自动移除不再需要的依赖包，这些包通常是跟随其他包安装但现在不再被使用。

- **`sudo apt-get clean`**：清理 `/var/cache/apt/archives/` 中已下载的软件包，释放硬盘空间。这些包缓存文件就跟曾经手机下载的安装包一样，安装完了就可以选择删掉，不影响已安装的软件包。

- **`sudo apt-get autoclean`**：只删除已过期的包。



---

## Conda

### 管理环境

- **创建新环境**
  - `conda create -n env_name python=3.8`：创建名为`env_name`、Python版本为3.8的环境。

- **激活环境**
  - `conda activate env_name`：激活`env_name`环境。

- **退出环境**
  - `conda deactivate`：退出当前激活的环境。

- **列出所有环境**
  - `conda env list`或`conda info --envs`：显示所有可用环境。

- **删除环境**
  - `conda remove -n env_name --all`：删除名为`env_name`的环境。
- **导出环境**
  - `conda env export > environment.yml`：将当前环境导出到 `environment.yml` 文件。
- **从文件创建环境**
  - `conda env create -f environment.yml`：根据 `environment.yml` 创建新环境。

### 管理包

- **安装包**
  - `conda install package_name`：在当前环境中安装`package_name`。

- **更新包**
  - `conda update package_name`：将`package_name`更新到最新版本。

- **移除包**
  - `conda remove package_name`：卸载`package_name`。

- **列出已安装包**
  - `conda list`：列出当前环境中已安装的包。

- **搜索包**
  - `conda search package_name`：搜索可用的包。
- **更新Conda**
  - `conda update conda`：更新`conda`本身。
  - `conda update --all`：更新当前环境中所有包，不过没事不需要更新。

---

## Pip

### 管理包

- **安装包**
  - `pip install package_name`：安装`package_name`。

- **升级包**
  - `pip install --upgrade package_name`：升级`package_name`到最新版本。

- **卸载包**
  - `pip uninstall package_name`：卸载`package_name`。

- **列出已安装包**
  - `pip list`：列出已安装的包。

- **安装特定版本的包**
  - `pip install package_name==1.2.3`：安装`package_name`的1.2.3版本。

- **从文件安装包**
  - `pip install -r requirements.txt`：根据`requirements.txt`文件安装包。

- **查看可用更新**
  - `pip list --outdated`：查看有哪些包有更新版本。

- **冻结当前环境的包**
  - `pip freeze > requirements.txt`：将当前环境的包及其版本输出到`requirements.txt`。
- **显示包信息**
  - `pip show package_name`：显示`package_name`的详细信息。

---

## Git

> 如果你想深入学习，这里是一些非常棒的资料：
>
> - [Pro Git](https://git-scm.com/book/zh/v2)：汉化版。
> - [Learn Git Branching](https://learngitbranching.js.org/?demo=&locale=zh_CN)：一个交互的学习平台。

### 仓库状态与更新

- **查看仓库状态**
  - `git status`：查看当前仓库状态，了解未提交的更改和未跟踪的文件。

- **查看远程更新**
  - `git fetch`：从远程仓库获取更新，但不合并到本地。
    - 之后可使用`git diff origin/main`查看与远程主分支的差异。

- **拉取更新**
  - `git pull`：获取远程仓库的更新并合并到当前分支。
    - 等同于`git fetch`加`git merge`。

### 常用 Git 命令

- **克隆仓库**
  - `git clone https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN.git`：将远程仓库克隆到本地。
- **添加更改**
  - `git add file`：将`file`的更改添加到暂存区。
  - `git add -u`：将当前仓库已跟踪文件的更改添加到暂存区。
- **提交更改**
  - `git commit -m "commit message"`：提交暂存区的更改并添加提交信息。
  - `git commit -a -m "commit message"`：跳过`git add`，直接提交所有已跟踪文件的更改。
- **查看差异**
  - `git diff`：查看工作目录和暂存区之间的差异。
  - `git diff --staged`：查看已暂存的更改与上次提交之间的差异。
  - `git diff HEAD..origin/master`：查看本地分支和远程 `master` 分支之间的差异（先执行 `git fetch origin`）。
- **撤销更改**
  - `git checkout -- file`：丢弃工作目录中对`file`的修改。
  - `git reset HEAD file`：取消暂存区中`file`的更改。
  - `git reset --hard`：重置当前分支到上次提交状态，丢弃所有未提交的更改。
    - **注意**：此操作不可逆。
- **推送更改**
  - `git push origin branch_name`：将本地分支推送到远程仓库。
- **查看提交日志**
  - `git log`：查看提交历史记录。
  - `git log --oneline`：以单行格式显示日志。
- **创建并切换分支**
  - `git branch new_branch`：创建`new_branch`分支。
  - `git checkout new_branch`：切换到`new_branch`分支，或许以后会被更轻量的 `switch` 替代。
  - `git checkout -b new_branch`：创建并切换到`new_branch`分支。
- **合并分支**
  - `git merge branch_name`：将`branch_name`合并到当前分支。
- **删除分支**
  - `git branch -d branch_name`：删除本地分支。
  - `git push origin --delete branch_name`：删除远程分支。

### 配置

- **设置全局用户名和邮箱**
  - `git config --global user.name "Your Name"`：设置用户名。
  - `git config --global user.email "you@example.com"`：设置邮箱。
- **查看配置信息**
  - `git config --list`：列出所有Git配置。



