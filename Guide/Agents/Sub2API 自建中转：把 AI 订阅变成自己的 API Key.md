# Sub2API 自建中转：把 AI 订阅变成自己的 API Key

> 这或许是一个必会的省钱手艺。为了文章发布后的“可读”性，本文仅针对**单一自用账户**进行完整配置指导，用于把订阅额度抽取为 API 接口，供学习调用、沉浸式翻译等个人使用场景，实现一定程度上的 Token 自由。
>
> 理论上来讲，只要一个客户端能自定义 `base_url` 和 `api_key`，就能接入这样中转出来的 API：Claude Code、Codex、Cherry Studio、沉浸式翻译、Zotero 插件、自己写的 Python 脚本...
>
> **实操前提**：拥有一个 ChatGPT 订阅账户。其余 AI 订阅也可以，只是在账号部分需要换一种方式认证。Claude 则完全不推荐在没有实操经验的情况下尝试。
>
> **写在前面**：文章没有提及到的部分可以询问 AI。

## 目录

- [背景](#背景)
   - [订阅对应的可用额度](#订阅对应的可用额度)
   - [Q：部署后能够得到什么？](#q部署后能够得到什么)
- [部署](#部署)
   - [（Windows）准备 WSL](#windows准备-wsl)
   - [安装 Docker](#安装-docker)
      - [Linux / WSL2](#linux--wsl2)
      - [macOS](#macos)
      - [Windows](#windows)
      - [验证安装](#验证安装)
   - [启动 Sub2API](#启动-sub2api)
   - [初始密码获取](#初始密码获取)
   - [登录](#登录)
   - [（可选）出口 IP 设置](#可选出口-ip-设置)
   - [创建分组](#创建分组)
   - [添加上游订阅账号](#添加上游订阅账号)
   - [创建 API Key](#创建-api-key)
- [接入客户端](#接入客户端)
   - [Claude Code](#claude-code)
   - [ChatGPT（Codex）](#chatgptcodex)
   - [OpenCode](#opencode)
   - [Pi](#pi)
   - [DeepSeek Harness](#deepseek-harness)
   - [（可选）写进 shell 配置](#可选写进-shell-配置)
      - [Linux / macOS / WSL2](#linux--macos--wsl2)
      - [Windows（PowerShell）](#windowspowershell)
   - [沉浸式翻译](#沉浸式翻译)
   - [其他客户端](#其他客户端)
- [拓展：让其他设备也能访问](#拓展让其他设备也能访问)
   - [服务器要求](#服务器要求)
   - [（可选）通过 AWS 赠送的 200$ 代金券获取练手的服务器](#可选通过-aws-赠送的-200-代金券获取练手的服务器)
   - [在服务器上部署](#在服务器上部署)
   - [（可选）域名 + Cloudflare](#可选域名--cloudflare)
      - [域名购买](#域名购买)
      - [把域名接入 Cloudflare](#把域名接入-cloudflare)
      - [把域名指向你的服务器](#把域名指向你的服务器)
      - [让用户无感访问非标准端口](#让用户无感访问非标准端口)
      - [SSL/TLS 模式](#ssltls-模式)
- [附录](#附录)
   - [Sub2API 部署 Prompt](#sub2api-部署-prompt)
   - [Cloudflare Tunnel 配置 Prompt](#cloudflare-tunnel-配置-prompt)

## 背景

[Sub2API](https://github.com/Wei-Shaw/sub2api)，顾名思义：Subscription to API，把订阅转换为 API。现在的 AI 客户端一般都支持 BYOK（Bring Your Own Key，自带密钥），允许接入自定义的模型服务。但 API 是按量付费的，直接充值调用堪称花钱如流水。与此相对的是订阅制，官方提供的额度远超用户订阅价格能买到的用量，完全是在发福利。可订阅本身并不会给我们一个 API Key，只能在官方客户端交互。

早期 Sub2API 并没有像对待 Claude 那样对 GPT 的请求做精细的伪装，所以能被转换为 API 很大程度上是因为官方“心善”。甚至可以说 OpenAI 没有对 Sub2API 以及第三方调用实施风控，这一点从上半年对于 OpenClaw 的态度就可以看出来：Anthropic [明令禁止](https://x.com/bcherny/status/2040206440556826908)订阅认证被第三方工具调用（不过后来 Anthropic 在 Fable 5 反复上下线和被其他模型冲击的时期逐步松口：从一开始[单独划一份 Agent SDK 额度](https://x.com/ClaudeDevs/status/2054610152817619388)到生效当天[暂停](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)），而 OpenAI 却[官宣欢迎](https://x.com/sama/status/2050357911915028689)。

本文将基于该开源项目和 ChatGPT 订阅服务，在 Windows / macOS / Linux 三种设备上尽可能完整地演示所有部署流程，将账户订阅转为 API Key，从而实现一定程度的跨平台 token 自由，甚至可选地进一步为亲朋好友提供 AI 服务，成为最“靓”的崽。

### 订阅对应的可用额度

首先了解实际的用量。文章的实操部分只会演示 GPT，不过刚好有两个平台的账户实测数据，所以这里把 Claude 也一起列出来做参照：当前 Claude 200\$ 提供的月用量在 13000\$ 左右（数据片面），GPT 月用量存在削减，测试在 9000\$ 左右（数据为 7 月多账户周限额的下限 × (4 + 2/7)）。

> [!note]
>
> 记得在 6 月测试的时候，个人账号的 Claude 的月用量上限在 6800\$ 左右，而 5 - 8 月明明都在官方声明[^1]的 50% 额外用量的时间内，所以这么大的估算差异，只能暂时不严谨地归因于个人使用习惯带来的波动了：在这期间，由于模型自身能力提升，我的使用方式从「superpowers + grill 辅助」变成了「仅 grill + 一些自定义的快捷 skill」，少了频繁使用 subagents 来回 review 的请求。

那么，如果我们真能把额度用完，每百万输入/输出 token（MTok）的实际价格是多少？

**以 Opus 5、Fable 5.1、GPT-6 Astra 和 GPT-5.6 的官方定价为例**：

| 模型          | 订阅月费 | 等价 API 用量 | 折扣系数 | 官方 API 定价（输入 / 输出 MTok） | 订阅等效定价（输入 / 输出 MTok） |
| ------------- | -------- | ------------- | -------- | --------------------------------- | -------------------------------- |
| Opus 5        | \$200    | \$13,000      | 1.54%    | \$5 / \$25                        | \$0.077 / \$0.385                |
| Fable 5.1     | \$200    | \$6,500       | 3.08%    | \$10 / \$50                       | \$0.308 / \$1.538                |
| GPT-6 Astra   | \$200    | \$9,000       | 2.22%    | \$10 / \$50                       | \$0.222 / \$1.111                |
| GPT-5.6 Sol   | \$200    | \$9,000       | 2.22%    | \$4 / \$20                        | \$0.089 / \$0.444                |
| GPT-5.6 Terra | \$200    | \$9,000       | 2.22%    | \$2 / \$12                        | \$0.044 / \$0.267                |
| GPT-5.6 Luna  | \$200    | \$9,000       | 2.22%    | \$0.20 / \$1.20                   | \$0.004 / \$0.027                |

除 Fable 5.1（约 0.3 折，因为 Fable 在 Max 上只能用到一半额度）外，其余模型折扣都在 0.25 折以下，Opus 5 更是低到 0.15 折，所以即便不转换为 API 使用，也是非常实惠的。按 1 美元 ≈ 6.8 元换算（200\$ 的订阅折合 1360 元）：**1 元 ≈ 9.56 美元的 Claude API 额度 ≈ 6.62 美元的 GPT API 额度**。用 token 展示或许更直观，按非缓存的价格，在能够用完周额度的情况下，订阅的 1 元大约等价于：

| 模型          | 输入 token | 输出 token |
| ------------- | ---------- | ---------- |
| Opus 5        | 191 万     | 38 万      |
| Fable 5.1     | 48 万      | 10 万      |
| GPT-6 Astra   | 66 万      | 13 万      |
| GPT-5.6 Sol   | 165 万     | 33 万      |
| GPT-5.6 Terra | 331 万     | 55 万      |
| GPT-5.6 Luna  | 3309 万    | 551 万     |

如果再把 [Tibo](https://x.com/thsottiaux) 时不时按下的用量重置算上（下图由 gpt-image-2 生成）：

<img src="assets/saint-tibo-usage-reset.jpg" alt="Saint Tibo" style="zoom: 33%;" />

最近几个月 GPT 订阅的折扣说是 0.1 折以下也不为过。

> [!tip]
>
> 1. 官方标识 200\$ 订阅是 20\$ 的 20 倍用量，折算成 20\$/100\$ 订阅的费率会和当前折扣有出入。
> 2. 目前两家的订阅用量相比去年都有所衰减，且用且珍惜。
> 3. 以上价格计算仅为理想自用情况，如果考虑到服务器等部署场景，成本会有所提升。
> 4. Claude 的 +50% 用量已延长到 2026-09-13[^1]，9 月 14 日起改为永久 +25%[^2]，也就是从 150% 降到 125%，届时用量约为当前的 83%（初步折算 Opus 5 约 10800\$，Fable 5.1 约 5400\$）。
> 5. GPT-5.6 Sol 目前是促销价，官方表示该定价至少持续到 2026-11-21[^3]。
> 6. **不要为了用完所有额度打乱自己的生活作息**。

[^1]: [Claude Code May–August 2026 weekly limits promotion](https://support.claude.com/en/articles/15910845-claude-code-may-august-2026-weekly-limits-promotion)
[^2]: [@ClaudeDevs 2026-08-29 的公告](https://x.com/ClaudeDevs/status/2093742321473065266)："Starting September 14, we're permanently raising standard weekly limits in Claude Code by 25% for Pro, Max, Team, and seat-based Enterprise plans. Until then, the current 50% increase will be in place." 随后的[补充帖](https://x.com/ClaudeDevs/status/2093742322525810912)承认 "Compared to today, this works out to a 17% reduction in weekly limits on Claude Code."
[^3]: [OpenAI API Pricing](https://developers.openai.com/api/docs/pricing)，页面原文："GPT-5.6 Sol's promotional pricing is available at least through November 21, 2026."

### Q：部署后能够得到什么？

支持 **OpenAI 和 Anthropic 请求格式**、可用于**任何地方**的、属于你自己的 API Key，仅此足矣。

## 部署

这里默认本机为演示环境。

后续模块将用 `<xxx>` 表示那些需要根据实际情况替换的值，比如 `<你的域名>`、`<你的服务器 IP>`、`<你的 API Key>`。截图中演示的 `goodgoodstudy.ai` 请对应理解为 `<你的域名>`。

### （Windows）准备 WSL

> macOS 和 Linux 用户可以跳过这一节。

打开菜单搜索 `powershell`，以管理员身份运行：

![以管理员身份运行 PowerShell](assets/1565.PNG)

执行以下命令：

```powershell
systeminfo | Select-String -Pattern "固件中已启用虚拟化|Virtualization Enabled In Firmware"
```

![固件中已启用虚拟化：是](assets/1586.PNG)

显示「是」就继续往下，显示「否」需要重启进 BIOS 打开（这部分需要结合 AI 和自身的主板型号搜索 BIOS 配置）。

安装 WSL：

```powershell
wsl --install
```

![wsl --install 的输出](assets/1587.PNG)

用下面的命令把网络模式设置为镜像，让 WSL 直接共享主机的网络：

```powershell
@"
[wsl2]
networkingMode=mirrored
"@ | Set-Content -Path "$env:USERPROFILE\.wslconfig" -Encoding utf8

Get-Content "$env:USERPROFILE\.wslconfig"
```

**重启电脑**，然后执行以下命令：

```powershell
wsl --install -d Ubuntu
```

装完会要求设置 **UNIX 用户名和密码**（这是新的用户身份，和当前电脑的登录身份无关）。

![设置 UNIX 用户名和密码](assets/1593.PNG)

确认以下命令输出的版本是 2：

```powershell
wsl -l -v
```

然后直接输入 `wsl` 并回车，进入 WSL，用下面的命令开启 systemd（需要输入密码）：

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true
EOF
cat /etc/wsl.conf
```

如图所示：

![在 WSL 中开启 systemd](assets/1597.PNG)

新开一个 PowerShell 或者直接 `exit` 退出 WSL，再执行下面的命令：

```powershell
wsl --shutdown
wsl -- systemctl is-system-running
```

返回 `running` 或 `degraded` 都代表配置生效（`degraded` 只是有部分服务没起来，不影响后面的步骤）。

### 安装 Docker

已经装过的可以跳过，按平台三选一。

#### Linux / WSL2

用官方脚本：

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

装完重新登录打开终端（也可以在 WSL 输入 `exit` 后回车重新 `wsl`），让 docker 用户组生效。

> [!note]
>
> 在 WSL 里跑这个脚本会弹一段提示，“建议改用 Docker Desktop”：
>
> ```
> WSL DETECTED: We recommend using Docker Desktop for Windows.
> ```
>
> 这里等 20 秒会继续装。
>
> 注意，apt 在后台自动跑更新的时候，执行官方脚本装 Docker 会报错：
>
> ```
> E: Could not get lock /var/lib/apt/lists/lock. It is held by process 506 (apt-get)
> ```
>
> 等它跑完再重试即可：
>
> ```bash
> while sudo fuser /var/lib/apt/lists/lock /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
>   echo "后台 apt 仍在运行，等待中..."; sleep 5
> done
> ```

（可选）配置镜像加速：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{
  "registry-mirrors": [
    "https://dockerproxy.net",
    "https://docker.1panel.live"
  ]
}
EOF
sudo systemctl restart docker
docker info | grep -A2 -i "registry mirrors"
```

#### macOS

装 [Docker Desktop](https://www.docker.com/products/docker-desktop)：

![Docker Desktop 下载页](assets/image-20260824011920888.png)

#### Windows

可选装 [Docker Desktop](https://www.docker.com/products/docker-desktop)（或者用之前 [Linux / WSL2](#linux--wsl2) 的命令进行安装，二选一）。选 Desktop 的话按默认配置直接安装（勾选「Use WSL 2 instead of Hyper-V」）：

![安装时勾选 Use WSL 2 instead of Hyper-V](assets/1601.PNG)

然后打开 Docker Desktop → 右上角齿轮 Settings → Resources → WSL Integration，把图示开关打开，点 Apply & Restart：

![在 WSL Integration 里打开 Ubuntu](assets/1602.PNG)

（可选）在 Settings → Docker Engine 里配置镜像加速：

```json
{
     "registry-mirrors": [
          "https://dockerproxy.net",
          "https://docker.1panel.live"
        ]
}
```

#### 验证安装

不论用哪种方式，装完都验证一下（Windows 需要先 `exit` 退出 WSL，再重新用 `wsl` 进入，让 WSL Integration 生效）：

```bash
docker --version && docker compose version
docker run --rm hello-world
```

> [!note]
>
> 如果 `docker run --rm hello-world` 报错：
>
> ```
> docker: error getting credentials - err: exec: "docker-credential-desktop.exe": executable file not found in $PATH
> ```
>
> 这个报错是因为 Desktop 在 WSL 里写的 `~/.docker/config.json` 指定了 Windows 侧的 `exe`，而 WSL 的 `$PATH` 里找不到它。
>
> **临时解决方法**：可以使用以下命令把凭证配置清掉再执行：
>
> ```bash
> echo '{}' > ~/.docker/config.json
> docker run --rm hello-world
> ```
>
> 这会让 `docker login` 不能登录私有仓库，但对本文场景没有影响。
>
> ![清空凭证配置后成功拉取镜像](assets/image-20260824021357829.png)

### 启动 Sub2API

拉取官方的部署脚本并执行：

```bash
mkdir -p ~/sub2api && cd ~/sub2api
curl -fsSL -o docker-deploy.sh \
  https://raw.githubusercontent.com/Wei-Shaw/sub2api/main/deploy/docker-deploy.sh
bash docker-deploy.sh
```

输出：

![部署脚本执行完成的输出](assets/image-20260823193130194.png)

打开 `.env` 文件，根据自己的实际情况修改几个值并**保存**（macOS 使用 Command + Shift + `.` 可以在 Finder 中显示隐藏文件）：

```bash
BIND_HOST=0.0.0.0       # 默认是 0.0.0.0，允许外部访问，如果只想本机访问，需要改为 127.0.0.1
SERVER_PORT=8081        # 默认 8080，演示使用 8081 端口
TZ=Asia/Singapore       # 默认 TZ=Asia/Shanghai（上海时区），演示使用新加坡时区

ADMIN_EMAIL=admin@sub2api.local   # 管理员邮箱，可改成你自己的
ADMIN_PASSWORD=                   # 留空 = 首次启动时随机生成
```

改完保存，执行命令启动服务：

```bash
cd ~/sub2api && docker compose up -d
```

顺利的话可以看到以下输出：

![docker compose up -d 的输出](assets/image-20260823201846812.png)

然后在浏览器打开 [http://127.0.0.1:8081](http://127.0.0.1:8081)，就能看到登录界面了（文章和截图统一以 `8081` 端口为例进行演示，如果填了别的 `SERVER_PORT` 需要对应替换）：

![浏览器打开后的登录界面](assets/image-20260823203830959.png)

> [!tip]
>
> **局域网访问**
>
> 执行命令查看 IP，macOS：
>
> ```bash
> ipconfig getifaddr $(route -n get default | awk '/interface:/{print $2}')
> ```
>
> Linux：
>
> ```bash
> ip route get 1 | grep -oP 'src \K[\d.]+'
> ```
>
> 假设输出为：
>
> ```
> 192.168.2.144
> ```
>
> 同网络下的其他设备可以通过 `http://192.168.2.144:8081` 进行访问：
>
> ![手机经局域网 IP 访问同一个服务](assets/IMG_3255.PNG)
>
> **公网访问**
>
> 执行命令查看 IP：
>
> ```bash
> curl -s icanhazip.com
>
> # 如果没输出就换下面的命令，也可以直接去云服务商控制台
> # curl -s ipinfo.io/ip
> ```
>
> 假设输出为：
>
> ```
> 1.2.3.4
> ```
>
> 所有设备都可以通过 `http://1.2.3.4:8081` 进行访问。

### 初始密码获取

如果 `.env` 中的 `ADMIN_PASSWORD` 没有设置，密码就会随机生成（在容器首次启动时输出到日志），命令行执行：

```bash
docker logs sub2api 2>&1 | grep -iE "admin password|one-time"
```

输出类似于：

```
Generated admin password (one-time): eb850b210f11bdd0fa123ff010d99104
```

其中的 `eb850b210f11bdd0fa123ff010d99104` 就是 admin 的密码，复制它。

### 登录

点击右上角的「登录」后输入账户和密码（默认的邮箱为 `admin@sub2api.local`）：

![输入账户和密码登录](assets/image-20260823205025426.png)

> [!tip]
>
> **密码重置（适用于忘记密码无法登录）**
>
> 生成一个 bcrypt 哈希写进数据库。把第一行换成你的新密码（保留单引号），`.env` 里改过 `ADMIN_EMAIL` 的话第二行同步，然后整段执行：
>
> ```bash
> NEW_PASSWORD='你的新密码'
> ADMIN_EMAIL='admin@sub2api.local'
> HASH=$(docker run --rm httpd:alpine htpasswd -bnBC 10 "" "$NEW_PASSWORD" | sed 's/^:\$2y/$2a/')
> docker exec sub2api-postgres psql -U sub2api -d sub2api \
>   -c "UPDATE users SET password_hash = '$HASH' WHERE email = '$ADMIN_EMAIL';"
> ```
>
> 输出 `UPDATE 1` 代表成功（`UPDATE 0` 说明邮箱没对上），然后直接用新密码登录就行。

在开始之前需要签署合规承诺：

![签署合规承诺](assets/image-20260823211843292.png)

确认并继续后，我们就得到了这样的控制台：

![登录后的控制台](assets/image-20260823212211491.png)

### （可选）出口 IP 设置

> 后续授权用的浏览器最好和部署 Sub2API 的机器保持相同的网络环境，但如果本身就处于国外，则此步可以跳过。

左侧栏「IP管理」→「添加代理」：

![IP 管理中添加代理](assets/image-20260823213016590.png)

假设本机的代理监听在宿主机的 `7890` 端口，按平台填：

| 部署环境                          | 代理地址                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| macOS / Windows（Docker Desktop） | `http://host.docker.internal:7890`                           |
| Linux                             | `http://<IP>:7890`（这里 `<IP>` 替换成 `ip -4 addr show docker0` 的输出） |

Linux 上也能用 `host.docker.internal`，需要先在 `docker-compose.yml` 里给 `sub2api` 服务额外配置 `extra_hosts: ["host.docker.internal:host-gateway"]` 再重建容器。不过这个地址的实际映射就是 `ip -4 addr show docker0` 的输出，所以直接执行 `ip -4 addr show docker0` 填对应 IP 更方便。

![代理配置表单](assets/image-20260823214942470.png)

另外记得让代理软件**允许来自局域网的连接**。

> [!note]
>
> 这里不能填 `127.0.0.1:7890`。因为 Sub2API 跑在 Docker 容器里，容器内的 `127.0.0.1` 指的是容器自己。

### 创建分组

需要先创建分组才能给后续的密钥分配账户，「分组管理」→「创建分组」：

![分组管理里创建分组](assets/image-20260823224920230.png)

**基础配置**：

1. 设置一个名称比如：`OpenAI`

   ![给分组设置名称](assets/image-20260823225851026.png)

2. 将平台切换为 `OpenAI`：

   ![把平台切换为 OpenAI](assets/image-20260823230010645.png)

3. 计费类型改为「订阅」（使用「余额」的话需要去「用户管理」界面在「余额」列点击充值才能正常使用），具体额度根据实际情况进行设置，也可以不设置：

   ![计费类型改为订阅](assets/image-20260823225341795.png)

   但注意，**设置为订阅后需要去左边栏的「订阅管理」部分分配订阅才能正常使用**：

   ![在订阅管理里分配订阅](assets/image-20260823234835621.png)

4. 如果需要使用 `gpt-image-2` 这样的生图模型，在「图片生成计费」部分勾选「允许当前分组生图」：

   ![勾选允许当前分组生图](assets/image-20260823230037552.png)

   > [!note]
   >
   > 目前网页版和 Codex 经测试均只能生成 1.5K 的图片。

5. 如果想通过 Claude Code 直接调用，在「OpenAI Messages 调度配置」部分启用「允许 /v1/messages 调度」：

   ![启用 /v1/messages 调度](assets/image-20260823225709082.png)

   上图的配置会让 claude-* 的模型被映射为对应的 GPT 模型，从而不需要用户更改现有的 model_id。

### 添加上游订阅账号

「账号管理」→「添加账号」：

![账号管理里添加账号](assets/image-20260823222034887.png)

选择对应的平台（Claude / OpenAI / Gemini / Grok / Antigravity / ...），这里以 OpenAI 为例，点击「OpenAI」→「OAuth」：

![选择 OpenAI 的 OAuth 方式](assets/image-20260823222129949.png)

下滑到「代理」部分，选择刚刚添加的 IP：

![在代理部分选择刚添加的 IP](assets/image-20260823222228365.png)

在分组部分选择刚才创建的分组：

![选择刚创建的分组](assets/image-20260823233316204.png)

点击「下一步」，然后点击「生成授权链接」后复制链接：

![生成并复制授权链接](assets/image-20260823222427353.png)

到浏览器打开，正常进行登录：

![浏览器中登录订阅账号](assets/image-20260823222644149.png)

选择后「继续」：

![选择账号后继续](assets/image-20260823222747407.png)

这时候会看到「**localhost** 拒绝了我们的连接请求」，不用在意它，直接复制 URL：

![回调页面提示 localhost 拒绝连接，直接复制 URL](assets/image-20260823223151390.png)

回到之前的界面直接粘贴后点击「完成授权」：

![粘贴回调 URL 完成授权](assets/image-20260823223241579.png)

这样就添加好了，可以通过「...」→「测试连接」：

![通过测试连接验证绑定](assets/image-20260823233550134.png)

切换到想测试的模型，比如 `GPT-5.6 Sol`，然后点击「开始测试」：

![选择模型后开始测试](assets/image-20260823223509983.png)

显示「测试完成」则代表可以正常拿到响应。

### 创建 API Key

左边栏「我的账户」→「API 密钥」→「创建密钥」：

![我的账户里创建 API 密钥](assets/image-20260823224058683.png)

「分组」选择刚才创建的订阅：

![选择刚创建的订阅](assets/image-20260823235002308.png)

> [!note]
>
> 如果需要进行拼车，建议**开启速率限制**，根据实际人数以 2100\$/周的基础数值进行 5h/日/周额度的配置，下面是一个 5 人拼车的额度示范：
>
> ![5 人拼车的速率限制配置示范](assets/image-20260824000305649.png)
>
> **注意**：官方只回调了 Plus 的 5h 限制，所以可以暂时不启用日限额，这一点也是选择 Pro 拼车的优势。

点击「使用密钥」：

![点击使用密钥](assets/image-20260823235112924.png)

直接切换到 `Claude Code` 或 `OpenCode` 界面，复制 `BASE_URL` 和 `AUTH_TOKEN`（`API Key`）的值留作后用：

![复制 BASE_URL 和 AUTH_TOKEN](assets/image-20260823235305539.png)

这是所有客户端通用的凭证。

## 接入客户端

> 受限于演示设备，本部分只在 macOS/Linux 上实际验证过。Windows 用户在 WSL2 里执行同样的命令也可以工作，其他设备场景可以复制本部分给 AI 并阐述自己的机器配置。

到这里，你已经有了属于自己的 API Key，可以接入任意客户端进行使用。下面演示几个常用客户端的配置，其他任何支持自定义 `base_url` 的工具同理。

先确认 base_url 该填什么：

| 场景 | base_url |
| ---- | -------- |
| 就在这台机器上用 | `http://127.0.0.1:8081` |
| 同一个 Wi-Fi 下的手机 / 另一台电脑 | `http://<本机局域网 IP>:8081` |
| 装在服务器上，任何地方都能访问（见[拓展](#拓展让其他设备也能访问)） | `http://<你的服务器 IP>:8081` |
| 域名访问（见[拓展](#可选域名--cloudflare)） | `https://<你的域名>` |

下文统一以本机（`http://127.0.0.1:8081`）为例，**其他场景需要替换地址**。另外注意，Sub2API 弹窗里给的示例配置有时会滞后，本文以写这篇时订阅侧可用的 gpt-5.6-* 模型族进行配置演示。

### Claude Code

如果「创建分组」时启用了「允许 `/v1/messages` 调度」，那么 `claude-*` 模型会被自动映射到对应的 GPT 模型：

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8081"
export ANTHROPIC_AUTH_TOKEN="<你的 API Key>"
claude
```

### ChatGPT（Codex）

编辑 `~/.codex/config.toml`：

```toml
model_provider = "openai-custom"
model = "gpt-5.6-sol"

[model_providers.openai-custom]
name = "openai-custom"
base_url = "http://127.0.0.1:8081/v1"
wire_api = "responses"
env_key = "OPENAI_CUSTOM_API_KEY"
```

然后把对应的密钥写进环境变量再启动（注意这里用的是 `OPENAI_CUSTOM_API_KEY`）：

```bash
export OPENAI_CUSTOM_API_KEY="<你的 API Key>"
codex
```

如果还想让 Codex 通过中转生图（对应前面分组里开的「允许生图」），还需要额外增加一行配置：

```toml
[model_providers.openai-custom]
# ...前面的配置保持不变...
http_headers = { "x-openai-actor-authorization" = "placeholder" }
```

因为 Codex 的 `image_gen` 工具默认只对官方登录开放，所以配置第三方 API 的时候没办法直接在 Codex 中生图。解决方法是给 provider 加一个探测头（上面的那行配置），值非空即可。

> [!note]
>
> - `preferred_auth_method = "apikey"` 在 2025 年 9 月（rust-v0.35.0）被[移除](https://github.com/openai/codex/pull/3189)，对应配置会被**静默忽略**。这时候如果没配 `env_key`，Codex 会不带任何认证头发请求，从而触发报错 `401 API_KEY_REQUIRED`。
> - provider 的 id 不能叫 `openai`（即 `model_providers.openai`），否则 Codex 会报错 `Built-in providers cannot be overridden`。

### OpenCode

编辑 `~/.config/opencode/opencode.json`：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "openai-custom": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8081/v1",
        "apiKey": "<你的 API Key>"
      },
      "models": {
        "gpt-5.6-sol": { "name": "GPT-5.6 Sol" }
      }
    }
  }
}
```

### Pi

[Pi](https://github.com/badlogic/pi-mono) 通过 npm 安装：

```bash
npm install -g @earendil-works/pi-coding-agent
```

编辑 `~/.pi/agent/models.json`：

```json
{
  "providers": {
    "openai-custom": {
      "baseUrl": "http://127.0.0.1:8081/v1",
      "api": "openai-completions",
      "apiKey": "<你的 API Key>",
      "models": [
        { "id": "gpt-5.6-sol", "name": "GPT-5.6 Sol", "contextWindow": 400000, "maxTokens": 128000 }
      ]
    }
  }
}
```

`contextWindow` 和 `maxTokens` 这两个值是自己声明的，pi 不会去问后端。用下面的命令确认模型被识别：

```bash
pi --list-models openai-custom
```

然后带 `--model` 启动（或进入会话后用 `/model` 切换）：

```bash
pi --model openai-custom/gpt-5.6-sol
```

> [!note]
>
> 如果终端里设置过 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量，pi 会把发往 `127.0.0.1` 的请求交给代理，从而报错 `503 status code (no body)`。所以运行前把本机地址排除掉：
>
> ```bash
> export NO_PROXY=127.0.0.1,localhost
> ```

### DeepSeek Harness

编辑 `~/.dsh/settings.yaml`（不存在就先创建）：

```yaml
agent-default-model:
  provider: openai-custom
  model: gpt-5.6-sol
llm-pi-ai:
  providers:
    openai-custom:
      apiKeyEnv: OPENAI_CUSTOM_API_KEY
      api: openai-completions
      baseURL: http://127.0.0.1:8081/v1
      models:
        - id: gpt-5.6-sol
```

`apiKeyEnv` 填的是环境变量名（Codex 的 `env_key` 也是这样）而不是 Key 本身，这里沿用 `OPENAI_CUSTOM_API_KEY`。

验证一下能否拿到回复（需要 Node 20.12 及以上，Node 18 会报 `does not provide an export named 'parseEnv'`）：

```bash
export OPENAI_CUSTOM_API_KEY="<你的 API Key>"
npx @deepseek-ai/dsh --profile headless "Hello"
```

然后用 `npx @deepseek-ai/dsh web` 打开 Web UI 进行对话。

### （可选）写进 shell 配置

> [!note]
>
> 这会让后续的每次启动都走这个配置。

#### Linux / macOS / WSL2

上面的 `export` 只对当前终端窗口有效，更建议写进持久的配置中：

```bash
RC=~/.${SHELL##*/}rc
KEY='<你的 API Key>'
sed -i.bak '/# >>> sub2api >>>/,/# <<< sub2api <<</d' "$RC" && rm -f "$RC.bak"
cat >> "$RC" <<EOF
# >>> sub2api >>>
export ANTHROPIC_BASE_URL="http://127.0.0.1:8081"  # 换成你的实际地址
export ANTHROPIC_AUTH_TOKEN="$KEY"
export OPENAI_CUSTOM_API_KEY="$KEY"
export NO_PROXY=127.0.0.1,localhost
# <<< sub2api <<<
EOF
source "$RC"
```

#### Windows（PowerShell）

```powershell
$KEY = "<你的 API Key>"
[Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", "http://127.0.0.1:8081", "User")  # 换成你的实际地址
[Environment]::SetEnvironmentVariable("ANTHROPIC_AUTH_TOKEN", $KEY, "User")
[Environment]::SetEnvironmentVariable("OPENAI_CUSTOM_API_KEY", $KEY, "User")
[Environment]::SetEnvironmentVariable("NO_PROXY", "127.0.0.1,localhost", "User")
```

### 沉浸式翻译

设置 →「翻译服务」→「添加自定义翻译服务」，然后填：

| 配置项 | 值（以本机部署为例） |
| ------ | ------ |
| 自定义 API 接口地址 | `http://127.0.0.1:8081/v1/chat/completions` |
| API Key | `<你的 API Key>` |
| 模型 | `gpt-5.6-sol` |

![添加自定义翻译服务并点击测试](assets/image-20260824221206631.png)

点击测试服务成功后，在「基础设置」中切换翻译服务：

![在基础设置中切换翻译服务](assets/image-20260824221657503.png)

### 其他客户端

任何支持 OpenAI / Anthropic SDK 的工具都可以跑通，比如自定义 Python 脚本：

- **OpenAI**

  ```python
  from openai import OpenAI

  client = OpenAI(
      base_url="http://127.0.0.1:8081/v1",  # 注意这里需要后缀 /v1
      api_key="<你的 API Key>",
  )
  resp = client.chat.completions.create(
      model="gpt-5.6-sol",
      messages=[{"role": "user", "content": "你好"}],
  )
  print(resp.choices[0].message.content)
  ```

- **OpenAI Responses API**

  ```python
  from openai import OpenAI

  client = OpenAI(
      base_url="http://127.0.0.1:8081/v1",
      api_key="<你的 API Key>",
  )
  resp = client.responses.create(
      model="gpt-5.6-sol",
      input="你好",
  )
  print(resp.output_text)
  ```

- **Anthropic**

  ```python
  import anthropic
  
  client = anthropic.Anthropic(
      base_url="http://127.0.0.1:8081",  # 注意这里不需要 /v1，SDK 会自己补
      api_key="<你的 API Key>",
  )
  resp = client.messages.create(
      model="claude-opus-4-5-20251101",  # claude-* 会被映射到对应的 GPT 模型
      max_tokens=1024,
      messages=[{"role": "user", "content": "你好"}],
  )
  print(resp.content[0].text)
  ```

> [!note]
>
> **OpenAI SDK 需要 `/v1` 后缀**。因为它只会在 `base_url` 后面拼 `/chat/completions`，写成 `http://127.0.0.1:8081` 时，请求会到 `/chat/completions` 而非 `/v1/chat/completions`，此时会返回 `index.html`。所以上面的脚本实测会报错：
>
> ```
> AttributeError: 'str' object has no attribute 'choices'
> ```
>
> 而 Anthropic 这边的 `base_url`（包括 Claude Code 的 `ANTHROPIC_BASE_URL`）不用加 `/v1`，SDK 会自动加。

至此，本机使用 API 的需求已经满足了，后续内容可以根据自身实际情况阅读。

## 拓展：让其他设备也能访问

局域网可以让实验室/舍友/家庭成员进行访问，但得一直开着电脑，而且换个网就用不了了。此时我们需要一台服务器，这样部署后直接用 `http://<你的服务器 IP>:8081` 就能从任何地方访问。

### 服务器要求

和本机一样，能正常访问 OpenAI 就行，如果用量比较大，2 核及以上会更好。1 核 1G 可以跑通自用，如果选择这个配置的话，可以购买阿里云/腾讯云/华为云等厂商的轻量应用服务器，价格约为 99 元/年。注意**国内地域的机器大概率直连不上 OpenAI**，需要按前面[（可选）出口 IP 设置](#可选出口-ip-设置)进行配置，或者直接选海外云服务商。

如果不知道选什么系统的话，可以选择 Ubuntu 22.04 或 24.04 LTS，这也是本文测试机器的环境。另外注意，服务器存在流量开销，一个 Pro 账户用完的情况下可能有 200GB 的出入流量。

### （可选）通过 AWS 赠送的 200\$ 代金券获取练手的服务器

> 这一步需要新的信用卡 & 手机号验证，由于作者已经领过了，所以仅进行注册路径分享。

打开[注册页](https://signin.aws.amazon.com/signup?request_type=register)，填写邮箱和账户名称后点击「验证电子邮件地址」：

![AWS 注册页：填写邮箱与账户名称](assets/image-20260818174222787.png)

在邮箱中查看验证码：

![AWS 发来的验证码邮件](assets/image-20260818212831617.png)

输入发送到邮箱的验证码：

![输入邮箱收到的验证码](assets/image-20260818212920266.png)

在设置密码后，你可以看到类似于下面的画面，这里选择免费计划：

![选择免费计划（Free plan）](assets/image-20260818213029157.png)

然后填写信息，地址可以通过[美国地址生成器](https://www.meiguodizhi.com/usa-address/ohio)进行填充：

![填写联系人信息](assets/image-20260818220429238.png)

第三步需要填写一个实际可用的信用卡，也可以通过闲鱼临时解决：

![填写信用卡信息](assets/image-20260818214150634.png)

后续通过赠送的抵扣金租赁服务器即可。

### 在服务器上部署

拿到服务器后，回到[部署](#部署)部分重跑一遍命令，注意：

- `.env` 里的 `BIND_HOST` 使用默认值 `0.0.0.0`，否则正常情况下只有服务器自己能访问。
- 云服务商的安全组 / 防火墙放行 `SERVER_PORT`（AWS / 腾讯云 / 阿里云是「安全组」，Lightsail 是「网络」→「IPv4 防火墙」）。

部署完之后把[接入客户端](#接入客户端)部分的 base_url 换成 `http://<你的服务器 IP>:8081`。

### （可选）域名 + Cloudflare

如果你觉得用 IP + HTTP 进行访问不够“优雅”，可以阅读本节。

#### 域名购买

自用的情况下，域名单纯是“为了折腾而折腾”，本质上是想让服务器换 IP 的时候不用修改客户端配置，并非必需，所以想跳过的同学可以跳过。

- **关于服务商**：国内的阿里云/腾讯云等云服务商（因为合规原因，在国内云服务商购买后，需要先进行网站备案），国外的 [Cloudflare Registrar](https://www.cloudflare.com/products/registrar/) / [Spaceship](https://www.spaceship.com/zh/) 等都可以，这里将以 [Spaceship](https://www.spaceship.com/) 为例进行讲解（因为早期购买导致的路径依赖）。比如，我想买一个未来可用于分享的学习网站，又不想太过严肃，那么，就“好好学习”吧，于是我输入 “goodgoodstudy”（替换成你想要的域名），点击搜索：

  ![在 Spaceship 搜索想要的域名](assets/image-20260818164353606.png)

  然后就可以在展开的界面中挑选合适的域名了，点击「加入购物车」后进行购买：

  ![挑选合适的域名后加入购物车](assets/image-20260818164226980.png)

- **关于后缀**：自用的情况下后缀完全不重要，挑便宜的，比如：`.site` / `.space` / `.top` 等，首年最多只需要 1\$。

  - 但还需要关注后续的续订价格，具体因站点而异，不要买那种首年 1 刀，续订几十刀的后缀，因为可能用着用着就不想换了，特别是维护多份机器配置的时候。

  - `.xyz` 域名仅建议购买 6 位数字域名，其他格式在去年都进行了涨价，特别是续订：

    ![.xyz 域名的续订价格](assets/image-20260818162150949.png)

- **关于备案**：如果域名解析和 CDN 都在 Cloudflare，服务器不在大陆，且不对国内提供服务，纯自用，就不需要进行备案。但如果你的服务器在国内，而且还想为其他人提供服务，那么按规定是要进行备案的。

#### 把域名接入 Cloudflare

买好域名后，可以将它托管到 [Cloudflare](https://dash.cloudflare.com/)（部分地方简写为 CF）。在这里我们将用到 Cloudflare 免费提供的三个功能：

1. **DNS 托管**：「购买的域名」到「服务器 IP」的解析。
2. **反向代理（小黄云）**：隐藏真实 IP，白嫖 HTTPS 证书和基础的 DDoS 防护。
3. **规则配置**：改写端口、做重定向等，适合一些不开放 80/443 端口的服务器厂商。

为了更方便阅读，我们先将语言调整为中文，点击右上角的图标后切换：

![切换 Cloudflare 控制台语言](assets/image-20260601232910096.png)

控制台左侧「[域名](https://dash.cloudflare.com/?to=/:account/domains/overview)」→「添加域名」（文中这类面板链接点开会先让你选账号和域名，再直接落到对应页面）：

![控制台左侧域名入口](assets/image-20260601232810439.png)

点击「连接域名」：

![点击连接域名](assets/image-20260601233353955.png)

填入你的域名，这里以刚购买的 `goodgoodstudy.ai` 为例进行演示。点击继续：

![填入待接入的域名](assets/image-20260601233528982.png)

对于个人，选择免费计划是完全够用的：

![选择 Free 套餐](assets/image-20260601233941666.png)

直接点「继续前往激活」：

![继续前往激活](assets/image-20260601235132098.png)

点击「确认」：

![确认激活](assets/image-20260601235249240.png)

分别复制接下来的界面提供的 Cloudflare 两个名称服务器：

![复制 Cloudflare 提供的两个名称服务器](assets/image-20260601235459522.png)

这里以 Spaceship 为例，在阿里云和其他云服务商购买也是类似的逻辑，搜寻 DNS 相关的界面进行配置：

![在域名注册商处找到 DNS 设置](assets/image-20260601235804283.png)

点击对应的域名（你购买的）：

![选择对应的域名](assets/image-20260601235928329.png)

点击「更改」：

![进入名称服务器更改界面](assets/image-20260602000048215.png)

选择「自定义名称服务器」，粘贴 Cloudflare 提供的两个地址，然后点击保存：

![填入 Cloudflare 的自定义名称服务器](assets/image-20260602000218438.png)

然后域名就会开始传播：

![域名开始传播](assets/image-20260602000323068.png)

回到 Cloudflare，点击下方的「我已更新名称服务器」：

![点击「我已更新名称服务器」](assets/image-20260602000422712.png)

需要等待一段时间才能感知到：

![等待 Cloudflare 检测名称服务器](assets/image-20260602000549943.png)

实测此次等待 6 分钟，会有邮件通知：

![域名激活成功的邮件通知](assets/image-20260602001259905.png)

切到左侧的「域名」→「[概览](https://dash.cloudflare.com/?to=/:account/:zone)」：

![左侧域名概览入口](assets/image-20260602001716233.png)

点击后，可以看到已经受到保护：

![域名已处于受保护状态](assets/image-20260602001843048.png)

#### 把域名指向你的服务器

点击左栏的「DNS」→「[记录](https://dash.cloudflare.com/?to=/:account/:zone/dns/records)」→「添加记录」：

![DNS 添加记录入口](assets/image-20260602002334015.png)

如果想把域名本身指向服务器 IP，那么名称部分需要填写 `@`，类型选 `A`，IPv4 地址填服务器公网 IP，**代理状态需要开启**，然后就可以看到俗称的“小黄云”了：

![添加根域名的 A 记录并开启代理](assets/image-20260602002524575.png)

接下来，你应该可以在界面上看到类似于下方的记录行，其中代理状态显示为「已代理」：

![DNS 记录列表中的已代理 A 记录](assets/image-20260602101221236.png)

#### 让用户无感访问非标准端口

有时候我们并不想让服务端口出现在 URL 里，而且有些服务器可能禁用了 80 和 443 端口，那应该怎么做才能让用户无感访问呢？

点击侧栏的「规则」→「[概述](https://dash.cloudflare.com/?to=/:account/:zone/rules/overview)」→「创建规则」→「Origin Rules」：

![规则概述中创建 Origin Rules](assets/image-20260602002852199.png)

字段选择主机名（Hostname），运算符选择等于（Equal），值填写你的域名：

![Origin Rules 的匹配条件](assets/image-20260602003129410.png)

往下拉到「目标端口」，选择「重写到...」并填入你的服务端口（这里是 `8081`，根据实际情况填写），其余保持「保留」，然后点击「部署」：

![把目标端口重写为 8081 并部署](assets/image-20260602003427973.png)

部署完成后可以在规则列表中看到它处于「活动」状态：

![Origin Rules 列表中处于活动状态的规则](assets/image-20260602003918560.png)

现在用户通过域名的访问会被拆成两个部分：

1. 从 `https://<你的域名>`（443 端口）连接到 Cloudflare

2. Cloudflare 再通过 `http://<你的服务器 IP>:8081` 进行连接

   > [!tip]
   >
   > 第二步也被称为**回源请求（Fetch to Origin）**。

这样就可以让其他人通过域名而非 IP:端口进行访问。

#### SSL/TLS 模式

现在还有最后一步，点击左栏的「SSL/TLS」→「[概述](https://dash.cloudflare.com/?to=/:account/:zone/ssl-tls)」→「配置」：

![SSL/TLS 概述与配置入口](assets/image-20260602005435614.png)

这个设置对应的是「让“回源”操作用 HTTP 还是 HTTPS」。在默认情况下是**完全（Full）**模式，也就是说 CF 会用 HTTPS 去连服务器，但 Sub2API 在对应端口上仅监听 HTTP，握手必然失败，此时访问 `https://<你的域名>` 会报 `Error code 525 — SSL handshake failed`：

![Full 模式下的 525 SSL handshake failed 报错](assets/image-20260602011659296.png)

如果想临时测试连通性，选择**灵活（Flexible）**模式，点击「保存」就可以通过 `https://<你的域名>` 进行访问了：

![保存 SSL/TLS 加密模式](assets/image-20260602005513729.png)

> [!important]
>
> 「灵活」模式意味着「CF → 你的服务器」这一段是 HTTP，前面的 Origin Rules 也只是把端口藏进了 URL，其他人依然能直接访问 `http://<你的服务器 IP>:8081` 绕过 CF 提供的防护。下面提供两种解决方案（感觉迷惑可以复制给 AI）：
>
> 1. **（推荐 / 附录）用 [Cloudflare Tunnel](#cloudflare-tunnel-配置-prompt)**，自带源站加密，会忽略这里的 SSL/TLS 模式设置，此时选「完全（严格）」就行，然后把 `.env` 的 `BIND_HOST` 改成 `127.0.0.1`，再执行一次 `docker compose up -d` 让它生效。
>
> 2. 如果只是自己和朋友用，也可以用 [Tailscale](https://tailscale.com/) 这类工具把几台设备组成虚拟局域网，让 Sub2API 只监听内网地址，这样域名和 CF 都可以跳过。

> [!tip]
>
> **1. 如果我们不想让根域名绑死这个服务怎么办？**
>
> 回到 [DNS](https://dash.cloudflare.com/?to=/:account/:zone/dns/records)，名称部分填写想要的子域名，比如填 `<子域名>`，就会得到 `<子域名>.<你的域名>`，假设我想这里用 enjoy：
>
> ![为子域名添加 A 记录](assets/image-20260602101411860.png)
>
> 然后回到「[规则](https://dash.cloudflare.com/?to=/:account/:zone/rules/overview)」里的那条 Origin Rule，把主机名（Hostname）改成「等于」+「完整子域名」即可：
>
> ![Origin Rule 匹配子域名](assets/image-20260602130529467.png)
>
> **2. 如果想直接跳过初始主页进行登录怎么办？**
>
> 规则 → [概述](https://dash.cloudflare.com/?to=/:account/:zone/rules/overview) → 创建规则 → 重定向规则：
>
> ![创建重定向规则](assets/image-20260602130937223.png)
>
> 自定义名称后，字段选择主机名（Hostname），运算符选择等于（Equal），值填写实际购买的域名（沿用上面的域名就是 `<子域名>.<你的域名>`）。然后点击右侧的 `And` 增加一条，字段选择 URI 路径，运算符选择等于，值填写 `/`：
>
> ![重定向规则的匹配条件与重定向目标](assets/image-20260602131253975.png)
>
> 表达式预览应该是：
>
> ```
> (http.host eq "<子域名>.<你的域名>" and http.request.uri.path eq "/")
> ```
>
> 在上图的「则...」中，URL 重定向类型选择「动态」，表达式填写 `concat("https://<子域名>.<你的域名>", "/login")`，状态代码选择 `302 - Temporary Redirect`。
>
> 这会让 `https://<子域名>.<你的域名>/` 被重定向为 `https://<子域名>.<你的域名>/login`。

配好之后把客户端的 base_url 换成 `https://<你的域名>`。

上面是比较“古法”的手操流程。现在有了 AI，只要知晓某个领域关键词作为 “seed”，就可以让它替你完成曾经存在知识壁垒的工作。所以，也可以直接把[附录](#cloudflare-tunnel-配置-prompt)部分的 prompt 交给 AI，但仍希望你经历一下前面的流程。

## 附录

> 欢迎在[讨论区](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/discussions)反馈任何卡住的地方，即便 Prompt 已经在多台 Linux 服务器和 Mac 上完整跑通过，也依旧可能在你的环境下出问题。

- [Sub2API 部署 Prompt](#sub2api-部署-prompt)
- [Cloudflare Tunnel 配置 Prompt](#cloudflare-tunnel-配置-prompt)

### Sub2API 部署 Prompt

如果需要 ssh 连接服务器，就在开头的“这台机器”文本后增加完整的 ssh 命令：

````
我想在这台机器上用 Docker Compose 部署 Sub2API（https://github.com/Wei-Shaw/sub2api），要求：

- 用官方部署脚本，不要自己拼 docker-compose.yml
- 端口、时区、管理员邮箱按我的要求定制
- 装完做健康校验，并把一次性管理员密码抓给我
- 幂等：如果已经部署过，检测出来问我，不要覆盖已有数据

## 阶段 0：先决条件检查（任一不满足立即停止，不要绕过）

```bash
# 系统类型：Linux / macOS / WSL2 都支持，Windows 原生不支持（让我改用 WSL2）
uname -s; uname -m
grep -qi microsoft /proc/version 2>/dev/null && echo "WSL=true"

# Docker 与 compose 插件
docker --version || { echo "FAIL: 未装 Docker"; exit 1; }
docker compose version || { echo "FAIL: 缺 compose 插件（不是 docker-compose 老版本）"; exit 1; }
docker info >/dev/null 2>&1 || { echo "FAIL: Docker 守护进程没跑起来"; exit 1; }

# 磁盘 ≥ 10GB，内存按 total ≥ 1GB 判断（三个容器空载约 170 MiB，Postgres 有负载后会涨）
df -h .; free -m 2>/dev/null || vm_stat | head -3

# 出站网络。GitHub 不通只是 WARN，Step 1 改走镜像；Docker Hub 那项按下面处理
curl -sI --max-time 8 https://raw.githubusercontent.com >/dev/null || echo "WARN: GitHub 不通，Step 1 走镜像"

# Docker Hub 直连在国内基本不通，先看是否已配镜像加速，已配就没问题
docker info 2>/dev/null | grep -A3 -i "registry mirrors" || echo "未配镜像加速"
curl -sI --max-time 8 https://registry-1.docker.io/v2/ >/dev/null || echo "WARN: Docker Hub 直连不通"
```

**「未配镜像加速」和「Docker Hub 直连不通」同时出现时**，用下面两个镜像源，不要自己猜别的地址：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'EOF'
{ "registry-mirrors": ["https://dockerproxy.net", "https://docker.1panel.live"] }
EOF
sudo systemctl restart docker
docker info | grep -A2 -i "registry mirrors"
```

已经存在 `/etc/docker/daemon.json` 的话不要覆盖，先把现有内容读给我，再把 `registry-mirrors` 合并进去。macOS 的 Docker Desktop 没有这个文件，停下让我在 Settings → Docker Engine 里填同样两个地址。

**Docker 缺失时按平台装**，不要用错方式：

- Linux / WSL2：`curl -fsSL https://get.docker.com | sh` 然后 `sudo usermod -aG docker $USER`，装完必须重开终端
- macOS：**`get.docker.com` 会直接拒绝执行**（脚本里明确 `Unsupported operating system 'macOS'`），让我自己装 Docker Desktop
- Windows 原生：停下，让我改用 WSL2 或远程 Linux 或让我自己装 Docker Desktop

## 阶段 1：环境探测

```bash
# 列出当前所有监听端口，供阶段 2 选端口时避开（默认的 8080 冲突概率很高）
ss -tlnp 2>/dev/null || lsof -iTCP -sTCP:LISTEN -P 2>/dev/null

# 是否已有部署（幂等关键）
docker ps -a --filter "name=sub2api" --format "{{.Names}}\t{{.Status}}"
ls -la ~/sub2api/.env /opt/sub2api/.env 2>/dev/null
```

**任一情况出现，停下报告，由我决定**：

| 情况 | 默认动作 |
|------|----------|
| 已存在 `sub2api` 容器 | 停，问我：复用 / 重建（会丢数据）/ 换目录 |
| 已存在 `.env` | 停，**绝对不要覆盖**——里面的 `JWT_SECRET`、`TOTP_ENCRYPTION_KEY` 一旦变了，所有会话失效、已绑定的 2FA 全部作废 |
| 目标端口被占 | 停，问我换端口还是停掉占用方 |
| 机器有公网 IP | 提示我 `BIND_HOST` 默认是 `0.0.0.0`，等于直接暴露到公网，问我是否改 `127.0.0.1`。**不要只看 `ip -4 addr`**：云主机大多是 NAT，网卡上只有 `172.x`/`10.x` 私网地址，看起来像内网机，实际仍有公网入口。用 `curl -s --max-time 8 icanhazip.com` 拿到的地址去判断；出站也被挡的话，直接问我这台机器有没有公网 IP |

## 阶段 2：一次性收集信息

一次问清楚，别来回打断我。每个问题给出你的推荐值：

| 项 | 说明 |
|----|------|
| 安装目录 | 推荐 `~/sub2api`（`/opt` 需要 sudo） |
| `SERVER_PORT` | 默认 8080，端口冲突再换 |
| `BIND_HOST` | 公网服务器且没有反代/防火墙 → 推荐 `127.0.0.1`；家里的机器要给局域网用 → `0.0.0.0` |
| `TZ` | 影响「今日用量」的日界，部署后再改不会迁移历史数据，推荐 `Asia/Singapore` |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | 密码留空 = 首次启动随机生成（只在日志里出现一次）；**填了就不会进日志**，更稳妥 |

## 阶段 3：执行

### Step 1：拉官方脚本并执行

```bash
mkdir -p <安装目录> && cd <安装目录>
curl -fsSL -o docker-deploy.sh \
  https://raw.githubusercontent.com/Wei-Shaw/sub2api/main/deploy/docker-deploy.sh
bash docker-deploy.sh
```

脚本只负责生成 `docker-compose.yml` 和带随机密钥的 `.env`，**不会启动服务**。预期输出里有 `Directory structure:` 和 `Next steps:`。

阶段 0 报了「GitHub 不通」的话改用镜像。注意脚本内部还会去 raw.githubusercontent.com 拉 compose 和 `.env` 模板，所以要把脚本里的地址一起替换：

```bash
M="https://ghproxy.net/https://raw.githubusercontent.com"
curl -fsSL -o docker-deploy.sh "$M/Wei-Shaw/sub2api/main/deploy/docker-deploy.sh"
sed -i.bak "s|https://raw.githubusercontent.com|$M|" docker-deploy.sh && rm -f docker-deploy.sh.bak
bash docker-deploy.sh
```

`ghproxy.net` 不通就把 `M` 换成 `https://gh-proxy.com/https://raw.githubusercontent.com` 再试一次，两个都不行停下问我，不要自己找别的加速站。

### Step 2：定制 `.env`

```bash
cd <安装目录>
sed -i.bak "s/^SERVER_PORT=.*/SERVER_PORT=<端口>/; s|^TZ=.*|TZ=<时区>|; s/^BIND_HOST=.*/BIND_HOST=<绑定地址>/" .env
grep -E "^(BIND_HOST|SERVER_PORT|TZ|ADMIN_EMAIL|ADMIN_PASSWORD)=" .env

# sed 匹配不到键时不会报错、退出码照样是 0，所以必须拿上面 grep 的输出逐条核对是不是改成了我要的值
# 核对完删掉备份：.env.bak 里是同一套密钥的明文副本
rm -f .env.bak
```

改完把这几行读给我确认。**不要动 `JWT_SECRET`、`TOTP_ENCRYPTION_KEY`、`POSTGRES_PASSWORD`**。

### Step 3：启动并等待健康

```bash
docker compose up -d --quiet-pull
for i in $(seq 1 24); do
    docker compose ps --format "{{.Name}}\t{{.Status}}"
    # 数本项目里健康的容器：docker compose ps 没有 health 过滤器，要自己 inspect
    # grep 用 -x 精确匹配，否则 unhealthy 也会被算进去
    n=$(docker compose ps -q | xargs -r docker inspect \
        --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' 2>/dev/null | grep -cx healthy)
    if [ "$n" -ge 3 ]; then break; fi
    sleep 5
done
```

三个容器（`sub2api` / `sub2api-postgres` / `sub2api-redis`）都要 healthy。首次启动 Sub2API 会等 Postgres 就绪并跑迁移，通常 30 秒内。

### Step 4：抓一次性管理员密码

```bash
docker logs sub2api 2>&1 | grep -iE "admin password|one-time"
# 预期：Generated admin password (one-time): <32 位十六进制>
```

抓到后**立刻**完整给我，并提醒我存进密码管理器。

### Step 5：验证

```bash
# 健康端点（--noproxy 很重要，见下）
curl -s --noproxy '*' -o /dev/null -w "health: %{http_code}\n" http://127.0.0.1:<端口>/health

# 实际监听地址，核对是否与 BIND_HOST 一致
ss -tlnp 2>/dev/null | grep :<端口> || lsof -iTCP:<端口> -sTCP:LISTEN -P
```

`/health` 返回 200 即成功。最后把 Web UI 地址给我：`http://<地址>:<端口>`。

## 执行约束

- 任何命令失败：停下，原文报告错误，等我决定，**不要自动重试或绕过**
- **唯一例外是拉镜像**：`docker compose up -d` 走镜像源时偶发 `failed to copy: httpReadSeeker: ... not found`，原样重跑就好，最多重试 3 次，仍失败再停下报告
- 需要我决策时：一次只问一个问题，列出可选项并标注你的推荐项（附一句理由），能通过探测环境自行确认的，不要来问我
- **不要覆盖已存在的 `.env`**，也不要重新生成里面的任何密钥
- 输出每条命令时同时输出「预期成功标志」
- 全程不要把 `.env` 的内容整段打印出来（里面有数据库密码和 JWT 密钥）。注意官方部署脚本自己会把生成的密钥打进 stdout，转述它的输出时把那几行滤掉

## 已知边界情况

- **健康检查被代理劫持**：机器上如果设了 `HTTP_PROXY` / `HTTPS_PROXY`，`curl` 和各类客户端会把发往 `127.0.0.1` 的请求也交给代理，表现为连不上或 `503`，而 Sub2API 日志里根本没有这条请求。校验时一律加 `--noproxy '*'`，客户端侧则 `export NO_PROXY=127.0.0.1,localhost`
- **`docker-compose` 与 `docker compose`**：本项目的脚本按新版插件写，老的 `docker-compose`（带横杠）可能解析失败
- **数据目录**：`data/`、`postgres_data/`、`redis_data/` 都在安装目录下，迁移机器直接打包整个目录即可，`.env` 一起带走就能保留所有账号
- **用量日志会持续增长**，繁忙网关约 1–5 GB/月，保留天数可在 `.env` 里的 `DASHBOARD_AGGREGATION_RETENTION_*` 调
- **升级**：`docker compose pull && docker compose up -d`，数据库迁移会在容器启动时自动跑；升级前先备份整个安装目录
````

### Cloudflare Tunnel 配置 Prompt

同样，如果需要 ssh 连接服务器，就在开头的“这台服务器”文本后增加完整的 ssh 命令：

````
这台服务器上面跑了多个 Web 服务（每个监听本机某个端口），我想用 Cloudflare Tunnel 把它们安全地暴露到公网，要求：

- 被隧道接管的服务不再需要公网端口，真实 IP 完全隐藏（机器上还跑着别的东西，哪些端口要一并收紧最后问我，别自作主张）
- 通过我自己的域名访问（域名已托管在 Cloudflare）
- 支持多个 Hostname → 不同本地端口/协议的映射
- 全程加密（端到端 HTTPS）
- 配置成 systemd 服务，开机自启、崩溃自动重启

## 阶段 0：先决条件检查（任一不满足立即停止，不要绕过）

```bash
# 必须 root
[ "$(id -u)" -ne 0 ] && { echo "FAIL: 请用 root 或 sudo -i 重新执行"; exit 1; }

# 时间必须同步（cert 验证依赖时间）
SYNCED=false
timedatectl status 2>/dev/null | grep -qi "synchronized: yes" && SYNCED=true
chronyc tracking 2>/dev/null | grep -qi "Leap status.*Normal" && SYNCED=true
[ "$SYNCED" = "false" ] && { 
    echo "FAIL: 时间未同步，运行: timedatectl set-ntp true || ntpdate -u pool.ntp.org"
    exit 1
}

# 出站 443 必须连得通 CF
curl -sI --max-time 5 https://api.cloudflare.com >/dev/null \
  || { echo "FAIL: 出站 443 不通"; exit 1; }

# 7844 是隧道真正用的端口（优先 UDP/QUIC，回退 TCP）。UDP 被拦是最常见的失败原因，
# 症状就是后面 Step 6 的「edge 连接未建立」，那时改用 --protocol http2 走 TCP 即可
timeout 6 bash -c "exec 3<>/dev/tcp/region1.v2.argotunnel.com/7844" 2>/dev/null \
  && echo "OK: TCP 7844 可达" || echo "WARN: TCP 7844 不通，隧道大概率建不起来"
```

## 阶段 0.5：依赖安装（不交互）

```bash
PKGS="jq openssl curl"
if command -v apt-get >/dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $PKGS dnsutils ca-certificates
elif command -v dnf >/dev/null; then
    dnf install -y -q $PKGS bind-utils ca-certificates
elif command -v yum >/dev/null; then
    yum install -y -q $PKGS bind-utils ca-certificates
elif command -v apk >/dev/null; then
    apk add --no-cache $PKGS bind-tools ca-certificates
else
    echo "FAIL: 未识别包管理器，请手动安装: $PKGS dig"
    exit 1
fi

# 验证关键命令
for cmd in jq openssl curl dig; do
    command -v $cmd >/dev/null || { echo "FAIL: $cmd 未成功安装"; exit 1; }
done

# DNS 解析校验
[ -n "$(dig api.cloudflare.com +short +time=3 +tries=1)" ] \
  || { echo "FAIL: DNS 解析不出结果"; exit 1; }
```

## 阶段 1：环境探测（并行执行，结果汇总后再继续）

```bash
# 系统信息
uname -a; uname -m
cat /etc/os-release

# libc 类型（musl/glibc）
if ldd --version 2>&1 | grep -qi musl; then echo "LIBC=musl"; else echo "LIBC=glibc"; fi

# 是否在容器内（影响 localhost 可达性）
if [ -f /.dockerenv ] || grep -qE "docker|lxc|containerd|kubepods" /proc/1/cgroup 2>/dev/null; then
    echo "CONTAINER=true"
else
    echo "CONTAINER=false"
fi

# init 系统
ps -p 1 -o comm= 2>/dev/null

# systemd 可用性
which systemctl && systemctl --version 2>/dev/null | head -1

# 是否已装 cloudflared + 版本（没装是正常的，Step 1 才装）
command -v cloudflared >/dev/null && cloudflared --version || echo "NO_CLOUDFLARED"

# 已有凭证（含归属用户）
ls -la /root/.cloudflared/ /etc/cloudflared/ 2>/dev/null
[ -f /root/.cloudflared/cert.pem ] && stat -c '%U:%G %a %n' /root/.cloudflared/cert.pem

# 已有隧道列表在这里查不了（要 Step 2 授权产生的 cert.pem），检查放在 Step 2 末尾，别在这里试

# 旧 cloudflared 服务状态 + 系统用户
systemctl status cloudflared 2>&1 | head -10
systemctl is-enabled cloudflared 2>&1
id cloudflared 2>/dev/null || echo "NO_CLOUDFLARED_USER"

# 当前监听端口
ss -tlnp 2>/dev/null | head -30

# 现有防火墙规则
iptables -L INPUT -n 2>/dev/null | head -20
ufw status 2>/dev/null | head -10
```

**任一情况出现，停下报告，根据实际情况给出推荐选项，由我决定**：

| 情况                                   | 默认动作                             |
| -------------------------------------- | ------------------------------------ |
| 没有 systemd                           | 停，问我是否用 OpenRC/sysvinit 替代  |
| `LIBC=musl`                            | 停，提示需用静态 binary，不能用 .deb |
| `CONTAINER=true`                       | 停，确认 cloudflared 能访问目标服务  |
| 非 x86_64/aarch64                      | 停，问我是否走静态 binary            |
| 已存在名为 `main` 的隧道               | 停，问我复用还是改名                 |
| 已存在旧版 cloudflared 且版本 < 2025.1 | 停，问我升级还是保留                 |
| `cert.pem` 存在但属于其他账号          | 停，列出 zone 列表让我确认归属       |
| 监听端口为 0.0.0.0                     | 列出来问我哪些要收紧到 127.0.0.1。**SSH（22）必须排除**，收了会把我锁在门外 |

## 阶段 2：一次性收集信息

**服务清单**：每条给出 `<hostname>` → `<本地协议>://<监听地址>:<端口>`

示例：
```
app.example.com    → http://localhost:3000
api.example.com    → https://localhost:8443  (自签证书)
grpc.example.com   → h2c://localhost:9090     (gRPC)
ws.example.com     → http://localhost:7000    (WebSocket 走 http 即可)
```

拿到后逐个验证端口确实在监听，未监听的提前报警避免后续 502：

```bash
ss -tlnp | grep -E "[:.]${PORT}\b"    # 用 \b 收尾，否则查 80 会把 8080、18789 也匹配上
```

某个端口没在监听时，先告诉我是哪个服务、让我确认是不是还没启动，不要直接往下走。

**域名归属确认**：所有 hostname 是否在同一 CF 账号下？跨账号要分隧道。

更常见的是**跨 zone**：`cloudflared tunnel login` 一次只授权一个 zone，cert.pem 里的 token 是 zone 作用域的。如果 hostname 分属多个 zone，Step 5 会报 `zone not found`——这时不是出错，而是要回 Step 2 对另一个 zone 重新 login 一次，再 route 属于它的 hostname。

还要问我一句：**zone 上有没有已存的重定向规则 / Bulk Redirects 会盖住新 hostname**。重定向在 CF 边缘执行、优先于隧道路由，命中时请求根本到不了隧道——症状是部署全部成功、隧道连接健康，但访问域名拿到 301、origin 一条日志都没有。这查不了（cert.pem 的 token 只能管 DNS），只能我去面板确认。

## 阶段 3：执行

### Step 1：安装 cloudflared

**优先官方 .deb/.rpm，禁止 OS 自带 apt repo 旧版本和 snap**。

```bash
# 变量不跨 shell 保留，这里重新检测一次（别依赖阶段 1 的 echo 结果）
if ldd --version 2>&1 | grep -qi musl; then LIBC=musl; else LIBC=glibc; fi
ARCH_RAW=$(uname -m)
case "$ARCH_RAW" in
    x86_64)  ARCH_DEB="amd64"; ARCH_RPM="x86_64"; ARCH_BIN="amd64" ;;
    aarch64) ARCH_DEB="arm64"; ARCH_RPM="aarch64"; ARCH_BIN="arm64" ;;
    armv7l)  ARCH_DEB="armhf"; ARCH_BIN="arm" ;;
    armv6l)  ARCH_BIN="arm" ;;   # 官方没有 armv6 资产，也没有对应 .deb，只能走静态 binary
    *) echo "FAIL: 未支持架构 $ARCH_RAW"; exit 1 ;;
esac

if command -v apt-get >/dev/null && [ "$LIBC" != "musl" ] && [ -n "$ARCH_DEB" ]; then
    curl -fsSL --retry 3 --max-time 1200 -o /tmp/cloudflared.deb \
      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH_DEB}.deb"
    dpkg -i /tmp/cloudflared.deb
elif command -v rpm >/dev/null; then
    curl -fsSL --retry 3 --max-time 1200 -o /tmp/cloudflared.rpm \
      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH_RPM}.rpm"
    rpm -i --force /tmp/cloudflared.rpm
else
    # musl / 静态 binary
    curl -fsSL --retry 3 --max-time 1200 -o /usr/local/bin/cloudflared \
      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH_BIN}"
    chmod +x /usr/local/bin/cloudflared
fi

cloudflared --version
# 预期：cloudflared version 2025.x 或更新
```

**版本低于 2025.1 就停下报告**，让我决定是否升级。

国内机器直连 GitHub 拉这个包可能只有十几 KB/s（实测 18 MB 花了 19 分钟）。太慢就停下告诉我，我来换镜像或手动上传，不要干等。

### Step 2：浏览器授权

```bash
cloudflared tunnel login
```

明确告诉我：

> 命令会打印一个 `https://dash.cloudflare.com/argotunnel?callback=...` 开头的 URL。
>
> **复制这个 URL 粘贴到你本地浏览器**打开（不需要服务器有图形界面）：
> 1. 登录你的 CF 账号
> 2. 选择要授权的域名（zone）
> 3. 点击 Authorize
>
> 服务器上的 cloudflared 会自动检测授权完成并下载 cert.pem。
>
> **等待命令返回，不要中断**。

授权完成验证：
```bash
ls -la /root/.cloudflared/cert.pem
# 预期：文件存在，权限 600，owner root
```

有了 cert.pem 才能查隧道列表，现在补上阶段 1 查不了的那项检查：

```bash
cloudflared tunnel list
```

**出现同名隧道（本文用 `main`）→ 停下问我**。注意「复用」不是无害选项：那等于把这台机器作为新副本加入现有隧道，流量会被分摊过来，而本机的 ingress 规则大概率和原来那台不一致，打过来的请求会掉进 404 兜底——除非我明确说就是要多副本，否则默认改用新名字。

### Step 3：创建/复用隧道

```bash
# 同名检查在 Step 2 末尾做过、并且我已拍板的前提下才走到这里
EXISTING=$(cloudflared tunnel list -o json 2>/dev/null | jq -r '.[] | select(.name=="main") | .id')
if [ -n "$EXISTING" ]; then
    echo "STOP: 已存在同名隧道 $EXISTING，先运行 cloudflared tunnel info main 看它有没有活跃连接，然后等我决定"
    exit 1
fi
cloudflared tunnel create main
TUNNEL_ID=$(cloudflared tunnel list -o json | jq -r '.[] | select(.name=="main") | .id')
[ -z "$TUNNEL_ID" ] && { echo "FAIL: 拿不到 TUNNEL_ID"; exit 1; }
echo "TUNNEL_ID=$TUNNEL_ID"
```

### Step 4：写 config.yml + 修复权限

```bash
mkdir -p /etc/cloudflared
cp /root/.cloudflared/${TUNNEL_ID}.json /etc/cloudflared/

cat > /etc/cloudflared/config.yml << EOF
tunnel: ${TUNNEL_ID}
credentials-file: /etc/cloudflared/${TUNNEL_ID}.json

ingress:
  # === 用户提供的每条 hostname → service ===
  - hostname: <hostname-1>
    service: http://localhost:<port-1>
  # HTTPS 自签后端示例
  - hostname: <hostname-2>
    service: https://localhost:<port-2>
    originRequest:
      noTLSVerify: true
  # gRPC 后端示例
  - hostname: <hostname-3>
    service: h2c://localhost:<port-3>
  # 兜底
  - service: http_status:404
EOF

# 修复 cloudflared 服务用户权限。全新安装时官方 .deb 不会建 cloudflared 用户，
# 这里会落到 root，属于正常情况；只有复用旧部署时才可能真的存在这个用户
SERVICE_USER="cloudflared"
if ! id $SERVICE_USER >/dev/null 2>&1; then
    SERVICE_USER="root"
fi
chown -R ${SERVICE_USER}:${SERVICE_USER} /etc/cloudflared/
chmod 600 /etc/cloudflared/*.json
chmod 644 /etc/cloudflared/config.yml

# 语法校验（--config 写在子命令前面最稳，旧版本后置会报 flag provided but not defined）
cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate \
  || { echo "FAIL: config.yml 语法错误"; exit 1; }

# 它能抓住缺兜底规则、缩进错、端口非数字这些，但**抓不到 scheme 拼错**
# （service: htp://... 照样报 OK，要到运行时才炸），所以自己再核对一遍每条 service 的协议头
```

### Step 5：注册 DNS

`route dns` 没有 dry-run 模式（只有 `--overwrite-dns`），先手动查现有解析，避免盲目覆盖：
```bash
for HOST in <hostname-1> <hostname-2> ...; do
    echo -n "$HOST -> "; dig +short "$HOST"
done
```

已有记录的 hostname 列出来先问我。确认后正式执行：
```bash
for HOST in <hostname-1> <hostname-2> ...; do
    cloudflared tunnel route dns main "$HOST" 2>&1
done
```

任何 `zone not found` 或 `unauthorized` 立即停下报告。

**如果报 `record with that host already exists`**，停下询问：
- 覆盖 → 用 `cloudflared tunnel route dns --overwrite-dns main "$HOST"`
- 跳过该 hostname
- 中止全流程

### Step 6：安装 systemd 服务

```bash
cloudflared service install   # 这一步已经包含 enable + start
systemctl restart cloudflared

# 等待 edge 连接建立
for i in 1 2 3 4 5 6; do
    sleep 5
    CONN_COUNT=$(journalctl -u cloudflared --since "1 minute ago" --no-pager 2>/dev/null \
      | grep -c "Registered tunnel connection")
    [ "$CONN_COUNT" -ge 2 ] && break
done

systemctl status cloudflared --no-pager | head -15
if [ "$CONN_COUNT" -lt 2 ]; then
    echo "FAIL: 30 秒内 edge 连接未建立"
    echo "常见原因是 UDP 7844 被拦，改用 TCP 重试：在 config.yml 顶层加 protocol: http2 后重启"
    journalctl -u cloudflared -n 50 --no-pager
    exit 1
fi
exit 0
```

### Step 7：验证

#### 7.1 隧道在线
```bash
journalctl -u cloudflared -n 50 --no-pager | grep "Registered tunnel connection"
# 预期：2-4 条记录
```

#### 7.2 DNS 已生效
```bash
for HOST in <hostname-1> <hostname-2> ...; do
    echo -n "$HOST -> "
    dig "$HOST" +short @1.1.1.1
done
# 预期：返回 CF 的 anycast IP。route dns 建的 CNAME 是 proxied 的，
# 公共解析器不会把 cfargotunnel.com 返回来，看不到它是正常的
```

#### 7.3 HTTPS 可访问
```bash
for HOST in <hostname-1> <hostname-2> ...; do
    echo -n "$HOST: "
    curl -sI "https://$HOST/" --max-time 15 -o /dev/null \
      -w "HTTP %{http_code} | TLS %{ssl_verify_result}\n"
done
# 预期：HTTP 200/30x/40x | TLS 0
# 502 / 1016 → ingress 写错或后端不可达
# 530 / 1033 → 隧道本身没连上，不是后端的问题
```

#### 7.4 证书链
```bash
curl -v "https://<hostname-1>/" --max-time 10 2>&1 \
  | grep -E "subject:|issuer:|SSL certificate verify"
# issuer 应为 Cloudflare 或 Google Trust Services
```

### Step 8：清理（逐条征求同意）

逐条询问，Y/N 由我决定：

1. **关闭之前为这些服务开的公网端口？**
   - 列出当前 iptables/ufw 中开放的相关端口
   - 给出关闭命令但不执行：
     ```bash
     iptables -D INPUT -p tcp --dport <port> -j ACCEPT
     ufw delete allow <port>
     ```
   - **注意**：云服务器的端口通常拦在厂商的安全组而不是本机防火墙上。如果 `iptables -L INPUT` 是空表且 policy 为 ACCEPT、`ufw status` 是 inactive，那么上面两条命令都是空操作，端口照样开着——这种情况要提醒我去云厂商控制台的安全组里关。

2. **删除针对相同 hostname 的现有 Cloudflare Origin Rules？**
   - 注意 `cert.pem` 内嵌的 token 只能管 DNS，删规则需要我在 <https://dash.cloudflare.com/profile/api-tokens> 另建一个一次性 token（Zone Settings / SSL and Certificates / Origin Rules 的 Edit 权限）
   - 用 CF API 列出现有规则，我确认后再删，删完提醒我吊销 token

3. **SSL/TLS 模式说明**（仅告知）：
   > Tunnel 自带源站加密，**完全忽略** SSL/TLS 模式（Full/Flexible/Off）。Edge Certificate（用户到 CF）仍受 Edge 设置影响。
   >
   > 如果域名只用 Tunnel 暴露这些服务，建议设为 **Full (strict)**——对 Tunnel 无影响，但保护未来可能加的非 Tunnel hostname。

## 执行约束

- 所有命令默认非交互式（`-y`、heredoc、stdin），避免卡 y/n
- **除了 `cloudflared tunnel login` 必须等浏览器授权**，其他必须能直接跑完
- 任何命令失败：停下，原文报告错误，等我决定，**不要自动重试或绕过**
- 需要我决策时：一次只问一个问题，列出可选项并标注你的推荐项（附一句理由），让我直接回复选项即可；能通过探测环境自行确认的，不要来问我
- 除了 Step 2 的浏览器授权和 Step 8 的建 token（这两件事只能在面板做），其余能用 CLI/API 完成的全部用 CLI，不要让我去点鼠标
- **整个流程在 root shell 内执行**
- 输出每条命令时同时输出「预期成功标志」

## 失败回退

```bash
# 停止并禁用服务
systemctl stop cloudflared 2>/dev/null
systemctl disable cloudflared 2>/dev/null

# 卸载 systemd unit
cloudflared service uninstall 2>/dev/null

# 删除隧道。注意它**不会**删掉 route dns 建的 CNAME，那些记录会变成悬挂记录（访问报 1016）
cloudflared tunnel delete -f main 2>/dev/null

# 悬挂 CNAME 用 cert.pem 内嵌的 token 删（它带本 zone 的 DNS 权限，此步要在删 cert.pem 之前做）
TOK=$(sed -n "/BEGIN ARGO TUNNEL TOKEN/,/END ARGO TUNNEL TOKEN/p" /root/.cloudflared/cert.pem | sed "1d;\$d" | base64 -d)
ZONE=$(echo "$TOK" | jq -r .zoneID); API=$(echo "$TOK" | jq -r .apiToken)
for HOST in <hostname-1> ...; do
  curl -s -H "Authorization: Bearer $API" \
    "https://api.cloudflare.com/client/v4/zones/$ZONE/dns_records?name=$HOST" | jq -r ".result[].id" | \
  while read -r ID; do curl -s -X DELETE -H "Authorization: Bearer $API" \
    "https://api.cloudflare.com/client/v4/zones/$ZONE/dns_records/$ID" | jq -r ".success"; done
done

# 清理配置
rm -rf /etc/cloudflared/ /root/.cloudflared/

# 卸载二进制
apt remove -y cloudflared 2>/dev/null || rpm -e cloudflared 2>/dev/null

# 确认 DNS 已清理
for HOST in <hostname-1> ...; do dig "$HOST" +short; done
```

## 已知边界情况

- **每个隧道无 hostname 数量硬限制**，受 DNS 记录数约束
- **流量完全免费**
- **不支持纯 TCP 服务**（SSH/RDP/MySQL）走普通 ingress，需要客户端装 `cloudflared access`
- **不支持 UDP 服务**（除非 WARP Connector）
- **WebSocket 自动支持**（透明转发）
- **Tunnel 出口流量从你的服务器出去**，仍消耗服务器带宽和上行
````
