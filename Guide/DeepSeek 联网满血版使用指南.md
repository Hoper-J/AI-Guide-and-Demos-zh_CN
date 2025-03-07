# DeepSeek 联网满血版使用指南

> **基于 Cherry Studio / Chatbox 进行配置，绕开 DeepSeek 网页对话的卡顿**。
>
> 需要特别说明的是，本文不涉及本地显卡部署方案，因为本地部署完整版 DeepSeek 对于个人使用者来说是一个伪需求，671B 的模型即便是 1-Bit 量化加载也需要约 84GB 的显存（模型参数量×量化位数/8），按显卡租赁的费用来算，每小时大概需要 8 块钱，而 1-Bit 量化模型基本不可用，满血版 DeepSeek（BF16）仅加载就需要 1342GB 显存，这意味着更高的租赁费用，对于个人来说投入和回报完全不成正比，所以使用 API 将是性价比最高的方案（各平台注册时都会赠送大量的 tokens：阿里 1000 万，百度 2000 万，硅基流动 2000 万）。
>
> **注意**：当前说法不适用于需要数据隔离的场景（比如金融/医疗），仅针对日常需求。

## ▌目录

- [为什么有网页版还要配置 API？](#为什么有网页版还要配置-api)
- [Cherry Studio 配置【推荐】](#cherry-studio-配置推荐)
   - [DeepSeek](#deepseek)
   - [添加自定义提供方](#添加自定义提供方)
      - [多平台参数对照表](#多平台参数对照表)
   - [QA](#qa)
      - [Q1：自定义提供方如何增加模型？](#q1自定义提供方如何增加模型)
      - [Q2：如何修改参数？](#q2如何修改参数)
      - [Q3：如何备份数据？](#q3如何备份数据)
      - [Q4：怎么修改默认模型？](#q4怎么修改默认模型)
      - [Q5：怎么修改默认助手的模版？](#q5怎么修改默认助手的模版)
- [Chatbox 配置](#chatbox-配置)
   - [DeepSeek](#deepseek-1)
   - [添加自定义提供方](#添加自定义提供方-1)
      - [多平台参数对照表](#多平台参数对照表-1)
   - [QA](#qa-1)
      - [Q1：自定义提供方如何增加模型？](#q1自定义提供方如何增加模型-1)
      - [Q2：如何切换平台？](#q2如何切换平台)
      - [Q3：如何导出或者分享聊天记录？](#q3如何导出或者分享聊天记录)
      - [Q4：温度（Temperature）应该怎么设置？](#q4温度temperature应该怎么设置)

## ▌为什么有网页版还要配置 API？

主要原因有五点：

1. 网页版（深度思考 - R1）由于访问过多经常服务器繁忙导致对话卡顿。
2. 可以在多个平台间进行无缝切换（聊天记录会“同步”），这意味着当前服务瘫痪时可以直接更换平台，从而减少等待时间。
3. 注册平台所赠送的 tokens 足以覆盖个人长期使用需求。
4. 可以自定义系统消息（System message），拥有更多的定制空间（注意，官方并不建议在使用推理模型的时候添加系统消息，对于该点需谨慎使用）。
5. 本地存储完整对话历史。

## ▌Cherry Studio 配置【推荐】

> 适用于电脑端，手机端可以尝试 [Chatbox](#chatbox-配置)。

访问[下载界面](https://cherry-ai.com/download)，选择合适的版本下载。

![image-20250210131143231](./assets/image-20250210131143231.png)

下载后打开，将看到一个清爽的界面，接下来点击左下角的 `设置图标`：

![Cherry Studio](./assets/image-20250210132307407.png)

### DeepSeek

如果已经获取了 DeepSeek 的 API（[获取步骤](./DeepSeek%20API%20的获取与对话示例.md#-deepseek-官方)），则从`设置` - `模型服务`界面中选择 `深度求索`，直接填充 API 密钥：

![填写API_Key](./assets/image-20250210131830180.png)

填写 API 密钥之后点击 `检查`，随意使用一个模型：

![检查 API](./assets/image-20250210132606271.png)

显示 `连接成功` 就意味着 API 可用，此时点击右上角 `开关按钮`，将其打开：

![image-20250210132909653](./assets/image-20250210132909653.png)

通过左边栏回到聊天界面，你可以在界面上方对平台/模型进行切换：

![切换模型](./assets/image-20250210133343815.png)

至此已经可以和 DeepSeek-R1 进行对话。

### 添加自定义提供方

通过 `设置` - `模型服务` - `+ 添加`：

![添加提供商](./assets/image-20250210133645807.png)

#### 多平台参数对照表

参照下表填写对应平台信息，最终呈现如右图：

|                       | 获取步骤                                                     | API 域名                                           | 模型 - 聊天             | 模型 - 推理             | 设置 - 推理                                              |
| --------------------- | ------------------------------------------------------------ | -------------------------------------------------- | ----------------------- | ----------------------- | -------------------------------------------------------- |
| 硅基流动              | [图文](./DeepSeek%20API%20的获取与对话示例.md#-硅基流动-)    | https://api.siliconflow.cn                         | deepseek-ai/DeepSeek-V3 | deepseek-ai/DeepSeek-R1 | ![设置 - 硅基流动](./assets/image-20250210134851206.png) |
| 阿里云百炼            | [图文](./DeepSeek%20API%20的获取与对话示例.md#-阿里云百炼-)  | https://dashscope.aliyuncs.com/compatible-mode/v1/ | deepseek-v3             | deepseek-r1             | ![设置 - 阿里云](./assets/image-20250210135415680.png)   |
| 百度智能云/百度云千帆 | [图文](./DeepSeek%20API%20的获取与对话示例.md#-百度智能云-)  | https://qianfan.baidubce.com/v2/                   | deepseek-v3             | deepseek-r1             | ![设置 - 百度](./assets/image-20250210140222077.png)     |
| 字节火山引擎          | [图文](./DeepSeek%20API%20的获取与对话示例.md#-字节火山引擎-) | https://ark.cn-beijing.volces.com/api/v3/          | "deepseek-v3-241226"    | "deepseek-r1-250120"    | ![设置 - 字节](./assets/image-20250210141105629.png)     |

### QA

#### Q1：自定义提供方如何增加模型？

以 `硅基流动` 为例，在右侧卡片底端点击 `添加`，对照上表输入模型 ID，然后点击保存。

![image-20250210141512888](./assets/image-20250210141512888.png)

填写 ID 后会自动复制填充 `模型名称` 和 `分组名称`，你可以定义 `分组名称` 使其与同类模型同组。

#### Q2：如何修改参数？

在聊天界面左侧点击 `设置`，可以修改 `模型温度` 和 `上下文数`：

![修改参数](./assets/image-20250210142152982.png)

参数 `上下文数` 意味着每次传入的对话轮数，过高会消耗大量的输入 token（对话历史 + 当前输入），你需要根据自己的实际需求去调节它。

> [!caution]
>
> 对于推理模型 `deepseek-reasoner`，传入参数 `temperature`、`top_p`、`presence_penalty`、`frequency_penalty`、`logprobs`、`top_logprobs` 均不会生效[^1]，故无需纠结温度设置。
>
> 也可以遵循官方的部署建议[^2]，将 `deepseek-reasoner` 的 `temperature` 默认设置为 0.6，以应对未来可能被允许的参数修改。

[^1]: [DeepSeek 官方文档](https://api-docs.deepseek.com/zh-cn/quick_start/parameter_settings).

[^2]: [Usage Recommendations - DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations).

#### Q3：如何备份数据？

左下角 `设置图标` -> `数据设置` -> `备份`：

![数据备份](./assets/image-20250210143452195.png)

#### Q4：怎么修改默认模型？

左下角 `设置图标` -> `默认模型` -> `默认助手模型`，从下拉框中进行选择：

![修改默认模型](./assets/image-20250210144519444.png)

#### Q5：怎么修改默认助手的模版？

左下角 `设置图标` -> `默认模型` -> `默认助手模型右侧设置图标`：

![修改默认助手](./assets/image-20250210144937894.png)

将弹出类似于下方的界面，此时可以修改默认助手的模版：

![默认助手](./assets/image-20250210144621718.png)

## ▌Chatbox 配置

根据系统下载对应版本的 APP[^3]：

- **电脑**
  - [Windows](https://chatboxai.app/?c=download-windows)
  - MacOS
    - [Intel 芯片](https://chatboxai.app/?c=download-mac-intel)
    - [M1/M2](https://chatboxai.app/?c=download-mac-aarch)
  - [Linux](https://chatboxai.app/?c=download-linux)
- **手机**
  - [苹果 / IOS](https://apps.apple.com/app/chatbox-ai/id6471368056)
  - [安卓/ Android](https://chatboxai.app/install?download=android_apk)

[^3]: [https://github.com/Bin-Huang/chatbox](https://github.com/Bin-Huang/chatbox).

下面以电脑端进行演示，打开 Chatbox，点击 `使用自己的 API Key 或本地模型`：

![使用 API Key](./assets/image-20250206112612860.png)

### DeepSeek

如果已经获取了 DeepSeek 的 API（[获取步骤](./DeepSeek%20API%20的获取与对话示例.md#-deepseek-官方)），则从打开的界面中选择 `DeepSeeK API`，直接填充 API 密钥，然后点击 `保存`：

![设置 DeepSeek API](./assets/image-20250206114220266.png)

至此已经可以和 DeepSeek-R1 进行对话。

### 添加自定义提供方

通过以下任一入口添加（点击 `添加自定义提供方`）：

| 首次使用时界面                                          | 已有配置时入口                                          |
| ------------------------------------------------------- | ------------------------------------------------------- |
| ![首次使用时界面](./assets/image-20250206112642637.png) | ![已有配置时入口](./assets/image-20250206120447013.png) |

#### 多平台参数对照表

参照下表填写对应平台信息，完成后点击保存，最终呈现如右图：

|              | 获取步骤                                                     | API 域名                                          | 模型 - 聊天             | 模型 - 推理             | 设置 - 推理                                              |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------- | ----------------------- | ----------------------- | -------------------------------------------------------- |
| 硅基流动     | [图文](./DeepSeek%20API%20的获取与对话示例.md#-硅基流动-)    | https://api.siliconflow.cn/v1                     | deepseek-ai/DeepSeek-V3 | deepseek-ai/DeepSeek-R1 | ![设置 - 硅基流动](./assets/image-20250206113805132.png) |
| 阿里云百炼   | [图文](./DeepSeek%20API%20的获取与对话示例.md#-阿里云百炼-)  | https://dashscope.aliyuncs.com/compatible-mode/v1 | deepseek-v3             | deepseek-r1             | ![设置 - 阿里云](./assets/image-20250206115713516.png)   |
| 百度智能云   | [图文](./DeepSeek%20API%20的获取与对话示例.md#-百度智能云-)  | https://qianfan.baidubce.com/v2                   | deepseek-v3             | deepseek-r1             | ![设置 - 百度](./assets/image-20250206120017418.png)     |
| 字节火山引擎 | [图文](./DeepSeek%20API%20的获取与对话示例.md#-字节火山引擎-) | https://ark.cn-beijing.volces.com/api/v3          | "deepseek-v3-241226"    | "deepseek-r1-250120"    | ![设置 - 字节](./assets/image-20250208215034817.png)     |

> [!note]
>
> 上表中的图例将 `上下文消息数量上限` 设置成了无限，这意味着每次都会传入完整的对话历史，并且消耗大量的输入 tokens，你需要根据自己的实际需求去调节它。

### QA

#### Q1：自定义提供方如何增加模型？

以 `硅基流动` 为例，在 `设置` 的模型中输入新的模型名称，然后点击 `+` 号并保存。

![image-20250206152625277](./assets/image-20250206152625277.png)

接下来可以在界面的右下角切换模型：
![切换模型](./assets/image-20250206152810904.png)

> [!note]
>
> 不同平台对于模型的标识可能不同。

#### Q2：如何切换平台？

点击 `设置`，从下拉选项中选择需要切换的平台，然后**点击** `保存`。

![切换平台](./assets/image-20250206152431349.png)

#### Q3：如何导出或者分享聊天记录？

`设置` -> `其他` -> `数据备份` -> 勾选需要导出的内容 -> `导出勾选数据`：

![image-20250206153355156](./assets/image-20250206153355156.png)

#### Q4：温度（Temperature）应该怎么设置？

默认情况为 1.0，实际使用可以遵循 [DeepSeek 官方文档](https://api-docs.deepseek.com/zh-cn/quick_start/parameter_settings)的建议，按使用场景设置：

| 场景                | 温度 |
| ------------------- | ---- |
| 代码生成/数学解题   | 0.0  |
| 数据抽取/分析       | 1.0  |
| 通用对话            | 1.3  |
| 翻译                | 1.3  |
| 创意类写作/诗歌创作 | 1.5  |

> [!CAUTION]
>
> 对于推理模型 `deepseek-reasoner`，传入参数 `temperature`、`top_p`、`presence_penalty`、`frequency_penalty`、`logprobs`、`top_logprobs` 均不会生效[^1]，故无需纠结。
>
> 可以遵循官方的部署建议[^2]，将 `deepseek-reasoner` 的 `temperature` 默认设置为 0.6，以应对未来可能被允许的参数修改。

**下一章**：[DeepSeek API 输出解析 - OpenAI SDK](./DeepSeek%20API%20输出解析%20-%20OpenAI%20SDK.md)

