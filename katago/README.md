# KataGo 文件放置目录

本项目不会把 KataGo 可执行文件和大型神经网络模型提交到 Git 仓库。

## 当前工作区配置

本机已经安装并验证以下内容（均被 `.gitignore` 排除）：

- `katago.exe`：KataGo v1.18.2，CUDA 13.2 / cuDNN 9.24 Windows x64
- `kata1-tf2-b10c384-s2941M-d5872M.bin.gz`：桌面对局使用的成熟主网络
- `source/`：固定在官方 v1.18.2 标签，供自我对弈、洗牌、PyTorch 训练和
  模型导出使用

官方 Windows CUDA 引擎所需的 CUDA/cuDNN DLL 由项目虚拟环境中的
`torch/lib` 提供，桌面程序和 `train_rl.py` 都会自动补充进程的 `PATH`，
无需再安装一份完整 CUDA Toolkit。强化学习用法见根目录 `README.md`。

可以把下列文件放在本目录，程序会在启动时自动发现：

- Windows：`katago.exe`
- 主网络：任意非 `human` 命名的 `*.bin.gz`
- 可选人类风格网络：文件名包含 `human` 的 `*.bin.gz`

也可以在程序顶部点击“KataGo 设置”，分别选择任意目录中的文件。设置会
保存在当前 Windows 用户的本地应用数据目录，而不是此 Git 仓库。

Windows 用户还可以在项目根目录运行 `./安装KataGo.ps1`，自动下载官方
OpenCL 引擎、主网络和 HumanSL 模型到本目录。旧版安装在
`vendor/katago/runtime` 与 `vendor/katago/models` 的文件也会继续被识别。

职业 1–9 段在安装人类风格网络后使用 KataGo 的 `proyear_2023` 现代
职业棋谱策略，并通过递增搜索量、逐级降低的选点温度和更严格的主网络评价
区分难度；没有安装时仍使用 KataGo 主网络。所有档位都不能视为真实职业
段位认证。

HumanSL 20 级到 9 段使用模型的 `rank_20k` 到 `rank_9d` profile，并直接
按对应人类棋谱策略采样非虚手落点；这些档位必须安装人类风格模型。

官方下载地址：

- 引擎：https://github.com/lightvector/KataGo/releases
- 主网络：https://katagotraining.org/
- 人类风格模型说明：https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide
