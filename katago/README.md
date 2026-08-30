# KataGo 文件放置目录

本项目不会把 KataGo 可执行文件和大型神经网络模型提交到 Git 仓库。

可以把下列文件放在本目录，程序会在启动时自动发现：

- Windows：`katago.exe`
- 主网络：任意非 `human` 命名的 `*.bin.gz`
- 可选人类风格网络：文件名包含 `human` 的 `*.bin.gz`

也可以在程序顶部点击“KataGo 设置”，分别选择任意目录中的文件。设置会
保存在当前 Windows 用户的本地应用数据目录，而不是此 Git 仓库。

职业 1–9 段在安装人类风格网络后使用 KataGo 的 `proyear_2023` 现代
职业棋谱策略，并通过递增搜索量、逐级降低的选点温度和更严格的主网络评价
区分难度；没有安装时仍使用 KataGo 主网络。所有档位都不能视为真实职业
段位认证。

官方下载地址：

- 引擎：https://github.com/lightvector/KataGo/releases
- 主网络：https://katagotraining.org/
- 人类风格模型说明：https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide
