# KataGo 混合棋盘训练：本机可行性

后续更新：已于同日完成 9／13 路及 9／13／19 路真实混训、MPS 更新恢复和 Metal 加载验证，见[完整实测记录](KATAGO_M4_MIXED_PROBE_20261023.md)。以下保留执行前的环境快照；引擎与源码缺失问题已在独立实验中补齐，项目正式入口仍待接入。

核查日期：2026-10-23。本文是分析记录，没有安装 KataGo、修改训练实现或启动正式训练。上游源码链接指向当天读取的 `master`，它不是本项目已经验收的固定训练源码版本。

## 结论

本机有条件做小规模混合棋盘实验；目前项目不能直接启动这套流程。应先补齐兼容的 macOS KataGo 引擎和官方训练源码，接入 Metal 自我对弈、MPS 训练，以及统一尺寸的数据通路，再做短闭环实测。尚无证据支持承诺从零训练到高棋力所需时间，或把 Windows RTX 5070 Ti 的吞吐、批次与精度设置搬到本机。

本次父任务实测：Mac mini M4，10 核 CPU / 10 核 GPU，16 GiB 统一内存，约 122 GiB 可用磁盘；`.venv` 为 Python 3.12.14 / PyTorch 2.14.1，MPS 构建与设备可用，CUDA 不可用。MPS 建议内存上限约 11.84 GiB，它不是独占可用显存。小型合成卷积网络 `[2,22,19,19]` 完成 MPS 前向、反向及 AdamW 一步，只证明基本训练算子可用，没有验证 KataGo 完整网络、损失函数、导出或吞吐。`katago/katago`、`katago/cpu/katago`、`katago/source` 和全局 KataGo 可执行文件均不存在。

## 官方本来支持同一模型混合棋盘

官方方法是把较小棋盘补零到批次的最大尺寸，用棋盘掩码区分真实交叉点与填充区域，并在归一化、池化中使用真实面积；这是同一网络、同一批次可以包含不同棋盘尺寸的依据。[官方方法说明，第 16–38 行](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#training-on-multiple-board-sizes-via-masking)。当前模型从输入第 0 通道提取掩码并求实际面积，[`model_pytorch.py` 第 3547–3549 行](https://github.com/lightvector/KataGo/blob/master/python/katago/train/model_pytorch.py#L3547-L3549)。

官方配置已经使用 `bSizes` 和 `bSizeRelProbs` 设定多尺寸；`dataBoardLen` 则必须不小于最大棋盘，并与 `train.py -pos-len` 相等，不同 `dataBoardLen` 的数据不能直接混洗。[`selfplay1.cfg` 第 8–13、78–80 行](https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/selfplay1.cfg#L8-L13)。数据读取也按唯一的 `pos_len` 校验打包长度和重塑张量，[`data_processing_pytorch.py` 第 80–88 行](https://github.com/lightvector/KataGo/blob/master/python/katago/train/data_processing_pytorch.py#L80-L88)。

因此若目标是 9 / 13 / 19 路混训，方案应是新生成的三种棋盘统一 `dataBoardLen=19`、训练统一 `-pos-len 19`。不能把已有原生 9×9、13×13、19×19 NPZ 简单合并。小棋盘仍占 19×19 张量空间；这是表示面积的差异，不能据此推算实际速度倍数。

配置中的相对权重是开局棋盘抽样权重，不是训练样本比例：源码在初始化棋局时抽取尺寸；不同局长、分叉与样本权重都会影响最终回放组成。若只要正方形应关闭矩形概率；任何建议比例都需后续按实际各尺寸的局数和训练行数检查。[`play.cpp` 第 124–159、441–484 行](https://github.com/lightvector/KataGo/blob/master/cpp/program/play.cpp#L124-L159)。

## Apple Silicon 的官方路径与尚待验证部分

| 环节 | 官方证据 | 对本机的含义 |
|---|---|---|
| 自我对弈 / 搜索推理 | 官方 macOS 编译文档推荐 `METAL`，并保留 `EIGEN` 选项 | 优先验证 Metal 引擎；纯 CPU Eigen 可作后备 |
| 神经网络训练 | 当前上游 `train.py` 明确按 CUDA → MPS → CPU 选择设备 | 上游已有 MPS 入口，不能称“训练必须 CUDA” |
| 混合精度 | 当前训练 AMP 分支仍使用 `autocast("cuda")` / `GradScaler("cuda")` | 首次验收应从 FP32、单设备、`-no-compile` 开始 |
| 本项目接入 | 硬件识别只检查 CUDA；doctor 按 CUDA / Eigen 校验 | 仍需添加 MPS 与 Metal 的独立支持和实机验收 |

来源：[官方 macOS 编译说明，第 203–229 行](https://github.com/lightvector/KataGo/blob/master/Compiling.md#macos)；[官方训练设备分支，第 479–495 行](https://github.com/lightvector/KataGo/blob/master/python/train.py#L479-L495)；[AMP 分支，第 1433–1440、1551–1565 行](https://github.com/lightvector/KataGo/blob/master/python/train.py#L1433-L1440)。PyTorch 官方说明 `mps` 是 macOS GPU 训练设备，[PyTorch 2.14 MPS 文档](https://docs.pytorch.org/docs/2.14/notes/mps.html)。

Metal 引擎负责搜索中的网络推理；PyTorch MPS 负责训练反向传播，两者是不同组件。引擎支持 Metal 不自动证明训练端已兼容。官方存在 MPS 设备分支同样不等于本机完整 KataGo 训练已经成功：真实损失、优化器、检查点、导出并重新加载均应各验证一次。Apple Neural Engine 的引擎推理能力也不能算作 PyTorch 训练算力。官方还明确 Apple Silicon 不原生支持 AVX2，不能使用 Windows x64 Eigen 安装包或照搬 AVX2 编译参数。[macOS 编译说明](https://github.com/lightvector/KataGo/blob/master/Compiling.md#macos)。

## 当前项目需要改的接缝

- `weiqi/katago_hardware.py:159` 的 `detect_hardware` 仅检测 CUDA，本机自动落入 CPU 档；CPU 档在 9 / 13 / 19 路批次上限分别为 16 / 8 / 4，不代表 MPS 的实测上限。
- `weiqi/katago_rl.py:125` 把 `dataBoardLen`、`bSizes` 都写为单个 `game.board_size`；`:517` 的 `-pos-len` 同样绑定该值。现有课程是按尺寸分阶段迁移，文档明确各阶段分开保存、禁止跨棋盘混洗，不能当成已实现混训。见 [`docs/KATAGO_TRAINING.md`](../KATAGO_TRAINING.md)。
- `weiqi/katago_rl.py:329` 的 `doctor` 对非 CPU 设备要求 CUDA，并把引擎后端期望设为 CUDA；不能只把配置字符串改为 `mps`。
- `weiqi/katago_rl.py:281` 的 CPU 子进程只屏蔽 CUDA；对当前上游自动选 MPS 的版本，这并不能强制真正 CPU 训练。应明确控制实际设备并验证日志。
- `scripts/install_katago_cpu.ps1:26` 固定的是 Windows Eigen **二进制** `v1.18.1`，不能据此断言缺失的 `katago/source` 训练源码版本。当前上游 Python 模块布局已位于 `python/katago/train/`，选择源码时必须核对现有启动器、模型格式与导出脚本的兼容性。

混训本身不要求重置模型或优化器：当前上游恢复时用请求的 `pos_len` 构造模型、加载模型与 SWA 状态，并尝试恢复同类型优化器；尺寸相关的目数向量是非持久缓冲区。是否能将具体旧检查点迁移到新尺寸，仍须以该检查点与选定源码实际加载为准，不应泛化到任意网络版本。[`train.py` 第 888–1005 行](https://github.com/lightvector/KataGo/blob/master/python/train.py#L888-L1005)；[`model_pytorch.py` 第 2611–2635 行](https://github.com/lightvector/KataGo/blob/master/python/katago/train/model_pytorch.py#L2611-L2635)。本项目现有课程主动重置优化器、阶段计数与数据记录，是编排策略，不是官方混合棋盘机制的硬性要求。

## 建议的验证顺序

1. 独立实验目录，选择与现有 `b10c128` 检查点兼容的源码和 macOS Metal 引擎，验证引擎真实后端与模型加载。
2. 先统一 13×13 张量做 9 / 13 路的小规模实验；若目标明确必须包含 19 路，再统一到 19×19。实验生成少量真实数据，并检查掩码、尺寸及各棋盘实际分布。
3. 小批次 FP32 / eager 完成真实 KataGo 损失、反向、优化器更新、检查点保存恢复、导出及引擎再加载；检测非有限损失与 CPU 回退。
4. 用整个闭环实测耗时、峰值统一内存、磁盘增长与各尺寸独立棋力，再决定是否长期运行。逐步调整批次、棋局线程和每手搜索，避免自我对弈与训练同时挤占 16 GiB 统一内存。

上述是后续实验建议，不是已经验收的参数。当前证据足以判断硬件有实验可行性；尚不足以判断完整训练吞吐和达到目标棋力的成本。
