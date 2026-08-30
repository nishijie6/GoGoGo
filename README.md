# 弈境 · Python 围棋

一个使用 Python/Tkinter 编写的本地围棋程序。简单、中等、困难模式无需
安装第三方 Python 依赖；职业棋手与 HumanSL 人类级段位模拟由用户自行配置的
KataGo 引擎提供。

## 功能

- 人机对战：可选择执黑或执白，电脑在后台思考，不会卡住界面
- 三组电脑选择：内置简单/中等/难、KataGo 模拟职业 1–9 段、HumanSL 模拟 20 级到 9 段
- 级段位档位全部由 KataGo/HumanSL 计算，不再把内置启发式 AI 标成业余或职业棋力
- 可选 KataGo 人类风格模型：使用 `proyear_2023` 现代职业棋谱策略
- 双人对战：两名玩家在同一台电脑上轮流落子
- 9×9、13×13、19×19 三种棋盘
- 完整的落子、提子、禁入点和全局同形（劫争）检查
- 虚手、连续两次虚手终局、认输和悔棋
- 对局内推理模式：保存正式棋局后临时推演黑白双方变化，支持逐手撤回，退出即恢复正式棋局
- 中国数子法自动计分，白方贴 6.5 目
- 每手实时显示黑白胜率、局面阶段和预计领先目数
- 程序内置完整中文围棋规则，可点击顶部“围棋规则”或按 `F1` 查看
- 推理训练：内置常见定式与基础死活题，支持点击试解、提示、逐手演示、回退和重置
- 每个训练落子都有“为什么这样下”的中文棋理解说；定式明确标注为局部参考次序
- 已接入 KataGo 官方强化学习流水线，当前从 9×9 开始，支持数据洗牌、
  CUDA/BF16 训练、模型导出、断点续训及后续独立的 19×19 阶段
- 棋谱列表、坐标、星位、上一手标记和落子预览

## 启动

需要 Python 3.9 或更新版本（已在 Python 3.11 测试）。

在项目目录打开 PowerShell：

```powershell
python main.py
```

Windows 也可以直接双击 `启动围棋.bat`。

不配置 KataGo 时，程序的简单、中等、困难人机模式和双人模式仍可正常使用。

## KataGo 与 HumanSL 模拟

KataGo 的可执行文件和神经网络体积较大、且需要按操作系统和显卡选择版本，
因此不会直接提交到本仓库。配置方法如下：

1. 从 [KataGo 官方 Releases](https://github.com/lightvector/KataGo/releases)
   下载适合电脑的版本。一般显卡可以先尝试 OpenCL；没有合适显卡时可使用
   Eigen CPU 版本。
2. 从 [KataGo Training](https://katagotraining.org/) 下载主神经网络
   `*.bin.gz`。官方建议通常选择较新、较强的 b18 网络。
3. 如果希望模仿真人段位行棋，再按照
   [KataGo Human SL 官方说明](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide)
   下载人类风格模型，例如 `b18c384nbt-humanv0.bin.gz`。
4. 启动本程序，点击顶部“KataGo 设置”，依次选择可执行文件、主网络和
   可选的人类风格网络，然后保存。
5. 在电脑难度中选择“KataGo 模拟职业1段”至“职业9段”并开始新局。

Windows 用户也可以在仓库根目录运行 `./安装KataGo.ps1`。脚本会从 KataGo
官方 GitHub Release 下载 OpenCL 引擎、小型主网络和 HumanSL 模型，校验可用的
SHA-256，并放入已被 Git 忽略的 `katago/` 目录；之后程序会自动识别这些文件。

也可以把文件直接放入项目的 `katago/` 目录：程序会自动识别
`katago.exe`、非 `human` 命名的主网络，以及文件名包含 `human` 的人类
风格网络。程序也会继续识别旧版 `vendor/katago/runtime` 和
`vendor/katago/models` 安装，不要求已有用户重新下载或复制。具体见
[`katago/README.md`](katago/README.md)。

程序通过 KataGo 官方 JSON 分析协议传递完整行棋历史，并使用与本程序一致的
9×9、13×13、19×19 棋盘、6.5 贴目、面积计分、禁自杀和位置全局同形规则。
职业段位从低到高设置 24、36、54、80、120、180、270、400、600 次最大
搜索访问量。首次启动 OpenCL 版本时可能先自动调优显卡，这是正常现象。

安装人类风格模型时，程序使用官方 `proyear_2023` 现代职业棋谱先验，
再结合主网络搜索评价选择合法落点；职业段位越高，访问量越大、随机温度越低、
对主网络评价的要求越严格。主网络也负责判断何时虚手以及估计胜率。未安装
人类风格模型时直接选择主网络最高评价落点。职业段位是面向训练和娱乐的模拟
档位，不是棋院认证，也无法保证与每位真实职业棋手的实际水平完全一致。

选择 `HumanSL 20级（模拟）` 到 `HumanSL 9段（模拟）` 时，程序使用官方
`rank_20k` 到 `rank_9d` profile。普通搜索用 64 次访问判断是否应当虚手；
非虚手落点从对应水平的 `humanPolicy` 中按概率采样，以保留该水平常见的选择
和失误分布。它模拟的是棋谱风格，不是棋力认证；这些 profile 主要来自 19×19
人类棋谱，在 9×9 和 13×13 上只应视作风格模拟。

## 操作

- 鼠标左键：在棋盘交叉点落子
- `Ctrl+N`：按当前设置开始新局
- `Ctrl+Z`：悔棋；人机模式通常会同时撤回玩家和电脑的一手
- `P`：虚手
- `F1`：打开程序内的围棋规则说明
- `F2`：打开“推理训练”，学习定式和死活题
- `F3`：开启或退出对局内“推理模式”；推演不会写入正式棋谱
- 推理训练内：鼠标猜落点，`H` 显示提示，`Space` 演示下一手，`Esc` 关闭
- 连续两次虚手后，程序按当前盘面自动计分

## 推理模式

在任意未结束的对局中点击顶部“开启推理”或按 `F3`，程序会把当前正式棋局
保留在内存中，并切换到一条隔离的临时变化。推理模式中黑白双方都由你落子，
可以使用落子、虚手、认输与 `Ctrl+Z` 撤回；撤回最多回到开启推理时的保存点，
不会越界删除正式棋谱。再次按 `F3` 会丢弃当前变化、恢复原棋局；若恢复时正轮到
电脑，电脑会从恢复后的正式局面继续思考。

## 推理训练

顶部“推理训练”窗口目前包含两套星位定式参考型，以及直三、曲三、
梅花五等基础死活课程。选择课程后可直接在棋盘猜下一手；错误落点不会
改变棋盘，正确落点或演示答案会显示该手的局部目的和后续思考题。

这里的“推理”指可读的教学棋理，例如方向、根据、眼位、气和断点，
不展示模型的隐藏思维过程。定式不是全盘唯一答案；实际选择仍需结合
周围子力、征子与行棋方向。

## KataGo 强化学习（RTX 5070 Ti）

本工作区已接入 KataGo 官方的单机训练流程：自我对弈生成数据、洗牌回放窗口、
PyTorch 更新网络、导出 KataGo 二进制模型，再由新模型继续自我对弈。训练入口
是 `train_rl.py`，本机专用配置为 `config/rl_training.rtx5070ti.json`。

课程训练会先接管并续跑现有 9×9 检查点，然后依次进入 13×13 和 19×19；
不会重置已有的 9×9 数据。三个阶段共用 `b10c128` 网络和迁移后的 SWA 权重，
但自我对弈、洗牌与检查点始终保存在各自棋盘目录中，禁止跨棋盘混洗。
9×9 的本机配置按 RTX 5070 Ti 16 GB 的实测结果设置：

| 项目 | 本机配置 |
|---|---:|
| 棋盘与规则 | 9×9、中国面积计分、位置全局同形、贴 6.5 目 |
| 数据张量 | `dataBoardLen=9`，不与 19×19 数据混用 |
| 网络 | 官方 `b10c128`，10 个残差块、128 通道，约 296 万参数 |
| 训练精度 | CUDA BF16 + TF32 |
| 训练批次 | 每 GPU 1024 |
| 每手搜索 | 200 visits |
| 自我对弈 | 128 个并行棋局线程、推理批次 128、每轮 128 局 |
| 推理后端 | CUDA FP16 + NHWC、2 个神经网络服务线程 |
| 每轮训练上限 | 4 步（首轮洗牌集也能形成完整 epoch） |
| 训练目录 | `training_runs/9x9/` |

本机训练吞吐实测如下，最终选择 1024 批次：

| BF16 批次 | 吞吐 | 峰值训练显存 |
|---:|---:|---:|
| 256 | 11,993 样本/秒 | 0.54 GiB |
| 512 | 16,485 样本/秒 | 0.98 GiB |
| 1024 | 17,609 样本/秒 | 1.84 GiB |

Windows 上当前 PyTorch 构建没有可供 `torch.compile` 使用的 Triton，因此本机
配置采用已实测稳定的 BF16 eager 路径；这不会关闭 CUDA、Tensor Core 或
TF32。KataGo 推理仍使用 FP16/NHWC。运行前可检查全部依赖：

```powershell
.\.venv\Scripts\python.exe train_rl.py doctor
.\.venv\Scripts\python.exe train_rl.py status
```

执行一次短闭环或仅持续当前棋盘训练：

```powershell
# 少量对局、低 visits，验证所有阶段
.\.venv\Scripts\python.exe train_rl.py cycle --smoke

# 按 RTX 配置持续训练；Ctrl+C 可安全停止
.\.venv\Scripts\python.exe train_rl.py continuous
```

推荐使用自动课程编排器长期训练：

```powershell
.\.venv\Scripts\python.exe train_rl.py curriculum `
  --curriculum-config config\rl_curriculum.rtx5070ti.json

# 只读取状态，不启动第二份训练
.\.venv\Scripts\python.exe train_rl.py curriculum-status
```

编排器在 9×9 达到最低 1000 万阶段样本且连续通过质量评测后，自动迁移到
13×13；13×13 新增至少 2500 万阶段样本并达标后迁移到 19×19。19×19
没有结束样本数，会持续训练并以滚动冠军评测记录胜率与近似 Elo，直到手动停止。
阶段迁移继承上一阶段的 SWA 权重，但会重置优化器、阶段计数和数据记录。
每新增 50 万样本执行固定 200 局评测；未达标或评测失败只会留在当前棋盘续训，
不会强制切换或破坏现有检查点。评测使用持久化的 100 个确定性开局，每个开局
交换候选与基准的黑白方各下一局，避免把同一空棋盘轨迹重复 200 次当成独立样本。
自我对弈关闭弱模型的原始策略开局初始化，探索仍由 MCTS 温度和 25% 根噪声提供。

正式接管前可用当前 SWA 检查点分别验证 13×13、19×19 的模型加载、4 局低
visits 自我对弈及单批训练；验证产物与正式阶段数据完全隔离：

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_curriculum_gpu.py `
  --source-checkpoint .\training_runs\9x9\torchmodels_toexport\<模型名>\model.ckpt
```

持续训练每接纳一个新模型都会原子更新本地监控面板和结构化历史。首次启用或需要
从已有检查点补录时运行：

```powershell
.\.venv\Scripts\python.exe train_rl.py dashboard
Start-Process .\training_runs\9x9\dashboard.html
```

面板每 15 秒自动刷新，展示累计训练样本、自我对弈数据量、每轮新增数据、官方
总损失 EMA，以及策略/价值/目数损失分量。原始记录同时保存在
`training_runs/9x9/metrics/history.csv`、`history.jsonl` 和 `latest.json`，可直接
用于 Excel、Python 或后续实验分析。损失趋势反映对当前训练目标的拟合情况，
不等同于 Elo 或实际棋力；课程面板中的固定条件模型对战用于判断实际进步。

课程和各棋盘面板可以同时查看；13×13、19×19 面板会在对应阶段首次启动后生成：

```powershell
Start-Process .\training_runs\curriculum\dashboard.html
Start-Process .\training_runs\9x9\dashboard.html
Start-Process .\training_runs\13x13\dashboard.html
Start-Process .\training_runs\19x19\dashboard.html
```

全局面板显示当前棋盘、切换门槛、预计剩余样本、评测胜率/Elo、黑白胜负、
双停率、极端棋局率和磁盘状态；棋盘面板保留该阶段的损失曲线和对局统计。
原子课程状态保存在 `training_runs/curriculum/state.json`，启动日志位于
`training_runs/curriculum/logs/`，各阶段原始指标位于
`training_runs/<棋盘>/metrics/`。可随时用 `curriculum-status` 查看状态；全局锁会
拒绝重复实例。

训练有互斥锁，不能误启两个进程写同一检查点。中断后再次执行 `continuous`
会从 `train/gogogo/checkpoint.ckpt` 续训；自我对弈、洗牌数据、日志、待导出
检查点和已接纳模型分别保存在训练目录的对应子目录。首次没有模型时由 KataGo
随机启动器生成种子数据。这个单显卡初始训练采用无 gatekeeper 模式，新导出
模型会直接接纳，以便更快完成早期迭代。

若希望登录 Windows 后自动续训，可为当前用户注册计划任务。任务隐藏启动、忽略
重复实例、异常退出 5 分钟后重试，且没有运行时限。任务还包含每 5 分钟一次的
轻量守护触发；正常运行时会因 `IgnoreNew` 自动跳过，外部中断未被 Windows 识别
为失败时则会在 5 分钟内恢复：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\register_curriculum_task.ps1 -StartNow
```

交互运行时按 `Ctrl+C` 会在安全边界保留可恢复产物，再次执行 `curriculum` 即可
恢复。若要暂停后台计划任务并防止自动重启，可执行：

```powershell
Stop-ScheduledTask -TaskName GoGoGo-KataGo-Curriculum
Disable-ScheduledTask -TaskName GoGoGo-KataGo-Curriculum

# 恢复后台训练
Enable-ScheduledTask -TaskName GoGoGo-KataGo-Curriculum
Start-ScheduledTask -TaskName GoGoGo-KataGo-Curriculum
```

电脑休眠期间训练自然暂停，唤醒后继续；注册脚本不会修改 Windows 电源计划。
剩余磁盘低于 50 GiB 时编排器会主动清理可回收的旧模型、检查点和 NPZ 回放窗；
低于 30 GiB 时会安全暂停并在全局面板报警。SGF、CSV、JSONL 记录会保留。
每个棋盘目录中的 `replay_rows.json` 原子记录已清理 NPZ 的累计行数，并传给
KataGo 官方 `shuffle.py -add-to-data-rows`，因此 50 万行物理回放窗口滚动后，
检查点和模型名中的逻辑数据水位仍会单调增长。不要手工修改或删除这个账本；
账本损坏时编排器会停止而不是用错误水位继续训练。

旧的 19×19 `b15c192` 试验数据和检查点仍保存在
`training_runs/high_performance/`，配置为
`config/rl_training.rtx5070ti.19x19.json`。它是独立的历史试验，不会进入新的
`b10c128` 课程训练；课程版 19×19 使用
`config/rl_training.rtx5070ti.curriculum.19x19.json`。

自行从零训练出的模型一开始很弱，达到成熟围棋网络的棋力需要大量 GPU 时间和
自我对弈数据。桌面程序默认使用 `katago/` 中下载的成熟主网络；本地强化学习
模型则在独立训练目录中持续成长，不会意外替换正常对局模型。

通用配置 `config/rl_training.json` 和高性能示例仍可用于覆盖参数。加载器会
拒绝拼错字段、无效棋盘尺寸、超过 100% 的显存比例，以及与当前规则引擎不一致
的计分或劫争规则。

## 规则说明

程序采用中国面积计分：棋盘上的活子数与围住的空点数相加，白方再加
6.5 目贴目。提子数只作展示，不重复计入面积得分。

程序不会自动判断复杂死活。准备结束时，请先在棋盘上把双方认可的死子
提净，再由双方连续虚手。劫争使用全局同形禁着，因此也能避免三劫循环。

内置简单、中等、困难电脑对手是快速启发式 AI，适合轻量休闲对局，并非
专业棋力引擎。KataGo 职业与 HumanSL 级段位必须成功启动外部 KataGo 进程；如果
文件缺失或引擎报错，程序会保留棋盘并打开设置，不会静默降级成内置弱 AI。

界面在每一步后都会立即更新胜率。普通模式及 KataGo 尚在思考时使用本地
启发式估算；KataGo 返回电脑着手后，界面优先显示该着手对应的 KataGo
神经网络胜率、领先目数和实际搜索访问量，并明确标注数据来源。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖三种尺寸、提子、自杀禁手、劫争、虚手终局、计分、悔棋、推理分支
隔离与恢复、AI 合法性、实时胜率估算、KataGo 坐标/棋谱/规则协议、职业段位与
HumanSL profile、CUDA 运行时发现、强化学习参数映射、配置读写、内置规则内容，
以及定式/死活课程的逐手合法回放。

## 项目结构

```text
GoGoGo/
├── main.py              # 桌面程序启动入口
├── 安装KataGo.ps1       # Windows 官方引擎与模型安装脚本
├── train_rl.py          # KataGo 强化学习命令行入口
├── requirements-rl.txt  # CUDA PyTorch 训练依赖
├── weiqi/
│   ├── engine.py        # 围棋规则与计分
│   ├── ai.py            # 本地启发式电脑对手
│   ├── katago.py        # KataGo JSON 协议、职业段位和进程管理
│   ├── katago_rl.py     # 官方自我对弈/洗牌/训练/导出编排
│   ├── replay_accounting.py # NPZ 清理后的原子累计行账本
│   ├── rl_curriculum.py # 课程配置、质量门槛、迁移与保留规则
│   ├── rl_curriculum_runtime.py # 可恢复课程运行时和全局面板
│   ├── rl_metrics.py    # 检查点指标、CSV/JSONL 历史和本地 HTML 图表
│   ├── katago_gui.py    # KataGo 文件配置窗口
│   ├── winrate.py       # 实时胜率与领先目数估算
│   ├── rl_config.py     # 强化学习默认/高性能预设、覆盖和校验
│   ├── reasoning.py     # 推理模式的正式棋局快照与临时变化隔离
│   ├── rules.py         # 程序内中文围棋规则内容
│   ├── training.py      # 定式/死活课程数据与局部次序回放
│   ├── training_gui.py  # 交互式推理训练窗口
│   └── gui.py           # Tkinter 界面
├── config/
│   ├── katago_analysis.cfg # 低内存、单局面 KataGo 分析配置
│   ├── rl_training.json # 普通电脑默认强化学习配置
│   ├── rl_training.high_performance.example.json # 高性能显卡示例
│   ├── rl_training.rtx5070ti.json # 当前 9×9 本机实测配置
│   ├── rl_training.rtx5070ti.13x13.json # 13×13 课程阶段
│   ├── rl_training.rtx5070ti.curriculum.19x19.json # 19×19 课程阶段
│   ├── rl_curriculum.rtx5070ti.json # 9→13→19 课程与门槛
│   └── rl_training.rtx5070ti.19x19.json # 保留的旧 19×19 试验配置
├── scripts/
│   ├── start_curriculum.ps1 # 隐藏计划任务入口与日志
│   ├── register_curriculum_task.ps1 # 当前用户登录自启注册
│   └── validate_curriculum_gpu.py # 13/19 GPU 冒烟验证
├── katago/
│   └── README.md        # 可选引擎和模型的本地放置说明
└── tests/
    ├── test_engine.py   # 规则与电脑对手测试
    ├── test_winrate.py  # 实时胜率估算测试
    ├── test_katago.py   # KataGo 协议、配置和职业段位测试
    ├── test_installer.py # Windows 安装器固定版本、摘要和目录安全测试
    ├── test_katago_rl.py # 官方强化学习流水线参数测试
    ├── test_replay_accounting.py # 回放累计行与中断恢复测试
    ├── test_rl_curriculum.py # 课程状态、门槛、评测与迁移测试
    ├── test_rl_curriculum_runtime.py # 编排、磁盘清理和恢复测试
    ├── test_rl_metrics.py # 训练指标解析、持久化与图表测试
    ├── test_rl_config.py # 强化学习配置与安全校验测试
    ├── test_reasoning.py # 推理分支、撤回边界与正式棋局恢复测试
    ├── test_rules.py    # 程序内规则内容测试
    └── test_training.py # 定式/死活课程与回放测试
```
