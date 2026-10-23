# KataGo 训练与课程续训

## 多电脑轮流接力

`train_rl.py` 支持导出、校验和导入 KataGo 接力包；训练仍在当前电脑的本地
硬盘和 GPU 上运行。接力包包含完整续训检查点、训练数据、模型、课程状态和
配置，并记录文件 SHA-256 及代码版本。导入只准备新目录，不会自动启动训练。
导出成功后原训练目录会进入接力暂停状态，防止旧计划任务重新接管同一份训练。

先禁用原电脑的自动训练任务并停止训练，再执行：

```powershell
py train_rl.py handoff-export --bundle handoff_bundles\handoff-0001.gogogo-handoff.zip
py train_rl.py handoff-verify --bundle handoff_bundles\handoff-0001.gogogo-handoff.zip
```

把接力包复制到另一台电脑后，导入到尚不存在的新目录，并核对完整检查点：

```powershell
py train_rl.py handoff-import `
  --bundle handoff_bundles\handoff-0001.gogogo-handoff.zip `
  --destination training_runs\relay-0001
py train_rl.py handoff-check --destination training_runs\relay-0001 --load-checkpoint
```

使用导入结果中的 `resume_command` 在新电脑继续训练。第一次跨电脑恢复还应
确认实际训练日志中的累计训练计数延续，并确认成功完成一次更新。检查点读取
验证不会运行 GPU 训练，不能代替该验收。

完整交接步骤、不同显卡配置、单棋盘模式和撤销接力见
[多电脑 KataGo 接力训练](TRAINING_HANDOFF.md)。模型与接力包保持在本地；
这些命令没有 Git 提交、推送或云端上传动作。

## Windows 三档硬件适配

`train_rl.py` 和 `scripts/start_curriculum.ps1` 在启动训练时默认自动检测硬件。
检测会执行一次真实 CUDA 张量运算，选择可用显卡中显存最大的一张；检测失败则
使用 CPU。`status`、`curriculum-status` 等查看状态的命令不需要初始化 GPU。

| 档位 | 自动选择条件 | 自我对弈资源 | 9 / 13 / 19 路训练批次上限 |
|---|---|---|---|
| 无显卡 `cpu` | 没有当前 PyTorch 可用的 CUDA 设备 | 最多 4 个棋局线程、推理批次 1、每轮最多 4 局 | 16 / 8 / 4 |
| 低端显卡 `low` | 可用 CUDA 未达到高端条件 | 32 个棋局线程、推理批次最多 16、每轮最多 32 局 | 显存 ≤4 GiB：64 / 32 / 16；其余：256 / 128 / 64 |
| 高端显卡 `high` | 显存 ≥12 GiB 且计算能力 ≥8.0 | 保留当前配置的并发与对局数 | 1024 / 2048 / 1280 |

这些阈值是保守的资源预设，尚未覆盖所有显卡的实机性能测试。可用 BF16 的显卡
使用 BF16，否则计算能力 ≥7.0 使用 FP16，更早的设备使用 FP32；CPU 使用 FP32。
旧显卡的可用性还取决于安装的 CUDA/PyTorch 版本。AMD、Intel 或无法执行 CUDA
运算的 NVIDIA 显卡目前归入 CPU 档，不启用 OpenCL/DirectML 训练。

三档保留配置内的网络结构、搜索访问量、规则、评测局数和晋级门槛，原来的
`b10c128` 模型可续训。换到较小设备时，会限制课程状态中缓存的训练批次，而不
改写已保存的批次。GPU 阶段迁移按当前精度及显存限制测试候选批次；CPU 使用小
批次，并在发布迁移目录前实际验证一次 CPU 训练。CPU 的完整课程和 200 局评测
会很慢，适合小规模验证；这里没有降低评测预算来改变晋级条件。
小批量对弈会持续累计至原来的最低回放样本数，再进行训练；健康检查按实际局数
向前扩展记录窗口，仍要求默认至少 10 个训练周期、1280 局对局记录。

```powershell
# 只检测硬件，不启动训练
.\.venv\Scripts\python.exe train_rl.py hardware

# 自动选择档位并续训课程
.\.venv\Scripts\python.exe train_rl.py curriculum

# 手动限制资源；不会另起一套模型
.\.venv\Scripts\python.exe train_rl.py curriculum --hardware low
.\.venv\Scripts\python.exe train_rl.py curriculum --hardware cpu

# 完全沿用原配置（兼容此前的启动行为）
.\.venv\Scripts\python.exe train_rl.py curriculum --hardware configured
```

CPU 自我对弈需要单独的 Eigen 引擎，首次安装执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_katago_cpu.ps1
.\.venv\Scripts\python.exe train_rl.py doctor --hardware cpu
```

安装脚本校验固定发布包的 SHA-256，并将 CPU 引擎放到 `katago/cpu/`。使用官方
v1.18.1 非 AVX2 Eigen 包；[v1.18.2 发布说明](https://github.com/lightvector/KataGo/releases/tag/v1.18.2)
明确要求非 CUDA 后端继续使用 v1.18.1。CPU 档只在训练子进程中屏蔽 CUDA，现有
CUDA 引擎和环境仍可用于其他进程。纯 CPU 电脑可用独立虚拟环境安装
`requirements-rl-cpu.txt`；CUDA 电脑继续使用 `requirements-rl.txt`。两者都需要
`katago/source` 中已固定版本的官方训练源码。

后台启动脚本也接受 `-Hardware auto/cpu/low/high/configured`，默认 `auto`。
正在运行的训练保持本次启动时的设置，下次启动才应用自动检测结果。

## 本机配置与启动

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
# 少量对局、低 visits、至多 32 的训练批次，验证当前棋盘闭环
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
阶段迁移继承上一阶段滚动冠军的 SWA 权重，而不是最新候选；缺少冠军或其检查点时拒绝迁移。
已有迁移任务的恢复仍使用其已记录的源模型及哈希，以保持原子迁移身份一致。
迁移会重置优化器、阶段计数和数据记录。
新阶段首轮若打乱数据不足以填满自动选择的训练批次，会仅对该轮临时缩小批次，
避免 KataGo 零步退出；后续批次仍根据实际数据量与自动调优设置选择。
训练中期每累计 200 万训练样本，对滚动冠军和每个固定历史基准分别执行 200 局评测；
距离当前阶段最低样本数不足 100 万时恢复每 50 万一次，19×19 持续阶段每 200 万一次。
配置由 `evaluation_midstage_interval_samples`、`evaluation_near_gate_samples` 和
`evaluation_interval_samples` 控制；省略中期间隔时沿用每 50 万一次。
本机另启用 `champion_evaluation_every_cycle: true`：每个训练周期导出模型后，
立即与当前滚动冠军进行完整 200 局固定开局换色评测。上述样本间隔继续控制固定历史基准；
恰逢整套评测时复用其中的冠军挑战，不重复下 200 局。中断的冠军评测会先恢复完成，再继续训练。
冠军挑战仍按胜率至少 60% 且 Wilson 下界超过 50% 晋级，不会增加课程切换的连续达标次数。
监控按各棋盘阶段从 T0 开始编号，晋级后加 1；当前胜率严格匹配最新候选与当前 T 级冠军，
逐轮历史保存在课程状态及评测目录，并按冠军分段画线。旧轮次缺失的评测不会补造。
未达标或评测失败只会留在当前棋盘续训，
不会强制切换或破坏现有检查点。评测使用持久化的 100 个确定性开局，每个开局
交换候选与基准的黑白方各下一局，避免把同一空棋盘轨迹重复 200 次当成独立样本。
高端显卡默认同时执行 4 组换色开局（`evaluation_parallel_pairs`），CPU 和低端档保持串行。
已完成的配对会校验模型、规则、开局清单和实际 SGF 后复用；中断的配对在独立重试目录重下，
最终统计只取每个开局的一次完整结果。重启可继续同一候选尚未完成的整套评测。
自我对弈关闭弱模型的原始策略开局初始化，探索仍由 MCTS 温度和 25% 根噪声提供。

13×13 使用 256 个并行棋局线程，每轮积累 512 局后再洗牌、训练和导出，
减少反复启动进程和重洗回放的开销；推理批次上限仍为 128。
本机继续使用 2048 训练批次，每轮训练预算请求 32 个批次；预算同时限制在本轮新增数据行数的
4 倍以内，避免扩大训练轮次后因官方训练桶不足而零步退出。旧设置请求每轮 4 个批次，
但官方训练按整文件读取，近期实际约 8 个批次；新上限对应约四轮旧训练量，保留相近的数据复用比例。
19×19 使用 256 个并行棋局线程、每轮 2048 局和 400 visits，推理批次上限仍为 128。
2026-10-23 将每轮自我对弈从 512 局扩大到 2048 局、训练预算从 24 个批次扩大到
256 个批次，使每次完整冠军评测之前先完成更多有效训练。本机保留自动调优得到的
训练批大小 1280，对应每轮请求处理 327,680 条样本；按近期每 512 局约新增 4 万行
估算，训练处理量/新增数据约为 2。请求预算仍受本轮新增数据 4 倍上限约束；
这只是预算上限，不保证官方训练器实际处理到该数量，以周期指标和优化器计数为准。
每轮 200 局冠军评测、胜率至少 60% 且 Wilson 下界超过 50% 的晋级标准保持不变。
官方训练以整文件为单位，实际训练批次数和处理量以周期指标为准。
9×9 的自对弈资源配置继续沿用已验证参数。

19×19 自对弈启用 `init_games_with_policy`，按 `policy_init_area_prop=0.04`
初始化开局（19×19 平均约 14 手，实际手数随机）；`early_fork_game_probability=0.04`
和 `fork_game_probability=0.01` 分别控制早期分叉和非早期分叉时的普通分叉概率。
分叉只探索替代走法，显式关闭分叉/开局后的贴目补偿、贴目翻转和双活规则分叉。
棋盘仍为 19×19，POSITIONAL/AREA、无自杀、无税、贴目 6.5 保持固定。
这些选项只传给自对弈引擎，每轮 200 局固定开局配对评测和晋级标准不受影响。
9×9、13×13 和无真实模型的随机初始化继续关闭这些开局和分叉选项。

可在停止课程训练后重复实测；脚本取得课程和棋盘锁，所有对照数据保存在独立目录：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_rl_efficiency.py --pairs 4 --games 512
```

实测报告分别记录评测耗时、有效新数据行/小时、GPU 利用率、显存和无效棋谱。
`--mode training --selfplay-input <独立自对弈目录> --training-steps <批次数>` 可从
当前正式检查点的副本验证不同训练预算；验证会核对每个优化器步数的连续增长、
模型参数确实改变、模型/SWA/优化器张量有限，并检查正式检查点哈希保持不变。
本次在同一个正式检查点副本上生成 512 局、39,451 行新数据后实测：原 24 批预算
实际更新 24 批、数据利用比例 0.779；按扩大周期比例设置的 64 批预算实际更新
60 批、比例 1.947。请求 256 批时，数据量保护将实际更新限制到 126 批、比例
4.088；官方整文件训练可能略超名义 4 倍边界，不能将其表述为逐样本硬限制。
三组均通过 90 组优化器步数连续性、450 个有限张量与模型导出检查，峰值整卡显存
约 11.9 GiB。报告保存在 `training_runs/efficiency-20261023-training-volume/comparison.json`。
这些验证确认更新量与数值稳定性，棋力收益需要后续完整 200 局冠军评测确认。
累计训练样本包含对历史数据的重复训练，不能当成独立新增局面数；面板同时展示
新增数据产出速度（训练循环及含评测等待两种口径）和训练处理量/新增数据比例。

2026-10-15 在本机 RTX 5070 Ti 上冻结 `gogogo-s20004864-d6298617` 的短测结果：

| 对照 | 原设置 | 新设置 | 吞吐变化 |
|---|---:|---:|---:|
| 8 局固定开局评测、300 visits | 串行 182.9 秒 | 4 组并行 143.8 秒 | +27.2% |
| 512 局自对弈、300 visits | 128 线程 188.4 秒 | 256 线程 155.6 秒 | +21.1% |

评测的 4 个开局均严格换色，逐开局、逐颜色的终局结果一致；这仅是吞吐对照，正式评测仍为每个对手 200 局。
两组自对弈共 1024 局，均无损坏、无结果或开局双停棋谱。选中的 256 线程组产出 19,045 行数据，
平均推理批次 69.6（128 线程组为 36.1），峰值设备显存约 1.75 GiB；4 组并行评测约 3.90 GiB。
复制原检查点、优化器及保留回放后，隔离续训实际完成 65,536 个训练样本处理量和模型导出，
每条新增数据对应约 3.44 次训练处理。自对弈、洗牌、训练、导出各阶段实测相加约 203.5 秒，
对比近期 50 轮均值折算的同等 512 局约 399.5 秒，吞吐约 1.96 倍；该比较不含评测等待。
原始报告、日志与备份保存在本机 `training_runs/efficiency-20261015/`，不提交 Git。

2026-10-19 在同一台 RTX 5070 Ti 上冻结 19×19 的 `gogogo-s1464320-d1886644`，
所有对照均使用 400 visits、相同规则与推理批次上限 128：

| 每轮局数 | 并行棋局线程 | 自对弈耗时 | 新数据行/小时 | 平均推理批次 |
|---|---:|---:|---:|---:|
| 128 | 128 | 211.6 秒 | 157,153 | 39.4 |
| 512 | 128 | 773.1 秒 | 169,594 | 42.5 |
| 512 | 256 | 679.5 秒 | 192,687 | 83.0 |

两组 512 局的新数据量分别为 36,418 和 36,371 行；256 线程的新数据产出速度比
128 线程提高约 13.6%。三组共 1152 局，均无损坏、无结果或开局双停棋谱。
复制原检查点、优化器及回放进行隔离续训后，旧配置实际完成 6×1280 个训练样本，
新配置实际完成 24×1280＝30,720 个训练样本；每条新增数据的训练处理量分别为 0.831 和 0.845。
新配置的 90 组优化器状态步数均由 1144 增至 1168，模型导出成功，原检查点 SHA-256 保持不变。
洗牌、训练、导出分别耗时 26.2、25.5、4.2 秒，训练设备显存峰值约 12.61 GiB。
与自对弈合计约 735.4 秒；旧配置最近 20 轮平均每 128 局为 249.2 秒，
折算同等 512 局约 996.8 秒，新方案吞吐约为其 1.36 倍。
这里的新周期耗时是隔离实测各阶段相加，两种口径均不含评测等待。
这些是吞吐测试；棋力继续由正式固定开局评测判断。
原始报告、日志与备份保存在本机 `training_runs/efficiency-19x19-20261019/`，不提交 Git。

大分差结果（终局绝对分差至少棋盘面积的 25%）继续记录并显示在面板上，
但不再作为课程晋级条件。质量窗口仍检查局数、开局双方立即停一手、
黑白得分平衡及无结果或损坏棋谱。移除大分差门槛的依据见
[9×9 大分差对局分析](9X9_EXTREME_RESULTS.md)。

三个棋盘阶段均保留两类评测：滚动冠军用于观察近期进步，固定历史版本用于
比较长期提升。候选模型战胜冠军达到 60% 且 Wilson 下界超过 50% 时更新冠军；
这项棋力判断独立于课程切换。9×9、13×13 的课程切换仍须满足最低样本数、
对每个固定历史基准分别达到胜率至少 60%、Wilson 下界超过 50%，满足对局健康门槛，
并连续通过两次。不得用不同对手的胜率直接判断棋力升降；冠军更替本身不会抬高固定晋级门槛。
同一模型兼任两类对手时只对战一次；候选不会与自身评测。

本机 9×9 暂定冠军为 `gogogo-s31967232-d4973021`，固定历史基准为
`gogogo-s9562112-d1610962`、`gogogo-s31655936-d4929572` 和
`gogogo-s31819776-d4952345`。初始名单在课程配置各阶段的
`initial_champion_model`、`fixed_baseline_models` 中设置，首次初始化后固化到
课程状态，避免随清理或重启漂移。配置中新增的固定基准会验证后追加到已有名单，
并清零连续达标计数；旧版只检查首个基准的达标计数也不会沿用。缺失模型会报错，
不会悄悄缩减对手池。冠军和历史基准的二进制
模型及检查点受保留保护。后续棋盘以迁移种子初始化两类对手。
本机正在进行的 13×13 阶段固定池为迁移种子 `gogogo-s0-d0` 和冻结的
1151 万样本历史冠军 `gogogo-s11507712-d3661143`，后者不会随新冠军出现而替换。
这是已有运行的模型名单；用于全新训练目录时需移除这些尚未生成的本机历史模型名。

人机对弈的“电脑难度 / 引擎”列表会按所选棋盘加载“训练·滚动冠军”和
“训练·历史基准”。这些选项使用对应的训练模型和 200 visits 搜索预算，
不叠加 HumanSL 风格，也不覆盖原先保存的 KataGo 主网络设置。

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

全局面板分别展示冠军挑战、各固定基准的胜率和 Wilson 区间，并保留真实对手
身份。相对 Elo 不能跨对手直接比较；全胜或全败时显示“饱和，无法估计”。
页面还显示最近训练记录时间，网页刷新不代表训练进程正在运行。
更新面板和初始化对手池、或单独执行一轮正式评测（均不启动训练）：

```powershell
.\.venv\Scripts\python.exe train_rl.py curriculum-dashboard
.\.venv\Scripts\python.exe train_rl.py curriculum-evaluate
```

本地直接打开 `training_runs/curriculum/dashboard.html` 即可查看。也可以在
项目目录启动仅本机可访问的网页服务，然后访问
`http://127.0.0.1:8765/curriculum/dashboard.html`：

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory training_runs
```

训练有互斥锁，不能误启两个进程写同一检查点。中断后再次执行 `continuous`
会从 `train/gogogo/checkpoint.ckpt` 续训；自我对弈、洗牌数据、日志、待导出
检查点和已接纳模型分别保存在训练目录的对应子目录。首次没有模型时由 KataGo
随机启动器生成种子数据。这个单显卡初始训练采用无 gatekeeper 模式，新导出
模型会直接接纳，以便更快完成早期迭代。

Windows 后台训练使用独立的计划任务守护，每分钟检查训练和监控进程。异常退出后
自动从已有检查点续训，监控服务可单独恢复；任务隐藏启动、忽略重复实例，没有
运行时限。守护不依赖 Codex，登录 Windows 后也会自动检查并恢复。安装或更新：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\register_training_recovery.ps1 -StartNow
```

注册脚本更新任务定义时保留正在运行的训练，不会为了安装守护而重启训练。
训练与监控任务分别为 `GoGoGo-KataGo-Training-Session`、
`GoGoGo-KataGo-Monitor-Session`，守护为 `GoGoGo-KataGo-Watchdog`。
旧 `GoGoGo-KataGo-Curriculum` 任务保持禁用，避免重复启动。

面板的“训练自动恢复”区域显示最近检查时间、恢复次数和异常/恢复记录。完整记录
保存在 `training_runs/curriculum/recovery_events.jsonl`，当前状态保存在同目录的
`recovery_status.json`。连续启动失败会逐步延长重试间隔，最长 15 分钟；进程仍在
运行时不会因低 GPU 利用率或日志暂时不更新而重启训练。

交互运行时按 `Ctrl+C` 会在安全边界保留可恢复产物。后台训练请用以下命令主动
暂停，守护不会将主动暂停当作故障：

```powershell
.\.venv\Scripts\python.exe -m weiqi.training_watchdog pause

# 恢复后台训练
.\.venv\Scripts\python.exe -m weiqi.training_watchdog resume
```

守护也尊重手动禁用/停止任务，以及课程或当前棋盘目录内的 `STOP`、`PAUSE` 和
接力导出锁；恢复前需解除相应暂停或接力锁。`pause` 产生的
`AUTO_RECOVERY_PAUSED` 标记由 `resume` 清除。监控服务在训练主动暂停时继续运行。

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
