# KataGo 训练与课程续训

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
阶段迁移继承上一阶段的 SWA 权重，但会重置优化器、阶段计数和数据记录。
每新增 50 万样本，对滚动冠军和每个固定历史基准分别执行 200 局评测；未达标或评测失败只会留在当前棋盘续训，
不会强制切换或破坏现有检查点。评测使用持久化的 100 个确定性开局，每个开局
交换候选与基准的黑白方各下一局，避免把同一空棋盘轨迹重复 200 次当成独立样本。
自我对弈关闭弱模型的原始策略开局初始化，探索仍由 MCTS 温度和 25% 根噪声提供。

9×9 的极端结果定义为终局分差至少 20.25 目。该阶段的最近 1280 局质量窗口
要求极端结果比例不超过 11%；13×13 和 19×19 沿用 5%。9×9 门槛由相同规则、
贴目、200 visits、根噪声和温度下的 512 局强网络自对弈校准：41 局极端结果，
比例 8.0%，95% Wilson 上界 10.7%。这是健康门槛的校准，不表示当前训练网络
已经降低了大分差比例；仍须以其最近 1280 局实际结果通过质量窗口。
这组强网络对照的黑方胜率为 68.4%，因此 11% 是当前 9×9 对局协议下的暂定
门槛，后续仍应结合黑白平衡及更大样本复核。
试验和校准依据见 [9×9 大分差对局校准](9X9_EXTREME_RESULTS.md)。

三个棋盘阶段均保留两类评测：滚动冠军用于观察近期进步，固定历史版本用于
比较长期提升。候选模型战胜冠军达到 60% 且 Wilson 下界超过 50% 时更新冠军；
这项棋力判断独立于课程切换。9×9、13×13 的课程切换仍须满足最低样本数、
对首个固定历史基准的胜率与置信区间门槛、对局健康门槛，并连续通过两次。
同一模型兼任两类对手时只对战一次；候选不会与自身评测。

本机 9×9 暂定冠军为 `gogogo-s31967232-d4973021`，固定历史基准为
`gogogo-s9562112-d1610962`、`gogogo-s31655936-d4929572` 和
`gogogo-s31819776-d4952345`。初始名单在课程配置各阶段的
`initial_champion_model`、`fixed_baseline_models` 中设置，首次初始化后固化到
课程状态，避免随清理或重启漂移；缺失的模型会报错。冠军和历史基准的二进制
模型及检查点受保留保护。后续棋盘以迁移种子初始化两类对手。

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
