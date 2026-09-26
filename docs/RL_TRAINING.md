# 本机 9×9 围棋强化学习

训练器使用现有 `GoGame` 的落子、提子、禁自杀、位置全局同形和面积计分规则。
流程是：最新候选与自己、轮换的历史冠军或冻结的近期候选对弈 → 回放池 → 策略/价值网络更新 → 固定局面与对手评测
→ 按条件进行晋级对战 → 保存检查点。
GUI 的“推理训练”仍是定式和死活课程；强化学习通过下面的命令启动。

## 环境与启动

当前机器的 Windows 默认 Python 没有 PyTorch。`训练围棋.ps1` 使用已安装的
WSL2 `Ubuntu-24.04`，以及 `/home/dev/.venvs/nanogpt/bin/python`。
脚本只转发命令，不会安装软件或修改这个虚拟环境。

在本项目目录的 PowerShell 中运行：

```powershell
# 查看命令。
.\训练围棋.ps1 --help

# 短验收：真实自我对弈和评测，搜索预算较小。
.\训练围棋.ps1 train --config config/rl_training.smoke.json --iterations 1

# 默认训练：9×9、64 通道、4 残差块、每手 64 次 MCTS、2 个自我对弈进程。
.\训练围棋.ps1 train --iterations 2

# 检查已保存模型。
.\训练围棋.ps1 inspect training_runs/balanced/latest.pt

# 再跑两轮，而不是把总轮数设为两轮。
.\训练围棋.ps1 train --resume training_runs/balanced/latest.pt --iterations 2

# 旧版完整检查点按新训练对手与晋级策略续训，仍保留原模型、回放和优化器。
.\训练围棋.ps1 train --resume training_runs/balanced_20260920/latest.pt --policy-overrides config/rl_training_policy.balanced.json --iterations 1
```

在已配置 PyTorch 的 Linux/Windows 环境中，可以直接使用
`python train.py train --iterations 2`。其他机器可在独立环境中安装
`requirements-training.txt`，并按其显卡和驱动选择合适的 PyTorch 版本。

默认新训练输出到 `training_runs/balanced/`。可用 `--output training_runs/my_run`
指定新目录。已有运行必须使用 `--resume`，以免误覆盖。训练过程按指定轮数结束，
不会自动建立长期后台任务。

## 默认规模和预热

| 参数 | balanced |
| --- | ---: |
| 棋盘 / 白方贴目 | 9×9 / 6.5 |
| 网络 | 64 通道，4 个残差块，326458 参数 |
| 自我对弈搜索 | 每手 64 次模拟 |
| CPU 对弈进程 / 最大推理批量 | 2 / 8 |
| 每轮自我对弈 | 16 局 |
| 对历史已接受冠军的对局 | 约 25%，成对换色 |
| 对近期冻结候选的对局 | 约 25%，成对换色；没有不同权重的存档时转为自我对弈 |
| 当前候选自我对弈 | 约 50%，加上未找到冻结对手的剩余对局 |
| 回放池容量 / 最少训练样本 | 50000 / 2048 |
| 训练 batch / 每轮更新 | 128 / 100 |
| 固定局面 | 黑白均衡的 96 个 9×9 局面，候选搜索 8 次/局面 |
| 每轮初筛 | 4 局，每手 8 次模拟；至少 25% 得分率才进入到期的晋级赛 |
| 历史模型对手池 | 每 3 轮评测一次，各对手 4 局、每手 8 次模拟 |
| 完整晋级赛 | 每 3 轮至多一次，40 局、每手 96 次模拟；对局长度上限为棋盘面积的 4 倍 |
| 晋级条件 | 得分率至少 55%、无截断、换色开局对的单侧精确符号检验 p ≤ 0.05 |
| 自动精度 / 显存分配上限 | FP32 / PyTorch 分配器的 60% |

样本不足 2048 条时，该轮只采集和保存，日志会标记 `replay_warmup`；继续运行或
续训即可积累样本。每局长短不同，不能保证第一轮就达到阈值。
初始模型是随机初始化网络，`best_iteration=0` 表示尚无候选通过评测。
最新候选即使暂未通过晋级，下一轮也会用当前参数继续收集样本。训练对局在轮开始时
按三类安排：约四分之一对已接受冠军、四分之一对不同权重的近期冻结候选、其余双方都
由当前候选控制。冻结对手使用同一开局交换黑白；没有可用的冠军或候选存档时，空缺转
为自我对弈。已接受冠军按版本轮换，近期候选从最多四个不同权重的存档轮换；
`best.pt` 仍只保存通过正式晋级赛的基准版本，`milestone_*.pt` 作为陪练和固定参照，
不冒充已接受冠军。

旧运行从 `latest.pt` 续训时，如果采用新策略且 `anchors/` 尚无可用的里程碑版本，
训练器会从保留的上一份完整历史检查点导出一份冻结候选供训练使用。导出不会覆盖历史
检查点，也不会修改当前模型、回放池或优化器。如果历史检查点不足，该份额仍转为自我
对弈。未明确传入 `--policy-overrides` 的旧检查点会继续采用它原有的对手比例与晋级规则。

每盘 JSON 棋谱的 `training_opponent` 和每轮 `selfplay_opponents` 汇总记录了对手名称、
模型摘要、对局数及有效样本数，便于核对实际轮换。对冠军执黑、执白各下一盘，以免
总让某一模型享受先手。双方的搜索访问分布和终局结果都可成为训练样本；冠军模型
来自先前已接受的本地快照，不使用 KataGo 作为对局训练对手。

`inference_batch_size` 是上限；两个同步对弈进程通常只会形成 1～2 条查询的批量。
增加并发进程前应测量 CPU、显存和整轮速度。主进程持有 CUDA 模型，子进程仅执行
规则和搜索。`data_loader_workers` 控制训练时的 CPU 数据加载进程。

`auto` 设备选择会明确记录 CUDA 或 CPU。显式指定无法使用的设备会直接报错，
不会悄悄切换。混合精度仅在支持它的 CUDA 设备上启用；`compile_model` 为可选
PyTorch 编译，本机默认关闭。60% 限制不包含显示驱动、CUDA 上下文或其他进程。

## 输出和续训

| 路径 | 内容 |
| --- | --- |
| `config.resolved.json` | 实际使用的完整参数 |
| `events.jsonl` | 设备、搜索进度、每局长度、损失、暂停和评测记录 |
| `latest.pt` | 最近完成轮次的模型、最佳模型、优化器、回放池和随机数状态 |
| `candidate.pt` | 最新候选网络，用于独立评测 |
| `best.pt` | 最近通过晋级评测的模型；初期可能仍为随机初始模型 |
| `anchors/accepted_*.pt`、`anchors/milestone_*.pt` | 冻结的最佳模型和每 5 轮的候选快照；新策略可从上一份保留的检查点补一份，不覆盖旧版本 |
| `checkpoints/iteration_*.pt` | 按配置保留的完整历史检查点 |
| `iterations/000001/selfplay/` | 自我对弈 JSON 棋谱和 SGF |
| `iterations/000001/position_quality.json` | 每个固定局面的模型着法和 KataGo 指标 |
| `iterations/000003/pool/` | 固定历史对手池和均匀策略 MCTS 的对战棋谱与汇总 |
| `iterations/000003/screening/`、`evaluation/` | 快速初筛和按条件执行的晋级赛棋谱 |
| `summary.json` | 最近完成轮次的统计和模型 SHA-256 |

检查点先写临时文件再替换，训练按完成轮次保存。`Ctrl+C` 或 STOP 中断后，恢复
最近一次完整检查点；未完成轮次会重新执行。GPU 上的浮点运算仍可能存在微小的
平台差异，恢复状态不等于承诺跨硬件逐位一致。

续训默认采用检查点的所有训练参数。若同时指定 `--config`，除输出路径外必须
与检查点一致；只想改变评测预算时，可额外传入 `--evaluation-overrides`，它不会
修改棋盘、网络、优化器或自我对弈参数。模型导出 `candidate.pt` / `best.pt` 不包含回放池和优化器，不能
用作 `--resume`。在别的目录恢复历史检查点时，显式指定 `--output`。
跨目录续训必须使用一个全新的空输出目录；已有运行只能从它自己的、内容未变化的
`latest.pt` 恢复，训练器会拒绝覆盖另一份运行结果。

## 暂停、停止和正常下棋

启动更新后的围棋程序后，未结束的对局会通过文件心跳通知训练器暂停，包括
推演期间。对局结束或关闭窗口后自动恢复；异常退出留下的心跳约 8 秒后失效。
此前已打开的旧版本窗口需要重启，才能发送这个心跳。

也可通过文件手动控制。以下示例使用默认输出目录：

```powershell
# 暂停；删除 PAUSE 后恢复。
New-Item -ItemType File -Path training_runs/balanced/PAUSE -Force
Remove-Item -LiteralPath training_runs/balanced/PAUSE

# 停止；再次续训前先删除 STOP。
New-Item -ItemType File -Path training_runs/balanced/STOP -Force
Remove-Item -LiteralPath training_runs/balanced/STOP
```

训练器会清理自己启动的对弈子进程。运行目录有进程锁，拒绝同一环境中的第二个
训练进程写入相同目录。请勿同时从 Windows 和 WSL 对同一目录启动两个训练器。

可以在另一个 PowerShell 中查看实时进度：

```powershell
Get-Content training_runs/balanced/events.jsonl -Tail 5 -Wait
```

各轮结束后，可在 Windows 上生成持续更新的可读趋势报告，不需要启动 PyTorch：

```powershell
py report_evaluation.py --run training_runs/balanced
```

报告写入运行目录的 `evaluation_trend.md`，并标出教师局面文件、模型或候选搜索预算
发生变化的轮次。只有同一评测协议下的局面分数才能直接对比。对手池数据和晋级赛
分别显示，不能把 2 局或 4 局的结果当作稳定棋力。

## 实时监控网页

在 Windows 项目目录运行 `py monitor_training.py`，或双击 `启动训练监控.bat`，
然后打开 <http://127.0.0.1:8766>。网页服务只有 Python 标准库依赖；WSL 中的
训练器继续按原命令运行，Windows 服务读取同一份 `training_runs/` 数据。

```powershell
# 指定页面默认选中的运行（参数是 training_runs 下的目录名）
py monitor_training.py --run balanced_20260920

# 读取另一份训练记录父目录，或更换端口
py monitor_training.py --root E:\other_training_runs --port 8767
```

页面每 2 秒自动刷新，支持切换训练记录，显示：

- 训练心跳、暂停或结束状态、当前轮次和阶段进度。
- 当前对局批次的对手名称、冻结版本摘要，以及逐局累积的胜、负、和、截断。
- 每轮初筛、晋级赛和固定对手池的历史成绩，最近完成轮次的训练对手分布。
- 累计参数更新、回放样本、损失及最近日志。

**胜率 = 胜局数 / 已产生结果的局数**；**得分率 =（胜 + 和 / 2）/ 局数**。
两者都把截断局保留在分母中，截断单列且不算胜。纯候选自我对弈没有独立对手，
其候选胜率显示为空。同一轮对多个冻结对手训练时，页面在对手表中逐个计胜率，
不会把不同对手的胜负合成一个单一胜率。训练冻结对手对局、固定对手池、初筛和正式晋级赛分别呈现；
搜索预算、对手或候选版本不同的成绩不能当成同一固定基准的提升。
旧运行的配对置信区间仍按原协议显示。采用新策略的晋级赛显示换色开局对的单侧检验
`p` 值和截断数；两种评估规则和不同样本量的结果不能混为同一条棋力提升证据。

更新后的 `train`、`evaluate` 和 `benchmark` 会产生 `monitor.json` 心跳，
`events.jsonl` 会新增开局、对手版本和逐局胜负。状态结束后，网页仍保留历史结果。
超过 15 秒没有心跳时显示“心跳过期”，而不是继续宣称正在训练；较长的阻塞操作
或计算也可能暂时出现此状态。暂停等待期间仍更新心跳。

旧版记录仍可读取已保存的汇总与棋谱，但无法事后补出实时状态或缺失的对手字段。
页面明确标为历史记录，缺失的即时胜率留空。正在运行的旧训练进程需在正常停止并
从 `latest.pt` 续训后，才能使用新增的逐局记录。

服务默认只监听本机 `127.0.0.1`，只提供读取接口。关闭服务窗口或按 `Ctrl+C`
只会停止监控，不影响训练；监控页面不提供训练启停按钮。

## 固定局面、KataGo 标注与历史对手

项目已经放入 `config/rl_eval_positions_9x9.json`：96 个按固定种子生成的合法局面，
涵盖开局、中盘和后段，黑白行棋各半。每条记录都保留完整走子历史，按本地禁手与
全局同形规则复盘。这些局面来自独立的合法随机棋局，**不是人类棋谱或业余段位题库**。
保持同一文件和 SHA-256 摘要，才能纵向比较模型版本。

`config/rl_eval_teacher_9x9.json` 是本机 Windows KataGo OpenCL 在固定模型、配置、
规则和每局面 64 次访问下生成的标注。标注文件记录了引擎和模型 SHA-256、原始
策略概率以及已经搜索过的候选着法目差。训练器只读取这个缓存，不会每一轮重新
启动 KataGo。重新生成标注可运行 `py benchmark_teacher.py label`；如果更换模型、
访问数或局面集，请指定**新输出文件**并在训练配置中更新路径。

每轮对相同的 96 个局面，候选模型搜索 8 次并选择一手，输出两个主要诊断量：

- `mean_teacher_log_loss`：KataGo 原始策略给该着法的负对数概率；越低通常越接近
  老师偏好的着法。所有合法候选着法都有这个指标，即使模型对 KataGo 完全赢不了。
- `teacher_top_move_rate`：和 KataGo 搜索首选着法一致的比例；越高通常越好。

同一报告还给出开局、中盘、后段的分项数值。`mean_score_delta_when_covered` 只统计
KataGo **在原局面确实搜索过**的候选落子，必须连同 `score_delta_coverage` 阅读；
未搜索的落子绝不拿原局面的 `rootInfo` 冒充这一步的评价。这些是诊断值，整体棋力
仍以固定对手实战为主。

需要完整的逐步目差时，Windows 可按需对候选的实际落子后局面另开 KataGo 查询：

```powershell
py benchmark_teacher.py score --quality training_runs/balanced/iterations/000003/position_quality.json --output training_runs/balanced/iterations/000003/katago_score_complete.json
```

生成的 `training_runs/eval_katago_score_cache.json` 按局面和落子复用先前查询结果。
完整报告显示平均的**带符号目差**；负值表示该着法可能取得更高目数，但 KataGo
的首选着法还会考虑胜率。落子后的搜索和原局面的候选搜索估计可能略有波动，
这项数据仍然是辅助指标。若候选着法恰好结束棋局，则使用本地面积计分，并在结果中
注明来源。更换老师模型、配置或访问数时缓存会拒绝混用。

每 3 轮，训练器从 `anchors/` 中选出不同年代的冻结版本，连同均匀策略 MCTS 组成
对手池。每个固定对手使用可重复的开局种子并交换黑白，后续候选可以对比同一参照。
相同参数的模型快照不会重复纳入对手池。冻结对手文件一旦生成，内容变化会报错。

不启动训练，也可以独立测某一检查点。对于之前创建、还没有 `anchors/` 的运行，
显式提供 `--opponent`：

```powershell
.\训练围棋.ps1 benchmark training_runs/balanced_20260920/candidate.pt --opponent training_runs/balanced_20260920/best.pt --games 4 --visits 8 --position-visits 8 --output training_runs/benchmark_candidate
```

`benchmark` 同时保存局面诊断、各对手的完整棋谱和换色对战统计。默认读取候选检查点
同级的 `anchors/`；也可多次传入 `--opponent` 指定额外的模型文件。独立运行时可用
`--positions`、`--teacher`、`--pool` 指向其它已冻结的评测资产。

## 分级晋级评测与不确定性

训练每轮先进行 4 局低搜索预算的初筛。只有到第 3、6、9… 轮且初筛达到门槛时，
才运行配置的 40 局正式晋级赛。正式赛将最大棋步数提高到棋盘面积的 4 倍，减少因
长度限制而没有终局的比赛；达到上限仍记为截断，不给它臆造胜负。候选须达到 55%
得分率、没有截断对局，且**对每组交换黑白的开局计算单侧精确符号检验，p ≤ 0.05**，
才能更新 `best.pt`。同一开局一黑一白算一个单位：这组候选得分超过一半记优势，
低于一半记劣势，正好一半记持平。没有决定性配对不晋级。

这是预先固定局数的一次晋级判定，不能在看到结果后不断追加对局来凑显著性；多轮
反复挑战也不等于获得全程 95% 的棋力保证。原来的保守区间仍随结果记录供诊断，
但新策略的晋级判断依据符号检验，而不是那个区间的下界。未晋级的候选保留参数并
继续训练。

`screening`、`evaluation` 和对手池分别报告胜、负、和、截断、局数、配对局数与区间。
新晋级赛还记录换色开局优势、劣势、持平组数，以及单侧 `p` 值。
局面集的策略指标不计入晋级门槛；KataGo 胜率也不作为早期晋级的唯一条件。
如需修改评测预算，新训练可在 JSON 配置 `overrides.evaluation` 中设置。
已有检查点续训可使用下面的评测专用覆盖文件；模型结构仍沿用原检查点：

```powershell
.\训练围棋.ps1 train --resume training_runs/balanced/latest.pt --evaluation-overrides config/rl_evaluation.more_games.example.json --iterations 1

# 同时调整训练对手与晋级策略；只接受策略相关字段。
.\训练围棋.ps1 train --resume training_runs/balanced_20260920/latest.pt --policy-overrides config/rl_training_policy.balanced.json --iterations 1
```

覆盖文件只接受评测组已有字段，例如 `games`、`full_every_iterations`、
`pool_games`、`confidence_level` 和固定局面文件路径。报告与下一检查点会记录
实际使用的配置；更换局面集后，应按 SHA-256 分开比较趋势。

## 独立对战

```powershell
# 对比最新候选和已接受模型，交换黑白。
.\训练围棋.ps1 evaluate training_runs/balanced/candidate.pt --opponent training_runs/balanced/best.pt --games 20 --output training_runs/eval_candidate

# 不指定 opponent：使用均匀策略、零价值的 MCTS 基线；它不是人类段位。
.\训练围棋.ps1 evaluate training_runs/balanced/candidate.pt --games 20 --output training_runs/eval_uniform
```

每对评测采用相同的两手随机合法开局并交换黑白；不加根节点探索噪声。
胜、负、和、截断分别报告，和棋计半分，截断不给分且阻止模型晋级。
默认 20 局的点估计仍然很粗；报告里的配对区间会显示样本不足。

## 规则和训练边界

- 价值标签以当前行棋方为视角，MCTS 每跨一手翻转价值符号；策略标签来自访问次数。
- 输入包含最近 8 个棋盘的双方棋子、行棋方、贴目、前一手虚手和合法落点。
  全局同形的完整历史保留在规则引擎中，用于禁止重复局面。
- 正常连续两次虚手或显式认输产生终局标签。初期默认不允许自动认输。
- 达到 `max_game_length_factor × 棋盘面积` 的对局记为截断，整局样本不进入回放池。
  日志保留当前盘面的分数用于诊断，这不是终局胜负。
- 面积计分沿用本项目规则，不自动判定复杂死活；双方需通过实际落子提净死子。
  训练行为受此规则定义约束。
- `.pt` 模型供本项目训练/评测命令使用。当前训练器没有提供 GUI 中的模型对手选择。
- `high_performance` 的 19×19 示例是可配置方向，本轮验收目标为 9×9。
  13×13/19×19 默认不加载这套 9×9 固定局面与老师标注；显式指定不匹配的数据会在
  读取配置时拒绝。

## 验证命令

```powershell
# Windows 的原有功能及无需训练依赖的测试。
py -m unittest discover -s tests -v

# WSL 中检查搜索、特征、终局标签、完整续训和多进程退出。
wsl -d Ubuntu-24.04 --cd /mnt/e/codex/weiqi_gui -- /home/dev/.venvs/nanogpt/bin/python -m unittest tests.test_rl_search tests.test_rl_runtime tests.test_rl_evaluation -v
```

Windows 没有 NumPy/PyTorch 时，依赖它们的测试会明确标记为跳过；实际训练验证
必须在含有训练依赖的环境中执行，不能把跳过测试当作 GPU 验证。
