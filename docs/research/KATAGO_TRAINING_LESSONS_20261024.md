# KataGo 训练方法对本地 M4 混合棋盘训练的启示

核对日期：2026-10-24。证据以本地 `katago/source` 的干净检出为准：tag `v1.18.2`、commit `fd0723fdbc0e9d82cf269c9630af8c27c57c07c4`；另核对 David J. Wu 的论文和官方在线说明。本文只研究，没有修改配置或干预训练。下文“建议”是针对当前条件的工程判断，不是官方给 M4 的推荐参数。

## 结论与优先级

当前值得优先补的是**按棋盘大小的效果评测、数据分布与算力账本**，而不是重新实现 KataGo 已有的算法。论文的整体加速结果来自不同硬件、数据量和整套方法，不能折算为本地提速倍数或达到某段位的时间承诺。[论文](https://arxiv.org/html/1902.10565v5)

| 优先级 | 建议的后续工作 | 机制、成本与判据 |
|---|---|---|
| P0 | 建立独立、固定的棋力评测 | 按 7/8/9/13/19 分开；冻结若干历史模型和至少一个可打得动的固定对手；相同规则、贴目、搜索预算与成对开局，交换候选模型执黑执白。保存双方版本、SGF、种子、胜率及置信区间。先低频评测并预先固定局数；区间过宽时另安排更大预算复测，避免边看胜率边决定何时停止，也避免评测吃掉大量自博弈预算。这是本文的实验设计建议。 |
| P0 | 把健康统计按尺寸拆开 | 分尺寸记录局数、局长、过早双 pass、极端结果、非法着、训练行数、搜索与先验的差异，以及各辅助 loss。混合总均值可能掩盖单一尺寸的问题；健康指标和 loss 不能替代棋力评测。 |
| P1 | 核算新鲜数据与重复使用 | 记录每轮新增训练行、累计实际训练样本、回放窗年龄和样本来源模型。surprise 会复制/重采样行，行数不等于独立局面数；累计训练量/累计数据行也不等于每个独立局面被训练的次数。先保证统计口径，再调回放与 train bucket。 |
| P1 | 受控比较搜索分配 | 固定墙钟预算，比较当前 full/cheap 分配与单变量调整；看“有用训练行/小时”和固定评测。低访问数在大棋盘是否足以产生比先验更好的策略目标，需要实测。不要直接照搬官方几百 visits。 |
| P2 | 先评估有效 SGF 起点、小比例 fork | 从合法、确有落子的开局局面小比例续弈，测覆盖、异常结束、数据成本与棋力；SGF 也须隔离评测集。弱 policy 可能过早 pass，不能盲目打开 policy init。教师数据、Muon 等更大改动另立实验。 |

官方 gatekeeper 是可选组件，说明明确允许直接接纳新模型；不开可节省计算，初期打开有助排错。**固定对手评测与接纳门控是两件事**：本机应先获得可信的独立评测，再决定是否值得每次导出都门控。门控只比较最新冠军也不足以描述全部尺寸和历史能力。[训练说明](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/SelfplayTraining.md)

## 已经继承的机制，不应当成待开发功能

| 方法 | 本地状态及源码证据 | 如何理解收益与限制 |
|---|---|---|
| Playout cap randomization | 当前 full 32 / cheap 8，full 概率 0.5；`cheapSearchTargetWeight=0`。官方 `play.cpp` 在 cheap 回合减少搜索、关闭探索噪声并默认降低训练权重。 | 让便宜着法推进更多完整对局，同时把较贵搜索用于较好的策略目标；这不是把每个 cheap 回合都用作独立的高质量策略标签。当前绝对预算远小于论文，不能仅凭“同一种方法”推断同等效果。 |
| 策略目标修剪与目标权重 | `extractPolicyTarget()` 调用 `getPlaySelectionValues()`，后者回看并削减过度探索分支的权重；训练端另读取 player/opponent policy 权重。 | 目的是避免把纯探索开销全部教给策略网络。“目标修剪”“廉价搜索权重”“棋盘外掩码”是三个不同机制。不要另写一个粗暴的 visits 直方图覆盖官方目标。 |
| Policy/value surprise 重加权 | 有效配置已是 policy 0.5、value 0.1。`play.cpp:2090–2165` 重分配写出期望权重。 | 优先学习搜索发现的意外着法/结果。关键例外：cheap 默认权重虽为 0，特别高的 policy surprise 仍可能令该行重新获得正权重；启用 reanalysis 后有额外条件。官方方法说明把 value surprise 的收益称为更实验性、尚未充分证实，不宜继续盲目调大。 |
| 辅助目标与 TD | 官方训练读取双方策略、胜负、分数分布、ownership、scoring、future position、seki、三组 TD value/score 与短期误差；无效目标有各自权重。`metrics_pytorch.py:563–584`、`739` 之后。 | 从一次对局取得比最终胜负更密集的监督，TD 还提供短期反馈。它们已接入，首先检查数据是否实际有有效标签及 loss，而不是再做一套辅助头。不要把总 loss 大小直接当棋力。 |
| Global pooling + mixed-size mask | `b10c128` 有 global-pooling blocks；`KataGPool` 用真实有效面积归一化，模型和 spatial loss 使用 mask。 | 共享局部战术并引入全局上下文；支持一个 batch 中混合尺寸。当前统一 `dataBoardLen=19` 会把小盘填充到 19，小盘对局短不意味着卷积训练每行按面积同比省算。棋盘外不可当普通空点。 |
| SWA / Lookahead / AdamW | 当前官方 run 已启用。`train.py:1706–1728` 实现 Lookahead 同步与 SWA；导出可选 SWA。 | 权重平均降低候选模型波动；SWA 导出不是运行时多网集成。已有机制无需重复叠加；收益仍应用固定评测确认。 |

以上分别可追溯到 [自博弈与重加权源码](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/program/play.cpp)、[搜索目标源码](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/search/searchresults.cpp)、[损失源码](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/katago/train/metrics_pytorch.py)、[网络源码](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/katago/train/model_pytorch.py)、[模型配置](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/katago/train/modelconfigs.py)、[训练器](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/train.py)。

官方方法说明补充了 masking、surprise 与短期目标的机制：TD 时间跨度按棋盘面积缩放；关于其辅助学习收益，官方也明确提醒近期未做完整受控重测。它们有实践依据，但不应包装成当前小数据 run 已证实的增益。[方法说明](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/docs/KataGoMethods.md)

## 尚未充分利用、但应有条件引入的做法

**开局覆盖。** 当前 `initGamesWithPolicy=false`、普通/早期 fork 为 0；但 `sidePositionProb=0.020` 已经继承，不能说当前完全没有替代着法训练。官方模板的 policy init 直接用策略先走一段，长度按面积缩放；fork 从已有局面试替代着法再续弈；SGF starts 允许载入棋谱局面、重新设置规则并续弈。三者都改变数据分布，fork 与贴目补偿还会增加计算。对当前早期弱策略，policy init 不保证改善双 pass；较可控的后续实验是先选合法且已有落子的 SGF 开局起点，小比例续弈，再分别尝试 fork 和 policy init。小盘贴目需按实际规则与独立评测校准，不能凭经验猜一个“公平的 7/8 路贴目”。SGF starts 不等于把整盘人类棋谱当监督标签。[官方配置](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/configs/training/selfplay1.cfg)、[初始化源码](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/program/play.cpp)

**回放新鲜度。** 官方 shuffler 选最新数据窗口，可用幂律增长，也支持固定窗；`keep-target-rows` 只是从该窗口保留的采样量。trainer 的 train bucket 按新增行允许一定训练量，`-no-repeat-files` 避免反复消费同一批已 shuffle 文件。它不能阻止下一次 shuffle 再抽到旧局面。当前 50,000 行上限本身既不能证明太旧，也不能证明足够多样；应结合尺寸、来源模型、样本年龄、训练/验证差距再调。官方常见每新增行允许 4 个训练样本是参考起点，不是本机最优值。[shuffler 源码](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/shuffle.py)、[训练说明](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/SelfplayTraining.md)

**贴目、规则与让子。** 官方模板提供多种劫规则、计分、禁自杀/允许自杀、随机贴目、让子及贴目补偿。其主要用途是覆盖不同局面和规则，并非保证每一种随机化都提高固定规则棋力。当前应先由有效生成配置确认哪些已启用；对小棋盘特别要按尺寸看贴目和结果分布。若产品只需一种规则，可比较适量多样性与目标规则专项训练；不能把极端胜负一概当坏样本删除，故意随机贴目/让子也会产生这种结果。[官方配置](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/configs/training/selfplay1.cfg)

**教师与迁移。** 可行路径是用更强的 C++ 推理网络生成格式兼容的自博弈训练数据，再用官方 trainer 训练本地网络；这是“借教师搜索产生标签”，不是当前已有自动蒸馏功能，成本包括教师推理和分布变化。SGF starts 只提供起始局面。`-initial-checkpoint` 则要求可加载、结构兼容的 PyTorch checkpoint；公开 `.bin.gz` 是 C++ 推理导出，不能当作完整 `.ckpt` 继续训练。若拿到真正 ckpt，需区分是否包含 optimizer/SWA/train_state；部分导出 ckpt 本身会跳过 optimizer。先核对版本、输入/目标格式、尺寸支持和来源，再做独立实验。[checkpoint 加载](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/katago/train/load_model.py)、[导出器](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/export_model_pytorch.py)、[训练器](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/train.py)

**暂缓增加模型与标签成本。** 在评测和数据覆盖尚未稳定时，不急于换更大架构；官方模板还提醒 `estimateLeadProb` 会花额外 visits，在低 visits 训练中设高可能相当慢。先核对已有设置与实测成本，再考虑增加 lead 标签。[官方配置](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/cpp/configs/training/selfplay1.cfg)

**Muon 与 MPS。** 本地版本包含 Muon/其变体，但当前 AdamW 已工作，不能仅凭新优化器名称换掉。Muon 的 Newton–Schulz 内部显式转 BF16 并做矩阵乘法；FP32 主网络并不代表这条分支是全 FP32。MPS 是否支持且划算需用当前 PyTorch/macOS 实测；trainer 的 AMP 分支还显式使用 CUDA autocast/GradScaler，不能假定 `-use-bf16/-use-fp16` 直接适用于 MPS。优化器名切换可能丢弃旧 optimizer state，应从隔离 checkpoint 比较数值稳定、每小时新数据消耗与棋力；没有证据承诺加速或涨棋。[Muon 源码](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/muon/muon.py)、[训练器](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/python/train.py)

## 当前快照的解释边界

当前参数背景：b10c128、MPS FP32、train batch 8；自博弈 2 threads / NN batch 4、每轮 24 局；棋盘权重 `[2,2,3,2,1]`。父任务在 `2026-10-24 00:22:24 +0800` 采集的只读本地快照如下（SGF、保留行与轮次 metrics 的边界不同，不能直接相加对账）：

| 棋盘 | 已解析 SGF | 开局双 pass | 平均着数 | 白胜 | 数据集保留行 |
|---|---:|---:|---:|---:|---:|
| 7 | 54 | 4（7.4%） | 20.3 | 43/54 | 593 |
| 8 | 44 | 7（15.9%） | 23.8 | 35/44 | 558 |
| 9 | 90 | 1（1.1%） | 26.3 | 76/90 | 1,239 |
| 13 | 61 | 0 | 15.1 | 55/61 | 465 |
| 19 | 39 | 0 | 7.4 | 37/39 | 167 |

最新轮次 metrics：累计 trained samples 6,964 / data rows 2,670；监控窗口 264 局、尚未完整，非法着率 0、极端结果率约 4.17%、开局双 pass 约 3.79%。程序报告 reuse 指标约 3.93，其口径应与脚本定义一起解释，不能用两个累计数简单相除代替。约 selfplay 17.9 秒、train 22.8 秒。[本地只读快照](/tmp/katago_training_analysis_snapshot_20261024.json)

值得调查的是 **8 路早期双 pass、13/19 路极短局和明显白胜偏向、19 路实际训练行很少**。这些现象可能与弱策略、贴目、终局判断和分布相互作用有关，现有小样本不能锁定因果，也不足以断言网络损坏。应按尺寸回看 SGF 的 pass 时点、规则、贴目、局面和模型来源，辅以固定对手评测。单看开局连续两 pass 会漏掉“大盘先随便走几手、再快速双 pass”的退化。游戏权重不是训练行权重：局长、full/cheap、surprise、side position 都会改变最终比例。

单机同步循环本身是官方支持的方式。当前轮次很小，可先量化启动、shuffle、导出和模型切换开销，适当增大每轮工作量以摊薄固定成本；不要直接把多机器全异步的效率结论搬到同一块 M4 GPU 上。是否异步还要测训练/推理争用、峰值内存和每小时有效数据。上述计时只是早期快照，不能据此认定最终瓶颈。[训练说明](https://github.com/lightvector/KataGo/blob/fd0723fdbc0e9d82cf269c9630af8c27c57c07c4/SelfplayTraining.md)

建议的下一次比较必须保存基线 checkpoint、有效配置、每尺寸数据量和相同评测套件。每次只改一个主要因素，比较相同墙钟预算下的结果；当前 run 继续作为基线积累证据。
