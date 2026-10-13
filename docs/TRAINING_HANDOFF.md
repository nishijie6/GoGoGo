# 多电脑接力训练

本功能用于两台或多台电脑轮流接力同一次普通版或 KataGo 训练。当前训练电脑使用自己的
本地 SSD 和 GPU，停止后生成一个完整接力包；下一台电脑在本地导入、检查、
手动启动续训。Git 用于同步程序，网盘、NAS 或移动硬盘可以用于搬运接力包。
程序不会上传模型、提交或推送 Git，也不会租用云 GPU。它只读取 Git 版本和忽略规则来校验接力包。

本地训练模型和数据默认不用移动到 Git。`training_runs/`、`handoff_bundles/`、
`*.gogogo-handoff.zip`、`*.ckpt`、`*.pt`、`*.pth`、`*.onnx` 和
`*.bin.gz` 均在项目的忽略规则内。仓库仅对
`model_snapshots/native/ordinary-stage-20261013.gogogo-handoff.zip` 设置发布例外，
通过 Git LFS 保存这一份指定的普通版阶段快照。KataGo 成果、日常导出的接力包和
导入后的训练目录仍被忽略。忽略规则对已被 Git 跟踪的文件无效；不要用
`git add -f` 批量添加训练目录。

## 从 GitHub 获取普通版阶段成果

仓库发布的普通版快照来自 `balanced_20260920/latest.pt`，保存第 8 轮、
700 次更新、11,182 条回放样本。它保留完整续训状态和运行目录，
`best_iteration` 仍为 0，尚无经过正式晋级评测的训练冠军。
它属于 `native` 后端，应通过 `train.py` 或 `训练围棋.ps1` 导入。

新电脑先安装 Git LFS，然后在项目目录下载真实接力包：

```powershell
git lfs install
git pull
git lfs pull

.\训练围棋.ps1 handoff-import `
  --bundle model_snapshots/native/ordinary-stage-20261013.gogogo-handoff.zip `
  --destination training_runs/native_from_github

.\训练围棋.ps1 handoff-check `
  --destination training_runs/native_from_github `
  --load-checkpoint
```

`training_runs/native_from_github` 必须尚不存在。只获取 Git LFS 指针文件而没有
完成 `git lfs pull` 时，文件不是可导入的 ZIP。读取完整检查点需要对应训练环境的
PyTorch；配置依赖后再执行检查和报告中的 `resume_command`。
训练仍使用新电脑自己的硬件，不会在 GitHub 上执行。
详见[普通版阶段快照说明](../model_snapshots/native/README.md)。

## 准备下一台电脑

两台电脑应使用相同的项目代码；KataGo 版还要使用相同的 KataGo 官方源码版本。
新电脑依照对应的训练说明准备 Python、PyTorch 和显卡依赖，KataGo 版还需
KataGo 引擎及其官方训练源码；接力包不包含软件安装环境。只校验
或导入接力包无需启动 KataGo，也无需 CUDA。`--load-checkpoint` 需要 PyTorch，
但只在 CPU 上读取检查点。

接力包以 ZIP 的无压缩存储方式封存训练目录和配置，逐文件记录 SHA-256。
导出电脑需要能容纳接力包的额外磁盘空间，导入电脑需要能容纳完整训练目录。
数据规模大时传输可能花较长时间；不要把正在训练的目录当作网盘实时同步目录。

同一次训练始终只由一台电脑运行。程序能阻止原目录在导出后自动恢复，但没有
跨电脑分布式锁；请自行协调哪台电脑接管。重复导入同一个包到不同电脑后，
同时运行会产生互相分叉的训练成果，后续不能通过简单复制合并。

## 选择对应的训练版本

| 训练版本 | 接力入口 | 完整续训检查点 | 恢复入口 |
| --- | --- | --- | --- |
| 普通版（项目自身的 PyTorch 训练器） | `train.py handoff-*` | `latest.pt` | `train.py train --resume ...` |
| KataGo 单棋盘或课程训练 | `train_rl.py handoff-*` | `train/gogogo/checkpoint.ckpt` | `train_rl.py continuous` 或 `curriculum` |

接力包在清单中保存 `trainer: "native"` 或 `trainer: "katago"`。两个入口会
拒绝另一版本的接力包和导入目录，不把 `.pt` 当成 KataGo 检查点，也不把
`.ckpt` 当成普通版成果。早期没有 `trainer` 字段的接力包按 KataGo 版识别。

## 普通版：A 导出，B 续训

A 先停止训练，等待训练主进程和自我对弈进程退出；尽量在完整一轮结束后交接。
前台训练可用 `Ctrl+C`，也可按普通版训练指南创建 `STOP` 文件。只暂停而未
退出的训练进程仍持有 `run.lock`，不能导出；不要删除锁文件绕过正在运行的进程。

普通版导出需要在已安装 PyTorch 的训练环境中读取 `latest.pt`，以其中保存的
配置作为依据，不能拿另一份 JSON 冒充现有成果的参数。Windows 推荐通过
`训练围棋.ps1` 转发到项目虚拟环境或现有 WSL 训练环境：

```powershell
.\训练围棋.ps1 handoff-export `
  --run-dir training_runs/balanced `
  --bundle handoff_bundles/native-0001.gogogo-handoff.zip

.\训练围棋.ps1 handoff-verify `
  --bundle handoff_bundles/native-0001.gogogo-handoff.zip
```

`--run-dir` 指向要交接的完整运行目录，该目录必须存在完整训练检查点
`latest.pt`。接力包保留模型、最佳模型、优化器、回放池、随机数状态和训练
计数，同时复制 `candidate.pt`、`best.pt`、`anchors/`、历史检查点、棋谱和
运行记录。单独的 `candidate.pt` 或 `best.pt` 用于推理评测，不能完整续训。
源目录的 `STOP`、`PAUSE`、进程锁和接力暂停标记不带入 B，避免恢复后立即
停止或永久暂停。导出成功后，A 的源目录留下 `HANDOFF.json`；新的训练及
独立评测写入会被锁保护拒绝，直到明确撤销接力。

B 复制接力包后导入到尚不存在的新目录：

```powershell
.\训练围棋.ps1 handoff-import `
  --bundle handoff_bundles/native-0001.gogogo-handoff.zip `
  --destination training_runs/native-relay-0001

.\训练围棋.ps1 handoff-check --destination training_runs/native-relay-0001
.\训练围棋.ps1 handoff-check --destination training_runs/native-relay-0001 --load-checkpoint

# 首次只续一轮，核对训练计数接续和实际更新。
.\训练围棋.ps1 train `
  --resume training_runs/native-relay-0001/run/latest.pt `
  --iterations 1
```

报告中的 `resume_checkpoint` 指向导入目录下的 `run/latest.pt`，
`training_config` 指向 `config/training.json`；报告还输出可复制的
`resume_command`。续训直接从完整检查点读取设置，无需传入 `--config`。
校验、导入和不带 `--load-checkpoint` 的检查无需 PyTorch；可直接用
`py train.py handoff-verify` 等入口。`--load-checkpoint` 在 CPU 上检查恢复
状态，不运行 GPU 自我对弈或更新；只有实际续训完成才能确认 B 的训练环境。

B 的显卡、精度、批次或进程数不同时，准备下文格式的 `local_hardware.json`，
导入时传入 `--hardware-profile`。报告会保存本机配置并生成带
`--hardware-overrides` 的续训命令，也可以通过训练脚本执行：

```powershell
.\训练围棋.ps1 handoff-import `
  --bundle handoff_bundles/native-0001.gogogo-handoff.zip `
  --destination training_runs/native-relay-0001 `
  --hardware-profile local_hardware.json

.\训练围棋.ps1 train `
  --resume training_runs/native-relay-0001/run/latest.pt `
  --hardware-overrides training_runs/native-relay-0001/config/hardware.json `
  --iterations 1
```

硬件覆盖只接受有效硬件字段、`self_play.workers`、
`self_play.inference_batch_size` 和 `optimizer.batch_size`，保留网络结构、
规则、学习率、回放和优化器状态。改批次或精度可能改变后续数值行为，首次
接管应做实际短训，并保留这份硬件配置。`--allow-version-mismatch` 可明确
允许项目版本差异，但仍检查哈希、禁止覆盖已有目录，兼容性需另外验收。

B 完成后，以导入目录下的 `run/` 再导出新编号：

```powershell
.\训练围棋.ps1 handoff-export `
  --run-dir training_runs/native-relay-0001/run `
  --bundle handoff_bundles/native-0002.gogogo-handoff.zip

# 只有在尚未交给 B，或已确认 B 停止后，才解冻 A 原来的运行。
.\训练围棋.ps1 handoff-cancel --run-dir training_runs/balanced
```

撤销只移除接力暂停标记，不会撤回别处的接力包，也不会删除源目录已有的
`STOP` 或 `PAUSE`；原电脑续训前按训练指南处理这些控制文件。正常返回 A
应导入最新包到另一个新目录，避免又从旧模型分叉。接力包和导入目录始终
作为本地训练备份，项目 Git 忽略规则会保护它们；前述指定的普通版发布快照是独立副本。

## KataGo：A 电脑导出

1. 先禁用自动训练任务，避免停止后又被守护触发重启。
2. 停止训练并等待训练进程及其 KataGo 子进程退出；尽量在完整一轮结束后交接。
3. 导出到训练目录外的新文件，随后校验接力包。

本项目默认 Windows 任务的暂停命令为：

```powershell
Disable-ScheduledTask -TaskName GoGoGo-KataGo-Curriculum
Stop-ScheduledTask -TaskName GoGoGo-KataGo-Curriculum
```

前台手动运行的训练可用 `Ctrl+C` 停止。导出会获取课程和各阶段互斥锁；有训练
进程占用时会失败，请等它退出后再执行。不要删除锁文件来绕过正在运行的进程。

默认导出 `config/rl_curriculum.rtx5070ti.json` 指定的课程及全部已存在阶段：

```powershell
py train_rl.py handoff-export `
  --bundle handoff_bundles\handoff-0001.gogogo-handoff.zip

py train_rl.py handoff-verify `
  --bundle handoff_bundles\handoff-0001.gogogo-handoff.zip
```

使用自己的课程配置时增加 `--curriculum-config <配置文件>`。仅接力一个棋盘、
没有使用课程编排器时，导出对应的训练配置：

```powershell
py train_rl.py handoff-export --single-run `
  --config config\rl_training.rtx5070ti.json `
  --bundle handoff_bundles\9x9-handoff-0001.gogogo-handoff.zip
```

导出包含完整的 `train/gogogo/` 续训检查点、自我对弈与洗牌数据、
`replay_rows.json`、已导出的模型、日志和课程记录等训练目录内容。只复制
`model.bin.gz` 无法恢复优化器和训练计数；导出目录中的 `model.ckpt` 也不能
代替训练器的完整 `train/gogogo/checkpoint.ckpt`。

导出成功会在源训练目录写入 `HANDOFF.json` 接力暂停标记。训练入口检查该标记，
旧电脑即使被计划任务重新触发，也会拒绝继续写入原训练目录。标记会保留到手动
撤销；原有模型和训练数据的内容不会被重新初始化或替换。

## KataGo：B 电脑导入并续训

先把接力包完整复制到 B。`--destination` 必须是尚不存在的新目录，程序会拒绝
覆盖任何已有目录，因此不会覆盖 B 上另一次训练的成果：

```powershell
py train_rl.py handoff-verify `
  --bundle handoff_bundles\handoff-0001.gogogo-handoff.zip

py train_rl.py handoff-import `
  --bundle handoff_bundles\handoff-0001.gogogo-handoff.zip `
  --destination training_runs\relay-0001

py train_rl.py handoff-check --destination training_runs\relay-0001
py train_rl.py handoff-check --destination training_runs\relay-0001 --load-checkpoint
```

导入会校验文件长度和哈希，生成指向新目录的本机配置，并重建需要迁移的绝对
路径。检查点和训练数据按原字节保存；导入后的训练目录可以独立使用，不需改动
项目中原来的课程配置。每个接力命令输出 JSON，失败时输出原因并返回非零状态。

在不支持原子目录发布的文件系统上，程序会独占创建目标目录，并用
`HANDOFF_IMPORTING.json` 标记阻止下方所有课程和阶段启动。完整内容发布后才会
移除标记。若断电或失败后仍保留该标记，请保留现场并重新导入到另一个新目录，
不要删除标记或通过 `handoff-cancel` 把未完成的导入当成可用成果。

导入报告的 `curriculum_config` 或 `training_config` 指向新生成的续训配置，
`resume_command` 是对应的启动命令。默认哈希检查应在首次启动前执行，因为
正式续训会修改检查点和数据，之后文件与导入时快照不同属于正常现象。

`--load-checkpoint` 使用受限的 PyTorch 加载器在 CPU 上核对模型、优化器及训练
状态，并检查模型张量。它不执行 GPU 模型推理、自我对弈或训练更新，也不能证明
新电脑的 CUDA 环境可用。完成首次接管时还需要执行报告中的启动命令，观察日志
中的训练计数是否接续，并确认至少一次实际更新成功。A 电脑保持停止。

默认导入会核对训练源码摘要；源码内容不同会被拒绝。Git 提交不同但训练源码摘要
一致时可以导入，例如发布文档或模型快照形成的新提交。优先同步到相同版本后重试。
确实需要在不同版本间迁移时，可显式增加 `--allow-version-mismatch`；结果会
保留兼容性警告，后续检查和实际续训验收仍须通过。该参数只允许版本差异，不会
跳过接力包的哈希校验或允许覆盖现有目录。

## 换显卡时调整硬件参数

可为 B 准备一个本机硬件覆盖文件，例如 `local_hardware.json`：

```json
{
  "hardware": {
    "device": "cuda:0",
    "precision": "amp_bfloat16",
    "gpu_memory_fraction": 0.85,
    "compile_model": false,
    "data_loader_workers": 2
  },
  "self_play": {
    "workers": 4,
    "inference_batch_size": 32
  },
  "optimizer": {
    "batch_size": 256
  }
}
```

这是一份字段格式示例。精度和批次应按 B 的显卡能力设置，以上数值没有经过
每种显卡的实际吞吐或显存验收。

```powershell
py train_rl.py handoff-import `
  --bundle handoff_bundles\handoff-0001.gogogo-handoff.zip `
  --destination training_runs\relay-0001 `
  --hardware-profile local_hardware.json
```

文件也可使用 `{"overrides": {...}}` 包装。允许覆盖的是 `hardware` 中的有效
硬件字段、`self_play.workers`、`self_play.inference_batch_size` 和
`optimizer.batch_size`。规则、棋盘、网络、搜索预算、学习率、课程门槛及数据
位置不能通过该文件改动，错误或不允许的字段会被拒绝。

KataGo 课程的覆盖会应用到全部导入阶段。显式指定批次后，课程状态中旧电脑保存的自动批次
会被清除，后续阶段的候选批次限制到该值，避免又选回旧显卡的大批次。检查点的
优化器状态仍被保留；批次调整可能改变后续训练行为，应在交接记录中保存这份
硬件配置。

## KataGo：返回 A 或撤销接力

B 停止后，用导入报告中的配置再导出下一份接力包：

```powershell
py train_rl.py handoff-export `
  --curriculum-config "<上次导入报告中的 curriculum_config>" `
  --bundle handoff_bundles\handoff-0002.gogogo-handoff.zip
```

如果是单棋盘训练，使用 `--single-run --config "<training_config>"`。A 收到
新包后导入另一个新目录，如 `training_runs\relay-0002`，从最新成果接管；
不要解冻旧的 `relay-0001` 来代替新成果。

尚未被其他电脑接管、确实要回到原目录继续时，手动撤销源目录的接力暂停：

```powershell
py train_rl.py handoff-cancel

# 使用过自定义课程配置时传入同一份配置
py train_rl.py handoff-cancel --curriculum-config "<导出时的课程配置>"

# 单棋盘训练
py train_rl.py handoff-cancel --single-run --config config\rl_training.rtx5070ti.json
```

撤销只移除原目录的接力暂停标记，不撤回已经复制或导入到别处的包，也不会
删除原训练成果。若包已被 B 接管，先确认 B 停止，再决定从哪份成果恢复，
避免两台电脑同时续跑旧快照。恢复 Windows 计划任务时，也应确认它的启动配置
指向当前接管的目录；原任务默认仍指向项目内原来的课程配置。

可以保留最近几份接力包以及各棋盘的阶段里程碑。每次交接使用不同编号；
导出不会覆盖已有接力包，复制完并校验成功后再决定是否清理旧备份。
