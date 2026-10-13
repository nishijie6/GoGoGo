# 普通版阶段训练快照

本目录通过 Git LFS 发布一份普通版（`native`）完整续训包：
[`ordinary-stage-20261013.gogogo-handoff.zip`](ordinary-stage-20261013.gogogo-handoff.zip)。
它来自本地正式训练运行 `balanced_20260920`，可导入新目录后继续本地训练。

| 项目 | 保存的状态 |
| --- | --- |
| 训练后端 | 项目自身的 PyTorch 训练器，入口为 `train.py` |
| 完整检查点 | `latest.pt` |
| 训练轮次 | 8 |
| 更新次数 | 700 |
| 回放样本 | 11,182 |
| 已接受最佳模型轮次 | 0，仍为初始化模型 |
| 原完整检查点 SHA-256 | `5e804fafb24c6ba4aa704aa755ddc5dff46ecaa5f58b708bff02e2f43b32e520` |

这是已有训练进度的阶段成果，尚未通过正式晋级评测，不能把它当作已证明棋力
最强的模型。包内包含完整模型、最佳模型、优化器、回放池、随机数状态、
训练计数、历史检查点和运行记录，不只是用于推理的权重。

## 下载、校验和恢复

在新电脑安装 Git LFS 和[普通版训练依赖](../../docs/RL_TRAINING.md)，然后在
仓库根目录执行：

```powershell
git lfs install
git pull
git lfs pull

.\训练围棋.ps1 handoff-verify `
  --bundle model_snapshots/native/ordinary-stage-20261013.gogogo-handoff.zip

.\训练围棋.ps1 handoff-import `
  --bundle model_snapshots/native/ordinary-stage-20261013.gogogo-handoff.zip `
  --destination training_runs/native_from_github

.\训练围棋.ps1 handoff-check `
  --destination training_runs/native_from_github `
  --load-checkpoint
```

`git pull` 获取项目代码和 Git LFS 指针；`git lfs pull` 获取真正的接力包。
如果克隆时已经成功下载 LFS 文件，再执行 `git lfs pull` 不会重复下载现有内容。
若文件只有几行 Git LFS 指针文本，先完成 LFS 下载再导入。

导入目标必须是尚不存在的新目录。导入和校验不会自动开始训练；成功报告中的
`resume_checkpoint` 指向 `training_runs/native_from_github/run/latest.pt`，
`resume_command` 给出对应的续训命令。首次可只续一轮：

```powershell
.\训练围棋.ps1 train `
  --resume training_runs/native_from_github/run/latest.pt `
  --iterations 1
```

检查点读取需要 PyTorch；`handoff-check --load-checkpoint` 在 CPU 上核对恢复
状态，实际续训才会使用当前电脑的 CPU 或 GPU。换显卡需要调整批次、精度或
并发数时，在导入时增加 `--hardware-profile`，并使用报告中带硬件覆盖的
`resume_command`。完整操作见[多电脑接力指南](../../docs/TRAINING_HANDOFF.md)。

## 后续训练成果

这一文件是固定的阶段快照。导入后的新检查点保存在被忽略的 `training_runs/`
中，接力程序不会自动提交或上传后续模型。该发布例外只覆盖此普通版快照；
KataGo 模型、检查点和接力包继续保存在本地。

同一次接力训练仍应由一台电脑运行。多台电脑分别导入后同时续训会产生独立
分支，无法通过复制检查点合并优化器和训练进度。
