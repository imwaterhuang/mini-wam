# Mini-WAM Colab 训练入口

## 当前 ACT 阶段（2026-09-14）

ACT（Action Chunking with Transformers，基于 Transformer 的动作分块）是当前唯一训练目标。全部 206 个示范回合用于训练，不设置示范验证集；验证改为随机模拟场景闭环执行。
模型、全量数据适配及训练/评测入口尚未实现，本页暂不提供可直接执行的 ACT 训练命令；下方现有命令训练的是历史模型。
当前规格见 [SPEC.md](SPEC.md)，阶段状态见 [PROGRESS.md](PROGRESS.md)。

ACT 云端训练接入顺序：

1. 完成模型审查、全量数据接口和 64 窗口过拟合，准备独立的 ACT 配置及运行目录。
2. 核对回合清单包含全部 206 个唯一回合，并在全量有效数据上重算 ACT 归一化统计；不加载旧 185/21 划分。
3. 在目标设备检查实际消费字段、短训练、数据/传输/计算/端到端耗时以及完整恢复。无示范验证数据加载器。
4. 根据测速确定批大小与总训练步数，登记候选保存频率、随机场景验证频率和选点规则，再启动种子 0 训练。
5. 使用同一批随机生成并记录种子的开发场景选择 `best.pt`；独立保存 `last.pt` 及恢复状态，镜像到持久存储。
6. 冻结选定模型后，在新随机场景上复核，记录结果与视频。原封存测试集保持未使用。

ACT 不继承旧命令中的离线 `val_loss` 选点方式、旧归一化文件或旧训练已完成的状态。

## 历史 action_only / future_aware 操作说明

> 以下为 2026-09-10 已收尾实验的训练/恢复说明，不是 ACT 入口。
> 历史两模型按最低离线验证损失选点；其数据划分、配置与检查点保持原样。
> 核对历史结果及重新评估使用 [docs/REPRODUCING.md](docs/REPRODUCING.md)。

本流程让训练在 Colab 的本地磁盘高速运行，并在每次验证和 checkpoint
（训练检查点）保存后，把完整运行目录增量镜像到 Google Drive。运行时意外断开时，
最多损失 1,000 个训练 step（步）；每 5,000 步额外保留一个编号归档。

## 1. 选择 GPU 运行时

在 Colab 中选择 GPU（Graphics Processing Unit，图形处理器）运行时，然后执行：

```python
import sys, torch
assert sys.version_info >= (3, 12), f"需要 Python 3.12+，当前为 {sys.version}"
assert torch.cuda.is_available(), "没有检测到 CUDA，请重新选择 GPU 运行时"
print(sys.version)
print(torch.cuda.get_device_name(0))
```

CUDA（Compute Unified Device Architecture，统一计算设备架构）由 Colab 当前的
PyTorch 提供。不要用本项目的本地完整锁定文件覆盖它。

## 2. 获取代码与安装依赖

```bash
%cd /content
!git clone https://github.com/mcintoshjody788-collab/mini-wam.git
%cd /content/mini-wam
!python -m pip install -q -r requirements-colab.txt
!python -m pip install -q -e . --no-deps
```

私有仓库必须先在 Colab 中完成 GitHub 授权。不要把 GitHub token（访问令牌）直接
写入 Notebook（笔记本）代码或输出。

安装依赖后如果 Colab 提示重启运行时，先重启，再回到项目目录继续执行。

## 3. 下载并验证数据

```bash
%cd /content/mini-wam
!python scripts/download_dataset.py
```

通过条件固定为 206 个 episode（回合）、25,650 帧。脚本还会输出数据指纹。

## 4. 挂载持久化目录并运行前置检查

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/mini-wam
!python scripts/check_training_ready.py \
  --config configs/action_only_seed0.yaml \
  --run-root /content/drive/MyDrive/mini-wam-runs \
  --require-cuda
```

只有输出 `"ready": true` 才进入训练。

### 4.1 训练速度诊断

正式训练前或发现 step 明显偏慢时，在同一个 GPU runtime 中运行独立诊断：

```bash
%cd /content/mini-wam
!python scripts/profile_action_only.py \
  --config configs/action_only_seed0.yaml \
  --checkpoint /content/drive/MyDrive/mini-wam-runs/action-only-seed0/checkpoints/last.pt \
  --device cuda \
  --num-workers 4 \
  --warmup-batches 5 \
  --measure-batches 20 \
  --output /content/drive/MyDrive/mini-wam-runs/action-only-seed0/performance_profile.json
```

先分别使用 `--num-workers 0`、`2`、`4` 和 `--data-only` 做短测试，再用最快且没有
worker 异常的值运行上面的完整诊断。本机实测 4 workers 最快，但 Colab 分配的 CPU
数量不同，不能直接把本机结果当作 A100 runtime 的结论。

该脚本不会覆盖 checkpoint 或训练指标。它在独立进程内对比旧完整数据路径与跳过未来
图像的 `action_only` 路径，并分别报告数据准备、Host-to-Device（主机到设备）传输、
GPU 计算和端到端耗时。只有 `used_fields_exactly_equal` 为 `true` 才允许从原 checkpoint
恢复。新版 sampler 只在 optimizer 更新完成后推进 checkpoint 位置，因此 worker 预取
不会再让恢复点提前；旧版 `num_workers=0` checkpoint 仍可直接加载。

## 5. GPU 冒烟测试

本地运行目录和 Drive 镜像目录必须不同：

```bash
%cd /content/mini-wam
!python scripts/train_action_only.py \
  --config configs/action_only_smoke.yaml \
  --run-dir /content/mini-wam-work/smoke \
  --mirror-dir /content/drive/MyDrive/mini-wam-runs/smoke \
  --device cuda
```

确认以下文件同时出现在 Drive 镜像目录：

- `checkpoints/last.pt`
- `checkpoints/best.pt`
- `train_metrics.csv`
- `val_metrics.csv`
- `environment.json`
- `normalization.json`

## 6. 正式训练 seed 0

```bash
%cd /content/mini-wam
!python scripts/train_action_only.py \
  --config configs/action_only_seed0.yaml \
  --run-dir /content/mini-wam-work/action-only-seed0 \
  --mirror-dir /content/drive/MyDrive/mini-wam-runs/action-only-seed0 \
  --num-workers 4 \
  --device cuda
```

checkpoint 选择规则预先固定为：使用离线验证损失最低的 `best.pt`。不得查看最终测试
场景来挑选模型。

### 断线后恢复

重新克隆代码、安装依赖、下载数据并挂载 Drive 后执行：

```bash
%cd /content/mini-wam
!python scripts/train_action_only.py \
  --config configs/action_only_seed0.yaml \
  --resume /content/drive/MyDrive/mini-wam-runs/action-only-seed0/checkpoints/last.pt \
  --run-dir /content/mini-wam-work/action-only-seed0 \
  --mirror-dir /content/drive/MyDrive/mini-wam-runs/action-only-seed0 \
  --num-workers 4 \
  --device cuda
```

训练器会先把 Drive 中已有的运行记录恢复到本地目录，再从下一批数据精确继续。命令中的
`4` 应替换成当前 Colab runtime 在 `0/2/4` 对照中实测最快且稳定的 worker 数。

## 7. 只在开发场景评估

```bash
%cd /content/mini-wam
!python scripts/evaluate_action_only.py \
  --checkpoint /content/drive/MyDrive/mini-wam-runs/action-only-seed0/checkpoints/best.pt \
  --output-dir /content/drive/MyDrive/mini-wam-reports/action-only-seed0-development \
  --device cuda
```

评估输出包含逐回合 CSV（Comma-Separated Values，逗号分隔值）、汇总 JSON
（JavaScript Object Notation，JavaScript 对象表示法）、全部 20 个开发场景视频、
成功率、覆盖率、回报、参数量、推理耗时和峰值显存。阶段 3 的脚本会拒绝最终测试
场景；50 个最终测试场景要到阶段 5 才能使用。

## 8. future_aware 的可选图片缓存

`future_aware` 每个窗口读取 2 张历史图片和最多 4 张未来图片。对于处理器紧张的
运行环境，可以把原读取路径产生的归一化图片一次性保存到本地内存映射文件，
让训练和验证直接取用完全相同的张量。缓存约占用 2.64 GiB（Gibibyte，吉比字节）
本地磁盘，不复制到 Drive；不同 worker 共享文件对应的内存页。

在 Colab 终端中构建，已有同名缓存时脚本会拒绝覆盖：

```bash
cd /content/mini-wam
python scripts/build_image_cache.py \
  --config configs/future_aware_seed0.yaml \
  --output /content/mini-wam-work/pusht-normalized-cache
```

读取器会核对源数据、归一化实现、依赖版本、帧数、形状和缓存校验值。只有设置
`MINI_WAM_IMAGE_CACHE` 才启用缓存；不设置时保留原读取路径。缓存不会改变训练
配置、动作、掩码、时间对齐、模型、优化器、学习率或采样器。

在正式训练已暂停、有效检查点已备份后，运行分阶段测速与恢复验证。该脚本会
做有限的参数更新作为诊断，但不会改写传入检查点或正式训练产物：

```bash
python scripts/profile_future_aware.py \
  --checkpoint /content/mini-wam-work/perf-transition/resume.pt \
  --cache /content/mini-wam-work/pusht-normalized-cache \
  --output /content/mini-wam-work/perf-transition/profile.json
```

必须看到 `TENSORS_EXACT`、三个 `RESUME_EXACT` 和 `PROFILE_VERIFIED`，再按
输出中的 `selected_workers` 选择加载进程数。诊断进程会开启 cuDNN（CUDA Deep Neural Network library）的确定性卷积，隔离非确定性运算对逐位恢复对照的影响；正式训练沿用原有运算设置。结果记录数据准备、主机到显卡传输、
计算和端到端耗时；不要在另一个训练同时占用显卡时解读其绝对吞吐。

后续从有效 `last.pt` 续跑时，在原有命令前设置环境变量，例如：

```bash
export MINI_WAM_IMAGE_CACHE=/content/mini-wam-work/pusht-normalized-cache
```

然后使用 `scripts/train_future_aware.py --resume ...` 及原训练配置、运行目录和
Drive 镜像目录。运行时重建后须先重新构建本地缓存；若不启用缓存，同一检查点
仍可通过原始数据路径恢复。训练环境记录中会保存启用的缓存路径和清单校验值。
