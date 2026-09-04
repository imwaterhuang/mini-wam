# Mini-WAM Colab 正式训练

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
