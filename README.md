# Mini-WAM

> 详细阶段进度与新窗口续接说明：[PROGRESS.md](PROGRESS.md)

在 Push-T 上公平比较 `action_only` 与带训练期未来表示辅助任务的
`future_aware`。完整模型和评估边界见 [SPEC.md](SPEC.md) 与
[EVALUATION.md](EVALUATION.md)。

## 总体路线

1. **环境**：固定 Python、LeRobot、PyTorch 与 Push-T 版本，并跑通环境自检。
2. **数据**：审计数据，按 episode 划分，建立时间窗口、mask 和归一化。
3. **最小过拟合**：让 `action_only` 在固定 64 个训练窗口上明显过拟合。
4. **闭环基线**：完成训练、保存、恢复与固定初始状态评测，超过随机动作。
5. **未来分支**：加入动作条件未来表示预测，验证正确动作优于打乱或反向动作。
6. **公平比较**：相同数据、初始化、训练预算和评测器下运行多个随机种子。
7. **项目包装**：输出结果表、曲线、模型图、成功与失败视频和结论边界。

每个阶段必须先通过正确性门禁，再进入下一阶段。

## Push-T 模型执行工作台

本地网页工作台只做三件事：检查并回放专家数据、检查并切换
`ActionOnlyPolicy` checkpoint（检查点）、在 Push-T 中执行真实闭环 rollout
（滚动执行）。它不会在网页中训练模型，也不会把刚导入的数据自动学进模型。

已有环境直接启动：

```bash
./.venv/bin/python app.py
```

然后打开 <http://127.0.0.1:7860>。服务只监听本机地址，不生成公开分享链接。

- “执行任务”：用 seed（随机种子）生成场景，或选择智能体/T 块后点击画布布置；
  模型每次预测 16 步、执行前 4 步，再重新观察，最多运行 300 步。
- “专家数据”：支持内置 `lerobot/pusht_image`、Hugging Face repo ID
  （仓库标识）与本地 LeRobot 目录，并导出 384×384 的 MP4（MPEG-4 Part 14）回放。
- “过拟合诊断”：读取 `action_only_overfit.pt` 中保存的 64 个确切训练窗口，
  对比两帧训练输入、16 步模型预测与专家动作，并重新计算训练误差；还可从
  图像近似恢复 T 块位姿，让模型从所选训练窗口开始闭环执行并导出视频。
- “模型管理”：自动扫描 `artifacts/*.pt`，也可上传同架构 `.pt`；加载使用
  `torch.load(weights_only=True)`，参数名、形状与元数据不兼容时直接拒绝。
- 每次闭环结果写入 `artifacts/studio/rollouts/`，视频旁保存同名 JSON
  （JavaScript Object Notation，JavaScript 对象表示法）指标。

第一版仅支持固定绿色目标的 Push-T 与当前 `ActionOnlyPolicy`：两帧
`[B,2,3,96,96]` 图像、两帧二维位置，输出 `[B,16,2]` 动作块。

## 第一步：建立环境

```bash
conda env create -p ./.venv -f environment.yml
conda run -p ./.venv python scripts/check_environment.py
```

自检通过后会生成 `artifacts/environment.json`。实际解析出的完整 Python 包版本
随后冻结到 `requirements-lock.txt`。

### 第一个 Push-T 演示

```bash
conda run -p ./.venv python examples/01_random_rollout.py
```

它会让随机动作控制智能体 120 步，并保存
`artifacts/random_rollout.gif`。这个脚本展示最小闭环：

```text
observation -> action -> env.step(action) -> next observation + reward
```

### 用鼠标操作 Push-T

```bash
conda run -p ./.venv python examples/02_mouse_control.py
```

- 移动鼠标：设置蓝色圆球下一步追踪的目标坐标。
- `R`：重置任务。
- `Esc` 或 `Q`：退出。
- 目标：将灰色 T 推入绿色 T；`coverage` 越接近 `1` 越好。

## 第二步：审计示范数据

```bash
conda run -p ./.venv python scripts/audit_dataset.py
```

输出：

- `artifacts/data_audit.json`：episode、帧、范围、连续性和训练集归一化统计。
- `artifacts/data_samples.png`：16 个“历史两帧—未来四帧”样本。
- `splits/episodes_seed42.json`：按完整 episode 固定的 90% 训练、10% 验证划分。

示范数据不另设测试集。最终测试使用一组独立生成并冻结的 Push-T 随机初始状态；
开发期间使用另一组不重叠的初始状态选择 checkpoint。

### 查看一个模型训练样本

```bash
conda run -p ./.venv python scripts/inspect_training_sample.py
```

### 验证数据契约

```bash
conda run -p ./.venv pytest tests/test_dataset.py -q
```

## 可恢复的 `action_only` 训练

先运行 100 步本地 smoke test（冒烟测试）：

```bash
./.venv/bin/python scripts/train_action_only.py \
  --config configs/action_only_smoke.yaml \
  --run-dir runs/action_only/0/my-smoke \
  --stop-after-step 50 \
  --device cpu

./.venv/bin/python scripts/train_action_only.py \
  --config configs/action_only_smoke.yaml \
  --resume runs/action_only/0/my-smoke/checkpoints/last.pt \
  --device cpu
```

第二条命令会从第 50 步继续到第 100 步。checkpoint（检查点）包含模型、
优化器、学习率调度器、确定性数据采样位置、随机数状态、完整配置、数据划分
哈希、归一化统计和源码指纹。训练指标与验证指标分别写入
`train_metrics.csv` 和 `val_metrics.csv`。

正式单种子配置为 `configs/action_only_seed0.yaml`。运行正式训练前，必须先冻结
开发评估和最终测试的 Push-T 初始状态；正式训练不得使用 smoke test 中限制验证
批次数的诊断配置。

## 固定评测场景与随机策略基线

```bash
./.venv/bin/python scripts/freeze_evaluation_scenes.py
./.venv/bin/python scripts/evaluate_random_policy.py
```

第一条命令固定 20 个开发场景和 50 个最终测试场景。场景文件采用写入后保护：
内容不同的文件不能被脚本静默覆盖。第二条命令只接受开发场景文件；如果误传
最终测试文件会直接拒绝运行。

当前随机策略在 20 个开发场景上的结果：成功率 `0%`，平均最终覆盖率 `0`，
平均最大覆盖率约 `0.120112`，平均回报约 `1.223907`。逐回合结果位于
`reports/baselines/random_policy_development.json` 和对应 CSV（Comma-Separated
Values，逗号分隔值）文件。

## 固定版本

- Python：3.12
- LeRobot：0.6.0
- LeRobot 上游提交：`30da8e687a6dfc617fcd94afc367ac7071c376ce`
- Push-T：由 `lerobot[pusht]==0.6.0` 的依赖约束解析并在锁定文件中记录
