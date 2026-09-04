# Mini-WAM 详细进度

最后更新：2026-09-04

这份文件是 Mini-WAM 的阶段交接记录。每次开启新窗口时，先读取本文件，再读取当前阶段对应的设计文件和代码；不要仅凭聊天记录判断进度。

## 新窗口工作规则

1. 先核对本文件、`SPEC.md`、当前代码和验证产物，确认记录没有落后于工作区。
2. 已通过验证的阶段不重复实现；如环境或依赖发生变化，只做必要的复查。
3. 每个阶段结束前必须满足该阶段的通过条件，并把验证结果和下一步写回本文件。
4. 模型代码由学习者本人编写。助手负责解释、拆分任务、审查代码和运行验证，不直接代写模型实现。
5. 最终测试状态只用于最后评价，不用于调参；开发过程使用训练集和验证集。

## 总体进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 本地环境、Push-T 运行与画面查看 | 已完成 |
| 1 | 数据审计、划分、窗口采样与归一化 | 已完成 |
| 2 | `action_only` 模型与 64 样本过拟合 | 已完成 |
| 3 | `action_only` 闭环训练与基线评测 | 进行中 |
| 4 | `future_aware` 模型与反事实检验 | 未开始 |
| 5 | 多随机种子正式评测 | 未开始 |
| 6 | 项目整理、复现实验与总结 | 未开始 |

## 当前交接点

- 当前阶段：阶段 3——`action_only` 闭环基线。
- 阶段 2 已通过：64 样本损失从 `0.423381` 降至 `0.006625`，下降约 98.4%；checkpoint 重新加载输出一致。
- 可恢复训练基础设施已完成：正式配置、smoke test 配置、训练/验证指标、完整 checkpoint 和确定性数据采样恢复。
- 已实测从第 50 步恢复到第 100 步；恢复训练与不中断训练的最终模型参数逐张量完全一致，学习率调度器和采样器状态也一致。
- 已冻结互不重叠的 20 个开发评估场景和 50 个最终测试场景；最终测试文件已标记为阶段 5 前禁止评测。
- 20 回合随机策略基线已完成：成功率 `0%`，平均最终覆盖率 `0`，平均最大覆盖率 `0.120112`，平均回报 `1.223907`。
- 未阻塞警告：macOS 运行模拟器时，OpenCV 与 Pygame 各自携带的 SDL（Simple DirectMedia Layer，跨平台多媒体库）会报告重复动态类；本轮 20 回合和全部测试均未崩溃，若后续出现图形相关异常再隔离依赖处理。
- 当前下一步：核对正式训练硬件与配置，然后启动种子 0 的 `action_only` 正式训练。
- `action_only_full.pt` 是早期 1000 步诊断模型，不视为正式阶段 3 模型；现有闭环运行尚无成功回合。

当前新窗口续接提示：

```text
请先读取 PROGRESS.md、SPEC.md、冻结的场景清单和随机策略报告。阶段 2 已完成，阶段 3 的可恢复训练、场景冻结和随机基线已验证。下一步核对正式训练硬件与配置，然后只启动种子 0 的 action_only 正式训练；不要开始 future_aware，也不要运行最终测试场景。
```

## 已确定的项目约定

- `observation`（观测）使用当前帧和前一帧：图像 `[2, 3, 96, 96]`，位置 `[2, 2]`。
- `action`（动作）是 Push-T 中的绝对目标坐标，不是相对位移。
- 每个样本预测 16 步动作，动作张量为 `[16, 2]`。
- 未来监督使用 4 帧，未来图像张量为 `[4, 3, 96, 96]`。
- 序列尾部不足时采用 padding（填充），并用布尔 mask（掩码）排除无效位置。
- 归一化参数只从训练集统计；验证数据也使用训练集参数。
- reward（奖励）只参与闭环评价，不作为当前行为克隆训练的输入或损失。
- 数据集只划分训练集和验证集；最终测试使用预先冻结的随机 Push-T 初始状态。
- 调参用的开发初始状态与最终测试初始状态必须分开冻结，避免把最终测试变成调参集。

## 阶段 0：本地环境与 Push-T

状态：已完成。

已完成：

- 创建本地 Python 环境并锁定主要依赖。
- 验证 PyTorch、TorchVision、LeRobot、Gymnasium 和 `gym-pusht` 可以导入。
- 验证 FFmpeg（Fast Forward Moving Picture Experts Group，多媒体处理工具）8.1.2 和 TorchCodec 0.11.1 可用。
- 验证 Push-T 的 `reset`、`step` 和 `render`。
- 生成随机动作演示 GIF（Graphics Interchange Format，图形交换格式）。
- 提供鼠标控制示例，用于亲自操作和观察 Push-T。

当前环境摘要：

- Python 3.12.14
- PyTorch 2.11.0
- TorchVision 0.26.0
- LeRobot 0.6.0
- Gymnasium 1.3.0
- `gym-pusht` 0.1.6
- 当前验证设备为 CPU（Central Processing Unit，中央处理器）。MPS（Metal Performance Shaders，Metal 性能着色器）已编译但在当前执行环境不可用。
- 本地 CPU 足够做数据处理、单元测试和小规模过拟合；正式训练计划使用云端 GPU（Graphics Processing Unit，图形处理器）。

验证证据：

- `artifacts/environment.json`
- `artifacts/random_rollout.gif`
- `scripts/check_environment.py`
- `examples/01_random_rollout.py`
- `examples/02_mouse_control.py`
- `environment.yml`
- `requirements-lock.txt`
- `splits/dev_scenes.json`：20 个固定开发场景。
- `splits/final_test_scenes.json`：50 个封存的最终测试场景。
- `reports/baselines/random_policy_development.json`：20 回合随机策略基线。

通过条件：环境检查通过，Push-T 可执行一步并渲染画面。已满足。

## 阶段 1：数据管线

状态：已完成。

已完成：

- 审计 `lerobot/pusht_image` 数据集。
- 确认共有 206 个 episode（回合）、25,650 帧，索引连续且无重复。
- 固定数据划分：训练集 185 个 episode，验证集 21 个 episode，无重叠。
- 实现基于 episode 的窗口采样，避免跨回合取样。
- 实现 2 帧历史、16 步动作、4 帧未来监督。
- 实现序列尾部填充及相应布尔掩码。
- 实现只使用训练集统计量的归一化。
- 验证完整训练集可产生 22,820 个训练窗口。
- 验证 DataLoader（数据加载器）的批次形状：图像 `[4, 2, 3, 96, 96]`，动作 `[4, 16, 2]`。
- 数据单元测试为 `4 passed`。

单样本数据契约：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `pixels` | `[2, 3, 96, 96]` | 前一帧与当前帧图像 |
| `agent_pos` | `[2, 2]` | 两帧对应的机械体位置 |
| `actions` | `[16, 2]` | 从当前时刻开始的 16 步动作 |
| `action_mask` | `[16]` | 有效动作位置 |
| `future_pixels` | `[4, 3, 96, 96]` | 从下一时刻开始的 4 帧未来图像 |
| `future_mask` | `[4]` | 有效未来帧位置 |
| `episode_id`、`frame_id` | 标量 | 样本来源和时间定位 |

验证证据：

- `artifacts/data_audit.json`
- `artifacts/data_samples.png`
- `splits/episodes_seed42.json`
- `src/mini_wam/data/dataset.py`
- `scripts/audit_dataset.py`
- `scripts/inspect_training_sample.py`
- `tests/test_dataset.py`

通过条件：数据语义明确、划分无泄漏、形状正确、尾部掩码正确、测试通过。已满足。

## 阶段 2：`action_only` 模型与小样本过拟合

状态：已完成。

目标：先实现一个只预测动作的行为克隆基线，证明模型、损失和训练循环可以工作。

待完成：

- [x] 创建 `src/mini_wam/models/__init__.py`。
- [x] 创建 `src/mini_wam/models/action_only.py`。
- [x] 学习者实现 `VisualEncoder`：`[B,2,3,96,96] -> [B,2,512]`。
- [x] 学习者实现位置编码和历史融合。
- [x] 学习者实现动作预测头。
- [x] 组合 `ActionOnlyPolicy`，输出 `[B,16,2]`。
- [x] 实现只计算有效动作位置的 masked Smooth L1 loss（带掩码的平滑 L1 损失）。
- [x] 增加模型形状和掩码损失测试。
- [x] 在固定的 64 个训练窗口上过拟合。

验证证据：

- `artifacts/action_only_overfit.pt`
- `scripts/overfit_action_only.py`
- `tests/test_action_only.py`
- 初始损失 `0.423381`，最终损失 `0.006625`。

通过条件：

- 模型输入、输出形状全部通过测试。
- padding 位置不参与损失。
- 固定 64 个窗口上的训练损失持续下降，并达到足以证明管线可学习的水平。
- 保存一次可重新载入的 checkpoint（检查点）。

阶段完成后的新窗口提示：

```text
请读取 PROGRESS.md 和阶段 2 的验证结果。阶段 2 已完成，现在只开始阶段 3：训练 action_only 基线并做闭环评测。先核对 checkpoint、训练配置和冻结的开发初始状态，不要改 future_aware 模型。
```

## 阶段 3：`action_only` 闭环基线

状态：进行中。

计划：

- [x] 建立可恢复的训练脚本和配置。
- [x] 在完整训练开始前分别冻结开发初始状态和最终测试初始状态；封存最终测试状态，不在开发阶段运行。
- [ ] 在完整训练集上训练 `action_only`。
- [ ] 在相同初始状态上比较学习策略与随机策略。
- [ ] 记录 reward、成功率、轨迹长度和推理耗时。
- [ ] 导出成功与失败视频，直接观察模型表现。

当前验证证据：

- `configs/action_only_smoke.yaml`：100 步本地恢复测试配置。
- `configs/action_only_seed0.yaml`：50,000 步正式种子 0 配置，尚未启动正式训练。
- `scripts/train_action_only.py`：训练与恢复入口。
- `src/mini_wam/training/action_only.py`：验证、指标、checkpoint 和确定性采样实现。
- `tests/test_training.py`：配置与精确恢复测试。
- `scripts/freeze_evaluation_scenes.py`：生成并保护 20/50 场景划分。
- `scripts/evaluate_random_policy.py`：只允许在开发场景上运行的随机策略评测。
- `splits/evaluation_scenes_manifest.json`：场景数量、哈希和最终测试封存状态。
- `reports/baselines/random_policy_development.json`：20 回合随机基线明细和汇总。
- `runs/action_only/0/resume-smoke-20260904-v2/`：50 步中断并恢复到 100 步的真实产物。
- `runs/action_only/0/SMOKE_VERIFICATION.md`：中断恢复、不中断对照及清理记录。
- 两条路径最终模型参数、学习率调度器、采样器状态及验证损失完全一致；为避免重复占用约 1.5 GB，只保留一个可执行的最终 smoke checkpoint。

通过条件：模型能够闭环控制 Push-T；评测可以复现；至少保留代表性的成功和失败视频。

阶段完成后的新窗口提示：

```text
请读取 PROGRESS.md、action_only 的训练配置和闭环评测产物。阶段 3 已完成，现在开始阶段 4：future_aware。先说明它相对 action_only 新增的输入、输出和损失，再让我逐个组件自己实现。
```

## 阶段 4：`future_aware` 模型

状态：未开始。

计划：

- [ ] 冻结或受控更新未来表征的目标编码器。
- [ ] 编码动作前缀并预测未来表征。
- [ ] 在动作损失之外加入 masked future loss（带掩码的未来损失）。
- [ ] 比较正确动作、打乱动作和反向动作条件下的未来预测。
- [ ] 确认未来分支确实依赖动作，而不是只外推画面。
- [ ] 在与 `action_only` 相同的开发初始状态上做闭环比较。

通过条件：未来预测对动作变化敏感，且对照实验能够排除“忽略动作”的退化解。

阶段完成后的新窗口提示：

```text
请读取 PROGRESS.md 和 future_aware 的反事实检验结果。阶段 4 已完成，现在开始阶段 5：正式多随机种子评测。不得用最终测试状态继续调参。
```

## 阶段 5：正式多随机种子评测

状态：未开始。

计划：

- [ ] 核对阶段 3 已冻结的开发初始状态和最终测试初始状态，确认两者不重叠且最终测试尚未用于调参。
- [ ] 固定至少 3 个训练随机种子。
- [ ] 保证 `action_only` 和 `future_aware` 使用相同数据、训练预算和测试初始状态。
- [ ] 汇报均值、离散程度、推理延迟和内存占用。
- [ ] 保存两种模型在相同测试状态下的视频。
- [ ] 完成最终测试后不再根据结果调参；如需修改模型，重新定义一轮新的最终测试。

通过条件：比较公平、可复现，并能客观判断未来表征是否改善闭环控制。

阶段完成后的新窗口提示：

```text
请读取 PROGRESS.md 和阶段 5 的最终评测表。不要再调参。现在开始阶段 6：整理复现命令、结果视频、限制和结论。
```

## 阶段 6：项目整理与总结

状态：未开始。

计划：

- [ ] 整理从环境、数据、训练到评测的复现命令。
- [ ] 固定最终配置、checkpoint 和数据划分。
- [ ] 汇总曲线、表格、成功视频和失败视频。
- [ ] 记录限制、失败案例和下一步改进方向。
- [ ] 更新 `README.md`，使新读者可以复现实验。

通过条件：另一台符合依赖要求的机器能够按文档重现实验和关键结论。

## 更新本文件的标准

只有满足以下任一条件时才把任务标记为“已完成”：

- 已运行与风险相称的测试，并记录真实结果；或
- 已生成可检查的实验产物，并人工核对其含义。

代码存在、命令开始运行、损失偶然下降一次、生成了一段视频，都不能单独证明阶段完成。
