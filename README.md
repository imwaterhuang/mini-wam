# Mini-WAM：从数据管线到闭环评测的 Push-T 机器人策略项目

Mini-WAM 是一个面向机器人算法与具身智能岗位的 Push-T 策略项目。当前阶段专注 ACT（Action Chunking with Transformers，基于 Transformer 的动作分块），目标是改善实际闭环控制表现。

## 当前阶段：ACT 动作策略（2026-09-14）

全部 **206 个示范回合、25,650 帧**作为训练数据来源，**不留出示范验证回合**。验证在模拟器中新随机生成的合法场景上执行，并保存随机种子和初始状态。
ACT 使用独立的全量回合清单与归一化统计，按随机开发场景成功率选择检查点，再在另一批新随机场景上复核。

实施顺序：**模型与数据接口 → 64 窗口过拟合 → 短训练/恢复 → 单种子训练 → 随机场景闭环评测**。
目前阶段文档已更新，ACT 代码及运行入口待实现，训练尚未启动。详见[当前进度](PROGRESS.md)、[ACT 规格](SPEC.md)和[验收协议](EVALUATION.md)。

本轮不加入 WAM（World Action Model，世界动作模型）或 Diffusion Policy（扩散策略）目标。项目名称沿用 Mini-WAM，下面保留的未来辅助监督实验仅为历史证据。

## 历史项目亮点（2026-09-10）

- **完整机器人学习闭环**：206 个示范回合、25,650 帧，覆盖数据划分、窗口构造、归一化、训练、检查点选择、闭环执行与统计分析。
- **可恢复训练系统**：保存并恢复模型、优化器、学习率调度器、随机数状态和已消费采样位置；避免多进程预取提前推进持久游标。
- **两种受控模型对比**：共享数据、视觉编码器、动作预测结构和评测器，仅改变训练期未来辅助监督。
- **冻结场景配对评测**：两模型在相同 20 个开发初始状态上运行，预测 16 步、执行 4 步后重新规划，严格成功标准为覆盖率 `> 0.95`。
- **可审计证据**：逐回合结果、训练日志、环境元数据、反事实实验、统计图和 6 段代表视频均保存在仓库中；证据可在不下载私人权重的情况下重算。

## 历史已解决的工程问题

| 模块 | 实现与验证 |
|---|---|
| 数据正确性 | 按完整回合划分训练/验证集；归一化仅使用训练集；窗口不跨回合；掩码排除动作与未来帧填充。 |
| 策略学习 | 基于两帧历史观测预测 16 步二维动作块，并以“执行 4 步后重规划”的方式闭环控制。 |
| 未来辅助监督 | 使用真实前 4 步动作预测 4 个未来表示；冻结的 ResNet-18 生成监督目标；推理时移除辅助分支。 |
| 训练可靠性 | 支持完整状态恢复、确定性采样与预取安全；检查点按预先定义的离线验证损失选择。 |
| 数据吞吐 | 为重叠窗口建立可校验的图像缓存；目标环境中缓存与 4 worker 配置相对原始路径达到 `2.097×` 端到端加速。 |
| 评测与诊断 | 配对闭环评测、McNemar 精确检验、正确/打乱/反向动作反事实检查，以及成功/失败视频对照。 |

## 历史模型与数据流

历史两帧图像与位置经过共享视觉编码器和时序模块得到历史表示，再由动作头输出 `[B, 16, 2]` 动作块。
`future_aware` 仅在训练时增加一条辅助路径：真实动作前缀 `[B, 4, 2]` 与历史表示共同预测未来特征 `[B, 4, 512]`，总损失为：

```text
L_total = L_action + 0.1 × L_future
```

其中 `L_future` 是带有效步掩码的余弦距离。固定正弦—余弦位置编码表示动作前缀中的时间位置。

![两种已实现的策略结构](reports/architecture/paper_style_architectures.png)

[可编辑架构图](reports/architecture/paper_style_architectures.svg) ·
[结构与梯度说明](reports/architecture/model_architectures.md) ·
[数据窗口可视化](reports/architecture/data_samples.png)

## 历史实验结果

两个模型均训练到 50,000 步，并按各自最低离线验证损失选择检查点。下表来自额外冻结的 20 个相同开发场景，每回合最多 300 个环境步：

| 指标 | `action_only` | `future_aware` |
|---|---:|---:|
| 严格成功率 | 2/20（10%） | 5/20（25%） |
| 平均最终覆盖率 | 0.556 | 0.478 |
| 中位最终覆盖率 | 0.810 | 0.337 |
| 最终覆盖率标准差 | 0.399 | 0.428 |
| 选中检查点步数 | 10,000 | 34,000 |

配对结果为：共同成功 1 个、仅基线成功 1 个、仅未来辅助模型成功 4 个、共同失败 14 个。McNemar 精确双侧检验 `p=0.375`，因此当前单随机种子实验**不能证明**未来辅助监督带来稳定提升。

![配对成功率与覆盖率](reports/closure/figures/paired_comparison.png)

在 2,368 个验证窗口、9,352 个有效未来步上，未来分支的余弦距离为：正确动作 `0.03147`、批内打乱动作 `0.03259`、合法反向动作 `0.04411`。反向动作会明显增大误差，但打乱动作影响较弱，说明动作条件化证据仍有限。

## 历史闭环视频

| 相同初始场景 | `action_only` | `future_aware` |
|---|---|---|
| `web-random-04`：共同成功 | [成功视频](reports/closure/representative_videos/shared_success__action_only__web-random-04.mp4) | [成功视频](reports/closure/representative_videos/shared_success__future_aware__web-random-04.mp4) |
| `web-random-09`：仅基线成功 | [成功视频](reports/closure/representative_videos/action_only_only_success__action_only__web-random-09.mp4) | [失败视频](reports/closure/representative_videos/action_only_only_success__future_aware__web-random-09.mp4) |
| `web-random-00`：仅未来辅助模型成功 | [失败视频](reports/closure/representative_videos/future_aware_only_success__action_only__web-random-00.mp4) | [成功视频](reports/closure/representative_videos/future_aware_only_success__future_aware__web-random-00.mp4) |

六段视频均完成解码检查，帧数与对应回合执行步数一致。更多观察见[失败分析](reports/failure_analysis.md)。

## 历史证据快速验证

Python 3.10 或以上可直接核验归档文件并重算统计，无需下载模型权重或数据：

```bash
python3 scripts/summarize_closure.py --output-dir outputs/closure-summary
```

预期成功标志：

```text
CLOSURE_AUDIT_OK files=14 scene_pairs=20 p=0.375
```

已有本地环境也可运行完整测试：

```bash
./.venv/bin/python -m pytest -q
```

2026-09-10 交付时回归记录为 57 项常规测试通过，另有 4 项默认跳过的多进程测试单独验证通过，共 61 项。这些记录不代表 ACT 已实现或当前代码已通过重新验证。

## 代码与证据入口

- [环境与数据安装](docs/GETTING_STARTED.md)
- [ACT 当前进度](PROGRESS.md)
- [ACT 云端准备与历史训练命令](COLAB.md)
- [历史评测和复现命令](docs/REPRODUCING.md)
- [完整实验报告](reports/PROJECT_CLOSURE.md)
- [逐回合配对结果](reports/closure/paired_comparison.csv)
- [归档与校验记录](reports/DELIVERY_VERIFICATION.md)
- [项目规格](SPEC.md)与[评测边界](EVALUATION.md)

## 历史结论与当前比较边界

新 ACT 将使用全部 206 个回合，旧模型只使用 185 个训练回合，并按旧离线验证协议选点。因此，新旧成绩不能直接解释为仅更换架构的效果；ACT 结果尚待训练和随机场景评测。

这是单随机种子的探索性开发实验，而不是多随机种子正式结论。封存最终测试场景未用于调参或本轮报告；尚未完成跨机器从头重训、亮度扰动、无动作辅助分支消融、完整逐步轨迹记录和真实机械臂验证。

因此，本项目能证明的是：**机器人策略的数据—训练—恢复—闭环评测—失败分析工程链路已打通，并形成可审计证据；它不能证明未来辅助监督在 Push-T 上稳定优于动作基线。**
