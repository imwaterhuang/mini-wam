> 历史快照：保留旧阶段交接记录。所有续接提示均已失效，当前状态以根目录 PROGRESS.md 为准。

# Mini-WAM 详细进度

最后更新：2026-09-10

这份文件是 Mini-WAM 的阶段交接记录。每次开启新窗口时，先读取本文件，再读取当前阶段对应的设计文件和代码；不要仅凭聊天记录判断进度。

## 2026-09-10 项目收尾状态

- 项目已按“单随机种子的探索性对照实验”收尾，不再追加训练。
- 检查点选择规则冻结为最低离线验证损失：`action_only` 为 step 10,000，
  SHA-256 为 `3b739cf3897dde0806b2b93b9404a41cec95975ce26a3483e9c17226747a3506`；
  `future_aware` 为 step 34,000，SHA-256 为
  `5b3ba0bb141564fa664054a514585b849b7f178c0f2cedca7ee03b974dc1a79b`。
- 新增 20 个相同初始状态的配对开发评估：`action_only` 成功 2/20，
  `future_aware` 成功 5/20；McNemar（配对二分类检验）精确双侧
  `p=0.375`，差异不显著。
- 两者中位最终覆盖率分别为 `0.810` 与 `0.337`，最终覆盖率标准差分别为
  `0.399` 与 `0.428`。`future_aware` 严格成功次数更多，但典型结果更低且略不稳定。
- 反事实验证中，正确、打乱和合法反向动作的未来损失分别为 `0.03147`、
  `0.03259`、`0.04411`：反向动作影响明显，打乱动作敏感性较弱。
- 封存最终测试集未使用。当前证据不足以声称 `future_aware` 稳定优于基线。
- Drive 完整证据包：`MyDrive/mini-wam-runs/project-closure-20260910/`；本地结论见
  [项目收尾报告](reports/PROJECT_CLOSURE.md)。

## 新窗口工作规则

1. 先核对本文件、`SPEC.md`、当前代码和验证产物，确认记录没有落后于工作区。
2. 已通过验证的阶段不重复实现；如环境或依赖发生变化，只做必要的复查。
3. 每个阶段结束前必须满足该阶段的通过条件，并把验证结果和下一步写回本文件。
4. 模型代码由学习者本人编写。助手负责解释、拆分任务、审查代码和运行验证，不直接代写模型实现。
5. 最终测试状态只用于最后评价，不用于调参；开发过程使用训练集和验证集。
6. 任何超过 1000 step 的云端训练前，必须逐项核对“模型实际使用字段”和“数据集实际读取、解码、拼接字段”；禁止为当前模型读取未进入 forward 或 loss 的图片。
7. 必须在目标 runtime 对 `num_workers=0/2/4` 做同批次短基准，分别记录数据准备、Host-to-Device（主机到设备）传输、GPU 计算和端到端耗时，再选择最快稳定值；不能沿用未经测量的默认值。
8. 任何数据路径或 worker 数量变更，必须先验证实际使用张量逐项完全一致，并验证中断恢复后的样本顺序、loss、模型、optimizer、scheduler 和随机状态一致，之后才允许继续正式训练。

## 2026-09-10 云端训练性能修复

- `future_aware` 重复逐帧读取、解码和预处理造成处理器瓶颈；已新增可选归一化图像缓存，模型代码与监督内容未改。
- 目标 T4 环境短基准：原路径 653.18 毫秒/步；缓存 workers=0/2/4 分别 419.03/342.27/311.47 毫秒/步，选择 4，约加速 2.10 倍。
- 1,378 个训练窗口全部字段与原路径完全一致，并检查验证集 episode 边界。确定性卷积诊断下，0/2/4 workers 的完整恢复状态逐项完全一致。本机 49 passed、4 skipped；云端 21 passed。
- 原训练保存到 step 2000，完整检查点另存 Drive 并核验后替换训练进程；原运行目录 `future-aware-seed0-w4` 从 step 2000 续跑至目标 50000。已观察 step 2038 → 2083 用时 15.01 秒（0.3337 秒/步），损失有限，单个主训练进程。该操作不代表阶段 4 训练或闭环验收完成。
- Drive `mini-wam-runs/future-aware-seed0-w4/performance/cache-migration/` 保存检查点备份、补丁压缩包、性能报告和校验清单。Colab 重建时先恢复补丁并重建本地缓存，再按 `COLAB.md` 第 8 节恢复最新检查点。
- 详见 [本次性能报告](reports/performance/2026-09-10-colab-future-aware-diagnosis.md)。

## 总体进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 本地环境、Push-T 运行与画面查看 | 已完成 |
| 1 | 数据审计、划分、窗口采样与归一化 | 已完成 |
| 2 | `action_only` 模型与 64 样本过拟合 | 已完成 |
| 3 | `action_only` 闭环训练与基线评测 | 已完成 |
| 4 | `future_aware` 模型、单种子训练与反事实检验 | 探索性完成，严格门禁未完全满足 |
| 5 | 多随机种子正式评测 | 未执行，封存最终测试未使用 |
| 6 | 项目整理、复现实验与总结 | 已完成（探索性结论） |

## 当前交接点

- 当前阶段：项目已收尾；如未来重启，应从多训练随机种子正式评测重新开始，
  不能把本轮开发场景结果当成封存最终测试结果。
- 阶段 2 已通过：64 样本损失从 `0.423381` 降至 `0.006625`，下降约 98.4%；checkpoint 重新加载输出一致。
- 可恢复训练基础设施已完成：正式配置、smoke test 配置、训练/验证指标、完整 checkpoint 和确定性数据采样恢复。
- Colab 前置流程已补齐：专用依赖、数据下载与校验、CUDA/数据/Drive 写入预检、本地高速训练加 Drive 增量镜像、断线恢复命令。
- `action_only` 正式闭环基线已完成：第 10,000 步 `best.pt` 在 20 个冻结开发场景中达到 2/20 严格成功（10%）；随机策略为 0/20。正式 checkpoint 与视频位于 Colab/Drive，未镜像到本机。
- 前置实现已完成本地验收：`34 passed`；数据下载脚本在已有数据上确认 206 回合、25,650 帧；训练从本地第 2 步镜像到持久化目录后，在全新本地目录精确恢复到第 4 步；临时 checkpoint 已跑通 20 个开发场景、20 个视频和汇总结果的短评估。
- 已实测从第 50 步恢复到第 100 步；恢复训练与不中断训练的最终模型参数逐张量完全一致，学习率调度器和采样器状态也一致。
- 已冻结互不重叠的 20 个开发评估场景和 50 个最终测试场景；最终测试文件已标记为阶段 5 前禁止评测。
- 20 回合随机策略基线已完成：成功率 `0%`，平均最终覆盖率 `0`，平均最大覆盖率 `0.120112`，平均回报 `1.223907`。
- 未阻塞警告：macOS 运行模拟器时，OpenCV 与 Pygame 各自携带的 SDL（Simple DirectMedia Layer，跨平台多媒体库）会报告重复动态类；本轮 20 回合和全部测试均未崩溃，若后续出现图形相关异常再隔离依赖处理。
- `action_only_full.pt` 是早期 1000 步诊断模型，不视为正式阶段 3 模型。训练耗时诊断确认：旧数据路径会为每个 `action_only` 样本额外解码并拼接 4 张未使用的未来图像。
- 已为 `MiniWAMDataset` 增加保持默认完整契约的 `include_future_observations` 模式；`action_only` 训练和验证固定关闭未来图像输出，不改变历史图像、位置、动作、mask、样本索引或配置文件。多批同索引检查和单元测试要求所有实际使用字段逐张量相等；当前完整回归为 `35 passed`。
- 本机 warm-cache（缓存已预热）微基准在预热 5 批后交错测量 50 批：batch size 128 的旧完整取数平均 `133.72 ms/batch`，跳过未来图像后平均 `46.69 ms/batch`，数据阶段快 `2.86x`；55 批共 7040 个样本的全部实际使用字段逐张量完全一致。这不是 A100 最终结论；`scripts/profile_action_only.py` 用于在同一 Colab runtime 中分别实测取数、主机到设备传输、计算和端到端耗时。
- sampler 已升级为“预取游标与已完成游标分离”：worker 请求 batch 不再推进 checkpoint 位置，只有 `optimizer.step()` 和 scheduler（学习率调度器）更新成功后才调用 `mark_consumed()`。新 sampler 状态带版本号，同时兼容旧版 `num_workers=0` checkpoint；worker seed 使用独立 generator，不推进模型训练的全局 PyTorch 随机状态。
- 已在允许 PyTorch 共享内存的本机进程环境中验证 `num_workers=2/4`：预取不会改变 checkpoint 的下一批位置；含 Dropout（随机失活层）的 toy 训练在不中断与第 4 步中断恢复到第 9 步之间，样本顺序、每步 loss、模型、optimizer、scheduler 和最终 PyTorch 随机状态全部完全一致。
- 真实 Mini-WAM 精简数据路径本机测量 20 批：`num_workers=0/2/4` 分别约为 `46.69/23.22/12.18 ms/batch`；2/4 workers 下完整与精简数据路径的实际使用字段仍完全一致。多进程结果依赖 Colab CPU 配额，A100 runtime 必须重新测量后再选 2 或 4。
- `future_aware` 模型、联合损失、训练/验证步骤和可恢复训练入口已实现；本地真实数据 2 步短跑通过，连续训练与第 1 步中断后恢复在模型参数、sampler、scheduler 和验证损失上完全一致；当前完整回归为 `46 passed, 4 skipped`。
- 阶段 4 尚未完成：正式训练前仍须审计冻结目标特征能否区分 Push-T 的位置与姿态变化，并完成 64 样本过拟合、动作反事实检查和开发场景闭环评测。`configs/future_aware_seed0.yaml` 只是候选正式配置，尚未通过目标 runtime 的吞吐与显存门槛，不得直接启动长训练。

当前新窗口续接提示：

```text
请先读取 PROGRESS.md、SPEC.md、`action_only` 的训练配置和闭环评测产物。阶段 3 已完成，现在只开始阶段 4：`future_aware`。先审计未来帧、动作前缀和掩码的时间对齐，再审计冻结目标特征对 Push-T 状态变化的可辨性；之后让我自己实现一个组件并用一个测试验证。不要运行最终测试场景。
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

状态：已完成。

计划：

- [x] 建立可恢复的训练脚本和配置。
- [x] 在完整训练开始前分别冻结开发初始状态和最终测试初始状态；封存最终测试状态，不在开发阶段运行。
- [x] 在完整训练集上训练 `action_only`。
- [x] 在相同初始状态上比较学习策略与随机策略。
- [x] 记录 reward、成功率、轨迹长度和推理耗时。
- [x] 导出成功与失败视频，直接观察模型表现。

当前验证证据：

- `configs/action_only_smoke.yaml`：100 步本地恢复测试配置。
- `configs/action_only_seed0.yaml`：50,000 步正式种子 0 配置；已用于阶段 3 正式训练。
- `scripts/train_action_only.py`：训练与恢复入口。
- `requirements-colab.txt`、`COLAB.md`：Colab 安装、训练、恢复和评估流程。
- `scripts/download_dataset.py`、`scripts/check_training_ready.py`：数据获取与正式训练前置检查。
- `scripts/evaluate_action_only.py`：正式 checkpoint 的开发集批量闭环评估；拒绝最终测试集。
- `src/mini_wam/training/action_only.py`：验证、指标、checkpoint 和确定性采样实现。
- `scripts/profile_action_only.py`：在不保存训练状态的独立进程中对比旧完整数据路径与 `action_only` 精简路径，并拆分数据、传输、计算和端到端耗时。
- `tests/test_training.py`：配置与精确恢复测试。
- `scripts/freeze_evaluation_scenes.py`：生成并保护 20/50 场景划分。
- `scripts/evaluate_random_policy.py`：只允许在开发场景上运行的随机策略评测。
- `splits/evaluation_scenes_manifest.json`：场景数量、哈希和最终测试封存状态。
- `reports/baselines/random_policy_development.json`：20 回合随机基线明细和汇总。
- `runs/action_only/0/resume-smoke-20260904-v2/`：50 步中断并恢复到 100 步的真实产物。
- `runs/action_only/0/SMOKE_VERIFICATION.md`：中断恢复、不中断对照及清理记录。
- 两条路径最终模型参数、学习率调度器、采样器状态及验证损失完全一致；为避免重复占用约 1.5 GB，只保留一个可执行的最终 smoke checkpoint。
- 正式训练的最佳检查点为第 10,000 步 `best.pt`，验证损失为 `0.06195011886979604`。该检查点与正式评测视频保存在 Colab/Drive 产物中，未镜像进当前本地工作区。
- 在冻结的 20 个开发场景中，以 `coverage > 0.95`（覆盖率大于 0.95）为严格成功标准：学习策略成功 2/20（10%），随机策略成功 0/20；平均最终/最大覆盖率为 `0.481638/0.550629`，平均 reward（回报）为 `99.947373`，平均推理耗时为 `3.169 ms`。
- 每个回合最多 300 个环境步；策略每次预测 16 步动作，实际执行 4 步后重新规划。`dev-001` 与 `dev-017` 为严格成功回合，已保留成功和失败视频以供人工复核。

通过条件：已满足。模型可在 Push-T 中闭环控制；开发集评测使用冻结场景并可复现；代表性的成功和失败视频已保留。该结果是 `action_only` 基线，不代表未来感知模块已有效，也不使用最终测试场景。

阶段完成后的新窗口提示：

```text
请读取 PROGRESS.md、SPEC.md、action_only 的训练配置和闭环评测产物。阶段 3 已完成，现在开始阶段 4：future_aware。先审计未来监督的帧/动作时间对齐和目标特征是否表达 Push-T 状态变化；再说明它相对 action_only 新增的输入、输出和损失，最后让我逐个组件自己实现。不要直接代写模型代码。
```

## 阶段 4：`future_aware` 模型

状态：探索性完成；严格通过条件未完全满足。

计划：

- [x] 冻结未来表征的目标编码器，并保持 eval 模式。
- [x] 编码前四步动作前缀并预测未来表征。
- [x] 在动作损失之外加入 masked future loss（带掩码的未来损失），并接入可恢复训练入口。
- [x] 比较正确动作、打乱动作和合法反向动作条件下的未来预测。
- [x] 检查未来分支的动作依赖；反向动作影响明显，但打乱动作差异较弱，证据有限。
- [x] 在与 `action_only` 相同的 20 个额外开发初始状态上做闭环比较。

通过条件未完全满足：反向动作使未来误差增加，但打乱动作只使损失从
`0.03147` 增至 `0.03259`，不足以强力排除“弱依赖动作”的退化解释。该限制已写入
最终结论。

阶段完成后的新窗口提示：

```text
请读取 PROGRESS.md 和 future_aware 的反事实检验结果。阶段 4 已完成，现在开始阶段 5：正式多随机种子评测。不得用最终测试状态继续调参。
```

## 阶段 5：正式多随机种子评测

状态：按收尾范围不执行。只完成单随机种子探索性比较，封存最终测试未使用。

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

状态：已完成探索性整理；未验证跨机器从头复现。

计划：

- [x] 整理从环境、数据、训练到评测的复现命令。
- [x] 固定最终配置、checkpoint 和数据划分。
- [x] 汇总曲线、表格、成功视频和失败视频。
- [x] 记录限制、失败案例和下一步改进方向。
- [x] 更新 `README.md`，说明实验结果与结论边界。

严格通过条件尚未验证：未在第二台机器从头重现实验。当前已完成产物校验与复现文档
整理，不把“文档存在”等同于“跨机器复现通过”。

## 更新本文件的标准

只有满足以下任一条件时才把任务标记为“已完成”：

- 已运行与风险相称的测试，并记录真实结果；或
- 已生成可检查的实验产物，并人工核对其含义。

代码存在、命令开始运行、损失偶然下降一次、生成了一段视频，都不能单独证明阶段完成。
