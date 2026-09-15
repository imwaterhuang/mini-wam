# Mini-WAM 工程规格

> 2026-09-14 ACT 阶段开始前的历史快照；下文状态和计划仅适用于旧实验，不作为当前续接指令。
> 当前状态见 [PROGRESS.md](../../PROGRESS.md)，当前规格见 [SPEC.md](../../SPEC.md)。

> 状态：单随机种子探索性实验已收尾；多随机种子正式结论未执行
> 定位：面向机器人数据、策略训练与闭环评测实习的作品项目  
> 环境：Push-T  
> 本文中的 WAM 指 World Action Model（世界动作模型）。

## 当前交付范围与原计划的关系

本次交付为单随机种子的探索性工程项目，不是下文 V0.2 正式实验版。
当前范围以 [PROGRESS.md](PROGRESS_before_act_20260914.md)、[收尾报告](../../reports/PROJECT_CLOSURE.md)
及 [复现说明](../../docs/REPRODUCING.md) 为准。下文三种子、50 个最终测试场景、
亮度扰动、小型模拟器反事实等条款保留为原计划；未执行条款不因收尾视为通过。
不再自动启动长训练或封存测试。实际入口是 `scripts/train_action_only.py`、
`scripts/train_future_aware.py`、`scripts/evaluate_paired_closure.py`、
`scripts/evaluate_counterfactual.py` 与 `scripts/summarize_closure.py`。
第 13 节通用文件树/命令仅是原设计示意，不是已存在入口的清单。

## 1. 项目目标

本项目首先建立一套完整、可复现的机器人学习工程流水线，其次研究一个小问题：

> 在相同轨迹数据、视觉编码器、动作头、训练更新次数和闭环评测器下，给动作分块策略增加“动作条件未来表示预测”辅助任务，是否改善 Push-T 的闭环控制表现？

项目必须展示：

1. 机器人轨迹的检查、划分、时间窗口采样、归一化和 padding（填充）处理。
2. 动作分块 Behavior Cloning（行为克隆）策略的训练、保存、恢复和推理。
3. 固定初始状态下的闭环控制、量化评测和失败视频分析。
4. 动作条件未来预测，以及正确、错误动作的对照验证。
5. 配置驱动、结果可追溯、他人可复现的训练与评测流程。

### 1.1 结论边界

`future_aware` 在训练时使用真实示范动作预测未来表示，推理时只运行动作路径。因此本项目检验的是：

> 动作条件未来监督能否改善直接动作策略的训练或表征。

它不检验“机器人是否在推理时先想象未来、再根据未来规划动作”。

### 1.2 非目标

第一版不做：

- 完整复现 ACT（Action Chunking with Transformers，基于 Transformer 的动作分块）。
- 完整复现 Fast-WAM 或其他大型视频模型。
- 像素级未来视频生成和长时递归生成。
- 推理期未来想象、搜索或 Model Predictive Control（模型预测控制）。
- Diffusion Policy（扩散策略）、VLA（Vision-Language-Action，视觉-语言-动作）或多任务策略。
- 真实机械臂采集、ROS（Robot Operating System，机器人操作系统）部署和硬件控制。
- 论文级因果结论或向其他机器人任务的直接外推。

以上内容不得阻塞核心版本完成。

## 2. 分阶段交付

### 2.1 V0.1：求职可投版

满足以下条件即可开始投递实习：

- 数据审计、episode（回合）划分和时间对齐测试通过。
- `action_only` 能在固定 64 个训练窗口上明显过拟合。
- `action_only` 能完成训练、保存、恢复和闭环推理。
- 闭环结果明显优于随机动作，并保存成功和失败视频。
- README 提供安装、训练和评测命令。

### 2.2 V0.2：完整 Mini-WAM 版

- `action_only` 与 `future_aware` 各完成 3 个正式训练种子。
- 两个模型使用相同的测试初始状态进行闭环比较。
- 未来分支完成正确、打乱和反向动作检查。
- 完成一种固定的观测扰动评测。
- 输出结果表、训练曲线、模型图、视频和失败分析。

### 2.3 选修版

核心完成后才允许增加：

- `future_no_action`：相同未来目标，但动作输入替换为空动作。
- 更多未来损失权重。
- 初始物体位置或角度的分布变化。
- Joint-WAM：推理期让动作头读取未来表示。
- 像素解码器或更长时域预测。

## 3. 可复现环境

首次初始化时必须记录：

- Python、PyTorch、LeRobot 和环境依赖版本。
- LeRobot 的固定提交哈希；正式实验不得只依赖持续变化的分支名。
- 操作系统、计算设备和精度模式。
- 数据集仓库、版本和许可信息。
- Push-T 的观测键、动作范围、控制频率和最大回合步数。

每次运行把这些信息保存为 `environment.json`，根目录维护依赖锁定文件。项目训练并保存自己的 checkpoint（训练检查点），不把旧在线 checkpoint 作为正式基线。

## 4. 数据契约

### 4.1 时间语义

对采样起点 `t`：

```text
历史输入：         o_{t-1}, o_t
状态输入：         p_{t-1}, p_t
动作监督：         a_t, ..., a_{t+15}
未来监督：         o_{t+1}, ..., o_{t+4}
未来分支动作输入： a_t, ..., a_{t+3}
```

其中：

- `p_t` 是 Push-T 智能体二维位置。
- `a_t` 必须对应 `o_t -> o_{t+1}`。
- 未来分支禁止读取 `a_{t+4}` 到 `a_{t+15}`，避免利用超出四步预测范围的信息。
- 样本至少包含有效的 `o_t`、`a_t` 和 `o_{t+1}`，且不得跨越 episode 边界。
- `t=0` 不生成样本，避免为历史输入再引入一套填充规则。

### 4.2 样本接口

单个样本：

```python
{
    "observation_history": float32[2, 3, 96, 96],
    "agent_position":      float32[2, 2],
    "action_chunk":        float32[16, 2],
    "action_valid_mask":   bool[16],
    "future_observations": float32[4, 3, 96, 96],
    "future_valid_mask":   bool[4],
    "episode_id":          int64[],
    "start_step":          int64[],
}
```

DataLoader（数据加载器）组成 batch（批次）后，所有字段前增加批次维度 `B`。

`future_observations` 与 `future_valid_mask` 是 `future_aware` 的训练监督。`action_only`
训练和验证允许使用同一数据集的精简视图并省略这两个字段，但其余字段、窗口索引、
归一化结果和 sampler（采样器）顺序必须与完整视图逐张量一致。该优化不能改变模型输入
或动作监督，只能跳过不会进入 `L_action` 的未来图像读取、解码、归一化与 batch 拼接。

Push-T 的目标区域固定在环境中，第一版不增加独立 `goal` 字段。若以后改造成可变目标任务，必须作为新实验版本处理。

### 4.3 尾部填充

当 episode 尾部不足 16 个动作或 4 个未来观测时：

- 缺失动作在归一化空间填 `0`，对应 `action_valid_mask=False`。
- 缺失未来图像重复最后一个有效观测，对应 `future_valid_mask=False`。
- 两种填充位置都不得参与损失。
- 未来分支只使用 `action_chunk[:4]`，并用 `future_valid_mask` 屏蔽无效位置。

### 4.4 预处理与归一化

- 原始图像保持宽高比缩放到 `96 x 96`，转为 `float32`，像素范围为 `[0, 1]`。
- 图像使用 ImageNet-1K 的均值和标准差，以匹配预训练视觉编码器。
- 训练增强只允许轻微平移和亮度变化；两个模型必须完全一致。
- 验证、在线评估和反事实评估不得使用随机增强。
- 智能体位置和动作的均值、标准差只从训练 episode 计算。
- 数值标准差下限为 `1e-6`。
- 策略输出先反归一化，再裁剪到环境合法范围。
- 统计量保存为 `normalization.json`，不得与 checkpoint 分离。

### 4.5 数据与环境划分

- 示范数据默认按完整 episode 做 `90% / 10%` 的训练、验证划分。
- 数据划分种子固定为 `42`；验证集原则上不少于 10 个完整 episode。
- 禁止逐帧随机划分，避免相邻帧泄漏到训练集和验证集。
- 数据划分一旦生成即冻结，保存到 `splits/*.json`。
- 验证 episode 用于离线损失、配置选择和 checkpoint 辅助选择，不另设示范数据测试集。
- 闭环环境初始状态分为开发评估集与最终测试集；两者均从环境随机生成后冻结，且彼此不重叠。
- 原正式实验计划以封存测试指标为主；本轮仅报告额外开发场景的探索性结论，不作为最终测试结论。

### 4.6 数据审计产物

`scripts/audit_dataset.py` 必须输出：

- episode 数、总帧数和回合长度分布。
- 图像尺寸、动作范围和智能体位置范围。
- 训练集归一化统计量。
- 16 个“历史观测—动作前缀—未来观测”样本图。
- 划分摘要、重复 ID 和跨 episode 检查结果。
- `artifacts/data_audit.json`。

## 5. 模型规格

第一版使用确定性动作分块 Behavior Cloning（行为克隆），不使用变分编码器或扩散生成。

### 5.1 共享动作路径

同一训练种子下，`action_only` 和 `future_aware` 的共享组件必须从相同权重初始化。

#### VisualEncoder

- 使用 ImageNet-1K 预训练的 ResNet-18。
- 删除最终分类层，每帧输出 512 维特征。
- 两个历史帧共享权重，正式训练时允许更新。

```text
[B, 2, 3, 96, 96] -> [B, 2, 512]
```

#### StateEncoder

- 将两步二维智能体位置展平为 4 维。
- 使用两层 MLP（Multilayer Perceptron，多层感知机）编码为 64 维。

#### HistoryFusion

- 将两帧视觉特征展平为 1024 维，与 64 维状态特征拼接。
- 使用两层 MLP 投影为共享历史表示 `h_t [B, 256]`。

#### ActionHead

- 两层 MLP：`256 -> 256 -> 32`。
- 输出重排为 `[B, 16, 2]`。
- 动作损失使用 Smooth L1 Loss（平滑 L1 损失）。
- 只在 `action_valid_mask=True` 的位置计算损失。

### 5.2 `action_only`

```text
历史图像 + 智能体位置 -> 共享动作路径 -> 16 步动作块
```

```text
L_total = L_action
```

### 5.3 `future_aware`

`future_aware` 保留相同动作路径，并在训练时增加以下模块。

#### FrozenTargetEncoder

- 使用独立的 ImageNet-1K 预训练 ResNet-18。
- 删除分类层，每个未来帧输出 512 维目标表示。
- 训练全程冻结，不接收梯度。
- 四帧输出为 `[B, 4, 512]`，最后一维做 L2 normalization（L2 归一化）。

#### ActionPrefixEncoder

- 只读取 `action_chunk[:, :4]`。
- 每步二维动作经过 Linear(2,64)、ReLU，加固定时间编码，再经过 Linear(64,64)、ReLU。
- 加入固定的 sine-cosine positional encoding（正弦余弦位置编码），展平后得到 `[B, 256]`。

#### FutureHead

- 拼接 `h_t [B, 256]` 与动作前缀表示 `[B, 256]`。
- 使用 Linear(512,1024)、ReLU、Linear(1024,2048)、ReLU，重排为 `[B, 4, 512]` 后沿最后一维做 L2 归一化。
- 未来损失使用逐步 cosine distance（余弦距离）。
- 只在 `future_valid_mask=True` 的位置取平均。

```text
L_total = L_action + lambda_future * L_future
lambda_future = 0.1
```

`lambda_future=0.1` 是核心版默认值。只能在正式三种子实验前用训练集和验证集完成一次试运行后修改，并把原因写入 `reports/decision_log.md`。

### 5.4 训练与推理路径

训练：

```text
历史观测 -> 共享历史表示 -> 动作头 -> 动作损失
                    |
真实前 4 步动作 ----+------> 未来头 -> 未来损失
未来 4 帧 -> 冻结目标编码器 -----------+
```

推理：

```text
历史观测 -> 共享历史表示 -> 动作头 -> 16 步动作块
```

推理时不得调用目标编码器、动作前缀编码器或未来头。

## 6. 训练协议

### 6.1 默认配置

```yaml
history_length: 2
action_horizon: 16
future_horizon: 4
n_action_steps: 4
image_size: [96, 96]
hidden_dim: 256
batch_size: 128
train_steps: 50000
learning_rate: 0.0003
weight_decay: 0.0001
warmup_steps: 1000
gradient_clip_norm: 1.0
validation_frequency: 1000
checkpoint_frequency: 5000
formal_seeds: [0, 1, 2]
```

若硬件无法支持默认批大小，可以减小批大小并使用梯度累积，但两个模型的有效批大小必须相同。

优化器使用 AdamW（Adaptive Moment Estimation with Decoupled Weight Decay，解耦权重衰减自适应矩估计），预热后使用余弦学习率衰减。

### 6.2 试运行与配置冻结

1. 只用种子 `0` 做数据、模型和损失调试。
2. 只使用训练集和验证集确定批大小、训练步数与未来损失权重。
3. 把选择结果与理由写入 `reports/decision_log.md`。
4. 冻结配置后再运行 `[0, 1, 2]` 三个正式种子。
5. 不得根据最终闭环测试结果单独修改某个模型。

### 6.3 checkpoint 与恢复

每个 checkpoint 至少包含：

- 模型、优化器和学习率调度器状态。
- 当前训练步数和随机数状态。
- sampler（采样器）的下一个未完成 batch 位置；worker 预取不得推进该持久位置。
- 完整训练配置。
- 归一化统计量引用。
- 数据划分哈希和源码提交哈希。

sampler 位置只能在 `optimizer.step()` 和学习率更新成功后推进。`num_workers>0` 时，
worker 可以提前生成 batch，但 checkpoint 不得把“已预取”误记为“已完成训练”。
DataLoader（数据加载器）使用独立随机数生成器，避免重建 worker 改变模型使用的全局
PyTorch 随机状态。如果未来加入随机数据增强，还必须让增强由样本索引和训练 step
确定性生成。

恢复后，相同输入的输出必须在数值容差内一致，并能继续累加训练步数。正式启用新的
worker 数前，必须验证不中断训练与中断恢复的样本顺序、模型、优化器、学习率调度器和
loss（损失）完全一致。

### 6.4 模型选择

- 每个模型以训练期间最低离线验证损失对应的 `best.pt` 作为唯一选中 checkpoint。
  `action_only` 使用动作验证损失；`future_aware` 使用其预先定义的联合验证损失。
- 开发闭环评估只用于报告控制表现和失败分析，不再反向参与 checkpoint 选择。
- 最终闭环测试只允许使用上述规则选中的单个 checkpoint；本次单随机种子探索性
  收尾没有运行封存最终测试。

### 6.5 运行目录

```text
runs/<model>/<seed>/<timestamp>/
├── config.yaml
├── environment.json
├── normalization.json
├── train_metrics.csv
├── val_metrics.csv
├── checkpoints/
├── curves/
└── run.log
```

异常退出、数值错误和被排除运行必须保留记录，不得静默删除。

## 7. 正确性测试与门禁

### Gate 0：环境与评测器

- 环境能 reset（重置）、step（执行一步）和渲染视频。
- 运行 20 个固定种子的随机动作回合。
- 输出随机策略覆盖率、成功率和平均回报。

### Gate 1：数据正确性

必须通过：

- 形状和数据类型测试。
- episode 边界与固定划分测试。
- `a_t` 对应 `o_t -> o_{t+1}` 的时间对齐测试。
- `action_chunk[:4]` 与四步未来目标的对齐测试。
- 归一化与反归一化往返测试。
- 两种有效掩码不计入填充位置损失的测试。
- 16 个样本的人工可视化检查。

Gate 1 未通过时禁止正式训练。

### Gate 2：64 样本过拟合

关闭数据增强：

- `action_only` 动作损失相对初始值下降至少 90%。
- `future_aware` 的动作损失和未来损失均持续下降。
- 输出随输入变化，不得退化为固定动作块。
- 保存和恢复 checkpoint 后输出一致。

无法过拟合时优先检查时间对齐、掩码、归一化和梯度，不扩大模型。

### Gate 3：动作基线

- 完成一个种子的 `action_only` 训练。
- 闭环覆盖率和成功率明显优于随机动作。
- 能执行“预测 16 步、执行 4 步、重新观测”的滚动时域控制。
- 保存至少一个成功和一个失败案例。

Gate 3 通过后，V0.1 完成并开始投递实习。

### Gate 4：未来分支

- 冻结目标编码器没有梯度。
- 未来损失能更新共享历史表示和未来头，但不更新动作头。
- 推理时关闭未来模块后动作输出不变。
- 打乱动作使未来预测误差上升，或使预测产生可测变化。

### Gate 5：正式实验

- 冻结数据、配置、模型选择规则和评测初始状态。
- 完成两个核心模型各三个种子。
- 汇总所有正式运行，包括失败与异常运行。

## 8. 核心实验矩阵

| ID | 模型 | 正式种子 | 用途 | 阻塞完成 |
|---|---|---:|---|:---:|
| A0 | `action_only` | 0, 1, 2 | 动作策略基线 | 是 |
| F1 | `future_aware` | 0, 1, 2 | 未来辅助监督 | 是 |
| CF | F1 反事实评估 | 无需重训 | 检查动作敏感性 | 是 |
| VIS-SHIFT | A0 与 F1 | 沿用已有模型 | 固定亮度扰动 | 是 |
| F-NULL | `future_no_action` | 选修 | 分离未来监督与动作条件 | 否 |

不得在 A0、F1 和 CF 完成前新增更多模型。

## 9. 闭环评测摘要

详细步骤由 `EVALUATION.md` 维护，核心约束如下。若旧版 `EVALUATION.md` 与本文冲突，以本文为准；正式评测前必须同步更新配套清单。

- 每次预测 16 步，实际执行前 4 步，然后重新观测。
- 最大回合长度固定为 300 个环境步。
- 每个训练种子评测 50 个固定测试初始状态。
- A0 与 F1 使用完全相同的初始状态和环境种子。
- 报告每个训练种子的独立结果，再报告三个种子的均值和标准差。
- 不得把所有回合混合后当作完全独立的训练重复。

策略指标：

- 平均和中位数覆盖率。
- 成功率、回报和成功回合的完成步数。
- 离线动作误差。

工程指标：

- 总参数量与推理动作路径参数量。
- batch size 为 1 时、包含图像预处理的动作推理延迟。
- 峰值显存或统一内存占用。

延迟先预热 20 次，再测量 100 次，报告中位数和 95 分位数。

## 10. 反事实动作评估

反事实评估只检查未来分支，不重新训练策略。

### 10.1 离线动作打乱

```text
正确：history_i + action_i -> future_i
打乱：history_i + action_j -> future_i, j != i
```

记录正确动作误差与打乱动作误差。两者几乎无差异时，不能声称未来分支利用了动作。

### 10.2 反向动作

Push-T 动作若表示绝对目标位置，禁止直接使用 `-action`。必须先根据当前智能体位置转换为位移，对位移取反，再转换回合法动作并裁剪到环境范围。环境适配层必须为该变换提供测试。

### 10.3 小型模拟器反事实集

- 固定 20 个可恢复模拟器状态。
- 每个状态执行正确动作前缀和一个合法替代动作前缀。
- 保存两条真实四步未来轨迹。
- 比较每个预测与匹配、非匹配真实未来的误差。
- 替代动作必须处于环境合法范围，优先落在训练动作分布主要范围内。

该实验用于区分“预测会变化”和“变化与真实环境一致”。

## 11. 固定观测扰动

核心版本只保留亮度扰动：

- 对相同 50 个测试初始状态评测原图、`0.75x` 和 `1.25x` 亮度。
- 像素裁剪到合法范围，不重新训练或调参。
- 分别报告覆盖率和成功率变化。

初始物体位置、角度、动作延迟和观测缺失均为选修。

## 12. 结论规则

可以写：

- “在 Push-T 和当前设置下，`future_aware` 在多个训练种子上表现出稳定改善。”
- “未来分支对动作敏感，并在小型反事实集上表现出匹配未来优势。”
- “未来辅助损失改善了离线优化，但闭环控制没有稳定提升。”

不可以写：

- 仅一个种子更好，就声称世界模型改善机器人控制。
- 仅未来表示误差更低，就声称模型理解物理或因果关系。
- 未做 `future_no_action`，就声称收益一定来自动作条件化，而不是一般辅助监督。
- 将 Push-T 结果直接外推到真实机械臂。
- 将训练期未来辅助监督描述成推理期规划。

负结果仍然有效。必须区分已观察的失败现象与尚未证实的机制；缺少排除性证据时，不强行归因于数据、优化、表征冲突或分布偏移。

## 13. 仓库与产物

```text
mini-wam/
├── README.md
├── SPEC.md
├── EVALUATION.md
├── requirements.lock
├── configs/
│   ├── data.yaml
│   ├── action_only.yaml
│   ├── future_aware.yaml
│   ├── eval.yaml
│   └── eval_brightness.yaml
├── splits/
├── src/mini_wam/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
├── tests/
├── scripts/
│   ├── audit_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── counterfactual_eval.py
│   └── summarize_results.py
├── artifacts/
├── runs/
├── results/
└── reports/
    ├── decision_log.md
    └── failure_analysis.md
```

大 checkpoint、原始数据和运行缓存不得提交到 Git；只提交代码、配置、固定划分、汇总结果和小型展示产物。

README 至少提供等价命令：

```bash
python scripts/audit_dataset.py --config configs/data.yaml
python scripts/train.py --config configs/action_only.yaml --seed 0
python scripts/train.py --config configs/future_aware.yaml --seed 0
python scripts/evaluate.py --run-dir <run_directory>
python scripts/counterfactual_eval.py --run-dir <future_aware_run_directory>
python scripts/summarize_results.py --runs-dir runs --output results/summary.csv
```

所有命令必须支持 `--help`，失败时返回非零退出状态，并打印实际配置路径和运行目录。

## 14. 求职展示验收

README 首页必须在两分钟内展示：

1. 一句话问题定义。
2. 数据样本和时间对齐图。
3. 两模型数据流图。
4. 闭环核心结果表。
5. 正确与错误动作的未来误差对比。
6. 成功和失败视频。
7. 复现命令。
8. 局限性和下一步。

最终至少保留一张架构图、一张训练曲线、一张闭环结果表、一张反事实图，以及每个核心模型各一个成功和失败视频。

项目完成标准不是“所有指标都提升”，而是：数据正确、训练可复现、闭环评测公平、结论与证据一致。

## 15. 范围变更规则

新增模型、指标或扰动前必须回答：

1. 它是否直接提升机器人数据、训练、评测或闭环能力的展示？
2. 它是否改变核心问题？
3. 它是否会推迟 V0.1 或 V0.2？

如果只增加论文感而不改善上述能力，则放入选修，不进入当前实现。
