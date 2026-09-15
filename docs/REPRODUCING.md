# 历史实验复现与证据使用

所有命令从仓库根目录执行。本文仅复现 2026-09-10 收尾的 `action_only` / `future_aware` 单种子实验，其 185/21 数据划分和最低离线验证损失选点规则保持不变。
当前 ACT（Action Chunking with Transformers，基于 Transformer 的动作分块）阶段改用全部 206 个示范回合训练、随机模拟场景验证，见[当前规格](../SPEC.md)与[验收协议](../EVALUATION.md)。
ACT 模型及入口待实现。下面的历史评估命令不支持 ACT，不会训练模型，也不接受原封存最终测试场景。

## 1. 不下载模型即可核对结论

需要 Python 3.10 或以上；仅使用标准库，无需数据集或图形设备：

```bash
python3 scripts/summarize_closure.py --output-dir outputs/closure-summary
```

成功标志：`CLOSURE_AUDIT_OK files=14 scene_pairs=20 p=0.375`。
脚本核对原证据包文件大小和 SHA-256（Secure Hash Algorithm 256-bit，256 位安全哈希算法），
从 40 条回合记录重新配对、计算成功率、覆盖率和 McNemar 精确双侧检验，
再核对报告及逐回合表。输出新的 `summary.json`、`paired_comparison.csv` 和 `SUMMARY.md`。
原始证据保持原字节，不被汇总脚本覆盖。

安装可选绘图库后，可额外生成带 Wilson 95% 区间的对比图及反事实损失图：

```bash
python3 -m pip install -r requirements-report.txt
python3 scripts/summarize_closure.py --output-dir outputs/closure-summary --plots
```

原始训练曲线保存在仓库；本入口的反事实图从已归档损失汇总绘制，**不会重新计算模型预测**。
训练曲线的全部逐步训练记录仍在原训练目录，仓库保留原验证记录、运行日志及完成审计。

## 2. 准备环境、数据及冻结检查点

环境和数据安装见 [GETTING_STARTED.md](GETTING_STARTED.md)、[COLAB.md](../COLAB.md)。
本地锁定环境与历史云端环境不同；原始版本记录见 `reports/closure/training/*/environment.json`。
原记录未涵盖所有模拟器依赖，因此不承诺跨机器逐位重现历史闭环轨迹。

两种模型都训练了 50,000 步，以各自最低离线验证损失选择检查点：

| 模型 | 选择指标 | 选中步数 | 检查点 |
|---|---|---:|---|
| action_only | 动作验证损失 | 10,000 | [原始 best.pt](https://drive.google.com/file/d/1leXbu8G57DD4BkxvCVVCUdo3iX1cqr_6/view?usp=drivesdk) |
| future_aware | 动作损失 + 0.1 × 未来损失 | 34,000 | [原始 best.pt](https://drive.google.com/file/d/1z2Ko-2CKBuiijN-yLngMnFRGq4cYyBwZ/view?usp=drivesdk) |

检查点分别约 143.5 MB 和 219.8 MB（Megabyte，兆字节），没有纳入 Git。
链接保持原 Drive 权限，只有已有权限的账号能下载。**仓库内的结果、图片和代表视频可直接查看；
历史权重的第三方下载尚不公开。** 无权重访问权的读者仍可核对结果，或自行重新训练，
但重新训练的模型不能冒充这些历史检查点。

下载后分别保存为 `artifacts/closure/action_only_best.pt` 和
`artifacts/closure/future_aware_best.pt`。哈希及大小见
[FINAL_REPORT.json](../reports/closure/FINAL_REPORT.json) 和
[CHECKPOINT_SOURCES.json](../reports/closure/CHECKPOINT_SOURCES.json)。
评估入口在加载前校验哈希，并核对归一化与训练/验证划分。

## 3. 重新执行 20 场景配对闭环

```bash
./.venv/bin/python scripts/evaluate_paired_closure.py \
  --action-checkpoint artifacts/closure/action_only_best.pt \
  --future-checkpoint artifacts/closure/future_aware_best.pt \
  --output-dir outputs/closure-paired-replay \
  --device cpu
```

成功标志：`PAIRED_COMPARISON_COMPLETE`。保存 40 段视频、逐回合结果、重新计算的统计量、
实际软件版本和源码校验值。输出目录必须不存在，异常会留下失败记录及已完成回合。
每个场景两模型交替先运行，固定最大 300 步，预测 16 步、执行 4 步。
默认使用 CPU（Central Processing Unit，中央处理器），与历史配对评估的设备类别相同。
其他设备可用于探索，但应报告差异；速度字段不是包含预处理和同步的正式延迟基准。

该入口只加载 `reports/closure/scenes.json`：从种子 2026091000 开始按顺序取前 20 个合法场景，
排除 2026091000、2026091005 两个非法初始接触场景，最后一个种子为 2026091021。
场景在运行模型前固定；它们是**额外开发集**，与 `splits/dev_scenes.json` 的早期开发集不同。

## 4. 重新执行离线反事实检查

```bash
./.venv/bin/python scripts/evaluate_counterfactual.py \
  --checkpoint artifacts/closure/future_aware_best.pt \
  --dataset-root data/lerobot/pusht_image \
  --output-dir outputs/closure-counterfactual-replay \
  --device cpu
```

成功标志：`COUNTERFACTUAL_COMPLETE`。使用冻结验证回合，按原窗口顺序、每批 64 个样本、
不打乱批次，并丢弃最后不足 64 个窗口的批次，以保持历史协议：2,368 个窗口、9,352 个有效未来步。
完整可用窗口数和丢弃数也会写入结果。不要把该数字描述为全部验证窗口。

- 正确动作：同一窗口的专家动作前缀。
- 打乱动作：每个完整批次内循环移位一个样本，无自配对；可能仍来自相邻、相似窗口，检验强度有限。
- 反向动作：先反归一化绝对目标，围绕当前智能体位置反射 `2 × position − action`，裁剪到 `[0,512]` 后再归一化。

这检查的是“错误动作是否使同一真实未来的预测损失上升”，没有为反事实动作执行模拟器来获得匹配未来，
因此不能证明预测变化符合真实物理。较大的反向动作偏移也可能引入分布外条件。

## 5. 原始执行代码与新的可运行入口

[original_cells](../reports/closure/original_cells/) 保留实际执行过的五段原始代码单元，
用于核对批次、场景、统计与视频选择来源；这些代码依赖当时笔记本的上下文，**不是独立脚本**。
当前 `scripts/` 入口将同一协议整理为独立命令，增加错误检查和运行记录。
它们已通过本地接口测试、真实数据小批检查及短闭环视频验证；本轮没有用正式权重重跑全部 40 回合，
也没有在第二台机器从头复现训练。历史结果与新入口的验证范围分别记录，不能互相替代。

## 6. 重新训练

原始配置见 `reports/closure/training/<model>/config.yaml`，与对应 `configs/*_seed0.yaml` 核对使用。
训练命令见 [COLAB.md](../COLAB.md)。长训练前仍须执行项目规定的数据审计、目标设备性能测量与恢复检查。
本次收尾不自动启动训练，封存测试集保持未使用。
