import csv, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

runs=Path("/content/drive/MyDrive/mini-wam-runs")
source=runs/"comparisons"/"web_random_20260910_20scenes_v1"
future_root=runs/"future-aware-seed0-w4"/"finalization"
out=runs/"project-closure-20260910"
(out/"representative_videos").mkdir(parents=True,exist_ok=True)
(out/"figures").mkdir(parents=True,exist_ok=True)

paired=json.loads((source/"results.json").read_text())
future_report=json.loads((future_root/"post_training_final_report.json").read_text())
counter=json.loads((future_root/"counterfactual"/"results.json").read_text())

for name in ["results.json","scenes.json","paired_comparison.csv"]:
    shutil.copy2(source/name,out/name)
if (source/"comparison.png").exists():
    shutil.copy2(source/"comparison.png",out/"figures"/"paired_comparison.png")
if (future_root/"counterfactual"/"counterfactual_losses.png").exists():
    shutil.copy2(future_root/"counterfactual"/"counterfactual_losses.png",out/"figures"/"counterfactual_losses.png")
if (future_root/"loss_curves.png").exists():
    shutil.copy2(future_root/"loss_curves.png",out/"figures"/"future_aware_loss_curves.png")

video_cases={
    "shared_success":"web-random-04",
    "action_only_only_success":"web-random-09",
    "future_aware_only_success":"web-random-00",
}
copied_videos=[]
for case,sid in video_cases.items():
    for model in ["action_only","future_aware"]:
        src=source/model/"videos"/f"{sid}.mp4"
        dst=out/"representative_videos"/f"{case}__{model}__{sid}.mp4"
        shutil.copy2(src,dst)
        copied_videos.append(str(dst.relative_to(out)))

closure={
    "schema_version":1,
    "status":"closed_as_single_seed_exploratory_study",
    "closed_at_utc":datetime.now(timezone.utc).isoformat(),
    "scope":{
        "task":"Push-T",
        "models":["action_only","future_aware"],
        "training_seeds_per_model":1,
        "checkpoint_selection_rule":"minimum offline validation loss; frozen before this closure",
        "sealed_final_test_used":False,
        "additional_training_performed_during_closure":False,
    },
    "primary_paired_development_result":{
        "scene_count":paired["protocol"]["episode_count"],
        "same_initial_states":paired["protocol"]["same_initial_states"],
        "action_only":paired["summary"]["action_only"],
        "future_aware":paired["summary"]["future_aware"],
        "future_minus_action_success_rate":paired["success_rate_difference_future_minus_action"],
        "paired_outcomes":paired["paired_outcomes"],
        "mcnemar_exact_two_sided_p":paired["mcnemar_exact_two_sided_p"],
        "uncertainty_and_dispersion":paired["uncertainty_and_dispersion"],
    },
    "future_aware_additional_development_result":future_report["development_closed_loop"],
    "future_branch_counterfactual":{
        "windows_evaluated":counter["windows_evaluated"],
        "valid_future_steps":counter["valid_future_steps"],
        "mean_masked_cosine_loss":counter["mean_masked_cosine_loss"],
        "deltas_vs_correct":counter["deltas_vs_correct"],
        "interpretation":"reversed actions increase loss clearly; shuffled-action delta is positive but weak",
    },
    "checkpoints":paired["checkpoints"],
    "unavailable_stability_metrics":{
        "action_reversal_count":"not collected",
        "action_change_magnitude":"not collected",
        "coverage_regression_count":"not collected",
        "reason":"step-level action and coverage trajectories were not persisted; videos are not used to fabricate quantitative values",
    },
    "claim_boundary":[
        "single training seed per model",
        "20-scene paired set is development evidence, not a sealed final test",
        "success-rate difference is not statistically significant",
        "the future-aware policy has lower median final coverage and slightly higher dispersion on the paired set",
        "no claim of robust superiority or generalization beyond Push-T",
    ],
    "representative_videos":copied_videos,
}

(out/"FINAL_REPORT.json").write_text(json.dumps(closure,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

md=f"""# Mini-WAM 项目收尾报告

## 项目状态

项目以“单随机种子的探索性对照实验”收尾。没有追加训练，也没有使用封存最终测试集。

## 冻结协议

- 两个模型各使用一个正式训练随机种子。
- checkpoint（检查点）统一按最低离线验证损失选择。
- 配对开发评估使用相同的 20 个初始状态。
- 每回合最多 300 个环境步；模型预测 16 步动作，每执行 4 步后重新规划。
- 严格成功标准为覆盖率大于 0.95。

## 主要结果

| 指标 | action_only | future_aware |
|---|---:|---:|
| 严格成功 | 2/20 (10%) | 5/20 (25%) |
| 平均最终覆盖率 | {paired["summary"]["action_only"]["mean_final_coverage"]:.3f} | {paired["summary"]["future_aware"]["mean_final_coverage"]:.3f} |
| 中位最终覆盖率 | {paired["summary"]["action_only"]["median_final_coverage"]:.3f} | {paired["summary"]["future_aware"]["median_final_coverage"]:.3f} |
| 最终覆盖率标准差 | {paired["uncertainty_and_dispersion"]["action_only_final_coverage_std"]:.3f} | {paired["uncertainty_and_dispersion"]["future_aware_final_coverage_std"]:.3f} |

配对结果为：共同成功 1 个、仅 action_only 成功 1 个、仅 future_aware 成功 4 个、共同失败 14 个。McNemar（配对二分类检验）精确双侧 p 值为 {paired["mcnemar_exact_two_sided_p"]:.3f}，不能认为成功率差异已经达到统计显著。

未来分支反事实检查覆盖 {counter["windows_evaluated"]} 个验证窗口。正确、打乱、合法反向动作的 masked cosine loss（带掩码余弦损失）分别为 {counter["mean_masked_cosine_loss"]["correct"]:.5f}、{counter["mean_masked_cosine_loss"]["shuffled"]:.5f}、{counter["mean_masked_cosine_loss"]["reversed"]:.5f}。反向动作使误差明显增加，但打乱动作只带来较弱变化，因此动作依赖证据有限。

## 结论

在相同数据、动作路径和闭环评估条件下，加入动作条件未来表征辅助损失后，模型在部分场景中获得了更高的严格成功率，但没有表现出更稳定的总体控制能力。现有配对实验的统计差异不显著，且 future_aware 模型的典型覆盖率更低、结果更加两极化。

因此，本实验不能证明未来辅助监督稳定优于 action_only 基线，只能说明它可能改变策略表征，并在少数场景中带来收益。该结果定位为探索性发现，而非确定性结论。

## 局限

- 每种模型只训练一个随机种子，无法估计训练随机性。
- 20 个配对场景只适合发现趋势，不足以证明稳定优势。
- 封存最终测试集保持未使用。
- 本轮未持久化逐步动作和覆盖率轨迹，因此不能可靠报告动作反转次数、动作变化幅度和覆盖率回退次数。
- 结论只适用于当前 Push-T 数据、模型结构和扰动范围。

## 证据目录

- FINAL_REPORT.json：机器可读汇总。
- results.json、paired_comparison.csv、scenes.json：配对评估原始结果与场景。
- figures/：成功率、覆盖率与反事实损失图。
- representative_videos/：共同成功、两种单模型独有成功的成对视频。
- ARTIFACT_MANIFEST.json：文件大小与 SHA-256（Secure Hash Algorithm 256-bit，256 位安全哈希算法）校验值。
"""
(out/"PROJECT_CONCLUSION.md").write_text(md,encoding="utf-8")

manifest=[]
for p in sorted(out.rglob("*")):
    if p.is_file() and p.name!="ARTIFACT_MANIFEST.json":
        manifest.append({
            "path":str(p.relative_to(out)),
            "bytes":p.stat().st_size,
            "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
        })
(out/"ARTIFACT_MANIFEST.json").write_text(json.dumps({
    "root":str(out),
    "file_count":len(manifest),
    "files":manifest,
},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

print("PROJECT_CLOSURE_COMPLETE",out)
print("MANIFEST_FILES",len(manifest))
print("VIDEOS",len(copied_videos))
print("FINAL_TEST_USED",closure["scope"]["sealed_final_test_used"])
print("REPORT_BYTES",(out/"FINAL_REPORT.json").stat().st_size)
print("CONCLUSION_BYTES",(out/"PROJECT_CONCLUSION.md").stat().st_size)
