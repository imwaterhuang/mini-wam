"""Verify archived evidence and regenerate the closure table and comparison figure."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Import only the standard-library audit module, avoiding simulation dependencies.
spec = importlib.util.spec_from_file_location("closure_audit", ROOT / "src/mini_wam/evaluation/closure.py")
closure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "reports/closure")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/closure-summary")
    parser.add_argument("--plots", action="store_true", help="Also regenerate plots; requires matplotlib")
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.evidence_dir.resolve()):
        parser.error("Write regenerated outputs outside the archived evidence directory")
    print(f"EVIDENCE {args.evidence_dir.resolve()}\nOUTPUT {args.output_dir.resolve()}")
    audit = closure.audit_archive(args.evidence_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    closure.write_json(args.output_dir / "summary.json", audit)
    with (args.output_dir / "paired_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit["paired_rows"][0]))
        writer.writeheader()
        writer.writerows(audit["paired_rows"])
    lines = ["# Recomputed closure results", "", "| Model | Success | Mean coverage | Median coverage | Coverage std |", "|---|---:|---:|---:|---:|"]
    for name, s in audit["summary"].items():
        lines.append(f"| {name} | {s['success_count']}/{s['episode_count']} | {s['mean_final_coverage']:.6f} | {s['median_final_coverage']:.6f} | {s['final_coverage_std']:.6f} |")
    lines += ["", f"Exact two-sided McNemar p = {audit['mcnemar_exact_two_sided_p']:.3f}.", "", audit["interval_scope"] + "."]
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        result = closure.read_json(args.evidence_dir / "results.json")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        names = list(closure.MODELS)
        for i, name in enumerate(names):
            s = audit["summary"][name]
            lo, hi = s["success_rate_wilson_95"]
            axes[0].bar(i, s["success_rate"], color=["#3b82f6", "#f97316"][i])
            axes[0].errorbar(i, s["success_rate"], yerr=[[s["success_rate"] - lo], [hi - s["success_rate"]]], color="black", capsize=5)
        axes[0].set(xticks=[0, 1], xticklabels=names, ylim=(0, 1), ylabel="Strict success rate", title="20 paired development scenes / Wilson 95%")
        axes[1].boxplot([[e["final_coverage"] for e in result["episodes"][n]] for n in names], tick_labels=names)
        axes[1].axhline(.95, ls="--", color="green")
        axes[1].set(ylim=(0, 1.05), ylabel="Final coverage", title=f"Exact McNemar p = {audit['mcnemar_exact_two_sided_p']:.3f}")
        fig.savefig(args.output_dir / "paired_comparison.png", dpi=180)
        plt.close(fig)
        counter = closure.read_json(args.evidence_dir / "FINAL_REPORT.json")["future_branch_counterfactual"]
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        losses = counter["mean_masked_cosine_loss"]
        ax.bar(list(losses), list(losses.values()), color=["#3b82f6", "#f59e0b", "#ef4444"])
        ax.set(ylabel="Masked cosine distance", title=f"{counter['windows_evaluated']} offline validation windows")
        fig.savefig(args.output_dir / "counterfactual_losses.png", dpi=180)
        plt.close(fig)
    print(f"CLOSURE_AUDIT_OK files={audit['original_files_verified']} scene_pairs={len(audit['paired_rows'])} p={audit['mcnemar_exact_two_sided_p']}")


if __name__ == "__main__":
    main()
