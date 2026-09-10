"""Audit the frozen exploratory study and recompute statistics from episodes.

This module uses only the standard library. Evaluation runners import torch
separately; auditing archived evidence never loads a model or runs final tests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = PROJECT_ROOT / "reports" / "closure"
MODELS = ("action_only", "future_aware")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_closure_scenes(path=EVIDENCE_ROOT / "scenes.json"):
    """Accept only the archived additional development scenes, never final test."""
    manifest = read_json(EVIDENCE_ROOT / "ARTIFACT_MANIFEST.json")
    expected = next(x["sha256"] for x in manifest["files"] if x["path"] == "scenes.json")
    if sha256_file(path) != expected:
        raise ValueError("Only the frozen closure development scenes are accepted")
    return read_json(path)


def wilson_interval(k, n, z=1.959963984540054):
    if not 0 <= k <= n or n <= 0:
        raise ValueError("Invalid success count")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - m, c + m]


def paired_statistics(episodes, scenes):
    expected = {x["scene_id"]: x for x in scenes["scenes"]}
    if len(expected) != len(scenes["scenes"]) or not expected:
        raise ValueError("Duplicate or empty scenes")
    indexed, summary = {}, {}
    for name in MODELS:
        rows = episodes[name]
        by_id = {x["scene_id"]: x for x in rows}
        if len(by_id) != len(rows) or set(by_id) != set(expected):
            raise ValueError(f"{name}: incomplete or duplicate scene pairing")
        for sid, row in by_id.items():
            if row["seed"] != expected[sid]["seed"]:
                raise ValueError(f"{name}/{sid}: seed mismatch")
            if type(row["is_success"]) is not bool or row.get("stopped", False):
                raise ValueError(f"{name}/{sid}: invalid or interrupted outcome")
            final, maximum = row["final_coverage"], row["max_coverage"]
            if not 0 <= final <= maximum <= 1:
                raise ValueError(f"{name}/{sid}: invalid coverage")
            if row["is_success"] != (final > 0.95):
                raise ValueError(f"{name}/{sid}: inconsistent strict success")
            if not 1 <= row["steps"] <= scenes["max_steps"]:
                raise ValueError(f"{name}/{sid}: invalid episode length")
        coverage = [r["final_coverage"] for r in rows]
        count = sum(r["is_success"] for r in rows)
        summary[name] = {
            "episode_count": len(rows), "success_count": count,
            "success_rate": count / len(rows),
            "mean_final_coverage": statistics.mean(coverage),
            "median_final_coverage": statistics.median(coverage),
            "final_coverage_std": statistics.pstdev(coverage),
            "success_rate_wilson_95": wilson_interval(count, len(rows)),
        }
        indexed[name] = by_id
    counts = dict.fromkeys(("both_success", "action_only_only", "future_aware_only", "neither_success"), 0)
    paired_rows = []
    for sid, scene in expected.items():
        a, f = (indexed[n][sid] for n in MODELS)
        key = ("both_success" if a["is_success"] and f["is_success"] else
               "action_only_only" if a["is_success"] else
               "future_aware_only" if f["is_success"] else "neither_success")
        counts[key] += 1
        paired_rows.append({"scene_id": sid, "seed": scene["seed"],
                            **{f"{n}_{k}": indexed[n][sid][k] for k in
                               ("is_success", "final_coverage", "max_coverage") for n in MODELS}})
    n = counts["action_only_only"] + counts["future_aware_only"]
    minority = min(counts["action_only_only"], counts["future_aware_only"])
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(minority + 1)) / 2**n) if n else 1.0
    return {"summary": summary, "paired_outcomes": counts,
            "mcnemar_exact_two_sided_p": p,
            "success_rate_difference_future_minus_action": summary["future_aware"]["success_rate"] - summary["action_only"]["success_rate"],
            "std_convention": "population, ddof=0, dispersion over these scenes",
            "interval_scope": "scene uncertainty conditional on one trained model; not training-seed uncertainty",
            "paired_rows": paired_rows}


def audit_archive(root=EVIDENCE_ROOT):
    root = Path(root).resolve()
    manifest = read_json(root / "ARTIFACT_MANIFEST.json")
    if manifest["file_count"] != len(manifest["files"]):
        raise ValueError("Manifest count mismatch")
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe evidence path: {entry['path']}")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {entry['path']}")
    result = read_json(root / "results.json")
    scenes = read_json(root / "scenes.json")
    report = read_json(root / "FINAL_REPORT.json")
    if result["protocol"]["sealed_final_test_used"] or report["scope"]["sealed_final_test_used"]:
        raise ValueError("Closure must not use the sealed final test")
    stats = paired_statistics(result["episodes"], scenes)
    if stats["paired_outcomes"] != result["paired_outcomes"]:
        raise ValueError("Stored paired outcomes disagree with episodes")
    if stats["mcnemar_exact_two_sided_p"] != result["mcnemar_exact_two_sided_p"]:
        raise ValueError("Stored p-value disagrees with episodes")
    for name in MODELS:
        for key in ("success_count", "success_rate", "mean_final_coverage", "median_final_coverage"):
            actual = stats["summary"][name][key]
            for saved in (result["summary"][name], report["primary_paired_development_result"][name]):
                if not math.isclose(actual, saved[key], abs_tol=1e-12):
                    raise ValueError(f"Stored {name}/{key} disagrees with episodes")
        recorded = result["uncertainty_and_dispersion"][f"{name}_final_coverage_std"]
        if not math.isclose(stats["summary"][name]["final_coverage_std"], recorded, abs_tol=1e-12):
            raise ValueError("Coverage dispersion mismatch")
    if result["checkpoints"] != report["checkpoints"]:
        raise ValueError("Checkpoint identities differ across reports")
    counter_path = root / "counterfactual/results.json"
    if counter_path.exists():
        counter = read_json(counter_path)
        for key in ("windows_evaluated", "valid_future_steps", "mean_masked_cosine_loss", "deltas_vs_correct"):
            if counter[key] != report["future_branch_counterfactual"][key]:
                raise ValueError(f"Counterfactual source/report mismatch: {key}")
    for name in MODELS:
        training = root / "training" / name
        if training.exists():
            with (training / "val_metrics.csv").open(newline="") as handle:
                unique = {int(row["step"]): row for row in csv.DictReader(handle)}
            loss_key = "validation_loss" if name == "action_only" else "total_loss"
            best = min(unique, key=lambda step: float(unique[step][loss_key]))
            if best != result["checkpoints"][name]["step"]:
                raise ValueError(f"{name}: selected checkpoint is not the validation minimum")
            if max(unique) != 50000 or "completed target step 50000" not in (training / "run.log").read_text():
                raise ValueError(f"{name}: training completion evidence is incomplete")
    supplemental = root / "DELIVERY_MANIFEST.json"
    supplementary_count = 0
    if supplemental.exists():
        for entry in read_json(supplemental)["files"]:
            path = (root / entry["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Supplemental evidence checksum mismatch: {entry['path']}")
            supplementary_count += 1
    with (root / "paired_comparison.csv").open(newline="") as handle:
        saved_rows = list(csv.DictReader(handle))
    if len(saved_rows) != len(stats["paired_rows"]):
        raise ValueError("Paired table row count mismatch")
    for saved, current in zip(saved_rows, stats["paired_rows"]):
        for key, value in current.items():
            old_key = key.replace("_is_success", "_success")
            if isinstance(value, (bool, str)):
                equal = str(value) == saved[old_key]
            else:
                equal = math.isclose(value, float(saved[old_key]), abs_tol=1e-12)
            if not equal:
                raise ValueError(f"Paired table mismatch: {key}")
    return {"original_files_verified": len(manifest["files"]),
            "supplementary_files_verified": supplementary_count, **stats}
