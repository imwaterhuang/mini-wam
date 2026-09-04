"""Evaluate the deterministic random-action baseline on development scenes."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from mini_wam.evaluation import evaluate_random_policy


def main() -> None:
    result = evaluate_random_policy(
        PROJECT_ROOT / "splits" / "dev_scenes.json",
        PROJECT_ROOT / "reports" / "baselines" / "random_policy_development.json",
        PROJECT_ROOT / "reports" / "baselines" / "random_policy_development.csv",
    )
    summary = result["summary"]
    print(f"回合：{result['episode_count']}")
    print(f"成功率：{summary['success_rate']:.1%}")
    print(f"平均最终覆盖率：{summary['mean_final_coverage']:.6f}")
    print(f"平均最大覆盖率：{summary['mean_max_coverage']:.6f}")
    print(f"平均回报：{summary['mean_reward']:.6f}")


if __name__ == "__main__":
    main()

