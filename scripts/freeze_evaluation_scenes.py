"""Generate immutable development and sealed final-test Push-T scenes."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from mini_wam.evaluation import freeze_evaluation_scenes


def main() -> None:
    manifest = freeze_evaluation_scenes(
        PROJECT_ROOT / "splits" / "dev_scenes.json",
        PROJECT_ROOT / "splits" / "final_test_scenes.json",
        PROJECT_ROOT / "splits" / "evaluation_scenes_manifest.json",
    )
    print(f"开发场景：{manifest['development']['count']} 个")
    print(f"最终测试场景：{manifest['final_test']['count']} 个（已封存，阶段 5 前禁止评测）")
    print(f"清单：{PROJECT_ROOT / 'splits' / 'evaluation_scenes_manifest.json'}")


if __name__ == "__main__":
    main()

