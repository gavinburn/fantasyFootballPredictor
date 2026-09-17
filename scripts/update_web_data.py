"""Copy the published rankings CSV into the static rankings browser."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/predictions_2026/v2_2026_rankings.csv"
DESTINATION = ROOT / "web/rankings.csv"


if __name__ == "__main__":
    shutil.copyfile(SOURCE, DESTINATION)
    print(f"Updated {DESTINATION.relative_to(ROOT)}")
