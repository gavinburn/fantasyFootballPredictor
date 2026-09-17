"""Plot identical-cohort MAE for the baseline, V1, and V2 by position."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import SELECTED_REPORT_TABLES_DIR, SELECTED_REPORTS_DIR


def build_plot() -> Path:
    comparison = pd.read_csv(
        SELECTED_REPORT_TABLES_DIR / "v2_v1_previous_ppg_comparison.csv"
    )
    positions = ["QB", "RB", "TE", "WR"]
    values = comparison.pivot(index="position", columns="model", values="mae_ppg")
    values = values.loc[positions]
    x = np.arange(len(positions))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        x - 0.24,
        values["Previous-season PPG"],
        width=0.24,
        label="Previous-season PPG",
    )
    axis.bar(
        x,
        values["V1"],
        width=0.24,
        label="V1 linear regression",
    )
    axis.bar(
        x + 0.24,
        values["V2"],
        width=0.24,
        label="V2 selected model",
    )
    axis.set_xticks(x, positions)
    axis.set_ylabel("Overall MAE (PPG)")
    axis.set_title("V2 versus V1 linear regression and primary baseline")
    axis.legend()
    figure.tight_layout()

    output_dir = SELECTED_REPORTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "v2_v1_baseline_comparison_by_position.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


if __name__ == "__main__":
    build_plot()
