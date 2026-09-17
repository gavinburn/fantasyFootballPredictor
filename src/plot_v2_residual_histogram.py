"""Plot the out-of-sample V2 prediction-error distribution by position."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    SELECTED_PREDICTIONS_DATA_DIR,
    SELECTED_REPORT_TABLES_DIR,
    SELECTED_REPORTS_DIR,
)

POSITIONS = ("QB", "RB", "TE", "WR")


def load_v2_residuals() -> pd.DataFrame:
    predictions = pd.read_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_predictions.parquet"
    )
    predictions = predictions[
        ~predictions["specification"].eq("SELECTED_WR_PARTICIPATION_FALLBACK")
    ].copy()
    predictions["prediction_error_ppg"] = (
        predictions["predicted_ppg"] - predictions["actual_ppg"]
    )
    return predictions


def summarize_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    return (
        residuals.groupby("position", as_index=False)
        .agg(
            player_seasons=("prediction_error_ppg", "size"),
            mean_error_ppg=("prediction_error_ppg", "mean"),
            median_error_ppg=("prediction_error_ppg", "median"),
            standard_deviation_ppg=("prediction_error_ppg", "std"),
            minimum_error_ppg=("prediction_error_ppg", "min"),
            maximum_error_ppg=("prediction_error_ppg", "max"),
        )
        .sort_values("position")
    )


def build_plot() -> Path:
    residuals = load_v2_residuals()
    summary = summarize_residuals(residuals)
    SELECTED_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        SELECTED_REPORT_TABLES_DIR / "v2_residual_distribution_summary.csv",
        index=False,
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for position, axis in zip(POSITIONS, axes.flat, strict=True):
        errors = residuals.loc[
            residuals["position"].eq(position), "prediction_error_ppg"
        ]
        mean_error = float(errors.mean())
        median_error = float(errors.median())
        sns.histplot(
            errors,
            bins=24,
            stat="percent",
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.8,
            ax=axis,
        )
        axis.axvline(0, color="#222222", linewidth=1.6, label="Perfect prediction")
        axis.axvline(
            mean_error,
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label="Mean error",
        )
        axis.set_title(position)
        axis.set_xlabel("Prediction error (predicted PPG − actual PPG)")
        axis.set_ylabel("Player-seasons (%)")
        axis.grid(axis="y", alpha=0.22)
        axis.text(
            0.98,
            0.95,
            f"n = {len(errors):,}\nMean = {mean_error:+.2f}\nMedian = {median_error:+.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle("V2 out-of-sample prediction-error distributions", y=0.99, fontsize=15)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=True,
    )
    figure.text(
        0.5,
        0.015,
        "Negative error = underprediction; positive error = overprediction",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.89))
    output_dir = SELECTED_REPORTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "v2_residual_histogram_by_position.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    print(build_plot())
