"""Create report figures and a summary table for V2 alone."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    SELECTED_PREDICTIONS_DATA_DIR,
    SELECTED_REPORT_TABLES_DIR,
    SELECTED_REPORTS_DIR,
)
from src.validation import TOP_N_BY_POSITION

POSITIONS = ("QB", "RB", "TE", "WR")
V2_COLOR = "#2ca02c"


def load_v2_predictions() -> pd.DataFrame:
    predictions = pd.read_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_predictions.parquet"
    )
    return predictions[
        ~predictions["specification"].eq("SELECTED_WR_PARTICIPATION_FALLBACK")
    ].copy()


def calculate_v2_yearly_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (position, season), group in predictions.groupby(
        ["position", "season"], sort=True
    ):
        top_n = min(TOP_N_BY_POSITION[position], len(group))
        actual_top = set(group.nlargest(top_n, "actual_ppg")["player_id"])
        predicted_top = set(group.nlargest(top_n, "predicted_ppg")["player_id"])
        rows.append(
            {
                "position": position,
                "season": int(season),
                "players": len(group),
                "mae_ppg": (
                    group["predicted_ppg"] - group["actual_ppg"]
                ).abs().mean(),
                "top_n": top_n,
                "top_n_overlap_percentage": (
                    100.0 * len(actual_top & predicted_top) / top_n
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_v2(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position, group in metrics.groupby("position", sort=True):
        best = group.loc[group["mae_ppg"].idxmin()]
        worst = group.loc[group["mae_ppg"].idxmax()]
        rows.append(
            {
                "position": position,
                "evaluation_period": (
                    f"{int(group['season'].min())}-{int(group['season'].max())}"
                ),
                "seasons": group["season"].nunique(),
                "player_seasons": int(group["players"].sum()),
                "mean_seasonal_mae_ppg": group["mae_ppg"].mean(),
                "best_season": int(best["season"]),
                "best_season_mae_ppg": best["mae_ppg"],
                "worst_season": int(worst["season"]),
                "worst_season_mae_ppg": worst["mae_ppg"],
                "mean_top_n_overlap_percentage": group[
                    "top_n_overlap_percentage"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def _plot(
    metrics: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    output_name: str,
    fixed_ylim: tuple[float, float] | None = None,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for position, axis in zip(POSITIONS, axes.flat, strict=True):
        series = metrics[metrics["position"].eq(position)]
        mean_value = float(series[value_column].mean())
        axis.plot(
            series["season"],
            series[value_column],
            color=V2_COLOR,
            marker="o",
            markersize=5,
            linewidth=2.2,
            label="V2",
        )
        axis.axhline(
            mean_value,
            color="#555555",
            linestyle="--",
            linewidth=1.3,
            label=f"Mean ({mean_value:.2f})",
        )
        axis.set_title(position)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")
        if fixed_ylim is not None:
            axis.set_ylim(*fixed_ylim)
        seasons = sorted(series["season"].unique())
        axis.set_xticks(seasons[::2] if len(seasons) > 7 else seasons)
        axis.tick_params(axis="x", rotation=45)

    figure.suptitle(title, y=0.985, fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir = SELECTED_REPORTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / output_name
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def build_outputs() -> tuple[Path, Path]:
    metrics = calculate_v2_yearly_metrics(load_v2_predictions())
    summary = summarize_v2(metrics)
    SELECTED_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        SELECTED_REPORT_TABLES_DIR / "v2_only_yearly_metrics.csv", index=False
    )
    summary.to_csv(
        SELECTED_REPORT_TABLES_DIR / "v2_only_yearly_summary.csv", index=False
    )
    mae = _plot(
        metrics,
        value_column="mae_ppg",
        ylabel="MAE (PPG)",
        title="V2 year-by-year prediction error",
        output_name="v2_only_yearly_mae.png",
    )
    overlap = _plot(
        metrics,
        value_column="top_n_overlap_percentage",
        ylabel="Top-N overlap (%)",
        title="V2 year-by-year Top-N overlap",
        output_name="v2_only_yearly_top_n_overlap.png",
        fixed_ylim=(0, 100),
    )
    return mae, overlap


if __name__ == "__main__":
    for output in build_outputs():
        print(output)
