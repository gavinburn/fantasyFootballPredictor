"""Plot annual V2, V1, and previous-season PPG performance trends."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    PREDICTIONS_DATA_DIR,
    SELECTED_PREDICTIONS_DATA_DIR,
    SELECTED_REPORT_TABLES_DIR,
    SELECTED_REPORTS_DIR,
)
from src.validation import TOP_N_BY_POSITION

METHOD_COLUMNS = {
    "Previous-season PPG": "previous_season_ppg_prediction",
    "V1 linear regression": "v1_prediction",
    "V2 selected model": "v2_prediction",
}
POSITIONS = ("QB", "RB", "TE", "WR")
COLORS = {
    "Previous-season PPG": "#1f77b4",
    "V1 linear regression": "#ff7f0e",
    "V2 selected model": "#2ca02c",
}
MARKERS = {
    "Previous-season PPG": "o",
    "V1 linear regression": "s",
    "V2 selected model": "^",
}


def load_identical_cohort() -> pd.DataFrame:
    selected = pd.read_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_predictions.parquet"
    )
    selected = selected[
        ~selected["specification"].eq("SELECTED_WR_PARTICIPATION_FALLBACK")
    ][["player_id", "player_name", "position", "season", "actual_ppg", "predicted_ppg"]]
    selected = selected.rename(columns={"predicted_ppg": "v2_prediction"})

    v1 = pd.read_parquet(
        PREDICTIONS_DATA_DIR / "historical_linear_regression_predictions.parquet"
    )[["player_id", "position", "season", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "v1_prediction"}
    )
    baseline = pd.read_parquet(
        PREDICTIONS_DATA_DIR / "historical_baseline_predictions.parquet"
    )
    baseline = baseline[baseline["model_name"].eq("previous_season_ppg")][
        ["player_id", "position", "season", "predicted_ppg"]
    ].rename(columns={"predicted_ppg": "previous_season_ppg_prediction"})

    keys = ["player_id", "position", "season"]
    return selected.merge(v1, on=keys, validate="one_to_one").merge(
        baseline, on=keys, validate="one_to_one"
    )


def calculate_yearly_metrics(cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (position, season), group in cohort.groupby(["position", "season"], sort=True):
        top_n = min(TOP_N_BY_POSITION[position], len(group))
        actual_top = set(group.nlargest(top_n, "actual_ppg")["player_id"])
        for method, column in METHOD_COLUMNS.items():
            predicted_top = set(group.nlargest(top_n, column)["player_id"])
            rows.append(
                {
                    "position": position,
                    "season": int(season),
                    "method": method,
                    "players": len(group),
                    "mae_ppg": (group[column] - group["actual_ppg"]).abs().mean(),
                    "top_n": top_n,
                    "top_n_overlap_percentage": (
                        100.0 * len(actual_top & predicted_top) / top_n
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_metric(
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
        position_data = metrics[metrics["position"].eq(position)]
        for method in METHOD_COLUMNS:
            series = position_data[position_data["method"].eq(method)]
            axis.plot(
                series["season"],
                series[value_column],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4,
                linewidth=1.8,
                label=method,
            )
        axis.set_title(position)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if fixed_ylim is not None:
            axis.set_ylim(*fixed_ylim)
        seasons = sorted(position_data["season"].unique())
        axis.set_xticks(seasons[::2] if len(seasons) > 7 else seasons)
        axis.tick_params(axis="x", rotation=45)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(title, y=0.99, fontsize=15)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=True,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    output_dir = SELECTED_REPORTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / output_name
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def build_plots() -> tuple[Path, Path]:
    metrics = calculate_yearly_metrics(load_identical_cohort())
    SELECTED_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(SELECTED_REPORT_TABLES_DIR / "yearly_v2_trend_metrics.csv", index=False)
    mae = _plot_metric(
        metrics,
        value_column="mae_ppg",
        ylabel="MAE (PPG)",
        title="Year-by-year prediction error: V2, V1, and previous-season PPG",
        output_name="yearly_mae_v2_v1_baseline.png",
    )
    overlap = _plot_metric(
        metrics,
        value_column="top_n_overlap_percentage",
        ylabel="Top-N overlap (%)",
        title="Year-by-year Top-N overlap: V2, V1, and previous-season PPG",
        output_name="yearly_top_n_overlap_v2_v1_baseline.png",
        fixed_ylim=(0, 100),
    )
    return mae, overlap


if __name__ == "__main__":
    for path in build_plots():
        print(path)
