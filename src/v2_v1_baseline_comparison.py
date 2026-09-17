"""Create an identical-cohort comparison of V2, V1, and prior-season PPG."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    PREDICTIONS_DATA_DIR,
    SELECTED_PREDICTIONS_DATA_DIR,
    SELECTED_REPORT_TABLES_DIR,
    SELECTED_REPORTS_DIR,
)


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "spearman": float(actual.corr(predicted, method="spearman")),
    }


def build_comparison() -> pd.DataFrame:
    selected = pd.read_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_predictions.parquet"
    )
    selected = selected[
        ~selected["specification"].eq("SELECTED_WR_PARTICIPATION_FALLBACK")
    ][["player_id", "position", "season", "actual_ppg", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "v2_prediction"}
    )

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

    paired = selected.merge(
        v1, on=["player_id", "position", "season"], validate="one_to_one"
    ).merge(
        baseline, on=["player_id", "position", "season"], validate="one_to_one"
    )

    rows: list[dict[str, object]] = []
    for position, group in paired.groupby("position", sort=True):
        model_metrics = {
            "V2": _metrics(group["actual_ppg"], group["v2_prediction"]),
            "V1": _metrics(group["actual_ppg"], group["v1_prediction"]),
            "Previous-season PPG": _metrics(
                group["actual_ppg"], group["previous_season_ppg_prediction"]
            ),
        }
        baseline_mae = model_metrics["Previous-season PPG"]["mae"]
        v1_mae = model_metrics["V1"]["mae"]
        for model_name in ("Previous-season PPG", "V1", "V2"):
            metric = model_metrics[model_name]
            rows.append(
                {
                    "position": position,
                    "evaluation_seasons": (
                        f"{int(group['season'].min())}-{int(group['season'].max())}"
                    ),
                    "player_seasons": len(group),
                    "model": model_name,
                    "mae_ppg": metric["mae"],
                    "rmse_ppg": metric["rmse"],
                    "spearman": metric["spearman"],
                    "mae_improvement_vs_previous_ppg": baseline_mae - metric["mae"],
                    "mae_percent_improvement_vs_previous_ppg": (
                        100 * (baseline_mae - metric["mae"]) / baseline_mae
                    ),
                    "v2_mae_improvement_vs_v1": (
                        v1_mae - metric["mae"] if model_name == "V2" else np.nan
                    ),
                }
            )

    result = pd.DataFrame(rows)
    SELECTED_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        SELECTED_REPORT_TABLES_DIR / "v2_v1_previous_ppg_comparison.csv", index=False
    )

    display = result.copy()
    for column in (
        "mae_ppg",
        "rmse_ppg",
        "spearman",
        "mae_improvement_vs_previous_ppg",
        "mae_percent_improvement_vs_previous_ppg",
        "v2_mae_improvement_vs_v1",
    ):
        display[column] = display[column].round(4)
    notes = [
        "V2, V1, AND PREVIOUS-SEASON PPG COMPARISON",
        "============================================================",
        "",
        "All three methods are evaluated on identical player-season rows within",
        "each position. Lower MAE and RMSE are better; higher Spearman is better.",
        "Coverage varies because the promoted RB market and WR participation",
        "features are not available for the full 2012-2025 validation period.",
        "",
        display.to_string(index=False),
        "",
    ]
    (SELECTED_REPORTS_DIR / "v2_v1_previous_ppg_comparison.txt").write_text(
        "\n".join(notes), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    build_comparison()
