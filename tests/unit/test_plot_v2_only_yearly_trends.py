import pandas as pd

from src.plot_v2_only_yearly_trends import (
    calculate_v2_yearly_metrics,
    summarize_v2,
)


def test_v2_only_metrics_and_summary() -> None:
    predictions = pd.DataFrame(
        {
            "player_id": ["a", "b", "a", "b"],
            "position": ["QB"] * 4,
            "season": [2020, 2020, 2021, 2021],
            "actual_ppg": [20.0, 10.0, 20.0, 10.0],
            "predicted_ppg": [19.0, 11.0, 17.0, 13.0],
        }
    )
    metrics = calculate_v2_yearly_metrics(predictions)
    summary = summarize_v2(metrics).iloc[0]
    assert metrics["mae_ppg"].tolist() == [1.0, 3.0]
    assert summary["mean_seasonal_mae_ppg"] == 2.0
    assert summary["best_season"] == 2020
    assert summary["worst_season"] == 2021
