import pandas as pd

from src.plot_v2_yearly_trends import calculate_yearly_metrics


def test_yearly_metrics_use_identical_rows_and_fixed_top_n() -> None:
    cohort = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "position": ["QB"] * 3,
            "season": [2020] * 3,
            "actual_ppg": [30.0, 20.0, 10.0],
            "v2_prediction": [29.0, 19.0, 9.0],
            "v1_prediction": [10.0, 20.0, 30.0],
            "previous_season_ppg_prediction": [28.0, 18.0, 8.0],
        }
    )
    metrics = calculate_yearly_metrics(cohort)
    assert set(metrics["method"]) == {
        "Previous-season PPG",
        "V1 linear regression",
        "V2 selected model",
    }
    assert metrics["players"].eq(3).all()
    assert metrics["top_n"].eq(3).all()
    v2 = metrics[metrics["method"].eq("V2 selected model")].iloc[0]
    assert v2["mae_ppg"] == 1.0
    assert v2["top_n_overlap_percentage"] == 100.0
