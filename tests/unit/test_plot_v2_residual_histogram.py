import pandas as pd

from src.plot_v2_residual_histogram import summarize_residuals


def test_residual_summary_preserves_signed_error() -> None:
    residuals = pd.DataFrame(
        {
            "position": ["QB", "QB", "QB"],
            "prediction_error_ppg": [-2.0, 1.0, 4.0],
        }
    )
    summary = summarize_residuals(residuals).iloc[0]
    assert summary["player_seasons"] == 3
    assert summary["mean_error_ppg"] == 1.0
    assert summary["median_error_ppg"] == 1.0
    assert summary["minimum_error_ppg"] == -2.0
    assert summary["maximum_error_ppg"] == 4.0
