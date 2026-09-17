import polars as pl
import pytest

from src.evaluate import (
    MODEL_NAME,
    POSITION_MEAN_BASELINE,
    PREVIOUS_BASELINE,
    aggregate_comparison,
    calculate_metrics_by_season,
    coefficient_stability,
)


def prediction_rows():
    rows = []
    actual = [10.0, 20.0]
    model_predictions = {
        PREVIOUS_BASELINE: [12.0, 18.0],
        POSITION_MEAN_BASELINE: [15.0, 15.0],
        MODEL_NAME: [11.0, 19.0],
    }
    for model, predicted in model_predictions.items():
        for player, (actual_ppg, predicted_ppg) in enumerate(
            zip(actual, predicted, strict=True)
        ):
            rows.append(
                {
                    "player_id": f"p{player}",
                    "position": "QB",
                    "season": 2020,
                    "model_name": model,
                    "actual_ppg": actual_ppg,
                    "predicted_ppg": predicted_ppg,
                    "absolute_error": abs(predicted_ppg - actual_ppg),
                    "actual_rank": float(2 - player),
                    "predicted_rank": float(2 - player),
                }
            )
    return pl.DataFrame(rows)


def test_season_metrics_report_mae_rmse_ranking_and_top_n():
    metrics = calculate_metrics_by_season(prediction_rows())
    linear = metrics.filter(pl.col("model_name") == MODEL_NAME).row(0, named=True)

    assert linear["MAE"] == pytest.approx(1.0)
    assert linear["RMSE"] == pytest.approx(1.0)
    assert linear["Spearman_rank_correlation"] == pytest.approx(1.0)
    assert linear["Top_N_overlap_percentage"] == pytest.approx(100.0)
    assert linear["number_of_predictions"] == 2


def test_aggregate_improvement_is_baseline_mae_minus_linear_mae():
    predictions = prediction_rows()
    metrics = calculate_metrics_by_season(predictions)
    exclusions = pl.DataFrame(
        schema={
            "player_id": pl.String,
            "position": pl.String,
            "prediction_season": pl.Int64,
            "exclusion_reason": pl.String,
        }
    )

    comparison = aggregate_comparison(predictions, metrics, exclusions)
    linear = comparison.filter(pl.col("model_name") == MODEL_NAME).row(0, named=True)

    assert linear["baseline_MAE"] == pytest.approx(2.0)
    assert linear["MAE_improvement_over_previous_PPG_baseline"] == pytest.approx(1.0)
    assert linear[
        "percentage_MAE_improvement_over_previous_PPG_baseline"
    ] == pytest.approx(50.0)
    assert linear["seasons_beating_previous_PPG_baseline"] == 1


def test_coefficient_stability_counts_adjacent_sign_changes():
    coefficients = pl.DataFrame(
        {
            "position": ["QB"] * 4,
            "feature": ["x"] * 4,
            "season": [2012, 2013, 2014, 2015],
            "original_unit_coefficient": [1.0, -1.0, -2.0, 2.0],
        }
    )

    stability = coefficient_stability(coefficients).row(0, named=True)

    assert stability["sign_changes"] == 2
    assert stability["coefficient_minimum"] == -2.0
    assert stability["coefficient_maximum"] == 2.0
    assert stability["unstable_coefficient"]
