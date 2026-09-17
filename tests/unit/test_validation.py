import polars as pl
import pytest

from src.validation import (
    ValidationError,
    determine_evaluation_start,
    expanding_folds,
    regression_metrics,
)


def feature_table(position, seasons=(2010, 2011, 2012, 2013)):
    rows = []
    for season in seasons:
        for index in range(2):
            rows.append(
                {
                    "player_id": f"{position}{index}",
                    "player_name": f"{position} Player {index}",
                    "position": position,
                    "prediction_season": season,
                    "target_ppg": float(season - 2000 + index),
                    "previous_season_ppg": (
                        None if season == min(seasons) else float(season - 2001 + index)
                    ),
                }
            )
    return pl.DataFrame(rows)


def test_expanding_folds_use_only_earlier_same_position_rows():
    table = feature_table("QB")

    folds = expanding_folds(
        table,
        "QB",
        start_season=2012,
        min_training_rows=1,
    )

    assert [fold.season for fold in folds] == [2012, 2013]
    assert all(fold.train["prediction_season"].max() < fold.season for fold in folds)
    assert all(fold.train["position"].unique().to_list() == ["QB"] for fold in folds)


def test_mixed_position_fold_is_rejected():
    mixed = feature_table("QB").vstack(feature_table("RB"))

    with pytest.raises(ValidationError, match="received positions"):
        expanding_folds(mixed, "QB", start_season=2012, min_training_rows=1)


def test_evaluation_start_requires_all_positions():
    tables = {
        position: feature_table(position) for position in ("QB", "RB", "WR", "TE")
    }

    assert (
        determine_evaluation_start(tables, preferred_start=2012, min_training_rows=1)
        == 2012
    )


def test_regression_metrics_match_simple_manual_example():
    predictions = pl.DataFrame(
        {
            "player_id": ["a", "b"],
            "actual_ppg": [10.0, 20.0],
            "predicted_ppg": [12.0, 18.0],
        }
    )

    metrics = regression_metrics(predictions, top_n=1)

    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)
    assert metrics["r_squared"] == pytest.approx(0.84)
    assert metrics["spearman_rank_correlation"] == pytest.approx(1.0)
    assert metrics["top_n_overlap"] == 1
