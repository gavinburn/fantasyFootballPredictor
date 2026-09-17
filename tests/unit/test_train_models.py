import numpy as np
import polars as pl
import pytest

from src.build_features import POSITIONS, V1_FEATURES
from src.train_models import (
    ModelTrainingError,
    coefficients_in_original_units,
    fit_position_pipeline,
    generate_linear_validation,
)


def modeling_table(position, seasons=None, players=3):
    if seasons is None:
        seasons = range(2010, 2014)
    rows = []
    features = V1_FEATURES[position]
    for season in seasons:
        for player in range(players):
            row = {
                "player_id": f"{position}{player}",
                "player_name": f"{position} Player {player}",
                "position": position,
                "prediction_season": season,
                "target_ppg": 5.0 + player + (season - 2010) * 0.5,
                "previous_season_ppg": (
                    None if season == min(seasons) else 4.5 + player
                ),
            }
            for index, feature in enumerate(features):
                row.setdefault(
                    feature,
                    None
                    if season == min(seasons) and feature.startswith("previous_")
                    else float(index + player + season - 2009),
                )
            rows.append(row)
    return pl.DataFrame(rows)


def test_original_unit_coefficients_reproduce_pipeline_prediction():
    table = modeling_table("QB")
    pipeline = fit_position_pipeline(table, "QB")
    features = V1_FEATURES["QB"]
    complete_row = table.filter(pl.col("prediction_season") == 2012).head(1)

    predicted = pipeline.predict(complete_row.select(features).to_numpy())[0]
    intercept, coefficients = coefficients_in_original_units(pipeline)
    reconstructed = intercept + np.dot(
        coefficients, complete_row.select(features).to_numpy()[0]
    )

    assert reconstructed == pytest.approx(predicted)


def test_position_model_rejects_mixed_training_rows():
    mixed = modeling_table("QB").vstack(
        modeling_table("QB").with_columns(pl.lit("RB").alias("position"))
    )

    with pytest.raises(ModelTrainingError, match="another position"):
        fit_position_pipeline(mixed, "QB")


def test_linear_validation_keeps_positions_and_folds_separate():
    tables = {position: modeling_table(position) for position in POSITIONS}

    predictions, metrics, coefficients = generate_linear_validation(
        tables,
        start_season=2012,
        min_training_rows=1,
    )

    assert predictions["position"].n_unique() == 4
    assert predictions["season"].unique().sort().to_list() == [2012, 2013]
    assert metrics.filter(pl.col("training_end_season") >= pl.col("season")).is_empty()
    assert (
        coefficients.group_by(["position", "season"])
        .agg(pl.col("feature").n_unique().alias("features"))
        .join(
            pl.DataFrame(
                {
                    "position": list(POSITIONS),
                    "expected": [len(V1_FEATURES[position]) for position in POSITIONS],
                }
            ),
            on="position",
        )
        .filter(pl.col("features") != pl.col("expected"))
        .is_empty()
    )
