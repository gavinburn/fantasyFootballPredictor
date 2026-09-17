import polars as pl

from src.build_features import V1_FEATURES, build_feature_tables
from tests.unit.test_feature_shifting import historical_rows, player_master


def _model_inputs(table, season):
    return (
        table.filter(pl.col("prediction_season") == season)
        .select(V1_FEATURES["QB"])
        .row(0, named=True)
    )


def test_changing_current_target_does_not_change_current_inputs():
    original = historical_rows()
    changed = original.with_columns(
        pl.when((pl.col("player_id") == "qb") & (pl.col("season") == 2022))
        .then(999.0)
        .otherwise(pl.col("fantasy_points_per_game"))
        .alias("fantasy_points_per_game")
    )

    original_qb = build_feature_tables(original, player_master())["QB"]
    changed_qb = build_feature_tables(changed, player_master())["QB"]

    assert _model_inputs(original_qb, 2022) == _model_inputs(changed_qb, 2022)
    assert (
        original_qb.filter(pl.col("prediction_season") == 2022)["target_ppg"].item()
        != changed_qb.filter(pl.col("prediction_season") == 2022)["target_ppg"].item()
    )


def test_changing_future_data_does_not_change_earlier_inputs():
    original = historical_rows()
    changed = original.with_columns(
        pl.when((pl.col("player_id") == "qb") & (pl.col("season") == 2023))
        .then(999.0)
        .otherwise(pl.col("fantasy_points_per_game"))
        .alias("fantasy_points_per_game")
    )

    original_qb = build_feature_tables(original, player_master())["QB"]
    changed_qb = build_feature_tables(changed, player_master())["QB"]

    assert _model_inputs(original_qb, 2022) == _model_inputs(changed_qb, 2022)
