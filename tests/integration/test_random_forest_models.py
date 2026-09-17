"""Contracts for the isolated random-forest experiment."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from src.config import (
    RANDOM_FOREST_PREDICTIONS_DATA_DIR,
    RANDOM_FOREST_REPORT_TABLES_DIR,
    RANDOM_STATE,
)
from src.random_forest_models import (
    FINAL_TREES,
    MAX_DEPTHS,
    MIN_SAMPLES_LEAF,
    PARAMETER_GRID,
    TUNING_TREES,
    build_forest_pipeline,
)


def test_forest_grid_is_constrained_and_reproducible() -> None:
    assert len(PARAMETER_GRID) == 36
    assert max(depth for depth in MAX_DEPTHS if depth is not None) == 6
    assert min(MIN_SAMPLES_LEAF) == 5
    assert TUNING_TREES == 150
    assert FINAL_TREES == 500
    for params in PARAMETER_GRID:
        pipeline = build_forest_pipeline(params)
        assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
        forest = pipeline.named_steps["regressor"]
        assert isinstance(forest, RandomForestRegressor)
        assert forest.random_state == RANDOM_STATE
        assert forest.bootstrap is True
        assert "scaler" not in pipeline.named_steps


def test_saved_forest_predictions_preserve_v2_cohorts() -> None:
    predictions = pd.read_parquet(
        RANDOM_FOREST_PREDICTIONS_DATA_DIR / "random_forest_predictions.parquet"
    )
    assert not predictions.duplicated(
        ["player_id", "position", "season", "specification"]
    ).any()
    direct = predictions[predictions["specification"].eq("RANDOM_FOREST_DIRECT")]
    assert direct.groupby("position").size().to_dict() == {
        "QB": 512, "RB": 410, "TE": 968, "WR": 929,
    }


def test_forest_promotion_decisions_match_metrics() -> None:
    comparison = pd.read_csv(
        RANDOM_FOREST_REPORT_TABLES_DIR / "comparison_vs_v2.csv"
    )
    promoted = comparison["decision"].eq("PROMOTE")
    assert (
        promoted
        == (
            comparison["mae_pass"]
            & comparison["season_consistency_pass"]
            & comparison["spearman_pass"]
        )
    ).all()
