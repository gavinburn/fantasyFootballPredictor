"""Contracts for the isolated gradient-boosting experiment."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from src.config import (
    GRADIENT_BOOSTING_PREDICTIONS_DATA_DIR,
    GRADIENT_BOOSTING_REPORT_TABLES_DIR,
    RANDOM_STATE,
)
from src.gradient_boosting_models import (
    LOSSES,
    MAX_LEAF_NODES,
    PARAMETER_GRID,
    build_boosting_pipeline,
)


def test_boosting_grid_is_constrained_and_reproducible() -> None:
    assert len(PARAMETER_GRID) == 48
    assert max(MAX_LEAF_NODES) == 15
    assert set(LOSSES) == {"squared_error", "absolute_error"}
    for params in PARAMETER_GRID:
        pipeline = build_boosting_pipeline(params)
        model = pipeline.named_steps["regressor"]
        assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
        assert isinstance(model, HistGradientBoostingRegressor)
        assert model.random_state == RANDOM_STATE
        assert model.early_stopping is False
        assert model.l2_regularization == 1.0
        assert "scaler" not in pipeline.named_steps


def test_saved_boosting_predictions_preserve_v2_cohorts() -> None:
    predictions = pd.read_parquet(
        GRADIENT_BOOSTING_PREDICTIONS_DATA_DIR
        / "gradient_boosting_predictions.parquet"
    )
    assert not predictions.duplicated(
        ["player_id", "position", "season", "specification"]
    ).any()
    direct = predictions[
        predictions["specification"].eq("GRADIENT_BOOSTING_DIRECT")
    ]
    assert direct.groupby("position").size().to_dict() == {
        "QB": 512, "RB": 410, "TE": 968, "WR": 929,
    }


def test_boosting_promotion_decisions_match_metrics() -> None:
    comparison = pd.read_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "comparison_vs_v2.csv"
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
