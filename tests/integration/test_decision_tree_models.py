"""Contracts for the isolated post-V2 decision-tree experiment."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    DECISION_TREE_PREDICTIONS_DATA_DIR,
    DECISION_TREE_REPORT_TABLES_DIR,
    RANDOM_STATE,
)
from src.decision_tree_models import (
    CCP_ALPHAS,
    MAX_DEPTHS,
    MIN_SAMPLES_LEAF,
    PARAMETER_GRID,
    TE_RESIDUAL_CAP,
    _features,
    _main_v2_predictions,
    build_tree_pipeline,
)
from src.selected_feature_models import WR_FALLBACK_SPECIFICATION


def test_tree_grid_is_constrained_and_reproducible() -> None:
    assert len(PARAMETER_GRID) == 64
    assert max(MAX_DEPTHS) == 5
    assert min(MIN_SAMPLES_LEAF) == 5
    assert min(CCP_ALPHAS) >= 0
    for params in PARAMETER_GRID:
        pipeline = build_tree_pipeline(params)
        assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
        tree = pipeline.named_steps["regressor"]
        assert isinstance(tree, DecisionTreeRegressor)
        assert tree.random_state == RANDOM_STATE
        assert "scaler" not in pipeline.named_steps


def test_tree_uses_exact_selected_feature_manifests() -> None:
    features = _features()
    assert set(features) == {"QB", "RB", "WR", "TE"}
    assert "market_implied_ppg" in features["RB"]
    assert "previous_pass_play_participation_rate" in features["WR"]
    assert "effective_receiving_competitor_count" in features["TE"]
    assert TE_RESIDUAL_CAP == 0.75


def test_v2_reference_excludes_wr_fallback_duplicate() -> None:
    reference = _main_v2_predictions()
    assert WR_FALLBACK_SPECIFICATION not in set(reference["specification"])
    assert not reference.duplicated(["player_id", "position", "season"]).any()
    assert reference.groupby("position").size().to_dict() == {
        "QB": 512, "RB": 410, "TE": 968, "WR": 929,
    }


def test_saved_tree_predictions_preserve_v2_cohorts() -> None:
    predictions = pd.read_parquet(
        DECISION_TREE_PREDICTIONS_DATA_DIR / "decision_tree_predictions.parquet"
    )
    assert not predictions.duplicated(
        ["player_id", "position", "season", "specification"]
    ).any()
    direct = predictions[predictions["specification"].eq("DECISION_TREE_DIRECT")]
    assert direct.groupby("position").size().to_dict() == {
        "QB": 512, "RB": 410, "TE": 968, "WR": 929,
    }


def test_saved_tree_candidates_are_not_promoted_when_they_fail_v2() -> None:
    comparison = pd.read_csv(
        DECISION_TREE_REPORT_TABLES_DIR / "comparison_vs_v2.csv"
    )
    assert set(comparison["decision"]) == {"REJECT"}
    assert (comparison["mae_improvement_vs_v2"] < 0).all()
