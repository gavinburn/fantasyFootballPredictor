"""Contracts for receiving-adjusted TE roster competition."""

import json

import pandas as pd

from src.config import (
    TE_COMPETITION_PROCESSED_DATA_DIR,
    TE_COMPETITION_REPORT_TABLES_DIR,
)
from src.te_competition_experiment import (
    CANDIDATE_MANIFESTS,
    INTRINSIC_ROLE_FEATURES,
    RESIDUAL_MANIFESTS,
)


def test_role_utility_is_intrinsic_and_non_circular() -> None:
    forbidden = {
        "effective_receiving_competitor_count",
        "max_competitor_receiving_role_probability",
        "probability_weighted_competitor_targets_per_game",
    }
    assert forbidden.isdisjoint(INTRINSIC_ROLE_FEATURES)


def test_crossfit_probabilities_preserve_unknown_competitors() -> None:
    probabilities = pd.read_parquet(
        TE_COMPETITION_PROCESSED_DATA_DIR / "crossfit_role_probabilities.parquet"
    )
    assert not probabilities.duplicated(["player_id", "prediction_season"]).any()
    no_history = probabilities["has_previous_history"].eq(0)
    assert probabilities.loc[
        no_history, "competitor_receiving_role_probability"
    ].isna().all()


def test_competition_features_exclude_direct_focal_role_inputs() -> None:
    features = {
        feature for manifest in CANDIDATE_MANIFESTS.values() for feature in manifest
    }
    assert "competitor_receiving_role_probability" not in features
    assert "previous_pass_play_participation_rate" not in features
    assert "previous_offensive_snap_share" not in features
    assert "previous_targets_per_team_pass_play" not in features
    saved = json.loads(
        (TE_COMPETITION_REPORT_TABLES_DIR / "feature_manifest.json").read_text()
    )
    assert saved == {
        "full_refit_controls": CANDIDATE_MANIFESTS,
        "frozen_v1_residual_adjustments": RESIDUAL_MANIFESTS,
    }
