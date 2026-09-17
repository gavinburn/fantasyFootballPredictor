"""Contracts for the position-specific selected feature set."""

from src.attempt3_features import feature_manifests as attempt3_manifests
from src.selected_feature_models import (
    SELECTED_SPECIFICATION,
    TE_SELECTED_SPECIFICATION,
    WR_FALLBACK_SPECIFICATION,
    selected_feature_manifests,
)


def test_selected_position_specific_contracts() -> None:
    selected = selected_feature_manifests()
    attempt3 = attempt3_manifests()

    assert selected["QB"][SELECTED_SPECIFICATION] == attempt3["QB"][
        "STRUCTURAL_CLEANUP"
    ]
    rb = selected["RB"][SELECTED_SPECIFICATION]
    assert rb[:-1] == attempt3["RB"]["VETERAN_ALIGNED_BASE"]
    assert rb[-1] == "market_implied_ppg"
    assert "preseason_ecr_available" not in rb
    assert "three_year_durability_index" not in rb

    wr = selected["WR"][SELECTED_SPECIFICATION]
    fallback = selected["WR"][WR_FALLBACK_SPECIFICATION]
    assert wr[-2:] == [
        "previous_pass_play_participation_rate",
        "previous_targets_per_pass_play_participated",
    ]
    assert fallback[-1] == "previous_pass_play_participation_rate"
    assert "previous_targets_per_pass_play_participated" not in fallback
    assert "participation_feature_available" not in wr

    te = selected["TE"][TE_SELECTED_SPECIFICATION]
    assert te[-2:] == [
        "effective_receiving_competitor_count",
        "probability_weighted_competitor_targets_per_game",
    ]


def test_selected_manifests_have_unique_features() -> None:
    for specifications in selected_feature_manifests().values():
        for features in specifications.values():
            assert len(features) == len(set(features))
