"""Contracts for the sealed constrained-TE confirmation study."""

import json

from src.config import (
    SELECTED_REPORT_TABLES_DIR,
)
from src.selected_feature_models import (
    TE_SELECTED_SPECIFICATION,
    selected_feature_manifests,
)
from src.te_constrained_confirmation import (
    CONFIRMATION_SEASONS,
    DISCOVERY_SEASONS,
)


def test_confirmation_and_discovery_seasons_are_disjoint() -> None:
    assert set(CONFIRMATION_SEASONS).isdisjoint(DISCOVERY_SEASONS)


def test_user_approved_practical_tolerance_promotes_te_wrapper() -> None:
    assert TE_SELECTED_SPECIFICATION in selected_feature_manifests()["TE"]
    override = json.loads(
        (
            SELECTED_REPORT_TABLES_DIR / "te_promotion_override.json"
        ).read_text()
    )
    assert override["rollback_model_preserved"] is True
    assert override["overridden_check"] == "featured_tier_bias_pass"
