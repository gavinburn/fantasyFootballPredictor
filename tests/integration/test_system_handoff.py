"""Completeness checks for the standalone feature-ideation handoff."""

import json

import pandas as pd

from src.config import ATTEMPT21_REPORT_TABLES_DIR, REPOSITORY_ROOT
from src.system_handoff import FEATURE_DEFINITIONS


def test_every_candidate_feature_has_a_definition() -> None:
    candidates = json.loads(
        (ATTEMPT21_REPORT_TABLES_DIR / "candidate_feature_sets.json").read_text()
    )
    features = {
        feature
        for position in candidates.values()
        for specification in position.values()
        for feature in specification
    }
    assert features.issubset(FEATURE_DEFINITIONS)


def test_same_cohort_comparison_contains_all_models_and_positions() -> None:
    comparison = pd.read_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_same_cohort_baseline_comparison.csv"
    )
    assert set(comparison.position) == {"QB", "RB", "WR", "TE"}
    assert set(comparison.model) == {
        "attempt21_adaptive",
        "frozen_same_cohort_v1",
        "original_v1",
        "previous_season_ppg",
        "historical_position_mean",
    }
    assert len(comparison) == 20


def test_handoff_contains_required_sections() -> None:
    text = (REPOSITORY_ROOT / "fantasy_football_model_feature_handoff.txt").read_text()
    for heading in (
        "RAW AND EXTERNAL INPUTS",
        "COMPLETE MODELED FEATURE CATALOG",
        "EXACT POSITION-SPECIFIC CANDIDATE SPECIFICATIONS",
        "DATA AND MODEL ISSUES FIXED",
        "CURRENT SAME-COHORT RESULTS",
        "HIGHEST-PRIORITY NEW FEATURE DIRECTIONS",
    ):
        assert heading in text
