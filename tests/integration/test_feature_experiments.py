"""Contracts for feature experiments 3 through 5."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE35_PROCESSED_DATA_DIR,
    FEATURE35_RAW_DATA_DIR,
    FEATURE35_REPORT_TABLES_DIR,
    POSITIONS,
)
from src.feature_experiment_models import evaluation_start
from src.feature_experiments import candidate_manifests


@pytest.mark.parametrize("position", POSITIONS)
def test_experiment_tables_preserve_veteran_cohort(position: str) -> None:
    table = pd.read_parquet(
        FEATURE35_PROCESSED_DATA_DIR
        / f"{position.lower()}_feature_experiments_dataset.parquet"
    )
    assert table["previous_season_ppg"].notna().all()
    assert not table.duplicated(["player_id", "prediction_season"]).any()


def test_xfp_representation_does_not_include_exact_ppg_dependency() -> None:
    for specifications in candidate_manifests().values():
        if "EXP3_XFP_REPRESENTATION" not in specifications:
            continue
        features = specifications["EXP3_XFP_REPRESENTATION"]
        assert "previous_expected_ppg_xfp" in features
        assert "previous_fpoe_per_game" in features
        assert "previous_season_ppg" not in features


@pytest.mark.parametrize("position", POSITIONS)
def test_candidate_designs_have_no_exact_dependencies(position: str) -> None:
    table = pd.read_parquet(
        FEATURE35_PROCESSED_DATA_DIR
        / f"{position.lower()}_feature_experiments_dataset.parquet"
    )
    for specification, features in candidate_manifests()[position].items():
        start = evaluation_start(position, specification)
        training = table[table["prediction_season"] < start][features]
        filled = training.fillna(training.median(numeric_only=True)).fillna(0.0)
        varying = filled.loc[:, filled.std(ddof=0) > 1e-12]
        standardized = (varying - varying.mean()) / varying.std(ddof=0)
        assert np.linalg.matrix_rank(standardized.to_numpy()) == varying.shape[1], (
            position, specification
        )


def test_share_features_have_valid_ranges() -> None:
    share_columns = [
        "previous_goal_line_carry_share", "previous_redzone_target_share",
        "previous_deep_target_share", "vacated_target_share",
        "vacated_carry_share", "vacated_redzone_touch_share",
        "previous_pass_play_participation_rate",
    ]
    for position in POSITIONS:
        table = pd.read_parquet(
            FEATURE35_PROCESSED_DATA_DIR
            / f"{position.lower()}_feature_experiments_dataset.parquet"
        )
        for column in share_columns:
            observed = table[column].dropna()
            assert observed.between(0, 1).all(), (position, column)


def test_route_and_inline_features_are_not_fabricated() -> None:
    omissions = json.loads(
        (FEATURE35_REPORT_TABLES_DIR / "schema_omissions.json").read_text()
    )
    serialized = json.dumps(candidate_manifests())
    assert "route_participation_rate" not in serialized
    assert "targets_per_route_run" not in serialized
    assert "inline_blocking_share" not in serialized
    assert set(omissions) == {
        "inline_blocking_share", "route_participation_rate", "targets_per_route_run"
    }


def test_frozen_opportunity_version_is_recorded() -> None:
    manifest = json.loads(
        (FEATURE35_RAW_DATA_DIR / "feature_experiments_raw_manifest.json").read_text()
    )
    assert manifest["ffopportunity_model_version"] == "v1.0.0"
