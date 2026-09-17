"""Contracts for the isolated Attempt 3 feature and modeling design."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.attempt3_features import QB_MISSING_FLAGS, feature_manifests
from src.attempt3_models import _apply_market_calibration
from src.config import (
    ATTEMPT3_PROCESSED_DATA_DIR,
    ATTEMPT3_RAW_DATA_DIR,
    POSITIONS,
)
from src.download_attempt3_data import RANKINGS_FILE


@pytest.mark.parametrize("position", POSITIONS)
def test_veteran_tables_have_no_missing_previous_ppg(position: str) -> None:
    table = pd.read_parquet(
        ATTEMPT3_PROCESSED_DATA_DIR
        / f"{position.lower()}_attempt3_modeling_dataset.parquet"
    )
    assert table["previous_season_ppg"].notna().all()
    assert not table.duplicated(["player_id", "prediction_season"]).any()


@pytest.mark.parametrize("position", POSITIONS)
def test_missing_qbr_flags_are_mutually_exclusive(position: str) -> None:
    table = pd.read_parquet(
        ATTEMPT3_PROCESSED_DATA_DIR
        / f"{position.lower()}_attempt3_modeling_dataset.parquet"
    )
    missing = table["projected_qb_previous_qbr"].isna()
    assert (table.loc[missing, QB_MISSING_FLAGS].sum(axis=1) == 1).all()
    assert (table.loc[~missing, QB_MISSING_FLAGS].sum(axis=1) == 0).all()


def test_market_snapshots_respect_cutoff_and_cover_expected_seasons() -> None:
    rankings = pd.read_parquet(ATTEMPT3_RAW_DATA_DIR / RANKINGS_FILE)
    snapshot = pd.to_datetime(rankings["snapshot_date"])
    cutoff = pd.to_datetime(rankings["prediction_season"].astype(str) + "-09-01")
    assert (snapshot <= cutoff).all()
    assert set(rankings["prediction_season"]) == set(range(2020, 2026))


def test_structural_and_market_manifests_avoid_known_exact_dependencies() -> None:
    manifests = feature_manifests()
    for position in POSITIONS:
        structural = manifests[position]["STRUCTURAL_CLEANUP"]
        market = manifests[position]["MARKET_ASSISTED"]
        assert "age_entering_season" not in structural
        assert "previous_season_games_played" not in structural
        assert "age_pre_apex" in structural and "age_post_apex" in structural
        assert "market_historical_ppg_delta" not in market
        assert "preseason_ecr" not in market
        assert len(structural) == len(set(structural))
        assert len(market) == len(set(market))


def test_market_calibration_uses_training_targets_only() -> None:
    train = pd.DataFrame(
        {"preseason_ecr": [1.0, 2.0, 4.0, 8.0] * 5, "target_ppg": [16.0, 12.0, 8.0, 4.0] * 5}
    )
    valid = pd.DataFrame({"preseason_ecr": [1.0, 10.0], "target_ppg": [999.0, 999.0]})
    _, transformed_a, calibration_a = _apply_market_calibration(train, valid)
    changed_valid = valid.assign(target_ppg=-999.0)
    _, transformed_b, calibration_b = _apply_market_calibration(train, changed_valid)
    assert calibration_a == calibration_b
    assert np.allclose(
        transformed_a["market_implied_ppg"], transformed_b["market_implied_ppg"]
    )
