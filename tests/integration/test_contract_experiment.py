"""Contracts for the leakage-conservative salary-cap ablation."""

import json

import pandas as pd

from src.config import (
    CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR,
    CONTRACT_EXPERIMENT_RAW_DATA_DIR,
    CONTRACT_EXPERIMENT_REPORT_TABLES_DIR,
)
from src.contract_features import (
    FOCAL_CONTRACT_FEATURES,
    LAGGED_CAP_FEATURES,
    ROOM_CONTRACT_FEATURES,
    TE_WEIGHTED_CONTRACT_FEATURES,
    _select_prior_contracts,
)
from src.contract_models import BASE_REFIT, candidate_manifests
from src.download_contract_data import CONTRACT_FILE
from src.selected_feature_models import (
    SELECTED_SPECIFICATION,
    selected_feature_manifests,
)


def test_contract_selector_excludes_prediction_year_signings() -> None:
    context = pd.DataFrame({
        "player_id": ["p1"], "prediction_season": [2020],
        "position": ["TE"], "target_team": ["AAA"],
    })
    contracts = pd.DataFrame({
        "gsis_id": ["p1", "p1"], "year_signed": [2019, 2020],
        "years": [2, 4], "value": [10.0, 100.0], "apy": [5.0, 25.0],
        "guaranteed": [3.0, 80.0], "apy_cap_pct": [0.03, 0.15],
        "inflated_apy": [6.0, 25.0], "draft_year": [2019, 2019],
        "team": ["AAA", "AAA"], "position": ["TE", "TE"],
    })
    selected, exclusions = _select_prior_contracts(context, contracts)
    assert selected.iloc[0]["year_signed"] == 2019
    assert exclusions.iloc[0]["same_season_contract_rows_excluded"] == 1


def test_frozen_contract_source_omits_current_status() -> None:
    source = pd.read_parquet(CONTRACT_EXPERIMENT_RAW_DATA_DIR / CONTRACT_FILE)
    assert "is_active" not in source.columns


def test_contract_features_are_unique_and_cutoff_safe() -> None:
    features = pd.read_parquet(
        CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR / "contract_context_features.parquet"
    )
    assert not features.duplicated(["player_id", "prediction_season"]).any()
    matched = features["year_signed"].notna()
    assert (
        features.loc[matched, "year_signed"]
        <= features.loc[matched, "prediction_season"] - 1
    ).all()


def test_contract_candidates_layer_on_current_position_bases() -> None:
    manifests = candidate_manifests()
    selected = selected_feature_manifests()
    for position in ("QB", "RB", "WR"):
        assert manifests[position][BASE_REFIT] == selected[position][SELECTED_SPECIFICATION]
    assert manifests["TE"]["BASE_PLUS_FOCAL_CONTRACT"][-len(FOCAL_CONTRACT_FEATURES):] == FOCAL_CONTRACT_FEATURES
    assert manifests["TE"]["BASE_PLUS_PRIOR_SEASON_CAP"][-len(LAGGED_CAP_FEATURES):] == LAGGED_CAP_FEATURES
    assert manifests["TE"]["BASE_PLUS_CONTRACT_ROOM"][-len(ROOM_CONTRACT_FEATURES):] == ROOM_CONTRACT_FEATURES
    assert manifests["TE"]["BASE_PLUS_RECEIVING_WEIGHTED_SALARY"][-len(TE_WEIGHTED_CONTRACT_FEATURES):] == TE_WEIGHTED_CONTRACT_FEATURES


def test_saved_contract_manifest_matches_code() -> None:
    saved = json.loads(
        (CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "candidate_feature_sets.json").read_text()
    )
    assert saved["full_refit"] == candidate_manifests()
