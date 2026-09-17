"""Contracts for the isolated TE receiving-role experiment."""

import numpy as np
import pandas as pd

from src.config import TE_ROLE_PROCESSED_DATA_DIR
from src.te_role_experiment import (
    FEATURE_MANIFESTS,
    PARTICIPATION_RATE,
    SNAP_SHARE,
    TARGET_RATE,
    TARGETS_PER_PARTICIPATED,
)


def test_te_role_table_has_point_in_time_features_and_target_label() -> None:
    table = pd.read_parquet(
        TE_ROLE_PROCESSED_DATA_DIR / "te_role_modeling_dataset.parquet"
    )
    assert not table.duplicated(["player_id", "prediction_season"]).any()
    assert table["target_targets_per_game"].notna().all()
    observed = table[[PARTICIPATION_RATE, TARGETS_PER_PARTICIPATED]].notna().all(axis=1)
    expected = (
        table.loc[observed, PARTICIPATION_RATE]
        * table.loc[observed, TARGETS_PER_PARTICIPATED]
    )
    assert np.allclose(table.loc[observed, TARGET_RATE], expected)
    assert table[SNAP_SHARE].dropna().between(0, 1).all()


def test_te_role_manifests_do_not_claim_unavailable_route_data() -> None:
    features = {feature for manifest in FEATURE_MANIFESTS.values() for feature in manifest}
    assert "route_participation_rate" not in features
    assert "targets_per_route_run" not in features
    assert "inline_blocking_share" not in features


def test_te_role_manifests_have_no_duplicate_features() -> None:
    for features in FEATURE_MANIFESTS.values():
        assert len(features) == len(set(features))
