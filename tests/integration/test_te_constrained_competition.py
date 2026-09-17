"""Contracts for the predeclared constrained TE competition experiment."""

import json

import pandas as pd

from src.config import (
    TE_CONSTRAINED_PREDICTIONS_DATA_DIR,
    TE_CONSTRAINED_REPORT_TABLES_DIR,
)
from src.te_constrained_competition import (
    ASYMMETRIC,
    BURDEN_THRESHOLDS,
    CENTERED_COUNT,
    COMBINED,
    COMPONENTS,
    CORRECTION_CAP,
    FROZEN,
    HIGH_BURDEN,
    MULTI_THRESHOLD,
    PREDECLARED_DEFINITIONS,
    RELATIVE_ROOM,
)


def test_requested_candidates_are_predeclared() -> None:
    assert COMPONENTS == [
        CENTERED_COUNT,
        HIGH_BURDEN,
        RELATIVE_ROOM,
        MULTI_THRESHOLD,
        ASYMMETRIC,
    ]
    assert {FROZEN, *COMPONENTS, COMBINED}.issubset(PREDECLARED_DEFINITIONS)
    assert BURDEN_THRESHOLDS == [2.0, 4.0, 6.0]


def test_saved_manifest_matches_code() -> None:
    saved = json.loads(
        (TE_CONSTRAINED_REPORT_TABLES_DIR / "predeclared_manifest.json").read_text()
    )
    assert saved["definitions"] == PREDECLARED_DEFINITIONS
    assert saved["candidate_order"] == [FROZEN, *COMPONENTS, COMBINED]


def test_frozen_predictions_are_exact_and_corrections_are_bounded() -> None:
    predictions = pd.read_parquet(
        TE_CONSTRAINED_PREDICTIONS_DATA_DIR / "te_constrained_predictions.parquet"
    )
    assert not predictions.duplicated(
        ["player_id", "season", "specification"]
    ).any()
    assert predictions["residual_correction"].abs().max() <= CORRECTION_CAP + 1e-12
    frozen = predictions[predictions["specification"] == FROZEN]
    assert (frozen["predicted_ppg"] == frozen["frozen_v1_predicted_ppg"]).all()


def test_combined_selection_is_inner_fold_only_and_bounded() -> None:
    selections = pd.read_csv(
        TE_CONSTRAINED_REPORT_TABLES_DIR / "combined_component_selections.csv"
    )
    for value in selections["selected_components"]:
        names = [] if value == "NONE" else value.split("|")
        assert len(names) <= 3
        assert set(names).issubset(COMPONENTS)
