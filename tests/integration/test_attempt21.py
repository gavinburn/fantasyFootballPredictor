"""Contracts for the pandas-only Attempt 2.1 experiment."""

import ast

import pandas as pd

from src.attempt21_features import (
    build_candidate_sets,
    dependency_audit,
    first_team_disagreement_exclusions,
    load_attempt2_tables,
    position_mismatch_exclusions,
)
from src.attempt21_models import _inner_seasons
from src.config import (
    ATTEMPT21_PROCESSED_DATA_DIR,
    ATTEMPT21_REPORT_TABLES_DIR,
    POSITIONS,
    REPOSITORY_ROOT,
)


def test_attempt21_modules_do_not_import_polars() -> None:
    for name in ("attempt21_features.py", "attempt21_models.py", "attempt21_report.py"):
        tree = ast.parse((REPOSITORY_ROOT / "src" / name).read_text())
        imports = [
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        ]
        assert "polars" not in imports


def test_corrected_candidates_have_no_exact_rank_deficiency() -> None:
    audit = dependency_audit(load_attempt2_tables(), build_candidate_sets())
    assert audit["passed"].all()


def test_inner_folds_are_strictly_earlier_than_outer_season() -> None:
    table = load_attempt2_tables()["QB"]
    for outer in range(2012, 2026):
        inner = _inner_seasons(table[table.prediction_season < outer])
        assert len(inner) >= 3
        assert max(inner) < outer


def test_parquet_round_trip_and_unique_keys() -> None:
    for position in POSITIONS:
        frame = pd.read_parquet(
            ATTEMPT21_PROCESSED_DATA_DIR
            / f"{position.lower()}_attempt21_modeling_dataset.parquet"
        )
        assert frame.index.name not in build_candidate_sets()[position]
        assert not frame.duplicated(["player_id", "prediction_season"]).any()
        assert frame["prediction_season"].min() == 2007


def test_pandas_parity_artifact_passes() -> None:
    audit = pd.read_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_pandas_parity_audit.csv"
    )
    assert audit.passed.all()
    assert audit.max_prediction_difference.max() <= 1e-10


def test_position_mismatch_seasons_are_absent_from_attempt21_tables() -> None:
    exclusions = position_mismatch_exclusions()[["player_id", "prediction_season"]]
    for position in POSITIONS:
        frame = pd.read_parquet(
            ATTEMPT21_PROCESSED_DATA_DIR
            / f"{position.lower()}_attempt21_modeling_dataset.parquet"
        )
        overlap = frame.merge(
            exclusions,
            on=["player_id", "prediction_season"],
            how="inner",
            validate="one_to_one",
        )
        assert overlap.empty


def test_first_team_disagreement_seasons_are_absent_from_attempt21_tables() -> None:
    exclusions = first_team_disagreement_exclusions()[
        ["player_id", "prediction_season"]
    ]
    assert len(exclusions) == 42
    for position in POSITIONS:
        frame = pd.read_parquet(
            ATTEMPT21_PROCESSED_DATA_DIR
            / f"{position.lower()}_attempt21_modeling_dataset.parquet"
        )
        overlap = frame.merge(
            exclusions,
            on=["player_id", "prediction_season"],
            how="inner",
            validate="one_to_one",
        )
        assert overlap.empty
