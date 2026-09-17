from pathlib import Path

import polars as pl
import pytest

from src.column_mapping import ColumnMappingError, validate_mapping
from src.inspect_data import InspectionError, find_mandatory_paths, inspect_frame


def test_validate_mapping_reports_missing_semantic_field():
    with pytest.raises(ColumnMappingError, match="player_id -> player_id"):
        validate_mapping("weekly", ["season"], {"player_id": "player_id"})


def test_find_mandatory_paths_uses_latest_dated_roster(tmp_path):
    (tmp_path / "player_weekly_stats_2006_2025.parquet").touch()
    (tmp_path / "players.parquet").touch()
    older = tmp_path / "rosters_2026_2026-07-24.parquet"
    latest = tmp_path / "rosters_2026_2026-07-25.parquet"
    older.touch()
    latest.touch()

    paths = find_mandatory_paths(tmp_path)

    assert paths["roster_2026"] == latest


def test_find_mandatory_paths_requires_roster(tmp_path):
    with pytest.raises(InspectionError, match="No 2026 roster snapshot"):
        find_mandatory_paths(tmp_path)


def test_inspect_frame_includes_required_sections(tmp_path):
    path = Path(tmp_path) / "sample.parquet"
    pl.DataFrame(
        {
            "player_id": ["a", "b"],
            "season": [2024, 2025],
            "position": ["QB", "RB"],
            "passing_yards": [100, None],
        }
    ).write_parquet(path)

    report = inspect_frame(
        "sample",
        path,
        {"player_id": "player_id", "season": "season"},
    )

    assert "FIRST FIVE ROWS" in report
    assert "NULL PERCENTAGE BY COLUMN" in report
    assert '"passing_yards"' in report
    assert '"player_id": "player_id"' in report
