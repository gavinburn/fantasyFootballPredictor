"""Validated semantic mappings for the inspected nflreadpy schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class ColumnMappingError(ValueError):
    """Raised when an inspected dataset no longer matches its expected schema."""


# These exact selections come from the Phase 2 inspection of nflreadpy 0.1.5 data.
WEEKLY_PLAYER_STATS_COLUMNS: dict[str, str] = {
    "player_id": "player_id",
    "player_name": "player_display_name",
    "season": "season",
    "week": "week",
    "season_type": "season_type",
    "position": "position",
    "position_group": "position_group",
    "recent_team": "team",
    "completions": "completions",
    "passing_attempts": "attempts",
    "passing_yards": "passing_yards",
    "passing_touchdowns": "passing_tds",
    "interceptions": "passing_interceptions",
    "rushing_attempts": "carries",
    "rushing_yards": "rushing_yards",
    "rushing_touchdowns": "rushing_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_touchdowns": "receiving_tds",
    "passing_two_point_conversions": "passing_2pt_conversions",
    "rushing_two_point_conversions": "rushing_2pt_conversions",
    "receiving_two_point_conversions": "receiving_2pt_conversions",
    "fumbles_lost": "fumbles_lost_total",
}

PLAYER_MASTER_COLUMNS: dict[str, str] = {
    "player_id": "gsis_id",
    "player_name": "display_name",
    "position": "position",
    "position_group": "position_group",
    "recent_team": "latest_team",
    "birth_date": "birth_date",
    "years_of_experience": "years_of_experience",
    "draft_round": "draft_round",
    "draft_pick": "draft_pick",
}

ROSTER_COLUMNS: dict[str, str] = {
    "player_id": "gsis_id",
    "player_name": "full_name",
    "season": "season",
    "position": "position",
    "team": "team",
    "years_of_experience": "years_exp",
    "draft_pick": "draft_number",
}


def validate_mapping(
    dataset_name: str,
    available_columns: Sequence[str],
    mapping: Mapping[str, str],
) -> dict[str, str]:
    """Return a mapping only when every selected physical column is available."""

    available = set(available_columns)
    missing = {
        semantic: physical
        for semantic, physical in mapping.items()
        if physical not in available
    }
    if missing:
        details = ", ".join(
            f"{semantic} -> {physical}" for semantic, physical in missing.items()
        )
        raise ColumnMappingError(
            f"{dataset_name} is missing mapped semantic fields: {details}"
        )
    return dict(mapping)
