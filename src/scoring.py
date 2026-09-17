"""Standard, non-PPR fantasy football scoring functions."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.column_mapping import WEEKLY_PLAYER_STATS_COLUMNS

STANDARD_FANTASY_POINTS_COLUMN = "fantasy_points_standard"


class ScoringError(ValueError):
    """Raised when a stat line cannot be scored safely."""


@dataclass(frozen=True)
class ScoringConfig:
    """Named weights for standard, non-PPR scoring."""

    passing_yards: float = 1.0 / 25.0
    passing_touchdowns: float = 4.0
    interceptions: float = -2.0
    rushing_yards: float = 1.0 / 10.0
    rushing_touchdowns: float = 6.0
    receiving_yards: float = 1.0 / 10.0
    receiving_touchdowns: float = 6.0
    receptions: float = 0.0
    two_point_conversions: float = 2.0
    fumbles_lost: float = -2.0


STANDARD_SCORING = ScoringConfig()

SCORING_SEMANTICS = (
    "passing_yards",
    "passing_touchdowns",
    "interceptions",
    "rushing_yards",
    "rushing_touchdowns",
    "receiving_yards",
    "receiving_touchdowns",
    "passing_two_point_conversions",
    "rushing_two_point_conversions",
    "receiving_two_point_conversions",
    "fumbles_lost",
)


def score_stat_line(
    *,
    passing_yards: float = 0.0,
    passing_touchdowns: float = 0.0,
    interceptions: float = 0.0,
    rushing_yards: float = 0.0,
    rushing_touchdowns: float = 0.0,
    receiving_yards: float = 0.0,
    receiving_touchdowns: float = 0.0,
    receptions: float = 0.0,
    passing_two_point_conversions: float = 0.0,
    rushing_two_point_conversions: float = 0.0,
    receiving_two_point_conversions: float = 0.0,
    fumbles_lost: float = 0.0,
    config: ScoringConfig = STANDARD_SCORING,
) -> float:
    """Score one stat line using standard scoring without PPR points."""

    two_point_conversions = (
        passing_two_point_conversions
        + rushing_two_point_conversions
        + receiving_two_point_conversions
    )
    return float(
        passing_yards * config.passing_yards
        + passing_touchdowns * config.passing_touchdowns
        + interceptions * config.interceptions
        + rushing_yards * config.rushing_yards
        + rushing_touchdowns * config.rushing_touchdowns
        + receiving_yards * config.receiving_yards
        + receiving_touchdowns * config.receiving_touchdowns
        + receptions * config.receptions
        + two_point_conversions * config.two_point_conversions
        + fumbles_lost * config.fumbles_lost
    )


def _numeric_stat(column: str) -> pl.Expr:
    """Treat null legitimate counting stats as zero and require numeric values."""

    return pl.col(column).cast(pl.Float64, strict=True).fill_null(0.0)


def calculate_weekly_fantasy_points(
    frame: pl.DataFrame,
    *,
    output_column: str = STANDARD_FANTASY_POINTS_COLUMN,
    config: ScoringConfig = STANDARD_SCORING,
) -> pl.DataFrame:
    """Add standard fantasy points to nflreadpy weekly player-stat rows.

    ``fumbles_lost_total`` is deliberately used as the one comprehensive fumble
    field. The play-type lost-fumble columns are not also added. nflreadpy 0.1.5
    provides passing, rushing, and receiving two-point conversion counts, so all
    three reliable components are included.
    """

    selected_columns = {
        semantic: WEEKLY_PLAYER_STATS_COLUMNS[semantic]
        for semantic in SCORING_SEMANTICS
    }
    missing = [
        column for column in selected_columns.values() if column not in frame.columns
    ]
    if missing:
        raise ScoringError(
            "Cannot calculate weekly fantasy points; missing stat columns: "
            + ", ".join(missing)
        )

    stat = {
        semantic: _numeric_stat(column) for semantic, column in selected_columns.items()
    }
    two_point_conversions = (
        stat["passing_two_point_conversions"]
        + stat["rushing_two_point_conversions"]
        + stat["receiving_two_point_conversions"]
    )
    points = (
        stat["passing_yards"] * config.passing_yards
        + stat["passing_touchdowns"] * config.passing_touchdowns
        + stat["interceptions"] * config.interceptions
        + stat["rushing_yards"] * config.rushing_yards
        + stat["rushing_touchdowns"] * config.rushing_touchdowns
        + stat["receiving_yards"] * config.receiving_yards
        + stat["receiving_touchdowns"] * config.receiving_touchdowns
        + two_point_conversions * config.two_point_conversions
        + stat["fumbles_lost"] * config.fumbles_lost
    )
    return frame.with_columns(points.alias(output_column))
