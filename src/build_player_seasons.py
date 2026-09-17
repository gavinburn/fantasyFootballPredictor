"""Build historical regular-season player PPG targets from weekly statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from src.column_mapping import WEEKLY_PLAYER_STATS_COLUMNS
from src.config import (
    HISTORICAL_END_SEASON,
    HISTORICAL_START_SEASON,
    INTERIM_DATA_DIR,
    MIN_TARGET_GAMES,
    POSITIONS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORT_TABLES_DIR,
)
from src.download_data import PLAYER_STATS_FILENAME
from src.scoring import (
    STANDARD_FANTASY_POINTS_COLUMN,
    calculate_weekly_fantasy_points,
)

AUDIT_FILENAME = "player_seasons_all.parquet"
TARGET_FILENAME = "player_season_targets.parquet"
QUALITY_REPORT_FILENAME = "target_quality_report.txt"
GROUP_KEYS = ("player_id", "season")

# Output name -> inspected nflreadpy semantic field.
SEASON_TOTAL_FIELDS = {
    "completions": "completions",
    "passing_attempts": "passing_attempts",
    "passing_yards": "passing_yards",
    "passing_touchdowns": "passing_touchdowns",
    "passing_interceptions": "interceptions",
    "rushing_attempts": "rushing_attempts",
    "rushing_yards": "rushing_yards",
    "rushing_touchdowns": "rushing_touchdowns",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_touchdowns": "receiving_touchdowns",
    "passing_two_point_conversions": "passing_two_point_conversions",
    "rushing_two_point_conversions": "rushing_two_point_conversions",
    "receiving_two_point_conversions": "receiving_two_point_conversions",
    "fumbles_lost": "fumbles_lost",
}

PER_GAME_FIELDS = (
    "passing_attempts",
    "passing_yards",
    "passing_touchdowns",
    "passing_interceptions",
    "rushing_attempts",
    "rushing_yards",
    "rushing_touchdowns",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_touchdowns",
    "opportunities",
)


class PlayerSeasonError(RuntimeError):
    """Raised when weekly data cannot produce trustworthy player-season targets."""


def _require_columns(frame: pl.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise PlayerSeasonError(
            "Weekly player stats are missing required columns: " + ", ".join(missing)
        )


def build_player_season_tables(
    weekly: pl.DataFrame,
    *,
    min_target_games: int = MIN_TARGET_GAMES,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the audit table and the minimum-games-filtered target table.

    Games played is the number of unique non-null game_id values. Inspection found
    complete game IDs and no duplicate player-game rows among eligible historical
    records. A zero-stat row still counts because game_id is participation evidence.
    """

    if min_target_games < 1:
        raise PlayerSeasonError("min_target_games must be at least 1")

    mapped_columns = {
        WEEKLY_PLAYER_STATS_COLUMNS[semantic]
        for semantic in SEASON_TOTAL_FIELDS.values()
    }
    identity_columns = {
        WEEKLY_PLAYER_STATS_COLUMNS[semantic]
        for semantic in (
            "player_id",
            "player_name",
            "season",
            "week",
            "season_type",
            "position",
            "recent_team",
        )
    }
    _require_columns(weekly, mapped_columns | identity_columns | {"game_id"})

    column = WEEKLY_PLAYER_STATS_COLUMNS
    eligible = weekly.filter(
        (pl.col(column["season_type"]) == "REG")
        & pl.col(column["position"]).is_in(POSITIONS)
        & pl.col(column["player_id"]).is_not_null()
        & pl.col("game_id").is_not_null()
    )
    if eligible.is_empty():
        raise PlayerSeasonError("No eligible regular-season QB/RB/WR/TE rows found")

    duplicate_games = (
        eligible.group_by([column["player_id"], "game_id"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate_games.height:
        raise PlayerSeasonError(
            f"Found {duplicate_games.height} duplicate player-game groups; "
            "refusing to double-count stats"
        )

    scored = calculate_weekly_fantasy_points(eligible)
    total_expressions = [
        pl.col(column[semantic])
        .cast(pl.Float64, strict=True)
        .fill_null(0.0)
        .sum()
        .alias(output_name)
        for output_name, semantic in SEASON_TOTAL_FIELDS.items()
    ]
    player_seasons = (
        scored.group_by([column["player_id"], column["season"]])
        .agg(
            pl.col(column["player_name"])
            .drop_nulls()
            .sort_by(column["week"])
            .last()
            .alias("player_name"),
            pl.col(column["position"]).sort_by(column["week"]).last().alias("position"),
            pl.col(column["recent_team"])
            .drop_nulls()
            .sort_by(column["week"])
            .last()
            .alias("team"),
            pl.col("game_id").n_unique().alias("games_played"),
            pl.col(STANDARD_FANTASY_POINTS_COLUMN).sum().alias("total_fantasy_points"),
            *total_expressions,
        )
        .rename(
            {
                column["player_id"]: "player_id",
                column["season"]: "season",
            }
        )
        .with_columns(
            (pl.col("rushing_attempts") + pl.col("targets")).alias("opportunities"),
            (pl.col("total_fantasy_points") / pl.col("games_played")).alias(
                "fantasy_points_per_game"
            ),
        )
        .with_columns(
            *[
                (pl.col(field) / pl.col("games_played")).alias(f"{field}_per_game")
                for field in PER_GAME_FIELDS
            ]
        )
        .sort(["season", "position", "player_id"])
    )

    _validate_player_seasons(player_seasons)
    targets = player_seasons.filter(
        (pl.col("games_played") >= min_target_games)
        & pl.col("position").is_in(POSITIONS)
        & pl.col("fantasy_points_per_game").is_not_null()
        & pl.col("fantasy_points_per_game").is_finite()
    )
    return player_seasons, targets


def _validate_player_seasons(player_seasons: pl.DataFrame) -> None:
    duplicate_count = (
        player_seasons.group_by(GROUP_KEYS).len().filter(pl.col("len") > 1).height
    )
    if duplicate_count:
        raise PlayerSeasonError(
            f"Player-season audit table has {duplicate_count} duplicate keys"
        )
    if player_seasons.filter(pl.col("games_played") <= 0).height:
        raise PlayerSeasonError("Player-season audit table contains non-positive games")
    invalid_seasons = player_seasons.filter(
        ~pl.col("season").is_between(
            HISTORICAL_START_SEASON, HISTORICAL_END_SEASON, closed="both"
        )
    )
    if invalid_seasons.height:
        raise PlayerSeasonError("Player-season audit table contains invalid seasons")

    arithmetic_errors = player_seasons.filter(
        (
            pl.col("fantasy_points_per_game")
            - pl.col("total_fantasy_points") / pl.col("games_played")
        ).abs()
        > 1e-10
    )
    if arithmetic_errors.height:
        raise PlayerSeasonError(
            f"Found {arithmetic_errors.height} player-seasons with incorrect PPG"
        )


def _write_parquet_validated(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        frame.write_parquet(temporary_path, use_pyarrow=True)
        if pl.read_parquet(temporary_path).shape != frame.shape:
            raise PlayerSeasonError(f"Read-back shape validation failed for {path}")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved {frame.height:,} rows x {frame.width:,} columns: {path.resolve()}")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def write_quality_report(
    player_seasons: pl.DataFrame,
    targets: pl.DataFrame,
    output_path: Path,
    *,
    min_target_games: int,
) -> Path:
    distribution = (
        targets.group_by("position")
        .agg(
            pl.len().alias("player_seasons"),
            pl.col("fantasy_points_per_game").mean().alias("mean_ppg"),
            pl.col("fantasy_points_per_game").median().alias("median_ppg"),
            pl.col("fantasy_points_per_game").std().alias("std_ppg"),
            pl.col("fantasy_points_per_game").min().alias("min_ppg"),
            pl.col("fantasy_points_per_game").max().alias("max_ppg"),
        )
        .sort("position")
    )
    review_columns = [
        "player_id",
        "player_name",
        "season",
        "position",
        "team",
        "games_played",
        "total_fantasy_points",
        "fantasy_points_per_game",
    ]
    highest = (
        targets.sort("fantasy_points_per_game", descending=True)
        .select(review_columns)
        .head(20)
    )
    lowest = targets.sort("fantasy_points_per_game").select(review_columns).head(20)
    report = [
        "FANTASY FOOTBALL PLAYER-SEASON TARGET QUALITY REPORT",
        "",
        "PARTICIPATION RULE",
        "games_played is the count of unique non-null game_id values per player-season.",
        "Zero-stat rows count because game_id is explicit participation evidence.",
        "Duplicate player-game rows are rejected before aggregation.",
        "",
        "FILTERS",
        "Regular season only (season_type == REG).",
        f"Positions: {', '.join(POSITIONS)}.",
        "Rows without a stable player_id are excluded.",
        f"Modeling target minimum games: {min_target_games}.",
        "",
        "QUALITY CHECKS PASSED",
        "No duplicate player_id-season rows.",
        "No zero or negative games_played.",
        "PPG equals total fantasy points / games played within 1e-10.",
        f"All seasons are within {HISTORICAL_START_SEASON}-{HISTORICAL_END_SEASON}.",
        "All target PPG values are finite and non-null.",
        "",
        f"AUDIT PLAYER-SEASONS: {player_seasons.height}",
        f"MODELING TARGET PLAYER-SEASONS: {targets.height}",
        "",
        "TARGET DISTRIBUTION BY POSITION",
        _json(distribution.to_dicts()),
        "",
        "20 HIGHEST HISTORICAL PPG ROWS",
        _json(highest.to_dicts()),
        "",
        "20 LOWEST HISTORICAL PPG ROWS",
        _json(lowest.to_dicts()),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved quality report: {output_path.resolve()}")
    return output_path


def run_build(
    *,
    input_path: Path = RAW_DATA_DIR / PLAYER_STATS_FILENAME,
    audit_path: Path = INTERIM_DATA_DIR / AUDIT_FILENAME,
    target_path: Path = PROCESSED_DATA_DIR / TARGET_FILENAME,
    report_path: Path = REPORT_TABLES_DIR / QUALITY_REPORT_FILENAME,
    min_target_games: int = MIN_TARGET_GAMES,
) -> tuple[Path, Path, Path]:
    if not input_path.is_file():
        raise PlayerSeasonError(f"Weekly player stats file not found: {input_path}")
    try:
        weekly = pl.read_parquet(input_path)
    except Exception as exc:
        raise PlayerSeasonError(f"Could not read {input_path}: {exc}") from exc

    print(f"Input: {input_path.resolve()} ({weekly.height:,} rows)")
    player_seasons, targets = build_player_season_tables(
        weekly, min_target_games=min_target_games
    )
    _write_parquet_validated(player_seasons, audit_path)
    _write_parquet_validated(targets, target_path)
    write_quality_report(
        player_seasons,
        targets,
        report_path,
        min_target_games=min_target_games,
    )
    return audit_path, target_path, report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=RAW_DATA_DIR / PLAYER_STATS_FILENAME
    )
    parser.add_argument(
        "--audit-output", type=Path, default=INTERIM_DATA_DIR / AUDIT_FILENAME
    )
    parser.add_argument(
        "--target-output", type=Path, default=PROCESSED_DATA_DIR / TARGET_FILENAME
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPORT_TABLES_DIR / QUALITY_REPORT_FILENAME,
    )
    parser.add_argument("--min-target-games", type=int, default=MIN_TARGET_GAMES)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_build(
            input_path=args.input,
            audit_path=args.audit_output,
            target_path=args.target_output,
            report_path=args.report_output,
            min_target_games=args.min_target_games,
        )
    except PlayerSeasonError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
