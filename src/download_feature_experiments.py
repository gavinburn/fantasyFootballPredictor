"""Snapshot the external inputs required by feature experiments 3 through 5."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.config import FEATURE35_RAW_DATA_DIR

FILES = {
    "opportunity_weekly": "ffopportunity_weekly_v1_2006_2025.parquet",
    "opportunity_pass": "ffopportunity_pass_v1_2006_2025.parquet",
    "opportunity_rush": "ffopportunity_rush_v1_2006_2025.parquet",
    "participation": "participation_2016_2025.parquet",
    "snap_counts": "snap_counts_2012_2025.parquet",
    "ngs_passing": "ngs_passing_2016_2025.parquet",
    "ngs_receiving": "ngs_receiving_2016_2025.parquet",
    "ngs_rushing": "ngs_rushing_2016_2025.parquet",
    "pfr_passing": "pfr_passing_2018_2025.parquet",
    "pfr_rushing": "pfr_rushing_2018_2025.parquet",
    "pfr_receiving": "pfr_receiving_2018_2025.parquet",
}

SELECTED_COLUMNS = {
    "opportunity_weekly": [
        "season", "posteam", "week", "game_id", "player_id", "full_name",
        "position", "total_fantasy_points", "total_fantasy_points_exp",
        "total_fantasy_points_diff",
    ],
    "opportunity_pass": [
        "game_id", "play_id", "receiver_player_id", "receiver_full_name",
        "receiver_position", "posteam", "season", "week", "pass_attempt",
        "two_point_attempt", "air_yards", "yardline_100",
    ],
    "opportunity_rush": [
        "game_id", "play_id", "rusher_player_id", "full_name", "position",
        "posteam", "season", "week", "rush_attempt", "two_point_attempt",
        "yardline_100", "qb_scramble",
    ],
    "participation": [
        "nflverse_game_id", "play_id", "possession_team", "offense_players",
        "route",
    ],
    "snap_counts": [
        "game_id", "season", "game_type", "week", "player", "pfr_player_id",
        "position", "team", "offense_snaps", "offense_pct",
    ],
    "ngs_passing": [
        "season", "season_type", "week", "player_gsis_id", "player_position",
        "team_abbr", "attempts", "avg_time_to_throw",
        "completion_percentage_above_expectation",
    ],
    "ngs_receiving": [
        "season", "season_type", "week", "player_gsis_id", "player_position",
        "team_abbr", "targets", "receptions", "avg_separation",
        "avg_yac_above_expectation",
    ],
    "ngs_rushing": [
        "season", "season_type", "week", "player_gsis_id", "player_position",
        "team_abbr", "rush_attempts", "avg_time_to_los",
        "rush_yards_over_expected_per_att",
    ],
    "pfr_passing": [
        "season", "week", "game_type", "team", "pfr_player_id",
        "times_sacked", "times_pressured",
    ],
    "pfr_rushing": [
        "season", "week", "game_type", "team", "pfr_player_id", "carries",
        "rushing_yards_after_contact", "rushing_broken_tackles",
        "receiving_broken_tackles",
    ],
    "pfr_receiving": [
        "season", "week", "game_type", "team", "pfr_player_id",
        "rushing_broken_tackles", "receiving_broken_tackles",
    ],
}


class FeatureExperimentDownloadError(RuntimeError):
    """Raised when an experiment input cannot be snapshotted safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.write_parquet(temporary, use_pyarrow=True)
        if pl.read_parquet(temporary).shape != frame.shape:
            raise FeatureExperimentDownloadError(f"Read-back validation failed: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_years(
    years: list[int], loader: Callable[[int], pl.DataFrame], columns: list[str]
) -> pl.DataFrame:
    frames = []
    for season in years:
        print(f"  loading {season}...")
        frame = loader(season)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise FeatureExperimentDownloadError(
                f"Season {season} is missing expected columns: {missing}"
            )
        frames.append(frame.select(columns))
    return pl.concat(frames, how="diagonal_relaxed")


def snapshot_feature_experiment_data(
    output_dir: Path = FEATURE35_RAW_DATA_DIR,
) -> Path:
    """Download frozen source versions and record checksums and coverage."""

    import nflreadpy as nfl

    output_dir.mkdir(parents=True, exist_ok=True)
    loaders: dict[str, tuple[list[int], Callable[[int], pl.DataFrame]]] = {
        "opportunity_weekly": (
            list(range(2006, 2026)),
            lambda year: nfl.load_ff_opportunity(year, "weekly", "v1.0.0"),
        ),
        "opportunity_pass": (
            list(range(2006, 2026)),
            lambda year: nfl.load_ff_opportunity(year, "pbp_pass", "v1.0.0"),
        ),
        "opportunity_rush": (
            list(range(2006, 2026)),
            lambda year: nfl.load_ff_opportunity(year, "pbp_rush", "v1.0.0"),
        ),
        "participation": (
            list(range(2016, 2026)), lambda year: nfl.load_participation(year)
        ),
        "snap_counts": (
            list(range(2012, 2026)), lambda year: nfl.load_snap_counts(year)
        ),
        "ngs_passing": (
            list(range(2016, 2026)), lambda year: nfl.load_nextgen_stats(year, "passing")
        ),
        "ngs_receiving": (
            list(range(2016, 2026)), lambda year: nfl.load_nextgen_stats(year, "receiving")
        ),
        "ngs_rushing": (
            list(range(2016, 2026)), lambda year: nfl.load_nextgen_stats(year, "rushing")
        ),
        "pfr_passing": (
            list(range(2018, 2026)),
            lambda year: nfl.load_pfr_advstats(year, "pass", "week"),
        ),
        "pfr_rushing": (
            list(range(2018, 2026)),
            lambda year: nfl.load_pfr_advstats(year, "rush", "week"),
        ),
        "pfr_receiving": (
            list(range(2018, 2026)),
            lambda year: nfl.load_pfr_advstats(year, "rec", "week"),
        ),
    }
    records = {}
    for name, (years, loader) in loaders.items():
        path = output_dir / FILES[name]
        if not path.exists():
            print(f"Downloading {name}...")
            _write(_load_years(years, loader, SELECTED_COLUMNS[name]), path)
        frame = pl.read_parquet(path)
        if frame.is_empty():
            raise FeatureExperimentDownloadError(f"{name} has no rows")
        records[name] = {
            "path": str(path.resolve()), "rows": frame.height,
            "columns": frame.width, "seasons": [min(years), max(years)],
            "sha256": _sha256(path),
        }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "ffopportunity_model_version": "v1.0.0",
        "point_in_time_rule": "Only season t-1 values are joined to prediction season t",
        "known_schema_limitations": {
            "participation_route": (
                "route labels describe the targeted receiver, not every route runner; "
                "pass-play participation is derived instead of claiming routes run"
            ),
            "inline_blocking": "not available and intentionally omitted",
        },
        "datasets": records,
    }
    path = output_dir / "feature_experiments_raw_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    snapshot_feature_experiment_data()
