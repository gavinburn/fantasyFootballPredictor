"""Snapshot the additional leakage-safe inputs required by Attempt 2."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.config import (
    ATTEMPT2_RAW_DATA_DIR,
    HISTORICAL_END_SEASON,
    HISTORICAL_START_SEASON,
    RAW_DATA_DIR,
    REPOSITORY_ROOT,
)

TEAM_WEEKLY_FILENAME = "team_weekly_stats_2006_2025.parquet"
ROSTERS_WEEKLY_FILENAME = "rosters_weekly_2006_2025.parquet"
DEPTH_CHARTS_FILENAME = "depth_charts_2006_2025.parquet"
DEPTH_CHARTS_2025_FILENAME = "depth_charts_2025_timestamped.parquet"
QBR_FILENAME = "qbr_season_level.parquet"
RANKINGS_FILENAME = "preseason_offensive_line_and_sos.parquet"
MANIFEST_FILENAME = "phase2_raw_data_manifest.json"
QBR_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "espn_data/qbr_season_level.parquet"
)
DEPTH_CHARTS_2025_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "depth_charts/depth_charts_2025.parquet"
)


class Phase2DownloadError(RuntimeError):
    """Raised when an Attempt 2 input cannot be snapshotted safely."""


def _write_validated(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.write_parquet(temporary, use_pyarrow=True)
        if pl.read_parquet(temporary).shape != frame.shape:
            raise Phase2DownloadError(f"Read-back validation failed for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_seasons(frame: pl.DataFrame, name: str) -> None:
    expected = set(range(HISTORICAL_START_SEASON, HISTORICAL_END_SEASON + 1))
    actual = set(frame.get_column("season").drop_nulls().cast(pl.Int32).to_list())
    missing = sorted(expected - actual)
    if missing:
        raise Phase2DownloadError(f"{name} is missing seasons {missing}")


def _load_rankings(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        truncate_ragged_lines=False,
        quote_char=None,
        schema_overrides={"ol_rank": pl.String, "sos_rank": pl.String},
    ).rename(
        {
            "season": "prediction_season",
            "team": "target_team",
            "ol_rank": "preseason_offensive_line_rank",
            "sos_rank": "preseason_strength_of_schedule_rank",
        }
    ).with_columns(
        pl.col("preseason_offensive_line_rank")
        .str.strip_chars('"')
        .cast(pl.Int32),
        pl.col("preseason_strength_of_schedule_rank")
        .str.strip_chars('"')
        .cast(pl.Int32),
    )
    if frame.height != 640:
        raise Phase2DownloadError(f"Expected 640 ranking rows, found {frame.height}")
    invalid = frame.filter(
        ~pl.col("preseason_offensive_line_rank").is_between(1, 32)
        | ~pl.col("preseason_strength_of_schedule_rank").is_between(1, 32)
    )
    if invalid.height:
        raise Phase2DownloadError("Ranking values must be within 1 through 32")
    counts = frame.group_by("prediction_season").len()
    if counts.filter(pl.col("len") != 32).height:
        raise Phase2DownloadError("Every ranking season must contain 32 teams")
    if frame.is_duplicated().sum() or frame.select(
        pl.struct("prediction_season", "target_team").is_duplicated().sum()
    ).item():
        raise Phase2DownloadError("Duplicate ranking team-season keys found")
    return frame.with_columns(
        ((32 - pl.col("preseason_offensive_line_rank")) / 31).alias(
            "preseason_offensive_line_score"
        ),
        ((pl.col("preseason_strength_of_schedule_rank") - 1) / 31).alias(
            "preseason_strength_of_schedule_score"
        ),
        pl.lit(1, dtype=pl.Int8).alias("offensive_line_rank_available"),
        pl.lit(1, dtype=pl.Int8).alias("strength_of_schedule_rank_available"),
        pl.lit("preseason_machine_learning_model").alias("source_method"),
        pl.lit("OL: 1=best; SOS: 1=hardest").alias("rank_direction"),
    )


def snapshot_phase2_data(
    *,
    rankings_source: Path,
    output_dir: Path = ATTEMPT2_RAW_DATA_DIR,
) -> Path:
    """Download Phase 2 data and preserve a checksum manifest."""

    import nflreadpy as nfl

    seasons = list(range(HISTORICAL_START_SEASON, HISTORICAL_END_SEASON + 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = {
        TEAM_WEEKLY_FILENAME: lambda: nfl.load_team_stats(
            seasons=seasons, summary_level="week"
        ),
        ROSTERS_WEEKLY_FILENAME: lambda: nfl.load_rosters_weekly(seasons=seasons),
        DEPTH_CHARTS_FILENAME: lambda: nfl.load_depth_charts(seasons=seasons),
        DEPTH_CHARTS_2025_FILENAME: lambda: pl.read_parquet(DEPTH_CHARTS_2025_URL),
        QBR_FILENAME: lambda: pl.read_parquet(QBR_URL),
        RANKINGS_FILENAME: lambda: _load_rankings(rankings_source),
    }
    records: dict[str, dict[str, object]] = {}
    for filename, loader in loaders.items():
        path = output_dir / filename
        if path.exists():
            frame = pl.read_parquet(path)
            print(f"Preserving existing Phase 2 snapshot: {path.resolve()}")
        else:
            print(f"Downloading/building {filename}...")
            frame = loader()
            if frame.is_empty():
                raise Phase2DownloadError(f"{filename} returned zero rows")
            _write_validated(frame, path)
        if (
            "season" in frame.columns
            and filename not in {QBR_FILENAME, DEPTH_CHARTS_FILENAME}
        ):
            _validate_seasons(frame, filename)
        if filename == DEPTH_CHARTS_FILENAME:
            expected_legacy = set(range(HISTORICAL_START_SEASON, 2025))
            actual_legacy = set(frame["season"].drop_nulls().cast(pl.Int32).to_list())
            if not expected_legacy.issubset(actual_legacy):
                raise Phase2DownloadError("Legacy depth charts must cover 2006-2024")
        records[filename] = {
            "path": str(path.resolve()),
            "rows": frame.height,
            "columns": frame.width,
            "sha256": _sha256(path),
        }

    # Record V1 files so preservation can be verified after Attempt 2 completes.
    attempt1_files: dict[str, str] = {}
    for base in (RAW_DATA_DIR, REPOSITORY_ROOT / "data/processed", REPOSITORY_ROOT / "data/predictions", REPOSITORY_ROOT / "models", REPOSITORY_ROOT / "reports"):
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file() and "attempt_2" not in path.parts:
                    attempt1_files[str(path.relative_to(REPOSITORY_ROOT))] = _sha256(path)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seasons": [HISTORICAL_START_SEASON, HISTORICAL_END_SEASON],
        "ranking_provenance": {
            "method": "machine-learning model",
            "timing": "inputs available before each season",
            "ol_rank_1": "best",
            "sos_rank_1": "hardest",
        },
        "datasets": records,
        "attempt1_checksums_before_attempt2": attempt1_files,
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Phase 2 manifest: {manifest_path.resolve()}")
    return manifest_path
