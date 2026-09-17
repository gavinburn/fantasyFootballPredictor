"""Snapshot point-in-time external inputs for the isolated Attempt 3 experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.config import ATTEMPT3_RAW_DATA_DIR

RANKINGS_FILE = "preseason_fantasypros_ecr_2020_2025.parquet"
INJURIES_FILE = "injuries_2009_2025.parquet"
MANIFEST_FILE = "attempt3_raw_data_manifest.json"


class Attempt3DownloadError(RuntimeError):
    """Raised when an Attempt 3 input cannot be safely snapshotted."""


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
            raise Attempt3DownloadError(f"Read-back validation failed for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _preseason_rankings(nfl: object) -> pl.DataFrame:
    raw = nfl.load_ff_rankings("all").with_columns(
        pl.col("scrape_date").str.to_date(strict=False).alias("snapshot_date")
    )
    eligible = raw.filter(
        pl.col("snapshot_date").dt.year().is_between(2020, 2025)
        & (pl.col("snapshot_date") <= pl.date(pl.col("snapshot_date").dt.year(), 9, 1))
        & (pl.col("ecr_type") == "ro")
        & pl.col("fp_page").str.contains("ppr-cheatsheets")
        & pl.col("pos").is_in(["QB", "RB", "WR", "TE"])
        & pl.col("ecr").is_not_null()
    )
    snapshots = eligible.group_by(
        pl.col("snapshot_date").dt.year().alias("prediction_season")
    ).agg(pl.col("snapshot_date").max())
    if set(snapshots["prediction_season"].to_list()) != set(range(2020, 2026)):
        raise Attempt3DownloadError("A preseason ECR snapshot is not available for every 2020-2025 season")
    selected = eligible.with_columns(
        pl.col("snapshot_date").dt.year().alias("prediction_season")
    ).join(snapshots, on=["prediction_season", "snapshot_date"], how="inner")
    # A few archived pages contain a duplicate player. Collapse only exact
    # player-position-season identities and preserve the mean consensus rank.
    result = selected.group_by(["prediction_season", "player", "pos"]).agg(
        pl.col("team").drop_nulls().first().alias("team"),
        pl.col("ecr").mean().alias("preseason_ecr"),
        pl.col("best").min().alias("preseason_ecr_best"),
        pl.col("worst").max().alias("preseason_ecr_worst"),
        pl.col("snapshot_date").first(),
        pl.len().alias("source_rows"),
    ).sort(["prediction_season", "pos", "preseason_ecr", "player"])
    if result.filter(pl.col("snapshot_date") > pl.date(pl.col("prediction_season"), 9, 1)).height:
        raise Attempt3DownloadError("A market snapshot is later than the September 1 cutoff")
    return result


def snapshot_attempt3_data(output_dir: Path = ATTEMPT3_RAW_DATA_DIR) -> Path:
    """Download and checksum the injury and preseason-market snapshots."""

    import nflreadpy as nfl

    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / RANKINGS_FILE
    injury_path = output_dir / INJURIES_FILE
    if not ranking_path.exists():
        print("Downloading and filtering historical preseason FantasyPros ECR...")
        _write(_preseason_rankings(nfl), ranking_path)
    if not injury_path.exists():
        print("Downloading historical injury reports...")
        _write(nfl.load_injuries(list(range(2009, 2026))), injury_path)

    rankings = pl.read_parquet(ranking_path)
    injuries = pl.read_parquet(injury_path)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "rankings": {
            "source": "DynastyProcess archive of FantasyPros expert consensus rankings",
            "ranking_type": "preseason PPR redraft overall ECR",
            "selection": "latest archived snapshot on or before September 1",
            "important_target_difference": "model target is standard non-PPR PPG",
            "seasons": [2020, 2025],
            "rows": rankings.height,
            "sha256": _sha256(ranking_path),
        },
        "injuries": {
            "source": "nflverse injuries",
            "seasons": [2009, 2025],
            "rows": injuries.height,
            "sha256": _sha256(injury_path),
        },
    }
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    snapshot_attempt3_data()
