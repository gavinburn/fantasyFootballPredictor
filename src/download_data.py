"""Download and snapshot the mandatory Phase 1 nflverse datasets."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.config import (
    HISTORICAL_END_SEASON,
    HISTORICAL_START_SEASON,
    POSITIONS,
    PREDICTION_SEASON,
    RAW_DATA_DIR,
)

PLAYER_STATS_FILENAME = "player_weekly_stats_2006_2025.parquet"
PLAYERS_FILENAME = "players.parquet"
MANIFEST_FILENAME = "raw_data_manifest.json"
POSITION_COLUMN_CANDIDATES = ("position", "position_group")


class DownloadError(RuntimeError):
    """Raised when a required dataset cannot be downloaded or validated."""


@dataclass(frozen=True)
class DatasetRecord:
    """Details needed to build one manifest dataset entry."""

    name: str
    path: Path
    frame: pl.DataFrame
    requested_seasons: list[int] | None


def _download(name: str, loader: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    print(f"Downloading {name}...")
    try:
        frame = loader()
    except Exception as exc:
        raise DownloadError(
            f"Failed to download required dataset '{name}': {exc}"
        ) from exc

    if not isinstance(frame, pl.DataFrame):
        raise DownloadError(
            f"Required dataset '{name}' returned {type(frame).__name__}, "
            "not a Polars DataFrame."
        )
    if frame.is_empty():
        raise DownloadError(f"Required dataset '{name}' returned zero rows.")
    return frame


def _find_column(frame: pl.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {column.lower(): column for column in frame.columns}
    return next(
        (lookup[name.lower()] for name in candidates if name.lower() in lookup),
        None,
    )


def _validate_historical_seasons(frame: pl.DataFrame) -> list[int]:
    season_column = _find_column(frame, ("season",))
    if season_column is None:
        raise DownloadError("Weekly player stats do not contain a 'season' column.")

    actual = sorted(
        int(value)
        for value in frame.get_column(season_column).drop_nulls().unique().to_list()
    )
    expected = set(range(HISTORICAL_START_SEASON, HISTORICAL_END_SEASON + 1))
    missing = sorted(expected.difference(actual))
    if missing:
        raise DownloadError(
            "Weekly player stats are missing requested seasons: "
            + ", ".join(map(str, missing))
        )
    return actual


def _validate_roster_positions(frame: pl.DataFrame) -> list[str]:
    position_column = _find_column(frame, POSITION_COLUMN_CANDIDATES)
    if position_column is None:
        raise DownloadError(
            "The 2026 roster has no recognized position column; checked "
            + ", ".join(POSITION_COLUMN_CANDIDATES)
        )

    actual = {
        str(value).strip().upper()
        for value in frame.get_column(position_column).drop_nulls().unique().to_list()
    }
    missing = sorted(set(POSITIONS).difference(actual))
    if missing:
        raise DownloadError(
            "The 2026 roster is missing required positions: " + ", ".join(missing)
        )
    return sorted(actual)


def _save_parquet(frame: pl.DataFrame, path: Path, *, preserve: bool = False) -> None:
    if preserve and path.exists():
        print(f"Preserving existing roster snapshot: {path.resolve()}")
        return
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        frame.write_parquet(temporary_path, use_pyarrow=True)
        saved_shape = pl.read_parquet(temporary_path).shape
        if saved_shape != frame.shape:
            raise DownloadError(
                f"Saved dataset validation failed for {path.name}: "
                f"expected shape {frame.shape}, read back {saved_shape}."
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved {frame.height:,} rows x {frame.width:,} columns: {path.resolve()}")


def _manifest_entry(record: DatasetRecord) -> dict[str, Any]:
    stat = record.path.stat()
    return {
        "requested_seasons": record.requested_seasons,
        "output_path": str(record.path.resolve()),
        "row_count": record.frame.height,
        "column_count": record.frame.width,
        "column_names": record.frame.columns,
        "file_size_bytes": stat.st_size,
    }


def download_mandatory_data(
    *,
    output_dir: Path = RAW_DATA_DIR,
    snapshot_date: date | None = None,
    nfl_module: Any | None = None,
) -> Path:
    """Download, validate, and save all mandatory Phase 1 datasets."""

    if nfl_module is None:
        import nflreadpy as nfl_module

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_date = snapshot_date or datetime.now(UTC).date()
    historical_seasons = list(range(HISTORICAL_START_SEASON, HISTORICAL_END_SEASON + 1))

    weekly = _download(
        "weekly player statistics (2006-2025)",
        lambda: nfl_module.load_player_stats(
            seasons=historical_seasons, summary_level="week"
        ),
    )
    players = _download("player master data", nfl_module.load_players)

    roster_path = output_dir / (
        f"rosters_{PREDICTION_SEASON}_{snapshot_date.isoformat()}.parquet"
    )
    if roster_path.exists():
        print(f"Using existing same-day roster snapshot: {roster_path.resolve()}")
        roster = pl.read_parquet(roster_path)
        if roster.is_empty():
            raise DownloadError(f"Existing roster snapshot is empty: {roster_path}")
    else:
        roster = _download(
            f"{PREDICTION_SEASON} seasonal roster",
            lambda: nfl_module.load_rosters(seasons=PREDICTION_SEASON),
        )

    represented_seasons = _validate_historical_seasons(weekly)
    represented_positions = _validate_roster_positions(roster)

    weekly_path = output_dir / PLAYER_STATS_FILENAME
    players_path = output_dir / PLAYERS_FILENAME
    _save_parquet(weekly, weekly_path)
    _save_parquet(players, players_path)
    _save_parquet(roster, roster_path, preserve=True)

    records = (
        DatasetRecord("player_weekly_stats", weekly_path, weekly, historical_seasons),
        DatasetRecord("players", players_path, players, None),
        DatasetRecord("rosters_2026", roster_path, roster, [PREDICTION_SEASON]),
    )
    manifest = {
        "download_timestamp": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "nflreadpy_version": importlib.metadata.version("nflreadpy"),
        "roster_snapshot_date": snapshot_date.isoformat(),
        "represented_historical_seasons": represented_seasons,
        "represented_roster_positions": represented_positions,
        "datasets": {record.name: _manifest_entry(record) for record in records},
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest: {manifest_path.resolve()}")
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Raw data directory (default: repository data/raw).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        download_mandatory_data(output_dir=args.output_dir)
    except DownloadError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
