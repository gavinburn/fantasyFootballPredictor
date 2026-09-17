"""Inspect mandatory raw datasets and save a reproducible schema report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from src.column_mapping import (
    PLAYER_MASTER_COLUMNS,
    ROSTER_COLUMNS,
    WEEKLY_PLAYER_STATS_COLUMNS,
    validate_mapping,
)
from src.config import RAW_DATA_DIR, REPORT_TABLES_DIR
from src.download_data import PLAYER_STATS_FILENAME, PLAYERS_FILENAME

REPORT_FILENAME = "data_schema_report.txt"
FANTASY_COLUMN_TERMS = (
    "passing",
    "rushing",
    "receiving",
    "receptions",
    "targets",
    "carries",
    "fumble",
    "fantasy",
    "2pt",
)


class InspectionError(RuntimeError):
    """Raised when mandatory raw data cannot be uniquely found or inspected."""


def find_mandatory_paths(raw_data_dir: Path) -> dict[str, Path]:
    """Resolve mandatory raw files, requiring exactly one dated roster snapshot."""

    roster_paths = sorted(raw_data_dir.glob("rosters_2026_*.parquet"))
    if not roster_paths:
        raise InspectionError(f"No 2026 roster snapshot found under {raw_data_dir}")

    paths = {
        "weekly_player_stats": raw_data_dir / PLAYER_STATS_FILENAME,
        "player_master": raw_data_dir / PLAYERS_FILENAME,
        "roster_2026": roster_paths[-1],
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise InspectionError("Missing mandatory raw files: " + ", ".join(missing))
    return paths


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def _distinct_values(frame: pl.DataFrame, column: str) -> list[Any]:
    return frame.get_column(column).drop_nulls().unique().sort().to_list()


def _candidate_id_columns(frame: pl.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.lower() == "id" or column.lower().endswith("_id")
    ]


def _fantasy_columns(frame: pl.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if any(term in column.lower() for term in FANTASY_COLUMN_TERMS)
    ]


def _null_percentages(frame: pl.DataFrame) -> dict[str, float]:
    null_counts = frame.null_count().row(0, named=True)
    return {
        column: round(100.0 * count / frame.height, 4)
        for column, count in null_counts.items()
    }


def inspect_frame(
    name: str,
    path: Path,
    mapping: Mapping[str, str],
) -> str:
    """Read one dataset, validate its mapping, and return its report section."""

    try:
        frame = pl.read_parquet(path)
    except Exception as exc:
        raise InspectionError(f"Could not read {name} at {path}: {exc}") from exc
    if frame.is_empty():
        raise InspectionError(f"Mandatory dataset {name} is empty: {path}")

    selected_mapping = validate_mapping(name, frame.columns, mapping)
    distinct_seasons = (
        _distinct_values(frame, "season") if "season" in frame.columns else []
    )
    distinct_positions = {
        column: _distinct_values(frame, column)
        for column in ("position", "position_group")
        if column in frame.columns
    }
    schema = {column: str(dtype) for column, dtype in frame.schema.items()}

    lines = [
        "=" * 88,
        f"DATASET: {name}",
        f"PATH: {path.resolve()}",
        f"SHAPE: {frame.height} rows x {frame.width} columns",
        "",
        "SCHEMA / DATA TYPES",
        _json(schema),
        "",
        "FIRST FIVE ROWS",
        _json(frame.head(5).to_dicts()),
        "",
        "DISTINCT SEASONS",
        _json(distinct_seasons),
        "",
        "NULL PERCENTAGE BY COLUMN",
        _json(_null_percentages(frame)),
        "",
        "CANDIDATE PLAYER ID COLUMNS",
        _json(_candidate_id_columns(frame)),
        "",
        "DISTINCT POSITION VALUES",
        _json(distinct_positions),
        "",
        "RELEVANT FANTASY-STAT COLUMNS",
        _json(_fantasy_columns(frame)),
        "",
        "SELECTED SEMANTIC COLUMN MAPPINGS",
        _json(selected_mapping),
        "",
    ]
    return "\n".join(lines)


def generate_schema_report(
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
    output_path: Path = REPORT_TABLES_DIR / REPORT_FILENAME,
) -> Path:
    """Inspect every mandatory raw dataset and write the Phase 2 report."""

    paths = find_mandatory_paths(raw_data_dir)
    specifications = (
        (
            "weekly_player_stats",
            paths["weekly_player_stats"],
            WEEKLY_PLAYER_STATS_COLUMNS,
        ),
        ("player_master", paths["player_master"], PLAYER_MASTER_COLUMNS),
        ("roster_2026", paths["roster_2026"], ROSTER_COLUMNS),
    )
    sections = [
        "FANTASY FOOTBALL RAW DATA SCHEMA REPORT",
        "Generated from mandatory Phase 1 Parquet snapshots.",
        "",
    ]
    for name, path, mapping in specifications:
        print(f"Inspecting {name}: {path.resolve()}")
        section = inspect_frame(name, path, mapping)
        sections.append(section)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved schema report: {output_path.resolve()}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_TABLES_DIR / REPORT_FILENAME,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        generate_schema_report(raw_data_dir=args.raw_data_dir, output_path=args.output)
    except (InspectionError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
