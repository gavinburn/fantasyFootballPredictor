"""Freeze the nflverse contract-history source for an isolated ablation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.config import CONTRACT_EXPERIMENT_RAW_DATA_DIR

CONTRACT_FILE = "nflverse_contracts_snapshot.parquet"
MANIFEST_FILE = "contract_raw_data_manifest.json"


class ContractDownloadError(RuntimeError):
    """Raised when the contract source cannot be frozen safely."""


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
            raise ContractDownloadError(f"Read-back validation failed for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def snapshot_contract_data(
    output_dir: Path = CONTRACT_EXPERIMENT_RAW_DATA_DIR,
) -> Path:
    """Download once, remove current-status state, and record a checksum."""

    import nflreadpy as nfl

    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / CONTRACT_FILE
    if not contract_path.exists():
        print("Downloading historical nflverse/OverTheCap contract records...")
        raw = nfl.load_contracts()
        required = {
            "player", "position", "team", "year_signed", "years", "value",
            "apy", "guaranteed", "apy_cap_pct", "inflated_apy", "gsis_id",
            "draft_year", "cols",
        }
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ContractDownloadError(f"Contract source is missing columns: {missing}")
        # is_active is current state, not historical point-in-time state. It is
        # intentionally absent from the frozen research source.
        keep = [column for column in raw.columns if column != "is_active"]
        _write(raw.select(keep), contract_path)

    frame = pl.read_parquet(contract_path)
    if frame.is_empty() or "is_active" in frame.columns:
        raise ContractDownloadError("Frozen contract source violates snapshot rules")
    years = frame.get_column("year_signed").drop_nulls()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source": "nflverse load_contracts / OverTheCap contract history",
        "rows": frame.height,
        "columns": frame.width,
        "year_signed_range": [int(years.min()), int(years.max())],
        "sha256": _sha256(contract_path),
        "point_in_time_rule": (
            "A season-t row may use only a contract with year_signed <= t-1. "
            "Same-season contracts are excluded because exact signing dates are unavailable."
        ),
        "excluded_fields": {
            "is_active": "Current status would leak future knowledge into historical folds."
        },
        "known_limitations": [
            "The source is a retrospectively maintained contract-history table.",
            "Prior-year-only eligibility omits legitimate offseason contracts signed in season t.",
            "Contract terms proxy organizational commitment and are not a direct receiving-role label.",
        ],
    }
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    snapshot_contract_data()
