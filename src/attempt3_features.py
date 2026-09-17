"""Build veteran-aligned structural and market features for Attempt 3."""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd

from src.attempt21_features import build_candidate_sets
from src.config import (
    ATTEMPT3_PROCESSED_DATA_DIR,
    ATTEMPT3_RAW_DATA_DIR,
    ATTEMPT3_REPORT_TABLES_DIR,
    DATA_DIR,
    POSITIONS,
)
from src.download_attempt3_data import INJURIES_FILE, RANKINGS_FILE


class Attempt3FeatureError(RuntimeError):
    """Raised when Attempt 3 feature contracts fail."""


BASE_SPECIFICATIONS = {
    "QB": "QB_B_VOLUME",
    "RB": "RB_COMP_CHANGE_TOTAL_MIX",
    "WR": "WR_WORKLOAD",
    # Attempt 2.1 TE was rejected, so the frozen V1 feature set remains the base.
    "TE": "TE_A_CORE",
}
AGE_KNOTS = {"QB": 36.0, "RB": 27.0, "WR": 30.0, "TE": 30.0}
QB_MISSING_FLAGS = [
    "projected_qb_missing_no_prior_workload",
    "projected_qb_missing_low_workload_injury",
    "projected_qb_missing_low_workload_other",
    "projected_qb_missing_source_omission",
]
STRUCTURAL_FEATURES = [
    "previous_availability_rate",
    "three_year_durability_index",
    "durability_history_seasons",
    "age_pre_apex",
    "age_post_apex",
    "projected_qb_previous_dropbacks",
    *QB_MISSING_FLAGS,
]
MARKET_FEATURES = ["market_implied_ppg", "preseason_ecr_available"]


def _normal_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text.lower())
    return re.sub(r"[^a-z0-9]", "", text)


def _availability_features() -> pd.DataFrame:
    weekly = pd.read_parquet(
        DATA_DIR / "raw/player_weekly_stats_2006_2025.parquet",
        columns=["player_id", "season", "season_type", "game_id"],
    )
    weekly = weekly[(weekly["season_type"] == "REG") & weekly["game_id"].notna()]
    games = weekly.groupby(["player_id", "season"], as_index=False)["game_id"].nunique()
    games = games.rename(columns={"game_id": "games_played"})
    games["scheduled_games"] = np.where(games["season"] >= 2021, 17.0, 16.0)
    games["availability"] = (games["games_played"] / games["scheduled_games"]).clip(upper=1.0)
    pieces = []
    weights = {1: 0.50, 2: 0.33, 3: 0.17}
    for lag, weight in weights.items():
        part = games[["player_id", "season", "availability"]].copy()
        part["prediction_season"] = part.pop("season") + lag
        part = part.rename(columns={"availability": f"availability_lag{lag}"})
        part[f"durability_weight_lag{lag}"] = weight
        pieces.append(part)
    merged = pieces[0]
    for part in pieces[1:]:
        merged = merged.merge(part, on=["player_id", "prediction_season"], how="outer")
    weighted = sum(
        merged[f"availability_lag{lag}"].fillna(0) * weight for lag, weight in weights.items()
    )
    present_weight = sum(
        merged[f"availability_lag{lag}"].notna().astype(float) * weight
        for lag, weight in weights.items()
    )
    merged["previous_availability_rate"] = merged["availability_lag1"]
    merged["three_year_durability_index"] = weighted / present_weight.replace(0, np.nan)
    merged["durability_history_seasons"] = sum(
        merged[f"availability_lag{lag}"].notna().astype(int) for lag in weights
    )
    return merged[[
        "player_id", "prediction_season", "previous_availability_rate",
        "three_year_durability_index", "durability_history_seasons",
    ]]


def _projected_qb_features() -> pd.DataFrame:
    context = pd.read_parquet(
        DATA_DIR / "attempt_2/interim/projected_qb_environment.parquet"
    )
    weekly = pd.read_parquet(
        DATA_DIR / "raw/player_weekly_stats_2006_2025.parquet",
        columns=["player_id", "season", "season_type", "attempts", "sacks_suffered"],
    )
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    for column in ("attempts", "sacks_suffered"):
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce").fillna(0)
    workload = weekly.groupby(["player_id", "season"], as_index=False)[
        ["attempts", "sacks_suffered"]
    ].sum()
    workload["projected_qb_previous_dropbacks"] = workload["attempts"] + workload["sacks_suffered"]
    workload["prediction_season"] = workload.pop("season") + 1
    workload = workload.rename(columns={"player_id": "projected_starting_qb_id"})

    injuries = pd.read_parquet(ATTEMPT3_RAW_DATA_DIR / INJURIES_FILE)
    injuries = injuries[(injuries["game_type"] == "REG") & injuries["gsis_id"].notna()].copy()
    injuries["documented_limiting_injury"] = injuries["report_status"].isin(["Out", "Doubtful"])
    injury = injuries.groupby(["gsis_id", "season"], as_index=False).agg(
        documented_limiting_injury=("documented_limiting_injury", "max")
    )
    injury["prediction_season"] = injury.pop("season").astype(int) + 1
    injury = injury.rename(columns={"gsis_id": "projected_starting_qb_id"})

    result = context.merge(
        workload[["projected_starting_qb_id", "prediction_season", "projected_qb_previous_dropbacks"]],
        on=["projected_starting_qb_id", "prediction_season"], how="left", validate="many_to_one",
    ).merge(
        injury, on=["projected_starting_qb_id", "prediction_season"], how="left", validate="many_to_one"
    )
    result["projected_qb_previous_dropbacks"] = result["projected_qb_previous_dropbacks"].fillna(0.0)
    result["documented_limiting_injury"] = (
        result["documented_limiting_injury"].astype("boolean").fillna(False).astype(bool)
    )
    missing = result["projected_qb_previous_qbr"].isna()
    db = result["projected_qb_previous_dropbacks"]
    result[QB_MISSING_FLAGS[0]] = (missing & db.eq(0)).astype(int)
    result[QB_MISSING_FLAGS[1]] = (missing & db.between(1, 99) & result["documented_limiting_injury"]).astype(int)
    result[QB_MISSING_FLAGS[2]] = (missing & db.between(1, 99) & ~result["documented_limiting_injury"]).astype(int)
    result[QB_MISSING_FLAGS[3]] = (missing & db.ge(100)).astype(int)
    if not (result.loc[missing, QB_MISSING_FLAGS].sum(axis=1) == 1).all():
        raise Attempt3FeatureError("Missing projected-QB QBR rows are not mutually exclusive")
    return result[[
        "prediction_season", "target_team", "projected_starting_qb_id",
        "projected_qb_previous_dropbacks", "documented_limiting_injury", *QB_MISSING_FLAGS,
    ]]


def _market_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_parquet(ATTEMPT3_RAW_DATA_DIR / RANKINGS_FILE)
    market = market[market["pos"].isin(POSITIONS)].copy()
    market["normalized_name"] = market["player"].map(_normal_name)
    if market.duplicated(["prediction_season", "pos", "normalized_name"]).any():
        raise Attempt3FeatureError("Normalized market keys are not unique")
    return market, market.groupby("prediction_season", as_index=False).agg(
        market_rows=("player", "size"), snapshot_date=("snapshot_date", "first")
    )


def feature_manifests() -> dict[str, dict[str, list[str]]]:
    attempt21 = build_candidate_sets()
    output = {}
    for position in POSITIONS:
        base = list(attempt21[position][BASE_SPECIFICATIONS[position]])
        structural_base = [
            feature for feature in base
            if feature not in {"age_entering_season", "previous_season_games_played"}
        ]
        output[position] = {
            "VETERAN_ALIGNED_BASE": base,
            "STRUCTURAL_CLEANUP": [*structural_base, *STRUCTURAL_FEATURES],
            "MARKET_ASSISTED": [*structural_base, *STRUCTURAL_FEATURES, *MARKET_FEATURES],
        }
    return output


def build_attempt3_features() -> dict[str, pd.DataFrame]:
    availability = _availability_features()
    qb = _projected_qb_features()
    market, market_audit = _market_features()
    outputs = {}
    cohort_rows = []
    coverage_rows = []
    for position in POSITIONS:
        source = pd.read_parquet(
            DATA_DIR / "attempt_2_1/processed" / f"{position.lower()}_attempt21_modeling_dataset.parquet"
        )
        before = len(source)
        source = source[source["previous_season_ppg"].notna()].copy()
        cohort_rows.append({
            "position": position, "rows_before": before,
            "no_history_rows_removed_from_all_model_fits": before - len(source),
            "rows_after": len(source),
        })
        source = source.merge(availability, on=["player_id", "prediction_season"], how="left", validate="one_to_one")
        source = source.drop(columns=[column for column in ["projected_qb_previous_dropbacks", *QB_MISSING_FLAGS] if column in source])
        source = source.merge(
            qb.drop(columns=["projected_starting_qb_id"]),
            on=["prediction_season", "target_team"], how="left", validate="many_to_one",
        )
        knot = AGE_KNOTS[position]
        source["age_pre_apex"] = source["age_entering_season"].clip(upper=knot)
        source["age_post_apex"] = (source["age_entering_season"] - knot).clip(lower=0)
        source["normalized_name"] = source["player_name"].map(_normal_name)
        position_market = market[market["pos"] == position][[
            "prediction_season", "normalized_name", "player", "team", "preseason_ecr", "snapshot_date"
        ]].rename(columns={"player": "market_player_name", "team": "market_team"})
        source = source.merge(
            position_market, on=["prediction_season", "normalized_name"], how="left", validate="many_to_one"
        )
        source["preseason_ecr_available"] = source["preseason_ecr"].notna().astype(int)
        source["market_implied_ppg"] = np.nan  # fitted independently inside every fold
        for season, group in source.groupby("prediction_season"):
            eligible = group["previous_season_ppg"].notna()
            coverage_rows.append({
                "position": position, "prediction_season": season,
                "eligible_veterans": int(eligible.sum()),
                "market_matches": int(group.loc[eligible, "preseason_ecr"].notna().sum()),
                "market_coverage_rate": float(group.loc[eligible, "preseason_ecr"].notna().mean()),
                "snapshot_date": group["snapshot_date"].dropna().min() if group["snapshot_date"].notna().any() else pd.NaT,
            })
        source = source.drop(columns="normalized_name")
        outputs[position] = source.sort_values(["prediction_season", "player_id"]).reset_index(drop=True)

    ATTEMPT3_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ATTEMPT3_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for position, frame in outputs.items():
        frame.to_parquet(ATTEMPT3_PROCESSED_DATA_DIR / f"{position.lower()}_attempt3_modeling_dataset.parquet", index=False)
    pd.DataFrame(cohort_rows).to_csv(ATTEMPT3_REPORT_TABLES_DIR / "veteran_training_cohort_audit.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(ATTEMPT3_REPORT_TABLES_DIR / "market_match_coverage.csv", index=False)
    market_audit.to_csv(ATTEMPT3_REPORT_TABLES_DIR / "market_snapshot_audit.csv", index=False)
    (ATTEMPT3_REPORT_TABLES_DIR / "candidate_feature_sets.json").write_text(
        json.dumps(feature_manifests(), indent=2) + "\n", encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    build_attempt3_features()
