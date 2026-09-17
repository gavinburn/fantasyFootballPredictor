"""Pandas-only feature manifests and migration audits for Attempt 2.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ATTEMPT2_INTERIM_DATA_DIR,
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    ATTEMPT2_PROCESSED_DATA_DIR,
    ATTEMPT21_DATA_DIR,
    ATTEMPT21_PROCESSED_DATA_DIR,
    ATTEMPT21_REPORT_TABLES_DIR,
    DATA_DIR,
    MODELS_DIR,
    POSITIONS,
    REPORTS_DIR,
    REPOSITORY_ROOT,
)

V1_FEATURES = {
    "QB": [
        "previous_season_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
        "previous_passing_attempts_per_game",
        "previous_passing_yards_per_attempt",
        "previous_passing_touchdowns_per_game",
        "previous_interceptions_per_game",
        "previous_qb_rushing_attempts_per_game",
        "previous_qb_rushing_yards_per_game",
        "previous_qb_rushing_touchdowns_per_game",
        "age_entering_season",
        "years_experience_entering_season",
    ],
    "RB": [
        "previous_season_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
        "previous_rushing_attempts_per_game",
        "previous_rushing_yards_per_attempt",
        "previous_rushing_touchdowns_per_game",
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_touchdowns_per_game",
        "previous_opportunities_per_game",
        "age_entering_season",
        "years_experience_entering_season",
    ],
    "WR": [
        "previous_season_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_yards_per_target",
        "previous_receiving_touchdowns_per_game",
        "previous_rushing_attempts_per_game",
        "previous_rushing_yards_per_game",
        "age_entering_season",
        "years_experience_entering_season",
    ],
    "TE": [
        "previous_season_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_yards_per_target",
        "previous_receiving_touchdowns_per_game",
        "age_entering_season",
        "years_experience_entering_season",
    ],
}

GROUPS = {
    "volume": {
        "QB": [
            "target_team_dropbacks_per_game_tminus1",
            "target_team_pass_rate_tminus1",
        ],
        "RB": [
            "target_team_carries_per_game_tminus1",
            "target_team_plays_per_game_tminus1",
        ],
        "WR": [
            "target_team_dropbacks_per_game_tminus1",
            "target_team_pass_rate_tminus1",
        ],
        "TE": [
            "target_team_dropbacks_per_game_tminus1",
            "target_team_pass_rate_tminus1",
        ],
    },
    "efficiency": {
        "QB": ["target_team_passing_tds_per_game_tminus1"],
        "RB": [
            "target_team_rushing_tds_per_game_tminus1",
            "target_team_rush_epa_per_carry_tminus1",
        ],
        "WR": [
            "target_team_passing_tds_per_game_tminus1",
            "target_team_pass_epa_per_dropback_tminus1",
        ],
        "TE": [
            "target_team_passing_tds_per_game_tminus1",
            "target_team_pass_epa_per_dropback_tminus1",
        ],
    },
    "workload": {
        "QB": ["previous_qb_dropback_share"],
        "RB": [
            "previous_carry_share",
            "previous_target_share",
            "previous_opportunity_share",
        ],
        "WR": ["previous_target_share", "previous_air_yards_share"],
        "TE": ["previous_target_share", "previous_air_yards_share"],
    },
    "competition": {
        "QB": [
            "top_competitor_previous_ppg",
            "top_competitor_age",
            "competition_pool_size",
            "competition_no_history_count",
        ],
        "RB": [
            "top_competitor_previous_total_points",
            "top_competitor_previous_ppg",
            "top_competitor_age",
            "second_competitor_previous_ppg",
            "competition_pool_size",
            "competition_no_history_count",
        ],
        "WR": [
            "top_competitor_previous_total_points",
            "top_competitor_previous_ppg",
            "top_competitor_age",
            "top_competitor_is_wr",
            "second_competitor_previous_ppg",
            "second_competitor_is_wr",
            "competition_pool_size",
            "competition_no_history_count",
        ],
        "TE": [
            "top_competitor_previous_total_points",
            "top_competitor_previous_ppg",
            "top_competitor_age",
            "top_competitor_is_wr",
            "second_competitor_previous_ppg",
            "second_competitor_is_wr",
            "competition_pool_size",
            "competition_no_history_count",
        ],
    },
    "quarterback": {
        p: [
            "projected_qb_previous_qbr",
            "projected_qb_previous_qb_plays",
            "projected_qb_has_previous_qbr",
        ]
        for p in POSITIONS
    },
    "team_change": {p: ["changed_team"] for p in POSITIONS},
    "offensive_line": {p: ["preseason_offensive_line_score"] for p in POSITIONS},
    "schedule": {p: ["preseason_strength_of_schedule_score"] for p in POSITIONS},
}


class Attempt21FeatureError(RuntimeError):
    """Raised when pandas migration or feature contracts fail."""


TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "LA": "LAR",
    "JAC": "JAX",
    "WSH": "WAS",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "LAR",
}


def position_mismatch_exclusions() -> pd.DataFrame:
    """Return seasons whose target and preseason roster positions disagree."""

    context = pd.read_parquet(
        ATTEMPT2_INTERIM_DATA_DIR / "prediction_context.parquet",
        columns=[
            "player_id",
            "prediction_season",
            "position",
            "roster_position",
            "target_team",
        ],
        engine="pyarrow",
    )
    exclusions = context[context["position"] != context["roster_position"]].copy()
    if exclusions.duplicated(["player_id", "prediction_season"]).any():
        raise Attempt21FeatureError(
            "Position-mismatch exclusions contain duplicate keys"
        )
    exclusions["exclusion_reason"] = (
        "target_position_differs_from_week1_roster_position"
    )
    return exclusions.sort_values(
        ["prediction_season", "player_id"], kind="stable"
    ).reset_index(drop=True)


def remove_position_mismatch_seasons(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Remove only flagged seasons, retaining unaffected seasons for each player."""

    exclusions = position_mismatch_exclusions()
    keys = exclusions[["player_id", "prediction_season"]]
    filtered: dict[str, pd.DataFrame] = {}
    removal_rows = []
    for position, table in tables.items():
        marked = table.merge(
            keys.assign(_position_mismatch=True),
            on=["player_id", "prediction_season"],
            how="left",
            validate="one_to_one",
        )
        remove = marked["_position_mismatch"].eq(True)
        removal_rows.append(
            {
                "position": position,
                "rows_before": len(table),
                "rows_removed": int(remove.sum()),
                "rows_after": int((~remove).sum()),
            }
        )
        filtered[position] = (
            marked[~remove].drop(columns="_position_mismatch").reset_index(drop=True)
        )
    audit = exclusions.merge(
        pd.DataFrame(removal_rows),
        on="position",
        how="left",
        validate="many_to_one",
    )
    audit["removed_from_attempt21"] = True
    return filtered, audit


def first_team_disagreement_exclusions() -> pd.DataFrame:
    """Return cutoff-rostered seasons with no appearance for the assigned team."""

    context = pd.read_parquet(
        ATTEMPT2_INTERIM_DATA_DIR / "prediction_context.parquet",
        columns=[
            "player_id",
            "prediction_season",
            "position",
            "target_team",
            "status",
        ],
        engine="pyarrow",
    )
    weekly = pd.read_parquet(
        DATA_DIR / "raw/player_weekly_stats_2006_2025.parquet",
        columns=["player_id", "season", "week", "season_type", "team"],
        engine="pyarrow",
    )
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    weekly["team"] = weekly["team"].replace(TEAM_ALIASES)
    first = (
        weekly.sort_values(["player_id", "season", "week", "team"], kind="stable")
        .drop_duplicates(["player_id", "season"])
        .rename(columns={"season": "prediction_season", "team": "actual_first_team"})[
            ["player_id", "prediction_season", "actual_first_team"]
        ]
    )
    compared = context.merge(
        first,
        on=["player_id", "prediction_season"],
        how="left",
        validate="one_to_one",
    )
    exclusions = compared[
        compared["actual_first_team"].notna()
        & compared["target_team"].ne(compared["actual_first_team"])
    ].copy()
    exclusions["exclusion_reason"] = (
        "waived_after_cutoff_before_first_regular_season_appearance"
    )
    if exclusions.duplicated(["player_id", "prediction_season"]).any():
        raise Attempt21FeatureError(
            "First-team disagreement exclusions contain duplicate keys"
        )
    return exclusions.sort_values(
        ["prediction_season", "player_id"], kind="stable"
    ).reset_index(drop=True)


def remove_first_team_disagreement_seasons(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Remove the confirmed waived-before-first-appearance player-seasons."""

    exclusions = first_team_disagreement_exclusions()
    keys = exclusions[["player_id", "prediction_season"]]
    filtered: dict[str, pd.DataFrame] = {}
    removal_rows = []
    for position, table in tables.items():
        marked = table.merge(
            keys.assign(_first_team_disagreement=True),
            on=["player_id", "prediction_season"],
            how="left",
            validate="one_to_one",
        )
        remove = marked["_first_team_disagreement"].eq(True)
        removal_rows.append(
            {
                "position": position,
                "rows_before": len(table),
                "rows_removed": int(remove.sum()),
                "rows_after": int((~remove).sum()),
            }
        )
        filtered[position] = (
            marked[~remove]
            .drop(columns="_first_team_disagreement")
            .reset_index(drop=True)
        )
    audit = exclusions.merge(
        pd.DataFrame(removal_rows),
        on="position",
        how="left",
        validate="many_to_one",
    )
    audit["removed_from_attempt21"] = True
    return filtered, audit


def remove_boundary_prediction_season(
    tables: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Keep 2006 as source history but remove it as a prediction target season."""

    filtered: dict[str, pd.DataFrame] = {}
    audits = []
    for position, table in tables.items():
        remove = table["prediction_season"].eq(2006)
        audit = table.loc[
            remove,
            ["player_id", "player_name", "position", "prediction_season"],
        ].copy()
        audit["exclusion_reason"] = (
            "boundary_history_only_2006_first_prediction_season_is_2007"
        )
        audit["rows_before"] = len(table)
        audit["rows_removed"] = int(remove.sum())
        audit["rows_after"] = int((~remove).sum())
        audit["removed_from_attempt21"] = True
        audits.append(audit)
        filtered[position] = table.loc[~remove].reset_index(drop=True)
    return filtered, pd.concat(audits, ignore_index=True)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _add(base: list[str], *groups: list[str]) -> list[str]:
    return _unique([*base, *(item for group in groups for item in group)])


def corrected_core(position: str, representation: str = "components") -> list[str]:
    features = list(V1_FEATURES[position])
    if position == "RB":
        if representation == "components":
            features.remove("previous_opportunities_per_game")
        elif representation == "total_mix":
            features.remove("previous_rushing_attempts_per_game")
            features.remove("previous_targets_per_game")
            features.append("previous_target_opportunity_rate")
        else:
            raise Attempt21FeatureError(f"Unknown RB representation: {representation}")
    return features


def build_candidate_sets() -> dict[str, dict[str, list[str]]]:
    candidates: dict[str, dict[str, list[str]]] = {p: {} for p in POSITIONS}

    qb = corrected_core("QB")
    candidates["QB"] = {
        "QB_A_CORE": qb,
        "QB_B_VOLUME": _add(qb, GROUPS["volume"]["QB"]),
        "QB_C_QB_ENV": _add(qb, GROUPS["quarterback"]["QB"]),
        "QB_D_COMPETITION": _add(qb, GROUPS["competition"]["QB"]),
        "QB_E_TEAM_CHANGE": _add(qb, GROUPS["team_change"]["QB"]),
        "QB_F_OL": _add(qb, GROUPS["offensive_line"]["QB"]),
    }
    qb_reduced = _add(
        qb,
        GROUPS["volume"]["QB"],
        ["previous_qb_dropback_share"],
        ["top_competitor_previous_ppg", "competition_pool_size"],
        GROUPS["quarterback"]["QB"],
        GROUPS["team_change"]["QB"],
    )
    candidates["QB"].update(
        {
            "QB_G_REDUCED_CONTEXT": qb_reduced,
            "QB_H_REDUCED_PLUS_OL": _add(qb_reduced, GROUPS["offensive_line"]["QB"]),
            "QB_I_REDUCED_PLUS_OL_SOS": _add(
                qb_reduced, GROUPS["offensive_line"]["QB"], GROUPS["schedule"]["QB"]
            ),
            "QB_FULL_NONREDUNDANT": _add(qb, *(GROUPS[g]["QB"] for g in GROUPS)),
        }
    )

    for representation, suffix in (
        ("components", "COMPONENTS"),
        ("total_mix", "TOTAL_MIX"),
    ):
        rb = corrected_core("RB", representation)
        candidates["RB"].update(
            {
                f"RB_CORE_{suffix}": rb,
                f"RB_WORKLOAD_{suffix}": _add(rb, GROUPS["workload"]["RB"]),
                f"RB_COMP_CHANGE_{suffix}": _add(
                    rb, GROUPS["competition"]["RB"], GROUPS["team_change"]["RB"]
                ),
            }
        )
    rb = corrected_core("RB", "components")
    rb_reduced = _add(
        rb,
        ["previous_opportunity_share"],
        ["target_team_carries_per_game_tminus1"],
        ["top_competitor_previous_total_points", "competition_pool_size"],
        ["projected_qb_previous_qbr", "projected_qb_has_previous_qbr"],
        GROUPS["team_change"]["RB"],
        GROUPS["offensive_line"]["RB"],
        GROUPS["schedule"]["RB"],
    )
    candidates["RB"].update(
        {
            "RB_REDUCED_CONTEXT": rb_reduced,
            "RB_FULL_NONREDUNDANT": _add(rb, *(GROUPS[g]["RB"] for g in GROUPS)),
        }
    )

    wr = corrected_core("WR")
    wr_reduced = _add(
        wr,
        ["previous_target_share"],
        ["target_team_dropbacks_per_game_tminus1"],
        [
            "top_competitor_previous_total_points",
            "competition_pool_size",
            "top_competitor_is_wr",
        ],
        ["projected_qb_previous_qbr", "projected_qb_has_previous_qbr"],
        GROUPS["team_change"]["WR"],
        GROUPS["offensive_line"]["WR"],
        GROUPS["schedule"]["WR"],
    )
    candidates["WR"] = {
        "WR_CORE": wr,
        "WR_WORKLOAD": _add(wr, GROUPS["workload"]["WR"]),
        "WR_COMP_CHANGE": _add(
            wr, GROUPS["competition"]["WR"], GROUPS["team_change"]["WR"]
        ),
        "WR_REDUCED_CONTEXT": wr_reduced,
        "WR_FULL_NONREDUNDANT": _add(wr, *(GROUPS[g]["WR"] for g in GROUPS)),
    }

    te = corrected_core("TE")
    sequence = {
        "TE_A_CORE": te,
        "TE_B_TARGET_SHARE": _add(te, ["previous_target_share"]),
        "TE_C_TARGET_AIR_SHARE": _add(
            te, ["previous_target_share", "previous_air_yards_share"]
        ),
    }
    sequence["TE_D_PLUS_VOLUME"] = _add(
        sequence["TE_C_TARGET_AIR_SHARE"], GROUPS["volume"]["TE"]
    )
    sequence["TE_E_PLUS_QB"] = _add(
        sequence["TE_D_PLUS_VOLUME"], GROUPS["quarterback"]["TE"]
    )
    sequence["TE_F_PLUS_COMPETITION"] = _add(
        sequence["TE_E_PLUS_QB"], GROUPS["competition"]["TE"]
    )
    sequence["TE_G_PLUS_EFFICIENCY"] = _add(
        sequence["TE_F_PLUS_COMPETITION"], GROUPS["efficiency"]["TE"]
    )
    sequence["TE_H_PLUS_OL"] = _add(
        sequence["TE_G_PLUS_EFFICIENCY"], GROUPS["offensive_line"]["TE"]
    )
    sequence["TE_I_PLUS_SOS"] = _add(sequence["TE_H_PLUS_OL"], GROUPS["schedule"]["TE"])
    sequence["TE_FULL_NONREDUNDANT"] = _add(te, *(GROUPS[g]["TE"] for g in GROUPS))
    candidates["TE"] = sequence
    return candidates


def load_attempt2_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for position in POSITIONS:
        path = (
            ATTEMPT2_PROCESSED_DATA_DIR
            / f"{position.lower()}_attempt2_modeling_dataset.parquet"
        )
        frame = pd.read_parquet(path, engine="pyarrow")
        frame = frame.sort_values(
            ["prediction_season", "player_id"], kind="stable"
        ).reset_index(drop=True)
        if frame.duplicated(["player_id", "prediction_season"]).any():
            raise Attempt21FeatureError(f"Duplicate {position} player-season keys")
        if position == "RB":
            denominator = frame["previous_opportunities_per_game"]
            frame["previous_target_opportunity_rate"] = np.where(
                denominator > 0,
                frame["previous_targets_per_game"] / denominator,
                np.nan,
            )
        tables[position] = frame
    return tables


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def run_pandas_parity(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet",
        engine="pyarrow",
    )
    frozen = frozen[frozen["model_name"] == "A_v1_same_cohort"].copy()
    rows = []
    for position in POSITIONS:
        table = tables[position]
        features = V1_FEATURES[position]
        for season in range(2012, 2026):
            train = table[table["prediction_season"] < season]
            valid = table[
                (table["prediction_season"] == season)
                & table["previous_season_ppg"].notna()
            ]
            model = _pipeline().fit(train[features], train["target_ppg"])
            predicted = model.predict(valid[features])
            current = pd.DataFrame(
                {
                    "player_id": valid["player_id"].to_numpy(),
                    "season": season,
                    "position": position,
                    "pandas_prediction": predicted,
                }
            )
            reference = frozen[
                (frozen["position"] == position) & (frozen["season"] == season)
            ][["player_id", "season", "position", "predicted_ppg"]]
            compared = current.merge(
                reference,
                on=["player_id", "season", "position"],
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            max_difference = float(
                np.max(
                    np.abs(compared["pandas_prediction"] - compared["predicted_ppg"])
                )
            )
            rows.append(
                {
                    "position": position,
                    "season": season,
                    "pandas_rows": len(current),
                    "frozen_rows": len(reference),
                    "unmatched_rows": int((compared["_merge"] != "both").sum()),
                    "max_prediction_difference": max_difference,
                    "feature_names_match": list(model.feature_names_in_) == features,
                    "passed": (
                        len(current) == len(reference)
                        and (compared["_merge"] == "both").all()
                        and max_difference <= 1e-10
                        and list(model.feature_names_in_) == features
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    if not audit["passed"].all():
        raise Attempt21FeatureError(
            "Pandas parity failed:\n" + audit[~audit["passed"]].to_string(index=False)
        )
    return audit


def dependency_audit(
    tables: dict[str, pd.DataFrame], candidates: dict[str, dict[str, list[str]]]
) -> pd.DataFrame:
    rows = []
    for position, specifications in candidates.items():
        table = tables[position]
        for name, features in specifications.items():
            for season in range(2012, 2026):
                train = table[table["prediction_season"] < season][features]
                filled = train.fillna(train.median(numeric_only=True)).fillna(0.0)
                varying = filled.loc[:, filled.std(ddof=0) > 1e-12]
                standardized = (varying - varying.mean()) / varying.std(ddof=0)
                rank = int(np.linalg.matrix_rank(standardized.to_numpy()))
                rows.append(
                    {
                        "position": position,
                        "specification": name,
                        "season": season,
                        "features": len(features),
                        "nonconstant_features": varying.shape[1],
                        "matrix_rank": rank,
                        "rank_deficiency": varying.shape[1] - rank,
                        "passed": rank == varying.shape[1],
                    }
                )
    audit = pd.DataFrame(rows)
    if not audit["passed"].all():
        failures = audit[~audit["passed"]]
        raise Attempt21FeatureError(
            "Corrected candidate still has exact dependencies:\n"
            + failures.head(20).to_string(index=False)
        )
    return audit


def correlation_decisions(
    tables: dict[str, pd.DataFrame], candidates: dict[str, dict[str, list[str]]]
) -> pd.DataFrame:
    rows = []
    for position in POSITIONS:
        full_name = next(
            name for name in candidates[position] if "FULL_NONREDUNDANT" in name
        )
        features = candidates[position][full_name]
        train = tables[position][tables[position]["prediction_season"] < 2012][features]
        corr = train.corr(numeric_only=True)
        for left, feature_1 in enumerate(features):
            for feature_2 in features[left + 1 :]:
                value = corr.loc[feature_1, feature_2]
                if pd.notna(value) and abs(value) >= 0.75:
                    rows.append(
                        {
                            "position": position,
                            "feature_1": feature_1,
                            "feature_2": feature_2,
                            "correlation_pre_2012": value,
                            "absolute_correlation": abs(value),
                            "decision": "exclude_together_in_reduced"
                            if abs(value) >= 0.90
                            else "allow_regularized_only",
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["position", "absolute_correlation"], ascending=[True, False]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_preserved_attempts() -> Path:
    output = ATTEMPT21_DATA_DIR / "preservation_checksums_before.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for base in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        for path in sorted(base.rglob("*")):
            if path.is_file() and "attempt_2_1" not in path.parts:
                checksums[str(path.relative_to(REPOSITORY_ROOT))] = _sha256(path)
    archive = REPOSITORY_ROOT / "attempt 2 results.zip"
    if archive.exists():
        checksums[archive.name] = _sha256(archive)
    output.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    return output


def run_attempt21_feature_build() -> dict[str, Path]:
    snapshot_preserved_attempts()
    frozen_tables = load_attempt2_tables()
    parity = run_pandas_parity(frozen_tables)
    parity["cohort"] = "frozen_pre_exclusion_parity"
    tables, mismatch_exclusions = remove_position_mismatch_seasons(frozen_tables)
    tables, first_team_exclusions = remove_first_team_disagreement_seasons(tables)
    tables, boundary_exclusions = remove_boundary_prediction_season(tables)
    candidates = build_candidate_sets()
    dependency = dependency_audit(tables, candidates)
    correlations = correlation_decisions(tables, candidates)

    ATTEMPT21_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ATTEMPT21_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for position, table in tables.items():
        table.to_parquet(
            ATTEMPT21_PROCESSED_DATA_DIR
            / f"{position.lower()}_attempt21_modeling_dataset.parquet",
            engine="pyarrow",
            index=False,
        )
    parity.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_pandas_parity_audit.csv", index=False
    )
    mismatch_exclusions.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "position_mismatch_exclusions.csv",
        index=False,
    )
    first_team_exclusions.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "first_team_disagreement_exclusions.csv",
        index=False,
    )
    boundary_exclusions.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "boundary_season_exclusions.csv",
        index=False,
    )
    dependency.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "exact_dependency_audit.csv", index=False
    )
    correlations.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "correlation_decisions.csv", index=False
    )
    (ATTEMPT21_REPORT_TABLES_DIR / "candidate_feature_sets.json").write_text(
        json.dumps(candidates, indent=2) + "\n", encoding="utf-8"
    )
    dictionary = [
        {
            "position": position,
            "specification": name,
            "feature_order": index + 1,
            "feature": feature,
        }
        for position, specs in candidates.items()
        for name, features in specs.items()
        for index, feature in enumerate(features)
    ]
    pd.DataFrame(dictionary).to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "feature_set_dictionary.csv", index=False
    )
    return {
        "parity": ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_pandas_parity_audit.csv",
        "manifest": ATTEMPT21_REPORT_TABLES_DIR / "candidate_feature_sets.json",
    }
