"""Sealed 2012-2016 confirmation of the locked constrained TE correction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.attempt21_models import TOP_N, _metrics
from src.config import (
    RANDOM_STATE,
    TE_CONSTRAINED_PREDICTIONS_DATA_DIR,
    TE_CONSTRAINED_PROCESSED_DATA_DIR,
    TE_CONSTRAINED_REPORT_TABLES_DIR,
    TE_CONSTRAINED_REPORTS_DIR,
)
from src.te_constrained_competition import (
    ADJUSTERS,
    CENTERED_COUNT,
    CORRECTION_CAP,
    HIGH_BURDEN,
    MULTI_THRESHOLD,
    PARAMETER_GRIDS,
    _clip,
    _parameter_complexity,
)

DISCOVERY_SEASONS = list(range(2017, 2026))
CONFIRMATION_SEASONS = list(range(2012, 2017))
LOCKED_COMPONENTS = [CENTERED_COUNT, HIGH_BURDEN, MULTI_THRESHOLD]
SHRINKAGE_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]

# Predeclared before reading confirmation outcomes.
MIN_MAE_IMPROVEMENT = 0.005
MIN_SEASONS_IMPROVED = 3
MIN_SPEARMAN_CHANGE = 0.0
MAX_LOW_TIER_BIAS_CHANGE = 0.0
MIN_FEATURED_TIER_BIAS_CHANGE = 0.0

CONFIRMATION_RULES = {
    "cohort": "Previously unused constrained-experiment seasons 2012-2016",
    "parameter_lock": "Components and parameters are selected only on 2017-2025",
    "minimum_mae_improvement_ppg": MIN_MAE_IMPROVEMENT,
    "minimum_seasons_improved": MIN_SEASONS_IMPROVED,
    "minimum_spearman_change": MIN_SPEARMAN_CHANGE,
    "maximum_low_tier_bias_change": MAX_LOW_TIER_BIAS_CHANGE,
    "minimum_featured_tier_bias_change": MIN_FEATURED_TIER_BIAS_CHANGE,
    "required": "All conditions must pass",
    "limitation": (
        "This is a sealed reverse-time holdout, not a prospective forward-season test."
    ),
}


class TEConfirmationError(RuntimeError):
    """Raised when confirmation-set isolation or scoring contracts fail."""


def _load_table() -> pd.DataFrame:
    table = pd.read_parquet(
        TE_CONSTRAINED_PROCESSED_DATA_DIR / "te_constrained_modeling_dataset.parquet"
    )
    seasons = set(table["prediction_season"].unique())
    required = set(DISCOVERY_SEASONS + CONFIRMATION_SEASONS)
    if not required.issubset(seasons):
        raise TEConfirmationError(
            f"Confirmation table is missing seasons: {sorted(required - seasons)}"
        )
    return table


def _fit_component_on_discovery(
    discovery: pd.DataFrame, specification: str
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for params in PARAMETER_GRIDS[specification]:
        adjustment = ADJUSTERS[specification](discovery, discovery, params)
        predicted = discovery["frozen_v1_predicted_ppg"].to_numpy() + adjustment
        rows.append({
            **params,
            "parameter_complexity": _parameter_complexity(params),
            "discovery_mae": float(np.mean(np.abs(
                predicted - discovery["target_ppg"].to_numpy()
            ))),
        })
    tuning = pd.DataFrame(rows).sort_values(
        ["discovery_mae", "parameter_complexity"], kind="stable"
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    params = {
        key: float(tuning.loc[0, key])
        for key in PARAMETER_GRIDS[specification][0]
    }
    return params, tuning


def lock_candidate(
    table: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], float, pd.DataFrame]:
    discovery = table[table["prediction_season"].isin(DISCOVERY_SEASONS)].copy()
    parameters = {}
    tuning_rows = []
    component_adjustments = []
    for specification in LOCKED_COMPONENTS:
        params, tuning = _fit_component_on_discovery(discovery, specification)
        parameters[specification] = params
        component_adjustments.append(
            ADJUSTERS[specification](discovery, discovery, params)
        )
        for row in tuning.to_dict("records"):
            tuning_rows.append({"specification": specification, **row})
    mean_adjustment = np.column_stack(component_adjustments).mean(axis=1)
    shrinkage_rows = []
    for shrinkage in SHRINKAGE_GRID:
        predicted = (
            discovery["frozen_v1_predicted_ppg"].to_numpy()
            + _clip(shrinkage * mean_adjustment)
        )
        shrinkage_rows.append({
            "specification": "COMBINED_SHRINKAGE",
            "combined_shrinkage": shrinkage,
            "discovery_mae": float(np.mean(np.abs(
                predicted - discovery["target_ppg"].to_numpy()
            ))),
        })
    shrinkage_tuning = pd.DataFrame(shrinkage_rows).sort_values(
        ["discovery_mae", "combined_shrinkage"], kind="stable"
    ).reset_index(drop=True)
    shrinkage_tuning["selected"] = False
    shrinkage_tuning.loc[0, "selected"] = True
    shrinkage = float(shrinkage_tuning.loc[0, "combined_shrinkage"])
    tuning_rows.extend(shrinkage_tuning.to_dict("records"))
    return parameters, shrinkage, pd.DataFrame(tuning_rows)


def score_confirmation(
    table: pd.DataFrame,
    parameters: dict[str, dict[str, float]],
    shrinkage: float,
) -> pd.DataFrame:
    discovery = table[table["prediction_season"].isin(DISCOVERY_SEASONS)].copy()
    confirmation = table[
        table["prediction_season"].isin(CONFIRMATION_SEASONS)
    ].copy()
    corrections = np.column_stack([
        ADJUSTERS[specification](discovery, confirmation, parameters[specification])
        for specification in LOCKED_COMPONENTS
    ])
    adjustment = _clip(shrinkage * corrections.mean(axis=1))
    result = confirmation[[
        "player_id", "player_name", "prediction_season", "target_ppg",
        "target_targets_per_game", "frozen_v1_predicted_ppg",
    ]].rename(columns={
        "prediction_season": "season", "target_ppg": "actual_ppg"
    })
    result["residual_correction"] = adjustment
    result["predicted_ppg"] = result["frozen_v1_predicted_ppg"] + adjustment
    if result["residual_correction"].abs().max() > CORRECTION_CAP + 1e-12:
        raise TEConfirmationError("Locked confirmation correction exceeded its cap")
    return result


def evaluate_confirmation(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fold_rows = []
    for season, fold in predictions.groupby("season"):
        candidate = _metrics(
            fold["actual_ppg"].to_numpy(), fold["predicted_ppg"].to_numpy(),
            fold["player_id"].to_numpy(), TOP_N["TE"],
        )
        frozen = _metrics(
            fold["actual_ppg"].to_numpy(),
            fold["frozen_v1_predicted_ppg"].to_numpy(),
            fold["player_id"].to_numpy(), TOP_N["TE"],
        )
        fold_rows.append({
            "season": season, "players": len(fold),
            "candidate_mae": candidate["mae"], "frozen_v1_mae": frozen["mae"],
            "mae_improvement": frozen["mae"] - candidate["mae"],
            "candidate_spearman": candidate["spearman"],
            "frozen_v1_spearman": frozen["spearman"],
            "spearman_change": candidate["spearman"] - frozen["spearman"],
        })
    folds = pd.DataFrame(fold_rows)
    candidate_abs = (predictions["predicted_ppg"] - predictions["actual_ppg"]).abs()
    frozen_abs = (
        predictions["frozen_v1_predicted_ppg"] - predictions["actual_ppg"]
    ).abs()
    improvement = frozen_abs - candidate_abs
    season_delta = improvement.groupby(predictions["season"]).mean()
    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap = np.asarray([
        rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
        for _ in range(100_000)
    ])

    tiered = predictions.copy()
    tiered["actual_role_tier"] = pd.cut(
        tiered["target_targets_per_game"], [-np.inf, 2, 4, np.inf],
        labels=["LOW_LT_2", "SECONDARY_2_TO_4", "FEATURED_GE_4"], right=False,
    )
    tiered["candidate_error"] = tiered["predicted_ppg"] - tiered["actual_ppg"]
    tiered["frozen_error"] = (
        tiered["frozen_v1_predicted_ppg"] - tiered["actual_ppg"]
    )
    tiers = tiered.groupby("actual_role_tier", observed=True, as_index=False).agg(
        players=("player_id", "size"),
        candidate_mean_bias=("candidate_error", "mean"),
        frozen_v1_mean_bias=("frozen_error", "mean"),
        candidate_mae=("candidate_error", lambda values: float(np.abs(values).mean())),
        frozen_v1_mae=("frozen_error", lambda values: float(np.abs(values).mean())),
    )
    tiers["bias_change"] = tiers["candidate_mean_bias"] - tiers["frozen_v1_mean_bias"]
    low_change = float(tiers.loc[
        tiers["actual_role_tier"] == "LOW_LT_2", "bias_change"
    ].iloc[0])
    featured_change = float(tiers.loc[
        tiers["actual_role_tier"] == "FEATURED_GE_4", "bias_change"
    ].iloc[0])
    mean_candidate_spearman = float(folds["candidate_spearman"].mean())
    mean_frozen_spearman = float(folds["frozen_v1_spearman"].mean())
    checks = {
        "mae_improvement": float(improvement.mean()),
        "mae_threshold_pass": bool(improvement.mean() >= MIN_MAE_IMPROVEMENT),
        "seasons_improved": int((season_delta > 0).sum()),
        "season_consistency_pass": bool(
            (season_delta > 0).sum() >= MIN_SEASONS_IMPROVED
        ),
        "mean_fold_spearman_change": mean_candidate_spearman - mean_frozen_spearman,
        "spearman_pass": bool(
            mean_candidate_spearman - mean_frozen_spearman >= MIN_SPEARMAN_CHANGE
        ),
        "low_tier_bias_change": low_change,
        "low_tier_bias_pass": bool(low_change <= MAX_LOW_TIER_BIAS_CHANGE),
        "featured_tier_bias_change": featured_change,
        "featured_tier_bias_pass": bool(
            featured_change >= MIN_FEATURED_TIER_BIAS_CHANGE
        ),
        "season_block_95_low": float(np.quantile(bootstrap, 0.025)),
        "season_block_95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_improvement": float((bootstrap > 0).mean()),
    }
    pass_fields = [key for key in checks if key.endswith("_pass")]
    checks["confirmation_decision"] = (
        "PASS" if all(checks[key] for key in pass_fields) else "FAIL"
    )
    return folds, tiers, checks


def _write_report(
    parameters: dict[str, dict[str, float]],
    shrinkage: float,
    folds: pd.DataFrame,
    tiers: pd.DataFrame,
    checks: dict[str, object],
) -> None:
    lines = [
        "LOCKED CONSTRAINED TE CONFIRMATION",
        "=" * 78, "",
        "The final three-component correction was locked using 2017-2025 only, then applied unchanged to the previously unused 2012-2016 confirmation seasons.",
        "This is a reverse-time holdout and cannot substitute for a prospective 2026 outcome test.", "",
        "Predeclared confirmation rules", "------------------------------",
        json.dumps(CONFIRMATION_RULES, indent=2), "",
        "Locked parameters", "-----------------",
        json.dumps({
            "components": LOCKED_COMPONENTS,
            "parameters": parameters,
            "combined_shrinkage": shrinkage,
        }, indent=2), "",
        "Confirmation checks", "-------------------", json.dumps(checks, indent=2), "",
        "Season results", "--------------", folds.to_string(index=False), "",
        "Role-tier results", "-----------------", tiers.to_string(index=False), "",
        "Replacement is authorized only when confirmation_decision is PASS. A PASS remains provisional because no untouched forward season currently exists.",
    ]
    (TE_CONSTRAINED_REPORTS_DIR / "te_constrained_confirmation_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_confirmation() -> dict[str, object]:
    TE_CONSTRAINED_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TE_CONSTRAINED_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    table = _load_table()
    parameters, shrinkage, tuning = lock_candidate(table)
    predictions = score_confirmation(table, parameters, shrinkage)
    folds, tiers, checks = evaluate_confirmation(predictions)
    predictions.to_parquet(
        TE_CONSTRAINED_PREDICTIONS_DATA_DIR / "te_confirmation_predictions.parquet",
        index=False,
    )
    tuning.to_csv(
        TE_CONSTRAINED_REPORT_TABLES_DIR / "confirmation_parameter_lock.csv",
        index=False,
    )
    folds.to_csv(
        TE_CONSTRAINED_REPORT_TABLES_DIR / "confirmation_fold_metrics.csv",
        index=False,
    )
    tiers.to_csv(
        TE_CONSTRAINED_REPORT_TABLES_DIR / "confirmation_role_tiers.csv",
        index=False,
    )
    (TE_CONSTRAINED_REPORT_TABLES_DIR / "confirmation_decision.json").write_text(
        json.dumps({
            "rules": CONFIRMATION_RULES,
            "locked_candidate": {
                "components": LOCKED_COMPONENTS,
                "parameters": parameters,
                "combined_shrinkage": shrinkage,
            },
            "checks": checks,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(parameters, shrinkage, folds, tiers, checks)
    return checks


if __name__ == "__main__":
    run_confirmation()
