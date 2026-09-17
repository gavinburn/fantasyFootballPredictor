"""Predeclared constrained residual ablations for TE roster competition."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.attempt21_models import TOP_N, _metrics
from src.config import (
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TE_COMPETITION_PROCESSED_DATA_DIR,
    TE_CONSTRAINED_PREDICTIONS_DATA_DIR,
    TE_CONSTRAINED_PROCESSED_DATA_DIR,
    TE_CONSTRAINED_REPORT_TABLES_DIR,
    TE_CONSTRAINED_REPORTS_DIR,
)

FROZEN = "FROZEN_V1"
CENTERED_COUNT = "V1_PLUS_CENTERED_EFFECTIVE_COUNT_RESIDUAL"
HIGH_BURDEN = "V1_PLUS_HIGH_BURDEN_PENALTY_ONLY"
RELATIVE_ROOM = "V1_PLUS_RELATIVE_TE_ROOM_CONTROL"
MULTI_THRESHOLD = "V1_PLUS_MULTI_THRESHOLD_COMPETITION_BURDENS"
ASYMMETRIC = "V1_PLUS_ASYMMETRIC_RESIDUAL_CORRECTION"
COMBINED = "COMBINED_INNER_SELECTED_CONSTRAINED"
COMPONENTS = [
    CENTERED_COUNT, HIGH_BURDEN, RELATIVE_ROOM, MULTI_THRESHOLD, ASYMMETRIC,
]

EFFECTIVE_COUNT = "effective_receiving_competitor_count"
BURDEN = "probability_weighted_competitor_targets_per_game"
ROOM_SHARE = "focal_position_room_contract_share"
CORRECTION_CAP = 0.75
HIGH_BURDEN_THRESHOLD = 6.0
BURDEN_THRESHOLDS = [2.0, 4.0, 6.0]
COMBINED_MAX_COMPONENTS = 3
COMBINED_COMPONENT_MIN_INNER_IMPROVEMENT = 0.0005
SCREEN_MIN_MAE_IMPROVEMENT = 0.005
SCREEN_SPEARMAN_TOLERANCE = -0.005
SCREEN_LOW_BIAS_TOLERANCE = 0.02
SCREEN_FEATURED_BIAS_TOLERANCE = 0.05

SIGNED_GRID = [-0.40, -0.25, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
PENALTY_GRID = [0.0, 0.025, 0.05, 0.10, 0.15, 0.25]
MULTI_GRID = [0.0, 0.025, 0.05, 0.10]
COMBINED_SHRINKAGE = [0.0, 0.25, 0.50, 0.75, 1.0]

PREDECLARED_DEFINITIONS = {
    FROZEN: "Exact frozen V1 outer-fold prediction; no refit.",
    CENTERED_COUNT: (
        "Training-centered and scaled effective receiving-competitor count with a "
        "signed residual coefficient; missing values use the training median."
    ),
    HIGH_BURDEN: (
        "Nonnegative penalty only for probability-weighted competitor targets above "
        f"{HIGH_BURDEN_THRESHOLD:.0f} per game."
    ),
    RELATIVE_ROOM: (
        "Training-centered focal contract APY-cap share of the target-team TE room; "
        "no focal receiving-use statistic enters the correction."
    ),
    MULTI_THRESHOLD: (
        "Nonnegative additive penalties on probability-weighted competitor-target "
        "hinges at 2, 4, and 6 targets per game."
    ),
    ASYMMETRIC: (
        "A high-burden penalty gated toward V1 projections below 4 PPG and a "
        "separately tuned protective uplift gated toward projections above 4 PPG."
    ),
    COMBINED: (
        "Average of at most three components that improve inner-fold MAE by at least "
        "0.0005 PPG, with an inner-selected shrinkage multiplier."
    ),
    "global_constraint": "Every candidate residual correction is clipped to +/-0.75 PPG.",
}


class TEConstrainedExperimentError(RuntimeError):
    """Raised when a constrained TE experiment contract fails."""


def _frozen_predictions() -> pd.DataFrame:
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    return frozen[
        (frozen["position"] == "TE")
        & (frozen["model_name"] == "A_v1_same_cohort")
    ][["player_id", "season", "predicted_ppg"]].rename(columns={
        "season": "prediction_season",
        "predicted_ppg": "frozen_v1_predicted_ppg",
    })


def build_modeling_table() -> pd.DataFrame:
    competition = pd.read_parquet(
        TE_COMPETITION_PROCESSED_DATA_DIR / "te_competition_modeling_dataset.parquet"
    )
    contract = pd.read_parquet(
        CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR / "te_contract_modeling_dataset.parquet",
        columns=["player_id", "prediction_season", ROOM_SHARE],
    )
    table = competition.merge(
        contract, on=["player_id", "prediction_season"],
        how="left", validate="one_to_one",
    ).merge(
        _frozen_predictions(), on=["player_id", "prediction_season"],
        how="inner", validate="one_to_one",
    )
    required = [
        "player_id", "player_name", "prediction_season", "target_ppg",
        "target_targets_per_game", "frozen_v1_predicted_ppg", EFFECTIVE_COUNT,
        BURDEN, ROOM_SHARE,
    ]
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise TEConstrainedExperimentError(f"Modeling table is missing: {missing}")
    if table.duplicated(["player_id", "prediction_season"]).any():
        raise TEConstrainedExperimentError("Duplicate constrained-TE rows")
    table = table.sort_values(["prediction_season", "player_id"]).reset_index(drop=True)
    TE_CONSTRAINED_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    table[required].to_parquet(
        TE_CONSTRAINED_PROCESSED_DATA_DIR / "te_constrained_modeling_dataset.parquet",
        index=False,
    )
    return table[required]


def _training_scaled(
    train: pd.DataFrame, valid: pd.DataFrame, column: str
) -> np.ndarray:
    median = float(train[column].median())
    train_values = train[column].fillna(median).to_numpy(dtype=float)
    valid_values = valid[column].fillna(median).to_numpy(dtype=float)
    mean = float(train_values.mean())
    scale = float(train_values.std())
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (valid_values - mean) / scale


def _training_filled(
    train: pd.DataFrame, valid: pd.DataFrame, column: str
) -> np.ndarray:
    median = float(train[column].median())
    return valid[column].fillna(median).to_numpy(dtype=float)


def _clip(adjustment: np.ndarray) -> np.ndarray:
    return np.clip(adjustment, -CORRECTION_CAP, CORRECTION_CAP)


def _centered_count_adjustment(
    train: pd.DataFrame, valid: pd.DataFrame, params: dict[str, float]
) -> np.ndarray:
    return _clip(params["coefficient"] * _training_scaled(train, valid, EFFECTIVE_COUNT))


def _high_burden_adjustment(
    train: pd.DataFrame, valid: pd.DataFrame, params: dict[str, float]
) -> np.ndarray:
    burden = _training_filled(train, valid, BURDEN)
    return _clip(-params["penalty"] * np.maximum(0.0, burden - HIGH_BURDEN_THRESHOLD))


def _relative_room_adjustment(
    train: pd.DataFrame, valid: pd.DataFrame, params: dict[str, float]
) -> np.ndarray:
    return _clip(params["coefficient"] * _training_scaled(train, valid, ROOM_SHARE))


def _multi_threshold_adjustment(
    train: pd.DataFrame, valid: pd.DataFrame, params: dict[str, float]
) -> np.ndarray:
    burden = _training_filled(train, valid, BURDEN)
    correction = np.zeros(len(valid), dtype=float)
    for threshold in BURDEN_THRESHOLDS:
        correction -= params[f"penalty_{int(threshold)}"] * np.maximum(
            0.0, burden - threshold
        )
    return _clip(correction)


def _asymmetric_adjustment(
    train: pd.DataFrame, valid: pd.DataFrame, params: dict[str, float]
) -> np.ndarray:
    burden = _training_filled(train, valid, BURDEN)
    high_burden = np.maximum(0.0, burden - 4.0)
    frozen = valid["frozen_v1_predicted_ppg"].to_numpy(dtype=float)
    low_gate = np.clip((4.0 - frozen) / 4.0, 0.0, 1.0)
    featured_gate = np.clip((frozen - 4.0) / 4.0, 0.0, 1.0)
    correction = (
        -params["low_penalty"] * high_burden * low_gate
        + params["featured_uplift"] * high_burden * featured_gate
    )
    return _clip(correction)


ADJUSTERS: dict[
    str, Callable[[pd.DataFrame, pd.DataFrame, dict[str, float]], np.ndarray]
] = {
    CENTERED_COUNT: _centered_count_adjustment,
    HIGH_BURDEN: _high_burden_adjustment,
    RELATIVE_ROOM: _relative_room_adjustment,
    MULTI_THRESHOLD: _multi_threshold_adjustment,
    ASYMMETRIC: _asymmetric_adjustment,
}

PARAMETER_GRIDS: dict[str, list[dict[str, float]]] = {
    CENTERED_COUNT: [{"coefficient": value} for value in SIGNED_GRID],
    HIGH_BURDEN: [{"penalty": value} for value in PENALTY_GRID],
    RELATIVE_ROOM: [{"coefficient": value} for value in SIGNED_GRID],
    MULTI_THRESHOLD: [
        {"penalty_2": p2, "penalty_4": p4, "penalty_6": p6}
        for p2, p4, p6 in itertools.product(MULTI_GRID, repeat=3)
    ],
    ASYMMETRIC: [
        {"low_penalty": penalty, "featured_uplift": uplift}
        for penalty, uplift in itertools.product(PENALTY_GRID, repeat=2)
    ],
}


def _inner_seasons(history: pd.DataFrame) -> list[int]:
    seasons = sorted(history["prediction_season"].unique())
    eligible = [
        season for season in seasons
        if (history["prediction_season"] < season).sum() >= 100
        and len([prior for prior in seasons if prior < season]) >= 2
    ]
    result = eligible[-5:]
    if len(result) < 2:
        raise TEConstrainedExperimentError("Insufficient inner seasons")
    return result


def _parameter_complexity(params: dict[str, float]) -> float:
    return float(sum(abs(value) for value in params.values()))


def _tune_component(
    history: pd.DataFrame, specification: str, inner_seasons: list[int]
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    adjuster = ADJUSTERS[specification]
    for params in PARAMETER_GRIDS[specification]:
        fold_mae = []
        for season in inner_seasons:
            train = history[history["prediction_season"] < season]
            valid = history[history["prediction_season"] == season]
            predicted = (
                valid["frozen_v1_predicted_ppg"].to_numpy()
                + adjuster(train, valid, params)
            )
            fold_mae.append(float(np.mean(np.abs(predicted - valid["target_ppg"]))))
        rows.append({
            **params,
            "parameter_complexity": _parameter_complexity(params),
            "inner_mean_mae": float(np.mean(fold_mae)),
        })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "parameter_complexity"], kind="stable"
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    params = {
        key: float(tuning.loc[0, key])
        for key in PARAMETER_GRIDS[specification][0]
    }
    return params, tuning


def _inner_baseline_mae(history: pd.DataFrame, seasons: list[int]) -> float:
    return float(np.mean([
        np.mean(np.abs(
            history.loc[
                history["prediction_season"] == season,
                "frozen_v1_predicted_ppg",
            ]
            - history.loc[
                history["prediction_season"] == season, "target_ppg"
            ]
        ))
        for season in seasons
    ]))


def _tune_combined(
    history: pd.DataFrame,
    inner_seasons: list[int],
    best_params: dict[str, dict[str, float]],
    inner_scores: dict[str, float],
) -> tuple[list[str], float, pd.DataFrame]:
    baseline_mae = _inner_baseline_mae(history, inner_seasons)
    selected = [
        name for name in sorted(COMPONENTS, key=lambda item: inner_scores[item])
        if baseline_mae - inner_scores[name]
        >= COMBINED_COMPONENT_MIN_INNER_IMPROVEMENT
    ][:COMBINED_MAX_COMPONENTS]
    rows = []
    for shrinkage in COMBINED_SHRINKAGE:
        fold_mae = []
        for season in inner_seasons:
            train = history[history["prediction_season"] < season]
            valid = history[history["prediction_season"] == season]
            if selected:
                corrections = np.column_stack([
                    ADJUSTERS[name](train, valid, best_params[name])
                    for name in selected
                ])
                adjustment = _clip(shrinkage * corrections.mean(axis=1))
            else:
                adjustment = np.zeros(len(valid))
            predicted = valid["frozen_v1_predicted_ppg"].to_numpy() + adjustment
            fold_mae.append(float(np.mean(np.abs(predicted - valid["target_ppg"]))))
        rows.append({
            "combined_shrinkage": shrinkage,
            "inner_mean_mae": float(np.mean(fold_mae)),
            "selected_components": "|".join(selected) if selected else "NONE",
        })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "combined_shrinkage"], kind="stable"
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    return selected, float(tuning.loc[0, "combined_shrinkage"]), tuning


def run_models(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    folds = []
    tuning_rows = []
    selection_rows = []
    for season in range(2017, 2026):
        history = table[table["prediction_season"] < season]
        valid = table[table["prediction_season"] == season]
        inner_seasons = _inner_seasons(history)
        best_params: dict[str, dict[str, float]] = {}
        inner_scores: dict[str, float] = {}
        adjustments: dict[str, np.ndarray] = {}
        for specification in COMPONENTS:
            params, tuning = _tune_component(history, specification, inner_seasons)
            best_params[specification] = params
            inner_scores[specification] = float(tuning.loc[0, "inner_mean_mae"])
            adjustments[specification] = ADJUSTERS[specification](
                history, valid, params
            )
            for row in tuning.to_dict("records"):
                tuning_rows.append({
                    "outer_season": season, "specification": specification, **row
                })

        selected, shrinkage, combined_tuning = _tune_combined(
            history, inner_seasons, best_params, inner_scores
        )
        for row in combined_tuning.to_dict("records"):
            tuning_rows.append({
                "outer_season": season, "specification": COMBINED, **row
            })
        if selected:
            combined_adjustment = _clip(
                shrinkage
                * np.column_stack([adjustments[name] for name in selected]).mean(axis=1)
            )
        else:
            combined_adjustment = np.zeros(len(valid))
        adjustments[COMBINED] = combined_adjustment
        selection_rows.append({
            "outer_season": season,
            "inner_baseline_mae": _inner_baseline_mae(history, inner_seasons),
            "selected_components": "|".join(selected) if selected else "NONE",
            "combined_shrinkage": shrinkage,
        })

        adjustments[FROZEN] = np.zeros(len(valid))
        for specification in [FROZEN, *COMPONENTS, COMBINED]:
            predicted = (
                valid["frozen_v1_predicted_ppg"].to_numpy()
                + adjustments[specification]
            )
            metric = _metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N["TE"],
            )
            folds.append({
                "season": season, "specification": specification,
                "mean_correction": float(adjustments[specification].mean()),
                "max_absolute_correction": float(
                    np.abs(adjustments[specification]).max()
                ),
                **metric,
            })
            frame = valid[[
                "player_id", "player_name", "prediction_season", "target_ppg",
                "target_targets_per_game", "frozen_v1_predicted_ppg",
            ]].rename(columns={
                "prediction_season": "season", "target_ppg": "actual_ppg"
            })
            frame["predicted_ppg"] = predicted
            frame["residual_correction"] = adjustments[specification]
            frame["specification"] = specification
            predictions.append(frame)
    result = pd.concat(predictions, ignore_index=True)
    if result.duplicated(["player_id", "season", "specification"]).any():
        raise TEConstrainedExperimentError("Duplicate constrained predictions")
    if result["residual_correction"].abs().max() > CORRECTION_CAP + 1e-12:
        raise TEConstrainedExperimentError("Correction cap was violated")
    return (
        result, pd.DataFrame(folds), pd.DataFrame(tuning_rows),
        pd.DataFrame(selection_rows),
    )


def compare_to_frozen(
    predictions: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    frozen = predictions[predictions["specification"] == FROZEN][[
        "player_id", "season", "predicted_ppg"
    ]].rename(columns={"predicted_ppg": "reference_predicted_ppg"})
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for specification, candidate in predictions.groupby("specification"):
        paired = candidate.merge(frozen, on=["player_id", "season"], validate="one_to_one")
        candidate_abs = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
        frozen_abs = (
            paired["reference_predicted_ppg"] - paired["actual_ppg"]
        ).abs()
        improvement = frozen_abs - candidate_abs
        season_delta = improvement.groupby(paired["season"]).mean()
        bootstrap = np.asarray([
            rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
            for _ in range(10_000)
        ])
        candidate_spearman = folds[
            folds["specification"] == specification
        ]["spearman"].mean()
        frozen_spearman = folds[folds["specification"] == FROZEN]["spearman"].mean()
        rows.append({
            "specification": specification,
            "folds": paired["season"].nunique(), "players": len(paired),
            "candidate_mae": float(candidate_abs.mean()),
            "frozen_v1_mae": float(frozen_abs.mean()),
            "mae_improvement": float(improvement.mean()),
            "seasons_improved": int((season_delta > 0).sum()),
            "season_block_95_low": float(np.quantile(bootstrap, 0.025)),
            "season_block_95_high": float(np.quantile(bootstrap, 0.975)),
            "candidate_mean_fold_spearman": float(candidate_spearman),
            "frozen_v1_mean_fold_spearman": float(frozen_spearman),
            "spearman_change": float(candidate_spearman - frozen_spearman),
        })
    return pd.DataFrame(rows).sort_values("mae_improvement", ascending=False)


def tier_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    data["actual_role_tier"] = pd.cut(
        data["target_targets_per_game"], [-np.inf, 2, 4, np.inf],
        labels=["LOW_LT_2", "SECONDARY_2_TO_4", "FEATURED_GE_4"], right=False,
    )
    data["error"] = data["predicted_ppg"] - data["actual_ppg"]
    return data.groupby(
        ["specification", "actual_role_tier"], observed=True, as_index=False
    ).agg(
        players=("player_id", "size"), mean_bias=("error", "mean"),
        mae=("error", lambda values: float(np.abs(values).mean())),
        mean_correction=("residual_correction", "mean"),
    )


def decision_table(
    comparison: pd.DataFrame, tiers: pd.DataFrame
) -> pd.DataFrame:
    frozen = tiers[tiers["specification"] == FROZEN].set_index("actual_role_tier")
    rows = []
    for row in comparison[comparison["specification"] != FROZEN].itertuples(index=False):
        candidate = tiers[tiers["specification"] == row.specification].set_index(
            "actual_role_tier"
        )
        low_delta = (
            candidate.loc["LOW_LT_2", "mean_bias"]
            - frozen.loc["LOW_LT_2", "mean_bias"]
        )
        featured_delta = (
            candidate.loc["FEATURED_GE_4", "mean_bias"]
            - frozen.loc["FEATURED_GE_4", "mean_bias"]
        )
        low_ok = low_delta <= 0
        featured_ok = featured_delta >= 0
        consistency = row.seasons_improved >= math.ceil(row.folds / 2)
        promoted = (
            row.mae_improvement > 0
            and row.season_block_95_low > 0
            and row.spearman_change >= 0
            and consistency and low_ok and featured_ok
        )
        screen = (
            row.mae_improvement >= SCREEN_MIN_MAE_IMPROVEMENT
            and consistency
            and row.spearman_change >= SCREEN_SPEARMAN_TOLERANCE
            and low_delta <= SCREEN_LOW_BIAS_TOLERANCE
            and featured_delta >= -SCREEN_FEATURED_BIAS_TOLERANCE
        )
        rows.append({
            "specification": row.specification,
            "promotion_decision": (
                "PROMOTE" if promoted else "REJECT_RETAIN_FROZEN_V1"
            ),
            "research_screen": "ADVANCE" if screen else "DO_NOT_ADVANCE",
            "mae_improvement": row.mae_improvement,
            "seasons_improved": row.seasons_improved,
            "season_block_95_low": row.season_block_95_low,
            "spearman_change": row.spearman_change,
            "low_tier_bias_change": float(low_delta),
            "featured_tier_bias_change": float(featured_delta),
        })
    return pd.DataFrame(rows)


def _write_report(
    comparison: pd.DataFrame,
    decisions: pd.DataFrame,
    tiers: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    lines = [
        "CONSTRAINED TE COMPETITION RESIDUAL ABLATION",
        "=" * 78, "",
        "All candidates were defined before this run. Frozen V1 is never refit. Only cross-fitted season-t receiving-competition context, cutoff-safe relative TE-room contract context, and the frozen V1 projection enter corrections.",
        "No focal target rate, role probability, route participation, snap share, or mixture-model output enters a correction.", "",
        "Predeclared definitions", "-----------------------",
        json.dumps(PREDECLARED_DEFINITIONS, indent=2), "",
        "Model comparison", "----------------", comparison.to_string(index=False), "",
        "Decisions", "---------", decisions.to_string(index=False), "",
        "Combined inner selections", "-------------------------",
        selections.to_string(index=False), "",
        "Role-tier diagnostics", "---------------------", tiers.to_string(index=False), "",
        "Promotion retains the existing strict rule. The separate research screen uses the previously discussed practical tolerances and cannot promote a model.",
    ]
    TE_CONSTRAINED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (TE_CONSTRAINED_REPORTS_DIR / "te_constrained_competition_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_constrained_te_experiment() -> dict[str, str]:
    TE_CONSTRAINED_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TE_CONSTRAINED_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    table = build_modeling_table()
    predictions, folds, tuning, selections = run_models(table)
    comparison = compare_to_frozen(predictions, folds)
    tiers = tier_diagnostics(predictions)
    decisions = decision_table(comparison, tiers)
    predictions.to_parquet(
        TE_CONSTRAINED_PREDICTIONS_DATA_DIR / "te_constrained_predictions.parquet",
        index=False,
    )
    folds.to_csv(TE_CONSTRAINED_REPORT_TABLES_DIR / "fold_metrics.csv", index=False)
    tuning.to_csv(TE_CONSTRAINED_REPORT_TABLES_DIR / "inner_tuning.csv", index=False)
    selections.to_csv(
        TE_CONSTRAINED_REPORT_TABLES_DIR / "combined_component_selections.csv",
        index=False,
    )
    comparison.to_csv(
        TE_CONSTRAINED_REPORT_TABLES_DIR / "model_comparison.csv", index=False
    )
    tiers.to_csv(
        TE_CONSTRAINED_REPORT_TABLES_DIR / "role_tier_diagnostics.csv", index=False
    )
    decisions.to_csv(
        TE_CONSTRAINED_REPORT_TABLES_DIR / "decisions.csv", index=False
    )
    (TE_CONSTRAINED_REPORT_TABLES_DIR / "predeclared_manifest.json").write_text(
        json.dumps({
            "definitions": PREDECLARED_DEFINITIONS,
            "constants": {
                "correction_cap": CORRECTION_CAP,
                "high_burden_threshold": HIGH_BURDEN_THRESHOLD,
                "burden_thresholds": BURDEN_THRESHOLDS,
                "combined_max_components": COMBINED_MAX_COMPONENTS,
                "combined_min_inner_improvement": (
                    COMBINED_COMPONENT_MIN_INNER_IMPROVEMENT
                ),
            },
            "candidate_order": [FROZEN, *COMPONENTS, COMBINED],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(comparison, decisions, tiers, selections)
    return {
        "summary": str(
            TE_CONSTRAINED_REPORTS_DIR / "te_constrained_competition_summary.txt"
        )
    }


if __name__ == "__main__":
    run_constrained_te_experiment()
