"""Chronologically evaluate contract capital without changing deployed models."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.attempt3_models import _apply_market_calibration, _tune
from src.attempt21_features import V1_FEATURES
from src.attempt21_models import TOP_N, _metrics, build_pipeline
from src.config import (
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    CONTRACT_EXPERIMENT_PREDICTIONS_DATA_DIR,
    CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR,
    CONTRACT_EXPERIMENT_REPORT_TABLES_DIR,
    CONTRACT_EXPERIMENT_REPORTS_DIR,
    POSITIONS,
    RANDOM_STATE,
    SELECTED_PREDICTIONS_DATA_DIR,
)
from src.contract_features import (
    FOCAL_CONTRACT_FEATURES,
    LAGGED_CAP_FEATURES,
    ROOM_CONTRACT_FEATURES,
    TE_WEIGHTED_CONTRACT_FEATURES,
)
from src.selected_feature_models import (
    EVALUATION_START,
    SELECTED_SPECIFICATION,
    selected_feature_manifests,
)
from src.te_competition_experiment import _tune_residual_adjustment

BASE_REFIT = "CURRENT_BASE_REFIT"
TE_RESIDUAL_MANIFESTS = {
    "FROZEN_V1_PLUS_PRIOR_SEASON_CAP_RESIDUAL": LAGGED_CAP_FEATURES,
    "FROZEN_V1_PLUS_CONTRACT_ROOM_RESIDUAL": [
        *FOCAL_CONTRACT_FEATURES, *ROOM_CONTRACT_FEATURES,
    ],
    "FROZEN_V1_PLUS_RECEIVING_WEIGHTED_SALARY_RESIDUAL": [
        *LAGGED_CAP_FEATURES,
        *FOCAL_CONTRACT_FEATURES,
        *ROOM_CONTRACT_FEATURES,
        *TE_WEIGHTED_CONTRACT_FEATURES,
    ],
}


class ContractModelError(RuntimeError):
    """Raised when contract-model validation contracts fail."""


def candidate_manifests() -> dict[str, dict[str, list[str]]]:
    selected = selected_feature_manifests()
    output: dict[str, dict[str, list[str]]] = {}
    for position in POSITIONS:
        base = (
            list(V1_FEATURES["TE"])
            if position == "TE"
            else list(selected[position][SELECTED_SPECIFICATION])
        )
        output[position] = {
            BASE_REFIT: base,
            "BASE_PLUS_PRIOR_SEASON_CAP": [*base, *LAGGED_CAP_FEATURES],
            "BASE_PLUS_FOCAL_CONTRACT": [*base, *FOCAL_CONTRACT_FEATURES],
            "BASE_PLUS_CONTRACT_ROOM": [
                *base, *FOCAL_CONTRACT_FEATURES, *ROOM_CONTRACT_FEATURES,
            ],
            "BASE_PLUS_PRIOR_CAP_AND_CONTRACT_ROOM": [
                *base,
                *LAGGED_CAP_FEATURES,
                *FOCAL_CONTRACT_FEATURES,
                *ROOM_CONTRACT_FEATURES,
            ],
        }
        if position == "TE":
            output[position]["BASE_PLUS_RECEIVING_WEIGHTED_SALARY"] = [
                *base,
                *LAGGED_CAP_FEATURES,
                *FOCAL_CONTRACT_FEATURES,
                *ROOM_CONTRACT_FEATURES,
                *TE_WEIGHTED_CONTRACT_FEATURES,
            ]
    for specifications in output.values():
        for features in specifications.values():
            if len(features) != len(set(features)):
                raise ContractModelError("Duplicate feature in contract manifest")
    return output


def _load_tables() -> dict[str, pd.DataFrame]:
    return {
        position: pd.read_parquet(
            CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR
            / f"{position.lower()}_contract_modeling_dataset.parquet"
        ).sort_values(["prediction_season", "player_id"]).reset_index(drop=True)
        for position in POSITIONS
    }


def _uses_market(position: str, features: list[str]) -> bool:
    return position == "RB" and "market_implied_ppg" in features


def _run_full_refits(
    tables: dict[str, pd.DataFrame], manifests: dict[str, dict[str, list[str]]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    for position, specifications in manifests.items():
        table = tables[position]
        for specification, features in specifications.items():
            start = EVALUATION_START[position]
            for season in range(start, 2026):
                train = table[table["prediction_season"] < season]
                valid = table[table["prediction_season"] == season]
                label = "MARKET_ASSISTED" if _uses_market(position, features) else "STRUCTURAL_CLEANUP"
                method, params, tuning = _tune(train, features, position, label)
                for row in tuning.to_dict("records"):
                    tuning_rows.append({
                        "position": position, "outer_season": season,
                        "specification": specification, **row,
                    })
                calibration = {"alpha": math.nan, "beta": math.nan, "training_rows": 0}
                if _uses_market(position, features):
                    train, valid, calibration = _apply_market_calibration(train, valid)
                model = build_pipeline(method, params).fit(
                    train[features], train["target_ppg"]
                )
                predicted = model.predict(valid[features])
                metric = _metrics(
                    valid["target_ppg"].to_numpy(), predicted,
                    valid["player_id"].to_numpy(), TOP_N[position],
                )
                folds.append({
                    "position": position, "season": season,
                    "specification": specification, "method": method,
                    "alpha": params.get("alpha", math.nan),
                    "l1_ratio": params.get("l1_ratio", math.nan),
                    "training_rows": len(train),
                    "market_calibration_alpha": calibration["alpha"],
                    "market_calibration_beta": calibration["beta"],
                    **metric,
                })
                frame = valid[[
                    "player_id", "player_name", "position", "prediction_season",
                    "target_ppg",
                ]].rename(columns={
                    "prediction_season": "season", "target_ppg": "actual_ppg"
                })
                frame["predicted_ppg"] = predicted
                frame["specification"] = specification
                predictions.append(frame)
                for feature, coefficient in zip(
                    features, model.named_steps["regressor"].coef_, strict=True
                ):
                    coefficients.append({
                        "position": position, "season": season,
                        "specification": specification, "method": method,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                    })
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(folds),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficients),
    )


def _frozen_v1_te() -> pd.DataFrame:
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    return frozen[
        (frozen["position"] == "TE")
        & (frozen["model_name"] == "A_v1_same_cohort")
    ][["player_id", "position", "season", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "frozen_v1_predicted_ppg"}
    )


def _run_te_residuals(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frozen = _frozen_v1_te().rename(columns={"season": "prediction_season"})
    residual = table.merge(
        frozen.drop(columns="position"),
        on=["player_id", "prediction_season"],
        how="inner",
        validate="one_to_one",
    )
    residual["v1_residual"] = residual["target_ppg"] - residual["frozen_v1_predicted_ppg"]
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for specification, features in TE_RESIDUAL_MANIFESTS.items():
        for season in range(2017, 2026):
            history = residual[residual["prediction_season"] < season]
            valid = residual[residual["prediction_season"] == season]
            alpha, shrinkage, tuning = _tune_residual_adjustment(history, features)
            for row in tuning.to_dict("records"):
                tuning_rows.append({
                    "position": "TE", "outer_season": season,
                    "specification": specification,
                    "method": "frozen_v1_residual_ridge", **row,
                })
            if shrinkage == 0:
                predicted = valid["frozen_v1_predicted_ppg"].to_numpy()
            else:
                model = build_pipeline("ridge", {"alpha": alpha}).fit(
                    history[features], history["v1_residual"]
                )
                predicted = (
                    valid["frozen_v1_predicted_ppg"].to_numpy()
                    + shrinkage * model.predict(valid[features])
                )
            metric = _metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N["TE"],
            )
            fold_rows.append({
                "position": "TE", "season": season,
                "specification": specification,
                "method": "frozen_v1_residual_ridge", "alpha": alpha,
                "l1_ratio": math.nan, "residual_shrinkage": shrinkage, **metric,
            })
            frame = valid[[
                "player_id", "player_name", "position", "prediction_season",
                "target_ppg",
            ]].rename(columns={
                "prediction_season": "season", "target_ppg": "actual_ppg"
            })
            frame["predicted_ppg"] = predicted
            frame["specification"] = specification
            predictions.append(frame)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows), pd.DataFrame(tuning_rows)


def _operational_reference(position: str) -> tuple[str, pd.DataFrame]:
    if position == "TE":
        return "FROZEN_V1", _frozen_v1_te().rename(
            columns={"frozen_v1_predicted_ppg": "reference_predicted_ppg"}
        )
    selected = pd.read_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_predictions.parquet"
    )
    selected = selected[
        (selected["position"] == position)
        & (selected["specification"] == SELECTED_SPECIFICATION)
    ][["player_id", "position", "season", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "reference_predicted_ppg"}
    )
    return "SELECTED_MAIN", selected


def _comparison_row(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    position: str,
    specification: str,
    comparison: str,
) -> dict[str, object]:
    paired = candidate.merge(
        reference, on=["player_id", "position", "season"], validate="one_to_one"
    )
    current_abs = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
    reference_abs = (
        paired["reference_predicted_ppg"] - paired["actual_ppg"]
    ).abs()
    improvement = reference_abs - current_abs
    season_delta = improvement.groupby(paired["season"]).mean()
    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap = np.asarray([
        rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
        for _ in range(10_000)
    ])
    current_spearman = []
    reference_spearman = []
    for _, fold in paired.groupby("season"):
        actual = fold["actual_ppg"].to_numpy()
        ids = fold["player_id"].to_numpy()
        current_spearman.append(_metrics(
            actual, fold["predicted_ppg"].to_numpy(), ids, TOP_N[position]
        )["spearman"])
        reference_spearman.append(_metrics(
            actual, fold["reference_predicted_ppg"].to_numpy(), ids, TOP_N[position]
        )["spearman"])
    return {
        "position": position,
        "specification": specification,
        "comparison": comparison,
        "seasons": f"{paired['season'].min()}-{paired['season'].max()}",
        "folds": paired["season"].nunique(),
        "players": len(paired),
        "mae": float(current_abs.mean()),
        "reference_mae": float(reference_abs.mean()),
        "mae_improvement": float(improvement.mean()),
        "rmse": float(np.sqrt(np.mean(
            (paired["predicted_ppg"] - paired["actual_ppg"]) ** 2
        ))),
        "mean_fold_spearman": float(np.nanmean(current_spearman)),
        "reference_mean_fold_spearman": float(np.nanmean(reference_spearman)),
        "spearman_change": float(
            np.nanmean(current_spearman) - np.nanmean(reference_spearman)
        ),
        "seasons_improved": int((season_delta > 0).sum()),
        "season_block_95_low": float(np.quantile(bootstrap, 0.025)),
        "season_block_95_high": float(np.quantile(bootstrap, 0.975)),
    }


def compare_models(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position in POSITIONS:
        position_rows = predictions[predictions["position"] == position]
        base = position_rows[position_rows["specification"] == BASE_REFIT][[
            "player_id", "position", "season", "predicted_ppg"
        ]].rename(columns={"predicted_ppg": "reference_predicted_ppg"})
        operational_label, operational = _operational_reference(position)
        for specification, candidate in position_rows.groupby("specification"):
            if specification != BASE_REFIT:
                rows.append(_comparison_row(
                    candidate, base, position=position,
                    specification=specification, comparison="CURRENT_BASE_REFIT",
                ))
            rows.append(_comparison_row(
                candidate, operational, position=position,
                specification=specification, comparison=operational_label,
            ))
    return pd.DataFrame(rows)


def _te_bias_guardrail(
    predictions: pd.DataFrame, specification: str
) -> tuple[bool, bool]:
    candidate = predictions[
        (predictions["position"] == "TE")
        & (predictions["specification"] == specification)
    ]
    _, reference = _operational_reference("TE")
    paired = candidate.merge(
        reference, on=["player_id", "position", "season"], validate="one_to_one"
    )
    paired["candidate_error"] = paired["predicted_ppg"] - paired["actual_ppg"]
    paired["reference_error"] = paired["reference_predicted_ppg"] - paired["actual_ppg"]
    low = paired["actual_ppg"] < 2
    featured = paired["actual_ppg"] >= 4
    low_ok = (
        paired.loc[low, "candidate_error"].mean()
        <= paired.loc[low, "reference_error"].mean()
    )
    featured_ok = (
        paired.loc[featured, "candidate_error"].mean()
        >= paired.loc[featured, "reference_error"].mean()
    )
    return bool(low_ok), bool(featured_ok)


def decision_table(
    comparisons: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    operational = comparisons[
        comparisons["comparison"].isin(["SELECTED_MAIN", "FROZEN_V1"])
        & comparisons["specification"].ne(BASE_REFIT)
    ]
    rows = []
    for row in operational.itertuples(index=False):
        low_ok, featured_ok = (True, True)
        if row.position == "TE":
            low_ok, featured_ok = _te_bias_guardrail(predictions, row.specification)
        consistent = row.seasons_improved >= math.ceil(row.folds / 2)
        accepted = (
            row.mae_improvement > 0
            and row.season_block_95_low > 0
            and row.spearman_change >= 0
            and consistent
            and low_ok
            and featured_ok
        )
        rows.append({
            "position": row.position,
            "specification": row.specification,
            "decision": "PROMOTE" if accepted else "REJECT_RETAIN_CURRENT_MODEL",
            "mae_improvement": row.mae_improvement,
            "seasons_improved": row.seasons_improved,
            "folds": row.folds,
            "season_block_95_low": row.season_block_95_low,
            "spearman_change": row.spearman_change,
            "te_low_tier_bias_not_worse": low_ok,
            "te_featured_tier_bias_not_worse": featured_ok,
        })
    return pd.DataFrame(rows)


def _salary_signal_diagnostics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for position, table in tables.items():
        for cohort, mask in {
            "all": pd.Series(True, index=table.index),
            "non_rookie_contract": table["contract_is_rookie_deal"].ne(1),
            "rookie_contract": table["contract_is_rookie_deal"].eq(1),
        }.items():
            observed = table.loc[mask, [
                "contract_apy_cap_pct_at_signing", "target_ppg"
            ]].dropna()
            rows.append({
                "position": position,
                "cohort": cohort,
                "rows": len(observed),
                "spearman_contract_apy_vs_target_ppg": (
                    observed.corr(method="spearman").iloc[0, 1]
                    if len(observed) >= 3 else np.nan
                ),
            })
    return pd.DataFrame(rows)


def _write_report(
    comparisons: pd.DataFrame,
    decisions: pd.DataFrame,
    signal: pd.DataFrame,
) -> None:
    lines = [
        "CONTRACT CAPITAL ABLATION",
        "=" * 78,
        "",
        "This experiment does not modify any deployed model or prior attempt artifact.",
        "Only contracts signed by t-1 and still covering season t are eligible. Current active status and same-season contract rows are excluded.",
        "Realized t-1 cap spending, focal-player commitment, and same-position roster-room salary competition are tested for QB, RB, WR, and TE. TE additionally receives cross-fitted receiving-role-probability-weighted competitor salary features and frozen-V1 residual tests.",
        "",
        "Model comparisons",
        "-----------------",
        comparisons.to_string(index=False),
        "",
        "Promotion decisions",
        "-------------------",
        decisions.to_string(index=False),
        "",
        "Univariate signal diagnostic (descriptive, not a promotion test)",
        "----------------------------------------------------------------",
        signal.to_string(index=False),
        "",
        "Promotion requires positive paired MAE improvement, a positive lower bound of the season-block 95% interval, nonnegative fold-Spearman change, improvement in at least half of folds, and—only for TE—no worsening of low-tier overprediction or featured-tier underprediction.",
        "Same-season exclusions make this a conservative test: it cannot capture new free-agent deals or extensions signed during the prediction-year offseason.",
    ]
    CONTRACT_EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (CONTRACT_EXPERIMENT_REPORTS_DIR / "contract_experiment_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_contract_experiment() -> dict[str, str]:
    CONTRACT_EXPERIMENT_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_EXPERIMENT_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = _load_tables()
    manifests = candidate_manifests()
    predictions, folds, tuning, coefficients = _run_full_refits(tables, manifests)
    residual_predictions, residual_folds, residual_tuning = _run_te_residuals(tables["TE"])
    predictions = pd.concat([predictions, residual_predictions], ignore_index=True)
    folds = pd.concat([folds, residual_folds], ignore_index=True)
    tuning = pd.concat([tuning, residual_tuning], ignore_index=True)
    if predictions.duplicated(
        ["player_id", "position", "season", "specification"]
    ).any():
        raise ContractModelError("Duplicate contract experiment predictions")
    comparisons = compare_models(predictions)
    decisions = decision_table(comparisons, predictions)
    signal = _salary_signal_diagnostics(tables)

    predictions.to_parquet(
        CONTRACT_EXPERIMENT_PREDICTIONS_DATA_DIR / "contract_predictions.parquet",
        index=False,
    )
    coefficients.to_parquet(
        CONTRACT_EXPERIMENT_PREDICTIONS_DATA_DIR / "contract_coefficients.parquet",
        index=False,
    )
    folds.to_csv(CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "fold_metrics.csv", index=False)
    tuning.to_csv(CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "inner_tuning.csv", index=False)
    comparisons.to_csv(
        CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "model_comparison.csv", index=False
    )
    decisions.to_csv(
        CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "decisions.csv", index=False
    )
    signal.to_csv(
        CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "salary_signal_diagnostics.csv",
        index=False,
    )
    (CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "candidate_feature_sets.json").write_text(
        json.dumps({
            "full_refit": manifests,
            "te_frozen_v1_residual": TE_RESIDUAL_MANIFESTS,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(comparisons, decisions, signal)
    return {
        "summary": str(CONTRACT_EXPERIMENT_REPORTS_DIR / "contract_experiment_summary.txt")
    }


if __name__ == "__main__":
    run_contract_experiment()
