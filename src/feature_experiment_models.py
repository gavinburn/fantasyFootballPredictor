"""Chronological ablation models for feature experiments 3 through 5."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.attempt3_models import _tune
from src.attempt21_models import TOP_N, _metrics, build_pipeline
from src.config import (
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    ATTEMPT3_PREDICTIONS_DATA_DIR,
    ATTEMPT21_PREDICTIONS_DATA_DIR,
    FEATURE35_PREDICTIONS_DATA_DIR,
    FEATURE35_PROCESSED_DATA_DIR,
    FEATURE35_REPORT_TABLES_DIR,
    FEATURE35_REPORTS_DIR,
    POSITIONS,
    RANDOM_STATE,
)
from src.feature_experiments import candidate_manifests


class FeatureExperimentModelError(RuntimeError):
    """Raised when experiment validation contracts fail."""


def evaluation_start(position: str, specification: str) -> int:
    """Return first fold with feature-aware chronological hyperparameter tuning."""

    if specification.startswith(("EXP5_PARTICIPATION", "EXP5_NGS")) or specification in {
        "EXP5_TARGETS_PER_PARTICIPATION_ONLY",
    }:
        return 2019
    if specification.startswith("EXP5_PFR"):
        return 2021
    if specification == "EXP5_ADVANCED_COMBINED":
        return 2019 if position == "TE" else 2021
    return 2012


def _load_tables() -> dict[str, pd.DataFrame]:
    return {
        position: pd.read_parquet(
            FEATURE35_PROCESSED_DATA_DIR
            / f"{position.lower()}_feature_experiments_dataset.parquet"
        ).sort_values(["prediction_season", "player_id"]).reset_index(drop=True)
        for position in POSITIONS
    }


def run_nested_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = _load_tables()
    manifests = candidate_manifests()
    predictions = []
    fold_rows = []
    tuning_rows = []
    coefficient_rows = []
    for position in POSITIONS:
        table = tables[position]
        for specification, features in manifests[position].items():
            if list(table[features].columns) != features:
                raise FeatureExperimentModelError(
                    f"Feature order changed for {position} {specification}"
                )
            for season in range(evaluation_start(position, specification), 2026):
                train = table[table["prediction_season"] < season]
                valid = table[table["prediction_season"] == season]
                method, params, tuning = _tune(
                    train, features, position, specification
                )
                for row in tuning.to_dict("records"):
                    tuning_rows.append({
                        "position": position, "outer_season": season,
                        "specification": specification, **row,
                    })
                model = build_pipeline(method, params).fit(
                    train[features], train["target_ppg"]
                )
                predicted = model.predict(valid[features])
                metrics = _metrics(
                    valid["target_ppg"].to_numpy(), predicted,
                    valid["player_id"].to_numpy(), TOP_N[position],
                )
                fold_rows.append({
                    "position": position, "season": season,
                    "specification": specification, "method": method, **params,
                    "training_start_season": int(train["prediction_season"].min()),
                    "training_end_season": int(train["prediction_season"].max()),
                    "training_rows": len(train), **metrics,
                })
                frame = valid[[
                    "player_id", "player_name", "position", "prediction_season", "target_ppg"
                ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
                frame["predicted_ppg"] = predicted
                frame["specification"] = specification
                predictions.append(frame)
                for feature, coefficient in zip(
                    features, model.named_steps["regressor"].coef_, strict=True
                ):
                    coefficient_rows.append({
                        "position": position, "season": season,
                        "specification": specification, "method": method,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                    })
    prediction_table = pd.concat(predictions, ignore_index=True)
    if prediction_table.duplicated(
        ["player_id", "position", "season", "specification"]
    ).any():
        raise FeatureExperimentModelError("Duplicate experiment predictions")
    folds = pd.DataFrame(fold_rows)
    if (folds["training_end_season"] >= folds["season"]).any():
        raise FeatureExperimentModelError("Outer-fold temporal leakage detected")
    return (
        prediction_table, folds, pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficient_rows),
    )


def _fold_rank_metrics(
    frame: pd.DataFrame, prediction_column: str, position: str
) -> tuple[float, float]:
    rows = []
    for _, fold in frame.groupby("season"):
        rows.append(_metrics(
            fold["actual_ppg"].to_numpy(), fold[prediction_column].to_numpy(),
            fold["player_id"].to_numpy(), TOP_N[position],
        ))
    return (
        float(np.nanmean([row["spearman"] for row in rows])),
        float(np.mean([row["top_n_overlap_rate"] for row in rows])),
    )


def compare_with_structural(predictions: pd.DataFrame) -> pd.DataFrame:
    structural = pd.read_parquet(
        ATTEMPT3_PREDICTIONS_DATA_DIR / "attempt3_predictions.parquet"
    )
    structural = structural[
        structural["specification"] == "STRUCTURAL_CLEANUP"
    ][["player_id", "position", "season", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "structural_predicted_ppg"}
    )
    attempt21 = pd.read_parquet(
        ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_predictions.parquet"
    )
    attempt21 = attempt21[
        attempt21["candidate"] == "ADAPTIVE_POSITION_SELECTED"
    ][["player_id", "position", "season", "predicted_ppg"]]
    frozen_v1 = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    frozen_v1 = frozen_v1[frozen_v1["model_name"] == "A_v1_same_cohort"][[
        "player_id", "position", "season", "predicted_ppg"
    ]]
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for (position, specification), candidate in predictions.groupby(
        ["position", "specification"]
    ):
        paired = candidate.merge(
            structural, on=["player_id", "position", "season"], validate="one_to_one"
        )
        operational = frozen_v1 if position == "TE" else attempt21
        operational_label = "FROZEN_V1" if position == "TE" else "ATTEMPT2_1_ADAPTIVE"
        paired = paired.merge(
            operational.rename(columns={"predicted_ppg": "operational_predicted_ppg"}),
            on=["player_id", "position", "season"], validate="one_to_one",
        )
        candidate_error = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
        structural_error = (
            paired["structural_predicted_ppg"] - paired["actual_ppg"]
        ).abs()
        operational_error = (
            paired["operational_predicted_ppg"] - paired["actual_ppg"]
        ).abs()
        improvement = structural_error - candidate_error
        season_delta = improvement.groupby(paired["season"]).mean()
        bootstrap = np.array([
            rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
            for _ in range(10_000)
        ])
        candidate_spearman, candidate_top = _fold_rank_metrics(
            paired, "predicted_ppg", position
        )
        structural_spearman, structural_top = _fold_rank_metrics(
            paired, "structural_predicted_ppg", position
        )
        operational_spearman, operational_top = _fold_rank_metrics(
            paired, "operational_predicted_ppg", position
        )
        candidate_rmse = float(np.sqrt(np.mean(
            (paired["predicted_ppg"] - paired["actual_ppg"]) ** 2
        )))
        structural_rmse = float(np.sqrt(np.mean(
            (paired["structural_predicted_ppg"] - paired["actual_ppg"]) ** 2
        )))
        operational_rmse = float(np.sqrt(np.mean(
            (paired["operational_predicted_ppg"] - paired["actual_ppg"]) ** 2
        )))
        rows.append({
            "position": position, "specification": specification,
            "seasons": f"{paired['season'].min()}-{paired['season'].max()}",
            "folds": paired["season"].nunique(), "players": len(paired),
            "candidate_mae": float(candidate_error.mean()),
            "structural_same_cohort_mae": float(structural_error.mean()),
            "mae_improvement": float(improvement.mean()),
            "operational_baseline": operational_label,
            "operational_same_cohort_mae": float(operational_error.mean()),
            "mae_improvement_vs_operational": float(
                (operational_error - candidate_error).mean()
            ),
            "candidate_rmse": candidate_rmse,
            "structural_same_cohort_rmse": structural_rmse,
            "rmse_improvement": structural_rmse - candidate_rmse,
            "operational_same_cohort_rmse": operational_rmse,
            "rmse_improvement_vs_operational": operational_rmse - candidate_rmse,
            "candidate_mean_fold_spearman": candidate_spearman,
            "structural_mean_fold_spearman": structural_spearman,
            "spearman_change": candidate_spearman - structural_spearman,
            "operational_mean_fold_spearman": operational_spearman,
            "spearman_change_vs_operational": candidate_spearman - operational_spearman,
            "candidate_mean_top_n_overlap": candidate_top,
            "structural_mean_top_n_overlap": structural_top,
            "top_n_overlap_change": candidate_top - structural_top,
            "operational_mean_top_n_overlap": operational_top,
            "top_n_overlap_change_vs_operational": candidate_top - operational_top,
            "seasons_improved": int((season_delta > 0).sum()),
            "season_block_95_low": float(np.quantile(bootstrap, 0.025)),
            "season_block_95_high": float(np.quantile(bootstrap, 0.975)),
        })
    return pd.DataFrame(rows).sort_values(
        ["position", "mae_improvement"], ascending=[True, False]
    ).reset_index(drop=True)


def decision_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in summary.itertuples(index=False):
        consistency_threshold = math.ceil(row.folds / 2)
        if row.mae_improvement_vs_operational <= 0:
            decision = "REJECT_DOES_NOT_BEAT_CURRENT_MAIN"
        elif row.mae_improvement <= 0:
            decision = "REJECT_NO_INCREMENTAL_MAE_GAIN"
        elif row.season_block_95_low > 0 and row.seasons_improved >= consistency_threshold:
            if row.spearman_change >= 0:
                decision = "ADVANCE_TO_MAIN_SET_REVIEW"
            elif row.spearman_change > -0.005 and row.top_n_overlap_change >= 0:
                decision = "ADVANCE_MAE_WITH_MINOR_RANK_TRADEOFF"
            else:
                decision = "PROMISING_RANK_TRADEOFF"
        elif row.spearman_change < 0:
            decision = "INCONCLUSIVE_RANKING_REGRESSION"
        else:
            decision = "PROMISING_NOT_CONFIRMED"
        if row.folds < 8 and decision == "ADVANCE_TO_MAIN_SET_REVIEW":
            decision = "PROMISING_LIMITED_HISTORY"
        if row.position == "WR" and row.specification in {
            "EXP5_PFR", "EXP5_PFR_AVAILABILITY_ONLY"
        }:
            decision = "REJECT_GAIN_DRIVEN_BY_SOURCE_AVAILABILITY"
        if row.position == "WR" and row.specification == "EXP5_PARTICIPATION":
            decision = "SUPERSEDED_BY_NO_FLAG_VARIANT"
        if row.specification.endswith("AVAILABILITY_ONLY"):
            decision = "REJECT_SOURCE_AVAILABILITY_PROXY"
        rows.append({
            "position": row.position, "specification": row.specification,
            "decision": decision, "folds": row.folds,
            "mae_improvement": row.mae_improvement,
            "mae_improvement_vs_operational": row.mae_improvement_vs_operational,
            "spearman_change": row.spearman_change,
            "top_n_overlap_change": row.top_n_overlap_change,
            "seasons_improved": row.seasons_improved,
            "season_block_95_low": row.season_block_95_low,
            "season_block_95_high": row.season_block_95_high,
        })
    return pd.DataFrame(rows)


def stability_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    return coefficients.groupby(
        ["position", "specification", "feature"], as_index=False
    )["standardized_coefficient"].agg(
        folds="size", mean="mean", std="std", minimum="min", maximum="max"
    ).assign(
        sign_changed=lambda d: (d["minimum"] < 0) & (d["maximum"] > 0)
    )


def _write_report(summary: pd.DataFrame, decisions: pd.DataFrame) -> None:
    advanced = decisions[
        decisions["decision"].isin([
            "ADVANCE_TO_MAIN_SET_REVIEW", "ADVANCE_MAE_WITH_MINOR_RANK_TRADEOFF",
            "PROMISING_LIMITED_HISTORY",
        ])
    ]
    lines = [
        "FEATURE EXPERIMENTS 3, 4, AND 5",
        "=" * 78, "",
        "Design",
        "------",
        "Every candidate is an isolated addition to the Attempt 3 structural foundation.",
        "Candidates are also paired against the current operational reference: Attempt 2.1 for QB/RB/WR and frozen V1 for TE.",
        "No candidate has been added to the main feature set or promoted for deployment.",
        "Rookies remain excluded from training and evaluation.",
        "Opportunity and vacated-workload models use outer folds 2012-2025.",
        "Participation and NGS models use 2019-2025 so inner tuning has prior feature-bearing folds.",
        "PFR and PFR-containing combined models use 2021-2025; TE advanced combined starts in 2019 because it has no PFR group.", "",
        "Important source correction",
        "---------------------------",
        "Participation data does not provide every player's route on a play. The experiment uses pass-play participation and targets per participated pass play, not falsely labeled routes run or TPRR.",
        "Inline blocking share is unavailable and was omitted.", "",
        "All paired candidate results versus the structural foundation",
        "---------------------------------------------------------",
        summary.to_string(index=False), "",
        "Decision screen",
        "---------------",
        decisions.to_string(index=False), "",
        "Candidates advancing to main-set review",
        "---------------------------------------",
        advanced.to_string(index=False) if not advanced.empty else "None", "",
        "Interpretation",
        "--------------",
        "Positive improvement means the candidate reduced error. Candidates are screened on MAE, rank direction, season consistency, and season-block uncertainty.",
        "Recent-era results remain less certain because they contain fewer outer seasons.",
    ]
    FEATURE35_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (FEATURE35_REPORTS_DIR / "feature_experiments_3_5_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_feature_experiment_models() -> dict[str, str]:
    predictions, folds, tuning, coefficients = run_nested_experiments()
    summary = compare_with_structural(predictions)
    decisions = decision_table(summary)
    stability = stability_summary(coefficients)
    FEATURE35_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE35_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        FEATURE35_PREDICTIONS_DATA_DIR / "feature_experiment_predictions.parquet",
        index=False,
    )
    coefficients.to_parquet(
        FEATURE35_PREDICTIONS_DATA_DIR / "feature_experiment_coefficients.parquet",
        index=False,
    )
    folds.to_csv(FEATURE35_REPORT_TABLES_DIR / "fold_metrics.csv", index=False)
    tuning.to_csv(FEATURE35_REPORT_TABLES_DIR / "inner_tuning.csv", index=False)
    summary.to_csv(FEATURE35_REPORT_TABLES_DIR / "experiment_comparison.csv", index=False)
    decisions.to_csv(FEATURE35_REPORT_TABLES_DIR / "experiment_decisions.csv", index=False)
    stability.to_csv(FEATURE35_REPORT_TABLES_DIR / "coefficient_stability.csv", index=False)
    _write_report(summary, decisions)
    return {"summary": str(FEATURE35_REPORTS_DIR / "feature_experiments_3_5_summary.txt")}


if __name__ == "__main__":
    run_feature_experiment_models()
