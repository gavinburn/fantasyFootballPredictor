"""Leakage-safe chronological evaluation for Attempt 3."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.attempt3_features import BASE_SPECIFICATIONS, feature_manifests
from src.attempt21_models import (
    ELASTIC_ALPHAS,
    ELASTIC_L1,
    RIDGE_ALPHAS,
    TOP_N,
    _allowed_methods,
    _inner_seasons,
    _metrics,
    build_pipeline,
)
from src.config import (
    ATTEMPT3_MODELS_DIR,
    ATTEMPT3_PREDICTIONS_DATA_DIR,
    ATTEMPT3_PROCESSED_DATA_DIR,
    ATTEMPT3_REPORT_TABLES_DIR,
    ATTEMPT3_REPORTS_DIR,
    ATTEMPT21_PREDICTIONS_DATA_DIR,
    ATTEMPT21_REPORT_TABLES_DIR,
    POSITIONS,
    RANDOM_STATE,
)

METHOD_GRID = [
    ("ols", {}),
    *[("ridge", {"alpha": alpha}) for alpha in RIDGE_ALPHAS],
    *[
        ("elastic_net", {"alpha": alpha, "l1_ratio": ratio})
        for alpha in ELASTIC_ALPHAS
        for ratio in ELASTIC_L1
    ],
]
EVALUATION_START = {
    "VETERAN_ALIGNED_BASE": 2012,
    "STRUCTURAL_CLEANUP": 2012,
    # 2020 is the first market season and supplies the first training examples.
    "MARKET_ASSISTED": 2021,
}


class Attempt3ModelError(RuntimeError):
    """Raised when Attempt 3 chronological-validation contracts fail."""


def _fit_market_power(train: pd.DataFrame) -> dict[str, float]:
    observed = train[["preseason_ecr", "target_ppg"]].dropna()
    observed = observed[(observed["preseason_ecr"] > 0) & (observed["target_ppg"] >= 0)]
    if len(observed) < 20:
        return {"alpha": math.nan, "beta": math.nan, "training_rows": len(observed)}
    rank = observed["preseason_ecr"].to_numpy(dtype=float)
    target = observed["target_ppg"].to_numpy(dtype=float)
    best: tuple[float, float, float] | None = None
    for beta in np.linspace(0.05, 1.50, 146):
        basis = rank ** (-beta)
        # For absolute error, the optimal no-intercept multiplier is the
        # weighted median of y/x with weights x.
        ratio = target / basis
        order = np.argsort(ratio)
        cumulative = np.cumsum(basis[order])
        alpha = float(ratio[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
        mae = float(np.mean(np.abs(alpha * basis - target)))
        if best is None or mae < best[0]:
            best = (mae, alpha, float(beta))
    assert best is not None
    return {"alpha": best[1], "beta": best[2], "training_rows": len(observed)}


def _apply_market_calibration(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    calibration = _fit_market_power(train)
    train = train.copy()
    valid = valid.copy()
    if pd.isna(calibration["alpha"]):
        train["market_implied_ppg"] = np.nan
        valid["market_implied_ppg"] = np.nan
    else:
        for frame in (train, valid):
            frame["market_implied_ppg"] = calibration["alpha"] * frame[
                "preseason_ecr"
            ].pow(-calibration["beta"])
    return train, valid, calibration


def _fold_data(
    table: pd.DataFrame, train_before: int, valid_season: int, specification: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train = table[table["prediction_season"] < train_before]
    valid = table[table["prediction_season"] == valid_season]
    if train.empty or valid.empty:
        raise Attempt3ModelError(f"Empty fold for {specification} season {valid_season}")
    if specification == "MARKET_ASSISTED":
        return _apply_market_calibration(train, valid)
    return train, valid, {"alpha": math.nan, "beta": math.nan, "training_rows": 0}


def _tune(
    outer_train: pd.DataFrame,
    features: list[str],
    position: str,
    specification: str,
) -> tuple[str, dict[str, float], pd.DataFrame]:
    rows = []
    allowed = _allowed_methods(position, BASE_SPECIFICATIONS[position])
    for method, params in METHOD_GRID:
        if method not in allowed:
            continue
        fold_metrics = []
        for season in _inner_seasons(outer_train):
            train = outer_train[outer_train["prediction_season"] < season]
            valid = outer_train[outer_train["prediction_season"] == season]
            if specification == "MARKET_ASSISTED":
                train, valid, _ = _apply_market_calibration(train, valid)
            model = build_pipeline(method, params).fit(train[features], train["target_ppg"])
            predicted = model.predict(valid[features])
            fold_metrics.append(_metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N[position],
            ))
        spearman_values = [
            row["spearman"] for row in fold_metrics if pd.notna(row["spearman"])
        ]
        rows.append({
            "method": method, **params,
            "inner_mean_mae": float(np.mean([row["mae"] for row in fold_metrics])),
            "inner_mean_spearman": float(np.mean(spearman_values)) if spearman_values else -1.0,
        })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "inner_mean_spearman", "method"],
        ascending=[True, False, True], kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    selected = tuning.iloc[0]
    params = {
        key: float(selected[key]) for key in ("alpha", "l1_ratio")
        if key in selected and pd.notna(selected[key])
    }
    return str(selected["method"]), params, tuning


def _load_tables() -> dict[str, pd.DataFrame]:
    return {
        position: pd.read_parquet(
            ATTEMPT3_PROCESSED_DATA_DIR / f"{position.lower()}_attempt3_modeling_dataset.parquet"
        ).sort_values(["prediction_season", "player_id"]).reset_index(drop=True)
        for position in POSITIONS
    }


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifests = feature_manifests()
    tables = _load_tables()
    prediction_frames = []
    fold_rows = []
    tuning_rows = []
    coefficient_rows = []
    for position in POSITIONS:
        table = tables[position]
        if table["previous_season_ppg"].isna().any():
            raise Attempt3ModelError("No-history rows remained in an Attempt 3 table")
        for specification, features in manifests[position].items():
            for season in range(EVALUATION_START[specification], 2026):
                outer_train = table[table["prediction_season"] < season]
                outer_valid = table[table["prediction_season"] == season]
                method, params, tuning = _tune(
                    outer_train, features, position, specification
                )
                for row in tuning.to_dict("records"):
                    tuning_rows.append({
                        "position": position, "outer_season": season,
                        "specification": specification, **row,
                    })
                calibration = {"alpha": math.nan, "beta": math.nan, "training_rows": 0}
                if specification == "MARKET_ASSISTED":
                    outer_train, outer_valid, calibration = _apply_market_calibration(
                        outer_train, outer_valid
                    )
                model = build_pipeline(method, params).fit(
                    outer_train[features], outer_train["target_ppg"]
                )
                predicted = model.predict(outer_valid[features])
                metrics = _metrics(
                    outer_valid["target_ppg"].to_numpy(), predicted,
                    outer_valid["player_id"].to_numpy(), TOP_N[position],
                )
                fold_rows.append({
                    "position": position, "season": season,
                    "specification": specification, "method": method, **params,
                    "training_rows": len(outer_train),
                    "market_calibration_alpha": calibration["alpha"],
                    "market_calibration_beta": calibration["beta"],
                    "market_calibration_rows": calibration["training_rows"],
                    **metrics,
                })
                frame = outer_valid[[
                    "player_id", "player_name", "position", "prediction_season", "target_ppg"
                ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
                frame["predicted_ppg"] = predicted
                frame["specification"] = specification
                prediction_frames.append(frame)
                for feature, coefficient in zip(features, model.named_steps["regressor"].coef_, strict=True):
                    coefficient_rows.append({
                        "position": position, "season": season,
                        "specification": specification, "method": method,
                        "feature": feature, "standardized_coefficient": float(coefficient),
                    })
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(fold_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficient_rows),
    )


def _paired_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_predictions.parquet")
    frozen = frozen[frozen["candidate"] == "ADAPTIVE_POSITION_SELECTED"][[
        "player_id", "position", "season", "predicted_ppg"
    ]].rename(columns={"predicted_ppg": "attempt21_predicted_ppg"})
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    for (position, specification), current in predictions.groupby(["position", "specification"]):
        paired = current.merge(frozen, on=["player_id", "position", "season"], validate="one_to_one")
        current_abs = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
        frozen_abs = (paired["attempt21_predicted_ppg"] - paired["actual_ppg"]).abs()
        improvement = frozen_abs - current_abs
        season_delta = improvement.groupby(paired["season"]).mean()
        boot = np.array([
            rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
            for _ in range(10_000)
        ])
        current_fold_metrics = []
        frozen_fold_metrics = []
        for _, fold in paired.groupby("season"):
            actual = fold["actual_ppg"].to_numpy()
            ids = fold["player_id"].to_numpy()
            current_fold_metrics.append(_metrics(
                actual, fold["predicted_ppg"].to_numpy(), ids, TOP_N[position]
            ))
            frozen_fold_metrics.append(_metrics(
                actual, fold["attempt21_predicted_ppg"].to_numpy(), ids, TOP_N[position]
            ))
        rows.append({
            "position": position, "specification": specification,
            "seasons": f"{paired['season'].min()}-{paired['season'].max()}",
            "folds": paired["season"].nunique(), "players": len(paired),
            "mae": float(current_abs.mean()), "attempt21_same_cohort_mae": float(frozen_abs.mean()),
            "mae_improvement_vs_attempt21": float(improvement.mean()),
            "rmse": float(np.sqrt(np.mean((paired["predicted_ppg"] - paired["actual_ppg"]) ** 2))),
            "attempt21_same_cohort_rmse": float(np.sqrt(np.mean((paired["attempt21_predicted_ppg"] - paired["actual_ppg"]) ** 2))),
            "mean_fold_spearman": float(np.nanmean([row["spearman"] for row in current_fold_metrics])),
            "attempt21_mean_fold_spearman": float(np.nanmean([row["spearman"] for row in frozen_fold_metrics])),
            "mean_fold_top_n_overlap": float(np.mean([row["top_n_overlap_rate"] for row in current_fold_metrics])),
            "attempt21_mean_fold_top_n_overlap": float(np.mean([row["top_n_overlap_rate"] for row in frozen_fold_metrics])),
            "seasons_improved": int((season_delta > 0).sum()),
            "season_block_95_low": float(np.quantile(boot, 0.025)),
            "season_block_95_high": float(np.quantile(boot, 0.975)),
        })
    return pd.DataFrame(rows)


def _comparison_row(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    position: str,
    comparison: str,
) -> dict[str, object]:
    paired = candidate.merge(
        reference[["player_id", "season", "predicted_ppg"]].rename(
            columns={"predicted_ppg": "reference_predicted_ppg"}
        ),
        on=["player_id", "season"], validate="one_to_one",
    )
    candidate_error = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
    reference_error = (
        paired["reference_predicted_ppg"] - paired["actual_ppg"]
    ).abs()
    improvement = reference_error - candidate_error
    season_delta = improvement.groupby(paired["season"]).mean()
    rng = np.random.default_rng(RANDOM_STATE)
    boot = np.array([
        rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
        for _ in range(10_000)
    ])
    return {
        "position": position, "comparison": comparison,
        "seasons": f"{paired['season'].min()}-{paired['season'].max()}",
        "folds": paired["season"].nunique(), "players": len(paired),
        "candidate_mae": float(candidate_error.mean()),
        "reference_mae": float(reference_error.mean()),
        "mae_improvement": float(improvement.mean()),
        "seasons_improved": int((season_delta > 0).sum()),
        "season_block_95_low": float(np.quantile(boot, 0.025)),
        "season_block_95_high": float(np.quantile(boot, 0.975)),
    }


def _fixed_attempt21_base_predictions(position: str) -> pd.DataFrame:
    metrics = pd.read_csv(ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_fold_metrics.csv")
    predictions = pd.read_parquet(
        ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_predictions.parquet"
    )
    specification = BASE_SPECIFICATIONS[position]
    selected_frames = []
    for season in range(2012, 2026):
        choice = metrics[
            (metrics["position"] == position)
            & (metrics["season"] == season)
            & (metrics["specification"] == specification)
        ].sort_values(
            ["inner_selection_mae", "inner_selection_spearman", "method", "candidate"],
            ascending=[True, False, True, True],
        ).iloc[0]
        selected_frames.append(predictions[
            (predictions["position"] == position)
            & (predictions["season"] == season)
            & (predictions["candidate"] == choice["candidate"])
        ][["player_id", "season", "predicted_ppg"]])
    return pd.concat(selected_frames, ignore_index=True)


def ablation_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position in POSITIONS:
        current = predictions[predictions["position"] == position]
        base = current[current["specification"] == "VETERAN_ALIGNED_BASE"]
        structural = current[current["specification"] == "STRUCTURAL_CLEANUP"]
        market = current[current["specification"] == "MARKET_ASSISTED"]
        rows.append(_comparison_row(
            base, _fixed_attempt21_base_predictions(position), position=position,
            comparison="veteran_aligned_base_vs_same_fixed_base_with_no_history_training_rows",
        ))
        rows.append(_comparison_row(
            structural, base, position=position,
            comparison="structural_cleanup_vs_veteran_aligned_base",
        ))
        structural_recent = structural[structural["season"].isin(market["season"])]
        rows.append(_comparison_row(
            market, structural_recent, position=position,
            comparison="market_assisted_vs_structural_cleanup",
        ))
    return pd.DataFrame(rows)


def _fit_research_models(
    tables: dict[str, pd.DataFrame], manifests: dict[str, dict[str, list[str]]],
    folds: pd.DataFrame,
) -> pd.DataFrame:
    ATTEMPT3_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for position in POSITIONS:
        for specification in ("STRUCTURAL_CLEANUP", "MARKET_ASSISTED"):
            recent = folds[(folds["position"] == position) & (folds["specification"] == specification)]
            method = Counter(recent["method"]).most_common(1)[0][0]
            method_rows = recent[recent["method"] == method]
            params: dict[str, float] = {}
            for key in ("alpha", "l1_ratio"):
                values = method_rows[key].dropna() if key in method_rows else pd.Series(dtype=float)
                if not values.empty:
                    params[key] = float(values.mode().iloc[0])
            table = tables[position].copy()
            calibration = {"alpha": math.nan, "beta": math.nan, "training_rows": 0}
            if specification == "MARKET_ASSISTED":
                table, _, calibration = _apply_market_calibration(table, table.iloc[:0].copy())
            features = manifests[position][specification]
            model = build_pipeline(method, params).fit(table[features], table["target_ppg"])
            artifact: dict[str, Any] = {
                "attempt": "3_research", "position": position,
                "specification": specification, "method": method,
                "parameters": params, "features": features,
                "market_calibration": calibration, "pipeline": model,
                "rookies_supported": False,
            }
            path = ATTEMPT3_MODELS_DIR / f"{position.lower()}_{specification.lower()}.joblib"
            joblib.dump(artifact, path)
            rows.append({
                "position": position, "specification": specification,
                "method": method, **params, "model_path": str(path),
                "promotion_status": "research_only_pending_results_review",
            })
    return pd.DataFrame(rows)


def _write_summary(
    summary: pd.DataFrame,
    ablations: pd.DataFrame,
    decisions: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    lines = [
        "ATTEMPT 3: VETERAN STRUCTURAL CLEANUP AND PRESEASON MARKET EXPERIMENT",
        "=" * 76, "",
        "Scope",
        "-----",
        "Rookies and all players without previous-season NFL PPG remain excluded from predictions.",
        "They are now also excluded from every veteran-model training fold.",
        "Attempt 1, Attempt 2, and Attempt 2.1 files were not overwritten.", "",
        "Market limitation",
        "-----------------",
        "The available FantasyPros archive supplies preseason PPR ECR for 2020-2025 only.",
        "The market model is therefore first tested in 2021, using 2020 as its first prior training season.",
        "It is market-assisted, not a football-data-only model, and its PPR rankings are used to predict standard PPG.", "",
        "Paired results versus Attempt 2.1",
        "---------------------------------",
        summary.to_string(index=False), "",
        "Direct ablation comparisons",
        "---------------------------",
        ablations.to_string(index=False), "",
        "Decisions",
        "---------",
        decisions.to_string(index=False), "",
        "Market matching coverage",
        "------------------------",
        coverage[coverage["prediction_season"].between(2020, 2025)].to_string(index=False), "",
        "Interpretation rule",
        "-------------------",
        "Positive MAE improvement favors Attempt 3. A confidence interval whose lower bound exceeds zero is stronger evidence.",
        "Market results have only five outer seasons and must be treated as preliminary even when average MAE improves.",
    ]
    ATTEMPT3_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (ATTEMPT3_REPORTS_DIR / "attempt3_evaluation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decision_table(summary: pd.DataFrame, ablations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position in POSITIONS:
        structural = summary[
            (summary["position"] == position)
            & (summary["specification"] == "STRUCTURAL_CLEANUP")
        ].iloc[0]
        market = ablations[
            (ablations["position"] == position)
            & (ablations["comparison"] == "market_assisted_vs_structural_cleanup")
        ].iloc[0]
        structural_decision = (
            "ACCEPT_RESEARCH_CANDIDATE"
            if structural["mae_improvement_vs_attempt21"] > 0
            and structural["season_block_95_low"] > 0
            else "REJECT_RETAIN_ATTEMPT2_1_OR_FROZEN_V1"
        )
        if market["mae_improvement"] <= 0:
            market_decision = "REJECT"
        elif market["season_block_95_low"] > 0:
            market_decision = "PROMISING_NOT_PROMOTED_LIMITED_HISTORY"
        else:
            market_decision = "INCONCLUSIVE_LIMITED_HISTORY"
        rows.extend([
            {
                "position": position, "stage": "structural_cleanup",
                "decision": structural_decision,
                "mae_improvement": structural["mae_improvement_vs_attempt21"],
                "ci_low": structural["season_block_95_low"],
                "ci_high": structural["season_block_95_high"],
                "note": (
                    "TE remains compared operationally with frozen V1 because Attempt 2.1 TE was rejected"
                    if position == "TE" else "Compared with the frozen Attempt 2.1 outer predictions"
                ),
            },
            {
                "position": position, "stage": "market_assisted",
                "decision": market_decision,
                "mae_improvement": market["mae_improvement"],
                "ci_low": market["season_block_95_low"],
                "ci_high": market["season_block_95_high"],
                "note": "Only five outer seasons; PPR ECR used for a standard-scoring target",
            },
        ])
    # Attempt 2.1 TE was not the deployed comparator, so it cannot be promoted
    # merely by beating that rejected model.
    for row in rows:
        if row["position"] == "TE" and row["stage"] == "structural_cleanup":
            row["decision"] = "REJECT_RETAIN_FROZEN_V1"
    return pd.DataFrame(rows)


def run_attempt3_models() -> dict[str, str]:
    predictions, folds, tuning, coefficients = run_evaluation()
    summary = _paired_summary(predictions)
    ablations = ablation_summary(predictions)
    decisions = decision_table(summary, ablations)
    tables = _load_tables()
    manifests = feature_manifests()
    deployment = _fit_research_models(tables, manifests, folds)
    coverage = pd.read_csv(ATTEMPT3_REPORT_TABLES_DIR / "market_match_coverage.csv")

    ATTEMPT3_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ATTEMPT3_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(ATTEMPT3_PREDICTIONS_DATA_DIR / "attempt3_predictions.parquet", index=False)
    coefficients.to_parquet(ATTEMPT3_PREDICTIONS_DATA_DIR / "attempt3_fold_coefficients.parquet", index=False)
    folds.to_csv(ATTEMPT3_REPORT_TABLES_DIR / "attempt3_fold_metrics.csv", index=False)
    tuning.to_csv(ATTEMPT3_REPORT_TABLES_DIR / "attempt3_inner_tuning.csv", index=False)
    summary.to_csv(ATTEMPT3_REPORT_TABLES_DIR / "attempt3_model_comparison.csv", index=False)
    ablations.to_csv(ATTEMPT3_REPORT_TABLES_DIR / "attempt3_ablation_comparison.csv", index=False)
    decisions.to_csv(ATTEMPT3_REPORT_TABLES_DIR / "attempt3_decisions.csv", index=False)
    deployment.to_csv(ATTEMPT3_REPORT_TABLES_DIR / "attempt3_research_models.csv", index=False)
    _write_summary(summary, ablations, decisions, coverage)
    return {"summary": str(ATTEMPT3_REPORTS_DIR / "attempt3_evaluation_summary.txt")}


if __name__ == "__main__":
    run_attempt3_models()
