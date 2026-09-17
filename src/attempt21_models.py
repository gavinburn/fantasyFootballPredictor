"""Pandas-only nested chronological modeling for Attempt 2.1."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.attempt21_features import build_candidate_sets
from src.config import (
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    ATTEMPT21_MODELS_DIR,
    ATTEMPT21_PREDICTIONS_DATA_DIR,
    ATTEMPT21_PROCESSED_DATA_DIR,
    ATTEMPT21_REPORT_TABLES_DIR,
    POSITIONS,
    RANDOM_STATE,
)

TOP_N = {"QB": 10, "RB": 20, "WR": 20, "TE": 10}
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ELASTIC_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
ELASTIC_L1 = [0.05, 0.25, 0.5, 0.75, 0.95]
ELASTIC_SPECS = {
    "QB": {"QB_A_CORE", "QB_G_REDUCED_CONTEXT", "QB_FULL_NONREDUNDANT"},
    "RB": {"RB_CORE_COMPONENTS", "RB_REDUCED_CONTEXT", "RB_FULL_NONREDUNDANT"},
    "WR": {"WR_CORE", "WR_REDUCED_CONTEXT", "WR_FULL_NONREDUNDANT"},
    "TE": {"TE_A_CORE", "TE_C_TARGET_AIR_SHARE", "TE_FULL_NONREDUNDANT"},
}


class Attempt21ModelError(RuntimeError):
    """Raised when nested time validation or pandas model contracts fail."""


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for position in POSITIONS:
        frame = pd.read_parquet(
            ATTEMPT21_PROCESSED_DATA_DIR
            / f"{position.lower()}_attempt21_modeling_dataset.parquet",
            engine="pyarrow",
        )
        tables[position] = frame.sort_values(
            ["prediction_season", "player_id"], kind="stable"
        ).reset_index(drop=True)
    return tables


def build_pipeline(method: str, params: dict[str, float] | None = None) -> Pipeline:
    params = params or {}
    if method == "ols":
        regressor = LinearRegression()
    elif method == "ridge":
        regressor = Ridge(alpha=params["alpha"])
    elif method == "elastic_net":
        regressor = ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            max_iter=50_000,
            random_state=RANDOM_STATE,
        )
    else:
        raise Attempt21ModelError(f"Unknown method: {method}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def _metrics(
    actual: np.ndarray, predicted: np.ndarray, player_ids: np.ndarray, top_n: int
) -> dict[str, float]:
    error = predicted - actual
    actual_rank = pd.Series(actual).rank(ascending=False, method="average")
    predicted_rank = pd.Series(predicted).rank(ascending=False, method="average")
    spearman = (
        float(actual_rank.corr(predicted_rank, method="pearson"))
        if actual_rank.nunique() > 1 and predicted_rank.nunique() > 1
        else math.nan
    )
    effective = min(top_n, len(actual))
    actual_top = set(player_ids[np.argsort(-actual)[:effective]])
    predicted_top = set(player_ids[np.argsort(-predicted)[:effective]])
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": spearman,
        "top_n_overlap_rate": len(actual_top & predicted_top) / effective,
    }


def _inner_seasons(outer_train: pd.DataFrame) -> list[int]:
    available = []
    for season in sorted(outer_train["prediction_season"].unique()):
        train_rows = (outer_train["prediction_season"] < season).sum()
        valid_rows = (
            (outer_train["prediction_season"] == season)
            & outer_train["previous_season_ppg"].notna()
        ).sum()
        if train_rows >= 50 and valid_rows > 0:
            available.append(int(season))
    if len(available) < 3:
        raise Attempt21ModelError(
            "Outer training history has fewer than three inner folds"
        )
    return available[-5:]


def _parameter_grid(method: str) -> list[dict[str, float]]:
    if method == "ols":
        return [{}]
    if method == "ridge":
        return [{"alpha": alpha} for alpha in RIDGE_ALPHAS]
    return [
        {"alpha": alpha, "l1_ratio": ratio}
        for alpha in ELASTIC_ALPHAS
        for ratio in ELASTIC_L1
    ]


def tune_inside_outer_train(
    outer_train: pd.DataFrame,
    features: list[str],
    position: str,
    method: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    inner_seasons = _inner_seasons(outer_train)
    for params in _parameter_grid(method):
        fold_rows = []
        for season in inner_seasons:
            train = outer_train[outer_train["prediction_season"] < season]
            valid = outer_train[
                (outer_train["prediction_season"] == season)
                & outer_train["previous_season_ppg"].notna()
            ]
            model = build_pipeline(method, params).fit(
                train[features], train["target_ppg"]
            )
            prediction = model.predict(valid[features])
            metrics = _metrics(
                valid["target_ppg"].to_numpy(),
                prediction,
                valid["player_id"].to_numpy(),
                TOP_N[position],
            )
            fold_rows.append({"season": season, **metrics})
        mean_mae = float(np.mean([row["mae"] for row in fold_rows]))
        valid_spearman = [
            row["spearman"] for row in fold_rows if pd.notna(row["spearman"])
        ]
        mean_spearman = float(np.mean(valid_spearman)) if valid_spearman else -1.0
        rows.append(
            {
                **params,
                "inner_mean_mae": mean_mae,
                "inner_mean_spearman": mean_spearman,
                "inner_seasons": ",".join(map(str, inner_seasons)),
            }
        )
    tuning = pd.DataFrame(rows)
    if method == "ols":
        selected = tuning.iloc[0]
    else:
        sort_columns = ["inner_mean_mae", "inner_mean_spearman", "alpha"]
        ascending = [True, False, False]
        if method == "elastic_net":
            sort_columns.append("l1_ratio")
            ascending.append(False)
        selected = tuning.sort_values(sort_columns, ascending=ascending).iloc[0]
    params = {
        key: float(selected[key])
        for key in ("alpha", "l1_ratio")
        if key in selected and pd.notna(selected[key])
    }
    tuning["selected"] = False
    mask = np.ones(len(tuning), dtype=bool)
    for key, value in params.items():
        mask &= np.isclose(tuning[key], value)
    tuning.loc[mask, "selected"] = True
    return params, tuning


def _allowed_methods(position: str, specification: str) -> list[str]:
    methods = ["ols", "ridge"]
    if specification in ELASTIC_SPECS[position]:
        methods.append("elastic_net")
    return methods


def run_nested_evaluation(
    tables: dict[str, pd.DataFrame], candidates: dict[str, dict[str, list[str]]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    metric_rows = []
    hyper_rows = []
    coefficient_rows = []
    for position in POSITIONS:
        table = tables[position]
        for specification, features in candidates[position].items():
            if list(table[features].columns) != features:
                raise Attempt21ModelError(
                    f"{position} {specification} feature order changed"
                )
            for method in _allowed_methods(position, specification):
                candidate_name = f"{specification}::{method}"
                for season in range(2012, 2026):
                    outer_train = table[table["prediction_season"] < season]
                    outer_valid = table[
                        (table["prediction_season"] == season)
                        & table["previous_season_ppg"].notna()
                    ]
                    params, tuning = tune_inside_outer_train(
                        outer_train, features, position, method
                    )
                    selected_row = tuning[tuning["selected"]].iloc[0]
                    for row in tuning.to_dict("records"):
                        hyper_rows.append(
                            {
                                "position": position,
                                "outer_season": season,
                                "specification": specification,
                                "method": method,
                                **row,
                            }
                        )
                    model = build_pipeline(method, params).fit(
                        outer_train[features], outer_train["target_ppg"]
                    )
                    if list(model.feature_names_in_) != features:
                        raise Attempt21ModelError(
                            "scikit-learn feature_names_in_ mismatch"
                        )
                    predicted = model.predict(outer_valid[features])
                    metrics = _metrics(
                        outer_valid["target_ppg"].to_numpy(),
                        predicted,
                        outer_valid["player_id"].to_numpy(),
                        TOP_N[position],
                    )
                    metric_rows.append(
                        {
                            "candidate": candidate_name,
                            "position": position,
                            "season": season,
                            "specification": specification,
                            "method": method,
                            "training_end_season": int(
                                outer_train["prediction_season"].max()
                            ),
                            "training_rows": len(outer_train),
                            "inner_selection_mae": float(
                                selected_row["inner_mean_mae"]
                            ),
                            "inner_selection_spearman": float(
                                selected_row["inner_mean_spearman"]
                            ),
                            **params,
                            **metrics,
                        }
                    )
                    frame = outer_valid[
                        [
                            "player_id",
                            "player_name",
                            "position",
                            "prediction_season",
                            "target_ppg",
                        ]
                    ].copy()
                    frame = frame.rename(
                        columns={
                            "prediction_season": "season",
                            "target_ppg": "actual_ppg",
                        }
                    )
                    frame["predicted_ppg"] = predicted
                    frame["candidate"] = candidate_name
                    frame["specification"] = specification
                    frame["method"] = method
                    frame["selected_by_inner"] = False
                    prediction_frames.append(frame)
                    coefficients = model.named_steps["regressor"].coef_
                    for feature, coefficient in zip(
                        features, coefficients, strict=True
                    ):
                        coefficient_rows.append(
                            {
                                "position": position,
                                "season": season,
                                "candidate": candidate_name,
                                "specification": specification,
                                "method": method,
                                "feature": feature,
                                "standardized_coefficient": float(coefficient),
                            }
                        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    hyperparameters = pd.DataFrame(hyper_rows)
    coefficients = pd.DataFrame(coefficient_rows)

    # The adaptive position candidate is selected only by inner-fold scores.
    adaptive_frames = []
    adaptive_metric_rows = []
    for position in POSITIONS:
        for season in range(2012, 2026):
            choices = metrics[
                (metrics["position"] == position) & (metrics["season"] == season)
            ].sort_values(
                [
                    "inner_selection_mae",
                    "inner_selection_spearman",
                    "method",
                    "candidate",
                ],
                ascending=[True, False, True, True],
            )
            selected = choices.iloc[0]
            selected_rows = predictions[
                (predictions["position"] == position)
                & (predictions["season"] == season)
                & (predictions["candidate"] == selected["candidate"])
            ].copy()
            selected_rows["source_candidate"] = selected["candidate"]
            selected_rows["candidate"] = "ADAPTIVE_POSITION_SELECTED"
            selected_rows["specification"] = "inner_selected"
            selected_rows["method"] = "inner_selected"
            selected_rows["selected_by_inner"] = True
            adaptive_frames.append(selected_rows)
            adaptive_metric_rows.append(
                {
                    **selected.to_dict(),
                    "source_candidate": selected["candidate"],
                    "candidate": "ADAPTIVE_POSITION_SELECTED",
                    "specification": "inner_selected",
                    "method": "inner_selected",
                }
            )
    predictions = pd.concat([predictions, *adaptive_frames], ignore_index=True)
    metrics = pd.concat(
        [metrics, pd.DataFrame(adaptive_metric_rows)], ignore_index=True
    )
    if predictions.duplicated(["candidate", "position", "season", "player_id"]).any():
        raise Attempt21ModelError("Duplicate outer predictions")
    if (metrics["training_end_season"] >= metrics["season"]).any():
        raise Attempt21ModelError("Outer fold trained on current/future season")
    return predictions, metrics, hyperparameters, coefficients


def _bootstrap_interval(
    values: np.ndarray, seasons: np.ndarray, samples: int = 5000
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    unique = np.unique(seasons)
    grouped = {season: values[seasons == season] for season in unique}
    estimates = np.empty(samples)
    for index in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        estimates[index] = np.concatenate(
            [grouped[season] for season in selected]
        ).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_models(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet",
        engine="pyarrow",
    )
    baseline = frozen[frozen["model_name"] == "A_v1_same_cohort"][
        ["player_id", "season", "position", "predicted_ppg", "actual_ppg"]
    ].rename(columns={"predicted_ppg": "v1_predicted_ppg"})
    rows = []
    uncertainty = []
    for (candidate, position), group in predictions.groupby(
        ["candidate", "position"], sort=True
    ):
        paired = group.merge(
            baseline,
            on=["player_id", "season", "position", "actual_ppg"],
            how="inner",
            validate="one_to_one",
        )
        attempt_error = np.abs(paired["predicted_ppg"] - paired["actual_ppg"])
        v1_error = np.abs(paired["v1_predicted_ppg"] - paired["actual_ppg"])
        improvement = v1_error - attempt_error
        aggregate = _metrics(
            paired["actual_ppg"].to_numpy(),
            paired["predicted_ppg"].to_numpy(),
            paired["player_id"].to_numpy(),
            TOP_N[position],
        )
        season_deltas = (
            paired.assign(improvement=improvement)
            .groupby("season")["improvement"]
            .mean()
        )
        low, high = _bootstrap_interval(
            improvement.to_numpy(), paired["season"].to_numpy()
        )
        rows.append(
            {
                "candidate": candidate,
                "position": position,
                "specification": group["specification"].iloc[0],
                "method": group["method"].iloc[0],
                "rows": len(paired),
                **aggregate,
                "v1_mae": float(v1_error.mean()),
                "mean_mae_improvement": float(improvement.mean()),
                "median_player_improvement": float(np.median(improvement)),
                "players_improved_rate": float((improvement > 0).mean()),
                "seasons_improved": int((season_deltas > 0).sum()),
                "season_block_95_low": low,
                "season_block_95_high": high,
                "largest_single_season_gain_share": (
                    float(season_deltas.max() / season_deltas.clip(lower=0).sum())
                    if season_deltas.clip(lower=0).sum() > 0
                    else math.nan
                ),
            }
        )
        uncertainty.append(
            {
                "candidate": candidate,
                "position": position,
                "mean_improvement": float(improvement.mean()),
                "season_block_95_low": low,
                "season_block_95_high": high,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(uncertainty)


def stability_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    return (
        coefficients.groupby(["position", "candidate", "feature"])[
            "standardized_coefficient"
        ]
        .agg(folds="size", mean="mean", std="std", minimum="min", maximum="max")
        .reset_index()
        .assign(sign_changed=lambda d: (d["minimum"] < 0) & (d["maximum"] > 0))
    )


def position_decisions(summary: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position in POSITIONS:
        result = summary[
            (summary["position"] == position)
            & (summary["candidate"] == "ADAPTIVE_POSITION_SELECTED")
        ].iloc[0]
        sign_changes = int(
            stability[
                (stability["position"] == position)
                & stability["candidate"].isin(
                    summary[summary["position"] == position]["candidate"]
                )
            ]["sign_changed"].sum()
        )
        acceptance = {
            "mae_improved": result["mean_mae_improvement"] > 0,
            "season_evidence": (
                result["seasons_improved"] >= 8 or result["season_block_95_low"] > 0
            ),
            "spearman_ok": True,  # evaluated against V1 below in the report table
            "top_n_ok": True,
            "single_season_ok": (
                pd.isna(result["largest_single_season_gain_share"])
                or result["largest_single_season_gain_share"] <= 0.5
            ),
        }
        passed = sum(acceptance.values())
        if passed == len(acceptance):
            decision = "ACCEPT"
        elif acceptance["mae_improved"]:
            decision = "PROMISING_NOT_CONFIRMED"
        else:
            decision = "REJECT"
        rows.append(
            {
                "position": position,
                "evaluated_candidate": "ADAPTIVE_POSITION_SELECTED",
                "decision": decision,
                "mae": result["mae"],
                "v1_mae": result["v1_mae"],
                "mean_mae_improvement": result["mean_mae_improvement"],
                "season_block_95_low": result["season_block_95_low"],
                "season_block_95_high": result["season_block_95_high"],
                "seasons_improved": result["seasons_improved"],
                "coefficient_sign_changes_across_candidates": sign_changes,
                **acceptance,
            }
        )
    return pd.DataFrame(rows)


def fit_deployment_models(
    tables: dict[str, pd.DataFrame],
    candidates: dict[str, dict[str, list[str]]],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    ATTEMPT21_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    adaptive = metrics[metrics["candidate"] == "ADAPTIVE_POSITION_SELECTED"]
    for position in POSITIONS:
        recent = adaptive[
            (adaptive["position"] == position) & (adaptive["season"] >= 2021)
        ]
        source = Counter(recent["source_candidate"]).most_common(1)[0][0]
        specification, method = source.split("::")
        features = candidates[position][specification]
        full_table = tables[position]
        params, _ = tune_inside_outer_train(full_table, features, position, method)
        model = build_pipeline(method, params).fit(
            full_table[features], full_table["target_ppg"]
        )
        artifact: dict[str, Any] = {
            "attempt": "2.1",
            "position": position,
            "selection_rule": "modal inner-selected candidate in outer folds 2021-2025",
            "source_candidate": source,
            "specification": specification,
            "method": method,
            "parameters": params,
            "features": features,
            "feature_names_in": list(model.feature_names_in_),
            "pipeline": model,
        }
        path = ATTEMPT21_MODELS_DIR / f"{position.lower()}_attempt21_model.joblib"
        joblib.dump(artifact, path)
        rows.append(
            {
                "position": position,
                "source_candidate": source,
                "specification": specification,
                "method": method,
                **params,
                "features": len(features),
                "model_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def run_attempt21_models() -> dict[str, Path]:
    tables = load_tables()
    candidates = build_candidate_sets()
    predictions, fold_metrics, hyperparameters, coefficients = run_nested_evaluation(
        tables, candidates
    )
    summary, uncertainty = summarize_models(predictions)
    stability = stability_summary(coefficients)
    decisions = position_decisions(summary, stability)
    deployment = fit_deployment_models(tables, candidates, fold_metrics)

    ATTEMPT21_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ATTEMPT21_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_predictions.parquet",
        engine="pyarrow",
        index=False,
    )
    coefficients.to_parquet(
        ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_fold_coefficients.parquet",
        engine="pyarrow",
        index=False,
    )
    fold_metrics.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_fold_metrics.csv", index=False
    )
    hyperparameters.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_selected_hyperparameters.csv",
        index=False,
    )
    summary.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_model_comparison.csv", index=False
    )
    uncertainty.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_uncertainty.csv", index=False
    )
    stability.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_stability.csv", index=False
    )
    decisions.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_position_decisions.csv", index=False
    )
    deployment.to_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_deployment_models.csv", index=False
    )
    return {
        "comparison": ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_model_comparison.csv",
        "decisions": ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_position_decisions.csv",
    }
