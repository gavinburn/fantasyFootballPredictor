"""Nested chronological histogram-gradient-boosting experiment on V2 features."""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.attempt3_models import _fit_market_power
from src.attempt21_models import TOP_N, _inner_seasons, _metrics
from src.config import (
    GRADIENT_BOOSTING_MODELS_DIR,
    GRADIENT_BOOSTING_PREDICTIONS_DATA_DIR,
    GRADIENT_BOOSTING_REPORT_TABLES_DIR,
    GRADIENT_BOOSTING_REPORTS_DIR,
    RANDOM_STATE,
)
from src.decision_tree_models import (
    TE_RESIDUAL_CAP,
    _comparisons,
    _features,
    _load_tables,
    _main_v2_predictions,
    _prepare_fold,
)
from src.selected_feature_models import EVALUATION_START

DIRECT = "GRADIENT_BOOSTING_DIRECT"
TE_RESIDUAL = "GRADIENT_BOOSTING_TE_RESIDUAL"
TE_RESIDUAL_START = 2017
LEARNING_RATES = (0.03, 0.07, 0.12)
MAX_LEAF_NODES = (7, 15)
MAX_ITERATIONS = (150, 300)
MIN_SAMPLES_LEAF = (10, 25)
LOSSES = ("squared_error", "absolute_error")
PARAMETER_GRID = [
    {
        "learning_rate": learning_rate,
        "max_leaf_nodes": leaves,
        "max_iter": iterations,
        "min_samples_leaf": min_leaf,
        "loss": loss,
    }
    for learning_rate, leaves, iterations, min_leaf, loss in product(
        LEARNING_RATES,
        MAX_LEAF_NODES,
        MAX_ITERATIONS,
        MIN_SAMPLES_LEAF,
        LOSSES,
    )
]


class GradientBoostingExperimentError(RuntimeError):
    """Raised when boosting violates an evaluation contract."""


def build_boosting_pipeline(params: dict[str, object]) -> Pipeline:
    """Build a deterministic boosted model with training-only imputation."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("regressor", HistGradientBoostingRegressor(
            early_stopping=False,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
            **params,
        )),
    ])


def _normalize_params(row: pd.Series) -> dict[str, object]:
    return {
        "learning_rate": float(row["learning_rate"]),
        "max_leaf_nodes": int(row["max_leaf_nodes"]),
        "max_iter": int(row["max_iter"]),
        "min_samples_leaf": int(row["min_samples_leaf"]),
        "loss": str(row["loss"]),
    }


def _tune(
    outer_train: pd.DataFrame,
    features: list[str],
    position: str,
    *,
    residual: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    target = "residual_target" if residual else "target_ppg"
    for params in PARAMETER_GRID:
        fold_metrics = []
        for season in _inner_seasons(outer_train):
            train = outer_train[outer_train["prediction_season"] < season]
            valid = outer_train[outer_train["prediction_season"] == season]
            if not residual:
                train, valid, _ = _prepare_fold(train, valid, position)
            model = build_boosting_pipeline(params).fit(
                train[features], train[target]
            )
            predicted = model.predict(valid[features])
            if residual:
                predicted = valid["v2_predicted_ppg"].to_numpy() + np.clip(
                    predicted, -TE_RESIDUAL_CAP, TE_RESIDUAL_CAP
                )
            fold_metrics.append(_metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N[position],
            ))
        rows.append({
            **params,
            "inner_mean_mae": float(np.mean([x["mae"] for x in fold_metrics])),
            "inner_mean_spearman": float(
                np.nanmean([x["spearman"] for x in fold_metrics])
            ),
        })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "inner_mean_spearman", "max_leaf_nodes", "max_iter"],
        ascending=[True, False, True, True], kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    return _normalize_params(tuning.iloc[0]), tuning


def _importance(
    model: Pipeline,
    valid: pd.DataFrame,
    features: list[str],
    target: str,
    position: str,
    season: int,
    specification: str,
) -> list[dict[str, object]]:
    result = permutation_importance(
        model,
        valid[features],
        valid[target],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return [{
        "position": position,
        "season": season,
        "specification": specification,
        "feature": feature,
        "importance_mean": float(mean),
        "importance_std": float(std),
    } for feature, mean, std in zip(
        features, result.importances_mean, result.importances_std, strict=True
    )]


def _prediction_frame(
    valid: pd.DataFrame, predicted: np.ndarray, specification: str
) -> pd.DataFrame:
    frame = valid[[
        "player_id", "player_name", "position", "prediction_season", "target_ppg"
    ]].rename(columns={
        "prediction_season": "season", "target_ppg": "actual_ppg"
    })
    frame["predicted_ppg"] = predicted
    frame["specification"] = specification
    return frame


def run_evaluation() -> tuple[pd.DataFrame, ...]:
    tables = _load_tables()
    manifests = _features()
    v2 = _main_v2_predictions()
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for position in ("QB", "RB", "WR", "TE"):
        table = tables[position]
        features = manifests[position]
        cohort = v2[v2["position"] == position]
        keys = set(zip(cohort["player_id"], cohort["season"]))
        for season in range(EVALUATION_START[position], 2026):
            outer_train = table[table["prediction_season"] < season]
            outer_valid = table[table["prediction_season"] == season]
            outer_valid = outer_valid[[
                (player_id, season) in keys for player_id in outer_valid["player_id"]
            ]]
            params, tuning = _tune(outer_train, features, position)
            tuning_rows.extend({
                "position": position,
                "outer_season": season,
                "specification": DIRECT,
                **row,
            } for row in tuning.to_dict("records"))
            train, valid, calibration = _prepare_fold(
                outer_train, outer_valid, position
            )
            model = build_boosting_pipeline(params).fit(
                train[features], train["target_ppg"]
            )
            predicted = model.predict(valid[features])
            train_predicted = model.predict(train[features])
            metrics = _metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N[position],
            )
            folds.append({
                "position": position,
                "season": season,
                "specification": DIRECT,
                **params,
                "training_rows": len(train),
                "training_mae": float(np.mean(np.abs(
                    train_predicted - train["target_ppg"].to_numpy()
                ))),
                "distinct_predictions": int(pd.Series(predicted).nunique()),
                "market_calibration_alpha": calibration["alpha"],
                "market_calibration_beta": calibration["beta"],
                **metrics,
            })
            predictions.append(_prediction_frame(valid, predicted, DIRECT))
            importance_rows.extend(_importance(
                model, valid, features, "target_ppg", position, season, DIRECT
            ))

    features = manifests["TE"]
    te = tables["TE"].merge(
        v2[v2["position"] == "TE"][["player_id", "season", "v2_predicted_ppg"]]
        .rename(columns={"season": "prediction_season"}),
        on=["player_id", "prediction_season"],
        validate="one_to_one",
    )
    te["residual_target"] = te["target_ppg"] - te["v2_predicted_ppg"]
    fallback = te[te["prediction_season"] < TE_RESIDUAL_START]
    predictions.append(_prediction_frame(
        fallback, fallback["v2_predicted_ppg"].to_numpy(), TE_RESIDUAL
    ))
    for season in range(TE_RESIDUAL_START, 2026):
        train = te[te["prediction_season"] < season]
        valid = te[te["prediction_season"] == season]
        params, tuning = _tune(train, features, "TE", residual=True)
        tuning_rows.extend({
            "position": "TE",
            "outer_season": season,
            "specification": TE_RESIDUAL,
            **row,
        } for row in tuning.to_dict("records"))
        model = build_boosting_pipeline(params).fit(
            train[features], train["residual_target"]
        )
        correction = np.clip(
            model.predict(valid[features]), -TE_RESIDUAL_CAP, TE_RESIDUAL_CAP
        )
        predicted = valid["v2_predicted_ppg"].to_numpy() + correction
        train_correction = np.clip(
            model.predict(train[features]), -TE_RESIDUAL_CAP, TE_RESIDUAL_CAP
        )
        metrics = _metrics(
            valid["target_ppg"].to_numpy(), predicted,
            valid["player_id"].to_numpy(), TOP_N["TE"],
        )
        folds.append({
            "position": "TE",
            "season": season,
            "specification": TE_RESIDUAL,
            **params,
            "training_rows": len(train),
            "training_mae": float(np.mean(np.abs(
                train["v2_predicted_ppg"].to_numpy() + train_correction
                - train["target_ppg"].to_numpy()
            ))),
            "distinct_predictions": int(pd.Series(predicted).nunique()),
            "market_calibration_alpha": math.nan,
            "market_calibration_beta": math.nan,
            **metrics,
        })
        predictions.append(_prediction_frame(valid, predicted, TE_RESIDUAL))
        importance_rows.extend(_importance(
            model, valid, features, "residual_target", "TE", season, TE_RESIDUAL
        ))

    result = pd.concat(predictions, ignore_index=True)
    if result.duplicated(
        ["player_id", "position", "season", "specification"]
    ).any():
        raise GradientBoostingExperimentError("Duplicate boosted predictions")
    return (
        result,
        pd.DataFrame(folds),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(importance_rows),
    )


def _final_params(
    folds: pd.DataFrame, position: str, specification: str
) -> dict[str, object]:
    recent = folds[
        (folds["position"] == position) & (folds["specification"] == specification)
    ].sort_values("season").tail(5)
    keys = [
        "learning_rate", "max_leaf_nodes", "max_iter", "min_samples_leaf", "loss"
    ]
    candidates = [tuple(row[key] for key in keys) for _, row in recent.iterrows()]
    selected = Counter(candidates).most_common(1)[0][0]
    return {
        "learning_rate": float(selected[0]),
        "max_leaf_nodes": int(selected[1]),
        "max_iter": int(selected[2]),
        "min_samples_leaf": int(selected[3]),
        "loss": str(selected[4]),
    }


def _fit_final_models(
    tables: dict[str, pd.DataFrame], folds: pd.DataFrame
) -> pd.DataFrame:
    manifests = _features()
    v2 = _main_v2_predictions()
    rows = []
    GRADIENT_BOOSTING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for position in ("QB", "RB", "WR", "TE"):
        params = _final_params(folds, position, DIRECT)
        table = tables[position].copy()
        calibration = {"alpha": math.nan, "beta": math.nan}
        if position == "RB":
            calibration = _fit_market_power(table)
            table["market_implied_ppg"] = calibration["alpha"] * table[
                "preseason_ecr"
            ].pow(-calibration["beta"])
        model = build_boosting_pipeline(params).fit(
            table[manifests[position]], table["target_ppg"]
        )
        filename = f"{position.lower()}_gradient_boosting_direct.joblib"
        joblib.dump(model, GRADIENT_BOOSTING_MODELS_DIR / filename)
        rows.append({
            "position": position,
            "specification": DIRECT,
            **params,
            "features": json.dumps(manifests[position]),
            "model_file": filename,
            "market_calibration_alpha": calibration["alpha"],
            "market_calibration_beta": calibration["beta"],
        })
    params = _final_params(folds, "TE", TE_RESIDUAL)
    te = tables["TE"].merge(
        v2[v2["position"] == "TE"][["player_id", "season", "v2_predicted_ppg"]]
        .rename(columns={"season": "prediction_season"}),
        on=["player_id", "prediction_season"],
        validate="one_to_one",
    )
    te["residual_target"] = te["target_ppg"] - te["v2_predicted_ppg"]
    model = build_boosting_pipeline(params).fit(
        te[manifests["TE"]], te["residual_target"]
    )
    filename = "te_gradient_boosting_residual.joblib"
    joblib.dump(model, GRADIENT_BOOSTING_MODELS_DIR / filename)
    rows.append({
        "position": "TE",
        "specification": TE_RESIDUAL,
        **params,
        "features": json.dumps(manifests["TE"]),
        "model_file": filename,
        "residual_cap": TE_RESIDUAL_CAP,
    })
    return pd.DataFrame(rows)


def _diagnostics(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    importance: pd.DataFrame,
    all_models: pd.DataFrame,
) -> None:
    importance.groupby(
        ["position", "specification", "feature"], as_index=False
    ).agg(importance_mean=("importance_mean", "mean")).sort_values(
        ["position", "specification", "importance_mean"],
        ascending=[True, True, False],
    ).to_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "mean_permutation_importance.csv",
        index=False,
    )
    diagnosed = predictions.assign(
        error=lambda x: x["predicted_ppg"] - x["actual_ppg"],
        absolute_error=lambda x: (x["predicted_ppg"] - x["actual_ppg"]).abs(),
    )
    diagnosed.sort_values(
        ["position", "specification", "absolute_error"],
        ascending=[True, True, False],
    ).groupby(["position", "specification"], as_index=False).head(25).to_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "largest_absolute_errors.csv",
        index=False,
    )
    diagnosed["actual_ppg_tier"] = diagnosed.groupby(
        ["position", "season"]
    )["actual_ppg"].transform(lambda values: pd.qcut(
        values.rank(method="first"),
        4,
        labels=["low", "lower_middle", "upper_middle", "high"],
    ))
    diagnosed.groupby(
        ["position", "specification", "actual_ppg_tier"],
        observed=True,
        as_index=False,
    ).agg(
        player_seasons=("player_id", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
    ).to_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "error_by_actual_ppg_tier.csv",
        index=False,
    )

    import matplotlib.pyplot as plt

    rows = all_models[
        all_models["model"].isin(["V2", DIRECT])
        & all_models["specification"].eq(DIRECT)
    ]
    pivot = rows.pivot(index="position", columns="model", values="mae").loc[
        ["QB", "RB", "TE", "WR"]
    ]
    x = np.arange(len(pivot))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - 0.18, pivot["V2"], width=0.36, label="V2 selected model")
    axis.bar(x + 0.18, pivot[DIRECT], width=0.36, label="Gradient boosting")
    axis.set_xticks(x, pivot.index)
    axis.set_ylabel("Overall MAE (PPG)")
    axis.set_title("Gradient-boosted decision trees versus V2")
    axis.legend()
    figure.tight_layout()
    GRADIENT_BOOSTING_REPORTS_DIR.joinpath("figures").mkdir(
        parents=True, exist_ok=True
    )
    figure.savefig(
        GRADIENT_BOOSTING_REPORTS_DIR / "figures" / "gradient_boosting_vs_v2_mae.png",
        dpi=160,
    )
    plt.close(figure)

    error = folds.groupby(
        ["position", "specification"], as_index=False
    ).agg(training_mae=("training_mae", "mean"), validation_mae=("mae", "mean"))
    labels = error["position"] + "\n" + error["specification"].replace({
        DIRECT: "direct", TE_RESIDUAL: "residual",
    })
    x = np.arange(len(error))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, error["training_mae"], width=0.36, label="Training")
    axis.bar(x + 0.18, error["validation_mae"], width=0.36, label="Outer validation")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean fold MAE (PPG)")
    axis.set_title("Gradient-boosting training and validation error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        GRADIENT_BOOSTING_REPORTS_DIR / "figures" / "training_validation_mae.png",
        dpi=160,
    )
    plt.close(figure)


def run_gradient_boosting_experiment() -> dict[str, str]:
    predictions, folds, tuning, importance = run_evaluation()
    comparison, all_models = _comparisons(predictions)
    deployment = _fit_final_models(_load_tables(), folds)
    GRADIENT_BOOSTING_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRADIENT_BOOSTING_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        GRADIENT_BOOSTING_PREDICTIONS_DATA_DIR / "gradient_boosting_predictions.parquet",
        index=False,
    )
    folds.to_csv(GRADIENT_BOOSTING_REPORT_TABLES_DIR / "fold_metrics.csv", index=False)
    tuning.to_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "inner_tuning.csv", index=False
    )
    importance.to_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "permutation_importance_by_fold.csv",
        index=False,
    )
    comparison.to_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "comparison_vs_v2.csv", index=False
    )
    all_models.to_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "all_model_metrics.csv", index=False
    )
    deployment.to_csv(
        GRADIENT_BOOSTING_REPORT_TABLES_DIR / "model_manifest.csv", index=False
    )
    _diagnostics(predictions, folds, importance, all_models)

    te_active = predictions[
        (predictions["position"] == "TE")
        & (predictions["specification"] == TE_RESIDUAL)
        & (predictions["season"] >= TE_RESIDUAL_START)
    ].merge(
        _main_v2_predictions()[[
            "player_id", "position", "season", "v2_predicted_ppg"
        ]],
        on=["player_id", "position", "season"],
        validate="one_to_one",
    )
    active_delta = (
        (te_active["v2_predicted_ppg"] - te_active["actual_ppg"]).abs()
        - (te_active["predicted_ppg"] - te_active["actual_ppg"]).abs()
    )
    active_seasons = active_delta.groupby(te_active["season"]).mean()
    lines = [
        "GRADIENT-BOOSTED DECISION TREE EXPERIMENT",
        "=" * 78,
        "",
        "Models use frozen V2 features and identical chronological outer cohorts.",
        "Learning rate, leaf count, iteration count, leaf size, and loss are selected",
        "inside each outer training history. Internal random early stopping is disabled.",
        "TE residual boosting uses V2 fallback through 2016 and bounded corrections thereafter.",
        "Positive MAE improvement favors gradient boosting over V2.",
        "",
        comparison.to_string(index=False),
        "",
        "TE residual active-period detail (2017-2025)",
        f"Mean MAE improvement versus V2: {active_delta.mean():.6f} PPG",
        f"Active seasons improved: {(active_seasons > 0).sum()} of {len(active_seasons)}",
        "",
    ]
    GRADIENT_BOOSTING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = GRADIENT_BOOSTING_REPORTS_DIR / "gradient_boosting_summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return {"summary": str(summary)}


if __name__ == "__main__":
    run_gradient_boosting_experiment()
