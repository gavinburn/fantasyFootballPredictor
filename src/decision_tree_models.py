"""Leakage-safe decision-tree regression experiment on the frozen V2 features."""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.attempt3_models import _apply_market_calibration, _fit_market_power
from src.attempt21_models import TOP_N, _inner_seasons, _metrics
from src.config import (
    DECISION_TREE_MODELS_DIR,
    DECISION_TREE_PREDICTIONS_DATA_DIR,
    DECISION_TREE_REPORT_TABLES_DIR,
    DECISION_TREE_REPORTS_DIR,
    FEATURE35_PROCESSED_DATA_DIR,
    POSITIONS,
    PREDICTIONS_DATA_DIR,
    RANDOM_STATE,
    SELECTED_PREDICTIONS_DATA_DIR,
    TE_CONSTRAINED_PROCESSED_DATA_DIR,
)
from src.selected_feature_models import (
    EVALUATION_START,
    SELECTED_SPECIFICATION,
    TE_SELECTED_SPECIFICATION,
    WR_FALLBACK_SPECIFICATION,
    selected_feature_manifests,
)

DIRECT = "DECISION_TREE_DIRECT"
TE_RESIDUAL = "DECISION_TREE_TE_RESIDUAL"
TE_RESIDUAL_START = 2017
TE_RESIDUAL_CAP = 0.75
MAX_DEPTHS = (2, 3, 4, 5)
MIN_SAMPLES_LEAF = (5, 10, 20, 40)
MAX_FEATURES = (None, 0.75)
CCP_ALPHAS = (0.0, 0.005)
PARAMETER_GRID = [
    {
        "max_depth": depth,
        "min_samples_leaf": leaf,
        "max_features": features,
        "ccp_alpha": alpha,
    }
    for depth, leaf, features, alpha in product(
        MAX_DEPTHS, MIN_SAMPLES_LEAF, MAX_FEATURES, CCP_ALPHAS
    )
]


class DecisionTreeExperimentError(RuntimeError):
    """Raised when a tree experiment violates a cohort or timing contract."""


def build_tree_pipeline(params: dict[str, object]) -> Pipeline:
    """Build a constrained tree with training-only median imputation."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("regressor", DecisionTreeRegressor(
            criterion="squared_error",
            random_state=RANDOM_STATE,
            **params,
        )),
    ])


def _main_v2_predictions() -> pd.DataFrame:
    selected = pd.read_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_predictions.parquet"
    )
    return selected[
        ~selected["specification"].eq(WR_FALLBACK_SPECIFICATION)
    ].rename(columns={"predicted_ppg": "v2_predicted_ppg"})


def _load_tables() -> dict[str, pd.DataFrame]:
    tables = {
        position: pd.read_parquet(
            FEATURE35_PROCESSED_DATA_DIR
            / f"{position.lower()}_feature_experiments_dataset.parquet"
        ).sort_values(["prediction_season", "player_id"], kind="stable")
        for position in POSITIONS
    }
    competition = pd.read_parquet(
        TE_CONSTRAINED_PROCESSED_DATA_DIR / "te_constrained_modeling_dataset.parquet"
    )[[
        "player_id", "prediction_season",
        "effective_receiving_competitor_count",
        "probability_weighted_competitor_targets_per_game",
    ]]
    tables["TE"] = tables["TE"].drop(
        columns=[
            "effective_receiving_competitor_count",
            "probability_weighted_competitor_targets_per_game",
        ],
        errors="ignore",
    ).merge(
        competition,
        on=["player_id", "prediction_season"],
        how="left",
        validate="one_to_one",
    )
    return {key: value.reset_index(drop=True) for key, value in tables.items()}


def _features() -> dict[str, list[str]]:
    manifests = selected_feature_manifests()
    return {
        "QB": manifests["QB"][SELECTED_SPECIFICATION],
        "RB": manifests["RB"][SELECTED_SPECIFICATION],
        "WR": manifests["WR"][SELECTED_SPECIFICATION],
        "TE": manifests["TE"][TE_SELECTED_SPECIFICATION],
    }


def _prepare_fold(
    train: pd.DataFrame, valid: pd.DataFrame, position: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if position == "RB":
        return _apply_market_calibration(train, valid)
    return train.copy(), valid.copy(), {
        "alpha": math.nan, "beta": math.nan, "training_rows": 0,
    }


def _tune_direct(
    outer_train: pd.DataFrame, features: list[str], position: str
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for params in PARAMETER_GRID:
        fold_metrics = []
        for season in _inner_seasons(outer_train):
            train = outer_train[outer_train["prediction_season"] < season]
            valid = outer_train[outer_train["prediction_season"] == season]
            train, valid, _ = _prepare_fold(train, valid, position)
            model = build_tree_pipeline(params).fit(
                train[features], train["target_ppg"]
            )
            predicted = model.predict(valid[features])
            fold_metrics.append(_metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N[position],
            ))
        rows.append({
            **params,
            "inner_mean_mae": float(np.mean([x["mae"] for x in fold_metrics])),
            "inner_mean_spearman": float(np.nanmean([x["spearman"] for x in fold_metrics])),
        })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "inner_mean_spearman", "max_depth", "min_samples_leaf"],
        ascending=[True, False, True, False], kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    params = {key: tuning.iloc[0][key] for key in (
        "max_depth", "min_samples_leaf", "max_features", "ccp_alpha"
    )}
    params["max_depth"] = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    params["ccp_alpha"] = float(params["ccp_alpha"])
    params["max_features"] = (
        None if pd.isna(params["max_features"]) else float(params["max_features"])
    )
    return params, tuning


def _tune_te_residual(
    outer_train: pd.DataFrame, features: list[str]
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for params in PARAMETER_GRID:
        fold_metrics = []
        for season in _inner_seasons(outer_train):
            train = outer_train[outer_train["prediction_season"] < season]
            valid = outer_train[outer_train["prediction_season"] == season]
            model = build_tree_pipeline(params).fit(
                train[features], train["residual_target"]
            )
            correction = np.clip(
                model.predict(valid[features]), -TE_RESIDUAL_CAP, TE_RESIDUAL_CAP
            )
            predicted = valid["v2_predicted_ppg"].to_numpy() + correction
            fold_metrics.append(_metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N["TE"],
            ))
        rows.append({
            **params,
            "inner_mean_mae": float(np.mean([x["mae"] for x in fold_metrics])),
            "inner_mean_spearman": float(np.nanmean([x["spearman"] for x in fold_metrics])),
        })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "inner_mean_spearman", "max_depth", "min_samples_leaf"],
        ascending=[True, False, True, False], kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    params = {key: tuning.iloc[0][key] for key in (
        "max_depth", "min_samples_leaf", "max_features", "ccp_alpha"
    )}
    params["max_depth"] = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    params["ccp_alpha"] = float(params["ccp_alpha"])
    params["max_features"] = (
        None if pd.isna(params["max_features"]) else float(params["max_features"])
    )
    return params, tuning


def _importance_rows(
    model: Pipeline,
    valid: pd.DataFrame,
    features: list[str],
    target: str,
    position: str,
    season: int,
    specification: str,
) -> list[dict[str, object]]:
    result = permutation_importance(
        model, valid[features], valid[target], scoring="neg_mean_absolute_error",
        n_repeats=5, random_state=RANDOM_STATE,
    )
    split_importance = model.named_steps["regressor"].feature_importances_
    return [{
        "position": position, "season": season, "specification": specification,
        "feature": feature, "importance_mean": float(mean),
        "importance_std": float(std), "split_used": bool(split > 0),
        "tree_impurity_importance": float(split),
    } for feature, mean, std, split in zip(
        features, result.importances_mean, result.importances_std,
        split_importance, strict=True
    )]


def run_evaluation() -> tuple[pd.DataFrame, ...]:
    tables = _load_tables()
    manifests = _features()
    v2 = _main_v2_predictions()
    prediction_rows: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for position in POSITIONS:
        table = tables[position]
        features = manifests[position]
        cohort = v2[v2["position"] == position]
        cohort_keys = set(zip(cohort["player_id"], cohort["season"]))
        for season in range(EVALUATION_START[position], 2026):
            outer_train = table[table["prediction_season"] < season]
            outer_valid = table[table["prediction_season"] == season]
            outer_valid = outer_valid[[
                (player_id, season) in cohort_keys
                for player_id in outer_valid["player_id"]
            ]]
            params, tuning = _tune_direct(outer_train, features, position)
            for row in tuning.to_dict("records"):
                tuning_rows.append({
                    "position": position, "outer_season": season,
                    "specification": DIRECT, **row,
                })
            train, valid, calibration = _prepare_fold(
                outer_train, outer_valid, position
            )
            model = build_tree_pipeline(params).fit(
                train[features], train["target_ppg"]
            )
            predicted = model.predict(valid[features])
            train_predicted = model.predict(train[features])
            metric = _metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N[position],
            )
            tree = model.named_steps["regressor"]
            folds.append({
                "position": position, "season": season, "specification": DIRECT,
                **params, "training_rows": len(train),
                "training_mae": float(np.mean(np.abs(
                    train_predicted - train["target_ppg"].to_numpy()
                ))),
                "tree_depth": tree.get_depth(), "terminal_leaves": tree.get_n_leaves(),
                "distinct_predictions": int(pd.Series(predicted).nunique()),
                "market_calibration_alpha": calibration["alpha"],
                "market_calibration_beta": calibration["beta"],
                **metric,
            })
            frame = valid[[
                "player_id", "player_name", "position", "prediction_season", "target_ppg"
            ]].rename(columns={
                "prediction_season": "season", "target_ppg": "actual_ppg"
            })
            frame["predicted_ppg"] = predicted
            frame["specification"] = DIRECT
            prediction_rows.append(frame)
            importance_rows.extend(_importance_rows(
                model, valid, features, "target_ppg", position, season, DIRECT
            ))

    # Residual TE uses only prior out-of-sample V2 residuals. The first five seasons
    # remain exact V2 fallbacks while enough residual history accumulates.
    te_features = manifests["TE"]
    te = tables["TE"].merge(
        v2[v2["position"] == "TE"][[
            "player_id", "season", "v2_predicted_ppg"
        ]].rename(columns={"season": "prediction_season"}),
        on=["player_id", "prediction_season"], validate="one_to_one",
    )
    te["residual_target"] = te["target_ppg"] - te["v2_predicted_ppg"]
    fallback = te[te["prediction_season"] < TE_RESIDUAL_START].copy()
    fallback_frame = fallback[[
        "player_id", "player_name", "position", "prediction_season", "target_ppg"
    ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
    fallback_frame["predicted_ppg"] = fallback["v2_predicted_ppg"].to_numpy()
    fallback_frame["specification"] = TE_RESIDUAL
    prediction_rows.append(fallback_frame)
    for season in range(TE_RESIDUAL_START, 2026):
        train = te[te["prediction_season"] < season]
        valid = te[te["prediction_season"] == season]
        params, tuning = _tune_te_residual(train, te_features)
        for row in tuning.to_dict("records"):
            tuning_rows.append({
                "position": "TE", "outer_season": season,
                "specification": TE_RESIDUAL, **row,
            })
        model = build_tree_pipeline(params).fit(
            train[te_features], train["residual_target"]
        )
        correction = np.clip(
            model.predict(valid[te_features]), -TE_RESIDUAL_CAP, TE_RESIDUAL_CAP
        )
        predicted = valid["v2_predicted_ppg"].to_numpy() + correction
        train_correction = np.clip(
            model.predict(train[te_features]), -TE_RESIDUAL_CAP, TE_RESIDUAL_CAP
        )
        metric = _metrics(
            valid["target_ppg"].to_numpy(), predicted,
            valid["player_id"].to_numpy(), TOP_N["TE"],
        )
        tree = model.named_steps["regressor"]
        folds.append({
            "position": "TE", "season": season, "specification": TE_RESIDUAL,
            **params, "training_rows": len(train),
            "training_mae": float(np.mean(np.abs(
                train["v2_predicted_ppg"].to_numpy() + train_correction
                - train["target_ppg"].to_numpy()
            ))),
            "tree_depth": tree.get_depth(), "terminal_leaves": tree.get_n_leaves(),
            "distinct_predictions": int(pd.Series(predicted).nunique()),
            "market_calibration_alpha": math.nan,
            "market_calibration_beta": math.nan,
            **metric,
        })
        frame = valid[[
            "player_id", "player_name", "position", "prediction_season", "target_ppg"
        ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
        frame["predicted_ppg"] = predicted
        frame["specification"] = TE_RESIDUAL
        prediction_rows.append(frame)
        importance_rows.extend(_importance_rows(
            model, valid, te_features, "residual_target", "TE", season, TE_RESIDUAL
        ))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    if predictions.duplicated(
        ["player_id", "position", "season", "specification"]
    ).any():
        raise DecisionTreeExperimentError("Duplicate decision-tree predictions")
    return (
        predictions, pd.DataFrame(folds), pd.DataFrame(tuning_rows),
        pd.DataFrame(importance_rows),
    )


def _comparisons(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v2 = _main_v2_predictions()[[
        "player_id", "position", "season", "actual_ppg", "v2_predicted_ppg"
    ]]
    v1 = pd.read_parquet(
        PREDICTIONS_DATA_DIR / "historical_linear_regression_predictions.parquet"
    )[["player_id", "position", "season", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "v1_predicted_ppg"}
    )
    baseline = pd.read_parquet(
        PREDICTIONS_DATA_DIR / "historical_baseline_predictions.parquet"
    )
    baseline = baseline[baseline["model_name"].eq("previous_season_ppg")][[
        "player_id", "position", "season", "predicted_ppg"
    ]].rename(columns={"predicted_ppg": "baseline_predicted_ppg"})
    rng = np.random.default_rng(RANDOM_STATE)
    summary_rows = []
    long_rows = []
    for (position, specification), candidate in predictions.groupby(
        ["position", "specification"]
    ):
        paired = candidate.merge(
            v2, on=["player_id", "position", "season"],
            suffixes=("", "_reference"), validate="one_to_one",
        ).merge(v1, on=["player_id", "position", "season"], validate="one_to_one").merge(
            baseline, on=["player_id", "position", "season"], validate="one_to_one"
        )
        candidate_error = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
        v2_error = (paired["v2_predicted_ppg"] - paired["actual_ppg"]).abs()
        delta = v2_error - candidate_error
        season_delta = delta.groupby(paired["season"]).mean()
        bootstrap = np.asarray([
            rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
            for _ in range(10_000)
        ])
        candidate_metrics = _metrics(
            paired["actual_ppg"].to_numpy(), paired["predicted_ppg"].to_numpy(),
            paired["player_id"].to_numpy(), TOP_N[position],
        )
        v2_metrics = _metrics(
            paired["actual_ppg"].to_numpy(), paired["v2_predicted_ppg"].to_numpy(),
            paired["player_id"].to_numpy(), TOP_N[position],
        )
        consistency_pass = int((season_delta > 0).sum()) >= math.ceil(len(season_delta) / 2)
        mae_pass = float(delta.mean()) > 0
        spearman_pass = (
            candidate_metrics["spearman"] - v2_metrics["spearman"] >= -0.005
        )
        summary_rows.append({
            "position": position, "specification": specification,
            "seasons": f"{paired['season'].min()}-{paired['season'].max()}",
            "player_seasons": len(paired), "candidate_mae": candidate_metrics["mae"],
            "v2_mae": v2_metrics["mae"], "mae_improvement_vs_v2": float(delta.mean()),
            "percent_improvement_vs_v2": float(100 * delta.mean() / v2_error.mean()),
            "seasons_improved": int((season_delta > 0).sum()),
            "seasons_evaluated": len(season_delta),
            "candidate_spearman": candidate_metrics["spearman"],
            "v2_spearman": v2_metrics["spearman"],
            "spearman_change": candidate_metrics["spearman"] - v2_metrics["spearman"],
            "season_block_95_low": float(np.quantile(bootstrap, 0.025)),
            "season_block_95_high": float(np.quantile(bootstrap, 0.975)),
            "mae_pass": mae_pass,
            "season_consistency_pass": consistency_pass,
            "spearman_pass": spearman_pass,
            "decision": "PROMOTE" if all(
                (mae_pass, consistency_pass, spearman_pass)
            ) else "REJECT",
        })
        for model, column in (
            ("Previous-season PPG", "baseline_predicted_ppg"),
            ("V1", "v1_predicted_ppg"), ("V2", "v2_predicted_ppg"),
            (specification, "predicted_ppg"),
        ):
            metrics = _metrics(
                paired["actual_ppg"].to_numpy(), paired[column].to_numpy(),
                paired["player_id"].to_numpy(), TOP_N[position],
            )
            long_rows.append({
                "position": position, "specification": specification,
                "seasons": f"{paired['season'].min()}-{paired['season'].max()}",
                "player_seasons": len(paired), "model": model, **metrics,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def _select_final_params(folds: pd.DataFrame, position: str, specification: str) -> dict[str, object]:
    recent = folds[
        (folds["position"] == position) & (folds["specification"] == specification)
    ].sort_values("season").tail(5)
    keys = ["max_depth", "min_samples_leaf", "max_features", "ccp_alpha"]
    values = [tuple(None if pd.isna(row[key]) else row[key] for key in keys)
              for _, row in recent.iterrows()]
    selected = Counter(values).most_common(1)[0][0]
    return {
        "max_depth": int(selected[0]), "min_samples_leaf": int(selected[1]),
        "max_features": None if selected[2] is None else float(selected[2]),
        "ccp_alpha": float(selected[3]),
    }


def _fit_final_models(
    tables: dict[str, pd.DataFrame], folds: pd.DataFrame
) -> pd.DataFrame:
    manifests = _features()
    v2 = _main_v2_predictions()
    rows = []
    DECISION_TREE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for position in POSITIONS:
        params = _select_final_params(folds, position, DIRECT)
        table = tables[position].copy()
        calibration = {"alpha": math.nan, "beta": math.nan, "training_rows": 0}
        if position == "RB":
            calibration = _fit_market_power(table)
            table["market_implied_ppg"] = calibration["alpha"] * table[
                "preseason_ecr"
            ].pow(-calibration["beta"])
        model = build_tree_pipeline(params).fit(
            table[manifests[position]], table["target_ppg"]
        )
        filename = f"{position.lower()}_decision_tree_direct.joblib"
        joblib.dump(model, DECISION_TREE_MODELS_DIR / filename)
        rows.append({
            "position": position, "specification": DIRECT, **params,
            "features": json.dumps(manifests[position]), "model_file": filename,
            "market_calibration_alpha": calibration["alpha"],
            "market_calibration_beta": calibration["beta"],
        })
    params = _select_final_params(folds, "TE", TE_RESIDUAL)
    te = tables["TE"].merge(
        v2[v2["position"] == "TE"][["player_id", "season", "v2_predicted_ppg"]]
        .rename(columns={"season": "prediction_season"}),
        on=["player_id", "prediction_season"], validate="one_to_one",
    )
    te["residual_target"] = te["target_ppg"] - te["v2_predicted_ppg"]
    model = build_tree_pipeline(params).fit(
        te[manifests["TE"]], te["residual_target"]
    )
    filename = "te_decision_tree_residual.joblib"
    joblib.dump(model, DECISION_TREE_MODELS_DIR / filename)
    rows.append({
        "position": "TE", "specification": TE_RESIDUAL, **params,
        "features": json.dumps(manifests["TE"]), "model_file": filename,
        "residual_cap": TE_RESIDUAL_CAP,
    })
    return pd.DataFrame(rows)


def run_decision_tree_experiment() -> dict[str, str]:
    predictions, folds, tuning, importance = run_evaluation()
    comparison, all_models = _comparisons(predictions)
    deployment = _fit_final_models(_load_tables(), folds)
    DECISION_TREE_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DECISION_TREE_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        DECISION_TREE_PREDICTIONS_DATA_DIR / "decision_tree_predictions.parquet",
        index=False,
    )
    folds.to_csv(DECISION_TREE_REPORT_TABLES_DIR / "fold_metrics.csv", index=False)
    tuning.to_csv(DECISION_TREE_REPORT_TABLES_DIR / "inner_tuning.csv", index=False)
    importance.to_csv(
        DECISION_TREE_REPORT_TABLES_DIR / "permutation_importance_by_fold.csv", index=False
    )
    comparison.to_csv(
        DECISION_TREE_REPORT_TABLES_DIR / "comparison_vs_v2.csv", index=False
    )
    all_models.to_csv(
        DECISION_TREE_REPORT_TABLES_DIR / "all_model_metrics.csv", index=False
    )
    deployment.to_csv(
        DECISION_TREE_REPORT_TABLES_DIR / "model_manifest.csv", index=False
    )
    importance.groupby(["position", "specification", "feature"], as_index=False).agg(
        importance_mean=("importance_mean", "mean"),
        selection_rate=("split_used", "mean"),
        mean_tree_impurity_importance=("tree_impurity_importance", "mean"),
    ).sort_values(
        ["position", "specification", "importance_mean"],
        ascending=[True, True, False],
    ).to_csv(
        DECISION_TREE_REPORT_TABLES_DIR / "mean_permutation_importance.csv", index=False
    )
    diagnosed = predictions.assign(
        error=lambda x: x["predicted_ppg"] - x["actual_ppg"],
        absolute_error=lambda x: (x["predicted_ppg"] - x["actual_ppg"]).abs(),
    )
    diagnosed.sort_values(
        ["position", "specification", "absolute_error"],
        ascending=[True, True, False],
    ).groupby(["position", "specification"], as_index=False).head(25).to_csv(
        DECISION_TREE_REPORT_TABLES_DIR / "largest_absolute_errors.csv", index=False
    )
    diagnosed["actual_ppg_tier"] = diagnosed.groupby(
        ["position", "season"]
    )["actual_ppg"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), 4,
            labels=["low", "lower_middle", "upper_middle", "high"],
        )
    )
    diagnosed.groupby(
        ["position", "specification", "actual_ppg_tier"],
        observed=True, as_index=False,
    ).agg(
        player_seasons=("player_id", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
    ).to_csv(
        DECISION_TREE_REPORT_TABLES_DIR / "error_by_actual_ppg_tier.csv", index=False
    )

    import matplotlib.pyplot as plt

    plot_rows = all_models[
        all_models["model"].isin(["V2", DIRECT])
        & all_models["specification"].eq(DIRECT)
    ].copy()
    pivot = plot_rows.pivot(index="position", columns="model", values="mae")
    pivot = pivot.loc[["QB", "RB", "TE", "WR"]]
    x = np.arange(len(pivot))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - 0.18, pivot["V2"], width=0.36, label="V2 selected model")
    axis.bar(x + 0.18, pivot[DIRECT], width=0.36, label="Decision tree")
    axis.set_xticks(x, pivot.index)
    axis.set_ylabel("Overall MAE (PPG)")
    axis.set_title("Decision tree regression versus V2")
    axis.legend()
    figure.tight_layout()
    DECISION_TREE_REPORTS_DIR.joinpath("figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(
        DECISION_TREE_REPORTS_DIR / "figures" / "decision_tree_vs_v2_mae.png",
        dpi=160,
    )
    plt.close(figure)

    mean_complexity = folds.groupby(
        ["position", "specification"], as_index=False
    ).agg(training_mae=("training_mae", "mean"), validation_mae=("mae", "mean"))
    labels = mean_complexity["position"] + "\n" + mean_complexity["specification"].replace({
        DIRECT: "direct", TE_RESIDUAL: "residual",
    })
    x = np.arange(len(mean_complexity))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, mean_complexity["training_mae"], width=0.36, label="Training")
    axis.bar(x + 0.18, mean_complexity["validation_mae"], width=0.36, label="Outer validation")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean fold MAE (PPG)")
    axis.set_title("Decision-tree training and validation error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        DECISION_TREE_REPORTS_DIR / "figures" / "training_validation_mae.png",
        dpi=160,
    )
    plt.close(figure)
    lines = [
        "DECISION TREE REGRESSION EXPERIMENT", "=" * 78, "",
        "All candidates use the frozen V2 feature definitions and identical outer cohorts.",
        "Hyperparameters are selected inside each chronological outer training fold.",
        "TE residual predictions use V2 fallback through 2016 and capped tree corrections thereafter.",
        "Positive MAE improvement favors the decision tree over V2.", "",
        comparison.to_string(index=False), "",
    ]
    DECISION_TREE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = DECISION_TREE_REPORTS_DIR / "decision_tree_summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")
    return {"summary": str(summary)}


if __name__ == "__main__":
    run_decision_tree_experiment()
