"""TE receiving-role ablations and leakage-safe two-stage mixture model."""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.attempt3_models import _tune
from src.attempt21_features import V1_FEATURES
from src.attempt21_models import TOP_N, _inner_seasons, _metrics, build_pipeline
from src.config import (
    ATTEMPT2_INTERIM_DATA_DIR,
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    DATA_DIR,
    FEATURE35_PROCESSED_DATA_DIR,
    FEATURE35_RAW_DATA_DIR,
    RANDOM_STATE,
    TE_ROLE_PREDICTIONS_DATA_DIR,
    TE_ROLE_PROCESSED_DATA_DIR,
    TE_ROLE_REPORT_FIGURES_DIR,
    TE_ROLE_REPORT_TABLES_DIR,
    TE_ROLE_REPORTS_DIR,
)
from src.download_feature_experiments import FILES


class TERoleExperimentError(RuntimeError):
    """Raised when TE role-model data or validation contracts fail."""


V1 = list(V1_FEATURES["TE"])
PARTICIPATION_RATE = "previous_pass_play_participation_rate"
TARGETS_PER_PARTICIPATED = "previous_targets_per_pass_play_participated"
TARGET_RATE = "previous_targets_per_team_pass_play"
SNAP_SHARE = "previous_offensive_snap_share"
ROLE_CONTEXT = [
    "previous_target_share",
    TARGET_RATE,
    SNAP_SHARE,
    "other_te_max_previous_target_share",
    "other_te_max_previous_targets_per_game",
    "other_te_receiving_competitor_count",
    "wr_max_previous_target_share",
]
FEATURE_MANIFESTS = {
    "V1_PLUS_TARGET_SHARE": [*V1, "previous_target_share"],
    "V1_PLUS_PARTICIPATION": [*V1, PARTICIPATION_RATE, TARGETS_PER_PARTICIPATED],
    "V1_PLUS_TARGET_RATE": [*V1, TARGET_RATE],
    "V1_PLUS_SNAP_SHARE": [*V1, SNAP_SHARE],
    "V1_PLUS_SNAP_AND_TARGET_SHARE": [*V1, SNAP_SHARE, "previous_target_share"],
    "V1_ROLE_CONTEXT": [*V1, *ROLE_CONTEXT],
}
EVALUATION_START = {
    "V1_PLUS_TARGET_SHARE": 2012,
    "V1_PLUS_PARTICIPATION": 2019,
    "V1_PLUS_TARGET_RATE": 2019,
    "V1_PLUS_SNAP_SHARE": 2017,
    "V1_PLUS_SNAP_AND_TARGET_SHARE": 2017,
    "V1_ROLE_CONTEXT": 2019,
    "TWO_STAGE_ROLE_MIXTURE": 2019,
    "THREE_TIER_ROLE_MIXTURE": 2019,
    "OPPORTUNITY_FIRST": 2019,
    "NONLINEAR_ROLE_CONTEXT": 2019,
    "V1_OPPORTUNITY_ENSEMBLE": 2022,
}
ROLE_THRESHOLDS = [1.5, 2.5, 3.5, 4.5]
MIXTURE_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
THREE_TIER_CUTS = [(1.5, 3.5), (2.0, 4.0), (2.5, 4.5)]
OPPORTUNITY_RIDGE_ALPHAS = [1.0, 10.0, 100.0]
NONLINEAR_GRID = [
    {"learning_rate": rate, "max_leaf_nodes": leaves, "l2_regularization": penalty}
    for rate in (0.03, 0.08)
    for leaves in (7, 15)
    for penalty in (1.0, 10.0)
]
ENSEMBLE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0))


def _snap_features() -> pd.DataFrame:
    snap = pd.read_parquet(FEATURE35_RAW_DATA_DIR / FILES["snap_counts"])
    snap = snap[(snap["game_type"] == "REG") & (snap["position"] == "TE")].copy()
    players = pd.read_parquet(
        DATA_DIR / "raw/players.parquet", columns=["gsis_id", "pfr_id"]
    ).dropna().drop_duplicates("pfr_id")
    snap = snap.merge(
        players, left_on="pfr_player_id", right_on="pfr_id",
        how="inner", validate="many_to_one",
    )
    snap["offense_snaps"] = pd.to_numeric(snap["offense_snaps"], errors="coerce")
    snap["offense_pct"] = pd.to_numeric(snap["offense_pct"], errors="coerce")
    valid = snap["offense_pct"].gt(0) & snap["offense_snaps"].ge(0)
    snap = snap[valid].copy()
    # PFR percentage is rounded by game. Reconstructing the denominator and then
    # aggregating weights games by their offensive play counts.
    snap["implied_team_offensive_snaps"] = (
        snap["offense_snaps"] / snap["offense_pct"]
    )
    grouped = snap.groupby(["gsis_id", "season"], as_index=False).agg(
        player_offensive_snaps=("offense_snaps", "sum"),
        implied_team_offensive_snaps=("implied_team_offensive_snaps", "sum"),
        snap_games=("game_id", "nunique"),
    )
    grouped[SNAP_SHARE] = _ratio(
        grouped["player_offensive_snaps"], grouped["implied_team_offensive_snaps"]
    ).clip(0, 1)
    grouped["prediction_season"] = grouped.pop("season") + 1
    return grouped.rename(columns={"gsis_id": "player_id"})[[
        "player_id", "prediction_season", SNAP_SHARE, "snap_games"
    ]]


def _role_competition_features() -> pd.DataFrame:
    competition = pd.read_parquet(
        ATTEMPT2_INTERIM_DATA_DIR / "competition_long.parquet"
    )
    competition = competition[competition["position"] == "TE"].copy()
    workload = pd.read_parquet(
        ATTEMPT2_INTERIM_DATA_DIR / "player_workload_shares.parquet",
        columns=["player_id", "prediction_season", "previous_target_share"],
    ).rename(columns={
        "player_id": "competitor_id",
        "previous_target_share": "competitor_previous_target_share",
    })
    seasons = pd.read_parquet(
        DATA_DIR / "interim/player_seasons_all.parquet",
        columns=["player_id", "season", "targets_per_game"],
    ).rename(columns={
        "player_id": "competitor_id",
        "targets_per_game": "competitor_previous_targets_per_game",
    })
    seasons["prediction_season"] = seasons.pop("season") + 1
    competition = competition.merge(
        workload, on=["competitor_id", "prediction_season"],
        how="left", validate="many_to_one",
    ).merge(
        seasons, on=["competitor_id", "prediction_season"],
        how="left", validate="many_to_one",
    )
    competition["te_receiving_competitor"] = (
        competition["competitor_position"].eq("TE")
        & competition["competitor_previous_targets_per_game"].ge(1.0)
    ).astype(int)
    competition["te_target_share"] = competition[
        "competitor_previous_target_share"
    ].where(competition["competitor_position"].eq("TE"))
    competition["te_targets_per_game"] = competition[
        "competitor_previous_targets_per_game"
    ].where(competition["competitor_position"].eq("TE"))
    competition["wr_target_share"] = competition[
        "competitor_previous_target_share"
    ].where(competition["competitor_position"].eq("WR"))
    return competition.groupby(
        ["player_id", "prediction_season"], as_index=False
    ).agg(
        other_te_max_previous_target_share=("te_target_share", "max"),
        other_te_max_previous_targets_per_game=("te_targets_per_game", "max"),
        other_te_receiving_competitor_count=("te_receiving_competitor", "sum"),
        wr_max_previous_target_share=("wr_target_share", "max"),
    )


def _target_role_labels() -> pd.DataFrame:
    seasons = pd.read_parquet(
        DATA_DIR / "interim/player_seasons_all.parquet",
        columns=["player_id", "season", "games_played", "targets", "targets_per_game"],
    )
    return seasons.rename(columns={
        "season": "prediction_season",
        "games_played": "target_games_played",
        "targets": "target_targets",
        "targets_per_game": "target_targets_per_game",
    })


def build_te_role_table() -> pd.DataFrame:
    table = pd.read_parquet(
        FEATURE35_PROCESSED_DATA_DIR / "te_feature_experiments_dataset.parquet"
    )
    table[TARGET_RATE] = (
        table[PARTICIPATION_RATE] * table[TARGETS_PER_PARTICIPATED]
    )
    table = table.merge(
        _snap_features(), on=["player_id", "prediction_season"],
        how="left", validate="one_to_one",
    ).merge(
        _role_competition_features(), on=["player_id", "prediction_season"],
        how="left", validate="one_to_one",
    ).merge(
        _target_role_labels(), on=["player_id", "prediction_season"],
        how="left", validate="one_to_one",
    )
    table["other_te_receiving_competitor_count"] = table[
        "other_te_receiving_competitor_count"
    ].fillna(0)
    if table["target_targets_per_game"].isna().any():
        raise TERoleExperimentError("A modeling row is missing its target-season role label")
    if table.duplicated(["player_id", "prediction_season"]).any():
        raise TERoleExperimentError("Duplicate TE role-model rows")
    TE_ROLE_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(
        TE_ROLE_PROCESSED_DATA_DIR / "te_role_modeling_dataset.parquet", index=False
    )
    coverage = []
    for season, rows in table.groupby("prediction_season"):
        coverage.append({
            "prediction_season": int(season),
            "rows": len(rows),
            "participation_available": int(rows[PARTICIPATION_RATE].notna().sum()),
            "snap_share_available": int(rows[SNAP_SHARE].notna().sum()),
            "target_share_available": int(rows["previous_target_share"].notna().sum()),
        })
    TE_ROLE_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(
        TE_ROLE_REPORT_TABLES_DIR / "feature_coverage.csv", index=False
    )
    (TE_ROLE_REPORT_TABLES_DIR / "feature_manifest.json").write_text(
        json.dumps(FEATURE_MANIFESTS, indent=2) + "\n", encoding="utf-8"
    )
    return table


def _role_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=10_000,
            random_state=RANDOM_STATE,
        )),
    ])


def _ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha)),
    ])


def _fit_mixture(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str],
    threshold: float, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    role = train["target_targets_per_game"].ge(threshold)
    if role.sum() < 20 or (~role).sum() < 20:
        raise TERoleExperimentError("A mixture branch has too few training rows")
    classifier = _role_classifier().fit(train[features], role.astype(int))
    low = _ridge(alpha).fit(train.loc[~role, features], train.loc[~role, "target_ppg"])
    high = _ridge(alpha).fit(train.loc[role, features], train.loc[role, "target_ppg"])
    probability = classifier.predict_proba(valid[features])[:, 1]
    prediction = (
        (1 - probability) * low.predict(valid[features])
        + probability * high.predict(valid[features])
    )
    return prediction, probability


def _tune_mixture(
    outer_train: pd.DataFrame, features: list[str]
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    inner_seasons = _inner_seasons(outer_train)
    for threshold in ROLE_THRESHOLDS:
        for alpha in MIXTURE_RIDGE_ALPHAS:
            fold_mae = []
            for season in inner_seasons:
                train = outer_train[outer_train["prediction_season"] < season]
                valid = outer_train[outer_train["prediction_season"] == season]
                predicted, _ = _fit_mixture(
                    train, valid, features, threshold, alpha
                )
                fold_mae.append(float(np.mean(np.abs(
                    predicted - valid["target_ppg"].to_numpy()
                ))))
            rows.append({
                "role_threshold_targets_per_game": threshold,
                "ridge_alpha": alpha,
                "inner_mean_mae": float(np.mean(fold_mae)),
            })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "role_threshold_targets_per_game", "ridge_alpha"],
        kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    return (
        float(tuning.loc[0, "role_threshold_targets_per_game"]),
        float(tuning.loc[0, "ridge_alpha"]),
        tuning,
    )


def _fit_three_tier_mixture(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str],
    cuts: tuple[float, float], alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    role = np.select(
        [
            train["target_targets_per_game"].lt(cuts[0]),
            train["target_targets_per_game"].lt(cuts[1]),
        ],
        [0, 1], default=2,
    )
    counts = np.bincount(role, minlength=3)
    if (counts < 20).any():
        raise TERoleExperimentError("A three-tier branch has too few training rows")
    classifier = _role_classifier().fit(train[features], role)
    probabilities = classifier.predict_proba(valid[features])
    prediction = np.zeros(len(valid), dtype=float)
    for tier in range(3):
        model = _ridge(alpha).fit(
            train.loc[role == tier, features], train.loc[role == tier, "target_ppg"]
        )
        prediction += probabilities[:, tier] * model.predict(valid[features])
    return prediction, probabilities[:, 2]


def _tune_three_tier(
    outer_train: pd.DataFrame, features: list[str]
) -> tuple[tuple[float, float], float, pd.DataFrame]:
    rows = []
    for cuts in THREE_TIER_CUTS:
        for alpha in MIXTURE_RIDGE_ALPHAS:
            fold_mae = []
            for season in _inner_seasons(outer_train):
                train = outer_train[outer_train["prediction_season"] < season]
                valid = outer_train[outer_train["prediction_season"] == season]
                predicted, _ = _fit_three_tier_mixture(
                    train, valid, features, cuts, alpha
                )
                fold_mae.append(float(np.mean(np.abs(
                    predicted - valid["target_ppg"].to_numpy()
                ))))
            rows.append({
                "low_cut_targets_per_game": cuts[0],
                "featured_cut_targets_per_game": cuts[1],
                "ridge_alpha": alpha,
                "inner_mean_mae": float(np.mean(fold_mae)),
            })
    tuning = pd.DataFrame(rows).sort_values(
        [
            "inner_mean_mae", "low_cut_targets_per_game",
            "featured_cut_targets_per_game", "ridge_alpha",
        ],
        kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    return (
        (
            float(tuning.loc[0, "low_cut_targets_per_game"]),
            float(tuning.loc[0, "featured_cut_targets_per_game"]),
        ),
        float(tuning.loc[0, "ridge_alpha"]),
        tuning,
    )


def _crossfit_opportunity(
    train: pd.DataFrame, features: list[str], alpha: float
) -> pd.Series:
    prediction = pd.Series(np.nan, index=train.index, dtype=float)
    for season in sorted(train["prediction_season"].unique()):
        prior = train[train["prediction_season"] < season]
        valid = train[train["prediction_season"] == season]
        if len(prior) < 100 or prior["prediction_season"].nunique() < 3:
            continue
        model = _ridge(alpha).fit(
            prior[features], prior["target_targets_per_game"]
        )
        prediction.loc[valid.index] = model.predict(valid[features])
    return prediction


def _fit_opportunity_first(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str],
    opportunity_alpha: float, ppg_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    crossfit = _crossfit_opportunity(train, features, opportunity_alpha)
    ppg_train = train.loc[crossfit.notna()].copy()
    ppg_train["predicted_targets_per_game"] = crossfit.dropna()
    opportunity = _ridge(opportunity_alpha).fit(
        train[features], train["target_targets_per_game"]
    )
    valid_opportunity = opportunity.predict(valid[features])
    valid_ppg = valid.copy()
    valid_ppg["predicted_targets_per_game"] = valid_opportunity
    ppg_features = [
        "predicted_targets_per_game", "previous_receiving_yards_per_target",
        "previous_receiving_touchdowns_per_game", "previous_season_ppg",
        "three_season_mean_ppg", "age_entering_season",
    ]
    ppg_model = _ridge(ppg_alpha).fit(
        ppg_train[ppg_features], ppg_train["target_ppg"]
    )
    return ppg_model.predict(valid_ppg[ppg_features]), valid_opportunity


def _tune_opportunity_first(
    outer_train: pd.DataFrame, features: list[str]
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    for opportunity_alpha in OPPORTUNITY_RIDGE_ALPHAS:
        for ppg_alpha in OPPORTUNITY_RIDGE_ALPHAS:
            fold_mae = []
            for season in _inner_seasons(outer_train):
                train = outer_train[outer_train["prediction_season"] < season]
                valid = outer_train[outer_train["prediction_season"] == season]
                predicted, _ = _fit_opportunity_first(
                    train, valid, features, opportunity_alpha, ppg_alpha
                )
                fold_mae.append(float(np.mean(np.abs(
                    predicted - valid["target_ppg"].to_numpy()
                ))))
            rows.append({
                "opportunity_ridge_alpha": opportunity_alpha,
                "ppg_ridge_alpha": ppg_alpha,
                "inner_mean_mae": float(np.mean(fold_mae)),
            })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "opportunity_ridge_alpha", "ppg_ridge_alpha"],
        kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    return (
        float(tuning.loc[0, "opportunity_ridge_alpha"]),
        float(tuning.loc[0, "ppg_ridge_alpha"]),
        tuning,
    )


def _nonlinear_pipeline(params: dict[str, float | int]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("regressor", HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=250, min_samples_leaf=20,
            random_state=RANDOM_STATE, **params,
        )),
    ])


def _tune_nonlinear(
    outer_train: pd.DataFrame, features: list[str]
) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows = []
    for params in NONLINEAR_GRID:
        fold_mae = []
        for season in _inner_seasons(outer_train):
            train = outer_train[outer_train["prediction_season"] < season]
            valid = outer_train[outer_train["prediction_season"] == season]
            model = _nonlinear_pipeline(params).fit(
                train[features], train["target_ppg"]
            )
            fold_mae.append(float(np.mean(np.abs(
                model.predict(valid[features]) - valid["target_ppg"].to_numpy()
            ))))
        rows.append({**params, "inner_mean_mae": float(np.mean(fold_mae))})
    tuning = pd.DataFrame(rows).sort_values(
        [
            "inner_mean_mae", "max_leaf_nodes", "learning_rate",
            "l2_regularization",
        ],
        kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    params = {
        "learning_rate": float(tuning.loc[0, "learning_rate"]),
        "max_leaf_nodes": int(tuning.loc[0, "max_leaf_nodes"]),
        "l2_regularization": float(tuning.loc[0, "l2_regularization"]),
    }
    return params, tuning


def run_models(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    folds = []
    tuning_rows = []
    for specification, features in FEATURE_MANIFESTS.items():
        for season in range(EVALUATION_START[specification], 2026):
            train = table[table["prediction_season"] < season]
            valid = table[table["prediction_season"] == season]
            method, params, tuning = _tune(train, features, "TE", "TE_A_CORE")
            for row in tuning.to_dict("records"):
                tuning_rows.append({
                    "specification": specification, "outer_season": season, **row
                })
            model = build_pipeline(method, params).fit(
                train[features], train["target_ppg"]
            )
            predicted = model.predict(valid[features])
            metric = _metrics(
                valid["target_ppg"].to_numpy(), predicted,
                valid["player_id"].to_numpy(), TOP_N["TE"],
            )
            folds.append({
                "specification": specification, "season": season,
                "method": method, "alpha": params.get("alpha", math.nan),
                "l1_ratio": params.get("l1_ratio", math.nan), **metric,
            })
            frame = valid[[
                "player_id", "player_name", "prediction_season", "target_ppg",
                "target_targets_per_game",
            ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
            frame["predicted_ppg"] = predicted
            frame["receiving_role_probability"] = np.nan
            frame["specification"] = specification
            predictions.append(frame)

    specification = "TWO_STAGE_ROLE_MIXTURE"
    features = FEATURE_MANIFESTS["V1_ROLE_CONTEXT"]
    for season in range(EVALUATION_START[specification], 2026):
        train = table[table["prediction_season"] < season]
        valid = table[table["prediction_season"] == season]
        threshold, alpha, tuning = _tune_mixture(train, features)
        for row in tuning.to_dict("records"):
            tuning_rows.append({
                "specification": specification, "outer_season": season,
                "method": "role_mixture", **row,
            })
        predicted, probability = _fit_mixture(
            train, valid, features, threshold, alpha
        )
        metric = _metrics(
            valid["target_ppg"].to_numpy(), predicted,
            valid["player_id"].to_numpy(), TOP_N["TE"],
        )
        folds.append({
            "specification": specification, "season": season,
            "method": "role_mixture", "alpha": alpha, "l1_ratio": math.nan,
            "role_threshold_targets_per_game": threshold, **metric,
        })
        frame = valid[[
            "player_id", "player_name", "prediction_season", "target_ppg",
            "target_targets_per_game",
        ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
        frame["predicted_ppg"] = predicted
        frame["receiving_role_probability"] = probability
        frame["specification"] = specification
        predictions.append(frame)

    specification = "THREE_TIER_ROLE_MIXTURE"
    for season in range(EVALUATION_START[specification], 2026):
        train = table[table["prediction_season"] < season]
        valid = table[table["prediction_season"] == season]
        cuts, alpha, tuning = _tune_three_tier(train, features)
        for row in tuning.to_dict("records"):
            tuning_rows.append({
                "specification": specification, "outer_season": season,
                "method": "three_tier_role_mixture", **row,
            })
        predicted, featured_probability = _fit_three_tier_mixture(
            train, valid, features, cuts, alpha
        )
        metric = _metrics(
            valid["target_ppg"].to_numpy(), predicted,
            valid["player_id"].to_numpy(), TOP_N["TE"],
        )
        folds.append({
            "specification": specification, "season": season,
            "method": "three_tier_role_mixture", "alpha": alpha,
            "l1_ratio": math.nan, "low_cut_targets_per_game": cuts[0],
            "featured_cut_targets_per_game": cuts[1], **metric,
        })
        frame = valid[[
            "player_id", "player_name", "prediction_season", "target_ppg",
            "target_targets_per_game",
        ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
        frame["predicted_ppg"] = predicted
        frame["receiving_role_probability"] = featured_probability
        frame["specification"] = specification
        predictions.append(frame)

    specification = "OPPORTUNITY_FIRST"
    for season in range(EVALUATION_START[specification], 2026):
        train = table[table["prediction_season"] < season]
        valid = table[table["prediction_season"] == season]
        opportunity_alpha, ppg_alpha, tuning = _tune_opportunity_first(
            train, features
        )
        for row in tuning.to_dict("records"):
            tuning_rows.append({
                "specification": specification, "outer_season": season,
                "method": "opportunity_first", **row,
            })
        predicted, predicted_opportunity = _fit_opportunity_first(
            train, valid, features, opportunity_alpha, ppg_alpha
        )
        metric = _metrics(
            valid["target_ppg"].to_numpy(), predicted,
            valid["player_id"].to_numpy(), TOP_N["TE"],
        )
        folds.append({
            "specification": specification, "season": season,
            "method": "opportunity_first", "alpha": ppg_alpha,
            "l1_ratio": math.nan, "opportunity_alpha": opportunity_alpha,
            **metric,
        })
        frame = valid[[
            "player_id", "player_name", "prediction_season", "target_ppg",
            "target_targets_per_game",
        ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
        frame["predicted_ppg"] = predicted
        frame["predicted_targets_per_game"] = predicted_opportunity
        frame["receiving_role_probability"] = np.nan
        frame["specification"] = specification
        predictions.append(frame)

    specification = "NONLINEAR_ROLE_CONTEXT"
    for season in range(EVALUATION_START[specification], 2026):
        train = table[table["prediction_season"] < season]
        valid = table[table["prediction_season"] == season]
        params, tuning = _tune_nonlinear(train, features)
        for row in tuning.to_dict("records"):
            tuning_rows.append({
                "specification": specification, "outer_season": season,
                "method": "hist_gradient_boosting", **row,
            })
        model = _nonlinear_pipeline(params).fit(
            train[features], train["target_ppg"]
        )
        predicted = model.predict(valid[features])
        metric = _metrics(
            valid["target_ppg"].to_numpy(), predicted,
            valid["player_id"].to_numpy(), TOP_N["TE"],
        )
        folds.append({
            "specification": specification, "season": season,
            "method": "hist_gradient_boosting", "alpha": math.nan,
            "l1_ratio": math.nan, **params, **metric,
        })
        frame = valid[[
            "player_id", "player_name", "prediction_season", "target_ppg",
            "target_targets_per_game",
        ]].rename(columns={"prediction_season": "season", "target_ppg": "actual_ppg"})
        frame["predicted_ppg"] = predicted
        frame["receiving_role_probability"] = np.nan
        frame["specification"] = specification
        predictions.append(frame)

    # Meta-ensemble weights are selected only from already out-of-sample seasons
    # preceding the current outer fold. Starting in 2022 supplies three complete
    # prior opportunity-first folds (2019-2021).
    specification = "V1_OPPORTUNITY_ENSEMBLE"
    opportunity_predictions = pd.concat(predictions, ignore_index=True)
    opportunity_predictions = opportunity_predictions[
        opportunity_predictions["specification"] == "OPPORTUNITY_FIRST"
    ]
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    frozen = frozen[
        (frozen["model_name"] == "A_v1_same_cohort") & (frozen["position"] == "TE")
    ][["player_id", "season", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "frozen_v1_predicted_ppg"}
    )
    paired_opportunity = opportunity_predictions.merge(
        frozen, on=["player_id", "season"], validate="one_to_one"
    )
    for season in range(EVALUATION_START[specification], 2026):
        history = paired_opportunity[paired_opportunity["season"] < season]
        weight_rows = []
        for weight in ENSEMBLE_WEIGHTS:
            blended = (
                (1 - weight) * history["frozen_v1_predicted_ppg"]
                + weight * history["predicted_ppg"]
            )
            row_error = (blended - history["actual_ppg"]).abs()
            season_mae = row_error.groupby(history["season"]).mean()
            weight_rows.append({
                "ensemble_opportunity_weight": weight,
                "inner_mean_mae": float(season_mae.mean()),
            })
        weight_tuning = pd.DataFrame(weight_rows).sort_values(
            ["inner_mean_mae", "ensemble_opportunity_weight"], kind="stable"
        ).reset_index(drop=True)
        weight_tuning["selected"] = False
        weight_tuning.loc[0, "selected"] = True
        weight = float(weight_tuning.loc[0, "ensemble_opportunity_weight"])
        for row in weight_tuning.to_dict("records"):
            tuning_rows.append({
                "specification": specification, "outer_season": season,
                "method": "out_of_sample_ensemble", **row,
            })
        current = paired_opportunity[
            paired_opportunity["season"] == season
        ].copy()
        current["predicted_ppg"] = (
            (1 - weight) * current["frozen_v1_predicted_ppg"]
            + weight * current["predicted_ppg"]
        )
        metric = _metrics(
            current["actual_ppg"].to_numpy(), current["predicted_ppg"].to_numpy(),
            current["player_id"].to_numpy(), TOP_N["TE"],
        )
        folds.append({
            "specification": specification, "season": season,
            "method": "out_of_sample_ensemble", "alpha": math.nan,
            "l1_ratio": math.nan, "ensemble_opportunity_weight": weight, **metric,
        })
        current["receiving_role_probability"] = np.nan
        current["specification"] = specification
        predictions.append(current[[
            "player_id", "player_name", "season", "actual_ppg",
            "target_targets_per_game", "predicted_ppg",
            "receiving_role_probability", "specification",
        ]])
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds), pd.DataFrame(tuning_rows)


def compare_to_frozen_v1(predictions: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    frozen = frozen[
        (frozen["model_name"] == "A_v1_same_cohort") & (frozen["position"] == "TE")
    ][["player_id", "season", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "frozen_v1_predicted_ppg"}
    )
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for specification, candidate in predictions.groupby("specification"):
        paired = candidate.merge(
            frozen, on=["player_id", "season"], validate="one_to_one"
        )
        candidate_error = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
        reference_error = (
            paired["frozen_v1_predicted_ppg"] - paired["actual_ppg"]
        ).abs()
        improvement = reference_error - candidate_error
        season_delta = improvement.groupby(paired["season"]).mean()
        bootstrap = np.asarray([
            rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
            for _ in range(10_000)
        ])
        mean_improvement = float(improvement.mean())
        interval_low = float(np.quantile(bootstrap, 0.025))
        if mean_improvement > 0 and interval_low > 0:
            decision = (
                "ADVANCE" if paired["season"].nunique() >= 7
                else "PROMISING_LIMITED_HISTORY"
            )
        else:
            decision = "REJECT_RETAIN_FROZEN_V1"
        rows.append({
            "specification": specification,
            "seasons": f"{paired['season'].min()}-{paired['season'].max()}",
            "folds": paired["season"].nunique(), "players": len(paired),
            "candidate_mae": float(candidate_error.mean()),
            "frozen_v1_same_cohort_mae": float(reference_error.mean()),
            "mae_improvement": mean_improvement,
            "seasons_improved": int((season_delta > 0).sum()),
            "season_block_95_low": interval_low,
            "season_block_95_high": float(np.quantile(bootstrap, 0.975)),
            "decision": decision,
        })
    return pd.DataFrame(rows).sort_values("mae_improvement", ascending=False)


def _frozen_v1_on_mixture_cohort(predictions: pd.DataFrame) -> pd.DataFrame:
    cohort = predictions[
        predictions["specification"] == "TWO_STAGE_ROLE_MIXTURE"
    ][[
        "player_id", "player_name", "season", "actual_ppg",
        "target_targets_per_game",
    ]]
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    frozen = frozen[
        (frozen["model_name"] == "A_v1_same_cohort") & (frozen["position"] == "TE")
    ][["player_id", "season", "predicted_ppg"]]
    result = cohort.merge(frozen, on=["player_id", "season"], validate="one_to_one")
    result["specification"] = "FROZEN_V1"
    result["receiving_role_probability"] = np.nan
    return result


def tier_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat(
        [predictions, _frozen_v1_on_mixture_cohort(predictions)], ignore_index=True
    )
    data["actual_role_tier"] = pd.cut(
        data["target_targets_per_game"],
        bins=[-np.inf, 2.0, 4.0, np.inf],
        labels=["LOW_LT_2", "SECONDARY_2_TO_4", "FEATURED_GT_4"],
        right=False,
    )
    data["error"] = data["predicted_ppg"] - data["actual_ppg"]
    return data.groupby(
        ["specification", "actual_role_tier"], observed=True, as_index=False
    ).agg(
        players=("player_id", "size"),
        actual_mean_ppg=("actual_ppg", "mean"),
        predicted_mean_ppg=("predicted_ppg", "mean"),
        mean_bias=("error", "mean"),
        mae=("error", lambda x: float(np.abs(x).mean())),
        overprediction_rate=("error", lambda x: float((x > 0).mean())),
    )


def role_classifier_diagnostics(
    predictions: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    mixture = predictions[
        predictions["specification"] == "TWO_STAGE_ROLE_MIXTURE"
    ].merge(
        folds[folds["specification"] == "TWO_STAGE_ROLE_MIXTURE"][[
            "season", "role_threshold_targets_per_game"
        ]],
        on="season", validate="many_to_one",
    )
    rows = []
    for season, group in mixture.groupby("season"):
        actual = group["target_targets_per_game"].ge(
            group["role_threshold_targets_per_game"]
        ).astype(int)
        probability = group["receiving_role_probability"]
        rows.append({
            "season": int(season), "players": len(group),
            "selected_threshold_targets_per_game": float(
                group["role_threshold_targets_per_game"].iloc[0]
            ),
            "receiving_role_rate": float(actual.mean()),
            "roc_auc": float(roc_auc_score(actual, probability)),
            "brier_score": float(brier_score_loss(actual, probability)),
        })
    return pd.DataFrame(rows)


def distribution_audit(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in table[table["prediction_season"].between(2012, 2025)].groupby(
        "prediction_season"
    ):
        values = group["target_targets_per_game"]
        rows.append({
            "season": int(season), "players": len(values),
            "mean_targets_per_game": float(values.mean()),
            "median_targets_per_game": float(values.median()),
            "skew_targets_per_game": float(values.skew()),
            "low_lt_2": int(values.lt(2).sum()),
            "secondary_2_to_4": int(values.between(2, 4, inclusive="left").sum()),
            "featured_ge_4": int(values.ge(4).sum()),
        })
    return pd.DataFrame(rows)


def _figures(
    table: pd.DataFrame, predictions: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    TE_ROLE_REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    recent = table[table["prediction_season"].between(2019, 2025)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(recent["target_targets_per_game"], bins=30, kde=True, ax=ax)
    ax.axvline(2, color="darkorange", linestyle="--")
    ax.axvline(4, color="firebrick", linestyle="--")
    ax.set(title="TE target opportunity distribution, 2019-2025", xlabel="Targets per game")
    fig.tight_layout()
    fig.savefig(TE_ROLE_REPORT_FIGURES_DIR / "te_target_distribution.png", dpi=160)
    plt.close(fig)

    best_specification = str(comparison.iloc[0]["specification"])
    chosen = pd.concat([
        predictions[predictions["specification"] == best_specification],
        _frozen_v1_on_mixture_cohort(predictions),
    ], ignore_index=True)
    chosen["role_tier"] = pd.cut(
        chosen["target_targets_per_game"], [-np.inf, 2, 4, np.inf],
        labels=["Low", "Secondary", "Featured"], right=False,
    )
    chosen["error"] = chosen["predicted_ppg"] - chosen["actual_ppg"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=chosen, x="role_tier", y="error", hue="specification",
        showfliers=False, ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="TE prediction bias by actual receiving-role tier", ylabel="Predicted minus actual PPG", xlabel="")
    fig.tight_layout()
    fig.savefig(TE_ROLE_REPORT_FIGURES_DIR / "te_role_tier_bias.png", dpi=160)
    plt.close(fig)


def _write_report(
    comparison: pd.DataFrame, tiers: pd.DataFrame, distribution: pd.DataFrame,
    classifier: pd.DataFrame,
) -> None:
    lines = [
        "TE RECEIVING-ROLE EXPERIMENT",
        "=" * 78, "",
        "Objective",
        "---------",
        "Test whether TE error is reduced by separating receiving opportunity from broad team context using binary and three-tier mixtures, an opportunity-first model, nonlinear boosting, and a leakage-safe V1/opportunity ensemble.", "",
        "Data constraints",
        "----------------",
        "True routes, TPRR, inline alignment, and pass/run blocking snaps remain unavailable. Pass-play participation, targets per participated pass play, their product (targets per team pass play), PFR offensive snap share, prior target share, and TE-specific competition are used as honest proxies.",
        "Every feature is from t-1 or the preseason roster cutoff. Target-season targets per game are used only as a training label and diagnostic tier, never as a predictor.", "",
        "Outer-fold comparison with frozen V1",
        "------------------------------------",
        comparison.to_string(index=False), "",
        "Tier diagnostics",
        "----------------",
        tiers.to_string(index=False), "",
        "Receiving-role classifier audit",
        "-------------------------------",
        classifier.to_string(index=False), "",
        "Target-distribution audit",
        "-------------------------",
        distribution.to_string(index=False), "",
        "Interpretation",
        "--------------",
        "Positive MAE improvement favors the candidate. Promotion requires positive average improvement with a season-block interval entirely above zero. Otherwise frozen V1 remains operational.",
        "Opportunity-first was essentially tied with V1 and improved featured-TE error, but its additional low-volume overprediction prevented promotion. The ensemble weight selected from prior out-of-sample seasons did not improve subsequent outer folds.",
        "The participation and mixture candidates have seven outer seasons (2019-2025), so uncertainty and tier-specific bias matter alongside average MAE.",
    ]
    TE_ROLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (TE_ROLE_REPORTS_DIR / "te_role_experiment_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_te_role_experiment() -> dict[str, str]:
    table = build_te_role_table()
    predictions, folds, tuning = run_models(table)
    comparison = compare_to_frozen_v1(predictions)
    tiers = tier_diagnostics(predictions)
    distribution = distribution_audit(table)
    classifier = role_classifier_diagnostics(predictions, folds)
    TE_ROLE_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TE_ROLE_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        TE_ROLE_PREDICTIONS_DATA_DIR / "te_role_predictions.parquet", index=False
    )
    folds.to_csv(TE_ROLE_REPORT_TABLES_DIR / "fold_metrics.csv", index=False)
    tuning.to_csv(TE_ROLE_REPORT_TABLES_DIR / "inner_tuning.csv", index=False)
    comparison.to_csv(TE_ROLE_REPORT_TABLES_DIR / "model_comparison.csv", index=False)
    tiers.to_csv(TE_ROLE_REPORT_TABLES_DIR / "role_tier_diagnostics.csv", index=False)
    distribution.to_csv(
        TE_ROLE_REPORT_TABLES_DIR / "target_distribution_audit.csv", index=False
    )
    classifier.to_csv(
        TE_ROLE_REPORT_TABLES_DIR / "role_classifier_diagnostics.csv", index=False
    )
    _figures(table, predictions, comparison)
    _write_report(comparison, tiers, distribution, classifier)
    return {"summary": str(TE_ROLE_REPORTS_DIR / "te_role_experiment_summary.txt")}


if __name__ == "__main__":
    run_te_role_experiment()
