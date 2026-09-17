"""Use cross-fitted TE receiving roles only as roster-competition context."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.attempt3_features import _availability_features
from src.attempt3_models import _tune
from src.attempt21_features import V1_FEATURES
from src.attempt21_models import TOP_N, _metrics, build_pipeline
from src.config import (
    ATTEMPT2_INTERIM_DATA_DIR,
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    DATA_DIR,
    FEATURE35_PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TE_COMPETITION_PREDICTIONS_DATA_DIR,
    TE_COMPETITION_PROCESSED_DATA_DIR,
    TE_COMPETITION_REPORT_TABLES_DIR,
    TE_COMPETITION_REPORTS_DIR,
)
from src.feature_experiments import build_participation_features
from src.te_role_experiment import _role_classifier, _snap_features


class TECompetitionExperimentError(RuntimeError):
    """Raised when receiving-adjusted competition contracts fail."""


ROLE_LABEL_TARGETS_PER_GAME = 1.5
ROLE_PROBABILITY_THRESHOLD = 0.5
V1 = list(V1_FEATURES["TE"])
INTRINSIC_ROLE_FEATURES = [
    "previous_season_ppg",
    "previous_targets_per_game",
    "previous_target_share",
    "previous_pass_play_participation_rate",
    "previous_targets_per_pass_play_participated",
    "previous_targets_per_team_pass_play",
    "previous_offensive_snap_share",
    "previous_receiving_yards_per_game",
    "previous_receiving_yards_per_target",
    "previous_receiving_touchdowns_per_game",
    "age_entering_season",
    "years_experience_entering_season",
    "previous_availability_rate",
    "three_year_durability_index",
    "durability_history_seasons",
]
RAW_COMPETITION_FEATURES = [
    "raw_other_te_count",
    "raw_other_te_max_previous_targets_per_game",
]
EFFECTIVE_COUNT_FEATURES = [
    "effective_receiving_competitor_count",
    "max_competitor_receiving_role_probability",
    "receiving_competitors_above_probability_threshold",
    "competitor_role_unknown_count",
]
WEIGHTED_WORKLOAD_FEATURES = [
    "probability_weighted_competitor_targets_per_game",
    "max_probability_weighted_competitor_targets_per_game",
    "probability_weighted_competitor_target_share",
    "competitor_role_unknown_count",
]
CANDIDATE_MANIFESTS = {
    "V1_PLUS_RAW_TE_COMPETITION": [*V1, *RAW_COMPETITION_FEATURES],
    "V1_PLUS_EFFECTIVE_RECEIVING_COUNT": [*V1, *EFFECTIVE_COUNT_FEATURES],
    "V1_PLUS_ROLE_WEIGHTED_WORKLOAD": [*V1, *WEIGHTED_WORKLOAD_FEATURES],
    "V1_PLUS_RECEIVING_ADJUSTED_COMPETITION": [
        *V1,
        *RAW_COMPETITION_FEATURES,
        *EFFECTIVE_COUNT_FEATURES,
        *[feature for feature in WEIGHTED_WORKLOAD_FEATURES
          if feature != "competitor_role_unknown_count"],
    ],
}
RESIDUAL_MANIFESTS = {
    "FROZEN_V1_PLUS_RAW_COMPETITION_RESIDUAL": RAW_COMPETITION_FEATURES,
    "FROZEN_V1_PLUS_EFFECTIVE_COUNT_RESIDUAL": EFFECTIVE_COUNT_FEATURES,
    "FROZEN_V1_PLUS_WEIGHTED_WORKLOAD_RESIDUAL": WEIGHTED_WORKLOAD_FEATURES,
    "FROZEN_V1_PLUS_FULL_COMPETITION_RESIDUAL": [
        *RAW_COMPETITION_FEATURES,
        *EFFECTIVE_COUNT_FEATURES,
        *[feature for feature in WEIGHTED_WORKLOAD_FEATURES
          if feature != "competitor_role_unknown_count"],
    ],
}
RESIDUAL_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
RESIDUAL_SHRINKAGE = [0.0, 0.25, 0.5, 0.75, 1.0]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0))


def build_role_utility_table() -> pd.DataFrame:
    context = pd.read_parquet(
        ATTEMPT2_INTERIM_DATA_DIR / "prediction_context.parquet"
    )
    context = context[
        (context["roster_position"] == "TE")
        & context["prediction_season"].between(2007, 2025)
    ].copy()
    seasons = pd.read_parquet(DATA_DIR / "interim/player_seasons_all.parquet")
    prior = seasons[[
        "player_id", "season", "fantasy_points_per_game", "games_played",
        "targets_per_game", "receiving_yards_per_game", "receiving_touchdowns_per_game",
        "receiving_yards", "targets",
    ]].copy()
    prior["prediction_season"] = prior.pop("season") + 1
    prior = prior.rename(columns={
        "fantasy_points_per_game": "previous_season_ppg",
        "games_played": "previous_season_games_played",
        "targets_per_game": "previous_targets_per_game",
        "receiving_yards_per_game": "previous_receiving_yards_per_game",
        "receiving_touchdowns_per_game": "previous_receiving_touchdowns_per_game",
    })
    prior["previous_receiving_yards_per_target"] = _ratio(
        prior["receiving_yards"], prior["targets"]
    )
    prior = prior.drop(columns=["receiving_yards", "targets"])

    target = seasons[[
        "player_id", "season", "games_played", "targets_per_game"
    ]].rename(columns={
        "season": "prediction_season",
        "games_played": "target_games_played",
        "targets_per_game": "target_targets_per_game",
    })
    workload = pd.read_parquet(
        ATTEMPT2_INTERIM_DATA_DIR / "player_workload_shares.parquet",
        columns=["player_id", "prediction_season", "previous_target_share"],
    )
    participation = build_participation_features()[[
        "player_id", "prediction_season", "previous_primary_team",
        "previous_pass_play_participation_rate",
        "previous_targets_per_pass_play_participated",
    ]]
    participation["previous_targets_per_team_pass_play"] = (
        participation["previous_pass_play_participation_rate"]
        * participation["previous_targets_per_pass_play_participated"]
    )
    snap = _snap_features()[[
        "player_id", "prediction_season", "previous_offensive_snap_share"
    ]]
    availability = _availability_features()
    players = pd.read_parquet(
        DATA_DIR / "raw/players.parquet",
        columns=["gsis_id", "rookie_season"],
    ).drop_duplicates("gsis_id").rename(columns={"gsis_id": "player_id"})

    utility = context.merge(
        prior, on=["player_id", "prediction_season"], how="left", validate="one_to_one"
    ).merge(
        workload, on=["player_id", "prediction_season"], how="left", validate="one_to_one"
    ).merge(
        participation,
        on=["player_id", "prediction_season", "previous_primary_team"],
        how="left", validate="one_to_one",
    ).merge(
        snap, on=["player_id", "prediction_season"], how="left", validate="one_to_one"
    ).merge(
        availability, on=["player_id", "prediction_season"],
        how="left", validate="one_to_one",
    ).merge(
        players, on="player_id", how="left", validate="many_to_one"
    ).merge(
        target, on=["player_id", "prediction_season"], how="left", validate="one_to_one"
    )
    birth = pd.to_datetime(utility["birth_date"], errors="coerce")
    cutoff = pd.to_datetime(
        utility["prediction_season"].astype(str) + "-09-01", errors="coerce"
    )
    utility["age_entering_season"] = (
        (cutoff - birth).dt.days / 365.2425
    )
    utility["years_experience_entering_season"] = (
        utility["prediction_season"] - utility["rookie_season"]
    ).clip(lower=0)
    utility["has_previous_history"] = utility["previous_season_ppg"].notna().astype(int)
    utility["target_receiving_role"] = (
        utility["target_targets_per_game"] >= ROLE_LABEL_TARGETS_PER_GAME
    ).astype("Int64")
    utility.loc[utility["target_targets_per_game"].isna(), "target_receiving_role"] = pd.NA
    if utility.duplicated(["player_id", "prediction_season"]).any():
        raise TECompetitionExperimentError("Duplicate preseason TE utility rows")
    TE_COMPETITION_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    utility.to_parquet(
        TE_COMPETITION_PROCESSED_DATA_DIR / "te_role_utility_dataset.parquet",
        index=False,
    )
    return utility


def crossfit_role_probabilities(
    utility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = []
    audits = []
    for season in sorted(utility["prediction_season"].unique()):
        train = utility[
            (utility["prediction_season"] < season)
            & utility["has_previous_history"].eq(1)
            & utility["target_games_played"].ge(4)
            & utility["target_receiving_role"].notna()
        ]
        valid = utility[utility["prediction_season"] == season].copy()
        valid["competitor_receiving_role_probability"] = np.nan
        class_counts = train["target_receiving_role"].value_counts()
        if len(train) >= 100 and len(class_counts) == 2 and class_counts.min() >= 20:
            model = _role_classifier().fit(
                train[INTRINSIC_ROLE_FEATURES],
                train["target_receiving_role"].astype(int),
            )
            scorable = valid["has_previous_history"].eq(1)
            valid.loc[scorable, "competitor_receiving_role_probability"] = (
                model.predict_proba(valid.loc[scorable, INTRINSIC_ROLE_FEATURES])[:, 1]
            )
            audited = valid[
                scorable & valid["target_games_played"].ge(4)
                & valid["target_receiving_role"].notna()
            ]
            if audited["target_receiving_role"].nunique() == 2:
                actual = audited["target_receiving_role"].astype(int)
                probability = audited["competitor_receiving_role_probability"]
                audits.append({
                    "season": int(season), "training_rows": len(train),
                    "scored_rows": int(scorable.sum()), "audit_rows": len(audited),
                    "roc_auc": float(roc_auc_score(actual, probability)),
                    "brier_score": float(brier_score_loss(actual, probability)),
                })
        outputs.append(valid[[
            "player_id", "prediction_season", "has_previous_history",
            "previous_targets_per_game", "previous_target_share",
            "competitor_receiving_role_probability",
        ]])
    probabilities = pd.concat(outputs, ignore_index=True)
    if probabilities.duplicated(["player_id", "prediction_season"]).any():
        raise TECompetitionExperimentError("Duplicate cross-fitted TE probabilities")
    probabilities.to_parquet(
        TE_COMPETITION_PROCESSED_DATA_DIR / "crossfit_role_probabilities.parquet",
        index=False,
    )
    audit = pd.DataFrame(audits)
    audit.to_csv(
        TE_COMPETITION_REPORT_TABLES_DIR / "role_classifier_audit.csv", index=False
    )
    return probabilities, audit


def build_competition_features(probabilities: pd.DataFrame) -> pd.DataFrame:
    competition = pd.read_parquet(
        ATTEMPT2_INTERIM_DATA_DIR / "competition_long.parquet"
    )
    competition = competition[
        (competition["position"] == "TE")
        & (competition["competitor_position"] == "TE")
    ].copy()
    competitor = probabilities.rename(columns={
        "player_id": "competitor_id",
        "has_previous_history": "competitor_has_prior_history",
        "previous_targets_per_game": "competitor_previous_targets_per_game",
        "previous_target_share": "competitor_previous_target_share",
    })
    competition = competition.merge(
        competitor,
        on=["competitor_id", "prediction_season"],
        how="left", validate="many_to_one",
    )
    probability = competition["competitor_receiving_role_probability"]
    competition["weighted_targets_per_game"] = (
        probability * competition["competitor_previous_targets_per_game"]
    )
    competition["weighted_target_share"] = (
        probability * competition["competitor_previous_target_share"]
    )

    def aggregate(group: pd.DataFrame) -> pd.Series:
        known = group["competitor_receiving_role_probability"].notna()
        weighted_tpg = group["weighted_targets_per_game"]
        weighted_share = group["weighted_target_share"]
        return pd.Series({
            "raw_other_te_count": len(group),
            "raw_other_te_max_previous_targets_per_game": group[
                "competitor_previous_targets_per_game"
            ].max(),
            "known_competitor_role_count": int(known.sum()),
            "competitor_role_unknown_count": int((~known).sum()),
            "effective_receiving_competitor_count": (
                float(group.loc[known, "competitor_receiving_role_probability"].sum())
                if known.any() else np.nan
            ),
            "max_competitor_receiving_role_probability": (
                float(group.loc[known, "competitor_receiving_role_probability"].max())
                if known.any() else np.nan
            ),
            "receiving_competitors_above_probability_threshold": (
                int((group.loc[known, "competitor_receiving_role_probability"]
                     >= ROLE_PROBABILITY_THRESHOLD).sum())
                if known.any() else np.nan
            ),
            "probability_weighted_competitor_targets_per_game": (
                float(weighted_tpg.sum(min_count=1))
            ),
            "max_probability_weighted_competitor_targets_per_game": weighted_tpg.max(),
            "probability_weighted_competitor_target_share": (
                float(weighted_share.sum(min_count=1))
            ),
        })

    features = competition.groupby(
        ["player_id", "prediction_season"], as_index=False
    ).apply(aggregate, include_groups=False).reset_index()
    if "level_2" in features:
        features = features.drop(columns="level_2")
    if features.duplicated(["player_id", "prediction_season"]).any():
        raise TECompetitionExperimentError("Duplicate competition feature rows")
    features.to_parquet(
        TE_COMPETITION_PROCESSED_DATA_DIR / "receiving_adjusted_competition.parquet",
        index=False,
    )
    return features


def build_modeling_table(features: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_parquet(
        FEATURE35_PROCESSED_DATA_DIR / "te_feature_experiments_dataset.parquet"
    )
    target = pd.read_parquet(
        DATA_DIR / "interim/player_seasons_all.parquet",
        columns=["player_id", "season", "targets_per_game"],
    ).rename(columns={
        "season": "prediction_season", "targets_per_game": "target_targets_per_game"
    })
    table = table.merge(
        features, on=["player_id", "prediction_season"],
        how="left", validate="one_to_one",
    ).merge(
        target, on=["player_id", "prediction_season"],
        how="left", validate="one_to_one",
    )
    for column in ["raw_other_te_count", "competitor_role_unknown_count"]:
        table[column] = table[column].fillna(0)
    if table["previous_season_ppg"].isna().any():
        raise TECompetitionExperimentError("No-history focal TE entered evaluation")
    table.to_parquet(
        TE_COMPETITION_PROCESSED_DATA_DIR / "te_competition_modeling_dataset.parquet",
        index=False,
    )
    return table


def _tune_residual_adjustment(
    history: pd.DataFrame, features: list[str]
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    seasons = sorted(history["prediction_season"].unique())
    inner_seasons = [
        season for season in seasons
        if (history["prediction_season"] < season).sum() >= 100
        and len([prior for prior in seasons if prior < season]) >= 2
    ][-5:]
    if len(inner_seasons) < 2:
        raise TECompetitionExperimentError(
            "Residual adjustment has insufficient inner history"
        )
    for alpha in RESIDUAL_RIDGE_ALPHAS:
        for shrinkage in RESIDUAL_SHRINKAGE:
            fold_mae = []
            for season in inner_seasons:
                train = history[history["prediction_season"] < season]
                valid = history[history["prediction_season"] == season]
                if shrinkage == 0:
                    predicted = valid["frozen_v1_predicted_ppg"].to_numpy()
                else:
                    model = build_pipeline("ridge", {"alpha": alpha}).fit(
                        train[features], train["v1_residual"]
                    )
                    predicted = (
                        valid["frozen_v1_predicted_ppg"].to_numpy()
                        + shrinkage * model.predict(valid[features])
                    )
                fold_mae.append(float(np.mean(np.abs(
                    predicted - valid["target_ppg"].to_numpy()
                ))))
            rows.append({
                "residual_ridge_alpha": alpha,
                "residual_shrinkage": shrinkage,
                "inner_mean_mae": float(np.mean(fold_mae)),
            })
    tuning = pd.DataFrame(rows).sort_values(
        ["inner_mean_mae", "residual_shrinkage", "residual_ridge_alpha"],
        kind="stable",
    ).reset_index(drop=True)
    tuning["selected"] = False
    tuning.loc[0, "selected"] = True
    return (
        float(tuning.loc[0, "residual_ridge_alpha"]),
        float(tuning.loc[0, "residual_shrinkage"]),
        tuning,
    )


def run_models(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    folds = []
    tuning_rows = []
    for specification, features in CANDIDATE_MANIFESTS.items():
        for season in range(2012, 2026):
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
            ]].rename(columns={
                "prediction_season": "season", "target_ppg": "actual_ppg"
            })
            frame["predicted_ppg"] = predicted
            frame["specification"] = specification
            predictions.append(frame)

    frozen = _frozen_predictions().rename(columns={"season": "prediction_season"})
    residual_table = table.merge(
        frozen, on=["player_id", "prediction_season"],
        how="inner", validate="one_to_one",
    )
    residual_table["v1_residual"] = (
        residual_table["target_ppg"]
        - residual_table["frozen_v1_predicted_ppg"]
    )
    for specification, features in RESIDUAL_MANIFESTS.items():
        for season in range(2017, 2026):
            history = residual_table[residual_table["prediction_season"] < season]
            valid = residual_table[residual_table["prediction_season"] == season]
            alpha, shrinkage, tuning = _tune_residual_adjustment(history, features)
            for row in tuning.to_dict("records"):
                tuning_rows.append({
                    "specification": specification, "outer_season": season,
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
            folds.append({
                "specification": specification, "season": season,
                "method": "frozen_v1_residual_ridge", "alpha": alpha,
                "l1_ratio": math.nan, "residual_shrinkage": shrinkage, **metric,
            })
            frame = valid[[
                "player_id", "player_name", "prediction_season", "target_ppg",
                "target_targets_per_game",
            ]].rename(columns={
                "prediction_season": "season", "target_ppg": "actual_ppg"
            })
            frame["predicted_ppg"] = predicted
            frame["specification"] = specification
            predictions.append(frame)
    result = pd.concat(predictions, ignore_index=True)
    if result.duplicated(["player_id", "season", "specification"]).any():
        raise TECompetitionExperimentError("Duplicate competition predictions")
    return result, pd.DataFrame(folds), pd.DataFrame(tuning_rows)


def _frozen_predictions() -> pd.DataFrame:
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    return frozen[
        (frozen["model_name"] == "A_v1_same_cohort") & (frozen["position"] == "TE")
    ][["player_id", "season", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "frozen_v1_predicted_ppg"}
    )


def compare_to_frozen_v1(
    predictions: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    frozen = _frozen_predictions()
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for specification, candidate in predictions.groupby("specification"):
        paired = candidate.merge(
            frozen, on=["player_id", "season"], validate="one_to_one"
        )
        candidate_error = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
        frozen_error = (
            paired["frozen_v1_predicted_ppg"] - paired["actual_ppg"]
        ).abs()
        improvement = frozen_error - candidate_error
        season_delta = improvement.groupby(paired["season"]).mean()
        bootstrap = np.asarray([
            rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
            for _ in range(10_000)
        ])
        candidate_spearman = folds[
            folds["specification"] == specification
        ]["spearman"].mean()
        frozen_spearman = []
        for _, group in paired.groupby("season"):
            frozen_spearman.append(_metrics(
                group["actual_ppg"].to_numpy(),
                group["frozen_v1_predicted_ppg"].to_numpy(),
                group["player_id"].to_numpy(), TOP_N["TE"],
            )["spearman"])
        rows.append({
            "specification": specification, "folds": paired["season"].nunique(),
            "players": len(paired), "candidate_mae": float(candidate_error.mean()),
            "frozen_v1_mae": float(frozen_error.mean()),
            "mae_improvement": float(improvement.mean()),
            "seasons_improved": int((season_delta > 0).sum()),
            "season_block_95_low": float(np.quantile(bootstrap, 0.025)),
            "season_block_95_high": float(np.quantile(bootstrap, 0.975)),
            "candidate_mean_fold_spearman": float(candidate_spearman),
            "frozen_v1_mean_fold_spearman": float(np.mean(frozen_spearman)),
            "spearman_change": float(candidate_spearman - np.mean(frozen_spearman)),
        })
    return pd.DataFrame(rows).sort_values("mae_improvement", ascending=False)


def tier_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    cohort = predictions[
        predictions["specification"] == "V1_PLUS_RECEIVING_ADJUSTED_COMPETITION"
    ][[
        "player_id", "player_name", "season", "actual_ppg", "target_targets_per_game"
    ]]
    frozen = cohort.merge(
        _frozen_predictions(), on=["player_id", "season"], validate="one_to_one"
    ).rename(columns={"frozen_v1_predicted_ppg": "predicted_ppg"})
    frozen["specification"] = "FROZEN_V1"
    data = pd.concat([predictions, frozen], ignore_index=True)
    data["actual_role_tier"] = pd.cut(
        data["target_targets_per_game"], [-np.inf, 2, 4, np.inf],
        labels=["LOW_LT_2", "SECONDARY_2_TO_4", "FEATURED_GE_4"], right=False,
    )
    data["error"] = data["predicted_ppg"] - data["actual_ppg"]
    return data.groupby(
        ["specification", "actual_role_tier"], observed=True, as_index=False
    ).agg(
        players=("player_id", "size"), mean_bias=("error", "mean"),
        mae=("error", lambda x: float(np.abs(x).mean())),
        overprediction_rate=("error", lambda x: float((x > 0).mean())),
    )


def decision_table(
    comparison: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    frozen_predictions = _frozen_predictions()
    rows = []
    for row in comparison.itertuples(index=False):
        paired = predictions[
            predictions["specification"] == row.specification
        ].merge(
            frozen_predictions, on=["player_id", "season"], validate="one_to_one"
        )
        paired["actual_role_tier"] = pd.cut(
            paired["target_targets_per_game"], [-np.inf, 2, 4, np.inf],
            labels=["LOW_LT_2", "SECONDARY_2_TO_4", "FEATURED_GE_4"],
            right=False,
        )
        paired["candidate_error"] = paired["predicted_ppg"] - paired["actual_ppg"]
        paired["frozen_error"] = (
            paired["frozen_v1_predicted_ppg"] - paired["actual_ppg"]
        )
        bias = paired.groupby("actual_role_tier", observed=True)[[
            "candidate_error", "frozen_error"
        ]].mean()
        low_bias_not_worse = (
            bias.loc["LOW_LT_2", "candidate_error"]
            <= bias.loc["LOW_LT_2", "frozen_error"]
        )
        featured_bias_not_worse = (
            bias.loc["FEATURED_GE_4", "candidate_error"]
            >= bias.loc["FEATURED_GE_4", "frozen_error"]
        )
        consistency_threshold = math.ceil(row.folds / 2)
        accepted = (
            row.mae_improvement > 0 and row.season_block_95_low > 0
            and row.seasons_improved >= consistency_threshold
            and row.spearman_change >= 0
            and low_bias_not_worse and featured_bias_not_worse
        )
        rows.append({
            "specification": row.specification,
            "decision": "PROMOTE" if accepted else "REJECT_RETAIN_FROZEN_V1",
            "low_tier_bias_not_worse": bool(low_bias_not_worse),
            "featured_tier_bias_not_worse": bool(featured_bias_not_worse),
            "mae_improvement": row.mae_improvement,
            "seasons_improved": row.seasons_improved,
            "season_block_95_low": row.season_block_95_low,
            "spearman_change": row.spearman_change,
        })
    return pd.DataFrame(rows)


def _write_report(
    comparison: pd.DataFrame, decisions: pd.DataFrame, tiers: pd.DataFrame,
    classifier_audit: pd.DataFrame,
) -> None:
    lines = [
        "TE RECEIVING-ADJUSTED COMPETITION EXPERIMENT",
        "=" * 78, "",
        "Frozen V1 remains unchanged. The completed direct receiving-role experiment remains a rejected hypothesis. Role, participation, snap-share, target-rate, and mixture features are not supplied directly to the focal TE PPG model.",
        "The retained classifier is trained only on intrinsic player history. Every season's competitor probabilities are generated by a classifier trained on earlier seasons, then aggregated across other TEs on the preseason-cutoff roster.",
        "No-history competitors remain explicit unknowns rather than being assigned zero receiving probability.", "",
        "Model comparison", "----------------", comparison.to_string(index=False), "",
        "Decisions", "---------", decisions.to_string(index=False), "",
        "Role-tier diagnostics", "---------------------", tiers.to_string(index=False), "",
        "Classifier audit", "----------------", classifier_audit.to_string(index=False), "",
        "Promotion requires lower MAE with a positive season-block interval, improvement in at least half of the available seasons, nonnegative Spearman change, and no worsening of low- or featured-tier directional bias.",
    ]
    TE_COMPETITION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (TE_COMPETITION_REPORTS_DIR / "te_competition_experiment_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_te_competition_experiment() -> dict[str, str]:
    TE_COMPETITION_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    utility = build_role_utility_table()
    probabilities, classifier_audit = crossfit_role_probabilities(utility)
    competition = build_competition_features(probabilities)
    table = build_modeling_table(competition)
    predictions, folds, tuning = run_models(table)
    comparison = compare_to_frozen_v1(predictions, folds)
    tiers = tier_diagnostics(predictions)
    decisions = decision_table(comparison, predictions)
    TE_COMPETITION_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        TE_COMPETITION_PREDICTIONS_DATA_DIR / "te_competition_predictions.parquet",
        index=False,
    )
    folds.to_csv(TE_COMPETITION_REPORT_TABLES_DIR / "fold_metrics.csv", index=False)
    tuning.to_csv(TE_COMPETITION_REPORT_TABLES_DIR / "inner_tuning.csv", index=False)
    comparison.to_csv(
        TE_COMPETITION_REPORT_TABLES_DIR / "model_comparison.csv", index=False
    )
    tiers.to_csv(
        TE_COMPETITION_REPORT_TABLES_DIR / "role_tier_diagnostics.csv", index=False
    )
    decisions.to_csv(
        TE_COMPETITION_REPORT_TABLES_DIR / "decisions.csv", index=False
    )
    (TE_COMPETITION_REPORT_TABLES_DIR / "feature_manifest.json").write_text(
        json.dumps({
            "full_refit_controls": CANDIDATE_MANIFESTS,
            "frozen_v1_residual_adjustments": RESIDUAL_MANIFESTS,
        }, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(comparison, decisions, tiers, classifier_audit)
    return {
        "summary": str(
            TE_COMPETITION_REPORTS_DIR / "te_competition_experiment_summary.txt"
        )
    }


if __name__ == "__main__":
    run_te_competition_experiment()
