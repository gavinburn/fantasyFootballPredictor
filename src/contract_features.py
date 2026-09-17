"""Build leakage-conservative contract-capital and roster-room features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    ATTEMPT2_INTERIM_DATA_DIR,
    CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR,
    CONTRACT_EXPERIMENT_RAW_DATA_DIR,
    CONTRACT_EXPERIMENT_REPORT_TABLES_DIR,
    FEATURE35_PROCESSED_DATA_DIR,
    POSITIONS,
    TE_COMPETITION_PROCESSED_DATA_DIR,
)
from src.download_contract_data import CONTRACT_FILE

KEYS = ["player_id", "prediction_season"]
LAGGED_CAP_FEATURES = [
    "previous_salary_data_available",
    "previous_season_cap_percent",
    "previous_season_log_cap_number",
    "previous_season_log_cash_paid",
    "previous_season_position_cap_percentile",
]
FOCAL_CONTRACT_FEATURES = [
    "contract_available",
    "contract_apy_cap_pct_at_signing",
    "contract_log_inflation_adjusted_apy",
    "contract_guaranteed_value_ratio",
    "contract_years_remaining_entering_season",
    "contract_is_rookie_deal",
    "contract_position_salary_percentile",
]
ROOM_CONTRACT_FEATURES = [
    "position_room_known_contract_count",
    "position_room_contract_apy_cap_pct_total",
    "other_position_room_contract_apy_cap_pct_total",
    "focal_position_room_contract_share",
    "focal_position_room_salary_rank",
    "max_other_position_contract_apy_cap_pct",
    "focal_to_max_other_contract_apy_ratio",
]
TE_WEIGHTED_CONTRACT_FEATURES = [
    "te_role_weighted_other_contract_apy_cap_pct",
    "te_max_role_weighted_other_contract_apy_cap_pct",
    "te_known_receiving_salary_competitor_count",
    "te_unknown_receiving_salary_competitor_count",
]


class ContractFeatureError(RuntimeError):
    """Raised when contract feature point-in-time or join contracts fail."""


def _load_context() -> pd.DataFrame:
    context = pd.read_parquet(
        ATTEMPT2_INTERIM_DATA_DIR / "prediction_context.parquet"
    )
    context = context[
        context["roster_position"].isin(POSITIONS)
        & context["prediction_season"].between(2007, 2025)
    ][[
        "player_id", "prediction_season", "roster_position", "target_team",
    ]].copy()
    context = context.rename(columns={"roster_position": "position"})
    if context.duplicated(KEYS).any():
        raise ContractFeatureError("Preseason context contains duplicate player-seasons")
    return context


def _previous_season_cap_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract realized t-1 cap information from the source's annual history."""

    annual = (
        pl.read_parquet(
            CONTRACT_EXPERIMENT_RAW_DATA_DIR / CONTRACT_FILE,
            columns=["gsis_id", "cols"],
        )
        .filter(pl.col("gsis_id").is_not_null())
        .explode("cols")
        .unnest("cols")
        .with_columns(pl.col("year").cast(pl.Int32, strict=False).alias("season"))
        .filter(pl.col("season").is_between(2006, 2024))
        .group_by(["gsis_id", "season"])
        .agg(
            pl.col("cap_percent").drop_nulls().first().alias("cap_percent"),
            pl.col("cap_percent").drop_nulls().n_unique().alias("cap_percent_values"),
            pl.col("cap_number").drop_nulls().first().alias("cap_number"),
            pl.col("cap_number").drop_nulls().n_unique().alias("cap_number_values"),
            pl.col("cash_paid").drop_nulls().first().alias("cash_paid"),
            pl.col("cash_paid").drop_nulls().n_unique().alias("cash_paid_values"),
        )
    )
    conflicts = annual.filter(
        (pl.col("cap_percent_values") > 1)
        | (pl.col("cap_number_values") > 1)
        | (pl.col("cash_paid_values") > 1)
    )
    # Ambiguous duplicated annual histories are rare and are nulled instead of
    # guessing which retrospectively stored value was known at the cutoff.
    annual = annual.with_columns(
        pl.when(pl.col("cap_percent_values") <= 1)
        .then(pl.col("cap_percent"))
        .otherwise(None)
        .alias("previous_season_cap_percent"),
        pl.when(pl.col("cap_number_values") <= 1)
        .then(pl.col("cap_number"))
        .otherwise(None)
        .alias("previous_season_cap_number"),
        pl.when(pl.col("cash_paid_values") <= 1)
        .then(pl.col("cash_paid"))
        .otherwise(None)
        .alias("previous_season_cash_paid"),
        (pl.col("season") + 1).alias("prediction_season"),
    ).select(
        pl.col("gsis_id").alias("player_id"),
        "prediction_season",
        "previous_season_cap_percent",
        "previous_season_cap_number",
        "previous_season_cash_paid",
    )
    return annual.to_pandas(), conflicts.to_pandas()


def _select_prior_contracts(
    context: pd.DataFrame, contracts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "gsis_id", "year_signed", "years", "value", "apy", "guaranteed",
        "apy_cap_pct", "inflated_apy", "draft_year", "team", "position",
    ]
    source = contracts[columns].rename(columns={
        "gsis_id": "player_id",
        "team": "contract_team",
        "position": "contract_position",
    }).copy()
    numeric = [
        "year_signed", "years", "value", "apy", "guaranteed", "apy_cap_pct",
        "inflated_apy", "draft_year",
    ]
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source[source["player_id"].notna() & source["year_signed"].notna()].copy()
    source["contract_end_season"] = source["year_signed"] + source["years"] - 1

    joined = context.merge(source, on="player_id", how="left", validate="many_to_many")
    eligible = joined[
        joined["year_signed"].le(joined["prediction_season"] - 1)
        & joined["contract_end_season"].ge(joined["prediction_season"])
    ].copy()
    # Extensions can overlap an older deal. The newest contract known by t-1 is
    # the relevant commitment; duplicate amendments in that year use the largest APY.
    eligible = eligible.sort_values(
        [*KEYS, "year_signed", "apy", "guaranteed"],
        ascending=[True, True, False, False, False],
        kind="stable",
    ).drop_duplicates(KEYS, keep="first")
    if (eligible["year_signed"] >= eligible["prediction_season"]).any():
        raise ContractFeatureError("A same-season contract entered the feature table")

    same_year = joined[
        joined["year_signed"].eq(joined["prediction_season"])
    ].groupby(["prediction_season", "position"], as_index=False).agg(
        same_season_contract_rows_excluded=("player_id", "nunique")
    )
    return eligible, same_year


def _focal_features(
    context: pd.DataFrame, selected: pd.DataFrame, annual: pd.DataFrame
) -> pd.DataFrame:
    keep = [
        *KEYS, "year_signed", "years", "contract_end_season", "value", "apy",
        "guaranteed", "apy_cap_pct", "inflated_apy", "draft_year",
        "contract_team", "contract_position",
    ]
    output = context.merge(selected[keep], on=KEYS, how="left", validate="one_to_one")
    output = output.merge(annual, on=KEYS, how="left", validate="one_to_one")
    output["previous_salary_data_available"] = output[
        "previous_season_cap_percent"
    ].notna().astype(int)
    output["previous_season_log_cap_number"] = np.log1p(
        output["previous_season_cap_number"].clip(lower=0)
    )
    output["previous_season_log_cash_paid"] = np.log1p(
        output["previous_season_cash_paid"].clip(lower=0)
    )
    output["previous_season_position_cap_percentile"] = output.groupby(
        ["prediction_season", "position"], observed=True
    )["previous_season_cap_percent"].rank(method="average", pct=True)
    output["contract_available"] = output["year_signed"].notna().astype(int)
    output["contract_apy_cap_pct_at_signing"] = output["apy_cap_pct"]
    output["contract_log_inflation_adjusted_apy"] = np.log1p(
        output["inflated_apy"].clip(lower=0)
    )
    output["contract_guaranteed_value_ratio"] = output["guaranteed"].div(
        output["value"].where(output["value"] > 0)
    ).clip(lower=0, upper=1)
    output["contract_years_remaining_entering_season"] = (
        output["contract_end_season"] - output["prediction_season"] + 1
    )
    output["contract_is_rookie_deal"] = (
        output["year_signed"].eq(output["draft_year"])
        & output["year_signed"].notna()
    ).astype(float)
    output.loc[output["year_signed"].isna(), "contract_is_rookie_deal"] = np.nan
    output["contract_position_salary_percentile"] = output.groupby(
        ["prediction_season", "position"], observed=True
    )["contract_apy_cap_pct_at_signing"].rank(method="average", pct=True)
    return output


def _room_features(focal: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["prediction_season", "target_team", "position"]
    salary = "contract_apy_cap_pct_at_signing"
    output = focal.copy()
    grouped = output.groupby(group_keys, observed=True)[salary]
    output["position_room_known_contract_count"] = grouped.transform("count")
    output["position_room_contract_apy_cap_pct_total"] = grouped.transform(
        lambda values: values.sum(min_count=1)
    )
    output["other_position_room_contract_apy_cap_pct_total"] = (
        output["position_room_contract_apy_cap_pct_total"] - output[salary].fillna(0)
    )
    output["focal_position_room_contract_share"] = output[salary].div(
        output["position_room_contract_apy_cap_pct_total"].where(
            output["position_room_contract_apy_cap_pct_total"] > 0
        )
    )
    output["focal_position_room_salary_rank"] = output.groupby(
        group_keys, observed=True
    )[salary].rank(method="min", ascending=False)

    competitors = focal[[*KEYS, *group_keys[1:], salary]].merge(
        focal[["player_id", *group_keys, salary]].rename(columns={
            "player_id": "competitor_id", salary: "competitor_contract_apy_cap_pct",
        }),
        on=group_keys,
        how="left",
        validate="many_to_many",
    )
    competitors = competitors[competitors["player_id"] != competitors["competitor_id"]]
    other_max = competitors.groupby(KEYS, as_index=False).agg(
        max_other_position_contract_apy_cap_pct=(
            "competitor_contract_apy_cap_pct", "max"
        )
    )
    output = output.merge(other_max, on=KEYS, how="left", validate="one_to_one")
    output["focal_to_max_other_contract_apy_ratio"] = output[salary].div(
        output["max_other_position_contract_apy_cap_pct"].where(
            output["max_other_position_contract_apy_cap_pct"] > 0
        )
    )
    return output


def _te_weighted_competition(focal: pd.DataFrame) -> pd.DataFrame:
    te = focal[focal["position"] == "TE"][[
        *KEYS, "target_team", "contract_apy_cap_pct_at_signing"
    ]].copy()
    probabilities = pd.read_parquet(
        TE_COMPETITION_PROCESSED_DATA_DIR / "crossfit_role_probabilities.parquet",
        columns=[
            "player_id", "prediction_season",
            "competitor_receiving_role_probability",
        ],
    )
    competitors = te.merge(
        te.rename(columns={
            "player_id": "competitor_id",
            "contract_apy_cap_pct_at_signing": "competitor_contract_apy_cap_pct",
        }),
        on=["prediction_season", "target_team"],
        how="left",
        validate="many_to_many",
    )
    competitors = competitors[competitors["player_id"] != competitors["competitor_id"]]
    probabilities = probabilities.rename(columns={"player_id": "competitor_id"})
    competitors = competitors.merge(
        probabilities,
        on=["competitor_id", "prediction_season"],
        how="left",
        validate="many_to_one",
    )
    probability = competitors["competitor_receiving_role_probability"]
    pay = competitors["competitor_contract_apy_cap_pct"]
    competitors["weighted_contract"] = probability * pay
    competitors["known_receiving_salary"] = (probability.notna() & pay.notna()).astype(int)
    competitors["unknown_receiving_salary"] = (probability.isna() & pay.notna()).astype(int)

    def _sum_or_nan(values: pd.Series) -> float:
        return float(values.sum(min_count=1))

    result = competitors.groupby(KEYS, as_index=False).agg(
        te_role_weighted_other_contract_apy_cap_pct=("weighted_contract", _sum_or_nan),
        te_max_role_weighted_other_contract_apy_cap_pct=("weighted_contract", "max"),
        te_known_receiving_salary_competitor_count=("known_receiving_salary", "sum"),
        te_unknown_receiving_salary_competitor_count=("unknown_receiving_salary", "sum"),
    )
    return result


def build_contract_features() -> dict[str, pd.DataFrame]:
    """Create contract features and merge them into untouched modeling cohorts."""

    CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_EXPERIMENT_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    context = _load_context()
    contracts = pd.read_parquet(CONTRACT_EXPERIMENT_RAW_DATA_DIR / CONTRACT_FILE)
    if "is_active" in contracts:
        raise ContractFeatureError("Current contract status must not enter this experiment")
    selected, exclusions = _select_prior_contracts(context, contracts)
    annual, annual_conflicts = _previous_season_cap_features()
    features = _room_features(_focal_features(context, selected, annual))
    features = features.merge(
        _te_weighted_competition(features), on=KEYS, how="left", validate="one_to_one"
    )
    feature_columns = [
        *LAGGED_CAP_FEATURES, *FOCAL_CONTRACT_FEATURES, *ROOM_CONTRACT_FEATURES,
        *TE_WEIGHTED_CONTRACT_FEATURES,
    ]
    if features.duplicated(KEYS).any():
        raise ContractFeatureError("Contract feature rows are not unique")
    if (features["year_signed"].dropna() >= features.loc[
        features["year_signed"].notna(), "prediction_season"
    ]).any():
        raise ContractFeatureError("Contract cutoff assertion failed")
    features[[*KEYS, "position", "target_team", "year_signed", *feature_columns]].to_parquet(
        CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR / "contract_context_features.parquet",
        index=False,
    )

    coverage = features.groupby(
        ["position", "prediction_season"], as_index=False, observed=True
    ).agg(
        roster_rows=("player_id", "size"),
        matched_contracts=("contract_available", "sum"),
        mean_apy_cap_pct=("contract_apy_cap_pct_at_signing", "mean"),
    )
    coverage["contract_match_rate"] = coverage["matched_contracts"] / coverage["roster_rows"]
    coverage.to_csv(
        CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "contract_coverage_by_season.csv",
        index=False,
    )
    exclusions.to_csv(
        CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "same_season_contract_exclusions.csv",
        index=False,
    )
    annual_conflicts.to_csv(
        CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "annual_cap_history_conflicts.csv",
        index=False,
    )

    outputs: dict[str, pd.DataFrame] = {}
    compact = features[[*KEYS, *feature_columns]]
    for position in POSITIONS:
        base = pd.read_parquet(
            FEATURE35_PROCESSED_DATA_DIR
            / f"{position.lower()}_feature_experiments_dataset.parquet"
        )
        table = base.merge(compact, on=KEYS, how="left", validate="one_to_one")
        if len(table) != len(base):
            raise ContractFeatureError(f"{position} cohort changed during contract merge")
        path = (
            CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR
            / f"{position.lower()}_contract_modeling_dataset.parquet"
        )
        table.to_parquet(path, index=False)
        outputs[position] = table

    (CONTRACT_EXPERIMENT_REPORT_TABLES_DIR / "feature_dictionary.json").write_text(
        json.dumps({
            "lagged_realized_cap": LAGGED_CAP_FEATURES,
            "focal_contract": FOCAL_CONTRACT_FEATURES,
            "position_room_contract": ROOM_CONTRACT_FEATURES,
            "te_receiving_weighted_contract_competition": TE_WEIGHTED_CONTRACT_FEATURES,
            "cutoff": "year_signed <= prediction_season - 1",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


if __name__ == "__main__":
    build_contract_features()
