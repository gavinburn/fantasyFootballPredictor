"""Build leakage-safe feature groups for experiments 3, 4, and 5."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.attempt3_features import feature_manifests as attempt3_manifests
from src.config import (
    ATTEMPT2_RAW_DATA_DIR,
    ATTEMPT3_PROCESSED_DATA_DIR,
    DATA_DIR,
    FEATURE35_PROCESSED_DATA_DIR,
    FEATURE35_RAW_DATA_DIR,
    FEATURE35_REPORT_TABLES_DIR,
    POSITIONS,
)
from src.download_feature_experiments import FILES
from src.phase2_features import ELIGIBLE_ROSTER_STATUSES, TEAM_ALIASES


class FeatureExperimentError(RuntimeError):
    """Raised when a feature-experiment contract fails."""


def _team(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_ALIASES)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0))


def _read_experiment_raw(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read snapshots while normalizing Arrow dictionary columns for pandas."""

    table = pq.read_table(FEATURE35_RAW_DATA_DIR / FILES[name], columns=columns)
    for index, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(index, field.name, table.column(index).cast(pa.string()))
    return table.to_pandas()


def build_opportunity_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = _read_experiment_raw("opportunity_weekly")
    weekly["season"] = pd.to_numeric(weekly["season"], errors="raise").astype(int)
    weekly["total_fantasy_points_exp"] = pd.to_numeric(
        weekly["total_fantasy_points_exp"], errors="coerce"
    )
    weekly["total_fantasy_points"] = pd.to_numeric(
        weekly["total_fantasy_points"], errors="coerce"
    )
    opportunity = weekly.groupby(["player_id", "season"], as_index=False).agg(
        opportunity_games=("game_id", "nunique"),
        expected_points=("total_fantasy_points_exp", "sum"),
        actual_opportunity_points=("total_fantasy_points", "sum"),
    )
    opportunity["previous_expected_ppg_xfp"] = _ratio(
        opportunity["expected_points"], opportunity["opportunity_games"]
    )
    opportunity["previous_fpoe_per_game"] = _ratio(
        opportunity["actual_opportunity_points"] - opportunity["expected_points"],
        opportunity["opportunity_games"],
    )
    opportunity["prediction_season"] = opportunity.pop("season") + 1
    opportunity = opportunity[[
        "player_id", "prediction_season", "previous_expected_ppg_xfp",
        "previous_fpoe_per_game", "opportunity_games",
    ]]

    passing = _read_experiment_raw("opportunity_pass")
    passing["season"] = pd.to_numeric(passing["season"], errors="raise").astype(int)
    passing["posteam"] = _team(passing["posteam"])
    passing = passing[
        passing["receiver_player_id"].notna()
        & passing["pass_attempt"].fillna(0).eq(1)
        & ~passing["two_point_attempt"].fillna(0).eq(1)
    ].copy()
    passing["redzone_target"] = passing["yardline_100"].le(20).astype(int)
    passing["deep_target"] = passing["air_yards"].ge(20).astype(int)
    player_pass = passing.groupby(
        ["receiver_player_id", "season", "posteam"], as_index=False
    ).agg(
        redzone_targets=("redzone_target", "sum"),
        deep_targets=("deep_target", "sum"),
    )
    team_pass = passing.groupby(["season", "posteam"], as_index=False).agg(
        team_redzone_targets=("redzone_target", "sum"),
        team_deep_targets=("deep_target", "sum"),
    )
    player_pass = player_pass.merge(
        team_pass, on=["season", "posteam"], validate="many_to_one"
    )
    player_pass["previous_redzone_target_share"] = _ratio(
        player_pass["redzone_targets"], player_pass["team_redzone_targets"]
    )
    player_pass["previous_deep_target_share"] = _ratio(
        player_pass["deep_targets"], player_pass["team_deep_targets"]
    )
    player_pass = player_pass.rename(columns={"receiver_player_id": "player_id"})

    rushing = _read_experiment_raw("opportunity_rush")
    rushing["season"] = pd.to_numeric(rushing["season"], errors="raise").astype(int)
    rushing["posteam"] = _team(rushing["posteam"])
    rushing = rushing[
        rushing["rusher_player_id"].notna()
        & rushing["rush_attempt"].fillna(0).eq(1)
        & ~rushing["two_point_attempt"].fillna(0).eq(1)
    ].copy()
    rushing["goal_line_carry"] = rushing["yardline_100"].le(5).astype(int)
    player_rush = rushing.groupby(
        ["rusher_player_id", "season", "posteam"], as_index=False
    ).agg(goal_line_carries=("goal_line_carry", "sum"))
    team_rush = rushing.groupby(["season", "posteam"], as_index=False).agg(
        team_goal_line_carries=("goal_line_carry", "sum")
    )
    player_rush = player_rush.merge(
        team_rush, on=["season", "posteam"], validate="many_to_one"
    )
    player_rush["previous_goal_line_carry_share"] = _ratio(
        player_rush["goal_line_carries"], player_rush["team_goal_line_carries"]
    )
    player_rush = player_rush.rename(columns={"rusher_player_id": "player_id"})

    high_value = player_pass.merge(
        player_rush,
        on=["player_id", "season", "posteam"], how="outer", validate="one_to_one",
    )
    high_value["prediction_season"] = high_value.pop("season") + 1
    high_value = high_value.rename(columns={"posteam": "previous_primary_team"})
    return opportunity, high_value


def build_vacated_features() -> pd.DataFrame:
    weekly = pd.read_parquet(
        DATA_DIR / "raw/player_weekly_stats_2006_2025.parquet",
        columns=[
            "player_id", "season", "season_type", "team", "targets", "carries"
        ],
    )
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    weekly["team"] = _team(weekly["team"])
    for column in ("targets", "carries"):
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce").fillna(0)
    workload = weekly.groupby(["player_id", "season", "team"], as_index=False)[
        ["targets", "carries"]
    ].sum()

    passing = _read_experiment_raw("opportunity_pass")
    passing["season"] = pd.to_numeric(passing["season"], errors="raise").astype(int)
    passing["posteam"] = _team(passing["posteam"])
    passing = passing[
        passing["receiver_player_id"].notna()
        & passing["pass_attempt"].fillna(0).eq(1)
        & ~passing["two_point_attempt"].fillna(0).eq(1)
        & passing["yardline_100"].le(20)
    ]
    rz_targets = passing.groupby(
        ["receiver_player_id", "season", "posteam"], as_index=False
    ).size().rename(columns={
        "receiver_player_id": "player_id", "posteam": "team", "size": "redzone_targets"
    })
    rushing = _read_experiment_raw("opportunity_rush")
    rushing["season"] = pd.to_numeric(rushing["season"], errors="raise").astype(int)
    rushing["posteam"] = _team(rushing["posteam"])
    rushing = rushing[
        rushing["rusher_player_id"].notna()
        & rushing["rush_attempt"].fillna(0).eq(1)
        & ~rushing["two_point_attempt"].fillna(0).eq(1)
        & rushing["yardline_100"].le(20)
    ]
    rz_carries = rushing.groupby(
        ["rusher_player_id", "season", "posteam"], as_index=False
    ).size().rename(columns={
        "rusher_player_id": "player_id", "posteam": "team", "size": "redzone_carries"
    })
    workload = workload.merge(
        rz_targets, on=["player_id", "season", "team"], how="outer", validate="one_to_one"
    ).merge(
        rz_carries, on=["player_id", "season", "team"], how="outer", validate="one_to_one"
    )
    for column in ("targets", "carries", "redzone_targets", "redzone_carries"):
        workload[column] = workload[column].fillna(0.0)
    workload["redzone_touches"] = workload["redzone_targets"] + workload["redzone_carries"]
    workload["prediction_season"] = workload["season"] + 1
    workload = workload.rename(columns={"team": "target_team"})

    rosters = pd.read_parquet(
        ATTEMPT2_RAW_DATA_DIR / "rosters_weekly_2006_2025.parquet",
        columns=["season", "week", "game_type", "team", "gsis_id", "status"],
    )
    rosters = rosters[
        rosters["week"].eq(1)
        & rosters["gsis_id"].notna()
        & rosters["status"].isin(ELIGIBLE_ROSTER_STATUSES)
    ].copy()
    rosters["team"] = _team(rosters["team"])
    retained = rosters[["season", "team", "gsis_id"]].drop_duplicates().rename(
        columns={"season": "prediction_season", "team": "target_team", "gsis_id": "player_id"}
    )
    marked = workload.merge(
        retained.assign(retained_at_cutoff=True),
        on=["prediction_season", "target_team", "player_id"],
        how="left", validate="many_to_one",
    )
    marked["departed"] = ~(
        marked["retained_at_cutoff"].astype("boolean").fillna(False).astype(bool)
    )
    for column in ("targets", "carries", "redzone_touches"):
        marked[f"departed_{column}"] = marked[column] * marked["departed"].astype(int)
    vacated = marked.groupby(
        ["prediction_season", "target_team"], as_index=False
    ).agg(
        previous_team_targets=("targets", "sum"),
        previous_team_carries=("carries", "sum"),
        previous_team_redzone_touches=("redzone_touches", "sum"),
        departed_targets=("departed_targets", "sum"),
        departed_carries=("departed_carries", "sum"),
        departed_redzone_touches=("departed_redzone_touches", "sum"),
        departed_player_count=("departed", "sum"),
    )
    vacated["vacated_target_share"] = _ratio(
        vacated["departed_targets"], vacated["previous_team_targets"]
    )
    vacated["vacated_carry_share"] = _ratio(
        vacated["departed_carries"], vacated["previous_team_carries"]
    )
    vacated["vacated_redzone_touch_share"] = _ratio(
        vacated["departed_redzone_touches"], vacated["previous_team_redzone_touches"]
    )
    return vacated


def build_participation_features() -> pd.DataFrame:
    participation = _read_experiment_raw("participation")
    participation["season"] = pd.to_numeric(
        participation["nflverse_game_id"].str.slice(0, 4), errors="raise"
    ).astype(int)
    participation["possession_team"] = _team(participation["possession_team"])
    passing = _read_experiment_raw(
        "opportunity_pass", columns=[
            "game_id", "play_id", "receiver_player_id", "posteam", "season",
            "pass_attempt", "two_point_attempt",
        ],
    )
    passing["season"] = pd.to_numeric(passing["season"], errors="raise").astype(int)
    passing["posteam"] = _team(passing["posteam"])
    passing = passing[
        passing["pass_attempt"].fillna(0).eq(1)
        & ~passing["two_point_attempt"].fillna(0).eq(1)
    ].copy()
    pass_plays = passing[["game_id", "play_id", "posteam", "season"]].drop_duplicates()
    on_field = participation.merge(
        pass_plays,
        left_on=["nflverse_game_id", "play_id", "possession_team", "season"],
        right_on=["game_id", "play_id", "posteam", "season"],
        how="inner", validate="one_to_one",
    )
    on_field = on_field[on_field["offense_players"].fillna("").ne("")].copy()
    on_field["player_id"] = on_field["offense_players"].str.split(";")
    on_field = on_field.explode("player_id")
    player_counts = on_field.groupby(
        ["player_id", "season", "possession_team"], as_index=False
    ).agg(
        player_pass_plays_on_field=("play_id", "size"),
        games_with_offensive_participation=("nflverse_game_id", "nunique"),
    )
    player_games = on_field[[
        "player_id", "season", "possession_team", "nflverse_game_id"
    ]].drop_duplicates()
    team_game = pass_plays.groupby(
        ["season", "posteam", "game_id"], as_index=False
    ).size().rename(columns={"size": "team_pass_plays_in_game"})
    player_games = player_games.merge(
        team_game,
        left_on=["season", "possession_team", "nflverse_game_id"],
        right_on=["season", "posteam", "game_id"],
        how="left", validate="many_to_one",
    )
    denominators = player_games.groupby(
        ["player_id", "season", "possession_team"], as_index=False
    )["team_pass_plays_in_game"].sum()
    targets = passing[passing["receiver_player_id"].notna()].groupby(
        ["receiver_player_id", "season", "posteam"], as_index=False
    ).size().rename(columns={
        "receiver_player_id": "player_id", "posteam": "possession_team",
        "size": "targets_on_tracked_pass_plays",
    })
    result = player_counts.merge(
        denominators, on=["player_id", "season", "possession_team"], validate="one_to_one"
    ).merge(
        targets, on=["player_id", "season", "possession_team"], how="left", validate="one_to_one"
    )
    result["targets_on_tracked_pass_plays"] = result["targets_on_tracked_pass_plays"].fillna(0)
    result["previous_pass_play_participation_rate"] = _ratio(
        result["player_pass_plays_on_field"], result["team_pass_plays_in_game"]
    )
    result["previous_targets_per_pass_play_participated"] = _ratio(
        result["targets_on_tracked_pass_plays"], result["player_pass_plays_on_field"]
    )
    result["prediction_season"] = result.pop("season") + 1
    return result.rename(columns={"possession_team": "previous_primary_team"})


def _weighted_group(
    frame: pd.DataFrame, values: list[str], weight: str
) -> pd.DataFrame:
    frame = frame.copy()
    frame[weight] = pd.to_numeric(frame[weight], errors="coerce").fillna(0)
    rows = []
    for (player_id, season), group in frame.groupby(["player_gsis_id", "season"]):
        row: dict[str, object] = {"player_id": player_id, "season": int(season)}
        for value in values:
            valid = group[value].notna() & group[weight].gt(0)
            row[value] = (
                np.average(group.loc[valid, value].astype(float), weights=group.loc[valid, weight])
                if valid.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_ngs_features() -> pd.DataFrame:
    passing = _read_experiment_raw("ngs_passing")
    receiving = _read_experiment_raw("ngs_receiving")
    rushing = _read_experiment_raw("ngs_rushing")
    passing = passing[(passing["season_type"] == "REG") & passing["week"].eq(0)]
    receiving = receiving[(receiving["season_type"] == "REG") & receiving["week"].eq(0)]
    rushing = rushing[(rushing["season_type"] == "REG") & rushing["week"].eq(0)]
    pass_features = _weighted_group(
        passing,
        ["completion_percentage_above_expectation", "avg_time_to_throw"],
        "attempts",
    ).rename(columns={
        "completion_percentage_above_expectation": "previous_ngs_cpoe",
        "avg_time_to_throw": "previous_ngs_avg_time_to_throw",
    })
    rec_features = _weighted_group(
        receiving, ["avg_separation", "avg_yac_above_expectation"], "targets"
    ).rename(columns={
        "avg_separation": "previous_ngs_avg_separation",
        "avg_yac_above_expectation": "previous_ngs_yac_over_expected",
    })
    rush_features = _weighted_group(
        rushing,
        ["rush_yards_over_expected_per_att", "avg_time_to_los"],
        "rush_attempts",
    ).rename(columns={
        "rush_yards_over_expected_per_att": "previous_ngs_rush_yoe_per_attempt",
        "avg_time_to_los": "previous_ngs_avg_time_to_los",
    })
    result = pass_features.merge(
        rec_features, on=["player_id", "season"], how="outer", validate="one_to_one"
    ).merge(
        rush_features, on=["player_id", "season"], how="outer", validate="one_to_one"
    )
    result["prediction_season"] = result.pop("season") + 1
    return result


def build_pfr_features() -> pd.DataFrame:
    players = pd.read_parquet(
        DATA_DIR / "raw/players.parquet", columns=["gsis_id", "pfr_id"]
    ).dropna().drop_duplicates("pfr_id")
    id_map = dict(zip(players["pfr_id"], players["gsis_id"], strict=True))
    passing = _read_experiment_raw("pfr_passing")
    rushing = _read_experiment_raw("pfr_rushing")
    receiving = _read_experiment_raw("pfr_receiving")
    for frame in (passing, rushing, receiving):
        frame["player_id"] = frame["pfr_player_id"].map(id_map)
        frame.dropna(subset=["player_id"], inplace=True)
    passing = passing[passing["game_type"] == "REG"]
    pass_group = passing.groupby(["player_id", "season"], as_index=False).agg(
        pfr_sacks=("times_sacked", "sum"), pfr_pressures=("times_pressured", "sum")
    )
    pass_group["previous_pfr_pressure_to_sack_rate"] = _ratio(
        pass_group["pfr_sacks"], pass_group["pfr_pressures"]
    )
    rushing = rushing[rushing["game_type"] == "REG"]
    rush_group = rushing.groupby(["player_id", "season"], as_index=False).agg(
        pfr_carries=("carries", "sum"),
        pfr_yards_after_contact=("rushing_yards_after_contact", "sum"),
        pfr_rush_broken=("rushing_broken_tackles", "sum"),
        pfr_rec_broken_from_rush=("receiving_broken_tackles", "sum"),
    )
    rush_group["previous_pfr_yards_after_contact_per_attempt"] = _ratio(
        rush_group["pfr_yards_after_contact"], rush_group["pfr_carries"]
    )
    receiving = receiving[receiving["game_type"] == "REG"]
    rec_group = receiving.groupby(["player_id", "season"], as_index=False).agg(
        pfr_rush_broken_rec_source=("rushing_broken_tackles", "sum"),
        pfr_rec_broken=("receiving_broken_tackles", "sum"),
    )
    weekly = pd.read_parquet(
        DATA_DIR / "raw/player_weekly_stats_2006_2025.parquet",
        columns=["player_id", "season", "season_type", "carries", "receptions"],
    )
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    touches = weekly.groupby(["player_id", "season"], as_index=False)[
        ["carries", "receptions"]
    ].sum()
    result = pass_group.merge(
        rush_group, on=["player_id", "season"], how="outer", validate="one_to_one"
    ).merge(
        rec_group, on=["player_id", "season"], how="outer", validate="one_to_one"
    ).merge(
        touches, on=["player_id", "season"], how="left", validate="one_to_one"
    )
    # The rush and receiving PFR feeds repeat the same broken-tackle categories.
    # Prefer the rush feed where present, otherwise use the receiving feed.
    rush_source = result["pfr_rush_broken"] + result["pfr_rec_broken_from_rush"]
    rec_source = result["pfr_rush_broken_rec_source"] + result["pfr_rec_broken"]
    result["pfr_broken_tackles"] = rush_source.combine_first(rec_source)
    result["previous_pfr_broken_tackle_per_touch"] = _ratio(
        result["pfr_broken_tackles"], result["carries"] + result["receptions"]
    )
    result["prediction_season"] = result.pop("season") + 1
    return result


GROUP_FEATURES = {
    "exp3_xfp": {
        "QB": [],
        "RB": ["previous_expected_ppg_xfp", "previous_fpoe_per_game"],
        "WR": ["previous_expected_ppg_xfp", "previous_fpoe_per_game"],
        "TE": ["previous_expected_ppg_xfp", "previous_fpoe_per_game"],
    },
    "exp3_high_value": {
        "QB": ["previous_goal_line_carry_share"],
        "RB": [
            "previous_goal_line_carry_share", "previous_redzone_target_share",
            "previous_deep_target_share",
        ],
        "WR": ["previous_redzone_target_share", "previous_deep_target_share"],
        "TE": ["previous_redzone_target_share", "previous_deep_target_share"],
    },
    "exp4_vacated": {
        "QB": [],
        "RB": ["vacated_target_share", "vacated_carry_share", "vacated_redzone_touch_share"],
        "WR": ["vacated_target_share", "vacated_carry_share", "vacated_redzone_touch_share"],
        "TE": ["vacated_target_share", "vacated_carry_share", "vacated_redzone_touch_share"],
    },
    "exp5_participation": {
        "QB": [],
        "RB": ["previous_pass_play_participation_rate", "previous_targets_per_pass_play_participated", "participation_feature_available"],
        "WR": ["previous_pass_play_participation_rate", "previous_targets_per_pass_play_participated", "participation_feature_available"],
        "TE": ["previous_pass_play_participation_rate", "previous_targets_per_pass_play_participated", "participation_feature_available"],
    },
    "exp5_ngs": {
        "QB": ["previous_ngs_cpoe", "previous_ngs_avg_time_to_throw", "ngs_feature_available"],
        "RB": [
            "previous_ngs_rush_yoe_per_attempt", "previous_ngs_avg_time_to_los",
            "ngs_feature_available",
        ],
        "WR": ["previous_ngs_avg_separation", "previous_ngs_yac_over_expected", "ngs_feature_available"],
        "TE": ["previous_ngs_avg_separation", "previous_ngs_yac_over_expected", "ngs_feature_available"],
    },
    "exp5_pfr": {
        "QB": ["previous_pfr_pressure_to_sack_rate", "pfr_feature_available"],
        "RB": [
            "previous_pfr_yards_after_contact_per_attempt",
            "previous_pfr_broken_tackle_per_touch", "pfr_feature_available",
        ],
        "WR": ["previous_pfr_broken_tackle_per_touch", "pfr_feature_available"],
        "TE": [],
    },
}


def candidate_manifests() -> dict[str, dict[str, list[str]]]:
    base = attempt3_manifests()
    output: dict[str, dict[str, list[str]]] = {}
    for position in POSITIONS:
        foundation = list(base[position]["STRUCTURAL_CLEANUP"])
        xfp = GROUP_FEATURES["exp3_xfp"][position]
        high = GROUP_FEATURES["exp3_high_value"][position]
        vacated = GROUP_FEATURES["exp4_vacated"][position]
        specs: dict[str, list[str]] = {}
        if xfp:
            xfp_base = [f for f in foundation if f != "previous_season_ppg"]
            specs["EXP3_XFP_REPRESENTATION"] = [*xfp_base, *xfp]
            specs["EXP3_OPPORTUNITY_COMBINED"] = [*xfp_base, *xfp, *high]
        elif high:
            specs["EXP3_HIGH_VALUE"] = [*foundation, *high]
        if high and xfp:
            specs["EXP3_HIGH_VALUE"] = [*foundation, *high]
        if vacated:
            specs["EXP4_VACATED_WORKLOAD"] = [*foundation, *vacated]
            combined_base = [f for f in foundation if not xfp or f != "previous_season_ppg"]
            specs["EXP3_4_COMBINED"] = [*combined_base, *xfp, *high, *vacated]
        participation = GROUP_FEATURES["exp5_participation"][position]
        ngs = GROUP_FEATURES["exp5_ngs"][position]
        pfr = GROUP_FEATURES["exp5_pfr"][position]
        if participation:
            specs["EXP5_PARTICIPATION"] = [*foundation, *participation]
        if ngs:
            specs["EXP5_NGS"] = [*foundation, *ngs]
        if pfr:
            specs["EXP5_PFR"] = [*foundation, *pfr]
        advanced = [*participation, *ngs, *pfr]
        if len([group for group in (participation, ngs, pfr) if group]) >= 2:
            specs["EXP5_ADVANCED_COMBINED"] = [*foundation, *advanced]
        if position == "WR":
            rate = "previous_pass_play_participation_rate"
            targets_per = "previous_targets_per_pass_play_participated"
            participation_flag = "participation_feature_available"
            pfr_metric = "previous_pfr_broken_tackle_per_touch"
            pfr_flag = "pfr_feature_available"
            specs.update({
                "EXP5_PARTICIPATION_RATE_ONLY": [*foundation, rate],
                "EXP5_TARGETS_PER_PARTICIPATION_ONLY": [*foundation, targets_per],
                "EXP5_PARTICIPATION_NO_FLAG": [*foundation, rate, targets_per],
                "EXP5_PARTICIPATION_AVAILABILITY_ONLY": [*foundation, participation_flag],
                "EXP5_PFR_METRIC_ONLY": [*foundation, pfr_metric],
                "EXP5_PFR_AVAILABILITY_ONLY": [*foundation, pfr_flag],
            })
        if position == "TE":
            v1 = list(base[position]["VETERAN_ALIGNED_BASE"])
            v1_xfp = [feature for feature in v1 if feature != "previous_season_ppg"]
            specs.update({
                "EXP3_TE_V1_DEEP_TARGET": [*v1, "previous_deep_target_share"],
                "EXP3_TE_V1_HIGH_VALUE": [
                    *v1, "previous_redzone_target_share", "previous_deep_target_share"
                ],
                "EXP3_TE_V1_OPPORTUNITY": [
                    *v1_xfp, "previous_expected_ppg_xfp", "previous_fpoe_per_game",
                    "previous_redzone_target_share", "previous_deep_target_share",
                ],
                "EXP5_NGS_YAC_ONLY": [*foundation, "previous_ngs_yac_over_expected"],
                "EXP5_NGS_SEPARATION_ONLY": [*foundation, "previous_ngs_avg_separation"],
                "EXP5_NGS_NO_FLAG": [
                    *foundation, "previous_ngs_avg_separation",
                    "previous_ngs_yac_over_expected",
                ],
                "EXP5_NGS_AVAILABILITY_ONLY": [*foundation, "ngs_feature_available"],
            })
        for name, features in specs.items():
            if len(features) != len(set(features)):
                raise FeatureExperimentError(f"Duplicate features in {position} {name}")
        output[position] = specs
    return output


def build_feature_experiment_tables() -> dict[str, pd.DataFrame]:
    opportunity, high_value = build_opportunity_features()
    vacated = build_vacated_features()
    participation = build_participation_features()
    ngs = build_ngs_features()
    pfr = build_pfr_features()
    outputs = {}
    coverage_rows = []
    for position in POSITIONS:
        table = pd.read_parquet(
            ATTEMPT3_PROCESSED_DATA_DIR / f"{position.lower()}_attempt3_modeling_dataset.parquet"
        )
        table = table.merge(
            opportunity, on=["player_id", "prediction_season"], how="left", validate="one_to_one"
        ).merge(
            high_value[[
                "player_id", "prediction_season", "previous_primary_team",
                "previous_redzone_target_share", "previous_deep_target_share",
                "previous_goal_line_carry_share",
            ]],
            on=["player_id", "prediction_season", "previous_primary_team"],
            how="left", validate="one_to_one",
        ).merge(
            vacated[[
                "prediction_season", "target_team", "vacated_target_share",
                "vacated_carry_share", "vacated_redzone_touch_share",
            ]],
            on=["prediction_season", "target_team"], how="left", validate="many_to_one",
        ).merge(
            participation[[
                "player_id", "prediction_season", "previous_primary_team",
                "previous_pass_play_participation_rate",
                "previous_targets_per_pass_play_participated",
            ]],
            on=["player_id", "prediction_season", "previous_primary_team"],
            how="left", validate="one_to_one",
        ).merge(
            ngs, on=["player_id", "prediction_season"], how="left", validate="one_to_one",
        ).merge(
            pfr[[
                "player_id", "prediction_season", "previous_pfr_pressure_to_sack_rate",
                "previous_pfr_yards_after_contact_per_attempt",
                "previous_pfr_broken_tackle_per_touch",
            ]],
            on=["player_id", "prediction_season"], how="left", validate="one_to_one",
        )
        high_value_columns = [
            "previous_redzone_target_share", "previous_deep_target_share",
            "previous_goal_line_carry_share",
        ]
        table[high_value_columns] = table[high_value_columns].fillna(0.0)
        availability_sources = {
            "participation_feature_available": [
                "previous_pass_play_participation_rate",
                "previous_targets_per_pass_play_participated",
            ],
            "ngs_feature_available": [
                feature for feature in GROUP_FEATURES["exp5_ngs"][position]
                if feature != "ngs_feature_available"
            ],
            "pfr_feature_available": [
                feature for feature in GROUP_FEATURES["exp5_pfr"][position]
                if feature != "pfr_feature_available"
            ],
        }
        for flag, source_columns in availability_sources.items():
            table[flag] = (
                table[source_columns].notna().any(axis=1).astype(int)
                if source_columns else 0
            )
        if table.duplicated(["player_id", "prediction_season"]).any():
            raise FeatureExperimentError(f"Duplicate {position} rows after feature joins")
        outputs[position] = table.sort_values(
            ["prediction_season", "player_id"]
        ).reset_index(drop=True)
        for group, by_position in GROUP_FEATURES.items():
            features = by_position[position]
            if not features:
                continue
            for season, rows in table.groupby("prediction_season"):
                coverage_rows.append({
                    "position": position, "feature_group": group,
                    "prediction_season": int(season), "eligible_rows": len(rows),
                    "complete_rows": int(rows[features].notna().all(axis=1).sum()),
                    "any_available_rows": int(rows[features].notna().any(axis=1).sum()),
                    "any_available_rate": float(rows[features].notna().any(axis=1).mean()),
                })

    FEATURE35_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE35_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for position, table in outputs.items():
        table.to_parquet(
            FEATURE35_PROCESSED_DATA_DIR / f"{position.lower()}_feature_experiments_dataset.parquet",
            index=False,
        )
    pd.DataFrame(coverage_rows).to_csv(
        FEATURE35_REPORT_TABLES_DIR / "feature_coverage_by_season.csv", index=False
    )
    (FEATURE35_REPORT_TABLES_DIR / "candidate_feature_sets.json").write_text(
        json.dumps(candidate_manifests(), indent=2) + "\n", encoding="utf-8"
    )
    omissions = {
        "inline_blocking_share": "not present in snap-count or participation schemas",
        "route_participation_rate": (
            "participation route labels identify only the targeted route; replaced with "
            "accurately named pass-play participation"
        ),
        "targets_per_route_run": (
            "routes run are unavailable; replaced with targets per pass play participated"
        ),
    }
    (FEATURE35_REPORT_TABLES_DIR / "schema_omissions.json").write_text(
        json.dumps(omissions, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


if __name__ == "__main__":
    build_feature_experiment_tables()
