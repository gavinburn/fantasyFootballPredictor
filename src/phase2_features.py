"""Build leakage-safe Attempt 2 team, roster, role, QB, OL, and SOS features."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.build_features import V1_FEATURES
from src.config import (
    ATTEMPT2_INTERIM_DATA_DIR,
    ATTEMPT2_PROCESSED_DATA_DIR,
    ATTEMPT2_RAW_DATA_DIR,
    ATTEMPT2_REPORT_TABLES_DIR,
    INTERIM_DATA_DIR,
    POSITIONS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.download_data import PLAYER_STATS_FILENAME, PLAYERS_FILENAME
from src.download_phase2_data import (
    DEPTH_CHARTS_2025_FILENAME,
    DEPTH_CHARTS_FILENAME,
    QBR_FILENAME,
    RANKINGS_FILENAME,
    ROSTERS_WEEKLY_FILENAME,
    TEAM_WEEKLY_FILENAME,
)

CONTEXT_FILENAME = "prediction_context.parquet"
TEAM_ENV_FILENAME = "target_team_environment.parquet"
WORKLOAD_FILENAME = "player_workload_shares.parquet"
COMPETITION_LONG_FILENAME = "competition_long.parquet"
COMPETITION_MODEL_FILENAME = "competition_model_features.parquet"
QB_ENV_FILENAME = "projected_qb_environment.parquet"
EXCLUSIONS_FILENAME = "attempt2_context_exclusions.csv"
AUDIT_FILENAME = "phase2_feature_audit.txt"

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "LA": "LAR",
    "JAC": "JAX",
    "WSH": "WAS",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "LAR",
}
ELIGIBLE_ROSTER_STATUSES = {"ACT", "INA", "RES", "SUS", "EXE"}
PROJECTED_QB_OVERRIDES = [
    # The historical week-1 file omits or mis-ranks these known opening-week depth
    # charts. These overrides use preseason starter information, not game results.
    {"prediction_season": 2013, "target_team": "LV", "projected_starting_qb_id": "00-0028825"},
    {"prediction_season": 2017, "target_team": "LAR", "projected_starting_qb_id": "00-0033106"},
    {"prediction_season": 2017, "target_team": "MIA", "projected_starting_qb_id": "00-0024226"},
    {"prediction_season": 2017, "target_team": "TB", "projected_starting_qb_id": "00-0031503"},
    {"prediction_season": 2018, "target_team": "LAR", "projected_starting_qb_id": "00-0033106"},
    {"prediction_season": 2018, "target_team": "TB", "projected_starting_qb_id": "00-0023682"},
]

GROUP_FEATURES: dict[str, dict[str, list[str]]] = {
    "volume": {
        "QB": ["target_team_dropbacks_per_game_tminus1", "target_team_pass_rate_tminus1"],
        "RB": ["target_team_carries_per_game_tminus1", "target_team_plays_per_game_tminus1"],
        "WR": ["target_team_dropbacks_per_game_tminus1", "target_team_pass_rate_tminus1"],
        "TE": ["target_team_dropbacks_per_game_tminus1", "target_team_pass_rate_tminus1"],
    },
    "efficiency": {
        "QB": ["target_team_passing_tds_per_game_tminus1"],
        "RB": ["target_team_rushing_tds_per_game_tminus1", "target_team_rush_epa_per_carry_tminus1"],
        "WR": ["target_team_passing_tds_per_game_tminus1", "target_team_pass_epa_per_dropback_tminus1"],
        "TE": ["target_team_passing_tds_per_game_tminus1", "target_team_pass_epa_per_dropback_tminus1"],
    },
    "workload": {
        "QB": ["previous_qb_dropback_share"],
        "RB": ["previous_carry_share", "previous_target_share", "previous_opportunity_share"],
        "WR": ["previous_target_share", "previous_air_yards_share"],
        "TE": ["previous_target_share", "previous_air_yards_share"],
    },
    "competition": {
        "QB": ["top_competitor_previous_ppg", "top_competitor_age", "competition_pool_size", "competition_no_history_count"],
        "RB": ["top_competitor_previous_total_points", "top_competitor_previous_ppg", "top_competitor_age", "second_competitor_previous_ppg", "competition_pool_size", "competition_no_history_count"],
        "WR": ["top_competitor_previous_total_points", "top_competitor_previous_ppg", "top_competitor_age", "top_competitor_is_wr", "top_competitor_is_te", "second_competitor_previous_ppg", "second_competitor_is_wr", "second_competitor_is_te", "competition_pool_size", "competition_no_history_count"],
        "TE": ["top_competitor_previous_total_points", "top_competitor_previous_ppg", "top_competitor_age", "top_competitor_is_wr", "top_competitor_is_te", "second_competitor_previous_ppg", "second_competitor_is_wr", "second_competitor_is_te", "competition_pool_size", "competition_no_history_count"],
    },
    "quarterback": {
        position: ["projected_qb_previous_qbr", "projected_qb_previous_qb_plays", "projected_qb_has_previous_qbr"]
        for position in POSITIONS
    },
    "team_change": {position: ["changed_team"] for position in POSITIONS},
    "offensive_line": {position: ["preseason_offensive_line_score"] for position in POSITIONS},
    "schedule": {position: ["preseason_strength_of_schedule_score"] for position in POSITIONS},
}


class Phase2FeatureError(RuntimeError):
    """Raised when Phase 2 feature construction violates a data contract."""


def normalize_team(column: str) -> pl.Expr:
    return pl.col(column).replace(TEAM_ALIASES).alias(column)


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr, alias: str) -> pl.Expr:
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None).alias(alias)


def _write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.write_parquet(temporary, use_pyarrow=True)
        if pl.read_parquet(temporary).shape != frame.shape:
            raise Phase2FeatureError(f"Read-back validation failed for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_roster_snapshot(rosters: pl.DataFrame) -> pl.DataFrame:
    """Return one eligible earliest-week roster assignment per player-season."""

    snapshot = (
        rosters.filter(
            (pl.col("week") == 1)
            & pl.col("gsis_id").is_not_null()
            & pl.col("position").is_in(POSITIONS)
            & pl.col("status").is_in(ELIGIBLE_ROSTER_STATUSES)
        )
        .with_columns(normalize_team("team"))
        .with_columns(
            pl.when(pl.col("status") == "ACT").then(0)
            .when(pl.col("status") == "INA").then(1)
            .otherwise(2)
            .alias("_status_priority")
        )
        .sort(["season", "gsis_id", "_status_priority", "team"])
        .unique(["season", "gsis_id"], keep="first", maintain_order=True)
        .select(
            pl.col("season").alias("prediction_season"),
            pl.col("gsis_id").alias("player_id"),
            pl.col("team").alias("target_team"),
            pl.col("position").alias("roster_position"),
            "status",
            "birth_date",
            "espn_id",
        )
    )
    if snapshot.select(pl.struct("prediction_season", "player_id").is_duplicated().sum()).item():
        raise Phase2FeatureError("Roster snapshot contains duplicate player-season keys")
    return snapshot


def build_previous_primary_team(player_weekly: pl.DataFrame) -> pl.DataFrame:
    weekly = player_weekly.filter(pl.col("season_type") == "REG").with_columns(normalize_team("team"))
    return (
        weekly.group_by("player_id", "season", "team")
        .agg(
            pl.len().alias("appearances"),
            (pl.col("attempts").fill_null(0) + pl.col("carries").fill_null(0) + pl.col("targets").fill_null(0)).sum().alias("offensive_opportunities"),
            pl.col("week").max().alias("latest_week"),
        )
        .sort(
            ["player_id", "season", "appearances", "offensive_opportunities", "latest_week", "team"],
            descending=[False, False, True, True, True, False],
        )
        .unique(["player_id", "season"], keep="first", maintain_order=True)
        .select(
            "player_id",
            (pl.col("season") + 1).alias("prediction_season"),
            pl.col("team").alias("previous_primary_team"),
        )
    )


def build_projected_starters(legacy: pl.DataFrame, timestamped: pl.DataFrame) -> pl.DataFrame:
    old = (
        legacy.filter(
            pl.col("season").is_between(2006, 2024)
            & (pl.col("week") == 1)
            & (pl.col("formation") == "Offense")
            & (pl.col("position") == "QB")
            & (pl.col("depth_team") == "1")
            & pl.col("gsis_id").is_not_null()
        )
        .with_columns(normalize_team("club_code"))
        .sort(["season", "club_code", "depth_position", "gsis_id"])
        .unique(["season", "club_code"], keep="first", maintain_order=True)
        .select(
            pl.col("season").alias("prediction_season"),
            pl.col("club_code").alias("target_team"),
            pl.col("gsis_id").alias("projected_starting_qb_id"),
        )
    )
    new = (
        timestamped.with_columns(
            pl.col("dt").str.to_datetime(
                "%Y-%m-%dT%H:%M:%SZ", time_zone="UTC"
            ).alias("_date"),
            normalize_team("team"),
        )
        .filter(
            (pl.col("_date") < pl.datetime(2025, 9, 4, time_zone="UTC"))
            & (pl.col("pos_abb") == "QB")
            & (pl.col("pos_rank") == 1)
            & pl.col("gsis_id").is_not_null()
        )
        .sort(["team", "_date", "pos_slot", "gsis_id"], descending=[False, True, False, False])
        .unique("team", keep="first", maintain_order=True)
        .select(
            pl.lit(2025).alias("prediction_season"),
            pl.col("team").alias("target_team"),
            pl.col("gsis_id").alias("projected_starting_qb_id"),
        )
    )
    override = pl.DataFrame(PROJECTED_QB_OVERRIDES).with_columns(
        pl.col("prediction_season").cast(pl.Int32)
    )
    override_keys = override.select("prediction_season", "target_team")
    starters = (
        old.vstack(new)
        .join(override_keys, on=["prediction_season", "target_team"], how="anti")
        .vstack(override)
        .sort(["prediction_season", "target_team"])
    )
    invalid_counts = starters.group_by("prediction_season").len().filter(pl.col("len") != 32)
    if invalid_counts.height:
        raise Phase2FeatureError(
            "Projected starter table must contain 32 teams per season: "
            + str(invalid_counts.to_dicts())
        )
    return starters


def build_context(
    targets: pl.DataFrame,
    roster_snapshot: pl.DataFrame,
    previous_team: pl.DataFrame,
    starters: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    base = targets.select(
        "player_id", pl.col("season").alias("prediction_season"), "position"
    )
    joined = (
        base.join(roster_snapshot, on=["player_id", "prediction_season"], how="left", validate="1:1")
        .join(previous_team, on=["player_id", "prediction_season"], how="left", validate="1:1")
        .join(starters, on=["prediction_season", "target_team"], how="left", validate="m:1")
        .with_columns(
            (pl.col("target_team") != pl.col("previous_primary_team")).cast(pl.Int8).alias("changed_team"),
            pl.date(pl.col("prediction_season"), 9, 1).alias("context_cutoff_date"),
        )
    )
    exclusions = (
        joined.filter(pl.col("target_team").is_null() | pl.col("projected_starting_qb_id").is_null())
        .with_columns(
            pl.when(pl.col("target_team").is_null())
            .then(pl.lit("no_eligible_week1_roster_assignment"))
            .otherwise(pl.lit("no_projected_starting_qb"))
            .alias("exclusion_reason")
        )
        .select("player_id", "prediction_season", "position", "exclusion_reason")
    )
    context = joined.filter(
        pl.col("target_team").is_not_null() & pl.col("projected_starting_qb_id").is_not_null()
    )
    if context.select(pl.struct("player_id", "prediction_season").is_duplicated().sum()).item():
        raise Phase2FeatureError("Prediction context contains duplicate keys")
    return context, exclusions


def build_team_environment(team_weekly: pl.DataFrame) -> pl.DataFrame:
    totals = (
        team_weekly.filter(pl.col("season_type") == "REG")
        .with_columns(normalize_team("team"))
        .group_by("season", "team")
        .agg(
            pl.col("game_id").n_unique().alias("games"),
            *[pl.col(c).fill_null(0).sum().alias(c) for c in (
                "attempts", "sacks_suffered", "carries", "passing_tds",
                "rushing_tds", "passing_epa", "rushing_epa",
            )],
        )
        .with_columns(
            (pl.col("attempts") + pl.col("sacks_suffered")).alias("team_dropbacks"),
            (pl.col("attempts") + pl.col("sacks_suffered") + pl.col("carries")).alias("team_offensive_plays"),
        )
        .with_columns(
            _safe_ratio(pl.col("team_offensive_plays"), pl.col("games"), "target_team_plays_per_game_tminus1"),
            _safe_ratio(pl.col("team_dropbacks"), pl.col("games"), "target_team_dropbacks_per_game_tminus1"),
            _safe_ratio(pl.col("carries"), pl.col("games"), "target_team_carries_per_game_tminus1"),
            _safe_ratio(pl.col("team_dropbacks"), pl.col("team_offensive_plays"), "target_team_pass_rate_tminus1"),
            _safe_ratio(pl.col("passing_tds"), pl.col("games"), "target_team_passing_tds_per_game_tminus1"),
            _safe_ratio(pl.col("rushing_tds"), pl.col("games"), "target_team_rushing_tds_per_game_tminus1"),
            _safe_ratio(pl.col("passing_epa"), pl.col("team_dropbacks"), "target_team_pass_epa_per_dropback_tminus1"),
            _safe_ratio(pl.col("rushing_epa"), pl.col("carries"), "target_team_rush_epa_per_carry_tminus1"),
        )
    )
    return totals.select(
        (pl.col("season") + 1).alias("prediction_season"),
        pl.col("team").alias("target_team"),
        *[c for c in totals.columns if c.startswith("target_team_")],
    )


def build_workload(player_weekly: pl.DataFrame, team_weekly: pl.DataFrame) -> pl.DataFrame:
    players = (
        player_weekly.filter(
            (pl.col("season_type") == "REG")
            & pl.col("position").is_in(POSITIONS)
        )
        .with_columns(normalize_team("team"))
        .select(
            "player_id", "season", "week", "team", "position",
            pl.col("attempts").fill_null(0).alias("player_attempts"),
            pl.col("sacks_suffered").fill_null(0).alias("player_sacks"),
            pl.col("carries").fill_null(0).alias("player_carries"),
            pl.col("targets").fill_null(0).alias("player_targets"),
            pl.col("receiving_air_yards").fill_null(0).alias("player_air_yards"),
        )
    )
    teams = (
        team_weekly.filter(pl.col("season_type") == "REG")
        .with_columns(normalize_team("team"))
        .select(
            "season", "week", "team",
            pl.col("attempts").fill_null(0).alias("team_attempts"),
            pl.col("sacks_suffered").fill_null(0).alias("team_sacks"),
            pl.col("carries").fill_null(0).alias("team_carries"),
            pl.col("targets").fill_null(0).alias("team_targets"),
            pl.col("receiving_air_yards").fill_null(0).alias("team_air_yards"),
        )
    )
    summed = (
        players.join(teams, on=["season", "week", "team"], how="inner", validate="m:1")
        .group_by("player_id", "season", "position")
        .agg(pl.exclude("player_id", "season", "position", "week", "team").sum())
    )
    return summed.with_columns(
        _safe_ratio(pl.col("player_attempts") + pl.col("player_sacks"), pl.col("team_attempts") + pl.col("team_sacks"), "previous_qb_dropback_share"),
        _safe_ratio(pl.col("player_carries"), pl.col("team_carries"), "previous_carry_share"),
        _safe_ratio(pl.col("player_targets"), pl.col("team_targets"), "previous_target_share"),
        _safe_ratio(pl.col("player_carries") + pl.col("player_targets"), pl.col("team_carries") + pl.col("team_targets"), "previous_opportunity_share"),
        _safe_ratio(pl.col("player_air_yards"), pl.col("team_air_yards"), "previous_air_yards_share"),
    ).select(
        "player_id", (pl.col("season") + 1).alias("prediction_season"),
        "previous_qb_dropback_share", "previous_carry_share", "previous_target_share",
        "previous_opportunity_share", "previous_air_yards_share",
    )


def build_competition(
    context: pl.DataFrame,
    roster_snapshot: pl.DataFrame,
    player_seasons: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    roster = roster_snapshot.select(
        "prediction_season", "target_team",
        pl.col("player_id").alias("competitor_id"),
        pl.col("roster_position").alias("competitor_position"),
        pl.col("birth_date").cast(pl.Date, strict=False).alias("_birth_date"),
    )
    prior = player_seasons.select(
        (pl.col("season") + 1).alias("prediction_season"),
        pl.col("player_id").alias("competitor_id"),
        pl.col("total_fantasy_points").alias("competitor_previous_total_points"),
        pl.col("fantasy_points_per_game").alias("competitor_previous_ppg"),
        pl.col("games_played").alias("competitor_previous_games"),
    )
    long = (
        context.select("player_id", "prediction_season", "position", "target_team")
        .join(roster, on=["prediction_season", "target_team"], how="left", validate="m:m")
        .filter(pl.col("competitor_id") != pl.col("player_id"))
        .filter(
            pl.when(pl.col("position") == "QB").then(pl.col("competitor_position") == "QB")
            .when(pl.col("position") == "RB").then(pl.col("competitor_position") == "RB")
            .otherwise(pl.col("competitor_position").is_in(["WR", "TE"]))
        )
        .join(prior, on=["prediction_season", "competitor_id"], how="left", validate="m:1")
        .with_columns(
            (pl.col("prediction_season") - pl.col("_birth_date").dt.year()
             - ((pl.col("_birth_date").dt.month() > 9) | ((pl.col("_birth_date").dt.month() == 9) & (pl.col("_birth_date").dt.day() > 1))).cast(pl.Int32)
            ).cast(pl.Float64).alias("competitor_age_entering_season"),
            pl.col("competitor_previous_total_points").is_not_null().cast(pl.Int8).alias("competitor_has_previous_history"),
        )
        .sort(
            ["player_id", "prediction_season", "competitor_has_previous_history", "competitor_previous_total_points", "competitor_previous_ppg", "competitor_id"],
            descending=[False, False, True, True, True, False], nulls_last=True,
        )
        .with_columns(pl.int_range(1, pl.len() + 1).over(["player_id", "prediction_season"]).alias("competition_rank"))
        .drop("_birth_date")
    )
    keys = ["player_id", "prediction_season"]
    pool = long.group_by(keys).agg(
        pl.len().alias("competition_pool_size"),
        (pl.col("competitor_has_previous_history") == 0).sum().alias("competition_no_history_count"),
    )
    top = long.filter(pl.col("competition_rank") == 1).select(
        *keys,
        pl.col("competitor_previous_total_points").alias("top_competitor_previous_total_points"),
        pl.col("competitor_previous_ppg").alias("top_competitor_previous_ppg"),
        pl.col("competitor_age_entering_season").alias("top_competitor_age"),
        (pl.col("competitor_position") == "WR").cast(pl.Int8).alias("top_competitor_is_wr"),
        (pl.col("competitor_position") == "TE").cast(pl.Int8).alias("top_competitor_is_te"),
    )
    second = long.filter(pl.col("competition_rank") == 2).select(
        *keys,
        pl.col("competitor_previous_ppg").alias("second_competitor_previous_ppg"),
        (pl.col("competitor_position") == "WR").cast(pl.Int8).alias("second_competitor_is_wr"),
        (pl.col("competitor_position") == "TE").cast(pl.Int8).alias("second_competitor_is_te"),
    )
    model = pool.join(top, on=keys, how="left", validate="1:1").join(second, on=keys, how="left", validate="1:1")
    return long, model


def build_qb_environment(
    context: pl.DataFrame, players: pl.DataFrame, qbr: pl.DataFrame
) -> pl.DataFrame:
    ids = players.select(
        pl.col("gsis_id").alias("projected_starting_qb_id"),
        pl.col("espn_id").cast(pl.String).alias("_espn_id"),
    ).unique("projected_starting_qb_id")
    history = qbr.filter(pl.col("season_type") == "Regular").select(
        (pl.col("season") + 1).alias("prediction_season"),
        pl.col("player_id").cast(pl.String).alias("_espn_id"),
        pl.col("qbr_total").alias("projected_qb_previous_qbr"),
        pl.col("qb_plays").alias("projected_qb_previous_qb_plays"),
    ).unique(["prediction_season", "_espn_id"])
    return (
        context.select("prediction_season", "target_team", "projected_starting_qb_id")
        .unique(["prediction_season", "target_team"])
        .join(ids, on="projected_starting_qb_id", how="left", validate="m:1")
        .join(history, on=["prediction_season", "_espn_id"], how="left", validate="m:1")
        .with_columns(
            pl.col("projected_qb_previous_qbr").is_not_null().cast(pl.Int8).alias("projected_qb_has_previous_qbr"),
            pl.col("projected_qb_previous_qb_plays").fill_null(0),
        )
        .drop("_espn_id")
    )


def assemble_tables(
    v1_tables: dict[str, pl.DataFrame], context: pl.DataFrame,
    team_environment: pl.DataFrame, workload: pl.DataFrame,
    competition: pl.DataFrame, qb_environment: pl.DataFrame,
    rankings: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    common = (
        context.select(
            "player_id", "prediction_season", "target_team", "previous_primary_team",
            "changed_team", "projected_starting_qb_id", "context_cutoff_date",
        )
        .join(team_environment, on=["prediction_season", "target_team"], how="left", validate="m:1")
        .join(workload, on=["player_id", "prediction_season"], how="left", validate="1:1")
        .join(competition, on=["player_id", "prediction_season"], how="left", validate="1:1")
        .join(qb_environment, on=["prediction_season", "target_team", "projected_starting_qb_id"], how="left", validate="m:1")
        .join(rankings, on=["prediction_season", "target_team"], how="left", validate="m:1")
    )
    tables = {}
    for position, v1 in v1_tables.items():
        table = (
            v1.drop("changed_team")
            .join(common, on=["player_id", "prediction_season"], how="inner", validate="1:1")
            .with_columns(pl.col("target_team").alias("prediction_team"))
            .sort(["prediction_season", "player_id"])
        )
        if table.select(pl.struct("player_id", "prediction_season").is_duplicated().sum()).item():
            raise Phase2FeatureError(f"Duplicate final rows for {position}")
        tables[position] = table
    return tables


def attempt2_features(position: str, groups: list[str]) -> list[str]:
    features = list(V1_FEATURES[position])
    for group in groups:
        features.extend(GROUP_FEATURES[group][position])
    return list(dict.fromkeys(features))


def run_phase2_feature_build() -> dict[str, pl.DataFrame]:
    targets = pl.read_parquet(PROCESSED_DATA_DIR / "player_season_targets.parquet")
    player_seasons = pl.read_parquet(INTERIM_DATA_DIR / "player_seasons_all.parquet")
    player_weekly = pl.read_parquet(RAW_DATA_DIR / PLAYER_STATS_FILENAME)
    players = pl.read_parquet(RAW_DATA_DIR / PLAYERS_FILENAME)
    rosters = pl.read_parquet(ATTEMPT2_RAW_DATA_DIR / ROSTERS_WEEKLY_FILENAME)
    team_weekly = pl.read_parquet(ATTEMPT2_RAW_DATA_DIR / TEAM_WEEKLY_FILENAME)
    legacy_depth = pl.read_parquet(ATTEMPT2_RAW_DATA_DIR / DEPTH_CHARTS_FILENAME)
    new_depth = pl.read_parquet(ATTEMPT2_RAW_DATA_DIR / DEPTH_CHARTS_2025_FILENAME)
    qbr = pl.read_parquet(ATTEMPT2_RAW_DATA_DIR / QBR_FILENAME)
    rankings = pl.read_parquet(ATTEMPT2_RAW_DATA_DIR / RANKINGS_FILENAME).with_columns(normalize_team("target_team"))
    v1_tables = {p: pl.read_parquet(PROCESSED_DATA_DIR / f"{p.lower()}_modeling_dataset.parquet") for p in POSITIONS}

    roster_snapshot = build_roster_snapshot(rosters)
    previous_team = build_previous_primary_team(player_weekly)
    starters = build_projected_starters(legacy_depth, new_depth)
    context, exclusions = build_context(targets, roster_snapshot, previous_team, starters)
    team_environment = build_team_environment(team_weekly)
    workload = build_workload(player_weekly, team_weekly)
    competition_long, competition_model = build_competition(context, roster_snapshot, player_seasons)
    qb_environment = build_qb_environment(context, players, qbr)
    tables = assemble_tables(v1_tables, context, team_environment, workload, competition_model, qb_environment, rankings)

    for frame, path in (
        (context, ATTEMPT2_INTERIM_DATA_DIR / CONTEXT_FILENAME),
        (team_environment, ATTEMPT2_INTERIM_DATA_DIR / TEAM_ENV_FILENAME),
        (workload, ATTEMPT2_INTERIM_DATA_DIR / WORKLOAD_FILENAME),
        (competition_long, ATTEMPT2_INTERIM_DATA_DIR / COMPETITION_LONG_FILENAME),
        (competition_model, ATTEMPT2_INTERIM_DATA_DIR / COMPETITION_MODEL_FILENAME),
        (qb_environment, ATTEMPT2_INTERIM_DATA_DIR / QB_ENV_FILENAME),
    ):
        _write(frame, path)
    for position, table in tables.items():
        _write(table, ATTEMPT2_PROCESSED_DATA_DIR / f"{position.lower()}_attempt2_modeling_dataset.parquet")

    ATTEMPT2_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    exclusions.write_csv(ATTEMPT2_REPORT_TABLES_DIR / EXCLUSIONS_FILENAME)
    audit = {
        "context_rows": context.height,
        "excluded_rows": exclusions.height,
        "competition_rows": competition_long.height,
        "position_rows": {p: t.height for p, t in tables.items()},
        "rank_coverage": {p: t["preseason_offensive_line_score"].is_not_null().sum() for p, t in tables.items()},
        "qbr_coverage": {p: t["projected_qb_previous_qbr"].is_not_null().sum() for p, t in tables.items()},
        "timing_contract": "season-t context/rankings; all statistical inputs t-1",
    }
    (ATTEMPT2_REPORT_TABLES_DIR / AUDIT_FILENAME).write_text(
        "PHASE 2 FEATURE AUDIT\n\n" + json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return tables
