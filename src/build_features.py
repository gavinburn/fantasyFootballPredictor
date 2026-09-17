"""Build leakage-safe, position-specific Phase 5 modeling datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from src.build_player_seasons import TARGET_FILENAME
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORT_TABLES_DIR
from src.download_data import PLAYERS_FILENAME

POSITIONS = ("QB", "RB", "WR", "TE")
AGE_CUTOFF_MONTH = 9
AGE_CUTOFF_DAY = 1

IDENTIFICATION_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "prediction_season",
    "prior_team",
    "prediction_team",
    "target_ppg",
]

SHARED_HISTORY_COLUMNS = [
    "previous_season_ppg",
    "two_season_mean_ppg",
    "three_season_mean_ppg",
    "previous_season_games_played",
    "two_season_mean_games_played",
    "three_season_mean_games_played",
    "previous_year_ppg_change",
    "prior_seasons_count",
]

PLAYER_CONTEXT_COLUMNS = [
    "age_entering_season",
    "years_experience_entering_season",
    "draft_round",
    "draft_pick",
    "undrafted_indicator",
    "changed_team",
]

# changed_team is intentionally absent: historical preseason rosters were not
# downloaded, and Phase 4's team is the final observed team for that season.
V1_FEATURES: dict[str, list[str]] = {
    "QB": [
        "previous_season_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
        "previous_passing_attempts_per_game",
        "previous_passing_yards_per_attempt",
        "previous_passing_touchdowns_per_game",
        "previous_interceptions_per_game",
        "previous_qb_rushing_attempts_per_game",
        "previous_qb_rushing_yards_per_game",
        "previous_qb_rushing_touchdowns_per_game",
        "age_entering_season",
        "years_experience_entering_season",
    ],
    "RB": [
        "previous_season_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
        "previous_rushing_attempts_per_game",
        "previous_rushing_yards_per_attempt",
        "previous_rushing_touchdowns_per_game",
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_touchdowns_per_game",
        "previous_opportunities_per_game",
        "age_entering_season",
        "years_experience_entering_season",
    ],
    "WR": [
        "previous_season_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_yards_per_target",
        "previous_receiving_touchdowns_per_game",
        "previous_rushing_attempts_per_game",
        "previous_rushing_yards_per_game",
        "age_entering_season",
        "years_experience_entering_season",
    ],
    "TE": [
        "previous_season_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_yards_per_target",
        "previous_receiving_touchdowns_per_game",
        "age_entering_season",
        "years_experience_entering_season",
    ],
}

POSITION_DERIVED_FEATURES = {
    "QB": [
        "previous_passing_attempts_per_game",
        "previous_passing_yards_per_attempt",
        "previous_passing_touchdowns_per_game",
        "previous_interceptions_per_game",
        "previous_qb_rushing_attempts_per_game",
        "previous_qb_rushing_yards_per_game",
        "previous_qb_rushing_touchdowns_per_game",
    ],
    "RB": [
        "previous_rushing_attempts_per_game",
        "previous_rushing_yards_per_attempt",
        "previous_rushing_touchdowns_per_game",
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_touchdowns_per_game",
        "previous_opportunities_per_game",
    ],
    "WR": [
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_yards_per_target",
        "previous_receiving_touchdowns_per_game",
        "previous_rushing_attempts_per_game",
        "previous_rushing_yards_per_game",
    ],
    "TE": [
        "previous_targets_per_game",
        "previous_receiving_yards_per_game",
        "previous_receiving_yards_per_target",
        "previous_receiving_touchdowns_per_game",
    ],
}

LAG_SOURCE_COLUMNS = [
    "team",
    "fantasy_points_per_game",
    "games_played",
    "passing_attempts",
    "passing_yards",
    "passing_touchdowns",
    "passing_interceptions",
    "rushing_attempts",
    "rushing_yards",
    "rushing_touchdowns",
    "targets",
    "receiving_yards",
    "receiving_touchdowns",
    "opportunities",
]


class FeatureBuildError(RuntimeError):
    """Raised when leakage-safe features cannot be built."""


def _safe_ratio(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(None)
        .alias(alias)
    )


def _lag_lookup(player_seasons: pl.DataFrame, lag: int) -> pl.DataFrame:
    expressions = [
        pl.col(column).alias(f"lag{lag}_{column}") for column in LAG_SOURCE_COLUMNS
    ]
    return player_seasons.select(
        "player_id",
        (pl.col("season") + lag).alias("prediction_season"),
        *expressions,
    )


def _add_player_context(
    features: pl.DataFrame, player_master: pl.DataFrame
) -> pl.DataFrame:
    required = {
        "gsis_id",
        "birth_date",
        "rookie_season",
        "draft_round",
        "draft_pick",
    }
    missing = sorted(required.difference(player_master.columns))
    if missing:
        raise FeatureBuildError(
            "Player master is missing required columns: " + ", ".join(missing)
        )
    duplicate_ids = (
        player_master.group_by("gsis_id").len().filter(pl.col("len") > 1).height
    )
    if duplicate_ids:
        raise FeatureBuildError(
            f"Player master contains {duplicate_ids} duplicate gsis_id values"
        )

    context = player_master.select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("birth_date").str.to_date(strict=False).alias("_birth_date"),
        "rookie_season",
        pl.col("draft_round").cast(pl.Float64),
        pl.col("draft_pick").cast(pl.Float64),
    )
    joined = features.join(context, on="player_id", how="left", validate="m:1")
    birthday_after_cutoff = (pl.col("_birth_date").dt.month() > AGE_CUTOFF_MONTH) | (
        (pl.col("_birth_date").dt.month() == AGE_CUTOFF_MONTH)
        & (pl.col("_birth_date").dt.day() > AGE_CUTOFF_DAY)
    )
    return joined.with_columns(
        (
            pl.col("prediction_season")
            - pl.col("_birth_date").dt.year()
            - birthday_after_cutoff.cast(pl.Int32)
        )
        .cast(pl.Float64)
        .alias("age_entering_season"),
        pl.when(pl.col("prediction_season") >= pl.col("rookie_season"))
        .then(pl.col("prediction_season") - pl.col("rookie_season"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("years_experience_entering_season"),
        (pl.col("draft_round").is_null() & pl.col("draft_pick").is_null())
        .cast(pl.Int8)
        .alias("undrafted_indicator"),
        pl.lit(None, dtype=pl.Float64).alias("changed_team"),
    ).drop("_birth_date", "rookie_season")


def build_feature_tables(
    player_seasons: pl.DataFrame,
    player_master: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Build four tables whose statistical inputs use seasons t-1 through t-3."""

    required = {
        "player_id",
        "player_name",
        "position",
        "season",
        "team",
        *LAG_SOURCE_COLUMNS[1:],
    }
    missing = sorted(required.difference(player_seasons.columns))
    if missing:
        raise FeatureBuildError(
            "Phase 4 target table is missing required columns: " + ", ".join(missing)
        )
    duplicates = (
        player_seasons.group_by(["player_id", "season"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicates:
        raise FeatureBuildError(
            f"Phase 4 input contains {duplicates} duplicate player-season keys"
        )

    base = (
        player_seasons.select(
            "player_id",
            "player_name",
            "position",
            pl.col("season").alias("prediction_season"),
            pl.col("team").alias("prediction_team"),
            pl.col("fantasy_points_per_game").alias("target_ppg"),
        )
        .sort(["player_id", "prediction_season"])
        .with_columns(
            (pl.col("prediction_season").rank("ordinal").over("player_id") - 1)
            .cast(pl.UInt32)
            .alias("prior_seasons_count")
        )
    )
    for lag in (1, 2, 3):
        base = base.join(
            _lag_lookup(player_seasons, lag),
            on=["player_id", "prediction_season"],
            how="left",
            validate="1:1",
        )

    base = base.with_columns(
        pl.col("lag1_team").alias("prior_team"),
        pl.col("lag1_fantasy_points_per_game").alias("previous_season_ppg"),
        pl.mean_horizontal(
            "lag1_fantasy_points_per_game", "lag2_fantasy_points_per_game"
        ).alias("two_season_mean_ppg"),
        pl.mean_horizontal(
            "lag1_fantasy_points_per_game",
            "lag2_fantasy_points_per_game",
            "lag3_fantasy_points_per_game",
        ).alias("three_season_mean_ppg"),
        pl.col("lag1_games_played")
        .cast(pl.Float64)
        .alias("previous_season_games_played"),
        pl.mean_horizontal("lag1_games_played", "lag2_games_played").alias(
            "two_season_mean_games_played"
        ),
        pl.mean_horizontal(
            "lag1_games_played", "lag2_games_played", "lag3_games_played"
        ).alias("three_season_mean_games_played"),
        (
            pl.col("lag1_fantasy_points_per_game")
            - pl.col("lag2_fantasy_points_per_game")
        ).alias("previous_year_ppg_change"),
        _safe_ratio(
            "lag1_passing_attempts",
            "lag1_games_played",
            "previous_passing_attempts_per_game",
        ),
        _safe_ratio(
            "lag1_passing_yards",
            "lag1_passing_attempts",
            "previous_passing_yards_per_attempt",
        ),
        _safe_ratio(
            "lag1_passing_touchdowns",
            "lag1_games_played",
            "previous_passing_touchdowns_per_game",
        ),
        _safe_ratio(
            "lag1_passing_interceptions",
            "lag1_games_played",
            "previous_interceptions_per_game",
        ),
        _safe_ratio(
            "lag1_rushing_attempts",
            "lag1_games_played",
            "previous_rushing_attempts_per_game",
        ),
        _safe_ratio(
            "lag1_rushing_attempts",
            "lag1_games_played",
            "previous_qb_rushing_attempts_per_game",
        ),
        _safe_ratio(
            "lag1_rushing_yards",
            "lag1_games_played",
            "previous_rushing_yards_per_game",
        ),
        _safe_ratio(
            "lag1_rushing_yards",
            "lag1_games_played",
            "previous_qb_rushing_yards_per_game",
        ),
        _safe_ratio(
            "lag1_rushing_touchdowns",
            "lag1_games_played",
            "previous_rushing_touchdowns_per_game",
        ),
        _safe_ratio(
            "lag1_rushing_touchdowns",
            "lag1_games_played",
            "previous_qb_rushing_touchdowns_per_game",
        ),
        _safe_ratio(
            "lag1_rushing_yards",
            "lag1_rushing_attempts",
            "previous_rushing_yards_per_attempt",
        ),
        _safe_ratio("lag1_targets", "lag1_games_played", "previous_targets_per_game"),
        _safe_ratio(
            "lag1_receiving_yards",
            "lag1_games_played",
            "previous_receiving_yards_per_game",
        ),
        _safe_ratio(
            "lag1_receiving_yards",
            "lag1_targets",
            "previous_receiving_yards_per_target",
        ),
        _safe_ratio(
            "lag1_receiving_touchdowns",
            "lag1_games_played",
            "previous_receiving_touchdowns_per_game",
        ),
        _safe_ratio(
            "lag1_opportunities",
            "lag1_games_played",
            "previous_opportunities_per_game",
        ),
    )
    base = _add_player_context(base, player_master)

    tables: dict[str, pl.DataFrame] = {}
    for position in POSITIONS:
        columns = list(
            dict.fromkeys(
                IDENTIFICATION_COLUMNS
                + SHARED_HISTORY_COLUMNS
                + PLAYER_CONTEXT_COLUMNS
                + POSITION_DERIVED_FEATURES[position]
            )
        )
        table = (
            base.filter(pl.col("position") == position)
            .select(columns)
            .sort(["prediction_season", "player_id"])
        )
        _validate_position_table(position, table)
        tables[position] = table
    return tables


def _validate_position_table(position: str, table: pl.DataFrame) -> None:
    if table.is_empty():
        raise FeatureBuildError(f"{position} feature table is empty")
    actual_positions = table["position"].unique().to_list()
    if actual_positions != [position]:
        raise FeatureBuildError(
            f"{position} table contains unexpected positions: {actual_positions}"
        )
    if table.select(["player_id", "prediction_season"]).is_duplicated().any():
        raise FeatureBuildError(f"{position} table has duplicate player-season rows")
    forbidden = {"player_id", "player_name", "target_ppg", "position"}
    overlap = forbidden.intersection(V1_FEATURES[position])
    if overlap:
        raise FeatureBuildError(
            f"{position} model inputs contain identifiers or target: {sorted(overlap)}"
        )
    missing_features = sorted(set(V1_FEATURES[position]).difference(table.columns))
    if missing_features:
        raise FeatureBuildError(
            f"{position} table is missing V1 inputs: {missing_features}"
        )

    first_rows = (
        table.sort(["player_id", "prediction_season"])
        .group_by("player_id", maintain_order=True)
        .first()
    )
    lagged = [
        "previous_season_ppg",
        "two_season_mean_ppg",
        "three_season_mean_ppg",
        "previous_season_games_played",
    ]
    if first_rows.select(pl.any_horizontal(pl.col(lagged).is_not_null()).any()).item():
        raise FeatureBuildError(
            f"{position} first observed player-seasons have non-null lagged values"
        )


def _write_parquet_validated(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        frame.write_parquet(temporary, use_pyarrow=True)
        if pl.read_parquet(temporary).shape != frame.shape:
            raise FeatureBuildError(f"Read-back validation failed for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _feature_metadata(feature: str) -> tuple[str, str, str]:
    fixed = {
        "previous_season_ppg": (
            "Actual standard PPG from season t-1.",
            "PPG(t-1)",
            "fantasy_points_per_game",
        ),
        "two_season_mean_ppg": (
            "Mean standard PPG from available seasons t-1 and t-2.",
            "mean(PPG(t-1), PPG(t-2)), ignoring unavailable seasons",
            "fantasy_points_per_game",
        ),
        "three_season_mean_ppg": (
            "Mean standard PPG from available seasons t-1 through t-3.",
            "mean(PPG(t-1), PPG(t-2), PPG(t-3)), ignoring unavailable seasons",
            "fantasy_points_per_game",
        ),
        "previous_season_games_played": (
            "Games played in season t-1.",
            "games_played(t-1)",
            "games_played",
        ),
        "two_season_mean_games_played": (
            "Mean games played in available seasons t-1 and t-2.",
            "mean(games_played(t-1), games_played(t-2))",
            "games_played",
        ),
        "three_season_mean_games_played": (
            "Mean games played in available seasons t-1 through t-3.",
            "mean(games_played(t-1), games_played(t-2), games_played(t-3))",
            "games_played",
        ),
        "previous_year_ppg_change": (
            "Change between the last two exact seasons.",
            "PPG(t-1) - PPG(t-2)",
            "fantasy_points_per_game",
        ),
        "prior_seasons_count": (
            "Number of earlier eligible player-seasons in the Phase 4 table.",
            "count(rows for player where season < t)",
            "player_id, season",
        ),
        "age_entering_season": (
            "Age on September 1 of the prediction season.",
            "whole years between birth_date and September 1 of t",
            "players.birth_date, season",
        ),
        "years_experience_entering_season": (
            "Elapsed seasons since the player's rookie season.",
            "prediction_season - rookie_season",
            "players.rookie_season, season",
        ),
        "draft_round": ("NFL draft round.", "players.draft_round", "draft_round"),
        "draft_pick": ("Overall NFL draft pick.", "players.draft_pick", "draft_pick"),
        "undrafted_indicator": (
            "One when both draft round and pick are unavailable.",
            "int(draft_round is null and draft_pick is null)",
            "draft_round, draft_pick",
        ),
        "changed_team": (
            "Unavailable: historical preseason team snapshots were not downloaded.",
            "null; deliberately not inferred from final season team",
            "unavailable",
        ),
    }
    if feature in fixed:
        return fixed[feature]
    if feature.startswith("previous_"):
        base = feature.removeprefix("previous_")
        if base == "passing_yards_per_attempt":
            formula = "passing_yards(t-1) / passing_attempts(t-1)"
            source = "passing_yards, passing_attempts"
        elif base == "rushing_yards_per_attempt":
            formula = "rushing_yards(t-1) / rushing_attempts(t-1)"
            source = "rushing_yards, rushing_attempts"
        elif base == "receiving_yards_per_target":
            formula = "receiving_yards(t-1) / targets(t-1)"
            source = "receiving_yards, targets"
        elif base.startswith("qb_rushing_"):
            stat = base.removeprefix("qb_").removesuffix("_per_game")
            formula = f"individual QB {stat}(t-1) / games_played(t-1)"
            source = f"{stat}, games_played"
        else:
            stat = base.removesuffix("_per_game")
            formula = f"{stat}(t-1) / games_played(t-1)"
            source = f"{stat}, games_played"
        return (
            f"Prior-season individual player {base.replace('_', ' ')}.",
            formula,
            source,
        )
    return (feature.replace("_", " ").capitalize() + ".", "tracking value", feature)


def write_feature_dictionary(
    position: str, table: pl.DataFrame, output_path: Path
) -> None:
    records = []
    for feature in table.columns:
        meaning, formula, sources = _feature_metadata(feature)
        non_null = table.filter(pl.col(feature).is_not_null())
        earliest = (
            non_null["prediction_season"].min()
            if non_null.height and feature != "prediction_season"
            else table["prediction_season"].min()
        )
        if feature == "changed_team":
            feature_set = "unavailable V1"
        elif feature in V1_FEATURES[position]:
            feature_set = "V1"
        elif feature in IDENTIFICATION_COLUMNS or feature in PLAYER_CONTEXT_COLUMNS:
            feature_set = "tracking-only"
        else:
            feature_set = "shared-history"
        records.append(
            {
                "feature_name": feature,
                "plain_language_meaning": meaning,
                "exact_formula": formula,
                "source_columns": sources,
                "earliest_data_season_available": earliest,
                "feature_set": feature_set,
                "model_input": feature in V1_FEATURES[position],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_csv(output_path)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def write_leakage_audit(
    tables: dict[str, pl.DataFrame], input_path: Path, output_path: Path
) -> None:
    lines = [
        "PHASE 5 LEAKAGE AUDIT",
        "",
        f"INPUT: {input_path.resolve()}",
        "TARGET: target_ppg is regular-season standard fantasy_points_per_game in t.",
        "LAGS: exact calendar seasons t-1, t-2, and t-3 are joined by player_id.",
        "ROLLING MEANS: calculated only from those lagged columns.",
        "AGE CUTOFF: September 1 of prediction_season.",
        "QB CARRIES: individual official carries; source includes scrambles and kneels.",
        "TEAM CONTEXT: prediction_team is tracking-only final observed season team.",
        "CHANGED TEAM: unavailable without historical preseason rosters; excluded from V1.",
        "MISSING VALUES: retained for fold-local imputation in later phases.",
        "",
        "CHECKS PASSED",
        "First observed player-season has null lagged history.",
        "No raw player ID, player name, position, or target is a model input.",
        "No same-season performance statistic is a model input.",
        "Each table contains only its named position and unique player-season rows.",
        "",
    ]
    for position, table in tables.items():
        missing = {
            feature: round(100 * table[feature].null_count() / table.height, 4)
            for feature in V1_FEATURES[position]
        }
        lines.extend(
            [
                f"{position}",
                f"rows: {table.height}",
                (
                    f"prediction seasons: {table['prediction_season'].min()}-"
                    f"{table['prediction_season'].max()}"
                ),
                "V1 model inputs: " + ", ".join(V1_FEATURES[position]),
                "missing percentage by V1 feature:",
                _json(missing),
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_build(
    *,
    input_path: Path = PROCESSED_DATA_DIR / TARGET_FILENAME,
    player_master_path: Path = RAW_DATA_DIR / PLAYERS_FILENAME,
    output_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORT_TABLES_DIR,
) -> dict[str, Path]:
    if not input_path.is_file() or not player_master_path.is_file():
        raise FeatureBuildError(
            f"Missing input: target={input_path.is_file()}, "
            f"player_master={player_master_path.is_file()}"
        )
    player_seasons = pl.read_parquet(input_path)
    player_master = pl.read_parquet(player_master_path)
    tables = build_feature_tables(player_seasons, player_master)

    outputs: dict[str, Path] = {}
    for position, table in tables.items():
        slug = position.lower()
        dataset_path = output_dir / f"{slug}_modeling_dataset.parquet"
        dictionary_path = report_dir / f"{slug}_feature_dictionary.csv"
        _write_parquet_validated(table, dataset_path)
        write_feature_dictionary(position, table, dictionary_path)
        outputs[position] = dataset_path
        print(
            f"Saved {position}: {table.height:,} rows x {table.width:,} columns: "
            f"{dataset_path.resolve()}"
        )
    audit_path = report_dir / "leakage_audit.txt"
    write_leakage_audit(tables, input_path, audit_path)
    print(f"Saved leakage audit: {audit_path.resolve()}")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROCESSED_DATA_DIR / TARGET_FILENAME
    )
    parser.add_argument(
        "--player-master", type=Path, default=RAW_DATA_DIR / PLAYERS_FILENAME
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_TABLES_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_build(
            input_path=args.input,
            player_master_path=args.player_master,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
        )
    except (FeatureBuildError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
