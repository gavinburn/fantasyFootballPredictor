"""Reusable expanding-window validation and regression metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

PREFERRED_EVALUATION_START = 2012
MIN_TRAINING_ROWS = 50
TOP_N_BY_POSITION = {"QB": 10, "RB": 20, "WR": 20, "TE": 10}


class ValidationError(RuntimeError):
    """Raised when a time-based validation fold is invalid."""


@dataclass(frozen=True)
class ExpandingFold:
    """One position-specific expanding-window fold."""

    position: str
    season: int
    train: pl.DataFrame
    validation: pl.DataFrame

    @property
    def name(self) -> str:
        return f"{self.position}_{self.season}"


def determine_evaluation_start(
    tables: dict[str, pl.DataFrame],
    *,
    preferred_start: int = PREFERRED_EVALUATION_START,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> int:
    """Find the first preferred-or-later season valid for every position."""

    if not tables:
        raise ValidationError("No position feature tables were supplied")
    common_seasons = set.intersection(
        *[
            set(table["prediction_season"].unique().to_list())
            for table in tables.values()
        ]
    )
    for season in sorted(year for year in common_seasons if year >= preferred_start):
        valid = True
        for table in tables.values():
            train_count = table.filter(pl.col("prediction_season") < season).height
            validation_count = table.filter(
                (pl.col("prediction_season") == season)
                & pl.col("previous_season_ppg").is_not_null()
            ).height
            if train_count < min_training_rows or validation_count == 0:
                valid = False
                break
        if valid:
            return int(season)
    raise ValidationError(
        "No common evaluation season has enough earlier training and validation rows"
    )


def expanding_folds(
    table: pl.DataFrame,
    position: str,
    *,
    start_season: int,
    end_season: int | None = None,
    require_previous_ppg: bool = True,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> list[ExpandingFold]:
    """Create expanding folds for one position without mixing positions."""

    if table.is_empty():
        raise ValidationError(f"{position} feature table is empty")
    actual_positions = table["position"].unique().to_list()
    if actual_positions != [position]:
        raise ValidationError(
            f"{position} validation received positions {actual_positions}"
        )
    final_season = end_season or int(table["prediction_season"].max())
    folds: list[ExpandingFold] = []
    for season in range(start_season, final_season + 1):
        train = table.filter(pl.col("prediction_season") < season)
        validation_filter = pl.col("prediction_season") == season
        if require_previous_ppg:
            validation_filter &= pl.col("previous_season_ppg").is_not_null()
        validation = table.filter(validation_filter)
        if train.height < min_training_rows or validation.is_empty():
            continue
        if int(train["prediction_season"].max()) >= season:
            raise ValidationError(
                f"{position} fold {season} contains current/future training data"
            )
        if validation["position"].unique().to_list() != [position]:
            raise ValidationError(f"{position} fold {season} mixed positions")
        folds.append(ExpandingFold(position, season, train, validation))
    if not folds:
        raise ValidationError(
            f"No valid {position} folds from {start_season} through {final_season}"
        )
    return folds


def add_prediction_diagnostics(
    predictions: pl.DataFrame,
    *,
    top_n: int,
) -> pl.DataFrame:
    """Add errors and within-fold actual/predicted ranks."""

    required = {"actual_ppg", "predicted_ppg"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValidationError(f"Predictions are missing columns: {sorted(missing)}")
    if predictions.is_empty():
        raise ValidationError("Cannot rank an empty prediction set")
    return predictions.with_columns(
        (pl.col("predicted_ppg") - pl.col("actual_ppg")).alias("error"),
        (pl.col("predicted_ppg") - pl.col("actual_ppg")).abs().alias("absolute_error"),
        pl.col("predicted_ppg")
        .rank(method="average", descending=True)
        .alias("predicted_rank"),
        pl.col("actual_ppg")
        .rank(method="average", descending=True)
        .alias("actual_rank"),
        pl.lit(min(top_n, predictions.height)).alias("top_n"),
    )


def regression_metrics(
    predictions: pl.DataFrame,
    *,
    top_n: int,
) -> dict[str, float | int]:
    """Calculate regression and ranking metrics for one fold-model pair."""

    diagnosed = add_prediction_diagnostics(predictions, top_n=top_n)
    actual = diagnosed["actual_ppg"]
    predicted = diagnosed["predicted_ppg"]
    errors = predicted - actual
    mse = float((errors**2).mean())
    actual_mean = float(actual.mean())
    total_variance = float(((actual - actual_mean) ** 2).sum())
    residual_variance = float((errors**2).sum())
    r_squared = (
        1.0 - residual_variance / total_variance if total_variance > 0 else float("nan")
    )
    actual_rank = actual.rank(method="average", descending=True)
    predicted_rank = predicted.rank(method="average", descending=True)
    actual_rank_centered = actual_rank - actual_rank.mean()
    predicted_rank_centered = predicted_rank - predicted_rank.mean()
    correlation_denominator = math.sqrt(
        float((actual_rank_centered**2).sum())
        * float((predicted_rank_centered**2).sum())
    )
    spearman = (
        float((actual_rank_centered * predicted_rank_centered).sum())
        / correlation_denominator
        if correlation_denominator > 0
        else float("nan")
    )

    effective_top_n = min(top_n, diagnosed.height)
    actual_top = set(
        diagnosed.sort("actual_ppg", descending=True)
        .head(effective_top_n)["player_id"]
        .to_list()
    )
    predicted_top = set(
        diagnosed.sort("predicted_ppg", descending=True)
        .head(effective_top_n)["player_id"]
        .to_list()
    )
    return {
        "mae": float(errors.abs().mean()),
        "rmse": math.sqrt(mse),
        "r_squared": r_squared,
        "spearman_rank_correlation": spearman,
        "mean_absolute_rank_error": float(
            (diagnosed["predicted_rank"] - diagnosed["actual_rank"]).abs().mean()
        ),
        "top_n": effective_top_n,
        "top_n_overlap": len(actual_top.intersection(predicted_top)),
        "top_n_overlap_rate": len(actual_top.intersection(predicted_top))
        / effective_top_n,
        "number_of_predictions": diagnosed.height,
    }
