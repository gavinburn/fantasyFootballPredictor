"""Generate position-specific baselines with expanding-window validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.build_features import POSITIONS
from src.config import PREDICTIONS_DATA_DIR, PROCESSED_DATA_DIR, REPORT_TABLES_DIR
from src.validation import (
    MIN_TRAINING_ROWS,
    PREFERRED_EVALUATION_START,
    TOP_N_BY_POSITION,
    add_prediction_diagnostics,
    determine_evaluation_start,
    expanding_folds,
    regression_metrics,
)

BASELINE_PREDICTIONS_FILENAME = "historical_baseline_predictions.parquet"
FOLD_METRICS_FILENAME = "metrics_by_position_and_season.csv"
SUMMARY_METRICS_FILENAME = "baseline_metrics_by_position.csv"


class BaselineError(RuntimeError):
    """Raised when baseline predictions cannot be produced safely."""


def load_feature_tables(
    input_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, pl.DataFrame]:
    tables = {}
    for position in POSITIONS:
        path = input_dir / f"{position.lower()}_modeling_dataset.parquet"
        if not path.is_file():
            raise BaselineError(f"Missing {position} feature table: {path}")
        table = pl.read_parquet(path)
        if table["position"].unique().to_list() != [position]:
            raise BaselineError(f"{path} does not contain only {position} rows")
        tables[position] = table
    return tables


def _fold_predictions(
    position: str,
    season: int,
    validation: pl.DataFrame,
    *,
    predicted_ppg: pl.Expr,
    model_name: str,
) -> pl.DataFrame:
    columns = [
        "player_id",
        "player_name",
        "prediction_season",
        "position",
        "target_ppg",
    ]
    predictions = (
        validation.with_columns(predicted_ppg.alias("predicted_ppg"))
        .select(columns + ["predicted_ppg"])
        .rename(
            {
                "prediction_season": "season",
                "target_ppg": "actual_ppg",
            }
        )
        .with_columns(
            pl.lit(model_name).alias("model_name"),
            pl.lit(f"{position}_{season}").alias("fold"),
        )
    )
    return add_prediction_diagnostics(predictions, top_n=TOP_N_BY_POSITION[position])


def generate_baseline_predictions(
    tables: dict[str, pl.DataFrame],
    *,
    start_season: int | None = None,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate both baseline models on identical position-fold validation rows."""

    evaluation_start = start_season or determine_evaluation_start(
        tables,
        preferred_start=PREFERRED_EVALUATION_START,
        min_training_rows=min_training_rows,
    )
    all_predictions = []
    metric_rows = []
    for position in POSITIONS:
        folds = expanding_folds(
            tables[position],
            position,
            start_season=evaluation_start,
            require_previous_ppg=True,
            min_training_rows=min_training_rows,
        )
        for fold in folds:
            position_mean = float(fold.train["target_ppg"].mean())
            models = {
                "previous_season_ppg": pl.col("previous_season_ppg"),
                "historical_position_mean": pl.lit(position_mean),
            }
            for model_name, expression in models.items():
                predictions = _fold_predictions(
                    position,
                    fold.season,
                    fold.validation,
                    predicted_ppg=expression,
                    model_name=model_name,
                )
                all_predictions.append(predictions)
                metrics = regression_metrics(
                    predictions, top_n=TOP_N_BY_POSITION[position]
                )
                metric_rows.append(
                    {
                        "position": position,
                        "season": fold.season,
                        "model_name": model_name,
                        "fold": fold.name,
                        "training_start_season": int(
                            fold.train["prediction_season"].min()
                        ),
                        "training_end_season": int(
                            fold.train["prediction_season"].max()
                        ),
                        "training_rows": fold.train.height,
                        "validation_rows": fold.validation.height,
                        **metrics,
                    }
                )
    if not all_predictions:
        raise BaselineError("No baseline fold predictions were generated")
    predictions = pl.concat(all_predictions).sort(
        ["position", "season", "model_name", "predicted_rank"]
    )
    metrics = pl.DataFrame(metric_rows).sort(["position", "season", "model_name"])
    _validate_prediction_pairing(predictions)
    return predictions, metrics


def _validate_prediction_pairing(predictions: pl.DataFrame) -> None:
    counts = predictions.group_by(
        ["position", "season", "player_id", "model_name"]
    ).len()
    if counts.filter(pl.col("len") != 1).height:
        raise BaselineError("Baseline predictions contain duplicate player-model rows")
    pairs = predictions.group_by(["position", "season", "player_id"]).agg(
        pl.col("model_name").n_unique().alias("models")
    )
    if pairs.filter(pl.col("models") != 2).height:
        raise BaselineError("Baseline models were not evaluated on identical rows")


def summarize_metrics(
    predictions: pl.DataFrame, fold_metrics: pl.DataFrame
) -> pl.DataFrame:
    """Calculate aggregate out-of-sample metrics by position and baseline."""

    rows = []
    for key, group in predictions.group_by(["position", "model_name"]):
        position, model_name = key
        pooled = regression_metrics(group, top_n=TOP_N_BY_POSITION[position])
        folds = fold_metrics.filter(
            (pl.col("position") == position) & (pl.col("model_name") == model_name)
        )
        rows.append(
            {
                "position": position,
                "model_name": model_name,
                "validation_start_season": int(group["season"].min()),
                "validation_end_season": int(group["season"].max()),
                "number_of_validation_seasons": group["season"].n_unique(),
                "mae": pooled["mae"],
                "rmse": pooled["rmse"],
                "r_squared": pooled["r_squared"],
                "mean_fold_spearman_rank_correlation": folds[
                    "spearman_rank_correlation"
                ].mean(),
                "mean_fold_absolute_rank_error": folds[
                    "mean_absolute_rank_error"
                ].mean(),
                "mean_fold_top_n_overlap_rate": folds["top_n_overlap_rate"].mean(),
                "number_of_predictions": group.height,
            }
        )
    return pl.DataFrame(rows).sort(["position", "model_name"])


def _write_parquet_validated(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        frame.write_parquet(temporary, use_pyarrow=True)
        if pl.read_parquet(temporary).shape != frame.shape:
            raise BaselineError(f"Read-back validation failed for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_baselines(
    *,
    input_dir: Path = PROCESSED_DATA_DIR,
    predictions_path: Path = (PREDICTIONS_DATA_DIR / BASELINE_PREDICTIONS_FILENAME),
    fold_metrics_path: Path = REPORT_TABLES_DIR / FOLD_METRICS_FILENAME,
    summary_path: Path = REPORT_TABLES_DIR / SUMMARY_METRICS_FILENAME,
) -> tuple[Path, Path, Path]:
    tables = load_feature_tables(input_dir)
    start = determine_evaluation_start(tables)
    print(f"Expanding-window evaluation seasons: {start}-2025")
    predictions, metrics = generate_baseline_predictions(tables, start_season=start)
    summary = summarize_metrics(predictions, metrics)

    _write_parquet_validated(predictions, predictions_path)
    fold_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(fold_metrics_path)
    summary.write_csv(summary_path)
    print(
        f"Saved {predictions.height:,} baseline predictions: "
        f"{predictions_path.resolve()}"
    )
    print(f"Saved fold metrics: {fold_metrics_path.resolve()}")
    print(f"Saved baseline summary: {summary_path.resolve()}")
    return predictions_path, fold_metrics_path, summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PREDICTIONS_DATA_DIR / BASELINE_PREDICTIONS_FILENAME,
    )
    parser.add_argument(
        "--fold-metrics-output",
        type=Path,
        default=REPORT_TABLES_DIR / FOLD_METRICS_FILENAME,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=REPORT_TABLES_DIR / SUMMARY_METRICS_FILENAME,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_baselines(
            input_dir=args.input_dir,
            predictions_path=args.predictions_output,
            fold_metrics_path=args.fold_metrics_output,
            summary_path=args.summary_output,
        )
    except (BaselineError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
