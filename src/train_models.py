"""Train and export four independent position-specific linear regressions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines import (
    FOLD_METRICS_FILENAME,
    load_feature_tables,
)
from src.build_features import POSITIONS, V1_FEATURES
from src.config import (
    MODELS_DIR,
    PREDICTIONS_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORT_EQUATIONS_DIR,
    REPORT_TABLES_DIR,
)
from src.validation import (
    TOP_N_BY_POSITION,
    add_prediction_diagnostics,
    determine_evaluation_start,
    expanding_folds,
    regression_metrics,
)

MODEL_NAME = "linear_regression_v1"
LINEAR_PREDICTIONS_FILENAME = "historical_linear_regression_predictions.parquet"
FOLD_COEFFICIENTS_FILENAME = "linear_regression_fold_coefficients.parquet"
LINEAR_SUMMARY_FILENAME = "linear_regression_metrics_by_position.csv"


class ModelTrainingError(RuntimeError):
    """Raised when a position-specific model cannot be trained safely."""


def build_linear_pipeline() -> Pipeline:
    """Create the required median-impute, scale, and OLS pipeline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def fit_position_pipeline(
    table: pl.DataFrame,
    position: str,
) -> Pipeline:
    """Fit one position model using only that position's V1 features."""

    if table["position"].unique().to_list() != [position]:
        raise ModelTrainingError(f"{position} training table contains another position")
    features = V1_FEATURES[position]
    pipeline = build_linear_pipeline()
    pipeline.fit(table.select(features).to_numpy(), table["target_ppg"].to_numpy())
    output_width = pipeline.named_steps["imputer"].statistics_.shape[0]
    if output_width != len(features):
        raise ModelTrainingError(
            f"{position} imputer changed feature width from "
            f"{len(features)} to {output_width}"
        )
    return pipeline


def coefficients_in_original_units(
    pipeline: Pipeline,
) -> tuple[float, np.ndarray]:
    """Convert standardized linear coefficients back to original input units."""

    scaler = pipeline.named_steps["scaler"]
    regressor = pipeline.named_steps["regressor"]
    coefficients = np.asarray(regressor.coef_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)
    original_coefficients = coefficients / scale
    original_intercept = float(
        regressor.intercept_ - np.sum(coefficients * mean / scale)
    )
    return original_intercept, original_coefficients


def _linear_fold_predictions(
    position: str,
    season: int,
    validation: pl.DataFrame,
    predictions: np.ndarray,
) -> pl.DataFrame:
    frame = (
        validation.select(
            "player_id",
            "player_name",
            "prediction_season",
            "position",
            "target_ppg",
        )
        .rename(
            {
                "prediction_season": "season",
                "target_ppg": "actual_ppg",
            }
        )
        .with_columns(
            pl.Series("predicted_ppg", predictions),
            pl.lit(MODEL_NAME).alias("model_name"),
            pl.lit(f"{position}_{season}").alias("fold"),
        )
    )
    return add_prediction_diagnostics(frame, top_n=TOP_N_BY_POSITION[position])


def generate_linear_validation(
    tables: dict[str, pl.DataFrame],
    *,
    start_season: int | None = None,
    min_training_rows: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fit four independent models through expanding-window validation."""

    evaluation_start = start_season or determine_evaluation_start(
        tables, min_training_rows=min_training_rows
    )
    prediction_frames = []
    metric_rows = []
    coefficient_rows = []
    for position in POSITIONS:
        features = V1_FEATURES[position]
        folds = expanding_folds(
            tables[position],
            position,
            start_season=evaluation_start,
            require_previous_ppg=True,
            min_training_rows=min_training_rows,
        )
        for fold in folds:
            pipeline = fit_position_pipeline(fold.train, position)
            predicted = pipeline.predict(fold.validation.select(features).to_numpy())
            predictions = _linear_fold_predictions(
                position, fold.season, fold.validation, predicted
            )
            prediction_frames.append(predictions)
            metrics = regression_metrics(predictions, top_n=TOP_N_BY_POSITION[position])
            metric_rows.append(
                {
                    "position": position,
                    "season": fold.season,
                    "model_name": MODEL_NAME,
                    "fold": fold.name,
                    "training_start_season": int(fold.train["prediction_season"].min()),
                    "training_end_season": int(fold.train["prediction_season"].max()),
                    "training_rows": fold.train.height,
                    "validation_rows": fold.validation.height,
                    **metrics,
                }
            )
            original_intercept, original_coefficients = coefficients_in_original_units(
                pipeline
            )
            regressor = pipeline.named_steps["regressor"]
            for index, feature in enumerate(features):
                coefficient_rows.append(
                    {
                        "position": position,
                        "season": fold.season,
                        "fold": fold.name,
                        "feature": feature,
                        "standardized_coefficient": float(regressor.coef_[index]),
                        "original_unit_coefficient": float(
                            original_coefficients[index]
                        ),
                        "standardized_intercept": float(regressor.intercept_),
                        "original_unit_intercept": original_intercept,
                    }
                )
    predictions = pl.concat(prediction_frames).sort(
        ["position", "season", "predicted_rank"]
    )
    metrics = pl.DataFrame(metric_rows).sort(["position", "season"])
    coefficients = pl.DataFrame(coefficient_rows).sort(
        ["position", "season", "feature"]
    )
    _validate_linear_predictions(predictions)
    return predictions, metrics, coefficients


def _validate_linear_predictions(predictions: pl.DataFrame) -> None:
    duplicates = (
        predictions.group_by(["position", "season", "player_id", "model_name"])
        .len()
        .filter(pl.col("len") != 1)
        .height
    )
    if duplicates:
        raise ModelTrainingError(
            f"Linear validation produced {duplicates} duplicate predictions"
        )
    if predictions.filter(~pl.col("predicted_ppg").is_finite()).height:
        raise ModelTrainingError("Linear validation produced non-finite predictions")


def _write_parquet_validated(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        frame.write_parquet(temporary, use_pyarrow=True)
        if pl.read_parquet(temporary).shape != frame.shape:
            raise ModelTrainingError(f"Read-back validation failed for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _aggregate_model_metrics(
    predictions: pl.DataFrame, fold_metrics: pl.DataFrame
) -> pl.DataFrame:
    rows = []
    for position, group in predictions.group_by("position"):
        position_name = position[0]
        metrics = regression_metrics(group, top_n=TOP_N_BY_POSITION[position_name])
        position_folds = fold_metrics.filter(pl.col("position") == position_name)
        rows.append(
            {
                "position": position_name,
                "model_name": MODEL_NAME,
                "validation_start_season": int(group["season"].min()),
                "validation_end_season": int(group["season"].max()),
                "number_of_validation_seasons": group["season"].n_unique(),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r_squared": metrics["r_squared"],
                "mean_fold_spearman_rank_correlation": position_folds[
                    "spearman_rank_correlation"
                ].mean(),
                "mean_fold_absolute_rank_error": position_folds[
                    "mean_absolute_rank_error"
                ].mean(),
                "mean_fold_top_n_overlap_rate": position_folds[
                    "top_n_overlap_rate"
                ].mean(),
                "number_of_predictions": group.height,
            }
        )
    return pl.DataFrame(rows).sort("position")


def _baseline_mae(baseline_predictions: pl.DataFrame, position: str) -> float:
    rows = baseline_predictions.filter(
        (pl.col("position") == position)
        & (pl.col("model_name") == "previous_season_ppg")
    )
    if rows.is_empty():
        return float("nan")
    return float(rows["absolute_error"].mean())


def export_equation(
    *,
    position: str,
    pipeline: Pipeline,
    training_table: pl.DataFrame,
    validation_predictions: pl.DataFrame,
    baseline_predictions: pl.DataFrame,
    output_path: Path,
) -> None:
    """Export standardized and original-unit forms of one final equation."""

    features = V1_FEATURES[position]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    regressor = pipeline.named_steps["regressor"]
    original_intercept, original_coefficients = coefficients_in_original_units(pipeline)
    linear_rows = validation_predictions.filter(pl.col("position") == position)
    linear_mae = float(linear_rows["absolute_error"].mean())
    baseline_mae = _baseline_mae(baseline_predictions, position)

    lines = [
        f"{position} LINEAR REGRESSION V1",
        "",
        (
            f"Training seasons: {training_table['prediction_season'].min()}-"
            f"{training_table['prediction_season'].max()}"
        ),
        f"Training rows: {training_table.height}",
        f"Rolling-validation MAE: {linear_mae:.6f}",
        f"Previous-season PPG baseline MAE: {baseline_mae:.6f}",
        f"MAE improvement over baseline: {baseline_mae - linear_mae:.6f}",
        "",
        "EXACT FEATURE ORDER",
        *[f"{index + 1}. {feature}" for index, feature in enumerate(features)],
        "",
        "TRAINING MEDIANS USED FOR MISSING VALUES",
        *[
            f"{feature}: {float(imputer.statistics_[index]):.10g}"
            for index, feature in enumerate(features)
        ],
        "",
        "STANDARDIZED EQUATION",
        f"intercept: {float(regressor.intercept_):.10g}",
        *[
            (f"{float(regressor.coef_[index]):+.10g} * standardized({feature})")
            for index, feature in enumerate(features)
        ],
        "",
        "STANDARDIZATION PARAMETERS",
        *[
            (
                f"{feature}: mean={float(scaler.mean_[index]):.10g}, "
                f"scale={float(scaler.scale_[index]):.10g}"
            )
            for index, feature in enumerate(features)
        ],
        "",
        "EQUATION IN ORIGINAL FEATURE UNITS",
        f"Predicted {position} PPG = {original_intercept:.10g}",
        *[
            f"{coefficient:+.10g} * {feature}"
            for feature, coefficient in zip(
                features, original_coefficients, strict=True
            )
        ],
        "",
        "COEFFICIENT UNITS",
        (
            "Standardized coefficients are PPG change per one training-set "
            "standard deviation."
        ),
        (
            "Original-unit coefficients are PPG change per one unit of the named "
            "feature, holding other features constant."
        ),
        ("Missing feature values are first replaced by the listed training median."),
        "",
        "WARNINGS",
        "Coefficients describe association, not causation.",
        (
            "Correlated football volume and efficiency features can make individual "
            "coefficients unstable; fold stability is evaluated in Phase 9."
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_fold_metrics(linear_metrics: pl.DataFrame, output_path: Path) -> None:
    if output_path.is_file():
        existing = pl.read_csv(output_path).filter(pl.col("model_name") != MODEL_NAME)
        combined = pl.concat([existing, linear_metrics], how="diagonal_relaxed").sort(
            ["position", "season", "model_name"]
        )
    else:
        combined = linear_metrics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(output_path)


def run_training(
    *,
    input_dir: Path = PROCESSED_DATA_DIR,
    predictions_path: Path = (PREDICTIONS_DATA_DIR / LINEAR_PREDICTIONS_FILENAME),
    coefficients_path: Path = (PREDICTIONS_DATA_DIR / FOLD_COEFFICIENTS_FILENAME),
    models_dir: Path = MODELS_DIR,
    equations_dir: Path = REPORT_EQUATIONS_DIR,
    fold_metrics_path: Path = REPORT_TABLES_DIR / FOLD_METRICS_FILENAME,
    summary_path: Path = REPORT_TABLES_DIR / LINEAR_SUMMARY_FILENAME,
    baseline_predictions_path: Path = (
        PREDICTIONS_DATA_DIR / "historical_baseline_predictions.parquet"
    ),
) -> dict[str, Path]:
    tables = load_feature_tables(input_dir)
    start = determine_evaluation_start(tables)
    predictions, metrics, coefficients = generate_linear_validation(
        tables, start_season=start
    )
    if not baseline_predictions_path.is_file():
        raise ModelTrainingError(
            f"Baseline predictions not found: {baseline_predictions_path}"
        )
    baseline_predictions = pl.read_parquet(baseline_predictions_path)

    _write_parquet_validated(predictions, predictions_path)
    _write_parquet_validated(coefficients, coefficients_path)
    _update_fold_metrics(metrics, fold_metrics_path)
    _aggregate_model_metrics(predictions, metrics).write_csv(summary_path)

    outputs: dict[str, Path] = {}
    for position in POSITIONS:
        table = tables[position]
        pipeline = fit_position_pipeline(table, position)
        model_path = models_dir / f"{position.lower()}_linear_regression.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact: dict[str, Any] = {
            "position": position,
            "model_name": MODEL_NAME,
            "features": V1_FEATURES[position],
            "training_start_season": int(table["prediction_season"].min()),
            "training_end_season": int(table["prediction_season"].max()),
            "training_rows": table.height,
            "pipeline": pipeline,
        }
        joblib.dump(artifact, model_path)
        equation_path = equations_dir / f"{position.lower()}_linear_regression.txt"
        export_equation(
            position=position,
            pipeline=pipeline,
            training_table=table,
            validation_predictions=predictions,
            baseline_predictions=baseline_predictions,
            output_path=equation_path,
        )
        outputs[position] = model_path
        print(f"Saved {position} model: {model_path.resolve()}")
        print(f"Saved {position} equation: {equation_path.resolve()}")

    print(
        f"Saved {predictions.height:,} rolling linear predictions: "
        f"{predictions_path.resolve()}"
    )
    print(f"Saved fold coefficients: {coefficients_path.resolve()}")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--equations-dir", type=Path, default=REPORT_EQUATIONS_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_training(
            input_dir=args.input_dir,
            models_dir=args.models_dir,
            equations_dir=args.equations_dir,
        )
    except (
        ModelTrainingError,
        pl.exceptions.PolarsError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
