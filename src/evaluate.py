"""Evaluate the first V1 linear-regression attempt against both baselines."""

from __future__ import annotations

import argparse
import importlib
import math
import os
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer

from src.baselines import load_feature_tables
from src.build_features import POSITIONS, V1_FEATURES
from src.config import (
    PREDICTIONS_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORT_FIGURES_DIR,
    REPORT_TABLES_DIR,
    REPORTS_DIR,
    REPOSITORY_ROOT,
)
from src.train_models import MODEL_NAME
from src.validation import TOP_N_BY_POSITION, regression_metrics

os.environ.setdefault("MPLCONFIGDIR", str(REPOSITORY_ROOT / ".matplotlib"))
matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")


PREVIOUS_BASELINE = "previous_season_ppg"
POSITION_MEAN_BASELINE = "historical_position_mean"
EXPECTED_MODELS = (PREVIOUS_BASELINE, POSITION_MEAN_BASELINE, MODEL_NAME)
STRONG_CORRELATION_THRESHOLD = 0.8


class EvaluationError(RuntimeError):
    """Raised when historical evaluation artifacts are invalid or incomparable."""


def load_out_of_sample_predictions(
    *,
    baseline_path: Path,
    linear_path: Path,
) -> pl.DataFrame:
    if not baseline_path.is_file() or not linear_path.is_file():
        raise EvaluationError(
            f"Missing prediction artifacts: baseline={baseline_path.is_file()}, "
            f"linear={linear_path.is_file()}"
        )
    baseline = pl.read_parquet(baseline_path)
    linear = pl.read_parquet(linear_path)
    predictions = pl.concat([baseline, linear], how="diagonal_relaxed")
    _validate_prediction_cohorts(predictions)
    return predictions


def _validate_prediction_cohorts(predictions: pl.DataFrame) -> None:
    if set(predictions["model_name"].unique()) != set(EXPECTED_MODELS):
        raise EvaluationError(
            f"Unexpected models: {predictions['model_name'].unique().to_list()}"
        )
    duplicates = (
        predictions.group_by(["position", "season", "player_id", "model_name"])
        .len()
        .filter(pl.col("len") != 1)
        .height
    )
    if duplicates:
        raise EvaluationError(f"Found {duplicates} duplicate prediction keys")
    cohorts = predictions.group_by(["position", "season", "player_id"]).agg(
        pl.col("model_name").n_unique().alias("model_count")
    )
    if cohorts.filter(pl.col("model_count") != len(EXPECTED_MODELS)).height:
        raise EvaluationError("Models were not evaluated on identical player rows")
    if predictions.filter(~pl.col("predicted_ppg").is_finite()).height:
        raise EvaluationError("Predictions contain non-finite values")


def calculate_metrics_by_season(predictions: pl.DataFrame) -> pl.DataFrame:
    """Calculate all required metrics for each position-season-model group."""

    rows = []
    for key, group in predictions.group_by(["position", "season", "model_name"]):
        position, season, model_name = key
        metrics = regression_metrics(group, top_n=TOP_N_BY_POSITION[position])
        rows.append(
            {
                "position": position,
                "validation_season": season,
                "model_name": model_name,
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
                "Spearman_rank_correlation": metrics["spearman_rank_correlation"],
                "Top_N": metrics["top_n"],
                "Top_N_overlap_count": metrics["top_n_overlap"],
                "Top_N_overlap_percentage": 100 * metrics["top_n_overlap_rate"],
                "number_of_predictions": metrics["number_of_predictions"],
            }
        )
    return pl.DataFrame(rows).sort(["position", "validation_season", "model_name"])


def _correlation(left: pl.Series, right: pl.Series) -> float:
    left_centered = left.cast(pl.Float64) - left.mean()
    right_centered = right.cast(pl.Float64) - right.mean()
    denominator = math.sqrt(
        float((left_centered**2).sum()) * float((right_centered**2).sum())
    )
    if denominator == 0:
        return float("nan")
    return float((left_centered * right_centered).sum()) / denominator


def build_evaluation_exclusions(
    tables: dict[str, pl.DataFrame],
    predictions: pl.DataFrame,
) -> pl.DataFrame:
    """List every 2012-2025 feature row omitted from the common comparison cohort."""

    evaluated = predictions.select(
        "player_id", "position", pl.col("season").alias("prediction_season")
    ).unique()
    frames = []
    for position, table in tables.items():
        candidates = table.filter(pl.col("prediction_season").is_between(2012, 2025))
        excluded = candidates.join(
            evaluated.filter(pl.col("position") == position),
            on=["player_id", "position", "prediction_season"],
            how="anti",
        ).select(
            "player_id",
            "player_name",
            "position",
            "prediction_season",
            pl.when(pl.col("previous_season_ppg").is_null())
            .then(pl.lit("missing exact previous-season PPG"))
            .otherwise(pl.lit("not present in common three-model cohort"))
            .alias("exclusion_reason"),
        )
        frames.append(excluded)
    return pl.concat(frames).sort(["position", "prediction_season", "player_id"])


def aggregate_comparison(
    predictions: pl.DataFrame,
    seasonal_metrics: pl.DataFrame,
    exclusions: pl.DataFrame,
) -> pl.DataFrame:
    """Build the required position-model aggregate comparison table."""

    rows = []
    for key, group in predictions.group_by(["position", "model_name"]):
        position, model_name = key
        seasonal = seasonal_metrics.filter(
            (pl.col("position") == position) & (pl.col("model_name") == model_name)
        )
        baseline_seasonal = seasonal_metrics.filter(
            (pl.col("position") == position)
            & (pl.col("model_name") == PREVIOUS_BASELINE)
        ).select(
            "validation_season",
            pl.col("MAE").alias("baseline_season_MAE"),
        )
        comparisons = seasonal.join(
            baseline_seasonal, on="validation_season", how="inner"
        ).with_columns(
            (pl.col("baseline_season_MAE") - pl.col("MAE")).alias("season_improvement")
        )
        errors = group["predicted_ppg"] - group["actual_ppg"]
        baseline_rows = predictions.filter(
            (pl.col("position") == position)
            & (pl.col("model_name") == PREVIOUS_BASELINE)
        )
        baseline_mae = float(baseline_rows["absolute_error"].mean())
        overall_mae = float(errors.abs().mean())
        improvement = baseline_mae - overall_mae
        rows.append(
            {
                "position": position,
                "model_name": model_name,
                "overall_MAE": overall_mae,
                "overall_RMSE": math.sqrt(float((errors**2).mean())),
                "overall_Spearman_rank_correlation": _correlation(
                    group["actual_rank"], group["predicted_rank"]
                ),
                "mean_seasonal_MAE": seasonal["MAE"].mean(),
                "mean_seasonal_RMSE": seasonal["RMSE"].mean(),
                "mean_seasonal_Spearman": seasonal["Spearman_rank_correlation"].mean(),
                "mean_top_N_overlap_percentage": seasonal[
                    "Top_N_overlap_percentage"
                ].mean(),
                "number_of_predictions": group.height,
                "number_of_validation_seasons": group["season"].n_unique(),
                "number_of_players_excluded": exclusions.filter(
                    pl.col("position") == position
                ).height,
                "baseline_MAE": baseline_mae,
                "MAE_improvement_over_previous_PPG_baseline": improvement,
                "percentage_MAE_improvement_over_previous_PPG_baseline": (
                    100 * improvement / baseline_mae
                ),
                "seasons_beating_previous_PPG_baseline": comparisons.filter(
                    pl.col("season_improvement") > 1e-12
                ).height,
                "seasons_tied_with_previous_PPG_baseline": comparisons.filter(
                    pl.col("season_improvement").abs() <= 1e-12
                ).height,
                "seasons_losing_to_previous_PPG_baseline": comparisons.filter(
                    pl.col("season_improvement") < -1e-12
                ).height,
            }
        )
    comparison = pl.DataFrame(rows)
    return _add_success_classification(comparison).sort(["position", "model_name"])


def _add_success_classification(comparison: pl.DataFrame) -> pl.DataFrame:
    baseline_rank = comparison.filter(pl.col("model_name") == PREVIOUS_BASELINE).select(
        "position",
        pl.col("overall_Spearman_rank_correlation").alias("_baseline_spearman"),
    )
    joined = comparison.join(baseline_rank, on="position", how="left")
    linear = pl.col("model_name") == MODEL_NAME
    successful = (
        (pl.col("MAE_improvement_over_previous_PPG_baseline") > 0)
        & (
            pl.col("seasons_beating_previous_PPG_baseline")
            > pl.col("seasons_losing_to_previous_PPG_baseline")
        )
        & (pl.col("overall_Spearman_rank_correlation") >= pl.col("_baseline_spearman"))
    )
    unsuccessful = (pl.col("MAE_improvement_over_previous_PPG_baseline") < 0) & (
        pl.col("seasons_losing_to_previous_PPG_baseline")
        > pl.col("seasons_beating_previous_PPG_baseline")
    )
    return joined.with_columns(
        pl.when(~linear)
        .then(pl.lit("Reference baseline"))
        .when(successful)
        .then(pl.lit("Successful"))
        .when(unsuccessful)
        .then(pl.lit("Unsuccessful"))
        .otherwise(pl.lit("Mixed"))
        .alias("first_attempt_classification")
    ).drop("_baseline_spearman")


def coefficient_stability(coefficients: pl.DataFrame) -> pl.DataFrame:
    """Summarize original-unit coefficient behavior across rolling folds."""

    rows = []
    for key, group in coefficients.group_by(["position", "feature"]):
        position, feature = key
        ordered = group.sort("season")["original_unit_coefficient"].to_numpy()
        signs = np.sign(ordered)
        nonzero_signs = signs[signs != 0]
        sign_changes = (
            int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
            if len(nonzero_signs) > 1
            else 0
        )
        mean = float(np.mean(ordered))
        std = float(np.std(ordered, ddof=1)) if len(ordered) > 1 else 0.0
        possible_changes = max(len(nonzero_signs) - 1, 1)
        sign_change_rate = sign_changes / possible_changes
        unstable = sign_change_rate >= 0.25 or std > abs(mean)
        rows.append(
            {
                "position": position,
                "feature": feature,
                "coefficient_mean": mean,
                "coefficient_standard_deviation": std,
                "coefficient_minimum": float(np.min(ordered)),
                "coefficient_maximum": float(np.max(ordered)),
                "sign_changes": sign_changes,
                "sign_change_rate": sign_change_rate,
                "folds": len(ordered),
                "unstable_coefficient": unstable,
            }
        )
    return pl.DataFrame(rows).sort(["position", "feature"])


def feature_correlation_matrix(
    table: pl.DataFrame, features: list[str]
) -> pl.DataFrame:
    """Return median-imputed V1 Pearson correlations for diagnostic use."""

    values = SimpleImputer(strategy="median").fit_transform(
        table.select(features).to_numpy()
    )
    correlation = np.corrcoef(values, rowvar=False)
    records = []
    for index, feature in enumerate(features):
        record = {"feature": feature}
        record.update(
            {
                compared: float(correlation[index, column_index])
                for column_index, compared in enumerate(features)
            }
        )
        records.append(record)
    return pl.DataFrame(records)


def _strong_pairs(matrix: pl.DataFrame) -> list[tuple[str, str, float]]:
    features = matrix["feature"].to_list()
    pairs = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            value = matrix[left_index, right]
            if (
                value is not None
                and math.isfinite(value)
                and abs(value) >= STRONG_CORRELATION_THRESHOLD
            ):
                pairs.append((left, right, float(value)))
    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)


def _save_figures(
    predictions: pl.DataFrame,
    seasonal_metrics: pl.DataFrame,
    comparison: pl.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    linear = predictions.filter(pl.col("model_name") == MODEL_NAME)
    for position in POSITIONS:
        rows = linear.filter(pl.col("position") == position)
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(
            rows["actual_ppg"],
            rows["predicted_ppg"],
            alpha=0.45,
            s=18,
        )
        lower = min(rows["actual_ppg"].min(), rows["predicted_ppg"].min())
        upper = max(rows["actual_ppg"].max(), rows["predicted_ppg"].max())
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="black")
        axis.set(
            title=f"{position} V1 rolling predictions",
            xlabel="Actual standard PPG",
            ylabel="Predicted standard PPG",
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"v1_predicted_vs_actual_{position.lower()}.png",
            dpi=160,
        )
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, position in zip(axes.flat, POSITIONS, strict=True):
        rows = seasonal_metrics.filter(pl.col("position") == position)
        for model in EXPECTED_MODELS:
            model_rows = rows.filter(pl.col("model_name") == model).sort(
                "validation_season"
            )
            axis.plot(
                model_rows["validation_season"],
                model_rows["MAE"],
                marker="o",
                markersize=3,
                label=model,
            )
        axis.set_title(position)
        axis.set_ylabel("MAE (PPG)")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Validation season")
    axes[-1, 1].set_xlabel("Validation season")
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("V1 MAE by season and position")
    figure.tight_layout()
    figure.savefig(output_dir / "v1_mae_by_season_and_position.png", dpi=160)
    plt.close(figure)

    linear_rows = comparison.filter(pl.col("model_name") == MODEL_NAME).sort("position")
    positions = linear_rows["position"].to_list()
    x = np.arange(len(positions))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        x - 0.18,
        linear_rows["baseline_MAE"],
        width=0.36,
        label="Previous-season PPG",
    )
    axis.bar(
        x + 0.18,
        linear_rows["overall_MAE"],
        width=0.36,
        label="V1 linear regression",
    )
    axis.set_xticks(x, positions)
    axis.set_ylabel("Overall MAE (PPG)")
    axis.set_title("V1 linear regression versus primary baseline")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "v1_baseline_comparison_by_position.png", dpi=160)
    plt.close(figure)


def _write_summary(
    comparison: pl.DataFrame,
    stability: pl.DataFrame,
    correlations: dict[str, pl.DataFrame],
    exclusions: pl.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "V1 FIRST LINEAR-REGRESSION EVALUATION",
        "",
        "All metrics use rolling historical out-of-sample predictions from 2012-2025.",
        "Primary baseline: previous-season PPG.",
        "Secondary baseline: fold-local historical position mean.",
        "Top-N is fixed at QB=10, RB=20, WR=20, TE=10.",
        "Success rules were fixed before classification: lower aggregate MAE, more",
        "season wins than losses, and Spearman ranking at least as high as baseline.",
        "",
    ]
    for position in POSITIONS:
        linear = comparison.filter(
            (pl.col("position") == position) & (pl.col("model_name") == MODEL_NAME)
        ).row(0, named=True)
        unstable = stability.filter(
            (pl.col("position") == position) & pl.col("unstable_coefficient")
        ).height
        pairs = _strong_pairs(correlations[position])
        lines.extend(
            [
                position,
                f"Classification: {linear['first_attempt_classification']}",
                f"Linear MAE: {linear['overall_MAE']:.4f} PPG",
                f"Baseline MAE: {linear['baseline_MAE']:.4f} PPG",
                (
                    "MAE improvement: "
                    f"{linear['MAE_improvement_over_previous_PPG_baseline']:.4f} "
                    f"PPG ({linear['percentage_MAE_improvement_over_previous_PPG_baseline']:.2f}%)"
                ),
                (
                    "Season record versus baseline: "
                    f"{linear['seasons_beating_previous_PPG_baseline']} wins, "
                    f"{linear['seasons_tied_with_previous_PPG_baseline']} ties, "
                    f"{linear['seasons_losing_to_previous_PPG_baseline']} losses"
                ),
                (
                    "Overall Spearman: "
                    f"{linear['overall_Spearman_rank_correlation']:.4f}"
                ),
                f"Excluded rows: {exclusions.filter(pl.col('position') == position).height}",
                f"Unstable coefficient flags: {unstable}",
                f"Strong feature-correlation pairs (|r| >= 0.8): {len(pairs)}",
                *[
                    f"  {left} vs {right}: r={value:.3f}"
                    for left, right, value in pairs
                ],
                "",
            ]
        )
    lines.extend(
        [
            "INTERPRETATION LIMITS",
            "Coefficient instability is diagnostic and does not change V1 features.",
            "Strong correlations can make individual coefficients hard to interpret.",
            "No injury causes are inferred, no 2026 predictions are generated, and",
            "no alternative model or optional feature is tested in this evaluation.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    *,
    baseline_path: Path = (
        PREDICTIONS_DATA_DIR / "historical_baseline_predictions.parquet"
    ),
    linear_path: Path = (
        PREDICTIONS_DATA_DIR / "historical_linear_regression_predictions.parquet"
    ),
    coefficients_path: Path = (
        PREDICTIONS_DATA_DIR / "linear_regression_fold_coefficients.parquet"
    ),
    feature_dir: Path = PROCESSED_DATA_DIR,
    tables_dir: Path = REPORT_TABLES_DIR,
    figures_dir: Path = REPORT_FIGURES_DIR,
    summary_path: Path = REPORTS_DIR / "v1_evaluation_summary.txt",
) -> dict[str, Path]:
    predictions = load_out_of_sample_predictions(
        baseline_path=baseline_path, linear_path=linear_path
    )
    tables = load_feature_tables(feature_dir)
    exclusions = build_evaluation_exclusions(tables, predictions)
    seasonal = calculate_metrics_by_season(predictions)
    comparison = aggregate_comparison(predictions, seasonal, exclusions)
    coefficients = pl.read_parquet(coefficients_path)
    stability = coefficient_stability(coefficients)
    correlations = {
        position: feature_correlation_matrix(tables[position], V1_FEATURES[position])
        for position in POSITIONS
    }

    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seasonal_metrics": tables_dir / "v1_metrics_by_position_and_season.csv",
        "comparison": tables_dir / "v1_model_comparison.csv",
        "stability": tables_dir / "v1_coefficient_stability.csv",
        "exclusions": tables_dir / "v1_evaluation_exclusions.csv",
    }
    seasonal.write_csv(paths["seasonal_metrics"])
    comparison.write_csv(paths["comparison"])
    stability.write_csv(paths["stability"])
    exclusions.write_csv(paths["exclusions"])
    for position, matrix in correlations.items():
        path = tables_dir / f"v1_feature_correlations_{position.lower()}.csv"
        matrix.write_csv(path)
        paths[f"{position}_correlations"] = path
    _save_figures(predictions, seasonal, comparison, figures_dir)
    _write_summary(comparison, stability, correlations, exclusions, summary_path)
    paths["summary"] = summary_path
    print(f"Saved evaluation comparison: {paths['comparison'].resolve()}")
    print(f"Saved evaluation summary: {summary_path.resolve()}")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=REPORT_TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=REPORT_FIGURES_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_evaluation(tables_dir=args.tables_dir, figures_dir=args.figures_dir)
    except (EvaluationError, pl.exceptions.PolarsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
