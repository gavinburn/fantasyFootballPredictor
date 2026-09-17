"""Run Attempt 2 expanding-window ablations and export comparable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ATTEMPT2_MODELS_DIR,
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    ATTEMPT2_PROCESSED_DATA_DIR,
    ATTEMPT2_REPORT_FIGURES_DIR,
    ATTEMPT2_REPORT_TABLES_DIR,
    ATTEMPT2_REPORTS_DIR,
    POSITIONS,
)
from src.phase2_features import attempt2_features
from src.validation import (
    TOP_N_BY_POSITION,
    add_prediction_diagnostics,
    expanding_folds,
    regression_metrics,
)

PREDICTIONS_FILENAME = "attempt2_ablation_predictions.parquet"
COEFFICIENTS_FILENAME = "attempt2_fold_coefficients.parquet"
FOLD_METRICS_FILENAME = "attempt2_metrics_by_position_season_variant.csv"
SUMMARY_FILENAME = "attempt2_ablation_summary.csv"
STABILITY_FILENAME = "attempt2_coefficient_stability.csv"
MISSINGNESS_FILENAME = "attempt2_feature_missingness.csv"
CORRELATION_FILENAME = "attempt2_feature_correlations.csv"

AUTOMATED_GROUPS = ["volume", "workload", "competition", "quarterback", "efficiency", "team_change"]
VARIANTS: dict[str, list[str]] = {
    "A_v1_same_cohort": [],
    "B_v1_plus_volume": ["volume"],
    "C_v1_plus_workload": ["workload"],
    "D_v1_plus_competition": ["competition"],
    "E_v1_plus_qb_environment": ["quarterback"],
    "F_v1_plus_efficiency": ["efficiency"],
    "G_all_automated": AUTOMATED_GROUPS,
    "H1_G_plus_offensive_line": [*AUTOMATED_GROUPS, "offensive_line"],
    "H2_G_plus_schedule": [*AUTOMATED_GROUPS, "schedule"],
    "H3_G_plus_ol_and_schedule": [*AUTOMATED_GROUPS, "offensive_line", "schedule"],
}
FINAL_VARIANT = "H3_G_plus_ol_and_schedule"


class Phase2ModelError(RuntimeError):
    """Raised when an Attempt 2 model or evaluation artifact is invalid."""


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def load_attempt2_tables() -> dict[str, pl.DataFrame]:
    return {
        p: pl.read_parquet(
            ATTEMPT2_PROCESSED_DATA_DIR / f"{p.lower()}_attempt2_modeling_dataset.parquet"
        )
        for p in POSITIONS
    }


def _prediction_frame(
    validation: pl.DataFrame, predictions: np.ndarray, position: str,
    season: int, variant: str,
) -> pl.DataFrame:
    return add_prediction_diagnostics(
        validation.select(
            "player_id", "player_name", "prediction_season", "position", "target_ppg"
        )
        .rename({"prediction_season": "season", "target_ppg": "actual_ppg"})
        .with_columns(
            pl.Series("predicted_ppg", predictions),
            pl.lit(variant).alias("model_name"),
            pl.lit(f"{position}_{season}_{variant}").alias("fold"),
        ),
        top_n=TOP_N_BY_POSITION[position],
    )


def run_ablation_validation(
    tables: dict[str, pl.DataFrame], *, start_season: int = 2012,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    predictions = []
    metric_rows = []
    coefficient_rows = []
    for variant, groups in VARIANTS.items():
        for position in POSITIONS:
            table = tables[position]
            features = attempt2_features(position, groups)
            missing = sorted(set(features) - set(table.columns))
            if missing:
                raise Phase2ModelError(f"{position} {variant} missing features: {missing}")
            for fold in expanding_folds(
                table, position, start_season=start_season,
                require_previous_ppg=True, min_training_rows=50,
            ):
                pipeline = build_pipeline()
                pipeline.fit(
                    fold.train.select(features).to_numpy(),
                    fold.train["target_ppg"].to_numpy(),
                )
                predicted = pipeline.predict(fold.validation.select(features).to_numpy())
                frame = _prediction_frame(
                    fold.validation, predicted, position, fold.season, variant
                )
                predictions.append(frame)
                metrics = regression_metrics(frame, top_n=TOP_N_BY_POSITION[position])
                metric_rows.append(
                    {
                        "variant": variant,
                        "position": position,
                        "season": fold.season,
                        "training_start_season": int(fold.train["prediction_season"].min()),
                        "training_end_season": int(fold.train["prediction_season"].max()),
                        "training_rows": fold.train.height,
                        **metrics,
                    }
                )
                regressor = pipeline.named_steps["regressor"]
                for feature, coefficient in zip(features, regressor.coef_, strict=True):
                    coefficient_rows.append(
                        {
                            "variant": variant,
                            "position": position,
                            "season": fold.season,
                            "feature": feature,
                            "standardized_coefficient": float(coefficient),
                        }
                    )
    prediction_frame = pl.concat(predictions).sort(["model_name", "position", "season", "predicted_rank"])
    if prediction_frame.group_by(["model_name", "position", "season", "player_id"]).len().filter(pl.col("len") != 1).height:
        raise Phase2ModelError("Ablation validation produced duplicate predictions")
    if prediction_frame.filter(~pl.col("predicted_ppg").is_finite()).height:
        raise Phase2ModelError("Ablation validation produced non-finite predictions")
    return prediction_frame, pl.DataFrame(metric_rows), pl.DataFrame(coefficient_rows)


def summarize_ablation(predictions: pl.DataFrame, folds: pl.DataFrame) -> pl.DataFrame:
    rows = []
    v1 = folds.filter(pl.col("variant") == "A_v1_same_cohort").select(
        "position", "season", pl.col("mae").alias("v1_mae")
    )
    with_baseline = folds.join(v1, on=["position", "season"], how="left", validate="m:1")
    for keys, group in predictions.group_by("model_name", "position"):
        variant, position = keys
        metrics = regression_metrics(group, top_n=TOP_N_BY_POSITION[position])
        fg = with_baseline.filter(
            (pl.col("variant") == variant) & (pl.col("position") == position)
        )
        rows.append(
            {
                "variant": variant,
                "position": position,
                "validation_start_season": int(group["season"].min()),
                "validation_end_season": int(group["season"].max()),
                "validation_seasons": group["season"].n_unique(),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mean_fold_spearman": float(fg["spearman_rank_correlation"].mean()),
                "mean_fold_top_n_overlap_rate": float(fg["top_n_overlap_rate"].mean()),
                "seasons_beating_v1": fg.filter(pl.col("mae") < pl.col("v1_mae")).height,
                "mean_mae_delta_vs_v1": float((fg["mae"] - fg["v1_mae"]).mean()),
                "predictions": group.height,
            }
        )
    return pl.DataFrame(rows).sort("variant", "position")


def coefficient_stability(coefficients: pl.DataFrame) -> pl.DataFrame:
    return (
        coefficients.group_by("variant", "position", "feature")
        .agg(
            pl.len().alias("folds"),
            pl.col("standardized_coefficient").mean().alias("mean_coefficient"),
            pl.col("standardized_coefficient").std().alias("coefficient_std"),
            pl.col("standardized_coefficient").min().alias("min_coefficient"),
            pl.col("standardized_coefficient").max().alias("max_coefficient"),
        )
        .with_columns(
            ((pl.col("min_coefficient") < 0) & (pl.col("max_coefficient") > 0))
            .cast(pl.Int8)
            .alias("sign_changed_across_folds")
        )
        .sort("variant", "position", "feature")
    )


def feature_diagnostics(tables: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame]:
    missing_rows = []
    correlation_frames = []
    for position, table in tables.items():
        features = attempt2_features(position, VARIANTS[FINAL_VARIANT])
        for feature in features:
            missing_rows.append(
                {
                    "position": position,
                    "feature": feature,
                    "missing_count": table[feature].null_count(),
                    "missing_rate": table[feature].null_count() / table.height,
                }
            )
        corr = table.select(features).corr().with_columns(
            pl.Series("feature", features), pl.lit(position).alias("position")
        ).select("position", "feature", *features)
        correlation_frames.append(corr)
    return pl.DataFrame(missing_rows), pl.concat(correlation_frames, how="diagonal")


def fit_final_models(tables: dict[str, pl.DataFrame]) -> None:
    ATTEMPT2_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for position, table in tables.items():
        features = attempt2_features(position, VARIANTS[FINAL_VARIANT])
        pipeline = build_pipeline()
        pipeline.fit(table.select(features).to_numpy(), table["target_ppg"].to_numpy())
        artifact = {
            "attempt": 2,
            "position": position,
            "variant": FINAL_VARIANT,
            "features": features,
            "training_start_season": int(table["prediction_season"].min()),
            "training_end_season": int(table["prediction_season"].max()),
            "training_rows": table.height,
            "pipeline": pipeline,
        }
        joblib.dump(artifact, ATTEMPT2_MODELS_DIR / f"{position.lower()}_attempt2_linear_regression.joblib")


def write_summary(summary: pl.DataFrame, tables: dict[str, pl.DataFrame]) -> Path:
    lines = [
        "ATTEMPT 2 PHASE 2 EVALUATION SUMMARY",
        "",
        "Evaluation: expanding-window seasons 2012-2025; training uses only earlier seasons.",
        "All variants use the same point-in-time roster-context cohort.",
        "Primary metric: MAE. Lower MAE and MAE delta are better.",
        "",
    ]
    for position in POSITIONS:
        rows = summary.filter(pl.col("position") == position).sort("mae")
        best = rows.row(0, named=True)
        final = rows.filter(pl.col("variant") == FINAL_VARIANT).row(0, named=True)
        v1 = rows.filter(pl.col("variant") == "A_v1_same_cohort").row(0, named=True)
        lines.extend(
            [
                position,
                f"  Modeling rows: {tables[position].height}",
                f"  Best variant: {best['variant']}",
                f"  Best MAE: {best['mae']:.4f}",
                f"  Same-cohort V1 MAE: {v1['mae']:.4f}",
                f"  Full Attempt 2 MAE: {final['mae']:.4f}",
                f"  Full delta versus V1: {final['mae'] - v1['mae']:+.4f}",
                f"  Full seasons beating V1: {final['seasons_beating_v1']} of {final['validation_seasons']}",
                f"  Full mean fold Spearman: {final['mean_fold_spearman']:.4f}",
                f"  Full mean Top-N overlap: {final['mean_fold_top_n_overlap_rate']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "INTERPRETATION",
            "Feature groups are evaluated independently before the combined model.",
            "The full model is saved for reproducibility, but the best out-of-sample",
            "variant by position should guide later feature retention.",
            "A small aggregate gain is not sufficient if it is unstable across seasons.",
            "",
        ]
    )
    path = ATTEMPT2_REPORTS_DIR / "attempt2_evaluation_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_figures(summary: pl.DataFrame, predictions: pl.DataFrame) -> None:
    ATTEMPT2_REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    variants = list(VARIANTS)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for axis, position in zip(axes.flat, POSITIONS, strict=True):
        rows = summary.filter(pl.col("position") == position)
        lookup = {r["variant"]: r["mae"] for r in rows.to_dicts()}
        axis.barh(range(len(variants)), [lookup[v] for v in variants])
        axis.set_yticks(range(len(variants)), variants, fontsize=7)
        axis.invert_yaxis()
        axis.set_title(f"{position} MAE by ablation")
        axis.set_xlabel("MAE (lower is better)")
    fig.savefig(ATTEMPT2_REPORT_FIGURES_DIR / "attempt2_ablation_mae.png", dpi=160)
    plt.close(fig)

    final = predictions.filter(pl.col("model_name") == FINAL_VARIANT)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for axis, position in zip(axes.flat, POSITIONS, strict=True):
        rows = final.filter(pl.col("position") == position)
        axis.scatter(rows["actual_ppg"], rows["predicted_ppg"], alpha=0.35, s=12)
        low = min(float(rows["actual_ppg"].min()), float(rows["predicted_ppg"].min()))
        high = max(float(rows["actual_ppg"].max()), float(rows["predicted_ppg"].max()))
        axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        axis.set_title(position)
        axis.set_xlabel("Actual PPG")
        axis.set_ylabel("Predicted PPG")
    fig.savefig(ATTEMPT2_REPORT_FIGURES_DIR / "attempt2_predicted_vs_actual.png", dpi=160)
    plt.close(fig)


def run_attempt2_models() -> dict[str, Path]:
    tables = load_attempt2_tables()
    predictions, folds, coefficients = run_ablation_validation(tables)
    summary = summarize_ablation(predictions, folds)
    stability = coefficient_stability(coefficients)
    missingness, correlations = feature_diagnostics(tables)

    ATTEMPT2_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ATTEMPT2_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(ATTEMPT2_PREDICTIONS_DATA_DIR / PREDICTIONS_FILENAME, use_pyarrow=True)
    coefficients.write_parquet(ATTEMPT2_PREDICTIONS_DATA_DIR / COEFFICIENTS_FILENAME, use_pyarrow=True)
    folds.write_csv(ATTEMPT2_REPORT_TABLES_DIR / FOLD_METRICS_FILENAME)
    summary.write_csv(ATTEMPT2_REPORT_TABLES_DIR / SUMMARY_FILENAME)
    stability.write_csv(ATTEMPT2_REPORT_TABLES_DIR / STABILITY_FILENAME)
    missingness.write_csv(ATTEMPT2_REPORT_TABLES_DIR / MISSINGNESS_FILENAME)
    correlations.write_csv(ATTEMPT2_REPORT_TABLES_DIR / CORRELATION_FILENAME)
    fit_final_models(tables)
    summary_path = write_summary(summary, tables)
    write_figures(summary, predictions)
    metadata = {
        "variants": VARIANTS,
        "final_variant": FINAL_VARIANT,
        "validation_seasons": [2012, 2025],
        "positions": list(POSITIONS),
    }
    metadata_path = ATTEMPT2_REPORT_TABLES_DIR / "attempt2_experiment_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary_path, "manifest": metadata_path}
