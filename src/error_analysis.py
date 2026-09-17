"""Analyze errors from the fixed V1 rolling-validation predictions."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import polars as pl

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
from src.validation import TOP_N_BY_POSITION

os.environ.setdefault("MPLCONFIGDIR", str(REPOSITORY_ROOT / ".matplotlib"))
matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")


class ErrorAnalysisError(RuntimeError):
    """Raised when V1 errors cannot be analyzed reliably."""


DETAIL_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "prediction_season",
    "prior_team",
    "prediction_team",
    "actual_ppg",
    "predicted_ppg",
    "signed_error",
    "absolute_error",
    "previous_season_ppg",
    "previous_season_games_played",
    "changed_team",
    "prior_seasons_count",
]


def enrich_linear_predictions(
    predictions: pl.DataFrame,
    tables: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Join each out-of-sample prediction to its tracking fields and missing count."""

    contexts = []
    for position, table in tables.items():
        context = table.with_columns(
            pl.sum_horizontal(
                [
                    pl.col(feature).is_null().cast(pl.Int32)
                    for feature in V1_FEATURES[position]
                ]
            ).alias("missing_input_count")
        ).select(
            "player_id",
            "position",
            "prediction_season",
            "prior_team",
            "prediction_team",
            "previous_season_ppg",
            "previous_season_games_played",
            "changed_team",
            "prior_seasons_count",
            "age_entering_season",
            "missing_input_count",
        )
        contexts.append(context)
    context = pl.concat(contexts, how="diagonal_relaxed")
    enriched = (
        predictions.rename({"season": "prediction_season"})
        .join(
            context,
            on=["player_id", "position", "prediction_season"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            (pl.col("predicted_ppg") - pl.col("actual_ppg")).alias("signed_error")
        )
    )
    if enriched.height != predictions.height:
        raise ErrorAnalysisError("Tracking join changed prediction row count")
    if enriched["previous_season_ppg"].null_count():
        raise ErrorAnalysisError("Evaluated rows are missing previous-season context")
    return add_error_buckets(enriched)


def add_error_buckets(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply documented fixed buckets for Phase 10 group analysis."""

    return frame.with_columns(
        pl.when(pl.col("age_entering_season") < 24)
        .then(pl.lit("<24"))
        .when(pl.col("age_entering_season") < 27)
        .then(pl.lit("24-26"))
        .when(pl.col("age_entering_season") < 30)
        .then(pl.lit("27-29"))
        .when(pl.col("age_entering_season") < 33)
        .then(pl.lit("30-32"))
        .otherwise(pl.lit("33+"))
        .alias("age_bucket"),
        pl.when(pl.col("previous_season_games_played") < 8)
        .then(pl.lit("4-7"))
        .when(pl.col("previous_season_games_played") < 12)
        .then(pl.lit("8-11"))
        .when(pl.col("previous_season_games_played") < 15)
        .then(pl.lit("12-14"))
        .otherwise(pl.lit("15+"))
        .alias("games_played_bucket"),
        pl.when(pl.col("previous_season_ppg") < 2)
        .then(pl.lit("<2"))
        .when(pl.col("previous_season_ppg") < 5)
        .then(pl.lit("2-4.99"))
        .when(pl.col("previous_season_ppg") < 10)
        .then(pl.lit("5-9.99"))
        .when(pl.col("previous_season_ppg") < 15)
        .then(pl.lit("10-14.99"))
        .otherwise(pl.lit("15+"))
        .alias("previous_ppg_bucket"),
        pl.when(pl.col("prior_seasons_count") == 0)
        .then(pl.lit("0"))
        .when(pl.col("prior_seasons_count") == 1)
        .then(pl.lit("1"))
        .when(pl.col("prior_seasons_count") == 2)
        .then(pl.lit("2"))
        .otherwise(pl.lit("3+"))
        .alias("prior_history_bucket"),
        pl.when(pl.col("changed_team").is_null())
        .then(pl.lit("Unknown"))
        .when(pl.col("changed_team") == 1)
        .then(pl.lit("Changed"))
        .otherwise(pl.lit("Same"))
        .alias("changed_team_bucket"),
        pl.when(pl.col("missing_input_count") == 0)
        .then(pl.lit("0"))
        .when(pl.col("missing_input_count") <= 2)
        .then(pl.lit("1-2"))
        .when(pl.col("missing_input_count") <= 5)
        .then(pl.lit("3-5"))
        .otherwise(pl.lit("6+"))
        .alias("missing_input_bucket"),
        pl.when(pl.col("actual_ppg") < 2)
        .then(pl.lit("<2"))
        .when(pl.col("actual_ppg") < 5)
        .then(pl.lit("2-4.99"))
        .when(pl.col("actual_ppg") < 10)
        .then(pl.lit("5-9.99"))
        .when(pl.col("actual_ppg") < 15)
        .then(pl.lit("10-14.99"))
        .otherwise(pl.lit("15+"))
        .alias("actual_ppg_bucket"),
    )


def largest_errors(
    enriched: pl.DataFrame, sort_column: str, *, descending: bool
) -> pl.DataFrame:
    """Return the twenty requested rows per position for one error ordering."""

    frames = []
    for position in POSITIONS:
        frames.append(
            enriched.filter(pl.col("position") == position)
            .sort(sort_column, descending=descending)
            .select(DETAIL_COLUMNS)
            .head(20)
        )
    return pl.concat(frames)


def grouped_errors(frame: pl.DataFrame, group_column: str) -> pl.DataFrame:
    """Calculate MAE and signed bias by position and a documented bucket."""

    return (
        frame.group_by(["position", group_column])
        .agg(
            pl.len().alias("sample_size"),
            pl.col("absolute_error").mean().alias("MAE"),
            pl.col("signed_error").mean().alias("mean_signed_error"),
            pl.col("signed_error").median().alias("median_signed_error"),
            pl.col("absolute_error").max().alias("maximum_absolute_error"),
        )
        .sort(["position", group_column])
    )


def ranking_errors(frame: pl.DataFrame) -> pl.DataFrame:
    """Classify Top-N misses and calculate signed rank error."""

    top_n = pl.col("position").replace_strict(TOP_N_BY_POSITION, return_dtype=pl.Int32)
    return (
        frame.with_columns(
            (pl.col("predicted_rank") - pl.col("actual_rank")).alias("rank_error"),
            (pl.col("predicted_rank") <= top_n).alias("predicted_top_n"),
            (pl.col("actual_rank") <= top_n).alias("actual_top_n"),
        )
        .with_columns(
            pl.when(pl.col("predicted_top_n") & pl.col("actual_top_n"))
            .then(pl.lit("correct_top_n"))
            .when(pl.col("predicted_top_n") & ~pl.col("actual_top_n"))
            .then(pl.lit("predicted_top_n_but_finished_outside"))
            .when(~pl.col("predicted_top_n") & pl.col("actual_top_n"))
            .then(pl.lit("missed_actual_top_n"))
            .otherwise(pl.lit("outside_top_n"))
            .alias("top_n_status")
        )
        .select(
            "player_id",
            "player_name",
            "position",
            "prediction_season",
            "actual_ppg",
            "predicted_ppg",
            "actual_rank",
            "predicted_rank",
            "rank_error",
            "predicted_top_n",
            "actual_top_n",
            "top_n_status",
        )
        .sort(
            ["position", "prediction_season", "rank_error"],
            descending=[False, False, True],
        )
    )


def _save_residual_figures(frame: pl.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for position in POSITIONS:
        rows = frame.filter(pl.col("position") == position)
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(
            rows["predicted_ppg"],
            rows["signed_error"],
            alpha=0.45,
            s=18,
        )
        axis.axhline(0, color="black", linestyle="--")
        axis.set(
            title=f"{position} V1 residuals",
            xlabel="Predicted standard PPG",
            ylabel="Signed error (predicted - actual)",
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"v1_residuals_{position.lower()}.png", dpi=160)
        plt.close(figure)


def _write_summary(
    frame: pl.DataFrame,
    grouped: dict[str, pl.DataFrame],
    rankings: pl.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "V1 FIRST-ATTEMPT ERROR ANALYSIS",
        "",
        "Signed error is predicted PPG minus actual PPG.",
        "Positive values are overpredictions; negative values are underpredictions.",
        "Buckets: age <24/24-26/27-29/30-32/33+; prior games 4-7/8-11/",
        "12-14/15+; previous and actual PPG <2/2-4.99/5-9.99/10-14.99/15+;",
        "prior seasons 0/1/2/3+; missing inputs 0/1-2/3-5/6+.",
        "Groups with small samples are reported but should not support conclusions.",
        "",
    ]
    for position in POSITIONS:
        rows = frame.filter(pl.col("position") == position)
        high = rows.filter(pl.col("actual_ppg") >= 15)
        limited = rows.filter(pl.col("previous_season_games_played") < 8)
        top_misses = rankings.filter(
            (pl.col("position") == position)
            & pl.col("top_n_status").is_in(
                [
                    "predicted_top_n_but_finished_outside",
                    "missed_actual_top_n",
                ]
            )
        )
        lines.extend(
            [
                position,
                f"Predictions: {rows.height}",
                f"Overall MAE: {rows['absolute_error'].mean():.4f}",
                f"Mean signed error: {rows['signed_error'].mean():.4f}",
                (
                    "Actual PPG >=15: "
                    f"n={high.height}, mean signed error="
                    f"{high['signed_error'].mean() if high.height else float('nan'):.4f}"
                ),
                (
                    "Previous games 4-7: "
                    f"n={limited.height}, MAE="
                    f"{limited['absolute_error'].mean() if limited.height else float('nan'):.4f}"
                ),
                f"Top-N false inclusions/omissions: {top_misses.height}",
                "",
            ]
        )
    lines.extend(
        [
            "DATA LIMITATIONS",
            "Changed-team status is Unknown for all evaluated rows because historical",
            "preseason roster snapshots were unavailable; no changed-team conclusion",
            "can be supported.",
            "Missing-input groups reflect only the fixed V1 feature set.",
            "No error is attributed to injury, suspension, or depth-chart events.",
            "Observed differences are descriptive and do not establish causation.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_error_analysis(
    *,
    predictions_path: Path = (
        PREDICTIONS_DATA_DIR / "historical_linear_regression_predictions.parquet"
    ),
    feature_dir: Path = PROCESSED_DATA_DIR,
    tables_dir: Path = REPORT_TABLES_DIR,
    figures_dir: Path = REPORT_FIGURES_DIR,
    summary_path: Path = REPORTS_DIR / "v1_error_analysis_summary.txt",
) -> dict[str, Path]:
    if not predictions_path.is_file():
        raise ErrorAnalysisError(f"Missing linear predictions: {predictions_path}")
    predictions = pl.read_parquet(predictions_path)
    tables = load_feature_tables(feature_dir)
    enriched = enrich_linear_predictions(predictions, tables)

    outputs = {
        "over": tables_dir / "v1_largest_overpredictions.csv",
        "under": tables_dir / "v1_largest_underpredictions.csv",
        "absolute": tables_dir / "v1_largest_absolute_errors.csv",
        "age": tables_dir / "v1_errors_by_age_bucket.csv",
        "games": tables_dir / "v1_errors_by_games_played_bucket.csv",
        "previous_ppg": tables_dir / "v1_errors_by_previous_ppg_bucket.csv",
        "history": tables_dir / "v1_errors_by_prior_history.csv",
        "changed_team": tables_dir / "v1_errors_by_changed_team.csv",
        "missing": tables_dir / "v1_errors_by_missing_inputs.csv",
        "actual_ppg": tables_dir / "v1_errors_by_actual_ppg.csv",
        "position_season": tables_dir / "v1_errors_by_position_and_season.csv",
        "ranking": tables_dir / "v1_ranking_errors.csv",
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    largest_errors(enriched, "signed_error", descending=True).write_csv(outputs["over"])
    largest_errors(enriched, "signed_error", descending=False).write_csv(
        outputs["under"]
    )
    largest_errors(enriched, "absolute_error", descending=True).write_csv(
        outputs["absolute"]
    )
    group_specs = {
        "age": "age_bucket",
        "games": "games_played_bucket",
        "previous_ppg": "previous_ppg_bucket",
        "history": "prior_history_bucket",
        "changed_team": "changed_team_bucket",
        "missing": "missing_input_bucket",
        "actual_ppg": "actual_ppg_bucket",
    }
    grouped = {}
    for key, column in group_specs.items():
        grouped[key] = grouped_errors(enriched, column)
        grouped[key].write_csv(outputs[key])
    grouped["position_season"] = grouped_errors(enriched, "prediction_season")
    grouped["position_season"].write_csv(outputs["position_season"])
    ranks = ranking_errors(enriched)
    ranks.write_csv(outputs["ranking"])
    _save_residual_figures(enriched, figures_dir)
    _write_summary(enriched, grouped, ranks, summary_path)
    outputs["summary"] = summary_path
    print(f"Saved error analysis summary: {summary_path.resolve()}")
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=REPORT_TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=REPORT_FIGURES_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_error_analysis(tables_dir=args.tables_dir, figures_dir=args.figures_dir)
    except (ErrorAnalysisError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
