"""Diagnose Attempt 2 out-of-sample errors before selecting an Attempt 3 design."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from src.config import (
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    ATTEMPT2_PROCESSED_DATA_DIR,
    ATTEMPT2_REPORT_TABLES_DIR,
    ATTEMPT2_REPORTS_DIR,
    INTERIM_DATA_DIR,
    POSITIONS,
    RANDOM_STATE,
)
from src.phase2_features import attempt2_features
from src.phase2_models import FINAL_VARIANT, PREDICTIONS_FILENAME, VARIANTS

DIAGNOSTIC_DIR = ATTEMPT2_REPORT_TABLES_DIR / "diagnostics"
PAIRED_ERRORS_FILENAME = "attempt2_paired_player_errors.parquet"
ABLATION_PAIRED_FILENAME = "attempt2_paired_ablation_diagnostics.csv"
SLICE_FILENAME = "attempt2_error_slices.csv"
SEASON_FILENAME = "attempt2_season_diagnostics.csv"
WORST_FILENAME = "attempt2_largest_final_errors.csv"
REGRESSIONS_FILENAME = "attempt2_largest_regressions_vs_v1.csv"
COLLINEARITY_FILENAME = "attempt2_high_feature_correlations.csv"
CONDITION_FILENAME = "attempt2_design_matrix_condition.csv"
MISSINGNESS_FILENAME = "attempt2_validation_missingness.csv"
STABILITY_FILENAME = "attempt2_final_coefficient_instability.csv"
REPORT_FILENAME = "attempt2_diagnostic_report.txt"


class Attempt2DiagnosticError(RuntimeError):
    """Raised when diagnostics cannot be constructed from comparable OOF rows."""


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    samples: int = 5000,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Return a deterministic 95% bootstrap interval for a mean."""

    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    if groups is None:
        for index in range(samples):
            estimates[index] = rng.choice(values, size=len(values), replace=True).mean()
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        grouped = {group: values[groups == group] for group in unique}
        for index in range(samples):
            selected = rng.choice(unique, size=len(unique), replace=True)
            estimates[index] = np.concatenate([grouped[group] for group in selected]).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _load_modeling_tables() -> pl.DataFrame:
    frames = []
    context_columns = [
        "player_id",
        "prediction_season",
        "position",
        "prediction_team",
        "previous_season_ppg",
        "previous_season_games_played",
        "age_entering_season",
        "years_experience_entering_season",
        "changed_team",
        "competition_pool_size",
        "competition_no_history_count",
        "projected_qb_has_previous_qbr",
        "preseason_offensive_line_rank",
        "preseason_strength_of_schedule_rank",
    ]
    position_role_column = {
        "QB": "previous_passing_attempts_per_game",
        "RB": "previous_opportunities_per_game",
        "WR": "previous_targets_per_game",
        "TE": "previous_targets_per_game",
    }
    for position in POSITIONS:
        table = pl.read_parquet(
            ATTEMPT2_PROCESSED_DATA_DIR
            / f"{position.lower()}_attempt2_modeling_dataset.parquet"
        )
        frames.append(
            table.select(
                *context_columns,
                pl.col(position_role_column[position]).alias("previous_role_volume"),
            )
        )
    return pl.concat(frames)


def build_paired_errors(predictions: pl.DataFrame) -> pl.DataFrame:
    key = ["player_id", "season", "position"]
    baseline = predictions.filter(pl.col("model_name") == "A_v1_same_cohort").select(
        *key,
        "player_name",
        "actual_ppg",
        pl.col("predicted_ppg").alias("v1_predicted_ppg"),
        pl.col("absolute_error").alias("v1_absolute_error"),
        pl.col("error").alias("v1_error"),
    )
    final = predictions.filter(pl.col("model_name") == FINAL_VARIANT).select(
        *key,
        pl.col("predicted_ppg").alias("attempt2_predicted_ppg"),
        pl.col("absolute_error").alias("attempt2_absolute_error"),
        pl.col("error").alias("attempt2_error"),
    )
    paired = baseline.join(final, on=key, how="inner", validate="1:1")
    if paired.height != baseline.height or paired.height != final.height:
        raise Attempt2DiagnosticError("V1 and final Attempt 2 rows are not identical")

    context = _load_modeling_tables().rename({"prediction_season": "season"})
    actual = pl.read_parquet(INTERIM_DATA_DIR / "player_seasons_all.parquet").select(
        "player_id",
        "season",
        "position",
        "games_played",
        "passing_attempts_per_game",
        "opportunities_per_game",
        "targets_per_game",
    ).with_columns(
        pl.when(pl.col("position") == "QB")
        .then(pl.col("passing_attempts_per_game"))
        .when(pl.col("position") == "RB")
        .then(pl.col("opportunities_per_game"))
        .otherwise(pl.col("targets_per_game"))
        .alias("actual_role_volume")
    ).select(
        "player_id", "season", "position", "games_played", "actual_role_volume"
    )
    paired = (
        paired.join(context, on=key, how="left", validate="1:1")
        .join(actual, on=key, how="left", validate="1:1")
        .with_columns(
            (pl.col("v1_absolute_error") - pl.col("attempt2_absolute_error")).alias(
                "attempt2_improvement"
            ),
            (pl.col("actual_role_volume") - pl.col("previous_role_volume")).alias(
                "role_volume_change"
            ),
        )
        .with_columns(
            pl.when(
                ((pl.col("position") == "QB") & (pl.col("role_volume_change") > 5))
                | ((pl.col("position") == "RB") & (pl.col("role_volume_change") > 2))
                | (pl.col("position").is_in(["WR", "TE"]) & (pl.col("role_volume_change") > 1.5))
            )
            .then(pl.lit("expanded"))
            .when(
                ((pl.col("position") == "QB") & (pl.col("role_volume_change") < -5))
                | ((pl.col("position") == "RB") & (pl.col("role_volume_change") < -2))
                | (pl.col("position").is_in(["WR", "TE"]) & (pl.col("role_volume_change") < -1.5))
            )
            .then(pl.lit("collapsed"))
            .otherwise(pl.lit("stable"))
            .alias("realized_role_change"),
            pl.when(pl.col("attempt2_error") > 0)
            .then(pl.lit("overprediction"))
            .otherwise(pl.lit("underprediction"))
            .alias("error_direction"),
        )
    )
    return paired


def paired_ablation_diagnostics(predictions: pl.DataFrame) -> pl.DataFrame:
    key = ["player_id", "season", "position"]
    baseline = predictions.filter(pl.col("model_name") == "A_v1_same_cohort").select(
        *key, pl.col("absolute_error").alias("v1_absolute_error")
    )
    rows = []
    for variant in VARIANTS:
        if variant == "A_v1_same_cohort":
            continue
        joined = predictions.filter(pl.col("model_name") == variant).select(
            *key, pl.col("absolute_error").alias("variant_absolute_error")
        ).join(baseline, on=key, how="inner", validate="1:1").with_columns(
            (pl.col("v1_absolute_error") - pl.col("variant_absolute_error")).alias(
                "improvement"
            )
        )
        for position in POSITIONS:
            group = joined.filter(pl.col("position") == position)
            values = group["improvement"].to_numpy()
            row_low, row_high = bootstrap_mean_interval(values)
            season_low, season_high = bootstrap_mean_interval(
                values, groups=group["season"].to_numpy(), seed=RANDOM_STATE + 1
            )
            rows.append(
                {
                    "variant": variant,
                    "position": position,
                    "rows": group.height,
                    "mean_mae_improvement": float(values.mean()),
                    "median_player_improvement": float(np.median(values)),
                    "players_improved_rate": float((values > 0).mean()),
                    "row_bootstrap_95_low": row_low,
                    "row_bootstrap_95_high": row_high,
                    "season_bootstrap_95_low": season_low,
                    "season_bootstrap_95_high": season_high,
                }
            )
    return pl.DataFrame(rows).sort("position", "mean_mae_improvement", descending=[False, True])


def _bucket_columns(paired: pl.DataFrame) -> pl.DataFrame:
    return paired.with_columns(
        pl.when(pl.col("age_entering_season") < 25).then(pl.lit("under_25"))
        .when(pl.col("age_entering_season") < 28).then(pl.lit("25_to_27"))
        .when(pl.col("age_entering_season") < 31).then(pl.lit("28_to_30"))
        .otherwise(pl.lit("31_plus")).alias("age_bucket"),
        pl.when(pl.col("previous_season_games_played") <= 8).then(pl.lit("4_to_8"))
        .when(pl.col("previous_season_games_played") <= 12).then(pl.lit("9_to_12"))
        .otherwise(pl.lit("13_plus")).alias("prior_games_bucket"),
        pl.when(pl.col("previous_season_ppg") < 5).then(pl.lit("under_5"))
        .when(pl.col("previous_season_ppg") < 10).then(pl.lit("5_to_10"))
        .when(pl.col("previous_season_ppg") < 15).then(pl.lit("10_to_15"))
        .otherwise(pl.lit("15_plus")).alias("prior_ppg_bucket"),
        pl.when(pl.col("competition_pool_size") <= 2).then(pl.lit("small_0_to_2"))
        .when(pl.col("competition_pool_size") <= 4).then(pl.lit("medium_3_to_4"))
        .otherwise(pl.lit("large_5_plus")).alias("competition_bucket"),
        (((pl.col("preseason_offensive_line_rank") - 1) // 8) + 1)
        .cast(pl.String).alias("ol_rank_quartile"),
        (((pl.col("preseason_strength_of_schedule_rank") - 1) // 8) + 1)
        .cast(pl.String).alias("sos_rank_quartile"),
        pl.col("changed_team").cast(pl.String).alias("changed_team_group"),
        pl.col("projected_qb_has_previous_qbr").cast(pl.String).alias("qb_history_group"),
    )


def error_slices(paired: pl.DataFrame) -> pl.DataFrame:
    frame = _bucket_columns(paired)
    slice_columns = [
        "age_bucket",
        "prior_games_bucket",
        "prior_ppg_bucket",
        "competition_bucket",
        "ol_rank_quartile",
        "sos_rank_quartile",
        "changed_team_group",
        "qb_history_group",
        "realized_role_change",
        "error_direction",
    ]
    outputs = []
    for column in slice_columns:
        outputs.append(
            frame.group_by("position", column)
            .agg(
                pl.len().alias("rows"),
                pl.col("v1_absolute_error").mean().alias("v1_mae"),
                pl.col("attempt2_absolute_error").mean().alias("attempt2_mae"),
                pl.col("attempt2_improvement").mean().alias("mean_improvement"),
                (pl.col("attempt2_improvement") > 0).mean().alias("improved_rate"),
                pl.col("attempt2_error").mean().alias("mean_signed_error"),
            )
            .rename({column: "slice_value"})
            .with_columns(pl.lit(column).alias("slice_name"))
            .select(
                "position", "slice_name", "slice_value", "rows", "v1_mae",
                "attempt2_mae", "mean_improvement", "improved_rate", "mean_signed_error",
            )
        )
    return pl.concat(outputs).sort("position", "slice_name", "slice_value")


def season_diagnostics(paired: pl.DataFrame) -> pl.DataFrame:
    return (
        paired.group_by("position", "season")
        .agg(
            pl.len().alias("rows"),
            pl.col("v1_absolute_error").mean().alias("v1_mae"),
            pl.col("attempt2_absolute_error").mean().alias("attempt2_mae"),
            pl.col("attempt2_improvement").mean().alias("mean_improvement"),
            (pl.col("attempt2_improvement") > 0).mean().alias("improved_rate"),
            pl.col("attempt2_error").mean().alias("mean_signed_error"),
            (pl.col("realized_role_change") == "collapsed").mean().alias("role_collapse_rate"),
            (pl.col("realized_role_change") == "expanded").mean().alias("role_expansion_rate"),
        )
        .sort("position", "season")
    )


def collinearity_diagnostics() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    pair_rows = []
    condition_rows = []
    missing_rows = []
    for position in POSITIONS:
        table = pl.read_parquet(
            ATTEMPT2_PROCESSED_DATA_DIR
            / f"{position.lower()}_attempt2_modeling_dataset.parquet"
        ).filter(
            (pl.col("prediction_season") >= 2012)
            & pl.col("previous_season_ppg").is_not_null()
        )
        features = attempt2_features(position, VARIANTS[FINAL_VARIANT])
        numeric = table.select(features)
        for feature in features:
            missing_rows.append(
                {
                    "position": position,
                    "feature": feature,
                    "validation_rows": table.height,
                    "missing_count": numeric[feature].null_count(),
                    "missing_rate": numeric[feature].null_count() / table.height,
                }
            )
        medians = numeric.select(pl.all().median()).row(0)
        filled = numeric.with_columns(
            *[
                pl.col(feature).fill_null(float(medians[index]))
                for index, feature in enumerate(features)
            ]
        )
        matrix = filled.to_numpy().astype(float)
        standard_deviation = matrix.std(axis=0)
        safe = standard_deviation > 1e-12
        standardized = (matrix[:, safe] - matrix[:, safe].mean(axis=0)) / standard_deviation[safe]
        condition_rows.append(
            {
                "position": position,
                "rows": table.height,
                "features": len(features),
                "nonconstant_features": int(safe.sum()),
                "matrix_rank": int(np.linalg.matrix_rank(standardized)),
                "rank_deficiency": int(
                    standardized.shape[1] - np.linalg.matrix_rank(standardized)
                ),
                "condition_number": float(np.linalg.cond(standardized)),
            }
        )
        correlation = np.corrcoef(matrix[:, safe], rowvar=False)
        safe_features = [feature for feature, keep in zip(features, safe, strict=True) if keep]
        for left in range(len(safe_features)):
            for right in range(left + 1, len(safe_features)):
                value = float(correlation[left, right])
                if abs(value) >= 0.75:
                    pair_rows.append(
                        {
                            "position": position,
                            "feature_1": safe_features[left],
                            "feature_2": safe_features[right],
                            "correlation": value,
                            "absolute_correlation": abs(value),
                        }
                    )
    pairs = pl.DataFrame(pair_rows).sort("absolute_correlation", descending=True)
    return pairs, pl.DataFrame(condition_rows), pl.DataFrame(missing_rows)


def _fmt(value: float) -> str:
    return f"{value:+.4f}"


def write_diagnostic_report(
    paired: pl.DataFrame,
    ablations: pl.DataFrame,
    slices: pl.DataFrame,
    seasons: pl.DataFrame,
    correlations: pl.DataFrame,
    conditions: pl.DataFrame,
    instability: pl.DataFrame,
) -> Path:
    lines = [
        "ATTEMPT 2 DIAGNOSTIC REPORT",
        "",
        "SCOPE",
        "All error comparisons use paired out-of-sample predictions from the same",
        "player-season cohort. Positive improvement means lower error than V1.",
        "",
        "1. PAIRED FULL-MODEL RESULTS",
        "",
    ]
    for position in POSITIONS:
        group = paired.filter(pl.col("position") == position)
        values = group["attempt2_improvement"].to_numpy()
        row_ci = bootstrap_mean_interval(values)
        season_ci = bootstrap_mean_interval(
            values, groups=group["season"].to_numpy(), seed=RANDOM_STATE + 1
        )
        lines.extend(
            [
                position,
                f"  Rows: {group.height}",
                f"  V1 MAE: {group['v1_absolute_error'].mean():.4f}",
                f"  Full Attempt 2 MAE: {group['attempt2_absolute_error'].mean():.4f}",
                f"  Mean improvement: {values.mean():+.4f}",
                f"  Median player improvement: {np.median(values):+.4f}",
                f"  Players improved: {(values > 0).mean():.1%}",
                f"  Row-bootstrap 95% interval: [{row_ci[0]:+.4f}, {row_ci[1]:+.4f}]",
                f"  Season-block 95% interval: [{season_ci[0]:+.4f}, {season_ci[1]:+.4f}]",
                "",
            ]
        )

    lines.extend(["2. PRINCIPAL FINDINGS", ""])
    for position in POSITIONS:
        pos_slices = slices.filter(pl.col("position") == position)
        role = pos_slices.filter(pl.col("slice_name") == "realized_role_change")
        worst_role = role.sort("mean_improvement").head(1)
        season = (
            seasons.filter(pl.col("position") == position)
            .sort("mean_improvement")
            .head(1)
        )
        best_ablation = ablations.filter(pl.col("position") == position).sort(
            "mean_mae_improvement", descending=True
        ).row(0, named=True)
        lines.extend(
            [
                f"{position}:",
                f"  Best tested feature specification: {best_ablation['variant']}",
                f"  Paired improvement: {_fmt(best_ablation['mean_mae_improvement'])}",
                (
                    f"  Worst realized-role group: {worst_role['slice_value'].item()} "
                    f"({_fmt(worst_role['mean_improvement'].item())})"
                ),
                (
                    f"  Worst season: {season['season'].item()} "
                    f"({_fmt(season['mean_improvement'].item())})"
                ),
                "",
            ]
        )

    lines.extend(["3. MODEL-STABILITY FINDINGS", ""])
    for row in conditions.sort("position").to_dicts():
        sign_changes = instability.filter(pl.col("position") == row["position"])[
            "sign_changed_across_folds"
        ].sum()
        lines.append(
            f"{row['position']}: condition number {row['condition_number']:.1f}; "
            f"rank deficiency {row['rank_deficiency']}; "
            f"{sign_changes} coefficients changed sign across folds."
        )
    lines.extend(
        [
            f"Highly correlated feature pairs (|r| >= 0.75): {correlations.height}",
            "",
            "Large condition numbers, opposing coefficients on similar opportunity",
            "features, and fold sign changes indicate that ordinary least squares is",
            "trying to divide credit among redundant predictors. This can suppress",
            "otherwise useful signal and makes individual coefficients unreliable.",
            "The RB matrix contains the exact relationship opportunities per game =",
            "carries per game + targets per game. WR and TE also include both sides of",
            "complementary WR/TE dummy encodings. One column from each exact dependency",
            "must be removed even if Ridge or another regularized model is introduced.",
            "",
            "4. DIAGNOSIS",
            "",
            "The primary limitation is contextual role uncertainty, followed by model",
            "instability. Team-level variables describe the environment but do not say",
            "how season-t touches, routes, or starts will be allocated. The full model",
            "therefore helps where team context is strongly connected to output (most",
            "clearly QB), but adds weak or redundant signals for RB/WR and harmful noise",
            "for TE. Small RB/WR aggregate gains should be treated as uncertain unless",
            "their season-block intervals exclude zero.",
            "The role slices show a further asymmetry: the new context helps on realized",
            "role collapses, but does not improve RB or WR role expansions. This points",
            "directly to missing forward-looking role allocation rather than missing",
            "generic team quality.",
            "",
            "5. REQUIRED ACTIONS BEFORE ATTEMPT 3",
            "",
            "1. Use position-specific winning specifications; do not force one full model.",
            "2. Compare reduced OLS with Ridge and Elastic Net inside every time fold.",
            "3. Remove or combine highly correlated opportunity and efficiency variables.",
            "4. Add historical preseason ADP/projections as a standalone baseline first.",
            "5. Add explicit depth-chart rank and expected role information.",
            "6. Add prior snap share, route participation, and red-zone opportunity.",
            "7. Add preseason injury, coaching/play-caller change, and vacated opportunity.",
            "8. Prefer raw OL/SOS prediction scores and uncertainty over ordinal ranks.",
            "9. Build a separate rookie/no-history evaluation rather than median-imputing it.",
            "",
            "The supporting tables identify the exact players, seasons, contexts, feature",
            "pairs, and missing fields behind these conclusions.",
            "",
        ]
    )
    output = ATTEMPT2_REPORTS_DIR / REPORT_FILENAME
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run_attempt2_diagnostics() -> dict[str, Path]:
    predictions = pl.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / PREDICTIONS_FILENAME
    )
    paired = build_paired_errors(predictions)
    ablations = paired_ablation_diagnostics(predictions)
    slices = error_slices(paired)
    seasons = season_diagnostics(paired)
    correlations, conditions, missingness = collinearity_diagnostics()
    stability = pl.read_csv(
        ATTEMPT2_REPORT_TABLES_DIR / "attempt2_coefficient_stability.csv"
    ).filter(pl.col("variant") == FINAL_VARIANT)

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    paired.write_parquet(DIAGNOSTIC_DIR / PAIRED_ERRORS_FILENAME, use_pyarrow=True)
    ablations.write_csv(DIAGNOSTIC_DIR / ABLATION_PAIRED_FILENAME)
    slices.write_csv(DIAGNOSTIC_DIR / SLICE_FILENAME)
    seasons.write_csv(DIAGNOSTIC_DIR / SEASON_FILENAME)
    paired.sort("attempt2_absolute_error", descending=True).head(100).write_csv(
        DIAGNOSTIC_DIR / WORST_FILENAME
    )
    paired.sort("attempt2_improvement").head(100).write_csv(
        DIAGNOSTIC_DIR / REGRESSIONS_FILENAME
    )
    correlations.write_csv(DIAGNOSTIC_DIR / COLLINEARITY_FILENAME)
    conditions.write_csv(DIAGNOSTIC_DIR / CONDITION_FILENAME)
    missingness.write_csv(DIAGNOSTIC_DIR / MISSINGNESS_FILENAME)
    stability.write_csv(DIAGNOSTIC_DIR / STABILITY_FILENAME)
    report = write_diagnostic_report(
        paired, ablations, slices, seasons, correlations, conditions, stability
    )
    manifest = {
        "paired_rows": paired.height,
        "positions": list(POSITIONS),
        "variants": list(VARIANTS),
        "bootstrap_samples": 5000,
        "positive_improvement_definition": "V1 absolute error - variant absolute error",
        "outputs": sorted(path.name for path in DIAGNOSTIC_DIR.iterdir()),
    }
    manifest_path = DIAGNOSTIC_DIR / "attempt2_diagnostic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"report": report, "manifest": manifest_path}
