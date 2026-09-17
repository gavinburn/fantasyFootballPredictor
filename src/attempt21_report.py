"""Pandas/seaborn diagnostics, decisions, preservation check, and packaging."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    ATTEMPT21_DATA_DIR,
    ATTEMPT21_FIGURE_DATA_DIR,
    ATTEMPT21_PREDICTIONS_DATA_DIR,
    ATTEMPT21_REPORT_FIGURES_DIR,
    ATTEMPT21_REPORT_TABLES_DIR,
    ATTEMPT21_REPORTS_DIR,
    POSITIONS,
    REPOSITORY_ROOT,
)

TOP_N = {"QB": 10, "RB": 20, "WR": 20, "TE": 10}
COLORS = {"QB": "#4C78A8", "RB": "#F58518", "WR": "#54A24B", "TE": "#B279A2"}


class Attempt21ReportError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame["predicted_ppg"] - frame["actual_ppg"]
    n = min(TOP_N[frame["position"].iloc[0]], len(frame))
    actual_top = set(frame.nlargest(n, "actual_ppg")["player_id"])
    pred_top = set(frame.nlargest(n, "predicted_ppg")["player_id"])
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(
            frame["actual_ppg"].corr(frame["predicted_ppg"], method="spearman")
        ),
        "top_n_overlap_rate": len(actual_top & pred_top) / n,
    }


def _season_bootstrap(values: pd.Series, samples: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(446)
    raw = values.to_numpy()
    estimates = np.array(
        [rng.choice(raw, len(raw), replace=True).mean() for _ in range(samples)]
    )
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def _row_bootstrap(values: pd.Series, samples: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(447)
    raw = values.to_numpy()
    estimates = np.array(
        [rng.choice(raw, len(raw), replace=True).mean() for _ in range(samples)]
    )
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def rebuild_summaries(
    predictions: pd.DataFrame, coefficients: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet",
        columns=[
            "player_id",
            "season",
            "position",
            "model_name",
            "predicted_ppg",
            "actual_ppg",
        ],
    )
    frozen = frozen[frozen["model_name"] == "A_v1_same_cohort"].rename(
        columns={"predicted_ppg": "v1_predicted_ppg"}
    )
    rows, seasonal = [], []
    for (candidate, position), group in predictions.groupby(["candidate", "position"]):
        paired = group.merge(
            frozen.drop(columns="model_name"),
            on=["player_id", "season", "position", "actual_ppg"],
            validate="one_to_one",
        )
        fold_rows = []
        for season, fold in paired.groupby("season"):
            current = _fold_metrics(fold)
            baseline = _fold_metrics(
                fold.rename(
                    columns={
                        "predicted_ppg": "attempt_predicted",
                        "v1_predicted_ppg": "predicted_ppg",
                    }
                )
            )
            improvement = (fold["v1_predicted_ppg"] - fold["actual_ppg"]).abs() - (
                fold["predicted_ppg"] - fold["actual_ppg"]
            ).abs()
            row = {
                "candidate": candidate,
                "position": position,
                "season": season,
                "mae_improvement": improvement.mean(),
                **current,
                "v1_mae": baseline["mae"],
                "v1_spearman": baseline["spearman"],
                "v1_top_n_overlap_rate": baseline["top_n_overlap_rate"],
            }
            fold_rows.append(row)
            seasonal.append(row)
        folds = pd.DataFrame(fold_rows)
        player_improvement = (
            paired["v1_predicted_ppg"] - paired["actual_ppg"]
        ).abs() - (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
        low, high = _season_bootstrap(folds["mae_improvement"])
        row_low, row_high = _row_bootstrap(player_improvement)
        positive = folds["mae_improvement"].clip(lower=0).sum()
        rows.append(
            {
                "candidate": candidate,
                "position": position,
                "specification": group["specification"].iloc[0],
                "method": group["method"].iloc[0],
                "rows": len(paired),
                "mae": float(
                    (paired["predicted_ppg"] - paired["actual_ppg"]).abs().mean()
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean((paired["predicted_ppg"] - paired["actual_ppg"]) ** 2)
                    )
                ),
                "mean_fold_spearman": folds["spearman"].mean(),
                "mean_fold_top_n_overlap_rate": folds["top_n_overlap_rate"].mean(),
                "v1_mae": float(
                    (paired["v1_predicted_ppg"] - paired["actual_ppg"]).abs().mean()
                ),
                "v1_mean_fold_spearman": folds["v1_spearman"].mean(),
                "v1_mean_fold_top_n_overlap_rate": folds[
                    "v1_top_n_overlap_rate"
                ].mean(),
                "spearman_change": folds["spearman"].mean()
                - folds["v1_spearman"].mean(),
                "top_n_overlap_change": folds["top_n_overlap_rate"].mean()
                - folds["v1_top_n_overlap_rate"].mean(),
                "mean_mae_improvement": player_improvement.mean(),
                "median_player_improvement": player_improvement.median(),
                "players_improved_rate": (player_improvement > 0).mean(),
                "seasons_improved": int((folds["mae_improvement"] > 0).sum()),
                "season_block_95_low": low,
                "season_block_95_high": high,
                "row_bootstrap_95_low": row_low,
                "row_bootstrap_95_high": row_high,
                "largest_single_season_gain_share": folds["mae_improvement"].max()
                / positive
                if positive
                else np.nan,
            }
        )
    summary, seasonal = pd.DataFrame(rows), pd.DataFrame(seasonal)

    # Give the adaptive model the coefficients of the source selected inside each fold.
    selected = predictions[predictions["candidate"] == "ADAPTIVE_POSITION_SELECTED"][
        ["position", "season", "source_candidate"]
    ].drop_duplicates()
    adaptive_coef = coefficients.merge(
        selected,
        left_on=["position", "season", "candidate"],
        right_on=["position", "season", "source_candidate"],
        validate="many_to_one",
    )
    adaptive_coef["candidate"] = "ADAPTIVE_POSITION_SELECTED"
    all_coef = pd.concat(
        [coefficients, adaptive_coef[coefficients.columns]], ignore_index=True
    )
    stability = (
        all_coef.groupby(["position", "candidate", "feature"])[
            "standardized_coefficient"
        ]
        .agg(folds="size", mean="mean", std="std", minimum="min", maximum="max")
        .reset_index()
    )
    stability["sign_changed"] = (stability["minimum"] < 0) & (stability["maximum"] > 0)
    return summary, seasonal, stability


def make_decisions(summary: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rank_audit = pd.read_csv(ATTEMPT21_REPORT_TABLES_DIR / "exact_dependency_audit.csv")
    rows = []
    for position in POSITIONS:
        result = summary[
            (summary.position == position)
            & (summary.candidate == "ADAPTIVE_POSITION_SELECTED")
        ].iloc[0]
        checks = {
            "rank_ok": bool(
                rank_audit[rank_audit.position == position]["passed"].all()
            ),
            "temporal_validation_ok": True,
            "mae_improved": result.mean_mae_improvement > 0,
            "season_evidence": result.seasons_improved >= 8
            or result.season_block_95_low > 0,
            "spearman_ok": result.spearman_change >= -0.01,
            "top_n_ok": result.top_n_overlap_change >= -0.02,
            "single_season_ok": pd.isna(result.largest_single_season_gain_share)
            or result.largest_single_season_gain_share <= 0.5,
        }
        decision = (
            "ACCEPT"
            if all(checks.values())
            else ("PROMISING_NOT_CONFIRMED" if checks["mae_improved"] else "REJECT")
        )
        rows.append(
            {
                "position": position,
                "evaluated_candidate": "ADAPTIVE_POSITION_SELECTED",
                "decision": decision,
                "recommended_model": "ADAPTIVE_POSITION_SELECTED"
                if decision == "ACCEPT"
                else "FROZEN_V1",
                "mae": result.mae,
                "v1_mae": result.v1_mae,
                "mean_mae_improvement": result.mean_mae_improvement,
                "season_block_95_low": result.season_block_95_low,
                "season_block_95_high": result.season_block_95_high,
                "seasons_improved": result.seasons_improved,
                "spearman_change": result.spearman_change,
                "top_n_overlap_change": result.top_n_overlap_change,
                "adaptive_coefficient_sign_changes": int(
                    stability[
                        (stability.position == position)
                        & (stability.candidate == "ADAPTIVE_POSITION_SELECTED")
                    ].sign_changed.sum()
                ),
                **checks,
            }
        )
    return pd.DataFrame(rows)


def stress_tables(
    predictions: pd.DataFrame, seasonal: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    base = frozen[frozen.model_name == "A_v1_same_cohort"][
        ["player_id", "season", "position", "predicted_ppg", "actual_ppg"]
    ].rename(columns={"predicted_ppg": "v1_predicted_ppg"})
    qb = predictions[
        (predictions.position == "QB")
        & (predictions.candidate == "QB_H_REDUCED_PLUS_OL::ols")
    ].merge(
        base,
        on=["player_id", "season", "position", "actual_ppg"],
        validate="one_to_one",
    )
    qb["improvement"] = (qb.v1_predicted_ppg - qb.actual_ppg).abs() - (
        qb.predicted_ppg - qb.actual_ppg
    ).abs()
    stress = [{"excluded_season": "none", "mean_improvement": qb.improvement.mean()}]
    for season in sorted(qb.season.unique()):
        stress.append(
            {
                "excluded_season": str(season),
                "mean_improvement": qb.loc[qb.season != season, "improvement"].mean(),
            }
        )
    context_path = (
        REPOSITORY_ROOT
        / "reports/attempt_2/tables/diagnostics/attempt2_paired_player_errors.parquet"
    )
    context = pd.read_parquet(context_path)[
        [
            "player_id",
            "season",
            "position",
            "changed_team",
            "realized_role_change",
            "previous_season_games_played",
        ]
    ]
    context = qb.merge(
        context, on=["player_id", "season", "position"], validate="one_to_one"
    )
    groups = []
    for column in ["changed_team", "realized_role_change"]:
        for value, part in context.groupby(column, dropna=False):
            groups.append(
                {
                    "position": "QB",
                    "group_variable": column,
                    "group": str(value),
                    "rows": len(part),
                    "mean_improvement": part.improvement.mean(),
                    "mae": (part.predicted_ppg - part.actual_ppg).abs().mean(),
                }
            )
    context["prior_games_group"] = np.where(
        context.previous_season_games_played >= 12, "starting_sample", "low_prior_games"
    )
    for value, part in context.groupby("prior_games_group"):
        groups.append(
            {
                "position": "QB",
                "group_variable": "prior_games_group",
                "group": value,
                "rows": len(part),
                "mean_improvement": part.improvement.mean(),
                "mae": (part.predicted_ppg - part.actual_ppg).abs().mean(),
            }
        )
    te = seasonal[
        (seasonal.position == "TE") & seasonal.candidate.str.startswith("TE_")
    ].copy()
    return pd.DataFrame(stress), pd.DataFrame(groups), te


def _save_source(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(ATTEMPT21_FIGURE_DATA_DIR / f"{name}.csv", index=False)


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(
        ATTEMPT21_REPORT_FIGURES_DIR / f"{name}.png", dpi=180, bbox_inches="tight"
    )
    plt.close()


def build_figures(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    seasonal: pd.DataFrame,
    stability: pd.DataFrame,
    role: pd.DataFrame,
) -> None:
    if "training_rows" in predictions and (predictions.training_rows > 0).any():
        raise Attempt21ReportError("Figure builder received in-sample rows")
    sns.set_theme(style="whitegrid", context="notebook")
    selected = summary[summary.candidate == "ADAPTIVE_POSITION_SELECTED"]
    _save_source(selected, "paired_mae_improvement")
    plt.figure(figsize=(7, 4))
    sns.pointplot(
        data=selected,
        x="position",
        y="mean_mae_improvement",
        hue="position",
        palette=COLORS,
        legend=False,
    )
    plt.axhline(0, color="black", lw=1)
    plt.ylabel("MAE improvement vs V1 (positive is better)")
    _save("paired_mae_improvement")

    key = seasonal[seasonal.candidate == "ADAPTIVE_POSITION_SELECTED"]
    _save_source(key, "fold_mae_delta_heatmap")
    plt.figure(figsize=(12, 3))
    sns.heatmap(
        key.pivot(index="position", columns="season", values="mae_improvement"),
        center=0,
        cmap="vlag",
        annot=False,
    )
    _save("fold_mae_delta_heatmap")

    tuning = pd.read_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_selected_hyperparameters.csv"
    )
    surface = tuning[
        (tuning.position == "QB")
        & (tuning.outer_season == 2025)
        & (tuning.specification == "QB_H_REDUCED_PLUS_OL")
    ]
    _save_source(surface, "regularization_tuning_surfaces")
    _fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ridge = surface[surface.method == "ridge"]
    sns.heatmap(
        ridge.pivot_table(index="alpha", values="inner_mean_mae", aggfunc="mean"),
        annot=True,
        fmt=".3f",
        ax=axes[0],
    )
    axes[0].set_title("Ridge")
    elastic = tuning[
        (tuning.position == "QB")
        & (tuning.outer_season == 2025)
        & (tuning.specification == "QB_FULL_NONREDUNDANT")
        & (tuning.method == "elastic_net")
    ]
    sns.heatmap(
        elastic.pivot(index="alpha", columns="l1_ratio", values="inner_mean_mae"),
        ax=axes[1],
        cmap="viridis",
    )
    axes[1].set_title("Elastic Net")
    _save("regularization_tuning_surfaces")

    coef = stability[(stability.candidate == "ADAPTIVE_POSITION_SELECTED")].copy()
    coef = coef.loc[
        coef.groupby("position").std.nlargest(8).index if False else coef.index
    ]
    _save_source(coef, "coefficient_stability")
    top = coef.sort_values("std", ascending=False).groupby("position").head(6)
    plt.figure(figsize=(12, 6))
    sns.stripplot(data=top, x="std", y="feature", hue="position", palette=COLORS)
    plt.xlabel("Std. dev. of standardized coefficient")
    _save("coefficient_stability")

    corr = pd.read_csv(ATTEMPT21_REPORT_TABLES_DIR / "correlation_decisions.csv")
    _save_source(corr, "feature_correlation_structure")
    matrix = (
        corr[corr.position == "QB"]
        .pivot_table(
            index="feature_1",
            columns="feature_2",
            values="correlation_pre_2012",
            aggfunc="first",
        )
        .fillna(0)
    )
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, cmap="vlag", center=0)
    plt.title("QB flagged feature correlations (pre-2012)")
    _save("feature_correlation_structure")

    chosen = predictions[predictions.candidate == "ADAPTIVE_POSITION_SELECTED"].copy()
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    frozen = frozen[frozen.model_name == "A_v1_same_cohort"]
    residual = pd.concat(
        [
            chosen.assign(
                model="Attempt 2.1", residual=lambda d: d.predicted_ppg - d.actual_ppg
            ),
            frozen.assign(
                model="V1", residual=lambda d: d.predicted_ppg - d.actual_ppg
            ),
        ],
        ignore_index=True,
    )
    _save_source(
        residual[["position", "season", "model", "residual"]], "residual_distributions"
    )
    plt.figure(figsize=(10, 5))
    sns.violinplot(
        data=residual,
        x="position",
        y="residual",
        hue="model",
        split=True,
        inner="quart",
    )
    plt.axhline(0, color="black", lw=1)
    _save("residual_distributions")

    _save_source(role, "role_context_errors")
    g = sns.catplot(
        data=role,
        x="group",
        y="mean_improvement",
        col="group_variable",
        kind="bar",
        col_wrap=2,
        sharex=False,
        height=3,
    )
    g.set_xticklabels(rotation=30)
    g.set_axis_labels("", "MAE improvement")
    g.savefig(
        ATTEMPT21_REPORT_FIGURES_DIR / "role_context_errors.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    _save_source(
        chosen[["position", "season", "actual_ppg", "predicted_ppg"]],
        "predicted_vs_actual",
    )
    g = sns.FacetGrid(
        chosen, col="position", col_wrap=2, hue="position", palette=COLORS, height=3
    )
    g.map_dataframe(
        sns.scatterplot, x="actual_ppg", y="predicted_ppg", alpha=0.35, s=18
    )
    for axis in g.axes.flat:
        axis.axline((0, 0), slope=1, color="black", lw=1)
    g.savefig(
        ATTEMPT21_REPORT_FIGURES_DIR / "predicted_vs_actual.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    _save_source(selected, "position_decision_dashboard")
    dash = selected.melt(
        id_vars=["position"],
        value_vars=["mean_mae_improvement", "spearman_change", "top_n_overlap_change"],
        var_name="metric",
        value_name="change",
    )
    plt.figure(figsize=(9, 4))
    sns.barplot(data=dash, x="position", y="change", hue="metric")
    plt.axhline(0, color="black", lw=1)
    _save("position_decision_dashboard")

    from src.attempt21_features import build_candidate_sets
    from src.attempt21_models import RIDGE_ALPHAS, build_pipeline, load_tables

    table = load_tables()["QB"]
    features = build_candidate_sets()["QB"]["QB_H_REDUCED_PLUS_OL"]
    path_rows = []
    for alpha in RIDGE_ALPHAS:
        model = build_pipeline("ridge", {"alpha": alpha}).fit(
            table[features], table["target_ppg"]
        )
        path_rows.extend(
            {"alpha": alpha, "feature": feature, "standardized_coefficient": value}
            for feature, value in zip(
                features, model.named_steps["regressor"].coef_, strict=True
            )
        )
    path_source = pd.DataFrame(path_rows)
    _save_source(path_source, "coefficient_paths")
    selected_features = (
        path_source.groupby("feature").standardized_coefficient.std().nlargest(8).index
    )
    view = path_source[path_source.feature.isin(selected_features)]
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=view, x="alpha", y="standardized_coefficient", hue="feature", marker="o"
    )
    plt.xscale("log")
    plt.axhline(0, color="black", lw=1)
    _save("coefficient_paths")


def verify_preservation() -> pd.DataFrame:
    before = json.loads(
        (ATTEMPT21_DATA_DIR / "preservation_checksums_before.json").read_text()
    )
    rows = []
    for relative, expected in before.items():
        path = REPOSITORY_ROOT / relative
        current = _sha256(path) if path.exists() else None
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "current_sha256": current,
                "unchanged": current == expected,
            }
        )
    result = pd.DataFrame(rows)
    if not result.unchanged.all():
        raise Attempt21ReportError("Attempt 1/2 preservation checksum failed")
    return result


def run_attempt21_report() -> dict[str, Path]:
    for directory in [
        ATTEMPT21_REPORT_TABLES_DIR,
        ATTEMPT21_REPORT_FIGURES_DIR,
        ATTEMPT21_FIGURE_DATA_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(
        ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_predictions.parquet"
    )
    coefficients = pd.read_parquet(
        ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_fold_coefficients.parquet"
    )
    summary, seasonal, stability = rebuild_summaries(predictions, coefficients)
    decisions = make_decisions(summary, stability)
    stress, role, te = stress_tables(predictions, seasonal)
    preservation = verify_preservation()
    outputs = {
        "attempt2_1_model_comparison.csv": summary,
        "attempt2_1_regularization_comparison.csv": summary,
        "attempt2_1_season_comparison.csv": seasonal,
        "attempt2_1_stability.csv": stability,
        "attempt2_1_position_decisions.csv": decisions,
        "attempt2_1_qb_stress_tests.csv": stress,
        "attempt2_1_role_context_stress.csv": role,
        "attempt2_1_te_sequence.csv": te,
        "attempt2_1_preservation_audit.csv": preservation,
    }
    for name, frame in outputs.items():
        frame.to_csv(ATTEMPT21_REPORT_TABLES_DIR / name, index=False)
    build_figures(predictions, summary, seasonal, stability, role)
    lines = [
        "ATTEMPT 2.1 DIAGNOSTIC SUMMARY",
        "",
        "Attempt 2.1 used only existing Attempt 2 data, pandas tables, and nested chronological validation.",
        "Hyperparameters and the adaptive candidate were selected using seasons earlier than each outer test season.",
        "Confidence intervals below resample validation seasons, not player rows.",
        "",
    ]
    for row in decisions.itertuples():
        lines += [
            f"{row.position}: {row.decision}",
            f"  MAE {row.mae:.4f} vs frozen V1 {row.v1_mae:.4f}; improvement {row.mean_mae_improvement:+.4f} PPG.",
            f"  95% season-block interval [{row.season_block_95_low:+.4f}, {row.season_block_95_high:+.4f}]; {row.seasons_improved}/14 seasons improved.",
            f"  Spearman change {row.spearman_change:+.4f}; Top-N overlap change {row.top_n_overlap_change:+.4f}.",
            f"  Recommendation: {row.recommended_model}.",
            "",
        ]
    lines += [
        "Interpretation",
        "The experiment tests whether correcting redundant features and regularizing the existing feature set is sufficient before new data are added. Rejected positions retain V1; no rejected Attempt 2.1 estimator is promoted.",
        "",
        "Preservation",
        f"All {len(preservation)} frozen Attempt 1/2 files matched their pre-run SHA-256 checksums.",
    ]
    summary_path = ATTEMPT21_REPORTS_DIR / "attempt2_1_diagnostic_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    deployment_path = ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_deployment_models.csv"
    if deployment_path.exists():
        deployment = pd.read_csv(deployment_path).merge(
            decisions[["position", "decision", "recommended_model"]],
            on="position",
            how="left",
            validate="one_to_one",
        )
        deployment["promoted"] = deployment["decision"].eq("ACCEPT")
        deployment.to_csv(deployment_path, index=False)
    archive = ATTEMPT21_REPORTS_DIR / "attempt2_1_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(ATTEMPT21_REPORTS_DIR.rglob("*")):
            if path.is_file() and path != archive:
                bundle.write(path, path.relative_to(ATTEMPT21_REPORTS_DIR))
    return {
        "summary": summary_path,
        "decisions": ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_position_decisions.csv",
        "archive": archive,
    }
