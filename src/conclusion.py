"""Write the evidence-backed conclusion for the fixed V1 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.build_features import POSITIONS
from src.config import REPORT_TABLES_DIR, REPORTS_DIR
from src.train_models import MODEL_NAME


class ConclusionError(RuntimeError):
    """Raised when required Phase 9 or 10 evidence is unavailable."""


def _model_row(comparison: pl.DataFrame, position: str, model: str) -> dict:
    rows = comparison.filter(
        (pl.col("position") == position) & (pl.col("model_name") == model)
    )
    if rows.height != 1:
        raise ConclusionError(
            f"Expected one {position}/{model} comparison row, found {rows.height}"
        )
    return rows.row(0, named=True)


def write_first_attempt_conclusion(
    *,
    comparison_path: Path = REPORT_TABLES_DIR / "v1_model_comparison.csv",
    stability_path: Path = REPORT_TABLES_DIR / "v1_coefficient_stability.csv",
    absolute_errors_path: Path = REPORT_TABLES_DIR / "v1_largest_absolute_errors.csv",
    output_path: Path = REPORTS_DIR / "v1_first_attempt_conclusion.md",
) -> Path:
    required = [comparison_path, stability_path, absolute_errors_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ConclusionError("Missing evaluation evidence: " + ", ".join(missing))

    comparison = pl.read_csv(comparison_path)
    stability = pl.read_csv(stability_path)
    absolute_errors = pl.read_csv(absolute_errors_path)
    linear_rows = comparison.filter(pl.col("model_name") == MODEL_NAME)
    strongest = linear_rows.sort(
        "percentage_MAE_improvement_over_previous_PPG_baseline",
        descending=True,
    ).row(0, named=True)
    weakest = linear_rows.sort(
        "percentage_MAE_improvement_over_previous_PPG_baseline"
    ).row(0, named=True)

    lines = [
        "# V1 First-Attempt Conclusion",
        "",
        "## Scope",
        "",
        (
            "This report evaluates four fixed position-specific linear regressions "
            "using rolling out-of-sample predictions from 2012–2025. It does not "
            "change features, train another model type, or generate 2026 predictions."
        ),
        "",
        "## Required Questions",
        "",
        "### 1. Did each model beat the previous-season PPG baseline?",
        "",
    ]
    for position in POSITIONS:
        row = _model_row(comparison, position, MODEL_NAME)
        answer = "Yes" if row["overall_MAE"] < row["baseline_MAE"] else "No"
        lines.append(
            f"- **{position}: {answer}.** MAE {row['overall_MAE']:.3f} versus "
            f"{row['baseline_MAE']:.3f} PPG."
        )
    lines.extend(["", "### 2. By how much did MAE improve or worsen?", ""])
    for position in POSITIONS:
        row = _model_row(comparison, position, MODEL_NAME)
        lines.append(
            f"- **{position}:** {row['MAE_improvement_over_previous_PPG_baseline']:.3f} "
            f"PPG ({row['percentage_MAE_improvement_over_previous_PPG_baseline']:.2f}%)."
        )
    lines.extend(["", "### 3. Did ranking accuracy improve?", ""])
    for position in POSITIONS:
        linear = _model_row(comparison, position, MODEL_NAME)
        baseline = _model_row(comparison, position, "previous_season_ppg")
        relation = (
            "improved"
            if linear["overall_Spearman_rank_correlation"]
            > baseline["overall_Spearman_rank_correlation"]
            else "did not improve"
        )
        lines.append(
            f"- **{position}:** {relation}; Spearman "
            f"{linear['overall_Spearman_rank_correlation']:.3f} versus "
            f"{baseline['overall_Spearman_rank_correlation']:.3f}, with mean "
            f"Top-N overlap {linear['mean_top_N_overlap_percentage']:.1f}% versus "
            f"{baseline['mean_top_N_overlap_percentage']:.1f}%."
        )
    lines.extend(["", "### 4. Was performance consistent across seasons?", ""])
    for position in POSITIONS:
        row = _model_row(comparison, position, MODEL_NAME)
        lines.append(
            f"- **{position}:** {row['seasons_beating_previous_PPG_baseline']} wins, "
            f"{row['seasons_tied_with_previous_PPG_baseline']} ties, and "
            f"{row['seasons_losing_to_previous_PPG_baseline']} losses across "
            f"{row['number_of_validation_seasons']} seasons."
        )
    lines.extend(
        [
            "",
            "### 5–6. Strongest and weakest positions",
            "",
            (
                f"**{strongest['position']} was strongest by percentage MAE "
                f"improvement ({strongest['percentage_MAE_improvement_over_previous_PPG_baseline']:.2f}%). "
                f"{weakest['position']} was weakest "
                f"({weakest['percentage_MAE_improvement_over_previous_PPG_baseline']:.2f}%).**"
            ),
            "",
            "### 7. Were there unusually large errors?",
            "",
        ]
    )
    for position in POSITIONS:
        maximum = absolute_errors.filter(pl.col("position") == position)[
            "absolute_error"
        ].max()
        lines.append(f"- **{position}:** largest absolute miss was {maximum:.3f} PPG.")
    lines.extend(["", "### 8. Were coefficient estimates stable?", ""])
    for position in POSITIONS:
        rows = stability.filter(pl.col("position") == position)
        unstable = rows.filter(pl.col("unstable_coefficient")).height
        lines.append(
            f"- **{position}:** {unstable} of {rows.height} coefficients met the "
            "predefined instability flag."
        )
    lines.extend(
        [
            "",
            "### 9. Did correlated features complicate interpretation?",
            "",
            (
                "Yes. Every position has strongly correlated V1 inputs, including "
                "previous PPG versus historical PPG or receiving/rushing volume, "
                "and age versus experience. Prediction can remain useful while "
                "individual coefficient meanings become unstable."
            ),
            "",
            "### 10. What limitations were discovered?",
            "",
            "- Exact previous-season history is required for the common comparison cohort.",
            "- Historical changed-team status is unavailable.",
            "- The models tend to underpredict the highest observed PPG ranges.",
            "- Large individual misses remain even when aggregate MAE improves.",
            "- Coefficient interpretation is limited by correlated inputs.",
            "- The target excludes injuries and games missed.",
            "",
            "## Position-Specific Verdicts",
            "",
        ]
    )
    for position in POSITIONS:
        row = _model_row(comparison, position, MODEL_NAME)
        lines.extend(
            [
                f"### {position}: {row['first_attempt_classification']}",
                "",
                (
                    f"MAE improved by "
                    f"{row['MAE_improvement_over_previous_PPG_baseline']:.3f} PPG "
                    f"({row['percentage_MAE_improvement_over_previous_PPG_baseline']:.2f}%). "
                    f"The model beat the baseline in "
                    f"{row['seasons_beating_previous_PPG_baseline']} of "
                    f"{row['number_of_validation_seasons']} seasons and had overall "
                    f"Spearman {row['overall_Spearman_rank_correlation']:.3f}."
                ),
                "",
            ]
        )
    verdicts = {
        row["position"]: row["first_attempt_classification"]
        for row in linear_rows.iter_rows(named=True)
    }
    lines.extend(
        [
            "## Overall Verdict",
            "",
            (
                "The first attempt is **promising but not complete**. Exact PPG MAE "
                "improved for all four positions. Ranking improved for RB, WR, and "
                "TE, while QB ranking was slightly worse than the primary baseline. "
                f"The supported verdicts are QB={verdicts['QB']}, "
                f"RB={verdicts['RB']}, WR={verdicts['WR']}, and TE={verdicts['TE']}."
            ),
            "",
            (
                "Prediction accuracy, ranking quality, stability, and interpretability "
                "are separate concerns: aggregate prediction improved, seasonal gains "
                "were not uniform, and multicollinearity limits causal or isolated "
                "coefficient interpretation."
            ),
            "",
            "## Future-Work Observations (Not Implemented)",
            "",
            "- Obtain historical preseason team context and team-change indicators.",
            "- Investigate target/carry shares and team offensive tendencies.",
            "- Improve handling of limited historical samples.",
            "- Reconsider redundant inputs after preserving this V1 benchmark.",
            "- Compare Ridge regression for coefficient stability.",
            "- Test nonlinear models only after the linear benchmark remains documented.",
            "",
            (
                "No 2026 rankings were generated, and this report does not declare "
                "the project complete."
            ),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved first-attempt conclusion: {output_path.resolve()}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "v1_first_attempt_conclusion.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        write_first_attempt_conclusion(output_path=args.output)
    except (ConclusionError, pl.exceptions.PolarsError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
