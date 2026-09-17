from pathlib import Path

import polars as pl

from src.conclusion import write_first_attempt_conclusion
from src.train_models import MODEL_NAME


def test_conclusion_reports_all_positions_and_stops_before_2026(tmp_path):
    comparison_rows = []
    for position in ("QB", "RB", "WR", "TE"):
        for model, mae, spearman, classification in (
            ("previous_season_ppg", 2.0, 0.6, "Reference baseline"),
            (MODEL_NAME, 1.8, 0.62, "Successful"),
        ):
            comparison_rows.append(
                {
                    "position": position,
                    "model_name": model,
                    "overall_MAE": mae,
                    "baseline_MAE": 2.0,
                    "MAE_improvement_over_previous_PPG_baseline": 2.0 - mae,
                    "percentage_MAE_improvement_over_previous_PPG_baseline": (
                        (2.0 - mae) / 2.0 * 100
                    ),
                    "overall_Spearman_rank_correlation": spearman,
                    "mean_top_N_overlap_percentage": 60.0,
                    "seasons_beating_previous_PPG_baseline": 10,
                    "seasons_tied_with_previous_PPG_baseline": 0,
                    "seasons_losing_to_previous_PPG_baseline": 4,
                    "number_of_validation_seasons": 14,
                    "first_attempt_classification": classification,
                }
            )
    comparison = tmp_path / "comparison.csv"
    pl.DataFrame(comparison_rows).write_csv(comparison)
    stability = tmp_path / "stability.csv"
    pl.DataFrame(
        {
            "position": ["QB", "RB", "WR", "TE"],
            "unstable_coefficient": [True, False, False, True],
        }
    ).write_csv(stability)
    errors = tmp_path / "errors.csv"
    pl.DataFrame(
        {
            "position": ["QB", "RB", "WR", "TE"],
            "absolute_error": [10.0, 9.0, 8.0, 7.0],
        }
    ).write_csv(errors)
    output = Path(tmp_path) / "conclusion.md"

    write_first_attempt_conclusion(
        comparison_path=comparison,
        stability_path=stability,
        absolute_errors_path=errors,
        output_path=output,
    )

    text = output.read_text()
    assert all(f"### {position}:" in text for position in ("QB", "RB", "WR", "TE"))
    assert "No 2026 rankings were generated" in text
