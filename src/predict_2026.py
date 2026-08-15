"""Generate 2026 rankings from frozen model-ready features and fitted models."""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import DATA_DIR, SELECTED_MODELS_DIR

INPUT_PATH = DATA_DIR / "model_inputs/v2_2026_model_inputs.parquet"
OUTPUT_DIR = DATA_DIR / "predictions_2026"
MODEL_FILES = {
    "QB": "qb_selected_main.joblib",
    "RB": "rb_selected_main.joblib",
    "WR": "wr_selected_main.joblib",
    "TE": "te_selected_constrained_residual.joblib",
}


class PredictionError(RuntimeError):
    """Raised when frozen inputs do not match a fitted model."""


def generate_rankings() -> pd.DataFrame:
    cohort = pd.read_parquet(INPUT_PATH)
    outputs: list[pd.DataFrame] = []
    for position in ("QB", "RB", "WR", "TE"):
        rows = cohort[cohort["position"].eq(position)].copy()
        model = joblib.load(SELECTED_MODELS_DIR / MODEL_FILES[position])
        features = list(model.feature_names_in_)
        missing = sorted(set(features) - set(rows.columns))
        if missing:
            raise PredictionError(f"{position} is missing features: {missing}")
        rows["projected_ppg"] = model.predict(rows[features])
        rows["rank"] = rows["projected_ppg"].rank(
            ascending=False, method="first"
        ).astype(int)
        outputs.append(rows[[
            "rank", "player_id", "player_name", "position", "target_team",
            "projected_ppg",
        ]])
    predictions = pd.concat(outputs, ignore_index=True).sort_values(
        ["position", "rank"]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "v2_2026_rankings.csv", index=False)
    predictions.to_parquet(
        OUTPUT_DIR / "v2_2026_rankings.parquet", index=False
    )
    return predictions


if __name__ == "__main__":
    result = generate_rankings()
    print(result[result["rank"] <= 5].to_string(index=False))
