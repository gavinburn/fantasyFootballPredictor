# Fantasy Football 2026 PPG Predictor

An inference-only machine-learning project that ranks NFL quarterbacks, running
backs, wide receivers, and tight ends by projected 2026 fantasy points per game
under standard scoring. The final deployment models were trained on historical
data available through the 2025 season.

The project uses position-specific regularized regression models selected with
expanding-window validation. QB, RB, and WR use the selected adaptive deployment
models; TE uses the validated constrained-residual model. The repository contains
the exact model-ready 2026 cohort so the published rankings are reproducible.

## What is included

- Final fitted 2026 selected-model artifacts (one per position)
- The frozen model-ready 2026 feature cohort supplied to those models
- Generated 2026 rankings in CSV and Parquet formats
- Source code and locked Python dependencies


## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)

## Setup

From the extracted package directory, install the locked environment:

```bash
uv sync --frozen
```

## Generate the 2026 rankings

Run the prediction workflow:

```bash
uv run python -m src.predict_2026
```

The command loads the frozen feature rows and included V2 models. It writes:

- `data/predictions_2026/v2_2026_rankings.csv`
- `data/predictions_2026/v2_2026_rankings.parquet`

It also prints the five highest-ranked players at each position.

## Scope of reproducibility

The included models are the final 2026 deployment versions trained using data
available through 2025. The historical tables in the report were generated
with expanding-window validation, which fitted a separate training model for
each evaluation season. Those historical fold models and previously generated
report outputs are intentionally excluded to keep this package compact.

Accordingly, this repository reproduces the 2026 predictions without
retraining or recalculating features. It does not provide a model-training
command. Historical experiments, superseded model versions, validation-fold
artifacts, and exploratory reports are intentionally excluded from this public
portfolio version.
