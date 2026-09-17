# Fantasy Football 2026 PPG Predictor

Position-specific models project **standard-scoring fantasy points per active regular-season game** for quarterbacks, running backs, wide receivers, and tight ends. This portfolio repo includes the historical modeling code, selected validation results, frozen 2026 inference inputs and models, and a small rankings browser.

**Try the rankings locally:** `python3 -m http.server 8000 --directory web`, then open `http://localhost:8000`. Search by player, filter by position or team, and sort the table. The page uses the checked-in 2026 prediction snapshot; it does not retrain or call a live API.

## What the model does

- Target: next-season standard/non-PPR fantasy points divided by active games, with a minimum of four target-season games in historical evaluation. This is **not** a projection of total points or games played.
- Population: veteran players with the required prior history. Rookies and other players without that history are outside these models.
- Separate models for QB, RB, WR, and TE. Player IDs, not names, are used for historical joins.
- Historical validation uses expanding chronological windows: each evaluation season is scored using earlier seasons only; preprocessing is fitted within each training fold.
- Primary metric: mean absolute error (MAE), in PPG. RMSE, Spearman correlation, and fixed Top-N overlap provide additional checks.

## Historical results

These are out-of-sample results for the selected position models. The previous-season PPG baseline is evaluated on the same player-season rows as each model.

| Position | Evaluation seasons | Player-seasons | Selected-model MAE | Previous-season PPG MAE |
| --- | --- | ---: | ---: | ---: |
| QB | 2012–2025 | 512 | 3.245 | 3.702 |
| RB | 2021–2025 | 410 | 2.362 | 2.634 |
| WR | 2019–2025 | 929 | 1.767 | 2.011 |
| TE | 2012–2025 | 968 | 1.319 | 1.497 |

Source: [V2/V1/baseline comparison](reports/selected_feature_set/v2_v1_previous_ppg_comparison.txt). The shorter RB and WR periods reflect feature availability. The TE model is a bounded residual correction to the preserved V1 model; its improvement over V1 is small and uncertain. See the [selected-model summary](reports/selected_feature_set/selected_feature_set_summary.txt), [yearly validation and limitations](reports/selected_feature_set/v2_post_selection_validation.md), and [leakage audit](reports/leakage_audit.txt).

The comparison against the previous-season baseline describes historical predictive accuracy, not a guarantee for the 2026 season. The 2026 rankings are a frozen snapshot made with data available through 2025.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/build_player_seasons.py`, `src/build_features.py`, `src/phase2_features.py`, `src/attempt3_features.py` | Target and feature construction |
| `src/validation.py`, `src/train_models.py`, `src/selected_feature_models.py` | Chronological evaluation and selected models |
| `src/te_constrained_competition.py`, `src/te_constrained_deployment.py` | TE follow-up and deployed correction |
| `src/run_pipeline.py` | Historical experiment entry points |
| `src/predict_2026.py` | Re-score the included frozen cohort |
| `tests/unit/`, `tests/integration/` | Self-contained checks and artifact-dependent research checks |
| `reports/selected_feature_set/` | Curated tables, plots, and selection notes |
| `data/model_inputs/`, `models/selected_feature_set/` | Frozen 2026 inputs and fitted artifacts |
| `data/predictions_2026/`, `web/` | Published ranking snapshot and browser |

The `src/` directory also retains isolated candidate experiments, including tree models, market features, roster context, and contract/TE ablations. They document model selection without adding their large generated outputs to Git.

## Run the published prediction snapshot

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra dev
uv run python -m src.predict_2026
uv run pytest
```

The prediction command reads `data/model_inputs/v2_2026_model_inputs.parquet` and the four included model artifacts, then rewrites the CSV and Parquet rankings. To refresh the frontend's copy after regenerating predictions:

```bash
python3 scripts/update_web_data.py
```

The model artifacts are local files; load them only from a repository or source you trust.

## Historical research reproducibility

The source and tests for feature construction, training, validation, and comparisons are included. Historical raw downloads, intermediate feature tables, per-fold models, and most exploratory reports are excluded to keep the portfolio compact. Reproducing every historical number from a fresh checkout requires rebuilding those inputs from their original sources; it is not a single-command workflow. Start with `uv run python -m src.download_data` and inspect `uv run python -m src.run_pipeline --help` for experiment stages. Some later experiments use external preseason rankings, contracts, or other snapshots whose availability may change. The included reports are the frozen evidence for the stated results. The default `pytest` run covers self-contained tests; `pytest tests/integration` requires the excluded historical artifacts.

Data sources include [nflverse](https://github.com/nflverse/nflverse-data) via `nflreadpy`, plus the preseason market and context sources documented in the corresponding downloader modules. Downloaded third-party datasets are not redistributed here.

## Known limitations

- PPG omits availability and future missed games, so these rankings are not full-season draft value estimates.
- Validation periods differ by position, and TE gains beyond V1 are modest.
- Some inputs include preseason market information; the model is not purely statistics based.
- The 2026 page is a snapshot. It does not update for injuries, trades, depth-chart changes, or new games.
