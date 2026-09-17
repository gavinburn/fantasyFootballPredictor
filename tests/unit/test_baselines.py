import polars as pl
import pytest

from src.baselines import generate_baseline_predictions
from tests.unit.test_validation import feature_table


def test_baselines_use_identical_rows_and_training_only_position_mean():
    tables = {
        position: feature_table(position) for position in ("QB", "RB", "WR", "TE")
    }

    predictions, metrics = generate_baseline_predictions(
        tables,
        start_season=2012,
        min_training_rows=1,
    )

    counts = predictions.group_by(["position", "season", "player_id"]).len()
    assert counts["len"].unique().to_list() == [2]
    assert predictions["season"].min() == 2012
    assert predictions["season"].max() == 2013
    assert metrics["training_end_season"].max() == 2012

    qb_2012_mean = predictions.filter(
        (pl.col("position") == "QB")
        & (pl.col("season") == 2012)
        & (pl.col("model_name") == "historical_position_mean")
    )
    # Training targets are [10, 11, 11, 12] from 2010 and 2011 only.
    assert qb_2012_mean["predicted_ppg"].unique().item() == pytest.approx(11.0)
