import pandas as pd
import pytest

from src.adp_benchmark import build_paired_cohort, calculate_metrics, normalize_name


def test_normalize_name_handles_suffixes_and_punctuation():
    assert normalize_name("D.J. Chark Jr.") == "dj chark"
    assert normalize_name("Brian Robinson Jr.") == "brian robinson"


def test_matching_requires_same_position_and_season():
    predictions = pd.DataFrame(
        [{"player_id": "nfl1", "player_name": "John Doe Jr.", "position": "WR", "season": 2020,
          "actual_ppg": 10.0, "predicted_ppg": 9.0, "specification": "SELECTED_MAIN"}]
    )
    adp = pd.DataFrame(
        [{"player_id": 1, "name": "John Doe", "name_key": "john doe", "position": "WR", "season": 2020,
          "team": "ABC", "adp": 25.0, "times_drafted": 100}]
    )
    paired, audit = build_paired_cohort(predictions, adp)
    assert len(paired) == 1
    assert audit.iloc[0]["match_status"] == "both"


def test_rank_metrics_reward_correct_ordering():
    paired = pd.DataFrame(
        {
            "position": ["QB"] * 3,
            "season": [2020] * 3,
            "actual_ppg": [30.0, 20.0, 10.0],
            "predicted_ppg": [29.0, 19.0, 9.0],
            "adp": [3.0, 2.0, 1.0],
        }
    )
    _, metrics = calculate_metrics(paired)
    v2 = metrics[metrics["method"].eq("V2")].iloc[0]
    market = metrics[metrics["method"].eq("FFC preseason ADP")].iloc[0]
    assert v2["spearman"] == pytest.approx(1.0)
    assert v2["mean_absolute_rank_error"] == pytest.approx(0.0)
    assert market["spearman"] == pytest.approx(-1.0)
