import polars as pl
import pytest

from src.column_mapping import WEEKLY_PLAYER_STATS_COLUMNS
from src.scoring import (
    STANDARD_FANTASY_POINTS_COLUMN,
    ScoringError,
    calculate_weekly_fantasy_points,
    score_stat_line,
)


def test_qb_stat_line():
    assert score_stat_line(
        passing_yards=300,
        passing_touchdowns=2,
        interceptions=1,
    ) == pytest.approx(18.0)


def test_rb_stat_line():
    assert score_stat_line(
        rushing_yards=100,
        rushing_touchdowns=1,
        receiving_yards=30,
    ) == pytest.approx(19.0)


def test_wr_stat_line_has_no_ppr_points():
    assert score_stat_line(
        receptions=8,
        receiving_yards=80,
        receiving_touchdowns=1,
    ) == pytest.approx(14.0)


def test_mixed_stat_player_with_lost_fumble_and_two_point_conversion():
    assert score_stat_line(
        passing_yards=50,
        rushing_yards=20,
        rushing_touchdowns=1,
        receiving_yards=30,
        rushing_two_point_conversions=1,
        fumbles_lost=1,
    ) == pytest.approx(13.0)


def test_zero_stat_appearance():
    assert score_stat_line() == 0.0


def test_negative_point_game_is_preserved():
    assert score_stat_line(interceptions=2, fumbles_lost=1) == -6.0


def _weekly_frame(**overrides):
    values = {
        WEEKLY_PLAYER_STATS_COLUMNS["passing_yards"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["passing_touchdowns"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["interceptions"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["rushing_yards"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["rushing_touchdowns"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["receiving_yards"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["receiving_touchdowns"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["passing_two_point_conversions"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["rushing_two_point_conversions"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["receiving_two_point_conversions"]: [0],
        WEEKLY_PLAYER_STATS_COLUMNS["fumbles_lost"]: [0],
    }
    values.update({key: [value] for key, value in overrides.items()})
    return pl.DataFrame(values)


def test_dataframe_scoring_fills_null_counting_stats_with_zero():
    frame = _weekly_frame(passing_yards=300, rushing_yards=None)

    scored = calculate_weekly_fantasy_points(frame)

    assert scored[STANDARD_FANTASY_POINTS_COLUMN].item() == pytest.approx(12.0)


def test_dataframe_scoring_rejects_missing_stat_columns():
    with pytest.raises(ScoringError, match="missing stat columns"):
        calculate_weekly_fantasy_points(pl.DataFrame({"passing_yards": [100]}))
