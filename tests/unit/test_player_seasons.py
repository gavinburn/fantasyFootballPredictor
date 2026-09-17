import polars as pl
import pytest

from src.build_player_seasons import PlayerSeasonError, build_player_season_tables


def _weekly_rows(rows):
    defaults = {
        "player_id": "p1",
        "player_display_name": "Test Player",
        "season": 2024,
        "week": 1,
        "season_type": "REG",
        "position": "RB",
        "team": "AAA",
        "game_id": "2024_01_AAA_BBB",
        "completions": 0,
        "attempts": 0,
        "passing_yards": 0,
        "passing_tds": 0,
        "passing_interceptions": 0,
        "carries": 0,
        "rushing_yards": 0,
        "rushing_tds": 0,
        "targets": 0,
        "receptions": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "passing_2pt_conversions": 0,
        "rushing_2pt_conversions": 0,
        "receiving_2pt_conversions": 0,
        "fumbles_lost_total": 0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_builds_ppg_counts_zero_stat_game_and_uses_latest_team():
    weekly = _weekly_rows(
        [
            {"rushing_yards": 100, "rushing_tds": 1},
            {
                "week": 2,
                "game_id": "2024_02_CCC_AAA",
                "team": "CCC",
            },
            {
                "week": 3,
                "game_id": "2024_03_AAA_DDD",
                "season_type": "POST",
                "rushing_yards": 200,
            },
            {"player_id": None, "game_id": "2024_04_AAA_EEE"},
            {"player_id": "k1", "position": "K", "game_id": "2024_05_AAA_FFF"},
        ]
    )

    audit, targets = build_player_season_tables(weekly, min_target_games=2)

    assert audit.height == 1
    row = audit.row(0, named=True)
    assert row["games_played"] == 2
    assert row["team"] == "CCC"
    assert row["total_fantasy_points"] == pytest.approx(16.0)
    assert row["fantasy_points_per_game"] == pytest.approx(8.0)
    assert targets.height == 1


def test_minimum_games_threshold_filters_only_modeling_targets():
    audit, targets = build_player_season_tables(_weekly_rows([{}]), min_target_games=2)

    assert audit.height == 1
    assert targets.is_empty()


def test_duplicate_player_game_is_rejected():
    weekly = _weekly_rows([{}, {"rushing_yards": 10}])

    with pytest.raises(PlayerSeasonError, match="duplicate player-game"):
        build_player_season_tables(weekly)


def test_invalid_minimum_games_is_rejected():
    with pytest.raises(PlayerSeasonError, match="at least 1"):
        build_player_season_tables(_weekly_rows([{}]), min_target_games=0)
