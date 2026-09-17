from pathlib import Path

import polars as pl
import pytest

from src.download_phase2_data import _load_rankings
from src.phase2_features import (
    attempt2_features,
    build_roster_snapshot,
    build_team_environment,
    build_workload,
)


def test_rankings_use_confirmed_directions_and_prediction_season(tmp_path: Path):
    rows = ["season,team,ol_rank,sos_rank"]
    for season in range(2001, 2021):
        for team_number in range(1, 33):
            rows.append(f"{season},T{team_number:02d},{team_number},{team_number}")
    path = tmp_path / "rankings.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    rankings = _load_rankings(path)
    best_line_hardest_schedule = rankings.filter(
        (pl.col("prediction_season") == 2020) & (pl.col("target_team") == "T01")
    ).row(
        0, named=True
    )
    assert best_line_hardest_schedule["prediction_season"] == 2020
    assert best_line_hardest_schedule["preseason_offensive_line_score"] == 1.0
    assert best_line_hardest_schedule["preseason_strength_of_schedule_score"] == 0.0


def test_roster_snapshot_uses_week_one_and_excludes_cut_players():
    rosters = pl.DataFrame(
        {
            "season": [2020, 2020, 2020],
            "week": [1, 1, 2],
            "gsis_id": ["active", "cut", "later"],
            "position": ["RB", "RB", "RB"],
            "status": ["ACT", "CUT", "ACT"],
            "team": ["OAK", "OAK", "OAK"],
            "birth_date": [None, None, None],
            "espn_id": ["1", "2", "3"],
        }
    )
    snapshot = build_roster_snapshot(rosters)
    assert snapshot["player_id"].to_list() == ["active"]
    assert snapshot["target_team"].to_list() == ["LV"]


def test_target_team_environment_is_shifted_exactly_one_season():
    team = pl.DataFrame(
        {
            "season": [2020],
            "week": [1],
            "team": ["SD"],
            "season_type": ["REG"],
            "game_id": ["g1"],
            "attempts": [30],
            "sacks_suffered": [2],
            "carries": [20],
            "passing_tds": [2],
            "rushing_tds": [1],
            "passing_epa": [8.0],
            "rushing_epa": [2.0],
        }
    )
    result = build_team_environment(team).row(0, named=True)
    assert result["prediction_season"] == 2021
    assert result["target_team"] == "LAC"
    assert result["target_team_dropbacks_per_game_tminus1"] == 32
    assert result["target_team_pass_rate_tminus1"] == pytest.approx(32 / 52)


def test_traded_player_workload_uses_only_matched_team_weeks():
    players = pl.DataFrame(
        {
            "player_id": ["p", "p"],
            "season": [2020, 2020],
            "week": [1, 2],
            "team": ["A", "B"],
            "season_type": ["REG", "REG"],
            "position": ["RB", "RB"],
            "attempts": [0, 0],
            "sacks_suffered": [0, 0],
            "carries": [10, 5],
            "targets": [2, 3],
            "receiving_air_yards": [10.0, 20.0],
        }
    )
    teams = pl.DataFrame(
        {
            "season": [2020, 2020],
            "week": [1, 2],
            "team": ["A", "B"],
            "season_type": ["REG", "REG"],
            "attempts": [30, 40],
            "sacks_suffered": [2, 3],
            "carries": [100, 50],
            "targets": [30, 40],
            "receiving_air_yards": [300.0, 400.0],
        }
    )
    row = build_workload(players, teams).row(0, named=True)
    assert row["prediction_season"] == 2021
    assert row["previous_carry_share"] == pytest.approx(15 / 150)
    assert row["previous_target_share"] == pytest.approx(5 / 70)
    assert row["previous_air_yards_share"] == pytest.approx(30 / 700)


def test_attempt2_feature_groups_extend_v1_without_identifiers():
    features = attempt2_features(
        "QB", ["volume", "competition", "quarterback", "offensive_line", "schedule"]
    )
    assert "target_team_dropbacks_per_game_tminus1" in features
    assert "projected_qb_previous_qbr" in features
    assert "preseason_offensive_line_score" in features
    assert "preseason_strength_of_schedule_score" in features
    assert "player_id" not in features
    assert len(features) == len(set(features))
