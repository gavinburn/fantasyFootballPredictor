import polars as pl
import pytest

from src.build_features import V1_FEATURES, build_feature_tables


def season_row(player_id, position, season, ppg, **overrides):
    row = {
        "player_id": player_id,
        "player_name": f"Player {player_id}",
        "position": position,
        "season": season,
        "team": f"T{season % 3}",
        "fantasy_points_per_game": ppg,
        "games_played": 16,
        "passing_attempts": 0.0,
        "passing_yards": 0.0,
        "passing_touchdowns": 0.0,
        "passing_interceptions": 0.0,
        "rushing_attempts": 0.0,
        "rushing_yards": 0.0,
        "rushing_touchdowns": 0.0,
        "targets": 0.0,
        "receiving_yards": 0.0,
        "receiving_touchdowns": 0.0,
        "opportunities": 0.0,
    }
    row.update(overrides)
    return row


def player_master():
    return pl.DataFrame(
        {
            "gsis_id": ["qb", "rb", "wr", "te"],
            "birth_date": ["1990-01-01"] * 4,
            "rookie_season": [2018] * 4,
            "draft_round": [1, 2, None, 3],
            "draft_pick": [1, 40, None, 75],
        }
    )


def historical_rows():
    rows = []
    for season, ppg in zip(range(2020, 2024), [10.0, 20.0, 30.0, 40.0]):
        rows.append(
            season_row(
                "qb",
                "QB",
                season,
                ppg,
                passing_attempts=400,
                passing_yards=3200,
                passing_touchdowns=24,
                passing_interceptions=8,
                rushing_attempts=80,
                rushing_yards=400,
                rushing_touchdowns=4,
                opportunities=80,
            )
        )
    for position, player_id, carries in (
        ("RB", "rb", 160),
        ("WR", "wr", 32),
        ("TE", "te", 0),
    ):
        rows.extend(
            [
                season_row(
                    player_id,
                    position,
                    2020,
                    5.0,
                    rushing_attempts=carries,
                    rushing_yards=carries * 4,
                    targets=80,
                    receiving_yards=800,
                    receiving_touchdowns=8,
                    opportunities=carries + 80,
                ),
                season_row(player_id, position, 2021, 6.0),
            ]
        )
    return pl.DataFrame(rows)


def test_first_season_is_null_and_exact_lags_feed_rolling_means():
    tables = build_feature_tables(historical_rows(), player_master())
    qb = tables["QB"]

    first = qb.filter(pl.col("prediction_season") == 2020).row(0, named=True)
    assert first["previous_season_ppg"] is None
    assert first["three_season_mean_ppg"] is None

    row_2023 = qb.filter(pl.col("prediction_season") == 2023).row(0, named=True)
    assert row_2023["previous_season_ppg"] == pytest.approx(30.0)
    assert row_2023["two_season_mean_ppg"] == pytest.approx(25.0)
    assert row_2023["three_season_mean_ppg"] == pytest.approx(20.0)
    assert row_2023["previous_year_ppg_change"] == pytest.approx(10.0)
    assert row_2023["prior_seasons_count"] == 3


def test_individual_position_rushing_attempts_are_used():
    tables = build_feature_tables(historical_rows(), player_master())

    qb = tables["QB"].filter(pl.col("prediction_season") == 2021).row(0, named=True)
    rb = tables["RB"].filter(pl.col("prediction_season") == 2021).row(0, named=True)
    wr = tables["WR"].filter(pl.col("prediction_season") == 2021).row(0, named=True)

    assert qb["previous_qb_rushing_attempts_per_game"] == pytest.approx(5.0)
    assert rb["previous_rushing_attempts_per_game"] == pytest.approx(10.0)
    assert wr["previous_rushing_attempts_per_game"] == pytest.approx(2.0)
    assert all("team_rushing" not in feature for feature in V1_FEATURES["QB"])


def test_position_tables_are_separate_and_have_no_ids_in_model_inputs():
    rows = historical_rows().vstack(pl.DataFrame([season_row("qb2", "QB", 2021, 7.0)]))
    master = player_master().vstack(
        pl.DataFrame(
            {
                "gsis_id": ["qb2"],
                "birth_date": ["1992-02-02"],
                "rookie_season": [2021],
                "draft_round": [None],
                "draft_pick": [None],
            }
        )
    )
    tables = build_feature_tables(rows, master)

    for position, table in tables.items():
        assert table["position"].unique().to_list() == [position]
        assert "player_id" not in V1_FEATURES[position]
        assert "player_name" not in V1_FEATURES[position]
        assert "target_ppg" not in V1_FEATURES[position]
        assert all(
            feature.startswith(("previous_", "three_", "age_", "years_"))
            for feature in V1_FEATURES[position]
        )
