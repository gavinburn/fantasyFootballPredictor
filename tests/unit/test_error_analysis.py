import polars as pl

from src.error_analysis import add_error_buckets, grouped_errors, ranking_errors


def error_frame():
    return pl.DataFrame(
        {
            "player_id": ["a", "b"],
            "player_name": ["A", "B"],
            "position": ["QB", "QB"],
            "prediction_season": [2020, 2020],
            "actual_ppg": [20.0, 10.0],
            "predicted_ppg": [18.0, 12.0],
            "signed_error": [-2.0, 2.0],
            "absolute_error": [2.0, 2.0],
            "actual_rank": [1.0, 2.0],
            "predicted_rank": [2.0, 1.0],
            "age_entering_season": [23.0, 30.0],
            "previous_season_games_played": [7.0, 15.0],
            "previous_season_ppg": [16.0, 8.0],
            "prior_seasons_count": [1, 3],
            "changed_team": [None, None],
            "missing_input_count": [0, 3],
        }
    )


def test_error_buckets_have_fixed_boundaries_and_group_sizes():
    bucketed = add_error_buckets(error_frame())
    assert bucketed["age_bucket"].to_list() == ["<24", "30-32"]
    assert bucketed["games_played_bucket"].to_list() == ["4-7", "15+"]

    grouped = grouped_errors(bucketed, "age_bucket")
    assert grouped["sample_size"].sum() == 2
    assert grouped["MAE"].unique().to_list() == [2.0]


def test_ranking_errors_identify_top_n_ordering_direction():
    ranked = ranking_errors(add_error_buckets(error_frame()))

    assert ranked.filter(pl.col("player_id") == "a")["rank_error"].item() == 1.0
    assert ranked.filter(pl.col("player_id") == "b")["rank_error"].item() == -1.0
