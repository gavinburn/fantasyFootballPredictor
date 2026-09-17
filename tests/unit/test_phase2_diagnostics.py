import numpy as np

from src.phase2_diagnostics import bootstrap_mean_interval


def test_bootstrap_interval_is_deterministic_and_contains_constant_mean():
    values = np.ones(20)
    first = bootstrap_mean_interval(values, samples=200)
    second = bootstrap_mean_interval(values, samples=200)
    assert first == second == (1.0, 1.0)


def test_season_block_bootstrap_resamples_complete_groups():
    values = np.array([1.0, 1.0, 3.0, 3.0])
    seasons = np.array([2020, 2020, 2021, 2021])
    low, high = bootstrap_mean_interval(
        values, groups=seasons, samples=1000, seed=446
    )
    assert low == 1.0
    assert high == 3.0
