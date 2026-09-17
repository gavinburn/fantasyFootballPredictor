from datetime import date

import polars as pl
import pytest

from src.download_data import DownloadError, download_mandatory_data


class FakeNflreadpy:
    def load_player_stats(self, *, seasons, summary_level):
        assert summary_level == "week"
        return pl.DataFrame({"season": seasons, "player_id": ["p"] * len(seasons)})

    def load_players(self):
        return pl.DataFrame({"gsis_id": ["p"], "display_name": ["Player"]})

    def load_rosters(self, *, seasons):
        assert seasons == 2026
        return pl.DataFrame(
            {
                "position": ["QB", "RB", "WR", "TE"],
                "gsis_id": ["q", "r", "w", "t"],
            }
        )


def test_download_writes_all_mandatory_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda _: "test-version")

    manifest = download_mandatory_data(
        output_dir=tmp_path,
        snapshot_date=date(2026, 7, 25),
        nfl_module=FakeNflreadpy(),
    )

    assert manifest.exists()
    assert (tmp_path / "player_weekly_stats_2006_2025.parquet").exists()
    assert (tmp_path / "players.parquet").exists()
    assert (tmp_path / "rosters_2026_2026-07-25.parquet").exists()


def test_empty_required_download_fails_clearly(tmp_path):
    fake = FakeNflreadpy()
    fake.load_players = lambda: pl.DataFrame()

    with pytest.raises(DownloadError, match="player master data.*zero rows"):
        download_mandatory_data(output_dir=tmp_path, nfl_module=fake)
