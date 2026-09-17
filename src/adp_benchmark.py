"""Compare frozen V2 player rankings with historical preseason market ADP."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.config import (
    ADP_BENCHMARK_PROCESSED_DATA_DIR,
    ADP_BENCHMARK_RAW_DATA_DIR,
    ADP_BENCHMARK_REPORT_TABLES_DIR,
    ADP_BENCHMARK_REPORTS_DIR,
    POSITIONS,
    SELECTED_PREDICTIONS_DATA_DIR,
)
from src.validation import TOP_N_BY_POSITION

API_URL = "https://fantasyfootballcalculator.com/api/v1/adp/standard"
MODEL_LABEL = "V2"
MARKET_LABEL = "FFC preseason ADP"


def normalize_name(value: object) -> str:
    """Create a conservative join key that tolerates punctuation and suffixes."""

    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.lower().replace("'", "").replace(".", "")
    tokens = re.sub(r"[^a-z0-9]+", " ", text).strip().split()
    while tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return " ".join(tokens)


def download_adp(*, seasons: range, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        destination = output_dir / f"ffc_standard_12_{season}.json"
        request = Request(
            f"{API_URL}?teams=12&year={season}",
            headers={"User-Agent": "fantasy-football-predictor/1.0"},
        )
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
        if payload.get("status") != "Success" or not payload.get("players"):
            warnings.warn(
                f"Skipping {season}: API returned {payload.get('errors', 'no data')}",
                stacklevel=2,
            )
            continue
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_adp(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("ffc_standard_12_*.json")):
        season = int(path.stem.rsplit("_", 1)[-1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["players"])
        frame["season"] = season
        frames.append(frame)
        metadata.append({"season": season, **payload["meta"]})
    if not frames:
        raise FileNotFoundError(f"No archived ADP responses in {raw_dir}")
    adp = pd.concat(frames, ignore_index=True)
    adp = adp[adp["position"].isin(POSITIONS)].copy()
    adp["name_key"] = adp["name"].map(normalize_name)
    duplicate = adp.duplicated(["season", "position", "name_key"], keep=False)
    if duplicate.any():
        raise RuntimeError("ADP contains duplicate normalized player keys")
    return adp, pd.DataFrame(metadata).sort_values("season")


def build_paired_cohort(
    predictions: pd.DataFrame, adp: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v2 = predictions[
        ~predictions["specification"].eq("SELECTED_WR_PARTICIPATION_FALLBACK")
    ].copy()
    v2["name_key"] = v2["player_name"].map(normalize_name)
    duplicate = v2.duplicated(["season", "position", "name_key"], keep=False)
    if duplicate.any():
        raise RuntimeError("V2 contains duplicate normalized player keys")

    keys = ["season", "position", "name_key"]
    merged = v2.merge(
        adp[[*keys, "player_id", "name", "team", "adp", "times_drafted"]],
        on=keys,
        how="left",
        suffixes=("_v2", "_ffc"),
        validate="one_to_one",
        indicator=True,
    )
    audit = merged[[
        "season", "position", "player_id_v2", "player_name", "name_key", "_merge"
    ]].rename(columns={"_merge": "match_status"})
    paired = merged[merged["_merge"].eq("both")].copy()
    return paired, audit


def _top_overlap(actual_rank: pd.Series, candidate_rank: pd.Series, n: int) -> float:
    effective_n = min(n, len(actual_rank))
    actual_top = set(actual_rank.nsmallest(effective_n).index)
    candidate_top = set(candidate_rank.nsmallest(effective_n).index)
    return 100.0 * len(actual_top & candidate_top) / effective_n


def calculate_metrics(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    player_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for (position, season), group in paired.groupby(["position", "season"], sort=True):
        group = group.copy()
        group["actual_rank"] = group["actual_ppg"].rank(ascending=False, method="average")
        group["v2_rank"] = group["predicted_ppg"].rank(ascending=False, method="average")
        group["adp_rank"] = group["adp"].rank(ascending=True, method="average")
        player_rows.append(group)
        top_n = TOP_N_BY_POSITION[position]
        for label, column in ((MODEL_LABEL, "v2_rank"), (MARKET_LABEL, "adp_rank")):
            metric_rows.append(
                {
                    "position": position,
                    "season": season,
                    "method": label,
                    "matched_players": len(group),
                    "spearman": group["actual_rank"].corr(group[column], method="spearman"),
                    "mean_absolute_rank_error": (group["actual_rank"] - group[column]).abs().mean(),
                    "top_n": min(top_n, len(group)),
                    "top_n_overlap_percentage": _top_overlap(
                        group["actual_rank"], group[column], top_n
                    ),
                }
            )
    return pd.concat(player_rows, ignore_index=True), pd.DataFrame(metric_rows)


def aggregate_metrics(seasonal: pd.DataFrame) -> pd.DataFrame:
    summary = seasonal.groupby(["position", "method"], as_index=False).agg(
        seasons=("season", "nunique"),
        player_seasons=("matched_players", "sum"),
        mean_seasonal_spearman=("spearman", "mean"),
        mean_seasonal_rank_mae=("mean_absolute_rank_error", "mean"),
        mean_top_n_overlap_percentage=("top_n_overlap_percentage", "mean"),
    )
    pivot = seasonal.pivot(index=["position", "season"], columns="method")
    wins: list[dict[str, object]] = []
    for position, group in pivot.groupby(level="position"):
        v2_s = group["spearman"][MODEL_LABEL]
        adp_s = group["spearman"][MARKET_LABEL]
        v2_e = group["mean_absolute_rank_error"][MODEL_LABEL]
        adp_e = group["mean_absolute_rank_error"][MARKET_LABEL]
        wins.append(
            {
                "position": position,
                "v2_seasons_higher_spearman": int((v2_s > adp_s).sum()),
                "adp_seasons_higher_spearman": int((adp_s > v2_s).sum()),
                "v2_seasons_lower_rank_mae": int((v2_e < adp_e).sum()),
                "adp_seasons_lower_rank_mae": int((adp_e < v2_e).sum()),
            }
        )
    return summary.merge(pd.DataFrame(wins), on="position", how="left")


def run_benchmark() -> pd.DataFrame:
    predictions = pd.read_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_predictions.parquet"
    )
    adp, metadata = load_adp(ADP_BENCHMARK_RAW_DATA_DIR)
    paired, audit = build_paired_cohort(predictions, adp)
    ranked, seasonal = calculate_metrics(paired)
    summary = aggregate_metrics(seasonal)

    ADP_BENCHMARK_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADP_BENCHMARK_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ADP_BENCHMARK_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(ADP_BENCHMARK_PROCESSED_DATA_DIR / "v2_adp_paired_rankings.parquet")
    audit.to_csv(ADP_BENCHMARK_REPORT_TABLES_DIR / "player_matching_audit.csv", index=False)
    metadata.to_csv(ADP_BENCHMARK_REPORT_TABLES_DIR / "adp_archive_metadata.csv", index=False)
    seasonal.to_csv(ADP_BENCHMARK_REPORT_TABLES_DIR / "metrics_by_season.csv", index=False)
    summary.to_csv(ADP_BENCHMARK_REPORT_TABLES_DIR / "ranking_comparison.csv", index=False)

    match_summary = audit.groupby(["position", "match_status"], observed=True).size().unstack(fill_value=0)
    lines = [
        "V2 VERSUS FANTASY FOOTBALL CALCULATOR PRESEASON ADP",
        "=" * 62,
        "",
        "Benchmark: archived 12-team non-PPR ADP. Each position-season uses",
        "the identical set of players successfully matched to frozen V2 outputs.",
        "Higher Spearman and Top-N overlap are better; lower rank MAE is better.",
        (
            f"Source coverage: {metadata['season'].min()}-"
            f"{metadata['season'].max()}; the API returned no 2025 archive."
        ),
        "V2-only players are primarily players who were not drafted often enough",
        "to appear in the finite ADP player pool; they are excluded from both methods.",
        "",
        "MATCHING",
        match_summary.to_string(),
        "",
        "AGGREGATE RESULTS (unweighted mean across seasons)",
        summary.round(4).to_string(index=False),
        "",
    ]
    (ADP_BENCHMARK_REPORTS_DIR / "adp_benchmark_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--start-season", type=int, default=2012)
    parser.add_argument("--end-season", type=int, default=2025)
    args = parser.parse_args()
    if args.download:
        download_adp(
            seasons=range(args.start_season, args.end_season + 1),
            output_dir=ADP_BENCHMARK_RAW_DATA_DIR,
        )
    print(run_benchmark().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
