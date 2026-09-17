"""Targeted, read-only audit of the existing team and roster feature data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, POSITIONS, REPORTS_DIR

OUTPUT_DIR = REPORTS_DIR / "data_quality_audit"
TABLES_DIR = OUTPUT_DIR / "tables"
TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "LA": "LAR",
    "JAC": "JAX",
    "WSH": "WAS",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "LAR",
}


def _read(relative: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / relative, columns=columns, engine="pyarrow")


def _normalize_team(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_ALIASES)


def _finding(
    category: str, check: str, severity: str, affected: int, total: int, detail: str
) -> dict:
    return {
        "category": category,
        "check": check,
        "severity": severity,
        "status": "PASS" if affected == 0 else "REVIEW",
        "affected_rows": int(affected),
        "total_rows": int(total),
        "affected_rate": affected / total if total else np.nan,
        "detail": detail,
    }


def audit_identifiers_and_positions() -> tuple[list[dict], pd.DataFrame]:
    targets = _read("processed/player_season_targets.parquet")
    context = _read("attempt_2/interim/prediction_context.parquet")
    frames = []
    findings = []
    for name, frame, keys in (
        ("targets", targets, ["player_id", "season"]),
        ("prediction_context", context, ["player_id", "prediction_season"]),
    ):
        missing = frame[keys].isna().any(axis=1)
        duplicates = frame.duplicated(keys, keep=False)
        findings.append(
            _finding(
                "identifiers",
                f"{name}_missing_keys",
                "critical",
                missing.sum(),
                len(frame),
                "Model keys must be complete.",
            )
        )
        findings.append(
            _finding(
                "identifiers",
                f"{name}_duplicate_keys",
                "critical",
                duplicates.sum(),
                len(frame),
                "Player-season keys must be unique.",
            )
        )
        if missing.any() or duplicates.any():
            issue = frame.loc[missing | duplicates, keys].copy()
            issue["source"] = name
            frames.append(issue)
    mismatch = context[context["position"] != context["roster_position"]][
        ["player_id", "prediction_season", "position", "roster_position", "target_team"]
    ].copy()
    names = targets[["player_id", "season", "player_name"]].rename(
        columns={"season": "prediction_season"}
    )
    mismatch = mismatch.merge(
        names,
        on=["player_id", "prediction_season"],
        how="left",
        validate="one_to_one",
    )
    mismatch = mismatch[
        [
            "player_id",
            "player_name",
            "prediction_season",
            "position",
            "roster_position",
            "target_team",
        ]
    ]
    mismatch["issue"] = "target_position_differs_from_week1_roster_position"
    findings.append(
        _finding(
            "positions",
            "target_vs_roster_position",
            "high",
            len(mismatch),
            len(context),
            "A mismatch changes the position model and competition-pool definition.",
        )
    )
    issues = (
        pd.concat([*frames, mismatch], ignore_index=True, sort=False)
        if frames or len(mismatch)
        else pd.DataFrame()
    )
    return findings, issues


def _actual_primary_team() -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = _read(
        "raw/player_weekly_stats_2006_2025.parquet",
        [
            "player_id",
            "season",
            "week",
            "season_type",
            "team",
            "attempts",
            "carries",
            "targets",
        ],
    )
    weekly = weekly[(weekly.season_type == "REG") & weekly.player_id.notna()].copy()
    weekly["team"] = _normalize_team(weekly["team"])
    weekly["opportunities"] = (
        weekly[["attempts", "carries", "targets"]].fillna(0).sum(axis=1)
    )
    grouped = (
        weekly.groupby(["player_id", "season", "team"], dropna=False)
        .agg(
            appearances=("week", "size"),
            offensive_opportunities=("opportunities", "sum"),
            latest_week=("week", "max"),
            earliest_week=("week", "min"),
        )
        .reset_index()
    )
    ordered = grouped.sort_values(
        [
            "player_id",
            "season",
            "appearances",
            "offensive_opportunities",
            "latest_week",
            "team",
        ],
        ascending=[True, True, False, False, False, True],
    )
    primary = ordered.drop_duplicates(["player_id", "season"]).rename(
        columns={"team": "actual_primary_team"}
    )
    first = (
        grouped.sort_values(["player_id", "season", "earliest_week", "team"])
        .drop_duplicates(["player_id", "season"])
        .rename(columns={"team": "actual_first_team"})
    )
    return primary[["player_id", "season", "actual_primary_team"]], first[
        ["player_id", "season", "actual_first_team"]
    ]


def audit_team_assignments() -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    context = _read("attempt_2/interim/prediction_context.parquet")
    names = _read(
        "processed/player_season_targets.parquet",
        ["player_id", "season", "player_name"],
    ).rename(columns={"season": "prediction_season"})
    primary, first = _actual_primary_team()
    compared = context.merge(
        primary,
        left_on=["player_id", "prediction_season"],
        right_on=["player_id", "season"],
        how="left",
        validate="one_to_one",
    ).merge(first, on=["player_id", "season"], how="left", validate="one_to_one")
    compared["target_differs_from_first_actual_team"] = (
        compared.target_team.ne(compared.actual_first_team)
        & compared.actual_first_team.notna()
    )
    compared["target_differs_from_primary_actual_team"] = (
        compared.target_team.ne(compared.actual_primary_team)
        & compared.actual_primary_team.notna()
    )
    no_actual = compared.actual_primary_team.isna()
    detail = compared[
        no_actual
        | compared.target_differs_from_first_actual_team
        | compared.target_differs_from_primary_actual_team
    ].copy()
    detail = detail.merge(
        names,
        on=["player_id", "prediction_season"],
        how="left",
        validate="one_to_one",
    )
    weekly = _read(
        "raw/player_weekly_stats_2006_2025.parquet",
        [
            "player_id",
            "season",
            "team",
            "position",
            "season_type",
            "week",
            "attempts",
            "carries",
            "targets",
        ],
    )
    weekly = weekly[
        (weekly.season_type == "REG") & weekly.position.isin(POSITIONS)
    ].copy()
    weekly["team"] = _normalize_team(weekly.team)
    counts = (
        weekly.groupby(["player_id", "season"])["team"]
        .nunique()
        .reset_index(name="teams_played_for")
    )
    multi = counts[counts.teams_played_for > 1].merge(
        primary, on=["player_id", "season"], validate="one_to_one"
    )
    next_context = context[
        [
            "player_id",
            "prediction_season",
            "target_team",
            "previous_primary_team",
            "changed_team",
        ]
    ].copy()
    multi["prediction_season"] = multi.season + 1
    multi = multi.merge(
        next_context,
        on=["player_id", "prediction_season"],
        how="left",
        validate="one_to_one",
    )
    findings = [
        _finding(
            "team_assignment",
            "no_actual_regular_season_appearance",
            "medium",
            no_actual.sum(),
            len(compared),
            "Week-1 roster assignment exists but the player has no recorded regular-season appearance.",
        ),
        _finding(
            "team_assignment",
            "preseason_team_vs_first_actual_team",
            "high",
            compared.target_differs_from_first_actual_team.sum(),
            compared.actual_first_team.notna().sum(),
            "Post-hoc diagnostic for cuts/trades or incorrect Week-1 roster assignment.",
        ),
        _finding(
            "team_assignment",
            "preseason_team_vs_primary_actual_team",
            "medium",
            compared.target_differs_from_primary_actual_team.sum(),
            compared.actual_primary_team.notna().sum(),
            "Differences can be legitimate in-season transactions; do not replace preseason team with this future value.",
        ),
        _finding(
            "team_assignment",
            "previous_multi_team_seasons",
            "medium",
            len(multi),
            len(counts),
            "Previous-primary-team and changed-team flags compress multi-team history to one club.",
        ),
    ]
    columns = [
        "player_id",
        "player_name",
        "prediction_season",
        "position",
        "target_team",
        "actual_first_team",
        "actual_primary_team",
        "status",
        "changed_team",
        "target_differs_from_first_actual_team",
        "target_differs_from_primary_actual_team",
    ]
    return findings, detail[columns], multi


def audit_projected_qbs() -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    context = _read("attempt_2/interim/prediction_context.parquet")
    projected = context[
        ["prediction_season", "target_team", "projected_starting_qb_id"]
    ].drop_duplicates()
    weekly = _read(
        "raw/player_weekly_stats_2006_2025.parquet",
        ["player_id", "season", "team", "position", "season_type", "attempts", "week"],
    )
    qb = weekly[(weekly.season_type == "REG") & (weekly.position == "QB")].copy()
    qb["team"] = _normalize_team(qb.team)
    actual = (
        qb.groupby(["season", "team", "player_id"])
        .agg(actual_attempts=("attempts", "sum"), first_week=("week", "min"))
        .reset_index()
        .sort_values(
            ["season", "team", "actual_attempts", "first_week", "player_id"],
            ascending=[True, True, False, True, True],
        )
        .drop_duplicates(["season", "team"])
    )
    compared = projected.merge(
        actual,
        left_on=["prediction_season", "target_team"],
        right_on=["season", "team"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"player_id": "actual_primary_qb_id"})
    compared["projected_matches_actual_primary_qb"] = (
        compared.projected_starting_qb_id.eq(compared.actual_primary_qb_id)
    )
    team_counts = (
        projected.groupby("prediction_season")
        .target_team.nunique()
        .reset_index(name="teams_with_context")
    )
    incomplete = team_counts[team_counts.teams_with_context != 32]
    findings = [
        _finding(
            "projected_qb",
            "context_team_season_coverage",
            "medium",
            int((32 - team_counts.teams_with_context).clip(lower=0).sum()),
            640,
            "Context-derived table omits team-seasons with no eligible target player; starter source construction itself requires 32 teams.",
        ),
        _finding(
            "projected_qb",
            "projected_vs_actual_primary_qb",
            "medium",
            (~compared.projected_matches_actual_primary_qb).sum(),
            compared.actual_primary_qb_id.notna().sum(),
            "Post-hoc accuracy check; disagreement is not automatically a data error.",
        ),
        _finding(
            "projected_qb",
            "duplicate_team_season_starters",
            "critical",
            projected.duplicated(["prediction_season", "target_team"]).sum(),
            len(projected),
            "One projected starter is required per team-season.",
        ),
    ]
    return findings, compared, incomplete


def audit_qbr_missingness() -> tuple[list[dict], pd.DataFrame]:
    environment = _read("attempt_2/interim/projected_qb_environment.parquet")
    players = _read("raw/players.parquet", ["gsis_id", "espn_id"]).drop_duplicates(
        "gsis_id"
    )
    qbr = _read(
        "attempt_2/raw/qbr_season_level.parquet",
        ["season", "season_type", "player_id", "qbr_total", "qb_plays"],
    )
    qbr = qbr[qbr.season_type == "Regular"].copy()
    qbr["player_id"] = qbr.player_id.astype("string")
    detail = environment.merge(
        players.rename(
            columns={"gsis_id": "projected_starting_qb_id", "espn_id": "mapped_espn_id"}
        ),
        on="projected_starting_qb_id",
        how="left",
        validate="many_to_one",
    )
    detail["mapped_espn_id"] = detail.mapped_espn_id.astype("string")
    history = qbr.rename(
        columns={
            "season": "history_season",
            "player_id": "mapped_espn_id",
            "qbr_total": "raw_qbr",
            "qb_plays": "raw_qb_plays",
        }
    )
    detail["history_season"] = detail.prediction_season - 1
    detail = detail.merge(
        history[["history_season", "mapped_espn_id", "raw_qbr", "raw_qb_plays"]],
        on=["history_season", "mapped_espn_id"],
        how="left",
        validate="many_to_one",
    )
    detail["missing_reason"] = np.select(
        [
            detail.projected_qb_previous_qbr.notna(),
            detail.mapped_espn_id.isna(),
            detail.raw_qbr.isna() & detail.raw_qb_plays.notna(),
        ],
        ["available", "no_gsis_to_espn_mapping", "qbr_row_present_but_value_null"],
        default="no_prior_season_qbr_row",
    )
    missing = detail.projected_qb_previous_qbr.isna()
    findings = [
        _finding(
            "qbr",
            "projected_qb_qbr_missing",
            "medium",
            missing.sum(),
            len(detail),
            "Missing values distinguish absent ID mapping, absent prior QBR history, and null source values; median imputation plus a presence flag is currently used.",
        )
    ]
    return findings, detail


def audit_competition() -> tuple[list[dict], pd.DataFrame]:
    context = _read("attempt_2/interim/prediction_context.parquet")
    long = _read("attempt_2/interim/competition_long.parquet")
    model = _read("attempt_2/interim/competition_model_features.parquet")
    duplicate = long.duplicated(
        ["player_id", "prediction_season", "competitor_id"], keep=False
    )
    self_rows = long.player_id.eq(long.competitor_id)
    position_bad = np.where(
        long.position.eq("QB"),
        ~long.competitor_position.eq("QB"),
        np.where(
            long.position.eq("RB"),
            ~long.competitor_position.eq("RB"),
            ~long.competitor_position.isin(["WR", "TE"]),
        ),
    )
    ranks = (
        long.groupby(["player_id", "prediction_season"])
        .competition_rank.agg(["min", "max", "nunique", "size"])
        .reset_index()
    )
    bad_rank_keys = ranks[
        (ranks["min"] != 1)
        | (ranks["max"] != ranks["size"])
        | (ranks["nunique"] != ranks["size"])
    ][["player_id", "prediction_season"]]
    missing_pool = context.merge(
        model[["player_id", "prediction_season"]],
        on=["player_id", "prediction_season"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    missing_pool = missing_pool[missing_pool._merge == "left_only"]
    issues = pd.concat(
        [
            long.loc[duplicate | self_rows | position_bad].assign(
                issue="duplicate_self_or_wrong_position"
            ),
            missing_pool.assign(issue="no_eligible_competitor_pool"),
        ],
        ignore_index=True,
        sort=False,
    )
    findings = [
        _finding(
            "competition",
            "duplicate_competitors",
            "critical",
            duplicate.sum(),
            len(long),
            "Competitors must be unique within player-season.",
        ),
        _finding(
            "competition",
            "target_in_own_pool",
            "critical",
            self_rows.sum(),
            len(long),
            "Target player must be excluded.",
        ),
        _finding(
            "competition",
            "wrong_position_pool",
            "critical",
            int(position_bad.sum()),
            len(long),
            "QB/RB use same position; WR/TE intentionally share a receiver pool.",
        ),
        _finding(
            "competition",
            "noncontiguous_ranks",
            "high",
            len(bad_rank_keys),
            len(ranks),
            "Competition ranks must start at one and be contiguous.",
        ),
        _finding(
            "competition",
            "missing_competition_pool",
            "medium",
            len(missing_pool),
            len(context),
            "No other eligible Week-1 player existed in the defined position pool.",
        ),
    ]
    return findings, issues


def audit_rankings() -> tuple[list[dict], pd.DataFrame]:
    rankings = _read("attempt_2/raw/preseason_offensive_line_and_sos.parquet")
    context = _read("attempt_2/interim/prediction_context.parquet")[
        ["prediction_season", "target_team"]
    ].drop_duplicates()
    duplicate = rankings.duplicated(["prediction_season", "target_team"], keep=False)
    counts = (
        rankings.groupby("prediction_season")
        .agg(
            teams=("target_team", "nunique"),
            ol_unique=("preseason_offensive_line_rank", "nunique"),
            sos_unique=("preseason_strength_of_schedule_rank", "nunique"),
            ol_missing=("preseason_offensive_line_rank", lambda s: s.isna().sum()),
            sos_missing=(
                "preseason_strength_of_schedule_rank",
                lambda s: s.isna().sum(),
            ),
        )
        .reset_index()
    )
    bad = counts[
        (counts.teams != 32) | (counts.ol_missing > 0) | (counts.sos_missing > 0)
    ]
    tied = counts[(counts.ol_unique != 32) | (counts.sos_unique != 32)]
    out_of_bounds = (~rankings.preseason_offensive_line_rank.between(1, 32)) | (
        ~rankings.preseason_strength_of_schedule_rank.between(1, 32)
    )
    expected_ol = (32 - rankings.preseason_offensive_line_rank) / 31
    expected_sos = (rankings.preseason_strength_of_schedule_rank - 1) / 31
    bad_score = ~np.isclose(
        rankings.preseason_offensive_line_score, expected_ol, atol=1e-12
    ) | ~np.isclose(
        rankings.preseason_strength_of_schedule_score, expected_sos, atol=1e-12
    )
    unmatched = context.merge(
        rankings[["prediction_season", "target_team"]],
        on=["prediction_season", "target_team"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = unmatched[unmatched._merge == "left_only"]
    findings = [
        _finding(
            "ol_sos",
            "duplicate_team_season_rankings",
            "critical",
            duplicate.sum(),
            len(rankings),
            "One ranking row is required per team-season.",
        ),
        _finding(
            "ol_sos",
            "complete_32_team_seasons",
            "critical",
            len(bad),
            rankings.prediction_season.nunique(),
            "Each season must contain 32 teams with nonmissing OL and SOS ranks.",
        ),
        _finding(
            "ol_sos",
            "rank_bounds",
            "critical",
            out_of_bounds.sum(),
            len(rankings),
            "OL and SOS ranks must be between 1 and 32.",
        ),
        _finding(
            "ol_sos",
            "score_transform_consistency",
            "critical",
            bad_score.sum(),
            len(rankings),
            "Scores must reproduce the documented rank direction exactly.",
        ),
        _finding(
            "ol_sos",
            "tied_rank_values",
            "low",
            len(tied),
            rankings.prediction_season.nunique(),
            "The source model assigned tied integer ranks in these seasons. Ties do not break joins or score construction, but should be documented.",
        ),
        _finding(
            "ol_sos",
            "context_join_coverage",
            "critical",
            len(unmatched),
            len(context),
            "Every prediction team-season must match a ranking row.",
        ),
    ]
    return findings, pd.concat(
        [
            bad.assign(issue="incomplete_season_ranking_set"),
            tied.assign(issue="tied_rank_values"),
            rankings[out_of_bounds | bad_score].assign(issue="invalid_rank_or_score"),
            unmatched.assign(issue="unmatched_context_team"),
        ],
        ignore_index=True,
        sort=False,
    )


def audit_cohort() -> tuple[list[dict], pd.DataFrame]:
    all_rows = _read("interim/player_seasons_all.parquet")
    targets = _read("processed/player_season_targets.parquet")
    context = _read("attempt_2/interim/prediction_context.parquet")
    target_keys = targets[["player_id", "season"]].assign(in_target=1)
    detail = all_rows.merge(
        target_keys, on=["player_id", "season"], how="left", validate="one_to_one"
    )
    detail["included_target"] = detail.in_target.fillna(0).astype(bool)
    detail["target_exclusion_reason"] = np.where(
        detail.included_target, "included", "fewer_than_4_games"
    )
    rows = []
    for position in POSITIONS:
        all_position = detail[detail.position == position]
        eligible = targets[targets.position == position]
        ctx = context[context.position == position]
        model = _read(
            f"attempt_2/processed/{position.lower()}_attempt2_modeling_dataset.parquet"
        )
        validation = model[
            (model.prediction_season.between(2012, 2025))
            & model.previous_season_ppg.notna()
        ]
        rows.append(
            {
                "position": position,
                "all_player_seasons": len(all_position),
                "target_rows_4plus_games": len(eligible),
                "removed_under_4_games": len(all_position) - len(eligible),
                "context_rows": len(ctx),
                "outer_validation_rows": len(validation),
            }
        )
    summary = pd.DataFrame(rows)
    removed = len(detail) - detail.included_target.sum()
    findings = [
        _finding(
            "cohort",
            "target_minimum_games_filter",
            "high",
            int(removed),
            len(detail),
            "Evaluation is conditional on playing at least four games in the season being predicted; this uses future availability and excludes many severe injury/roster outcomes.",
        ),
        _finding(
            "cohort",
            "week1_context_exclusions",
            "high",
            len(targets) - len(context),
            len(targets),
            "Eligible targets without an accepted Week-1 roster assignment are excluded before Attempt 2 modeling.",
        ),
    ]
    detail = detail[~detail.included_target][
        [
            "player_id",
            "season",
            "player_name",
            "position",
            "team",
            "games_played",
            "target_exclusion_reason",
        ]
    ]
    return findings, summary, detail


def run_data_quality_audit() -> dict[str, Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    findings = []
    identifier_findings, position_issues = audit_identifiers_and_positions()
    findings += identifier_findings
    team_findings, team_issues, multi_team = audit_team_assignments()
    findings += team_findings
    qb_findings, qb_detail, qb_coverage = audit_projected_qbs()
    findings += qb_findings
    qbr_findings, qbr_detail = audit_qbr_missingness()
    findings += qbr_findings
    competition_findings, competition_issues = audit_competition()
    findings += competition_findings
    ranking_findings, ranking_issues = audit_rankings()
    findings += ranking_findings
    cohort_findings, cohort_summary, cohort_removed = audit_cohort()
    findings += cohort_findings
    tables = {
        "data_quality_findings.csv": pd.DataFrame(findings),
        "identifier_position_issues.csv": position_issues,
        "team_assignment_issues.csv": team_issues,
        "multi_team_seasons.csv": multi_team,
        "projected_qb_posthoc_audit.csv": qb_detail,
        "projected_qb_coverage_issues.csv": qb_coverage,
        "qbr_missingness_reasons.csv": qbr_detail,
        "competition_issues.csv": competition_issues,
        "ol_sos_issues.csv": ranking_issues,
        "cohort_summary.csv": cohort_summary,
        "cohort_removed_under_four_games.csv": cohort_removed,
    }
    for name, frame in tables.items():
        frame.to_csv(TABLES_DIR / name, index=False)
    findings_frame = tables["data_quality_findings.csv"]
    qbr_counts = qbr_detail.missing_reason.value_counts().to_dict()
    qb_match = qb_detail.projected_matches_actual_primary_qb.mean()
    lines = [
        "EXISTING DATA QUALITY AUDIT",
        "",
        "Scope: existing Attempts 1, 2, and 2.1 inputs only. No model tables were changed.",
        "Post-hoc actual team/QB comparisons are diagnostics and must not be used as preseason features.",
        "",
        "EXECUTIVE FINDING",
        "Core identifiers, joins, competition construction, and OL/SOS coverage are tested structurally below. The most material risk is cohort selection: evaluation requires four games in the future target season and therefore omits short/injured seasons.",
        "",
        "SUMMARY CHECKS",
    ]
    for row in findings_frame.itertuples():
        lines.append(
            f"- [{row.severity.upper()}] {row.category}/{row.check}: {row.affected_rows}/{row.total_rows} ({row.affected_rate:.1%}) — {row.detail}"
        )
    lines += [
        "",
        "QBR MISSINGNESS",
        json.dumps(qbr_counts, indent=2),
        "",
        f"Projected QB matched the eventual team leader in pass attempts in {qb_match:.1%} of auditable team-seasons. This measures preseason uncertainty, not leakage-safe correctness.",
        "",
        "RECOMMENDED ACTIONS",
        "1. Keep the existing identifier, competition, and OL/SOS joins if their structural checks pass.",
        "2. Before adding new features, manually review the exported position mismatches and preseason-team disagreements.",
        "3. Treat no-QBR-history as a football state (rookie/backup/no prior qualifying play), not merely a value to median-impute.",
        "4. Report current accuracy as conditional on at least four target-season games.",
        "5. For the next experiment, create a second all-rostered-player evaluation cohort with availability/zero-game outcomes handled explicitly.",
    ]
    report = OUTPUT_DIR / "existing_data_quality_audit.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = OUTPUT_DIR / "audit_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "read_only_inputs": True,
                "posthoc_values_used_as_features": False,
                "tables": sorted(tables),
                "report": report.name,
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "report": report,
        "findings": TABLES_DIR / "data_quality_findings.csv",
        "cohort": TABLES_DIR / "cohort_summary.csv",
    }


if __name__ == "__main__":
    run_data_quality_audit()
