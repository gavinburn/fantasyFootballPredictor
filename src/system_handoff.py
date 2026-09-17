"""Generate the standalone model/feature handoff for external feature ideation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    ATTEMPT21_PREDICTIONS_DATA_DIR,
    ATTEMPT21_REPORT_TABLES_DIR,
    PREDICTIONS_DATA_DIR,
    REPOSITORY_ROOT,
)

OUTPUT = REPOSITORY_ROOT / "fantasy_football_model_feature_handoff.txt"
BASELINE_TABLE = (
    ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_same_cohort_baseline_comparison.csv"
)
TOP_N = {"QB": 10, "RB": 20, "WR": 20, "TE": 10}

FEATURE_DEFINITIONS = {
    "previous_season_ppg": "Standard-scoring fantasy points / games played in t-1.",
    "three_season_mean_ppg": "Mean available standard PPG from exact seasons t-1, t-2 and t-3; missing seasons are ignored.",
    "previous_season_games_played": "Unique regular-season games played in t-1.",
    "age_entering_season": "Whole-year age on September 1 of prediction season t.",
    "years_experience_entering_season": "t minus the player master rookie season.",
    "previous_passing_attempts_per_game": "Passing attempts(t-1) / games(t-1).",
    "previous_passing_yards_per_attempt": "Passing yards(t-1) / passing attempts(t-1); null when denominator is zero.",
    "previous_passing_touchdowns_per_game": "Passing TD(t-1) / games(t-1).",
    "previous_interceptions_per_game": "Passing interceptions(t-1) / games(t-1).",
    "previous_qb_rushing_attempts_per_game": "QB rush attempts(t-1) / games(t-1).",
    "previous_qb_rushing_yards_per_game": "QB rush yards(t-1) / games(t-1).",
    "previous_qb_rushing_touchdowns_per_game": "QB rush TD(t-1) / games(t-1).",
    "previous_rushing_attempts_per_game": "Rush attempts(t-1) / games(t-1).",
    "previous_rushing_yards_per_attempt": "Rush yards(t-1) / rush attempts(t-1); null when denominator is zero.",
    "previous_rushing_touchdowns_per_game": "Rush TD(t-1) / games(t-1).",
    "previous_targets_per_game": "Targets(t-1) / games(t-1).",
    "previous_receiving_yards_per_game": "Receiving yards(t-1) / games(t-1).",
    "previous_receiving_yards_per_target": "Receiving yards(t-1) / targets(t-1); null when denominator is zero.",
    "previous_receiving_touchdowns_per_game": "Receiving TD(t-1) / games(t-1).",
    "previous_rushing_yards_per_game": "Rush yards(t-1) / games(t-1).",
    "previous_opportunities_per_game": "[Rush attempts(t-1) + targets(t-1)] / games(t-1).",
    "previous_target_opportunity_rate": "Targets per game(t-1) / opportunities per game(t-1); safe null division. Used only in the RB total-plus-mix representation.",
    "target_team_plays_per_game_tminus1": "Target team offensive plays in t-1 / target team games in t-1; plays = dropbacks + carries.",
    "target_team_dropbacks_per_game_tminus1": "Target team [pass attempts + sacks suffered] in t-1 / games.",
    "target_team_carries_per_game_tminus1": "Target team carries in t-1 / games.",
    "target_team_pass_rate_tminus1": "Target team dropbacks(t-1) / [dropbacks(t-1) + carries(t-1)].",
    "target_team_passing_tds_per_game_tminus1": "Target team passing TD(t-1) / games.",
    "target_team_rushing_tds_per_game_tminus1": "Target team rushing TD(t-1) / games.",
    "target_team_pass_epa_per_dropback_tminus1": "Target team passing EPA(t-1) / dropbacks(t-1).",
    "target_team_rush_epa_per_carry_tminus1": "Target team rushing EPA(t-1) / carries(t-1).",
    "previous_qb_dropback_share": "Player [attempts + sacks](t-1) / matched-team [attempts + sacks](t-1), summed only over the player's actual team-weeks.",
    "previous_carry_share": "Player carries(t-1) / matched-team carries(t-1), supporting traded players week by week.",
    "previous_target_share": "Player targets(t-1) / matched-team targets(t-1).",
    "previous_opportunity_share": "Player [carries + targets](t-1) / matched-team [carries + targets](t-1).",
    "previous_air_yards_share": "Player receiving air yards(t-1) / matched-team receiving air yards(t-1).",
    "top_competitor_previous_total_points": "Highest-ranked eligible same-pool teammate's total standard fantasy points in t-1.",
    "top_competitor_previous_ppg": "Highest-ranked eligible same-pool teammate's PPG in t-1.",
    "top_competitor_age": "Top competitor's age on September 1 of t.",
    "second_competitor_previous_ppg": "Second-ranked eligible competitor's PPG in t-1.",
    "competition_pool_size": "Count of other eligible Week-1 roster players in the position pool.",
    "competition_no_history_count": "Count of pool competitors with no eligible t-1 player-season history.",
    "top_competitor_is_wr": "1 if top WR/TE pooled competitor is a WR; TE is the reference category.",
    "second_competitor_is_wr": "1 if second WR/TE pooled competitor is a WR; TE is the reference category.",
    "projected_qb_previous_qbr": "Projected starting QB's ESPN Total QBR in t-1; missing when the source has no prior row.",
    "projected_qb_previous_qb_plays": "Projected starting QB's QBR-source QB plays in t-1; currently filled with zero when the QBR row is absent.",
    "projected_qb_has_previous_qbr": "1 when projected_qb_previous_qbr is present, else 0.",
    "changed_team": "1 when season-t Week-1 roster team differs from the player's selected primary team in t-1; null when no previous team exists.",
    "preseason_offensive_line_score": "Machine-learning preseason OL rank transformed as (32-rank)/31; 1 is best and 0 is worst.",
    "preseason_strength_of_schedule_score": "Machine-learning preseason SOS rank transformed as (rank-1)/31; 0 is hardest and 1 is easiest.",
}

CALCULATED_NOT_CURRENT_CANDIDATES = {
    "two_season_mean_ppg": "Mean available PPG from t-1 and t-2.",
    "two_season_mean_games_played": "Mean games from t-1 and t-2.",
    "three_season_mean_games_played": "Mean games from t-1 through t-3.",
    "previous_year_ppg_change": "PPG(t-1) - PPG(t-2).",
    "prior_seasons_count": "Count of earlier eligible player-season rows.",
    "draft_round": "NFL draft round from player master.",
    "draft_pick": "Overall NFL draft pick from player master.",
    "undrafted_indicator": "1 when draft round and pick are both missing.",
    "preseason_offensive_line_rank": "Original OL ordinal rank, retained for audit but models use score.",
    "preseason_strength_of_schedule_rank": "Original SOS ordinal rank, retained for audit but models use score.",
    "top_competitor_is_te": "Complement of top_competitor_is_wr in the WR/TE pool; calculated in Attempt 2 but removed from Attempt 2.1 to prevent exact dependency.",
    "second_competitor_is_te": "Complement of second_competitor_is_wr; calculated but removed from Attempt 2.1.",
    "offensive_line_rank_available": "Coverage flag supplied with preseason OL rankings; currently always one in the complete historical source.",
    "strength_of_schedule_rank_available": "Coverage flag supplied with preseason SOS rankings; currently always one.",
    "context_cutoff_date": "September 1 of prediction season t; audit/timing field, never a numeric model input.",
    "previous_primary_team": "Selected t-1 primary club used to calculate changed_team; tracking/join field.",
    "projected_starting_qb_id": "Season-t preseason projected starter GSIS ID; join field, never passed directly to the regressor.",
    "source_method": "Ranking provenance label (preseason machine-learning model); audit-only.",
    "rank_direction": "Audit label documenting OL 1=best and SOS 1=hardest.",
}


def _fold_stats(
    frame: pd.DataFrame, prediction: str, position: str
) -> tuple[float, float]:
    spearman, overlap = [], []
    for _, fold in frame.groupby("season"):
        value = (
            fold["actual_ppg"].corr(fold[prediction], method="spearman")
            if fold[prediction].nunique() > 1
            else np.nan
        )
        spearman.append(value)
        n = min(TOP_N[position], len(fold))
        actual_top = set(fold.nlargest(n, "actual_ppg")["player_id"])
        predicted_top = set(fold.nlargest(n, prediction)["player_id"])
        overlap.append(len(actual_top & predicted_top) / n)
    valid = [value for value in spearman if pd.notna(value)]
    return (float(np.mean(valid)) if valid else np.nan, float(np.mean(overlap)))


def same_cohort_comparison() -> pd.DataFrame:
    current = pd.read_parquet(
        ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_predictions.parquet"
    )
    current = current[current["candidate"] == "ADAPTIVE_POSITION_SELECTED"][
        ["player_id", "season", "position", "actual_ppg", "predicted_ppg"]
    ].rename(columns={"predicted_ppg": "attempt21_adaptive"})
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    frozen = frozen[frozen["model_name"] == "A_v1_same_cohort"][
        ["player_id", "season", "position", "actual_ppg", "predicted_ppg"]
    ].rename(columns={"predicted_ppg": "frozen_same_cohort_v1"})
    original_v1 = pd.read_parquet(
        PREDICTIONS_DATA_DIR / "historical_linear_regression_predictions.parquet"
    )[["player_id", "season", "position", "actual_ppg", "predicted_ppg"]].rename(
        columns={"predicted_ppg": "original_v1"}
    )
    naive = (
        pd.read_parquet(
            PREDICTIONS_DATA_DIR / "historical_baseline_predictions.parquet"
        )
        .pivot(
            index=["player_id", "season", "position", "actual_ppg"],
            columns="model_name",
            values="predicted_ppg",
        )
        .reset_index()
    )
    paired = (
        current.merge(
            frozen,
            on=["player_id", "season", "position", "actual_ppg"],
            validate="one_to_one",
        )
        .merge(
            original_v1,
            on=["player_id", "season", "position", "actual_ppg"],
            validate="one_to_one",
        )
        .merge(
            naive,
            on=["player_id", "season", "position", "actual_ppg"],
            validate="one_to_one",
        )
    )
    models = [
        "attempt21_adaptive",
        "frozen_same_cohort_v1",
        "original_v1",
        "previous_season_ppg",
        "historical_position_mean",
    ]
    rows = []
    for position, group in paired.groupby("position"):
        current_season_mae = (
            group.assign(
                absolute_error=(group["attempt21_adaptive"] - group["actual_ppg"]).abs()
            )
            .groupby("season")["absolute_error"]
            .mean()
        )
        for model in models:
            error = group[model] - group["actual_ppg"]
            spearman, top_n = _fold_stats(group, model, position)
            model_season_mae = (
                group.assign(absolute_error=error.abs())
                .groupby("season")["absolute_error"]
                .mean()
            )
            rows.append(
                {
                    "position": position,
                    "model": model,
                    "rows": len(group),
                    "mae": error.abs().mean(),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mean_fold_spearman": spearman,
                    "mean_fold_top_n_overlap": top_n,
                    "attempt21_mae_improvement_vs_model": error.abs().mean()
                    - (group["attempt21_adaptive"] - group["actual_ppg"]).abs().mean(),
                    "attempt21_seasons_better_than_model": int(
                        (current_season_mae < model_season_mae).sum()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(BASELINE_TABLE, index=False)
    return result


def _feature_catalog(candidates: dict) -> list[str]:
    lines = [
        "7. COMPLETE MODELED FEATURE CATALOG",
        "",
        "Every unique input that appears in at least one Attempt 2.1 candidate is listed below. All statistical quantities are point-in-time: player and team statistics are from t-1 or earlier; preseason roster/ranking context is for t and must exist before the season.",
        "",
    ]
    for position, specifications in candidates.items():
        unique = []
        for features in specifications.values():
            unique.extend(feature for feature in features if feature not in unique)
        lines.append(f"{position}: {len(unique)} unique candidate features")
        for feature in unique:
            lines.append(f"  - {feature}: {FEATURE_DEFINITIONS[feature]}")
        lines.append("")
    lines.extend(
        [
            "Calculated and retained in source/modeling tables but not used by the current Attempt 2.1 candidate manifest:",
            *[
                f"  - {feature}: {definition}"
                for feature, definition in CALCULATED_NOT_CURRENT_CANDIDATES.items()
            ],
            "",
        ]
    )
    return lines


def _candidate_catalog(candidates: dict) -> list[str]:
    lines = [
        "8. EXACT POSITION-SPECIFIC CANDIDATE SPECIFICATIONS",
        "",
        "These are the exact ordered feature lists supplied to scikit-learn. OLS and Ridge are tested for every specification. Elastic Net is limited to predefined core/reduced/full candidates to control computation and model-search breadth.",
        "",
    ]
    for position, specifications in candidates.items():
        lines.append(position)
        for name, features in specifications.items():
            lines.append(f"  {name} ({len(features)} features)")
            lines.append("    " + ", ".join(features))
        lines.append("")
    return lines


def generate_system_handoff() -> Path:
    candidates = json.loads(
        (ATTEMPT21_REPORT_TABLES_DIR / "candidate_feature_sets.json").read_text()
    )
    comparison = same_cohort_comparison()
    attempt_summary = pd.read_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_model_comparison.csv"
    )
    attempt_summary = attempt_summary[
        attempt_summary["candidate"] == "ADAPTIVE_POSITION_SELECTED"
    ]
    decisions = pd.read_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_position_decisions.csv"
    )
    deployment = pd.read_csv(
        ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_deployment_models.csv"
    )
    folds = pd.read_csv(ATTEMPT21_REPORT_TABLES_DIR / "attempt2_1_fold_metrics.csv")
    selected = folds[folds["candidate"] == "ADAPTIVE_POSITION_SELECTED"]

    lines = [
        "FANTASY FOOTBALL PREDICTION SYSTEM",
        "DETAILED INPUT, FEATURE, DATA-CORRECTION, AND RESULT HANDOFF",
        "Generated from the current corrected Attempt 2.1 artifacts.",
        "",
        "1. PURPOSE OF THIS DOCUMENT",
        "",
        "This document is a standalone technical description intended for an AI or researcher proposing additional features. It distinguishes raw inputs, calculated features, tested feature subsets, evaluation design, corrected data issues, unresolved limitations, and current empirical results. Repository access should not be necessary to understand the system.",
        "",
        "2. PREDICTION OBJECTIVE AND TARGET",
        "",
        "The system predicts each player's next-season regular-season fantasy points per game (PPG), separately for QB, RB, WR, and TE. The target is standard non-PPR scoring:",
        "  passing yards / 25",
        "  + 4 * passing touchdowns",
        "  - 2 * interceptions",
        "  + rushing yards / 10",
        "  + 6 * rushing touchdowns",
        "  + receiving yards / 10",
        "  + 6 * receiving touchdowns",
        "  + 2 * passing/rushing/receiving two-point conversions",
        "  - 2 * fumbles lost.",
        "Receptions are worth zero. PPG = total regular-season standard points / unique games played.",
        "",
        "The target table requires at least four games in prediction season t. Therefore, reported accuracy is conditional on the player ultimately playing four or more games. This is an intentional scope choice, but membership cannot be known at the preseason forecast date and must be stated when interpreting results.",
        "",
        "Rookies and all other players without a non-null previous-season PPG are not included in outer-fold predictions. Rookie rows currently exist in historical training tables and can influence fits through imputation; a separate rookie/no-history system remains recommended.",
        "",
        "3. TIME DESIGN AND EVALUATION POPULATION",
        "",
        "Raw history covers 2006-2025. The corrected design uses 2006 only as historical source data. Prediction season 2007 is the first modeling row and uses 2006 as t-1. There are no 2006 target/model rows. Out-of-sample validation covers 2012-2025 (14 seasons). For test season t, outer training uses prediction seasons strictly earlier than t. Hyperparameters and candidate selection use expanding inner folds wholly inside that outer training period. The earliest inner validation season is 2008.",
        "",
        "Current Attempt 2.1 outer prediction counts after all exclusions: QB=512, RB=1,100, WR=1,741, TE=968. Every result comparison later in this document is paired on these exact player-position-season rows.",
        "",
        "Explicit season exclusions applied before both training and testing:",
        "  - 366 remaining prediction-season 2006 rows: removed because 2006 is history-only.",
        "  - 66 player-seasons whose season outcome position differs from the Week-1 preseason roster position: removed because midseason/hybrid position changes make position-specific evaluation ambiguous.",
        "  - 42 player-seasons rostered at the cutoff but waived before recording a game for that team: removed by user-defined scope. Two overlap the position list, so position plus first-team rules contain 106 unique player-season keys.",
        "  - Attempts 1 and 2 remain frozen; these corrections are isolated in Attempt 2.1.",
        "",
        "4. RAW AND EXTERNAL INPUTS",
        "",
        "A. Weekly player statistics, 2006-2025 (nflverse/nflreadpy; 356,565 rows). Relevant fields include player/team/position IDs, season/week/game, passing attempts/yards/TD/interceptions/sacks, carries/rushing yards/TD, targets/receptions/receiving yards/TD/air yards, two-point conversions, fumbles lost, and source fantasy fields. We recalculate standard fantasy points rather than trusting a PPR field.",
        "B. Player master (25,035 rows). Used for GSIS and ESPN IDs, display name, birth date, rookie season, draft round/pick, and identifier joins.",
        "C. Weekly team statistics, 2006-2025 (10,862 rows). Used for team plays, attempts, sacks, carries, TD, EPA, targets, and air yards denominators.",
        "D. Weekly historical rosters, 2006-2025 (782,594 rows). The prediction context uses eligible Week-1 statuses ACT, INA, RES, SUS, and EXE; CUT is excluded. One assignment per player-season is selected by status priority and stable team ordering.",
        "E. Historical depth charts, 2006-2024, plus a timestamped 2025 depth-chart snapshot. Used to identify one projected starting QB per team-season. Six documented preseason overrides repair known missing/mis-ranked historical opening depth charts.",
        "F. ESPN season-level Total QBR (1,523 rows). GSIS IDs map to ESPN IDs through the player master. Only t-1 QBR is used.",
        "G. Preseason offensive-line and strength-of-schedule ranks (640 rows = 32 teams * 20 seasons). They were produced by a separate machine-learning model using only information available before each season. OL rank 1 is best; SOS rank 1 is hardest. All team-season joins and score transformations pass structural audits. Tied integer ranks occur in 14 seasons and are retained/documented.",
        "H. 2026 roster snapshot exists for future prediction preparation but is not part of the 2012-2025 evaluation reported here.",
        "",
        "5. POINT-IN-TIME TEAM, ROSTER, AND COMPETITION CONSTRUCTION",
        "",
        "Season-t target team comes from the eligible Week-1 roster snapshot. Team aliases are normalized (for example OAK->LV, SD->LAC, STL/LA->LAR, JAC->JAX, WSH->WAS). Previous primary team is selected from t-1 weekly rows by most appearances, then most offensive opportunities, then latest week, then stable team order. The changed-team flag compares these two point-in-time assignments.",
        "",
        "For traded players, workload shares use each actual t-1 player-team-week joined to the matching team-week denominator before aggregation. This avoids assigning an entire traded season to one team's denominator.",
        "",
        "Competition pools use the season-t Week-1 roster, exclude the target player, and rank competitors by whether t-1 history exists, t-1 total points, t-1 PPG, then ID. QB competes with QB; RB with RB; WR and TE share one receiving competition pool. For WR/TE, TE is the dummy reference and only is-WR flags are retained.",
        "",
        "Projected-QB context is joined by prediction season, target team, and projected starter ID. Team environment statistics are always shifted exactly one season. OL and SOS are preseason season-t context.",
        "",
        "6. PREPROCESSING AND MODEL BOUNDARY",
        "",
        "Attempt 2.1 uses pandas for all new tables and scikit-learn pipelines. Identifiers and target columns never enter X. Within each inner/outer fit: numeric missing values receive training-only median imputation; features receive training-only standard scaling; then OLS, Ridge, or Elastic Net is fitted. DataFrames retain exact ordered names and saved pipelines expose feature_names_in_. Player ID plus prediction season is unique. All joins declare one-to-one, many-to-one, or many-to-many cardinality explicitly.",
        "",
    ]
    lines.extend(_feature_catalog(candidates))
    lines.extend(_candidate_catalog(candidates))
    lines.extend(
        [
            "9. MODEL SEARCH AND SELECTION",
            "",
            "OLS and Ridge are evaluated for every declared candidate. Ridge alpha grid: 0.01, 0.1, 1, 10, 100, 1000. Elastic Net alpha grid: 0.001, 0.01, 0.1, 1, 10; l1_ratio: 0.05, 0.25, 0.50, 0.75, 0.95. Elastic Net is restricted to selected core/reduced/full specifications.",
            "",
            "For each outer season, each candidate/method is tuned using the last up to five valid expanding inner seasons, with at least three required. Lowest mean inner MAE wins; mean inner Spearman and deterministic method/candidate ordering break ties. The ADAPTIVE_POSITION_SELECTED prediction for an outer fold is the candidate/method with the best inner-fold evidence only. Outer outcomes never choose that fold's model.",
            "",
            "Metrics: MAE (primary), RMSE, mean fold Spearman, mean fold Top-N overlap (QB/TE top 10; RB/WR top 20), paired player error, seasons improved, row bootstrap, and season-block bootstrap. Positive MAE improvement means the comparison model has higher error than Attempt 2.1.",
            "",
            "10. DATA AND MODEL ISSUES FIXED",
            "",
            "1. Boundary-year error: 2006 had incorrectly existed as a prediction row without 2005 history. It is now history-only; 2007 is first prediction year. This removed 366 residual 2006 rows and removed 32 artificial 2006 missing-QBR team-seasons from active scope.",
            "2. RB exact dependency: opportunities/game equaled carries/game + targets/game. Candidate sets now use either components without total, or total plus target-opportunity mix without both components.",
            "3. WR/TE exact dummy dependency: is-WR + is-TE = 1 in the receiver pool. TE is now the reference category; only is-WR is modeled.",
            "4. Position ambiguity: 66 mismatch seasons are removed from train and test, but other seasons for those player IDs remain.",
            "5. Waived-before-appearance cohort: 42 confirmed seasons are removed; 2 overlap the position list.",
            "6. Multi-team workload calculation: player weeks use their matching team-week denominators before seasonal aggregation.",
            "7. Team aliases, unique player-season keys, merge cardinality, projected-QB uniqueness, competition self-exclusion/rank continuity, and OL/SOS coverage/direction now have explicit audits.",
            "8. OLS instability: reduced nonredundant sets plus nested Ridge/Elastic Net were introduced. All corrected candidate matrices have full rank after excluding constants.",
            "9. Leakage-safe tuning: random CV was replaced by nested chronological expanding folds. Imputation, scaling, hyperparameters, and adaptive selection are training-only.",
            "10. Pandas migration: the frozen Attempt 2 OLS process was reproduced within 1e-10 before intentional cohort corrections.",
            "",
            "11. IMPORTANT OPEN LIMITATIONS FOR NEW-FEATURE DESIGN",
            "",
            "A. Future role allocation remains the dominant missing signal. Errors are much larger for expanded/collapsed roles than stable roles. Existing team totals describe environment but not who will receive starts, snaps, routes, carries, targets, red-zone work, or goal-line work.",
            "B. Rookies/no-history players receive no reported outer predictions. They should have a separate model/evaluation using draft capital, college production, ADP, depth chart and projected role. Current historical training still contains rookie rows imputed on missing history, creating a train/test population mismatch worth correcting.",
            "C. The four-game target rule makes evaluation conditional on future survival/availability. This is accepted project scope, but results are not accuracy for all players identifiable preseason.",
            "D. QBR missingness is heterogeneous. In active 2007-2025 scope, 96 of 606 projected-QB team-seasons have no t-1 QBR row. Cases include true rookies, veterans with no prior attempts, limited backups, injured veterans, and 24 cases with 100+ attempts omitted by the QBR source. The current binary presence flag and zero-filled QBR plays do not distinguish these states. Known t-1 attempts/dropbacks from weekly NFL data should replace false zero QBR plays, and categorical missing reasons should be tested.",
            "E. A single preseason target team supplies context for the entire player-season. Later trades cannot be represented dynamically. This is acceptable for a point-in-time preseason forecast but should be recognized.",
            "F. Previous primary team compresses a multi-team t-1 season to one club for changed_team, although workload shares themselves correctly use all matched team-weeks.",
            "G. OL/SOS are single ordinal-derived scores without uncertainty. Raw underlying model scores and prediction uncertainty would be preferable.",
            "H. Accepted QB/RB/WR season-block and row-bootstrap intervals still cross zero. Acceptance occurred because the predefined rule also allows at least 8 of 14 seasons improved while ranking thresholds hold. Treat gains as modest, not conclusive.",
            "I. The adaptive winner changes across seasons. This is an unbiased evaluation of inner selection, but it is not one fixed interpretable formula. Deployment uses the modal inner-selected candidate from recent folds and is documented below.",
            "",
            "12. CURRENT SAME-COHORT RESULTS",
            "",
            "Definitions:",
            "  attempt21_adaptive = candidate/method selected separately inside each outer fold using inner history only.",
            "  frozen_same_cohort_v1 = frozen Attempt 2 V1 OLS predictions, paired to the corrected current rows.",
            "  original_v1 = the first linear-regression attempt, paired to the corrected current rows.",
            "  previous_season_ppg = primary original naive baseline: predict PPG(t) = PPG(t-1).",
            "  historical_position_mean = secondary naive baseline: fold-local mean target PPG for the position; its within-fold Spearman is undefined because every player receives one value.",
            "",
        ]
    )
    for position in ("QB", "RB", "WR", "TE"):
        lines.append(position)
        position_rows = comparison[comparison["position"] == position]
        for row in position_rows.itertuples():
            spearman = (
                "N/A"
                if pd.isna(row.mean_fold_spearman)
                else f"{row.mean_fold_spearman:.4f}"
            )
            lines.append(
                f"  {row.model}: n={row.rows}; MAE={row.mae:.4f}; RMSE={row.rmse:.4f}; mean-fold Spearman={spearman}; mean-fold Top-N overlap={row.mean_fold_top_n_overlap:.4f}."
            )
        for baseline in (
            "frozen_same_cohort_v1",
            "original_v1",
            "previous_season_ppg",
            "historical_position_mean",
        ):
            row = position_rows[position_rows["model"] == baseline].iloc[0]
            percentage = row.attempt21_mae_improvement_vs_model / row.mae * 100
            lines.append(
                f"  Attempt 2.1 vs {baseline}: MAE improvement={row.attempt21_mae_improvement_vs_model:+.4f} PPG ({percentage:+.2f}%); Attempt 2.1 lower MAE in {int(row.attempt21_seasons_better_than_model)}/14 seasons."
            )
        detail = attempt_summary[attempt_summary["position"] == position].iloc[0]
        decision = decisions[decisions["position"] == position].iloc[0]
        lines.append(
            f"  Paired-vs-frozen-V1 uncertainty: season-block 95% [{detail.season_block_95_low:+.4f}, {detail.season_block_95_high:+.4f}]; row-bootstrap 95% [{detail.row_bootstrap_95_low:+.4f}, {detail.row_bootstrap_95_high:+.4f}]; median player improvement={detail.median_player_improvement:+.4f}; players improved={detail.players_improved_rate:.1%}; seasons improved={detail.seasons_improved}/14."
        )
        lines.append(
            f"  Decision={decision.decision}; Spearman change vs frozen V1={decision.spearman_change:+.4f}; Top-N overlap change={decision.top_n_overlap_change:+.4f}."
        )
        lines.append("")

    lines.extend(
        [
            "Interpretation of current results:",
            "  - QB improves MAE and Spearman versus both V1 versions and both naive baselines. The absolute gain over frozen V1 is modest and uncertainty crosses zero. The fixed predeclared QB_B_VOLUME OLS candidate is stronger diagnostically (MAE 3.4559; +0.0783 versus frozen V1; 13/14 seasons; season-block interval entirely positive), but choosing it after outer results would not be an unbiased adaptive-selection estimate.",
            "  - RB beats frozen same-cohort V1 by 0.0205 MAE and previous-season PPG by about 0.2107, but is nearly tied with original V1 (only about +0.0012 MAE) and has worse RMSE/ranking overlap than those V1 comparisons. Its accepted label should be treated cautiously.",
            "  - WR improves modestly versus frozen and original V1 and substantially versus previous-season PPG; ranking metrics also improve slightly. Confidence intervals narrowly cross zero.",
            "  - TE adaptive context is worse than frozen/original V1. The recommended TE model remains V1. This is evidence to reject broad TE context rather than evidence the experiment failed.",
            "",
            "13. ADAPTIVE SELECTION AND DEPLOYMENT DETAILS",
            "",
        ]
    )
    for position in ("QB", "RB", "WR", "TE"):
        counts = (
            selected[selected["position"] == position]["source_candidate"]
            .value_counts()
            .to_dict()
        )
        deployed = deployment[deployment["position"] == position].iloc[0]
        lines.append(f"{position} outer-fold inner-selected source frequencies:")
        for candidate, count in counts.items():
            lines.append(f"  {candidate}: {count}/14 folds")
        lines.append(
            f"  Saved recent-modal deployment artifact: {deployed.source_candidate}; alpha={deployed.alpha if pd.notna(deployed.alpha) else 'N/A'}; l1_ratio={deployed.l1_ratio if pd.notna(deployed.l1_ratio) else 'N/A'}; features={deployed.features}; promoted={deployed.promoted}. Recommended model label={deployed.recommended_model}."
        )
        lines.append("")

    lines.extend(
        [
            "14. HIGHEST-PRIORITY NEW FEATURE DIRECTIONS",
            "",
            "Features should be available before the season for every historical year and should first be tested as isolated groups. Highest expected value:",
            "  1. Preseason ADP or auction value as a market-information baseline.",
            "  2. Explicit depth-chart rank, expected starter/committee/backup status, and assignment confidence.",
            "  3. Prior snap share, route participation, routes per team dropback, and blocking-versus-route usage for TE.",
            "  4. Vacated carries, targets, routes, air yards, red-zone touches, and goal-line carries caused by offseason roster changes.",
            "  5. Preseason injuries, surgery recovery, PUP/IR status, expected Week-1 availability, and prior games missed.",
            "  6. Rookie-specific draft capital, college production/efficiency, age, combine measures, team investment, and ADP.",
            "  7. Coaching/play-caller/quarterback changes and prior scheme tendencies, defined point-in-time.",
            "  8. Projected-QB missing-history categories and weekly-stat attempts/dropbacks when QBR is unavailable.",
            "  9. Red-zone and end-zone opportunity shares, which should help touchdowns without duplicating generic volume.",
            "  10. Raw OL/SOS model predictions and uncertainty rather than rank-derived point estimates.",
            "",
            "Any new feature group should retain the 2007 start, 2012-2025 outer folds, training-only transformations, exact corrected cohort, position separation, frozen comparison models, and nested chronological tuning. It should be evaluated for MAE, RMSE, fold Spearman, Top-N overlap, season-block uncertainty, role-change slices, and coefficient stability. Do not use season-t realized snaps, injuries, roles, teams, or outcomes to construct preseason inputs.",
            "",
            "END OF HANDOFF",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(generate_system_handoff())
