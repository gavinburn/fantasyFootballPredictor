"""Contracts for the targeted existing-data audit."""

import pandas as pd

from src.config import REPORTS_DIR
from src.data_quality_audit import (
    audit_competition,
    audit_identifiers_and_positions,
    audit_rankings,
    audit_team_assignments,
)


def test_position_issue_export_keeps_id_and_adds_name() -> None:
    _, issues = audit_identifiers_and_positions()
    position_issues = issues[
        issues["issue"] == "target_position_differs_from_week1_roster_position"
    ]
    assert {"player_id", "player_name"}.issubset(position_issues.columns)
    assert position_issues["player_id"].notna().all()
    assert position_issues["player_name"].notna().all()


def test_team_issue_export_keeps_id_and_adds_name() -> None:
    _, issues, _ = audit_team_assignments()
    first_team_issues = issues[issues["target_differs_from_first_actual_team"]]
    assert {"player_id", "player_name"}.issubset(first_team_issues.columns)
    assert first_team_issues["player_id"].notna().all()
    assert first_team_issues["player_name"].notna().all()


def test_competition_structure_has_no_critical_issues() -> None:
    findings, _ = audit_competition()
    critical = [row for row in findings if row["severity"] == "critical"]
    assert all(row["affected_rows"] == 0 for row in critical)


def test_rankings_join_every_context_team_season() -> None:
    findings, _ = audit_rankings()
    coverage = next(row for row in findings if row["check"] == "context_join_coverage")
    assert coverage["affected_rows"] == 0


def test_audit_records_conditional_evaluation_cohort() -> None:
    findings = pd.read_csv(
        REPORTS_DIR / "data_quality_audit/tables/data_quality_findings.csv"
    )
    cohort = findings[findings.check == "target_minimum_games_filter"].iloc[0]
    assert cohort.affected_rows > 0
    assert cohort.severity == "high"
