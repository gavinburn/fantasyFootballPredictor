"""Central project configuration."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DATA_DIR = DATA_DIR / "predictions"
REPORTS_DIR = REPOSITORY_ROOT / "reports"
REPORT_TABLES_DIR = REPORTS_DIR / "tables"
REPORT_EQUATIONS_DIR = REPORTS_DIR / "equations"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = REPOSITORY_ROOT / "models"

# Attempt 2 is intentionally isolated from the V1 paths above. Phase 2 code may read
# V1 inputs, but every generated artifact is written below one of these directories.
ATTEMPT2_DATA_DIR = DATA_DIR / "attempt_2"
ATTEMPT2_RAW_DATA_DIR = ATTEMPT2_DATA_DIR / "raw"
ATTEMPT2_INTERIM_DATA_DIR = ATTEMPT2_DATA_DIR / "interim"
ATTEMPT2_PROCESSED_DATA_DIR = ATTEMPT2_DATA_DIR / "processed"
ATTEMPT2_PREDICTIONS_DATA_DIR = ATTEMPT2_DATA_DIR / "predictions"
ATTEMPT2_REPORTS_DIR = REPORTS_DIR / "attempt_2"
ATTEMPT2_REPORT_TABLES_DIR = ATTEMPT2_REPORTS_DIR / "tables"
ATTEMPT2_REPORT_FIGURES_DIR = ATTEMPT2_REPORTS_DIR / "figures"
ATTEMPT2_MODELS_DIR = MODELS_DIR / "attempt_2"

ATTEMPT21_DATA_DIR = DATA_DIR / "attempt_2_1"
ATTEMPT21_PROCESSED_DATA_DIR = ATTEMPT21_DATA_DIR / "processed"
ATTEMPT21_PREDICTIONS_DATA_DIR = ATTEMPT21_DATA_DIR / "predictions"
ATTEMPT21_REPORTS_DIR = REPORTS_DIR / "attempt_2_1"
ATTEMPT21_REPORT_TABLES_DIR = ATTEMPT21_REPORTS_DIR / "tables"
ATTEMPT21_REPORT_FIGURES_DIR = ATTEMPT21_REPORTS_DIR / "figures"
ATTEMPT21_FIGURE_DATA_DIR = ATTEMPT21_REPORTS_DIR / "figure_data"
ATTEMPT21_MODELS_DIR = MODELS_DIR / "attempt_2_1"

# Attempt 3 is the isolated veteran-cleanup and preseason-market experiment.
ATTEMPT3_DATA_DIR = DATA_DIR / "attempt_3"
ATTEMPT3_RAW_DATA_DIR = ATTEMPT3_DATA_DIR / "raw"
ATTEMPT3_PROCESSED_DATA_DIR = ATTEMPT3_DATA_DIR / "processed"
ATTEMPT3_PREDICTIONS_DATA_DIR = ATTEMPT3_DATA_DIR / "predictions"
ATTEMPT3_REPORTS_DIR = REPORTS_DIR / "attempt_3"
ATTEMPT3_REPORT_TABLES_DIR = ATTEMPT3_REPORTS_DIR / "tables"
ATTEMPT3_MODELS_DIR = MODELS_DIR / "attempt_3"

# Isolated candidate-group experiments 3-5. These read the Attempt 3 structural
# tables but never overwrite any prior attempt.
FEATURE35_DATA_DIR = DATA_DIR / "feature_experiments_3_5"
FEATURE35_RAW_DATA_DIR = FEATURE35_DATA_DIR / "raw"
FEATURE35_PROCESSED_DATA_DIR = FEATURE35_DATA_DIR / "processed"
FEATURE35_PREDICTIONS_DATA_DIR = FEATURE35_DATA_DIR / "predictions"
FEATURE35_REPORTS_DIR = REPORTS_DIR / "feature_experiments_3_5"
FEATURE35_REPORT_TABLES_DIR = FEATURE35_REPORTS_DIR / "tables"
FEATURE35_MODELS_DIR = MODELS_DIR / "feature_experiments_3_5"

# Position-specific feature set selected after the Attempt 3 structural/market
# ablations and feature experiments 3-5. Historical experiment artifacts remain
# immutable; this directory contains only the combined, promoted candidate.
SELECTED_DATA_DIR = DATA_DIR / "selected_feature_set"
SELECTED_PREDICTIONS_DATA_DIR = SELECTED_DATA_DIR / "predictions"
SELECTED_REPORTS_DIR = REPORTS_DIR / "selected_feature_set"
SELECTED_REPORT_TABLES_DIR = SELECTED_REPORTS_DIR / "tables"
SELECTED_MODELS_DIR = MODELS_DIR / "selected_feature_set"

# First post-V2 nonlinear experiment. It reads the frozen V2 feature inputs and
# predictions but never overwrites the operational V2 artifacts.
DECISION_TREE_DATA_DIR = DATA_DIR / "decision_tree"
DECISION_TREE_PREDICTIONS_DATA_DIR = DECISION_TREE_DATA_DIR / "predictions"
DECISION_TREE_REPORTS_DIR = REPORTS_DIR / "decision_tree"
DECISION_TREE_REPORT_TABLES_DIR = DECISION_TREE_REPORTS_DIR / "tables"
DECISION_TREE_REPORT_FIGURES_DIR = DECISION_TREE_REPORTS_DIR / "figures"
DECISION_TREE_MODELS_DIR = MODELS_DIR / "decision_tree"

# Bagged-tree experiment built on the same frozen V2 contracts.
RANDOM_FOREST_DATA_DIR = DATA_DIR / "random_forest"
RANDOM_FOREST_PREDICTIONS_DATA_DIR = RANDOM_FOREST_DATA_DIR / "predictions"
RANDOM_FOREST_REPORTS_DIR = REPORTS_DIR / "random_forest"
RANDOM_FOREST_REPORT_TABLES_DIR = RANDOM_FOREST_REPORTS_DIR / "tables"
RANDOM_FOREST_REPORT_FIGURES_DIR = RANDOM_FOREST_REPORTS_DIR / "figures"
RANDOM_FOREST_MODELS_DIR = MODELS_DIR / "random_forest"

# Sequential boosted-tree experiment on the frozen V2 feature contracts.
GRADIENT_BOOSTING_DATA_DIR = DATA_DIR / "gradient_boosting"
GRADIENT_BOOSTING_PREDICTIONS_DATA_DIR = GRADIENT_BOOSTING_DATA_DIR / "predictions"
GRADIENT_BOOSTING_REPORTS_DIR = REPORTS_DIR / "gradient_boosting"
GRADIENT_BOOSTING_REPORT_TABLES_DIR = GRADIENT_BOOSTING_REPORTS_DIR / "tables"
GRADIENT_BOOSTING_REPORT_FIGURES_DIR = GRADIENT_BOOSTING_REPORTS_DIR / "figures"
GRADIENT_BOOSTING_MODELS_DIR = MODELS_DIR / "gradient_boosting"

# Historical market-ranking benchmark. Raw files are archived responses from the
# free Fantasy Football Calculator API; all comparisons use frozen V2 outputs.
ADP_BENCHMARK_DATA_DIR = DATA_DIR / "adp_benchmark"
ADP_BENCHMARK_RAW_DATA_DIR = ADP_BENCHMARK_DATA_DIR / "raw"
ADP_BENCHMARK_PROCESSED_DATA_DIR = ADP_BENCHMARK_DATA_DIR / "processed"
ADP_BENCHMARK_REPORTS_DIR = REPORTS_DIR / "adp_benchmark"
ADP_BENCHMARK_REPORT_TABLES_DIR = ADP_BENCHMARK_REPORTS_DIR / "tables"

# Isolated TE receiving-role and mixture-model experiment.
TE_ROLE_DATA_DIR = DATA_DIR / "te_role_experiment"
TE_ROLE_PROCESSED_DATA_DIR = TE_ROLE_DATA_DIR / "processed"
TE_ROLE_PREDICTIONS_DATA_DIR = TE_ROLE_DATA_DIR / "predictions"
TE_ROLE_REPORTS_DIR = REPORTS_DIR / "te_role_experiment"
TE_ROLE_REPORT_TABLES_DIR = TE_ROLE_REPORTS_DIR / "tables"
TE_ROLE_REPORT_FIGURES_DIR = TE_ROLE_REPORTS_DIR / "figures"
TE_ROLE_MODELS_DIR = MODELS_DIR / "te_role_experiment"

# Isolated experiment that repurposes the rejected TE role classifier only as a
# preseason roster-competition utility. Frozen V1 remains the scoring base.
TE_COMPETITION_DATA_DIR = DATA_DIR / "te_competition_experiment"
TE_COMPETITION_PROCESSED_DATA_DIR = TE_COMPETITION_DATA_DIR / "processed"
TE_COMPETITION_PREDICTIONS_DATA_DIR = TE_COMPETITION_DATA_DIR / "predictions"
TE_COMPETITION_REPORTS_DIR = REPORTS_DIR / "te_competition_experiment"
TE_COMPETITION_REPORT_TABLES_DIR = TE_COMPETITION_REPORTS_DIR / "tables"

# Isolated, point-in-time contract-capital ablation. Contract rows signed during
# prediction season t are deliberately excluded because the source has no exact
# signing date with which to enforce the preseason cutoff.
CONTRACT_EXPERIMENT_DATA_DIR = DATA_DIR / "contract_experiment"
CONTRACT_EXPERIMENT_RAW_DATA_DIR = CONTRACT_EXPERIMENT_DATA_DIR / "raw"
CONTRACT_EXPERIMENT_PROCESSED_DATA_DIR = CONTRACT_EXPERIMENT_DATA_DIR / "processed"
CONTRACT_EXPERIMENT_PREDICTIONS_DATA_DIR = CONTRACT_EXPERIMENT_DATA_DIR / "predictions"
CONTRACT_EXPERIMENT_REPORTS_DIR = REPORTS_DIR / "contract_experiment"
CONTRACT_EXPERIMENT_REPORT_TABLES_DIR = CONTRACT_EXPERIMENT_REPORTS_DIR / "tables"

# Follow-up constrained TE competition ablation. This consumes only frozen V1,
# cross-fitted receiving-competition utilities, and point-in-time contract context.
TE_CONSTRAINED_DATA_DIR = DATA_DIR / "te_constrained_competition"
TE_CONSTRAINED_PROCESSED_DATA_DIR = TE_CONSTRAINED_DATA_DIR / "processed"
TE_CONSTRAINED_PREDICTIONS_DATA_DIR = TE_CONSTRAINED_DATA_DIR / "predictions"
TE_CONSTRAINED_REPORTS_DIR = REPORTS_DIR / "te_constrained_competition"
TE_CONSTRAINED_REPORT_TABLES_DIR = TE_CONSTRAINED_REPORTS_DIR / "tables"

HISTORICAL_START_SEASON = 2006
HISTORICAL_END_SEASON = 2025
PREDICTION_SEASON = 2026
POSITIONS = ("QB", "RB", "WR", "TE")
MIN_TARGET_GAMES = 4
RANDOM_STATE = 446
