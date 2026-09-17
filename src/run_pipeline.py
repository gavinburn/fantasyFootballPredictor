"""Run reproducible stages of the V1 fantasy-football experiment."""

from __future__ import annotations

import argparse
from collections.abc import Callable


def run_features() -> None:
    from src.build_features import run_build

    run_build()


def run_baselines() -> None:
    from src.baselines import run_baselines

    run_baselines()


def run_validate_linear() -> None:
    from src.train_models import run_training

    run_training()


def run_evaluate_v1() -> None:
    from src.evaluate import run_evaluation

    run_evaluation()


def run_error_analysis_v1() -> None:
    from src.error_analysis import run_error_analysis

    run_error_analysis()


def run_conclusion() -> None:
    from src.conclusion import write_first_attempt_conclusion

    write_first_attempt_conclusion()


def run_phase2_features() -> None:
    from src.phase2_features import run_phase2_feature_build

    run_phase2_feature_build()


def run_attempt2_models() -> None:
    from src.phase2_models import run_attempt2_models

    run_attempt2_models()


def run_attempt2_diagnostics() -> None:
    from src.phase2_diagnostics import run_attempt2_diagnostics

    run_attempt2_diagnostics()


def run_attempt21_features() -> None:
    from src.attempt21_features import run_attempt21_feature_build

    run_attempt21_feature_build()


def run_attempt21_models() -> None:
    from src.attempt21_models import run_attempt21_models

    run_attempt21_models()


def run_attempt21_report() -> None:
    from src.attempt21_report import run_attempt21_report

    run_attempt21_report()


def run_data_quality_audit() -> None:
    from src.data_quality_audit import run_data_quality_audit

    run_data_quality_audit()


def run_system_handoff() -> None:
    from src.system_handoff import generate_system_handoff

    generate_system_handoff()


def run_attempt3_download() -> None:
    from src.download_attempt3_data import snapshot_attempt3_data

    snapshot_attempt3_data()


def run_attempt3_features() -> None:
    from src.attempt3_features import build_attempt3_features

    build_attempt3_features()


def run_attempt3_models() -> None:
    from src.attempt3_models import run_attempt3_models as run

    run()


def run_feature35_download() -> None:
    from src.download_feature_experiments import snapshot_feature_experiment_data

    snapshot_feature_experiment_data()


def run_feature35_features() -> None:
    from src.feature_experiments import build_feature_experiment_tables

    build_feature_experiment_tables()


def run_feature35_models() -> None:
    from src.feature_experiment_models import run_feature_experiment_models

    run_feature_experiment_models()


def run_selected_models() -> None:
    from src.selected_feature_models import run_selected_feature_models

    run_selected_feature_models()


def run_decision_tree_models() -> None:
    from src.decision_tree_models import run_decision_tree_experiment

    run_decision_tree_experiment()


def run_random_forest_models() -> None:
    from src.random_forest_models import run_random_forest_experiment

    run_random_forest_experiment()


def run_gradient_boosting_models() -> None:
    from src.gradient_boosting_models import run_gradient_boosting_experiment

    run_gradient_boosting_experiment()


def run_te_role_models() -> None:
    from src.te_role_experiment import run_te_role_experiment

    run_te_role_experiment()


def run_te_competition_models() -> None:
    from src.te_competition_experiment import run_te_competition_experiment

    run_te_competition_experiment()


def run_contract_download() -> None:
    from src.download_contract_data import snapshot_contract_data

    snapshot_contract_data()


def run_contract_features() -> None:
    from src.contract_features import build_contract_features

    build_contract_features()


def run_contract_models() -> None:
    from src.contract_models import run_contract_experiment

    run_contract_experiment()


def run_te_constrained_competition() -> None:
    from src.te_constrained_competition import run_constrained_te_experiment

    run_constrained_te_experiment()


def run_te_constrained_confirmation() -> None:
    from src.te_constrained_confirmation import run_confirmation

    run_confirmation()


STAGES: dict[str, Callable[[], None]] = {
    "features": run_features,
    "baselines": run_baselines,
    "validate_linear": run_validate_linear,
    "evaluate_v1": run_evaluate_v1,
    "error_analysis_v1": run_error_analysis_v1,
    "conclusion_v1": run_conclusion,
    "phase2_features": run_phase2_features,
    "attempt2_models": run_attempt2_models,
    "attempt2_diagnostics": run_attempt2_diagnostics,
    "attempt21_features": run_attempt21_features,
    "attempt21_models": run_attempt21_models,
    "attempt21_report": run_attempt21_report,
    "data_quality_audit": run_data_quality_audit,
    "system_handoff": run_system_handoff,
    "attempt3_download": run_attempt3_download,
    "attempt3_features": run_attempt3_features,
    "attempt3_models": run_attempt3_models,
    "feature35_download": run_feature35_download,
    "feature35_features": run_feature35_features,
    "feature35_models": run_feature35_models,
    "selected_models": run_selected_models,
    "decision_tree_models": run_decision_tree_models,
    "random_forest_models": run_random_forest_models,
    "gradient_boosting_models": run_gradient_boosting_models,
    "te_role_models": run_te_role_models,
    "te_competition_models": run_te_competition_models,
    "contract_download": run_contract_download,
    "contract_features": run_contract_features,
    "contract_models": run_contract_models,
    "te_constrained_competition": run_te_constrained_competition,
    "te_constrained_confirmation": run_te_constrained_confirmation,
}


def run_first_attempt_all() -> None:
    """Run Phase 5 onward without downloading or rebuilding Phases 1-4."""

    for name in (
        "features",
        "baselines",
        "validate_linear",
        "evaluate_v1",
        "error_analysis_v1",
        "conclusion_v1",
    ):
        print(f"\n=== {name} ===")
        STAGES[name]()


def run_attempt2_all() -> None:
    """Build Phase 2 features and run all Attempt 2 ablations."""

    for name in ("phase2_features", "attempt2_models", "attempt2_diagnostics"):
        print(f"\n=== {name} ===")
        STAGES[name]()


def run_attempt21_all() -> None:
    """Run the isolated pandas-only Attempt 2.1 experiment."""
    for name in ("attempt21_features", "attempt21_models", "attempt21_report"):
        print(f"\n=== {name} ===")
        STAGES[name]()


def run_attempt3_all() -> None:
    """Run the isolated veteran-cleanup and market-feature experiment."""
    for name in ("attempt3_download", "attempt3_features", "attempt3_models"):
        print(f"\n=== {name} ===")
        STAGES[name]()


def run_feature35_all() -> None:
    """Run isolated feature-group experiments 3 through 5."""
    for name in ("feature35_download", "feature35_features", "feature35_models"):
        print(f"\n=== {name} ===")
        STAGES[name]()


def run_contract_all() -> None:
    """Run the isolated point-in-time contract-capital experiment."""
    for name in ("contract_download", "contract_features", "contract_models"):
        print(f"\n=== {name} ===")
        STAGES[name]()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=[*STAGES, "first_attempt_all", "attempt2_all", "attempt21_all", "attempt3_all", "feature35_all", "contract_all"]
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.stage == "first_attempt_all":
        run_first_attempt_all()
    elif args.stage == "attempt2_all":
        run_attempt2_all()
    elif args.stage == "attempt21_all":
        run_attempt21_all()
    elif args.stage == "attempt3_all":
        run_attempt3_all()
    elif args.stage == "feature35_all":
        run_feature35_all()
    elif args.stage == "contract_all":
        run_contract_all()
    else:
        STAGES[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
