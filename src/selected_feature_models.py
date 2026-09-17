"""Evaluate and fit the position-specific feature set selected after ablations."""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd

from src.attempt3_features import feature_manifests as attempt3_manifests
from src.attempt3_models import _apply_market_calibration, _fit_market_power, _tune
from src.attempt21_features import V1_FEATURES
from src.attempt21_models import TOP_N, _metrics, build_pipeline
from src.config import (
    ATTEMPT2_PREDICTIONS_DATA_DIR,
    ATTEMPT21_PREDICTIONS_DATA_DIR,
    FEATURE35_PROCESSED_DATA_DIR,
    POSITIONS,
    RANDOM_STATE,
    SELECTED_MODELS_DIR,
    SELECTED_PREDICTIONS_DATA_DIR,
    SELECTED_REPORT_TABLES_DIR,
    SELECTED_REPORTS_DIR,
    TE_CONSTRAINED_PREDICTIONS_DATA_DIR,
    TE_CONSTRAINED_PROCESSED_DATA_DIR,
)
from src.te_constrained_competition import BURDEN, COMBINED, EFFECTIVE_COUNT
from src.te_constrained_deployment import (
    TE_DEPLOYMENT_FILENAME,
    build_te_deployment_model,
)


class SelectedFeatureModelError(RuntimeError):
    """Raised when the promoted feature-set contracts fail."""


SELECTED_SPECIFICATION = "SELECTED_MAIN"
WR_FALLBACK_SPECIFICATION = "SELECTED_WR_PARTICIPATION_FALLBACK"
TE_SELECTED_SPECIFICATION = "SELECTED_TE_CONSTRAINED_RESIDUAL"
EVALUATION_START = {
    "QB": 2012,
    # Historical preseason ECR begins in 2020 and needs one prior calibration year.
    "RB": 2021,
    # Participation data begins in 2016 and feature-aware inner tuning starts in 2019.
    "WR": 2019,
    "TE": 2012,
}


def selected_feature_manifests() -> dict[str, dict[str, list[str]]]:
    """Return the selected main set and the explicit simpler WR fallback."""

    attempt3 = attempt3_manifests()
    participation_rate = "previous_pass_play_participation_rate"
    targets_per_participated = "previous_targets_per_pass_play_participated"
    output = {
        # QB retains the veteran-aligned cohort and structural/QBR corrections.
        "QB": {
            SELECTED_SPECIFICATION: list(attempt3["QB"]["STRUCTURAL_CLEANUP"]),
        },
        # RB rejected the structural bundle. Add only the calibrated market signal
        # to the veteran-aligned Attempt 2.1 foundation (no source-availability flag).
        "RB": {
            SELECTED_SPECIFICATION: [
                *attempt3["RB"]["VETERAN_ALIGNED_BASE"], "market_implied_ppg"
            ],
        },
        "WR": {
            SELECTED_SPECIFICATION: [
                *attempt3["WR"]["STRUCTURAL_CLEANUP"],
                participation_rate,
                targets_per_participated,
            ],
            WR_FALLBACK_SPECIFICATION: [
                *attempt3["WR"]["STRUCTURAL_CLEANUP"], participation_rate
            ],
        },
        # The original V1 pipeline is preserved as the base and rollback model.
        # The promoted wrapper adds only the approved bounded competition residual.
        "TE": {
            TE_SELECTED_SPECIFICATION: [
                *V1_FEATURES["TE"], EFFECTIVE_COUNT, BURDEN,
            ],
        },
    }
    for position, specifications in output.items():
        for name, features in specifications.items():
            if len(features) != len(set(features)):
                raise SelectedFeatureModelError(
                    f"Duplicate features in {position} {name}"
                )
    return output


def _load_tables() -> dict[str, pd.DataFrame]:
    return {
        position: pd.read_parquet(
            FEATURE35_PROCESSED_DATA_DIR
            / f"{position.lower()}_feature_experiments_dataset.parquet"
        ).sort_values(["prediction_season", "player_id"]).reset_index(drop=True)
        for position in POSITIONS
    }


def _uses_market(position: str, specification: str) -> bool:
    return position == "RB" and specification == SELECTED_SPECIFICATION


def _tuning_label(position: str, specification: str) -> str:
    # Attempt 3's tuner performs fold-local market calibration under this label.
    return "MARKET_ASSISTED" if _uses_market(position, specification) else "STRUCTURAL_CLEANUP"


def _te_promoted_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use V1 fallback through 2016 and nested constrained predictions thereafter."""

    table = pd.read_parquet(
        TE_CONSTRAINED_PROCESSED_DATA_DIR / "te_constrained_modeling_dataset.parquet"
    )
    frozen = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    frozen = frozen[
        (frozen["position"] == "TE")
        & (frozen["model_name"] == "A_v1_same_cohort")
    ][["player_id", "season", "predicted_ppg"]].rename(columns={
        "predicted_ppg": "frozen_v1_predicted_ppg"
    })
    selected = table[[
        "player_id", "player_name", "prediction_season", "target_ppg"
    ]].rename(columns={
        "prediction_season": "season", "target_ppg": "actual_ppg"
    }).merge(frozen, on=["player_id", "season"], validate="one_to_one")
    selected["predicted_ppg"] = selected["frozen_v1_predicted_ppg"]
    discovery = pd.read_parquet(
        TE_CONSTRAINED_PREDICTIONS_DATA_DIR / "te_constrained_predictions.parquet"
    )
    discovery = discovery[discovery["specification"] == COMBINED][[
        "player_id", "season", "predicted_ppg"
    ]].rename(columns={"predicted_ppg": "constrained_predicted_ppg"})
    selected = selected.merge(
        discovery, on=["player_id", "season"], how="left", validate="one_to_one"
    )
    corrected = selected["constrained_predicted_ppg"].notna()
    selected.loc[corrected, "predicted_ppg"] = selected.loc[
        corrected, "constrained_predicted_ppg"
    ]
    selected["position"] = "TE"
    selected["specification"] = TE_SELECTED_SPECIFICATION
    selected = selected[[
        "player_id", "player_name", "position", "season", "actual_ppg",
        "predicted_ppg", "specification",
    ]]
    fold_rows = []
    for season, fold in selected.groupby("season"):
        metric = _metrics(
            fold["actual_ppg"].to_numpy(), fold["predicted_ppg"].to_numpy(),
            fold["player_id"].to_numpy(), TOP_N["TE"],
        )
        fold_rows.append({
            "position": "TE", "season": season,
            "specification": TE_SELECTED_SPECIFICATION,
            "method": (
                "frozen_v1_fallback" if season < 2017
                else "frozen_v1_plus_constrained_residual"
            ),
            "alpha": math.nan, "l1_ratio": math.nan,
            "training_rows": int((table["prediction_season"] < season).sum()),
            "market_calibration_alpha": math.nan,
            "market_calibration_beta": math.nan,
            "market_calibration_rows": 0,
            **metric,
        })
    return selected, pd.DataFrame(fold_rows)


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = _load_tables()
    manifests = selected_feature_manifests()
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for position, specifications in manifests.items():
        if position == "TE":
            continue
        table = tables[position]
        if table["previous_season_ppg"].isna().any():
            raise SelectedFeatureModelError("No-history row entered veteran modeling")
        for specification, features in specifications.items():
            if list(table[features].columns) != features:
                raise SelectedFeatureModelError(
                    f"Feature order changed for {position} {specification}"
                )
            for season in range(EVALUATION_START[position], 2026):
                train = table[table["prediction_season"] < season]
                valid = table[table["prediction_season"] == season]
                method, params, tuning = _tune(
                    train, features, position, _tuning_label(position, specification)
                )
                for row in tuning.to_dict("records"):
                    tuning_rows.append({
                        "position": position,
                        "outer_season": season,
                        "specification": specification,
                        **row,
                    })
                calibration = {"alpha": math.nan, "beta": math.nan, "training_rows": 0}
                if _uses_market(position, specification):
                    train, valid, calibration = _apply_market_calibration(train, valid)
                model = build_pipeline(method, params).fit(
                    train[features], train["target_ppg"]
                )
                predicted = model.predict(valid[features])
                metrics = _metrics(
                    valid["target_ppg"].to_numpy(), predicted,
                    valid["player_id"].to_numpy(), TOP_N[position],
                )
                fold_rows.append({
                    "position": position,
                    "season": season,
                    "specification": specification,
                    "method": method,
                    "alpha": params.get("alpha", math.nan),
                    "l1_ratio": params.get("l1_ratio", math.nan),
                    "training_rows": len(train),
                    "market_calibration_alpha": calibration["alpha"],
                    "market_calibration_beta": calibration["beta"],
                    "market_calibration_rows": calibration["training_rows"],
                    **metrics,
                })
                frame = valid[[
                    "player_id", "player_name", "position", "prediction_season",
                    "target_ppg",
                ]].rename(columns={
                    "prediction_season": "season", "target_ppg": "actual_ppg"
                })
                frame["predicted_ppg"] = predicted
                frame["specification"] = specification
                predictions.append(frame)
                for feature, coefficient in zip(
                    features, model.named_steps["regressor"].coef_, strict=True
                ):
                    coefficient_rows.append({
                        "position": position,
                        "season": season,
                        "specification": specification,
                        "method": method,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                    })
    te_predictions, te_folds = _te_promoted_evaluation()
    predictions.append(te_predictions)
    fold_rows.extend(te_folds.to_dict("records"))
    result = pd.concat(predictions, ignore_index=True)
    if result.duplicated(
        ["player_id", "position", "season", "specification"]
    ).any():
        raise SelectedFeatureModelError("Duplicate selected-model predictions")
    return (
        result,
        pd.DataFrame(fold_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(coefficient_rows),
    )


def _comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    attempt21 = pd.read_parquet(
        ATTEMPT21_PREDICTIONS_DATA_DIR / "attempt2_1_predictions.parquet"
    )
    attempt21 = attempt21[
        attempt21["candidate"] == "ADAPTIVE_POSITION_SELECTED"
    ][["player_id", "position", "season", "predicted_ppg"]]
    frozen_v1 = pd.read_parquet(
        ATTEMPT2_PREDICTIONS_DATA_DIR / "attempt2_ablation_predictions.parquet"
    )
    frozen_v1 = frozen_v1[
        frozen_v1["model_name"] == "A_v1_same_cohort"
    ][["player_id", "position", "season", "predicted_ppg"]]
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for (position, specification), candidate in predictions.groupby(
        ["position", "specification"]
    ):
        reference = frozen_v1 if position == "TE" else attempt21
        label = "FROZEN_V1" if position == "TE" else "ATTEMPT2_1_ADAPTIVE"
        paired = candidate.merge(
            reference.rename(columns={"predicted_ppg": "reference_predicted_ppg"}),
            on=["player_id", "position", "season"], validate="one_to_one",
        )
        candidate_error = (paired["predicted_ppg"] - paired["actual_ppg"]).abs()
        reference_error = (
            paired["reference_predicted_ppg"] - paired["actual_ppg"]
        ).abs()
        improvement = reference_error - candidate_error
        season_delta = improvement.groupby(paired["season"]).mean()
        bootstrap = np.asarray([
            rng.choice(season_delta.to_numpy(), len(season_delta), replace=True).mean()
            for _ in range(10_000)
        ])
        rows.append({
            "position": position,
            "specification": specification,
            "seasons": f"{paired['season'].min()}-{paired['season'].max()}",
            "folds": paired["season"].nunique(),
            "players": len(paired),
            "candidate_mae": float(candidate_error.mean()),
            "reference": label,
            "reference_same_cohort_mae": float(reference_error.mean()),
            "mae_improvement": float(improvement.mean()),
            "seasons_improved": int((season_delta > 0).sum()),
            "season_block_95_low": float(np.quantile(bootstrap, 0.025)),
            "season_block_95_high": float(np.quantile(bootstrap, 0.975)),
        })
    return pd.DataFrame(rows).sort_values(["position", "specification"])


def _fit_models(
    tables: dict[str, pd.DataFrame], folds: pd.DataFrame
) -> pd.DataFrame:
    manifests = selected_feature_manifests()
    rows = []
    SELECTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for position, specifications in manifests.items():
        for specification, features in specifications.items():
            if position == "TE":
                _, override = build_te_deployment_model()
                rows.append({
                    "position": position,
                    "specification": specification,
                    "method": "frozen_v1_plus_constrained_residual",
                    "alpha": math.nan,
                    "l1_ratio": math.nan,
                    "features": json.dumps(features),
                    "market_calibration_alpha": math.nan,
                    "market_calibration_beta": math.nan,
                    "model_file": TE_DEPLOYMENT_FILENAME,
                    "rollback_model": override["previous_model"],
                })
                continue
            table = tables[position].copy()
            recent = folds[
                (folds["position"] == position)
                & (folds["specification"] == specification)
            ].sort_values("season").iloc[-5:]
            choice = recent.groupby(
                ["method", "alpha", "l1_ratio"], dropna=False
            ).size().sort_values(ascending=False).index[0]
            method, alpha, l1_ratio = choice
            params = {}
            if pd.notna(alpha):
                params["alpha"] = float(alpha)
            if pd.notna(l1_ratio):
                params["l1_ratio"] = float(l1_ratio)
            calibration = {"alpha": math.nan, "beta": math.nan, "training_rows": 0}
            if _uses_market(position, specification):
                calibration = _fit_market_power(table)
                table["market_implied_ppg"] = (
                    calibration["alpha"]
                    * table["preseason_ecr"].pow(-calibration["beta"])
                )
            model = build_pipeline(str(method), params).fit(
                table[features], table["target_ppg"]
            )
            filename = f"{position.lower()}_{specification.lower()}.joblib"
            joblib.dump(model, SELECTED_MODELS_DIR / filename)
            rows.append({
                "position": position,
                "specification": specification,
                "method": method,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "features": json.dumps(features),
                "market_calibration_alpha": calibration["alpha"],
                "market_calibration_beta": calibration["beta"],
                "model_file": filename,
            })
    return pd.DataFrame(rows)


def _write_report(comparison: pd.DataFrame) -> None:
    lines = [
        "SELECTED POSITION-SPECIFIC FEATURE SET",
        "=" * 78,
        "",
        "This combines only features selected after the Attempt 3 and experiments 3-5 ablations.",
        "Historical Attempt 1, Attempt 2, Attempt 2.1, Attempt 3, and isolated experiment artifacts were not overwritten.",
        "Rookies and other no-prior-history rows remain outside these veteran models.",
        "RB uses veteran-aligned Attempt 2.1 inputs plus fold-calibrated market-implied PPG; it does not inherit the rejected RB structural bundle or an ECR-availability flag.",
        "WR uses the accepted structural group plus pass-play participation rate and targets per participated pass play. A participation-rate-only fallback is fit and reported separately.",
        "TE uses the promoted bounded three-component competition residual on top of the preserved V1 pipeline. The original V1 model remains available for rollback.",
        "",
        "Chronological outer-fold comparison",
        "-----------------------------------",
        comparison.to_string(index=False),
        "",
        "Positive MAE improvement favors the selected candidate over the current operational reference on the identical cohort.",
        "RB market results remain limited to 2021-2025; WR participation results cover 2019-2025. TE uses V1 fallback for 2012-2016 and nested constrained residual predictions for 2017-2025.",
    ]
    SELECTED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (SELECTED_REPORTS_DIR / "selected_feature_set_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_selected_feature_models() -> dict[str, str]:
    predictions, folds, tuning, coefficients = run_evaluation()
    comparison = _comparison(predictions)
    deployment = _fit_models(_load_tables(), folds)
    SELECTED_PREDICTIONS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_predictions.parquet", index=False
    )
    coefficients.to_parquet(
        SELECTED_PREDICTIONS_DATA_DIR / "selected_fold_coefficients.parquet",
        index=False,
    )
    folds.to_csv(SELECTED_REPORT_TABLES_DIR / "fold_metrics.csv", index=False)
    tuning.to_csv(SELECTED_REPORT_TABLES_DIR / "inner_tuning.csv", index=False)
    comparison.to_csv(SELECTED_REPORT_TABLES_DIR / "model_comparison.csv", index=False)
    deployment.to_csv(SELECTED_REPORT_TABLES_DIR / "model_manifest.csv", index=False)
    (SELECTED_REPORT_TABLES_DIR / "feature_manifest.json").write_text(
        json.dumps(selected_feature_manifests(), indent=2) + "\n", encoding="utf-8"
    )
    _write_report(comparison)
    return {"summary": str(SELECTED_REPORTS_DIR / "selected_feature_set_summary.txt")}


if __name__ == "__main__":
    run_selected_feature_models()
