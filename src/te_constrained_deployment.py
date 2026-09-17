"""Package frozen V1 plus the approved constrained TE residual for deployment."""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from src.attempt21_features import V1_FEATURES
from src.config import (
    MODELS_DIR,
    SELECTED_MODELS_DIR,
    SELECTED_REPORT_TABLES_DIR,
    TE_CONSTRAINED_PROCESSED_DATA_DIR,
)
from src.te_constrained_competition import BURDEN, CORRECTION_CAP, EFFECTIVE_COUNT
from src.te_constrained_confirmation import DISCOVERY_SEASONS

TE_DEPLOYMENT_FILENAME = "te_selected_constrained_residual.joblib"
TE_PROMOTION_OVERRIDE_FILENAME = "te_promotion_override.json"
PRACTICAL_FEATURED_BIAS_TOLERANCE = 0.02
APPROVED_CONFIRMATION_CHECKS = {
    "confirmation_decision": "FAIL",
    "featured_tier_bias_change": -0.011362257655036512,
    "mae_threshold_pass": True,
    "season_consistency_pass": True,
    "spearman_pass": True,
    "low_tier_bias_pass": True,
}
APPROVED_LOCKED_CANDIDATE = {
    "components": [
        "V1_PLUS_CENTERED_EFFECTIVE_COUNT_RESIDUAL",
        "V1_PLUS_HIGH_BURDEN_PENALTY_ONLY",
        "V1_PLUS_MULTI_THRESHOLD_COMPETITION_BURDENS",
    ],
    "parameters": {
        "V1_PLUS_CENTERED_EFFECTIVE_COUNT_RESIDUAL": {"coefficient": -0.15},
        "V1_PLUS_HIGH_BURDEN_PENALTY_ONLY": {"penalty": 0.1},
        "V1_PLUS_MULTI_THRESHOLD_COMPETITION_BURDENS": {
            "penalty_2": 0.05,
            "penalty_4": 0.0,
            "penalty_6": 0.0,
        },
    },
    "combined_shrinkage": 1.0,
}


class TEDeploymentError(RuntimeError):
    """Raised when the approved TE deployment contract is invalid."""


@dataclass
class TEConstrainedResidualModel:
    """One deployable predictor containing V1 and its bounded correction."""

    base_v1_model: object
    v1_features: list[str]
    effective_count_median: float
    effective_count_mean: float
    effective_count_scale: float
    burden_median: float
    centered_count_coefficient: float
    high_burden_penalty: float
    multi_penalty_2: float
    multi_penalty_4: float
    multi_penalty_6: float
    combined_shrinkage: float

    @property
    def feature_names_in_(self) -> np.ndarray:
        return np.asarray([*self.v1_features, EFFECTIVE_COUNT, BURDEN], dtype=object)

    def residual_correction(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted({EFFECTIVE_COUNT, BURDEN} - set(frame.columns))
        if missing:
            raise TEDeploymentError(f"TE correction is missing features: {missing}")
        effective = pd.to_numeric(frame[EFFECTIVE_COUNT], errors="coerce").fillna(
            self.effective_count_median
        ).to_numpy(dtype=float)
        burden = pd.to_numeric(frame[BURDEN], errors="coerce").fillna(
            self.burden_median
        ).to_numpy(dtype=float)
        centered = np.clip(
            self.centered_count_coefficient
            * ((effective - self.effective_count_mean) / self.effective_count_scale),
            -CORRECTION_CAP,
            CORRECTION_CAP,
        )
        high = np.clip(
            -self.high_burden_penalty * np.maximum(0.0, burden - 6.0),
            -CORRECTION_CAP,
            CORRECTION_CAP,
        )
        multi = np.clip(
            -self.multi_penalty_2 * np.maximum(0.0, burden - 2.0)
            - self.multi_penalty_4 * np.maximum(0.0, burden - 4.0)
            - self.multi_penalty_6 * np.maximum(0.0, burden - 6.0),
            -CORRECTION_CAP,
            CORRECTION_CAP,
        )
        return np.clip(
            self.combined_shrinkage * np.column_stack([centered, high, multi]).mean(axis=1),
            -CORRECTION_CAP,
            CORRECTION_CAP,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame):
            raise TEDeploymentError("TE deployment prediction requires a pandas DataFrame")
        missing = sorted(set(self.v1_features) - set(frame.columns))
        if missing:
            raise TEDeploymentError(f"TE V1 base is missing features: {missing}")
        base_model = self.base_v1_model
        # Early V1 artifacts stored metadata and the fitted sklearn pipeline in a
        # dictionary; later artifacts stored the pipeline directly. Supporting both
        # formats keeps the promoted TE wrapper compatible with the frozen V1 model.
        if isinstance(base_model, dict):
            base_model = base_model.get("pipeline")
        if base_model is None or not hasattr(base_model, "predict"):
            raise TEDeploymentError("TE V1 deployment artifact has no predictor")
        base = base_model.predict(frame[self.v1_features].to_numpy())
        return base + self.residual_correction(frame)


def build_te_deployment_model() -> tuple[TEConstrainedResidualModel, dict[str, object]]:
    """Build and save the user-approved model while preserving V1 for rollback."""

    checks = APPROVED_CONFIRMATION_CHECKS
    if checks["featured_tier_bias_change"] < -PRACTICAL_FEATURED_BIAS_TOLERANCE:
        raise TEDeploymentError("Confirmation exceeds approved featured-tier tolerance")
    required_checks = [
        "mae_threshold_pass", "season_consistency_pass", "spearman_pass",
        "low_tier_bias_pass",
    ]
    if not all(checks[name] for name in required_checks):
        raise TEDeploymentError("Confirmation failed a non-overridden promotion check")

    table = pd.read_parquet(
        TE_CONSTRAINED_PROCESSED_DATA_DIR / "te_constrained_modeling_dataset.parquet"
    )
    discovery = table[table["prediction_season"].isin(DISCOVERY_SEASONS)]
    effective_median = float(discovery[EFFECTIVE_COUNT].median())
    effective = discovery[EFFECTIVE_COUNT].fillna(effective_median).to_numpy(dtype=float)
    effective_scale = float(effective.std())
    if not np.isfinite(effective_scale) or effective_scale == 0:
        effective_scale = 1.0
    locked = APPROVED_LOCKED_CANDIDATE
    parameters = locked["parameters"]
    model = TEConstrainedResidualModel(
        base_v1_model=joblib.load(MODELS_DIR / "te_linear_regression.joblib"),
        v1_features=list(V1_FEATURES["TE"]),
        effective_count_median=effective_median,
        effective_count_mean=float(effective.mean()),
        effective_count_scale=effective_scale,
        burden_median=float(discovery[BURDEN].median()),
        centered_count_coefficient=float(
            parameters["V1_PLUS_CENTERED_EFFECTIVE_COUNT_RESIDUAL"]["coefficient"]
        ),
        high_burden_penalty=float(
            parameters["V1_PLUS_HIGH_BURDEN_PENALTY_ONLY"]["penalty"]
        ),
        multi_penalty_2=float(
            parameters["V1_PLUS_MULTI_THRESHOLD_COMPETITION_BURDENS"]["penalty_2"]
        ),
        multi_penalty_4=float(
            parameters["V1_PLUS_MULTI_THRESHOLD_COMPETITION_BURDENS"]["penalty_4"]
        ),
        multi_penalty_6=float(
            parameters["V1_PLUS_MULTI_THRESHOLD_COMPETITION_BURDENS"]["penalty_6"]
        ),
        combined_shrinkage=float(locked["combined_shrinkage"]),
    )
    SELECTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SELECTED_MODELS_DIR / TE_DEPLOYMENT_FILENAME)
    override = {
        "status": "PROMOTED_BY_USER_APPROVED_PRACTICAL_TOLERANCE",
        "previous_model": "models/te_linear_regression.joblib",
        "rollback_model_preserved": True,
        "selected_model": f"models/selected_feature_set/{TE_DEPLOYMENT_FILENAME}",
        "strict_confirmation_decision": checks["confirmation_decision"],
        "overridden_check": "featured_tier_bias_pass",
        "featured_tier_bias_change_ppg": checks["featured_tier_bias_change"],
        "approved_practical_tolerance_ppg": PRACTICAL_FEATURED_BIAS_TOLERANCE,
        "other_confirmation_checks": {name: checks[name] for name in required_checks},
        "locked_candidate": locked,
    }
    SELECTED_REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (SELECTED_REPORT_TABLES_DIR / TE_PROMOTION_OVERRIDE_FILENAME).write_text(
        json.dumps(override, indent=2) + "\n", encoding="utf-8"
    )
    return model, override


if __name__ == "__main__":
    build_te_deployment_model()
