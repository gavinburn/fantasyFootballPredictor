"""Minimal deployment classes required to load fitted model artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EFFECTIVE_COUNT = "effective_receiving_competitor_count"
BURDEN = "probability_weighted_competitor_targets_per_game"
CORRECTION_CAP = 0.75


class DeploymentModelError(RuntimeError):
    """Raised when a deployment model receives incompatible scoring data."""


@dataclass
class TEConstrainedResidualModel:
    """Frozen V1 tight-end predictor with its bounded competition correction."""

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
            raise DeploymentModelError(f"TE correction is missing features: {missing}")
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
            self.combined_shrinkage
            * np.column_stack([centered, high, multi]).mean(axis=1),
            -CORRECTION_CAP,
            CORRECTION_CAP,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame):
            raise DeploymentModelError("TE prediction requires a pandas DataFrame")
        missing = sorted(set(self.v1_features) - set(frame.columns))
        if missing:
            raise DeploymentModelError(f"TE V1 base is missing features: {missing}")
        base_model = self.base_v1_model
        if isinstance(base_model, dict):
            base_model = base_model.get("pipeline")
        if base_model is None or not hasattr(base_model, "predict"):
            raise DeploymentModelError("TE deployment artifact has no predictor")
        base = base_model.predict(frame[self.v1_features].to_numpy())
        return base + self.residual_correction(frame)
