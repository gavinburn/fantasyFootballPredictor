"""Contracts for the promoted TE deployment wrapper."""

import joblib
import numpy as np
import pandas as pd

from src.config import (
    SELECTED_MODELS_DIR,
    TE_CONSTRAINED_PROCESSED_DATA_DIR,
)
from src.te_constrained_deployment import TE_DEPLOYMENT_FILENAME


def test_deployment_correction_is_bounded_and_finite() -> None:
    model = joblib.load(SELECTED_MODELS_DIR / TE_DEPLOYMENT_FILENAME)
    table = pd.read_parquet(
        TE_CONSTRAINED_PROCESSED_DATA_DIR / "te_constrained_modeling_dataset.parquet"
    )
    confirmation = table[table["prediction_season"].between(2012, 2016)].copy()
    correction = model.residual_correction(confirmation)
    assert np.isfinite(correction).all()
    assert np.abs(correction).max() <= 0.75


def test_original_v1_model_is_preserved_for_rollback() -> None:
    from src.config import MODELS_DIR

    assert (MODELS_DIR / "te_linear_regression.joblib").exists()
