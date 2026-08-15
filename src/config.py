"""Paths used by the inference-only submission package."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "data"
MODELS_DIR = REPOSITORY_ROOT / "models"
SELECTED_MODELS_DIR = MODELS_DIR / "selected_feature_set"
