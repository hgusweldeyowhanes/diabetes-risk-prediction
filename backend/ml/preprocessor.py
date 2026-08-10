"""Feature cleaning and engineering (delegates to pipeline)."""
from __future__ import annotations

from .pipeline import DiabetesPredictor, ORIGINAL_FEATURES, ZERO_AS_MISSING


class Preprocessor:
    def __init__(self, predictor: DiabetesPredictor | None = None):
        self.predictor = predictor or DiabetesPredictor()

    def clean(self, frame):
        return self.predictor._clean_clinical_zeros(frame)

    def engineer(self, frame):
        return self.predictor._create_features(frame)

    def prepare_input(self, input_data: dict):
        return self.predictor._prepare_input_data(input_data)


__all__ = ["Preprocessor", "ORIGINAL_FEATURES", "ZERO_AS_MISSING"]
