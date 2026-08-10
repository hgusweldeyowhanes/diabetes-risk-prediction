"""Explainability helpers: local drivers, SHAP, counterfactuals."""
from __future__ import annotations


class Explainer:
    def __init__(self, predictor):
        self.predictor = predictor

    def local(self, X_imp, raw):
        return self.predictor._local_explanation(X_imp, raw)

    def shap(self, X_imp):
        return self.predictor._shap_values(X_imp)

    def counterfactuals(self, inputs, base_p):
        return self.predictor._counterfactuals(inputs, base_p)

    def feature_importance(self):
        return self.predictor._get_feature_importance()
