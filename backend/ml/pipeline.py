"""
Advanced diabetes risk prediction pipeline (aligned with modern ML practice).

Implemented from the "Advanced Approaches" checklist (where the Pima dataset allows):
  - Gradient boosting (sklearn GB + optional XGBoost / LightGBM)
  - Stacking ensemble + soft blending with clinical rules
  - Clinical feature engineering (HOMA-IR proxy, interactions, polynomials,
    metabolic-syndrome style flags)
  - Zero-as-missing + IterativeImputer, then SMOTE on train only
  - Class weights / cost-sensitive learning (FN penalty)
  - Probability calibration + uncertainty (model-disagreement bands)
  - Stratified CV ROC-AUC, subgroup (age-band) metrics, learning curves
  - Explainability: RF importance, PDP, local approximation, SHAP if installed
  - Counterfactual "what-if" risk reductions
  - Multi-tier risk stratification
  - Disk cache of trained artifacts

Not implementable with single-visit Pima features alone (need more data):
  LSTM time series, Cox survival, true causal graphs, genomic/wearable inputs,
  TG/HDL, pulse pressure, multi-task complications, GNNs, production drift monitors.
"""
from __future__ import annotations

import base64
import io
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.conf import settings
from django.utils import timezone
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import RobustScaler

from predictions.models import (
    FeatureStatistics,
    MLModel,
    ModelPerformance,
    Prediction,
)
from utils.helpers import client_ip

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

ORIGINAL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Cost-sensitive: missing a diabetic (FN) is costlier than a false alarm
POS_CLASS_WEIGHT = 2.5

MODEL_DIR = Path(
    getattr(
        settings,
        "ML_ARTIFACT_DIR",
        Path(getattr(settings, "BASE_DIR", Path.cwd())) / "ml_artifacts",
    )
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
STATIC_MODELS = Path(
    getattr(
        settings,
        "ML_STATIC_MODELS",
        Path(getattr(settings, "BASE_DIR", Path.cwd())) / "static" / "models",
    )
)
STATIC_MODELS.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = MODEL_DIR / "diabetes_stack_v3_advanced.joblib"


class DiabetesPredictor:
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.meta_model = None
        self.model_rf = None
        self.model_gb = None
        self.model_lr = None
        self.model_et = None
        self.model_xgb = None
        self.model_lgbm = None
        self.feature_names_ = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False
        self.metrics_ = {}
        self.cv_auc_ = None
        self.subgroup_metrics_ = {}
        self.original_features = ORIGINAL_FEATURES
        self.django_model = None
        self.rf_django_model = None
        self.ensemble_model = None
        self.background_X_ = None  # for SHAP / local explain

    # ------------------------------------------------------------------ public
    def load_and_train(self, force_retrain: bool = False) -> bool:
        if not force_retrain and self._load_from_disk():
            logger.info("Loaded v3 advanced artifact from disk")
            self.is_trained = True
            try:
                self.ensemble_model = MLModel.objects.filter(
                    name__icontains="Stacking Ensemble"
                ).order_by("-id").first()
                self.django_model = self.ensemble_model or MLModel.objects.filter(
                    is_active=True
                ).first()
            except Exception:
                self.django_model = None
            return True
        try:
            data_path = Path(
                getattr(
                    settings,
                    "ML_DATA_PATH",
                    Path(settings.BASE_DIR) / "data" / "diabetes.csv",
                )
            )
            if not data_path.exists():
                data_path = Path(settings.BASE_DIR) / "diabetes.csv"
            if not data_path.exists():
                data_path = Path("diabetes.csv")
            data = pd.read_csv(data_path)
            X = data.drop("Outcome", axis=1)
            y = data["Outcome"].astype(int)

            X = self._clean_clinical_zeros(X)
            X = self._create_features(X)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Multiple imputation (advanced vs simple mean)
            self.imputer = IterativeImputer(
                max_iter=20, random_state=42, initial_strategy="median"
            )
            X_tr_imp = pd.DataFrame(
                self.imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index
            )
            X_te_imp = pd.DataFrame(
                self.imputer.transform(X_te), columns=X_te.columns, index=X_te.index
            )

            smote = SMOTE(random_state=42, k_neighbors=5)
            X_res, y_res = smote.fit_resample(X_tr_imp, y_tr)
            X_res = pd.DataFrame(X_res, columns=X_tr_imp.columns)

            sample_w = np.where(y_res == 1, POS_CLASS_WEIGHT, 1.0)

            self.scaler = RobustScaler()
            X_res_s = self.scaler.fit_transform(X_res)
            X_te_s = self.scaler.transform(X_te_imp)

            self.X_train, self.y_train = X_res, y_res
            self.X_test, self.y_test = X_te_imp, y_te
            self.feature_names_ = list(X.columns)
            self.background_X_ = X_tr_imp.sample(
                n=min(80, len(X_tr_imp)), random_state=42
            )

            self.model_rf = RandomForestClassifier(
                n_estimators=450,
                max_depth=12,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight={0: 1, 1: POS_CLASS_WEIGHT},
                random_state=42,
                n_jobs=-1,
            )
            self.model_gb = GradientBoostingClassifier(
                n_estimators=280,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42,
            )
            self.model_et = ExtraTreesClassifier(
                n_estimators=400,
                max_depth=14,
                min_samples_leaf=2,
                class_weight={0: 1, 1: POS_CLASS_WEIGHT},
                random_state=42,
                n_jobs=-1,
            )
            self.model_lr = LogisticRegression(
                C=1.2,
                max_iter=2500,
                class_weight={0: 1, 1: POS_CLASS_WEIGHT},
                random_state=42,
            )

            estimators = [
                ("rf", self.model_rf),
                ("gb", self.model_gb),
                ("et", self.model_et),
                ("lr", self.model_lr),
            ]
            if HAS_XGB:
                self.model_xgb = XGBClassifier(
                    n_estimators=320,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=POS_CLASS_WEIGHT,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                )
                estimators.append(("xgb", self.model_xgb))
            if HAS_LGBM:
                self.model_lgbm = LGBMClassifier(
                    n_estimators=320,
                    learning_rate=0.05,
                    num_leaves=31,
                    scale_pos_weight=POS_CLASS_WEIGHT,
                    random_state=42,
                    verbosity=-1,
                )
                estimators.append(("lgbm", self.model_lgbm))

            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(
                    C=1.0,
                    max_iter=2500,
                    class_weight={0: 1, 1: POS_CLASS_WEIGHT},
                    random_state=42,
                ),
                stack_method="predict_proba",
                cv=5,
                n_jobs=-1,
            )
            stack.fit(X_res_s, y_res, sample_weight=sample_w)

            self.meta_model = CalibratedClassifierCV(stack, method="isotonic", cv=3)
            self.meta_model.fit(X_res_s, y_res)

            # Base learners on native/scaled matrices
            self.model_rf.fit(X_res, y_res, sample_weight=sample_w)
            self.model_gb.fit(X_res, y_res, sample_weight=sample_w)
            self.model_et.fit(X_res, y_res, sample_weight=sample_w)
            self.model_lr.fit(X_res_s, y_res, sample_weight=sample_w)
            if self.model_xgb is not None:
                self.model_xgb.fit(X_res, y_res, sample_weight=sample_w)
            if self.model_lgbm is not None:
                self.model_lgbm.fit(X_res, y_res, sample_weight=sample_w)

            self.metrics_ = self._evaluate_all(X_te_imp, X_te_s, y_te)
            self.subgroup_metrics_ = self._subgroup_age_metrics(X_te_imp, X_te_s, y_te)

            # Stratified CV AUC (imputed train, no SMOTE leak into CV scores)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            X_tr_s = self.scaler.transform(X_tr_imp)
            self.cv_auc_ = float(
                cross_val_score(
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        class_weight={0: 1, 1: POS_CLASS_WEIGHT},
                        random_state=42,
                        n_jobs=-1,
                    ),
                    X_tr_s,
                    y_tr,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                ).mean()
            )
            self.metrics_["cv_auc_rf"] = self.cv_auc_

            self._save_model_to_db()
            self.django_model = self.ensemble_model or self.rf_django_model
            self._save_feature_statistics()
            self._save_to_disk()
            self._export_static_artifacts()
            self.is_trained = True
            logger.info(
                "v3 train done stack_auc=%.4f cv_auc=%.4f",
                self.metrics_.get("stack", {}).get("roc_auc", 0),
                self.cv_auc_ or 0,
            )
            return True
        except Exception as e:
            logger.exception("Training failed")
            print(f"Error in training: {e}")
            return False

    def _export_static_artifacts(self):
        """Publish lightweight copies under static/models for deployment."""
        import json

        try:
            model = self.model_xgb if self.model_xgb is not None else self.model_rf
            if model is not None:
                joblib.dump(model, STATIC_MODELS / "xgboost_model.pkl")
            if self.scaler is not None:
                joblib.dump(self.scaler, STATIC_MODELS / "scaler.pkl")
            (STATIC_MODELS / "feature_names.json").write_text(
                json.dumps(self.feature_names_, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("static artifact export failed: %s", e)

    def predict(self, input_data, request=None, patient=None):
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        try:
            engineered = self._prepare_input_data(input_data)
            X_imp = pd.DataFrame(
                self.imputer.transform(engineered), columns=engineered.columns
            )
            X_s = self.scaler.transform(X_imp)

            proba_ml = float(self.meta_model.predict_proba(X_s)[0][1])
            rule_score, fired_rules = self._clinical_rule_score(input_data)
            # Hybrid: blend learned probability with guideline-style risk mass
            proba = float(np.clip(0.82 * proba_ml + 0.18 * rule_score, 0.0, 1.0))
            stack_pred = int(proba >= 0.5)

            preds = {
                "stack": int(self.meta_model.predict(X_s)[0]),
                "rf": int(self.model_rf.predict(X_imp)[0]),
                "gb": int(self.model_gb.predict(X_imp)[0]),
                "et": int(self.model_et.predict(X_imp)[0]),
                "lr": int(self.model_lr.predict(X_s)[0]),
            }
            if self.model_xgb is not None:
                preds["xgb"] = int(self.model_xgb.predict(X_imp)[0])
            if self.model_lgbm is not None:
                preds["lgbm"] = int(self.model_lgbm.predict(X_imp)[0])

            votes = list(preds.values())
            vote_std = float(np.std(votes)) if votes else 0.0
            # Uncertainty: wider when models disagree
            uncertainty = float(np.clip(0.08 + 0.25 * vote_std + abs(proba - 0.5) * 0.1, 0.05, 0.35))
            ci_low = float(np.clip(proba - uncertainty, 0, 1))
            ci_high = float(np.clip(proba + uncertainty, 0, 1))

            risk_tier = self._multi_class_risk(proba)
            guidance = self._clinical_guidance(input_data, proba, risk_tier, fired_rules)
            row = X_imp.iloc[0]
            feature_analysis = self._analyze_prediction_features(row)
            local_exp = self._local_explanation(X_imp, input_data)
            counterfactuals = self._counterfactuals(input_data, proba)
            shap_vals = self._shap_values(X_imp)

            prediction_instance = self._save_prediction_to_db(
                input_data,
                row.to_dict(),
                stack_pred,
                proba,
                request,
                patient=patient,
                extras={
                    "risk_level": risk_tier,
                    "risk_class": self._risk_class_code(proba),
                    "model_votes": preds,
                    "clinical_guidance": guidance,
                    "local_explanation": local_exp,
                    "counterfactuals": counterfactuals,
                },
            )
            analytics = self._generate_analytics()

            return {
                "final_prediction": stack_pred,
                "probability": proba,
                "probability_ml": proba_ml,
                "probability_rule": rule_score,
                "confidence": max(proba, 1 - proba),
                "uncertainty": uncertainty,
                "probability_ci": [ci_low, ci_high],
                "risk_level": risk_tier,
                "risk_class": self._risk_class_code(proba),
                "model_agreement": f"{sum(votes)}/{len(votes)}",
                "majority_vote": int(sum(votes) >= len(votes) / 2),
                "ensemble_prediction": preds["stack"],
                "svm_linear_prediction": preds["lr"],
                "svm_rbf_prediction": preds.get("xgb", preds["gb"]),
                "rf_prediction": preds["rf"],
                "gb_prediction": preds["gb"],
                "et_prediction": preds["et"],
                "xgb_prediction": preds.get("xgb"),
                "lgbm_prediction": preds.get("lgbm"),
                "model_votes": preds,
                "hybrid_rules_fired": fired_rules,
                "clinical_guidance": guidance,
                "local_explanation": local_exp,
                "shap_values": shap_vals,
                "counterfactuals": counterfactuals,
                "subgroup_metrics": self.subgroup_metrics_,
                "analytics": analytics,
                "feature_analysis": feature_analysis,
                "prediction_id": prediction_instance.id if prediction_instance else None,
                "cv_auc": self.cv_auc_,
                "techniques": self._technique_manifest(),
            }
        except Exception as e:
            logger.exception("Prediction failed")
            return {"error": f"Prediction error: {e}"}

    def _technique_manifest(self) -> dict:
        """What this build covers vs the Advanced Approaches checklist."""
        return {
            "implemented": [
                "Gradient boosting (sklearn GB" + (", XGBoost" if HAS_XGB else "") + (", LightGBM" if HAS_LGBM else "") + ")",
                "Stacking ensemble + calibrated probabilities",
                "Hybrid rule-based (ADA-style thresholds) + ML blend",
                "Feature engineering: interactions, polynomials, HOMA-IR proxy, metabolic flags",
                "Iterative (multiple) imputation + SMOTE + cost-sensitive class weights",
                "Stratified CV AUC + age subgroup metrics + learning curves",
                "Calibration curve (reliability) + Brier score",
                "Explainability: feature importance, PDP, local drivers" + (", SHAP" if HAS_SHAP else " (install shap for full SHAP)"),
                "Counterfactual what-if scenarios",
                "Multi-tier risk stratification + uncertainty band from model disagreement",
            ],
            "requires_richer_data": [
                "LSTM temporal biomarkers / glucose trajectory",
                "Cox survival (time-to-diabetes)",
                "Causal inference / GNN biomarker graphs",
                "TG/HDL, pulse pressure, genomics, wearables, diet",
                "Longitudinal multi-visit cohorts, multi-task complications",
                "Production drift & demographic fairness monitoring pipelines",
            ],
            "libraries": {
                "xgboost": HAS_XGB,
                "lightgbm": HAS_LGBM,
                "shap": HAS_SHAP,
            },
        }

    # ---------------------------------------------------------------- cleaning
    def _clean_clinical_zeros(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in ZERO_AS_MISSING:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return X

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for f in ORIGINAL_FEATURES:
            if f not in X.columns:
                raise ValueError(f"Missing required feature: {f}")

        def col_fill(name, default):
            s = X[name]
            med = s.median() if s.notna().any() else default
            return s.fillna(med if pd.notna(med) else default)

        g = col_fill("Glucose", 120.0)
        bmi = col_fill("BMI", 28.0)
        age = col_fill("Age", 30.0)
        bp = col_fill("BloodPressure", 70.0)
        insulin = X["Insulin"].fillna(0)
        dpf = col_fill("DiabetesPedigreeFunction", 0.5)
        preg = X["Pregnancies"].fillna(0)
        skin = X["SkinThickness"].fillna(0)

        # Interactions
        X["BMI_Age_Interaction"] = bmi * age
        X["Age_BP_Interaction"] = age * bp
        X["Glucose_BMI_Ratio"] = g / (bmi + 1e-3)
        X["BloodPressure_Glucose_Interaction"] = bp * g
        X["Weight_BMI_Proxy"] = bmi * bmi  # no weight column → BMI² as body-size nonlinear

        # Polynomials
        X["BMI_sq"] = bmi ** 2
        X["Glucose_sq"] = g ** 2
        X["Age_sq"] = age ** 2

        # Derived clinical metrics available from Pima fields
        X["HOMA_IR_Proxy"] = (g * insulin.replace(0, np.nan)).fillna(g * 15) / 405.0
        X["Pulse_Pressure_Proxy"] = bp * 0.45  # no systolic → informative proxy scaled BP
        X["Metabolic_Risk"] = g + bmi + age / 10.0
        X["Blood_Risk"] = bp + skin
        X["MetSynd_Score"] = (
            (bmi >= 30).astype(float)
            + (g >= 100).astype(float)
            + (bp >= 85).astype(float)
            + (insulin > 150).astype(float)
        )

        X["Obesity_Flag"] = (bmi >= 30).astype(float)
        X["Prediabetes_Glucose"] = ((g >= 100) & (g < 126)).astype(float)
        X["Diabetes_Glucose_Flag"] = (g >= 126).astype(float)
        X["Hypertension_Flag"] = (bp >= 90).astype(float)
        X["Age_Over_45"] = (age >= 45).astype(float)
        X["High_DPF"] = (dpf >= 0.8).astype(float)
        X["Multi_Preg"] = (preg >= 3).astype(float)
        X["Risk_Composite"] = (
            X["Diabetes_Glucose_Flag"] * 3
            + X["Prediabetes_Glucose"] * 1.5
            + X["Obesity_Flag"] * 1.2
            + X["Hypertension_Flag"]
            + X["Age_Over_45"]
            + X["High_DPF"] * 1.1
            + np.clip(g / 200.0, 0, 1) * 2
            + np.clip(bmi / 40.0, 0, 1)
            + X["MetSynd_Score"] * 0.5
        )
        # Fixed clinical bins work for both training batches and single-row predict
        X["Glucose_Bin"] = pd.cut(
            g, bins=[-np.inf, 90, 110, 125, 150, np.inf], labels=False
        )
        X["Age_Bin"] = pd.cut(
            age, bins=[-np.inf, 30, 40, 50, 60, np.inf], labels=False
        )
        return X.fillna(0)

    def _prepare_input_data(self, input_data: dict) -> pd.DataFrame:
        row = {k: input_data[k] for k in ORIGINAL_FEATURES}
        df = pd.DataFrame([row])
        df = self._clean_clinical_zeros(df)
        return self._create_features(df)

    def _clinical_rule_score(self, inputs: dict) -> tuple[float, list]:
        """ADA-ish / NICE-inspired heuristic mass in [0,1] for hybrid blend."""
        g = float(inputs.get("Glucose", 0))
        bmi = float(inputs.get("BMI", 0))
        bp = float(inputs.get("BloodPressure", 0))
        age = float(inputs.get("Age", 0))
        dpf = float(inputs.get("DiabetesPedigreeFunction", 0))
        score = 0.08
        fired = []
        if g >= 126:
            score += 0.45
            fired.append("Rule: glucose ≥126 (diabetes-range threshold)")
        elif g >= 100:
            score += 0.2
            fired.append("Rule: glucose 100–125 (prediabetes-range)")
        if bmi >= 30:
            score += 0.15
            fired.append("Rule: obesity BMI ≥30")
        elif bmi >= 25:
            score += 0.07
            fired.append("Rule: overweight BMI ≥25")
        if bp >= 90:
            score += 0.08
            fired.append("Rule: elevated diastolic BP ≥90")
        if age >= 45:
            score += 0.08
            fired.append("Rule: age ≥45 screening priority")
        if dpf >= 0.8:
            score += 0.1
            fired.append("Rule: high diabetes pedigree (family-risk proxy)")
        if g >= 100 and bmi >= 30:
            score += 0.08
            fired.append("Rule: metabolic dual flag (glucose + obesity)")
        return float(np.clip(score, 0, 1)), fired

    # ---------------------------------------------------------------- metrics
    def _calc(self, y_true, y_pred, y_proba=None):
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if y_proba is not None:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                out["avg_precision"] = float(average_precision_score(y_true, y_proba))
                out["brier"] = float(brier_score_loss(y_true, y_proba))
            except Exception:
                out["roc_auc"] = out["avg_precision"] = out["brier"] = 0.0
        return out

    def _evaluate_all(self, X_test_imp, X_test_s, y_test):
        metrics = {}
        proba = self.meta_model.predict_proba(X_test_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics["stack"] = self._calc(y_test, pred, proba)
        metrics["stack"]["model_name"] = "Calibrated Stack"
        metrics["rf"] = self._calc(
            y_test,
            self.model_rf.predict(X_test_imp),
            self.model_rf.predict_proba(X_test_imp)[:, 1],
        )
        metrics["rf"]["model_name"] = "Random Forest"
        metrics["gb"] = self._calc(
            y_test,
            self.model_gb.predict(X_test_imp),
            self.model_gb.predict_proba(X_test_imp)[:, 1],
        )
        metrics["gb"]["model_name"] = "Gradient Boosting"
        metrics["lr"] = self._calc(
            y_test,
            self.model_lr.predict(X_test_s),
            self.model_lr.predict_proba(X_test_s)[:, 1],
        )
        metrics["lr"]["model_name"] = "Logistic Regression"
        metrics["ensemble"] = metrics["stack"]
        metrics["svm_linear"] = metrics["lr"]
        metrics["svm_rbf"] = metrics["gb"]
        return metrics

    def _subgroup_age_metrics(self, X_imp, X_s, y):
        """Stratified performance by age band (subgroup analysis)."""
        ages = X_imp["Age"].values if "Age" in X_imp.columns else np.zeros(len(y))
        bands = {
            "under_40": ages < 40,
            "40_to_54": (ages >= 40) & (ages < 55),
            "55_plus": ages >= 55,
        }
        out = {}
        proba = self.meta_model.predict_proba(X_s)[:, 1]
        for name, mask in bands.items():
            if mask.sum() < 8:
                continue
            yp = (proba[mask] >= 0.5).astype(int)
            out[name] = self._calc(y.values[mask], yp, proba[mask])
            out[name]["n"] = int(mask.sum())
        return out

    def _multi_class_risk(self, p: float) -> str:
        if p >= 0.75:
            return "Critical / Very High Risk"
        if p >= 0.55:
            return "High Risk"
        if p >= 0.35:
            return "Moderate Risk"
        if p >= 0.2:
            return "Low–Moderate Risk"
        return "Low Risk"

    def _risk_class_code(self, p: float) -> str:
        if p >= 0.55:
            return "HIGH"
        if p >= 0.35:
            return "MODERATE"
        return "LOW"

    def _clinical_guidance(self, inputs, proba, risk, fired_rules):
        tips = list(fired_rules)
        g = inputs.get("Glucose", 0)
        bmi = inputs.get("BMI", 0)
        if g >= 126:
            tips.append(
                "Confirmatory clinical testing (A1C / OGTT) is recommended — do not self-diagnose."
            )
        elif g >= 100:
            tips.append("Prediabetes-range glucose: lifestyle change can materially cut risk.")
        if bmi >= 30:
            tips.append("Obesity range BMI: weight and activity interventions are high-leverage.")
        tips.append(
            f"Hybrid probability {proba:.0%} (ML + guideline blend). Tier: {risk}."
        )
        tips.append(
            "Missing a true case is costlier than a false alarm — model weights sensitivity higher."
        )
        tips.append("Educational / screening aid only — not a medical diagnosis.")
        return tips

    def _local_explanation(self, X_imp: pd.DataFrame, raw: dict) -> list:
        """Lightweight local drivers (RF importance × standardized deviations)."""
        imp = self._get_feature_importance()
        if self.X_train is None or not imp:
            return []
        row = X_imp.iloc[0]
        means = self.X_train.mean()
        stds = self.X_train.std().replace(0, 1)
        items = []
        for f, w in imp.items():
            if f not in row.index:
                continue
            z = float((row[f] - means.get(f, 0)) / (stds.get(f, 1) or 1))
            items.append(
                {
                    "feature": f,
                    "value": float(row[f]),
                    "z_score": z,
                    "direction": "raises risk" if z > 0 and w > 0 else "lowers / baseline",
                    "strength": float(abs(z) * w),
                }
            )
        items.sort(key=lambda x: x["strength"], reverse=True)
        return items[:8]

    def _shap_values(self, X_imp: pd.DataFrame) -> list:
        if not HAS_SHAP or self.model_rf is None or self.background_X_ is None:
            return []
        try:
            cols = [c for c in self.background_X_.columns if c in X_imp.columns]
            explainer = shap.TreeExplainer(self.model_rf)
            sv = explainer.shap_values(X_imp[cols])
            if isinstance(sv, list):
                sv = sv[1]
            vals = sv[0]
            pairs = sorted(
                zip(cols, vals), key=lambda t: abs(float(t[1])), reverse=True
            )[:10]
            return [{"feature": f, "shap": float(v)} for f, v in pairs]
        except Exception as e:
            logger.debug("SHAP failed: %s", e)
            return []

    def _counterfactuals(self, inputs: dict, base_p: float) -> list:
        """Simple 'what-if' scenarios by intervening on modifiable labs."""
        scenarios = [
            ("Lower glucose by 15%", {"Glucose": max(70, inputs["Glucose"] * 0.85)}),
            ("Lower BMI by 10%", {"BMI": max(18, inputs["BMI"] * 0.9)}),
            (
                "Improve glucose −15% and BMI −10%",
                {
                    "Glucose": max(70, inputs["Glucose"] * 0.85),
                    "BMI": max(18, inputs["BMI"] * 0.9),
                },
            ),
            ("Normalize BP to 80", {"BloodPressure": 80}),
        ]
        out = []
        for label, patch in scenarios:
            alt = dict(inputs)
            alt.update(patch)
            try:
                eng = self._prepare_input_data(alt)
                Xi = pd.DataFrame(
                    self.imputer.transform(eng), columns=eng.columns
                )
                Xs = self.scaler.transform(Xi)
                p_ml = float(self.meta_model.predict_proba(Xs)[0][1])
                r, _ = self._clinical_rule_score(alt)
                p = float(np.clip(0.82 * p_ml + 0.18 * r, 0, 1))
                out.append(
                    {
                        "scenario": label,
                        "probability": round(p * 100, 1),
                        "delta": round((p - base_p) * 100, 1),
                        "risk_level": self._multi_class_risk(p),
                    }
                )
            except Exception:
                continue
        return out

    # --------------------------------------------------------------- persistence
    def _save_to_disk(self):
        joblib.dump(
            {
                "imputer": self.imputer,
                "scaler": self.scaler,
                "meta_model": self.meta_model,
                "model_rf": self.model_rf,
                "model_gb": self.model_gb,
                "model_et": self.model_et,
                "model_lr": self.model_lr,
                "model_xgb": self.model_xgb,
                "model_lgbm": self.model_lgbm,
                "feature_names": self.feature_names_,
                "metrics": self.metrics_,
                "cv_auc": self.cv_auc_,
                "subgroup_metrics": self.subgroup_metrics_,
                "X_test": self.X_test,
                "y_test": self.y_test,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "background_X": self.background_X_,
            },
            ARTIFACT_PATH,
        )

    def _load_from_disk(self) -> bool:
        if not ARTIFACT_PATH.exists():
            return False
        try:
            art = joblib.load(ARTIFACT_PATH)
            self.imputer = art["imputer"]
            self.scaler = art["scaler"]
            self.meta_model = art["meta_model"]
            self.model_rf = art["model_rf"]
            self.model_gb = art["model_gb"]
            self.model_et = art["model_et"]
            self.model_lr = art["model_lr"]
            self.model_xgb = art.get("model_xgb")
            self.model_lgbm = art.get("model_lgbm")
            self.feature_names_ = art["feature_names"]
            self.metrics_ = art.get("metrics", {})
            self.cv_auc_ = art.get("cv_auc")
            self.subgroup_metrics_ = art.get("subgroup_metrics", {})
            self.X_test = art.get("X_test")
            self.y_test = art.get("y_test")
            self.X_train = art.get("X_train")
            self.y_train = art.get("y_train")
            self.background_X_ = art.get("background_X")
            return True
        except Exception as e:
            logger.warning("Artifact load failed: %s", e)
            return False

    def _save_model_to_db(self):
        try:
            stack_m = self.metrics_.get("stack", {})
            rf_m = self.metrics_.get("rf", {})
            imp = self._get_feature_importance()
            self.ensemble_model, _ = MLModel.objects.update_or_create(
                name="Diabetes Stacking Ensemble v3 Advanced",
                model_type="stacking",
                defaults={
                    "version": "3.0",
                    "accuracy": stack_m.get("accuracy", 0),
                    "precision": stack_m.get("precision", 0),
                    "recall": stack_m.get("recall", 0),
                    "f1_score": stack_m.get("f1", 0),
                    "roc_auc": stack_m.get("roc_auc"),
                    "feature_importance": imp,
                    "trained_on": timezone.now(),
                    "is_active": True,
                },
            )
            self.rf_django_model, _ = MLModel.objects.update_or_create(
                name="Diabetes Random Forest v3",
                model_type="random_forest",
                defaults={
                    "version": "3.0",
                    "accuracy": rf_m.get("accuracy", 0),
                    "precision": rf_m.get("precision", 0),
                    "recall": rf_m.get("recall", 0),
                    "f1_score": rf_m.get("f1", 0),
                    "feature_importance": imp,
                    "trained_on": timezone.now(),
                    "is_active": False,
                },
            )
            for mobj, key in [
                (self.ensemble_model, "stack"),
                (self.rf_django_model, "rf"),
            ]:
                m = self.metrics_.get(key, {})
                ModelPerformance.objects.create(
                    ml_model=mobj,
                    accuracy=m.get("accuracy", 0),
                    precision=m.get("precision", 0),
                    recall=m.get("recall", 0),
                    f1_score=m.get("f1", 0),
                    roc_auc=m.get("roc_auc"),
                    test_size=len(self.y_test) if self.y_test is not None else 0,
                )
            MLModel.objects.exclude(id=self.ensemble_model.id).filter(
                is_active=True
            ).update(is_active=False)
        except Exception as e:
            logger.warning("DB model save failed: %s", e)

    def _save_feature_statistics(self):
        try:
            if self.django_model is None or self.X_train is None:
                return
            imp = self._get_feature_importance()
            for feature in self.X_train.columns:
                FeatureStatistics.objects.update_or_create(
                    ml_model=self.django_model,
                    feature_name=feature,
                    defaults={
                        "mean_value": float(self.X_train[feature].mean()),
                        "std_value": float(self.X_train[feature].std() or 0),
                        "min_value": float(self.X_train[feature].min()),
                        "max_value": float(self.X_train[feature].max()),
                        "importance_score": float(imp.get(feature, 0)),
                    },
                )
        except Exception as e:
            logger.warning("Feature stats save failed: %s", e)

    def _save_prediction_to_db(
        self,
        original_input,
        engineered,
        prediction,
        proba,
        request,
        patient=None,
        extras=None,
    ):
        try:
            if not self.django_model:
                self.django_model = MLModel.objects.filter(is_active=True).first()
            extras = extras or {}
            user = None
            if request is not None and getattr(request, "user", None) is not None:
                if getattr(request.user, "is_authenticated", False):
                    user = request.user
            inst = Prediction.objects.create(
                ml_model=self.django_model,
                patient=patient,
                user=user,
                pregnancies=original_input["Pregnancies"],
                glucose=original_input["Glucose"],
                blood_pressure=original_input["BloodPressure"],
                skin_thickness=original_input["SkinThickness"],
                insulin=original_input["Insulin"],
                bmi=original_input["BMI"],
                diabetes_pedigree_function=original_input["DiabetesPedigreeFunction"],
                age=original_input["Age"],
                sex=extras.get("sex", ""),
                screening=extras.get("screening"),
                prediction_result=bool(prediction),
                probability=float(proba),
                confidence=float(max(proba, 1 - proba)),
                risk_level=extras.get("risk_level", ""),
                risk_class=extras.get("risk_class", ""),
                model_votes=extras.get("model_votes"),
                clinical_guidance=extras.get("clinical_guidance"),
                local_explanation=extras.get("local_explanation"),
                counterfactuals=extras.get("counterfactuals"),
                feature_analysis=self._analyze_prediction_features(
                    pd.Series(engineered)
                ),
            )
            if request is not None:
                try:
                    inst.session_id = request.session.session_key or "anonymous"
                except Exception:
                    inst.session_id = "anonymous"
                inst.ip_address = client_ip(request)
                inst.save()
            return inst
        except Exception as e:
            logger.warning("Prediction save failed: %s", e)
            return None

    def _analyze_prediction_features(self, features: pd.Series) -> dict:
        imp = self._get_feature_importance()
        analysis = {}
        for feature, importance in imp.items():
            if feature in features.index:
                val = float(features[feature])
                analysis[feature] = {
                    "importance": float(importance),
                    "value": val,
                    "contribution": float(importance) * val,
                }
        return dict(
            sorted(
                analysis.items(), key=lambda x: abs(x[1]["contribution"]), reverse=True
            )[:10]
        )

    def _get_feature_importance(self) -> dict:
        if self.model_rf is None or self.X_train is None:
            return {}
        return dict(
            zip(self.X_train.columns, self.model_rf.feature_importances_.tolist())
        )

    def _generate_analytics(self):
        base = {
            "ensemble_metrics": self.metrics_.get("stack", {}),
            "svm_linear_metrics": self.metrics_.get("lr", {}),
            "svm_rbf_metrics": self.metrics_.get("gb", {}),
            "rf_metrics": self.metrics_.get("rf", {}),
            "plots": {},
            "feature_importance": self._get_feature_importance(),
            "roc_auc": self.metrics_.get("stack", {}).get("roc_auc"),
            "cv_auc": self.cv_auc_,
            "brier": self.metrics_.get("stack", {}).get("brier"),
            "subgroup_metrics": self.subgroup_metrics_,
            "techniques": self._technique_manifest(),
        }
        if self.X_test is None or self.y_test is None:
            return base
        X_s = self.scaler.transform(self.X_test)
        plots = {
            "ensemble_cm": self._plot_cm(
                confusion_matrix(self.y_test, self.meta_model.predict(X_s)),
                "Stacking Ensemble",
            ),
            "rf_cm": self._plot_cm(
                confusion_matrix(self.y_test, self.model_rf.predict(self.X_test)),
                "Random Forest",
            ),
            "feature_importance": self._plot_feature_importance(),
            "performance_comparison": self._plot_performance(),
            "roc_curve": self._plot_roc(X_s),
            "calibration": self._plot_calibration(X_s),
            "learning_curve": self._plot_learning_curve(),
            "pdp": self._plot_pdp(),
        }
        base["plots"] = plots
        return base

    def _plot_cm(self, cm, title):
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            pd.DataFrame(cm, index=["Actual−", "Actual+"], columns=["Pred−", "Pred+"]),
            annot=True,
            fmt="g",
            cmap="YlGnBu",
        )
        plt.title(title)
        plt.tight_layout()
        return self._fig_to_base64()

    def _plot_feature_importance(self):
        imp = self._get_feature_importance()
        items = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:14]
        names = [i[0] for i in items][::-1]
        vals = [i[1] for i in items][::-1]
        plt.figure(figsize=(10, 6.5))
        plt.barh(names, vals, color="#0f766e")
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        return self._fig_to_base64()

    def _plot_performance(self):
        keys = ["stack", "rf", "gb", "lr"]
        labels = ["Stack", "RF", "GB", "LogReg"]
        mks = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(labels))
        w = 0.18
        plt.figure(figsize=(11, 6))
        for i, mk in enumerate(mks):
            vals = [self.metrics_.get(k, {}).get(mk, 0) for k in keys]
            plt.bar(x + (i - 1.5) * w, vals, w, label=mk)
        plt.xticks(x, labels)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.title("Model Performance (cost-sensitive stack)")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return self._fig_to_base64()

    def _plot_roc(self, X_test_s):
        plt.figure(figsize=(7, 6))
        proba = self.meta_model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, proba)
        auc = self.metrics_.get("stack", {}).get("roc_auc", 0)
        plt.plot(fpr, tpr, label=f"Stack (AUC={auc:.3f})", color="#0f766e", lw=2)
        plt.plot([0, 1], [0, 1], "--", color="#999")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        return self._fig_to_base64()

    def _plot_calibration(self, X_test_s):
        """Calibration analysis — probabilities vs outcomes."""
        plt.figure(figsize=(7, 6))
        proba = self.meta_model.predict_proba(X_test_s)[:, 1]
        try:
            frac_pos, mean_pred = calibration_curve(
                self.y_test, proba, n_bins=8, strategy="quantile"
            )
            plt.plot(mean_pred, frac_pos, "o-", color="#0f766e", label="Stack")
        except Exception:
            pass
        plt.plot([0, 1], [0, 1], "--", color="#999", label="Perfect")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.tight_layout()
        return self._fig_to_base64()

    def _plot_learning_curve(self):
        if self.X_train is None:
            return ""
        try:
            # Use a light RF on a sample of engineered features for speed
            X = self.scaler.transform(self.X_train.iloc[: min(600, len(self.X_train))])
            y = (
                self.y_train.iloc[: len(X)]
                if hasattr(self.y_train, "iloc")
                else self.y_train[: len(X)]
            )
            train_sizes, train_scores, val_scores = learning_curve(
                RandomForestClassifier(
                    n_estimators=80, max_depth=8, random_state=42, n_jobs=-1
                ),
                X,
                y,
                cv=3,
                train_sizes=np.linspace(0.2, 1.0, 5),
                scoring="roc_auc",
                n_jobs=-1,
            )
            plt.figure(figsize=(7, 5))
            plt.plot(train_sizes, train_scores.mean(axis=1), label="Train AUC")
            plt.plot(train_sizes, val_scores.mean(axis=1), label="CV AUC")
            plt.xlabel("Training examples")
            plt.ylabel("ROC-AUC")
            plt.title("Learning Curve (data sufficiency)")
            plt.legend()
            plt.tight_layout()
            return self._fig_to_base64()
        except Exception as e:
            logger.debug("Learning curve failed: %s", e)
            return ""

    def _plot_pdp(self):
        """Partial dependence for Glucose & BMI (global explainability)."""
        if self.model_rf is None or self.X_train is None:
            return ""
        try:
            feats = [c for c in ["Glucose", "BMI"] if c in self.X_train.columns]
            if not feats:
                return ""
            fig, ax = plt.subplots(1, len(feats), figsize=(5 * len(feats), 4))
            if len(feats) == 1:
                ax = [ax]
            PartialDependenceDisplay.from_estimator(
                self.model_rf,
                self.X_train.sample(n=min(400, len(self.X_train)), random_state=42),
                features=feats,
                ax=ax,
            )
            fig.suptitle("Partial Dependence (risk response)")
            plt.tight_layout()
            return self._fig_to_base64()
        except Exception as e:
            logger.debug("PDP failed: %s", e)
            return ""

    def _fig_to_base64(self):
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


