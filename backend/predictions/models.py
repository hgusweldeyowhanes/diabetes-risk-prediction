from django.conf import settings
from django.db import models


class MLModel(models.Model):
    MODEL_TYPES = [
        ("random_forest", "Random Forest"),
        ("xgboost", "XGBoost"),
        ("stacking", "Stacking Ensemble"),
        ("ensemble", "Ensemble"),
    ]

    name = models.CharField(max_length=120)
    model_type = models.CharField(max_length=32, choices=MODEL_TYPES)
    version = models.CharField(max_length=20, default="1.0")
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    roc_auc = models.FloatField(null=True, blank=True)
    feature_importance = models.JSONField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    trained_on = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} v{self.version}"


class Prediction(models.Model):
    ml_model = models.ForeignKey(
        MLModel, on_delete=models.SET_NULL, null=True, blank=True, related_name="predictions"
    )
    patient = models.ForeignKey(
        "patients.Patient",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="predictions",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="predictions",
    )

    pregnancies = models.IntegerField()
    glucose = models.IntegerField()
    blood_pressure = models.IntegerField()
    skin_thickness = models.IntegerField()
    insulin = models.IntegerField()
    bmi = models.FloatField()
    diabetes_pedigree_function = models.FloatField()
    age = models.IntegerField()
    sex = models.CharField(max_length=1, blank=True)
    screening = models.JSONField(null=True, blank=True)

    prediction_result = models.BooleanField()
    probability = models.FloatField(null=True, blank=True)
    risk_level = models.CharField(max_length=64, blank=True)
    risk_class = models.CharField(max_length=16, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    model_votes = models.JSONField(null=True, blank=True)
    clinical_guidance = models.JSONField(null=True, blank=True)
    local_explanation = models.JSONField(null=True, blank=True)
    counterfactuals = models.JSONField(null=True, blank=True)
    feature_analysis = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        label = "Diabetic" if self.prediction_result else "Not Diabetic"
        return f"Prediction {self.pk} — {label}"


class ModelPerformance(models.Model):
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name="performances")
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    roc_auc = models.FloatField(null=True, blank=True)
    test_size = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class FeatureStatistics(models.Model):
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name="feature_stats")
    feature_name = models.CharField(max_length=100)
    mean_value = models.FloatField()
    std_value = models.FloatField()
    min_value = models.FloatField()
    max_value = models.FloatField()
    importance_score = models.FloatField(default=0.0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ["ml_model", "feature_name"]
