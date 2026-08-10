from django.contrib import admin

from .models import FeatureStatistics, MLModel, ModelPerformance, Prediction


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ("name", "model_type", "version", "accuracy", "roc_auc", "is_active")
    list_filter = ("model_type", "is_active")


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "prediction_result",
        "probability",
        "risk_level",
        "age",
        "glucose",
        "created_at",
    )
    list_filter = ("prediction_result", "risk_class")


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ("ml_model", "accuracy", "recall", "f1_score", "created_at")


@admin.register(FeatureStatistics)
class FeatureStatisticsAdmin(admin.ModelAdmin):
    list_display = ("feature_name", "ml_model", "importance_score", "mean_value")
