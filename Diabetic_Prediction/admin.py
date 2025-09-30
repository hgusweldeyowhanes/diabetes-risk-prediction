from django.contrib import admin
from .models import MLModel, Prediction, ModelPerformance, FeatureStatistics

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'version', 'accuracy', 'precision', 'recall', 'f1_score', 'is_active', 'created_at']
    list_filter = ['model_type', 'is_active', 'created_at']
    readonly_fields = ['created_at', 'trained_on']
    search_fields = ['name', 'model_type']
    list_editable = ['is_active']  
    def get_queryset(self, request):
        return super().get_queryset(request)

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'ml_model', 'prediction_result', 'glucose', 'bmi', 'age', 'created_at', 'ip_address']
    list_filter = ['prediction_result', 'created_at', 'ml_model']
    readonly_fields = ['created_at']
    search_fields = ['session_id', 'ip_address', 'ml_model__name']
    date_hierarchy = 'created_at'

@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display = ['ml_model', 'accuracy', 'precision', 'recall', 'f1_score', 'test_size', 'created_at']
    list_filter = ['ml_model', 'created_at']
    readonly_fields = ['created_at']
    search_fields = ['ml_model__name']

@admin.register(FeatureStatistics)
class FeatureStatisticsAdmin(admin.ModelAdmin):
    list_display = ['ml_model', 'feature_name', 'importance_score', 'mean_value', 'std_value', 'updated_at']
    list_filter = ['ml_model', 'feature_name']
    readonly_fields = ['updated_at']
    search_fields = ['feature_name', 'ml_model__name']
    ordering = ['-importance_score']