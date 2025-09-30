from django.db import models

class MLModel(models.Model):
    """Store trained ML model information"""
    MODEL_TYPES = [
        ('random_forest', 'Random Forest'),
        ('svm', 'Support Vector Machine'),
        ('ensemble', 'Ensemble'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    version = models.CharField(max_length=20, default='1.0')
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    feature_importance = models.JSONField(null=True, blank=True)  
    model_file = models.FileField(upload_to='ml_models/', null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    trained_on = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.name} v{self.version}"
    
    class Meta:
        ordering = ['-created_at']


class Prediction(models.Model):
    """Store individual predictions"""
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='predictions')
    
    # Input features
    pregnancies = models.IntegerField()
    glucose = models.IntegerField()
    blood_pressure = models.IntegerField()
    skin_thickness = models.IntegerField()
    insulin = models.IntegerField()
    bmi = models.FloatField()
    diabetes_pedigree_function = models.FloatField()
    age = models.IntegerField()
    
    # Engineered features (optional)
    bmi_age_interaction = models.FloatField(null=True, blank=True)
    glucose_bmi_ratio = models.FloatField(null=True, blank=True)
    
    # Predictions
    prediction_result = models.BooleanField()  # True = Diabetic, False = Not Diabetic
    confidence = models.FloatField(null=True, blank=True)  
    feature_analysis = models.JSONField(null=True, blank=True)  
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, blank=True)  
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    def __str__(self):
        return f"Prediction {self.id} - {'Diabetic' if self.prediction_result else 'Not Diabetic'}"
    
    class Meta:
        ordering = ['-created_at']


class ModelPerformance(models.Model):
    """Store model performance metrics over time"""
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='performances')
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    test_size = models.IntegerField()  # Number of test samples
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.ml_model.name} Performance - {self.created_at.date()}"
    
    class Meta:
        ordering = ['-created_at']


class FeatureStatistics(models.Model):
    """Store statistics about features for analysis"""
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='feature_stats')
    feature_name = models.CharField(max_length=100)
    mean_value = models.FloatField()
    std_value = models.FloatField()
    min_value = models.FloatField()
    max_value = models.FloatField()
    importance_score = models.FloatField()
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.feature_name} Stats"
    
    class Meta:
        unique_together = ['ml_model', 'feature_name']