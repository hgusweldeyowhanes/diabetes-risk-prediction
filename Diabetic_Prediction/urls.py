from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.DiabetesPredictionView.as_view(), name='diabetes-predict'),
    
    # DRF API endpoints
    path('api/predict/', views.DiabetesPredictionAPIView.as_view(), name='api-diabetes-predict'),
    
    # Legacy API endpoints (for compatibility)
    path('api/predict/legacy/', views.api_predict_diabetes, name='api-predict-legacy'),
    path('api/health/', views.health_check, name='health-check'),
]