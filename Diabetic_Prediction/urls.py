from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.DiabetesPredictionView.as_view(), name='predict_diabetes'),
    path('api/predict/', views.api_predict_diabetes, name='api_predict_diabetes'),
    path('health/', views.health_check, name='health_check'),
]