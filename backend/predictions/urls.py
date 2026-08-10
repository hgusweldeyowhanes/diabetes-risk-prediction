from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import MLModelViewSet, PredictView, PredictionHistoryViewSet

router = DefaultRouter()
router.register(r"history", PredictionHistoryViewSet, basename="prediction-history")
router.register(r"models", MLModelViewSet, basename="ml-models")

urlpatterns = [
    path("run/", PredictView.as_view(), name="predict-run"),
    path("", include(router.urls)),
]
