from django.db.models import Count, OuterRef, Subquery
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from predictions.models import Prediction
from predictions.serializers import PredictionSerializer

from .models import Patient
from .permissions import IsPatientOwner
from .serializers import PatientSerializer


class PatientViewSet(viewsets.ModelViewSet):
    serializer_class = PatientSerializer
    permission_classes = [permissions.IsAuthenticated, IsPatientOwner]

    def get_queryset(self):
        latest = Prediction.objects.filter(patient_id=OuterRef("pk")).order_by("-created_at")
        return (
            Patient.objects.filter(created_by=self.request.user)
            .annotate(
                prediction_count=Count("predictions", distinct=True),
                latest_probability=Subquery(latest.values("probability")[:1]),
                latest_risk_level=Subquery(latest.values("risk_level")[:1]),
                latest_prediction_at=Subquery(latest.values("created_at")[:1]),
            )
        )

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    @action(detail=True, methods=["get"])
    def predictions(self, request, pk=None):
        patient = self.get_object()
        rows = (
            Prediction.objects.filter(patient=patient, user=request.user)
            .order_by("created_at")
        )
        return Response(PredictionSerializer(rows, many=True).data)

    @action(detail=True, methods=["get"], url_path="trend")
    def trend(self, request, pk=None):
        patient = self.get_object()
        rows = (
            Prediction.objects.filter(patient=patient, user=request.user)
            .order_by("created_at")
        )
        return Response([_trend_point(p) for p in rows])


def _trend_point(prediction):
    screening = prediction.screening or {}
    return {
        "prediction_id": prediction.id,
        "created_at": prediction.created_at,
        "probability": prediction.probability,
        "glucose": prediction.glucose,
        "bmi": prediction.bmi,
        "risk_level": prediction.risk_level,
        "screening_score": screening.get("score"),
        "screening_label": screening.get("label"),
    }
