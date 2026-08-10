from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from ml.model_loader import get_predictor
from patients.models import Patient

from .models import MLModel, Prediction
from .permissions import CanViewPrediction
from .serializers import (
    BiomarkerInputSerializer,
    MLModelSerializer,
    PredictionSerializer,
)


class PredictView(APIView):
    """Public predict endpoint (optional auth attaches user/patient)."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = BiomarkerInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        predictor = get_predictor()
        if not predictor.is_trained:
            ok = predictor.load_and_train()
            if not ok:
                return Response(
                    {"error": "Model training failed"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        input_data = {
            "Pregnancies": data["pregnancies"],
            "Glucose": data["glucose"],
            "BloodPressure": data["blood_pressure"],
            "SkinThickness": data["skin_thickness"],
            "Insulin": data["insulin"],
            "BMI": data["bmi"],
            "DiabetesPedigreeFunction": data["diabetes_pedigree_function"],
            "Age": data["age"],
        }

        patient = None
        patient_id = data.get("patient_id")
        if patient_id and request.user.is_authenticated:
            patient = Patient.objects.filter(
                pk=patient_id, created_by=request.user
            ).first()

        result = predictor.predict(input_data, request=request, patient=patient)
        if "error" in result:
            return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(
            {
                "status": "success",
                "prediction": "Diabetic"
                if result.get("final_prediction") == 1
                else "Not Diabetic",
                "probability": result.get("probability"),
                "probability_ml": result.get("probability_ml"),
                "probability_rule": result.get("probability_rule"),
                "probability_ci": result.get("probability_ci"),
                "confidence": result.get("confidence"),
                "risk_level": result.get("risk_level"),
                "risk_class": result.get("risk_class"),
                "clinical_guidance": result.get("clinical_guidance"),
                "model_agreement": result.get("model_agreement"),
                "model_votes": result.get("model_votes"),
                "hybrid_rules_fired": result.get("hybrid_rules_fired"),
                "local_explanation": result.get("local_explanation"),
                "shap_values": result.get("shap_values"),
                "counterfactuals": result.get("counterfactuals"),
                "feature_analysis": result.get("feature_analysis"),
                "techniques": result.get("techniques"),
                "analytics": {
                    "ensemble_metrics": result.get("analytics", {}).get(
                        "ensemble_metrics", {}
                    ),
                    "feature_importance": result.get("analytics", {}).get(
                        "feature_importance", {}
                    ),
                    "roc_auc": result.get("analytics", {}).get("roc_auc"),
                    "cv_auc": result.get("analytics", {}).get("cv_auc"),
                    "plots": result.get("analytics", {}).get("plots", {}),
                    "subgroup_metrics": result.get("analytics", {}).get(
                        "subgroup_metrics", {}
                    ),
                },
                "prediction_id": result.get("prediction_id"),
                "input_data": input_data,
            }
        )


class PredictionHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticated, CanViewPrediction]

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)


class MLModelViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [permissions.AllowAny]

    @action(detail=False, methods=["get"])
    def active(self, request):
        model = MLModel.objects.filter(is_active=True).first()
        if not model:
            return Response({"detail": "No active model"}, status=404)
        return Response(MLModelSerializer(model).data)
