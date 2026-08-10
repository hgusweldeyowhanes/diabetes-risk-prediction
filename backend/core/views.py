from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from ml.model_loader import get_predictor


@api_view(["GET"])
@permission_classes([AllowAny])
def health(request):
    predictor = get_predictor()
    if not predictor.is_trained:
        # Fast path when artifacts already exist on disk
        predictor.load_and_train(force_retrain=False)
    return Response(
        {
            "status": "ok",
            "service": "diabetes-risk-prediction",
            "model_ready": bool(predictor.is_trained),
        }
    )
