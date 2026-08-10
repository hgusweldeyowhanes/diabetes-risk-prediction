from functools import wraps

from rest_framework.response import Response
from rest_framework import status


def require_trained_model(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from ml.model_loader import get_predictor

        predictor = get_predictor()
        if not predictor.is_trained and not predictor.load_and_train():
            return Response(
                {"error": "ML model is not ready"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return func(*args, **kwargs)

    return wrapper
