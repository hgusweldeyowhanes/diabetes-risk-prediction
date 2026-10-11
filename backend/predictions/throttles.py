from rest_framework.throttling import AnonRateThrottle, UserRateThrottle


class PredictAnonThrottle(AnonRateThrottle):
    scope = "predict_anon"


class PredictUserThrottle(UserRateThrottle):
    scope = "predict_user"
