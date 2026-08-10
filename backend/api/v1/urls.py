from django.urls import include, path

urlpatterns = [
    path("", include("core.urls")),
    path("auth/", include("accounts.urls")),
    path("patients/", include("patients.urls")),
    path("predictions/", include("predictions.urls")),
]
