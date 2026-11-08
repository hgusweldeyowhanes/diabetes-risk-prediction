from rest_framework import serializers

from .models import Patient


class PatientSerializer(serializers.ModelSerializer):
    prediction_count = serializers.IntegerField(read_only=True, default=0)
    latest_probability = serializers.FloatField(read_only=True, allow_null=True, default=None)
    latest_risk_level = serializers.CharField(read_only=True, allow_null=True, default=None)
    latest_prediction_at = serializers.DateTimeField(read_only=True, allow_null=True, default=None)

    class Meta:
        model = Patient
        fields = (
            "id",
            "external_id",
            "first_name",
            "last_name",
            "sex",
            "date_of_birth",
            "notes",
            "prediction_count",
            "latest_probability",
            "latest_risk_level",
            "latest_prediction_at",
            "created_at",
            "updated_at",
        )
        read_only_fields = (
            "prediction_count",
            "latest_probability",
            "latest_risk_level",
            "latest_prediction_at",
            "created_at",
            "updated_at",
        )
