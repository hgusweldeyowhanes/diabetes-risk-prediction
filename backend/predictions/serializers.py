from rest_framework import serializers

from .models import MLModel, Prediction


class BiomarkerInputSerializer(serializers.Serializer):
    pregnancies = serializers.IntegerField(min_value=0, max_value=20)
    glucose = serializers.IntegerField(min_value=0, max_value=400)
    blood_pressure = serializers.IntegerField(min_value=0, max_value=202)
    skin_thickness = serializers.IntegerField(min_value=0, max_value=200)
    insulin = serializers.IntegerField(min_value=0, max_value=900)
    bmi = serializers.FloatField(min_value=0.0, max_value=70.0)
    diabetes_pedigree_function = serializers.FloatField(min_value=0.0, max_value=3.0)
    age = serializers.IntegerField(min_value=1, max_value=120)
    sex = serializers.ChoiceField(choices=["F", "M", "U"], required=False, default="U")
    waist_cm = serializers.FloatField(min_value=40, max_value=200, required=False, allow_null=True)
    physically_active = serializers.BooleanField(required=False, default=True)
    family_history = serializers.BooleanField(required=False, default=False)
    patient_id = serializers.IntegerField(required=False, allow_null=True)

    def validate(self, data):
        if (
            data["glucose"] == 0
            and data["blood_pressure"] == 0
            and data["bmi"] == 0
            and data["age"] == 0
        ):
            raise serializers.ValidationError(
                "Glucose, blood pressure, BMI, and age cannot all be zero."
            )
        return data


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = (
            "id",
            "patient",
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree_function",
            "age",
            "sex",
            "screening",
            "prediction_result",
            "probability",
            "risk_level",
            "risk_class",
            "confidence",
            "model_votes",
            "clinical_guidance",
            "local_explanation",
            "counterfactuals",
            "feature_analysis",
            "created_at",
        )
        read_only_fields = fields


class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = (
            "id",
            "name",
            "model_type",
            "version",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "is_active",
            "trained_on",
            "feature_importance",
        )
