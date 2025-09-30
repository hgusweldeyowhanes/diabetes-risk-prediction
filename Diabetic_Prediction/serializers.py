from rest_framework import serializers

class DiabetesPredictionSerializer(serializers.Serializer):
    pregnancies = serializers.IntegerField(
        min_value=0,
        max_value=20,
        help_text='Number of times pregnant'
    )
    
    glucose = serializers.IntegerField(
        min_value=0,
        max_value=400,
        help_text='Plasma glucose concentration'
    )
    
    blood_pressure = serializers.IntegerField(
        min_value=0,
        max_value=202,
        help_text='Diastolic blood pressure (mm Hg)'
    )
    
    skin_thickness = serializers.IntegerField(
        min_value=0,
        max_value=200,
        help_text='Triceps skin fold thickness (mm)'
    )
    
    insulin = serializers.IntegerField(
        min_value=0,
        max_value=900,
        help_text='2-Hour serum insulin (mu U/ml)'
    )
    
    bmi = serializers.FloatField(
        min_value=0.0,
        max_value=70.0,
        help_text='Body mass index'
    )
    
    diabetes_pedigree_function = serializers.FloatField(
        min_value=0.0,
        max_value=3.0,
        help_text='Diabetes pedigree function'
    )
    
    age = serializers.IntegerField(
        min_value=1,
        max_value=120,
        help_text='Age in years'
    )

    def validate(self, data):
        """
        Custom validation if needed
        """
        # Example: Add any cross-field validation here
        if data['glucose'] == 0 and data['blood_pressure'] == 0:
            raise serializers.ValidationError("Glucose and blood pressure cannot both be zero.")
        
        return data