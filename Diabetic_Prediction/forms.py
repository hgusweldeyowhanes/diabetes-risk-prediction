from django import forms

class DiabetesPredictionForm(forms.Form):
    pregnancies = forms.IntegerField(
        label='Pregnancies',
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of times pregnant'
        })
    )
    
    glucose = forms.IntegerField(
        label='Glucose Level',
        min_value=0,
        max_value=400,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Plasma glucose concentration'
        })
    )
    
    blood_pressure = forms.IntegerField(
        label='Blood Pressure',
        min_value=0,
        max_value=202,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Diastolic blood pressure (mm Hg)'
        })
    )
    
    skin_thickness = forms.IntegerField(
        label='Skin Thickness',
        min_value=0,
        max_value=200,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Triceps skin fold thickness (mm)'
        })
    )
    
    insulin = forms.IntegerField(
        label='Insulin Level',
        min_value=0,
        max_value=900,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': '2-Hour serum insulin (mu U/ml)'
        })
    )
    
    bmi = forms.FloatField(
        label='BMI',
        min_value=0.0,
        max_value=70.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Body mass index',
            'step': '0.1'
        })
    )
    
    diabetes_pedigree_function = forms.FloatField(
        label='Diabetes Pedigree Function',
        min_value=0.0,
        max_value=3.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Diabetes pedigree function',
            'step': '0.001'
        })
    )
    
    age = forms.IntegerField(
        label='Age',
        min_value=1,
        max_value=120,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Age in years'
        })
    )