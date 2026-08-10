import pytest
from patients.models import Patient
from predictions.models import Prediction


@pytest.mark.django_db
def test_patient_str(user):
    p = Patient.objects.create(created_by=user, first_name="Ada", last_name="Lovelace")
    assert "Ada" in str(p)


@pytest.mark.django_db
def test_prediction_str(user):
    pred = Prediction.objects.create(
        user=user,
        pregnancies=0,
        glucose=120,
        blood_pressure=70,
        skin_thickness=20,
        insulin=80,
        bmi=28.0,
        diabetes_pedigree_function=0.5,
        age=40,
        prediction_result=False,
        probability=0.2,
    )
    assert "Not Diabetic" in str(pred)
