import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from patients.models import Patient
from predictions.models import Prediction


@pytest.fixture
def users(db):
    User = get_user_model()
    owner = User.objects.create_user(username="owner", password="pass12345")
    other = User.objects.create_user(username="other", password="pass12345")
    return owner, other


@pytest.fixture
def client():
    return APIClient()


def _predict(user, **extra):
    fields = {
        "user": user,
        "pregnancies": 1,
        "glucose": 140,
        "blood_pressure": 80,
        "skin_thickness": 20,
        "insulin": 90,
        "bmi": 30.0,
        "diabetes_pedigree_function": 0.5,
        "age": 45,
        "prediction_result": True,
        "probability": 0.72,
        "risk_level": "Elevated",
        "screening": {"score": 12, "max": 26, "label": "High"},
    }
    fields.update(extra)
    return Prediction.objects.create(**fields)


@pytest.mark.django_db
def test_patient_crud_and_annotations(client, users):
    owner, _ = users
    client.force_authenticate(user=owner)
    res = client.post(
        "/api/v1/patients/",
        {
            "first_name": "Abebe",
            "last_name": "Kebede",
            "external_id": "C-100",
            "sex": "M",
        },
        format="json",
    )
    assert res.status_code == 201, res.data
    patient_id = res.data["id"]
    assert res.data["prediction_count"] == 0

    patient = Patient.objects.get(pk=patient_id)
    _predict(owner, patient=patient, glucose=150, probability=0.81)
    _predict(owner, patient=patient, glucose=130, probability=0.55)

    res = client.get(f"/api/v1/patients/{patient_id}/")
    assert res.status_code == 200
    assert res.data["prediction_count"] == 2
    assert res.data["latest_probability"] == pytest.approx(0.55)

    preds = client.get(f"/api/v1/patients/{patient_id}/predictions/")
    assert preds.status_code == 200
    assert len(preds.data) == 2
    assert preds.data[0]["glucose"] == 150


@pytest.mark.django_db
def test_patient_owned_only(client, users):
    owner, other = users
    patient = Patient.objects.create(created_by=owner, first_name="Only", last_name="Mine")
    client.force_authenticate(user=other)
    assert client.get(f"/api/v1/patients/{patient.id}/").status_code == 404
    listed = client.get("/api/v1/patients/")
    rows = listed.data.get("results", listed.data)
    assert rows == []


@pytest.mark.django_db
def test_predict_links_patient(client, users):
    owner, _ = users
    patient = Patient.objects.create(created_by=owner, first_name="Link", last_name="Me", sex="F")
    client.force_authenticate(user=owner)
    res = client.post(
        "/api/v1/predictions/run/",
        {
            "pregnancies": 1,
            "glucose": 148,
            "blood_pressure": 72,
            "skin_thickness": 35,
            "insulin": 0,
            "bmi": 33.6,
            "diabetes_pedigree_function": 0.627,
            "age": 50,
            "sex": "F",
            "patient_id": patient.id,
        },
        format="json",
    )
    assert res.status_code == 200, res.data
    pred_id = res.data.get("prediction_id")
    assert pred_id
    pred = Prediction.objects.get(pk=pred_id)
    assert pred.patient_id == patient.id
    assert pred.user_id == owner.id


@pytest.mark.django_db
def test_history_trend_order_and_filter(client, users):
    owner, other = users
    patient = Patient.objects.create(created_by=owner, first_name="Trend")
    foreign = Patient.objects.create(created_by=other, first_name="Nope")
    a = _predict(owner, patient=patient, glucose=110, probability=0.3)
    b = _predict(owner, patient=patient, glucose=160, probability=0.8)
    _predict(owner, glucose=120, probability=0.4)  # no patient
    _predict(other, patient=foreign, glucose=200, probability=0.9)

    client.force_authenticate(user=owner)
    trend = client.get("/api/v1/predictions/history/trend/")
    assert trend.status_code == 200
    assert len(trend.data) == 3
    assert [p["prediction_id"] for p in trend.data] == sorted(
        [p["prediction_id"] for p in trend.data],
        key=lambda i: Prediction.objects.get(pk=i).created_at,
    )
    assert trend.data[0]["probability"] == pytest.approx(0.3)

    filtered = client.get(f"/api/v1/predictions/history/trend/?patient_id={patient.id}")
    assert len(filtered.data) == 2
    assert {p["prediction_id"] for p in filtered.data} == {a.id, b.id}

    patient_trend = client.get(f"/api/v1/patients/{patient.id}/trend/")
    assert patient_trend.status_code == 200
    assert len(patient_trend.data) == 2
    assert patient_trend.data[0]["glucose"] == 110


@pytest.mark.django_db
def test_history_detail_ownership(client, users):
    owner, other = users
    pred = _predict(owner)
    client.force_authenticate(user=other)
    assert client.get(f"/api/v1/predictions/history/{pred.id}/").status_code == 404
    client.force_authenticate(user=owner)
    res = client.get(f"/api/v1/predictions/history/{pred.id}/")
    assert res.status_code == 200
    assert res.data["id"] == pred.id
