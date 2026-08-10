import pytest
from django.urls import reverse


@pytest.mark.django_db
def test_health(api_client):
    res = api_client.get("/api/v1/health/")
    assert res.status_code == 200
    assert res.data["status"] == "ok"


@pytest.mark.django_db
def test_register(api_client):
    res = api_client.post(
        "/api/v1/auth/register/",
        {
            "username": "newuser",
            "password": "StrongPass123!",
            "email": "n@example.com",
        },
        format="json",
    )
    assert res.status_code in (200, 201)
