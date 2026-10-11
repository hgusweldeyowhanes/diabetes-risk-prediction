import pytest


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


@pytest.mark.django_db
def test_history_requires_login(api_client):
    res = api_client.get("/api/v1/predictions/history/")
    assert res.status_code in (401, 403)


@pytest.mark.django_db
def test_history_empty_for_user(api_client, user):
    api_client.force_authenticate(user=user)
    res = api_client.get("/api/v1/predictions/history/")
    assert res.status_code == 200
    rows = res.data.get("results", res.data)
    assert rows == []
