from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_read_model_bad_token():
    response = client.post(
        "/predict-house/v1/",
        headers={"X-Token": "pacmann"},
        json={"OverallQual": "10", "GrLivArea": "10", "TotalBsmtSF": "20",
              "FirstFlrSF": "10", "GarageCars": "2", "GarageArea": "5"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Token header"}


def test_read_model_success_token():
    response = client.post(
        "/predict-house/v1/",
        headers={"X-Token": "pacmannpmdata"},
        json={"OverallQual": "10", "GrLivArea": "10", "TotalBsmtSF": "20",
              "FirstFlrSF": "10", "GarageCars": "2", "GarageArea": "5"},
    )
    assert response.status_code == 200
    assert response.json() == {"result": 63928.82692857142}