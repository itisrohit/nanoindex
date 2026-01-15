import pytest
from fastapi.testclient import TestClient

from app.main import get_application

client = TestClient(get_application())


def test_read_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_index_and_search_workflow() -> None:
    # 0. Reset index
    client.delete("/api/v1/index/reset")

    # 1. Add vectors
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    ids = [1, 2]
    add_response = client.post(
        "/api/v1/index/add/", json={"vectors": vectors, "ids": ids}
    )
    assert add_response.status_code == 200
    assert add_response.json()["count"] == 2

    # 2. Search
    search_response = client.post(
        "/api/v1/search/", json={"vector": [1.0, 0.0], "top_k": 1}
    )
    assert search_response.status_code == 200
    results = search_response.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["score"] == pytest.approx(0.0)  # L2 distance
