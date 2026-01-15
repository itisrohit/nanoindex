"""
Simple integration test to verify the add and search workflow.
"""

import httpx
import numpy as np

BASE_URL = "http://localhost:8000/api/v1"


def test_workflow() -> None:
    # 1. Check health
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Health: {response.json()}")

    # 2. Add vectors
    vectors = np.random.rand(10, 128).tolist()
    ids = list(range(10))
    response = httpx.post(
        f"{BASE_URL}/index/add", json={"vectors": vectors, "ids": ids}
    )
    print(f"Add vectors: {response.json()}")

    # 3. Search
    query = np.random.rand(128).tolist()
    response = httpx.post(f"{BASE_URL}/search", json={"vector": query, "top_k": 5})
    print(f"Search: {response.json()}")


if __name__ == "__main__":
    # Start the server first: just dev
    # Then run this script: python scripts/test_api.py
    try:
        test_workflow()
    except Exception as e:
        print(f"Error: {e}. Make sure the server is running with 'just dev'.")
