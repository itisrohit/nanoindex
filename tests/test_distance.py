import numpy as np
import pytest

from app.internal.distance import cosine_similarity, l2_distance, normalize_vectors


def test_l2_distance_basic() -> None:
    query = np.array([1, 0], dtype="float32")
    vectors = np.array([[1, 0], [0, 1], [2, 0]], dtype="float32")

    distances = l2_distance(query, vectors)

    assert distances.shape == (3,)
    assert distances[0] == pytest.approx(0.0)
    assert distances[1] == pytest.approx(np.sqrt(2.0))
    assert distances[2] == pytest.approx(1.0)


def test_cosine_similarity_basic() -> None:
    query = np.array([1, 0], dtype="float32")
    vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype="float32")

    similarities = cosine_similarity(query, vectors)

    assert similarities.shape == (3,)
    assert similarities[0] == pytest.approx(1.0)
    assert similarities[1] == pytest.approx(0.0)
    assert similarities[2] == pytest.approx(-1.0)


def test_normalize_vectors() -> None:
    vectors = np.array([[3, 0], [0, 4]], dtype="float32")
    normalized = normalize_vectors(vectors)

    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert normalized[0, 0] == pytest.approx(1.0)
    assert normalized[1, 1] == pytest.approx(1.0)


def test_distance_numerical_stability() -> None:
    # Test with very large and very small numbers
    query = np.array([1e-10, 0], dtype="float32")
    vectors = np.array([[1e-10, 0], [1e10, 0]], dtype="float32")

    distances = l2_distance(query, vectors)
    assert distances[0] == pytest.approx(0.0)
    assert distances[1] > 1e9
