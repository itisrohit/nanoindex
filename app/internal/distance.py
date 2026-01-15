from typing import cast

import numpy as np


def l2_distance(
    query: np.ndarray, vectors: np.ndarray, v_sq: np.ndarray | None = None
) -> np.ndarray:
    """
    Computes the L2 (Euclidean) distance between a query vector and an array of vectors.

    Args:
        query: 1D numpy array of shape (d,)
        vectors: 2D numpy array of shape (n, d)
        v_sq: Pre-calculated squared norms of vectors. If None, it will be calculated.

    Returns:
        1D numpy array of shape (n,) containing the distances.
    """
    # Using the expansion (x - y)^2 = x^2 + y^2 - 2xy for performance
    q_sq = np.sum(query**2)
    if v_sq is None:
        v_sq = np.sum(vectors**2, axis=1)

    # dot(vectors, query) is the fastest way for matrix-vector product
    dot = np.dot(vectors, query)

    # Use maximum(..., 0) to avoid negative numbers due to floating point precision
    distances_sq = np.maximum(q_sq + v_sq - 2 * dot, 0)
    return cast(np.ndarray, np.sqrt(distances_sq))


def cosine_similarity(
    query: np.ndarray, vectors: np.ndarray, v_norms: np.ndarray | None = None
) -> np.ndarray:
    """
    Computes the Cosine similarity between a query vector and an array of vectors.

    Args:
        query: 1D numpy array of shape (d,)
        vectors: 2D numpy array of shape (n, d)
        v_norms: Pre-calculated L2 norms of vectors.

    Returns:
        1D numpy array of shape (n,) containing the similarities.
    """
    q_norm = np.linalg.norm(query)
    if v_norms is None:
        v_norms = np.linalg.norm(vectors, axis=1)

    dot = np.dot(vectors, query)

    # Add a small epsilon to avoid division by zero
    return cast(np.ndarray, dot / (q_norm * v_norms + 1e-10))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes vectors to unit length (L2 norm = 1).

    Args:
        vectors: 1D or 2D numpy array.

    Returns:
        Normalized vectors of the same shape.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return cast(np.ndarray, vectors / (norm + 1e-10))
    else:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return cast(np.ndarray, vectors / (norms + 1e-10))
