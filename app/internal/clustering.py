"""
Minimal K-Means implementation for vector clustering.
"""

import numpy as np


def kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int = 10,
    tol: float = 1e-4,
    subsample_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform K-Means clustering on the given data.
    Uses an optimized batch assignment step and data subsampling for large datasets.

    Args:
        data: Input vectors (N, D).
        k: Number of centroids.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        subsample_size: Maximum number of points to use for training (Inspired by FAISS).

    Returns:
        centroids: The K calculated centroids.
        labels: The cluster index for each original data point.
    """
    n_samples, n_features = data.shape

    # 1. Subsample heavy datasets to speed up training iterations
    # FAISS usually uses a subset of vectors for centroid training.
    train_data = data
    if n_samples > subsample_size:
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        train_data = data[indices]

    n_train = train_data.shape[0]

    # 2. Randomly initialize centroids from the training subset
    centroid_indices = np.random.choice(n_train, k, replace=False)
    centroids = train_data[centroid_indices].copy()

    train_data_sq = np.sum(train_data**2, axis=1, keepdims=True)

    for _ in range(max_iter):
        old_centroids = centroids.copy()

        # 3. Optimized Batch Assignment
        # dist^2 = ||x||^2 + ||c||^2 - 2 * x.c
        centroid_sq = np.sum(centroids**2, axis=1)
        dot = np.dot(train_data, centroids.T)

        dist_sq = train_data_sq + centroid_sq - 2 * dot
        labels = np.argmin(dist_sq, axis=1)

        # 4. Update: calculate new centroids as means of clusters
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centroids[i] = np.mean(train_data[mask], axis=0)
            else:
                # Handle empty cluster by re-initializing to a random point
                centroids[i] = train_data[np.random.choice(n_train)]

        # 5. Check for convergence
        if np.all(np.abs(centroids - old_centroids) < tol):
            break

    # After training on subset, we assign ALL data points to labels
    # We use the same dot-product trick for the full dataset assignment
    all_data_sq = np.sum(data**2, axis=1, keepdims=True)
    all_centroid_sq = np.sum(centroids**2, axis=1)
    all_dot = np.dot(data, centroids.T)
    final_dist_sq = all_data_sq + all_centroid_sq - 2 * all_dot
    final_labels = np.argmin(final_dist_sq, axis=1)

    return centroids, final_labels
