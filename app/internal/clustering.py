"""
Minimal K-Means implementation for vector clustering.
"""

import numpy as np


def kmeans(
    data: np.ndarray, k: int, max_iter: int = 10, tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform K-Means clustering on the given data.

    Returns:
        centroids: The K calculated centroids.
        labels: The cluster index for each data point.
    """
    n_samples, n_features = data.shape

    # 1. Randomly initialize centroids from the data
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[indices].copy()

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        old_centroids = centroids.copy()

        # 2. Assignment: find nearest centroid for each point
        # We can reuse our l2_distance but it calculates squared distance if optimized
        # For assignment, we need distance of each point to ALL centroids
        # Since l2_distance is (query, dataset), we can iterate or use broadcasting
        for i in range(n_samples):
            # Distance from point i to all centroids
            # Using broadcasted subtraction for speed
            diff = centroids - data[i]
            dist_sq = np.sum(diff**2, axis=1)
            labels[i] = np.argmin(dist_sq)

        # 3. Update: calculate new centroids as means of clusters
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)

        # 4. Check for convergence
        if np.all(np.abs(centroids - old_centroids) < tol):
            break

    return centroids, labels
