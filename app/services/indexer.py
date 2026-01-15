"""
Inverted File Index (IVF) implementation for efficient vector search.
"""

import numpy as np

from app.internal.clustering import kmeans


class IVFIndex:
    """
    Implements a simple Inverted File Index.
    Partitions the vector space into cells (clusters) and only searches
    relevant cells at query time.
    """

    def __init__(self, n_cells: int = 100):
        self.n_cells = n_cells
        self.centroids: np.ndarray | None = None
        # Map cluster_id -> list of indices in the generic datastore
        self.cells: dict[int, list[int]] = {}
        self.is_trained = False

    def train(self, data: np.ndarray) -> None:
        """Calculate centroids using a subset of data."""
        if len(data) < self.n_cells:
            # Fallback for very small datasets
            self.n_cells = max(1, len(data) // 10)

        self.centroids, labels = kmeans(data, self.n_cells)
        self.is_trained = True

        # Initial population from training data
        self.cells = {i: [] for i in range(self.n_cells)}
        for idx, label in enumerate(labels):
            self.cells[int(label)].append(idx)

    def add_vectors(self, vectors: np.ndarray, base_index: int) -> None:
        """Map new vectors to existing centroids."""
        if not self.is_trained or self.centroids is None:
            return

        for i in range(len(vectors)):
            # Find nearest centroid
            diff = self.centroids - vectors[i]
            dist_sq = np.sum(diff**2, axis=1)
            label = int(np.argmin(dist_sq))
            self.cells[label].append(base_index + i)

    def search(self, query: np.ndarray, nprobe: int = 10) -> np.ndarray:
        """
        Search for nearest neighbors by only looking into the nearest cells.
        Returns indices into the base datastore.
        """
        if not self.is_trained or self.centroids is None:
            return np.array([], dtype=int)

        # 1. Find the nprobe nearest centroids to the query
        diff = self.centroids - query
        dist_sq = np.sum(diff**2, axis=1)
        nearest_cells = np.argsort(dist_sq)[:nprobe]

        # 2. Collect all vector indices from these cells
        candidate_indices = []
        for cell_id in nearest_cells:
            candidate_indices.extend(self.cells[cell_id])

        return np.array(candidate_indices, dtype=int)


# Global indexer instance
indexer = IVFIndex()
