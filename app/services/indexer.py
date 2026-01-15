"""
Inverted File Index (IVF) implementation for efficient vector search.
"""

import json
import os

import numpy as np

from app.core.config import settings
from app.internal.clustering import kmeans


class IVFIndex:
    """
    Implements a simple Inverted File Index.
    Partitions the vector space into cells (clusters) and only searches
    relevant cells at query time.
    """

    def __init__(self, directory: str = settings.DATA_DIR):
        self.directory = directory
        self.n_cells = 0
        self.centroids: np.ndarray | None = None
        self.centroid_norms: np.ndarray | None = None
        # Map cluster_id -> list of indices in the generic datastore
        self.cells: dict[int, list[int]] = {}
        self.is_trained = False
        self.save_path = os.path.join(directory, "indexer_state.json")
        self.centroids_path = os.path.join(directory, "centroids.npy")

        # FAISS-like search parameters
        self.max_codes = 50000  # Cap search to 50k vectors max by default

        # Auto-load if existing
        self.load()

    def train(self, data: np.ndarray, n_cells: int = 100) -> None:
        """Calculate centroids using a subset of data."""
        self.n_cells = n_cells
        if len(data) < self.n_cells:
            # Fallback for very small datasets
            self.n_cells = max(1, len(data) // 10)

        # Basic K-Means clustering with subsampling internally
        self.centroids, labels = kmeans(data, self.n_cells)
        self.centroid_norms = np.sum(self.centroids**2, axis=1)
        self.is_trained = True

        # Initial population from training data
        self.cells = {i: [] for i in range(self.n_cells)}
        # Optimized grouping
        for i in range(self.n_cells):
            indices = np.where(labels == i)[0]
            if len(indices) > 0:
                self.cells[i].extend(indices.tolist())

        self.save()

    def add_vectors(
        self, vectors: np.ndarray, base_index: int, v_sq: np.ndarray | None = None
    ) -> None:
        """Map new vectors to clusters using batch quantization."""
        if not self.is_trained or self.centroids is None or self.centroid_norms is None:
            return

        # Faster batch assignment using matrix multiplication
        # dist^2 = ||v||^2 + ||c||^2 - 2 * v.c
        if v_sq is None:
            v_sq = np.sum(vectors**2, axis=1, keepdims=True)
        else:
            if v_sq.ndim == 1:
                v_sq = v_sq[:, np.newaxis]

        c_sq = self.centroid_norms
        dot = np.dot(vectors, self.centroids.T)

        dist_sq = v_sq + c_sq - 2 * dot
        labels = np.argmin(dist_sq, axis=1)

        # Batch update cells to avoid slow Python loops for every vector
        for i in range(self.n_cells):
            matching_indices = np.where(labels == i)[0]
            if len(matching_indices) > 0:
                if i not in self.cells:
                    self.cells[i] = []
                # base_index + matching_indices gives the global indices in DataStore
                self.cells[i].extend((base_index + matching_indices).tolist())

        self.save()

    def search(self, query: np.ndarray, nprobe: int = 10) -> np.ndarray:
        """
        Search for nearest neighbors by only looking into the nearest cells.
        Returns indices into the base datastore.
        """
        if not self.is_trained or self.centroids is None or self.centroid_norms is None:
            return np.array([], dtype=int)

        # 1. Find the nprobe nearest centroids to the query
        q_sq = np.sum(query**2)
        c_sq = self.centroid_norms
        dot = np.dot(self.centroids, query)
        dist_sq = q_sq + c_sq - 2 * dot

        nearest_cells = np.argsort(dist_sq)[:nprobe]

        # 2. Collect indices from these cells, respecting max_codes search budget
        candidate_indices = []
        n_visited = 0
        for cell_id in nearest_cells:
            cid = int(cell_id)
            if cid in self.cells:
                cell_data = self.cells[cid]
                candidate_indices.extend(cell_data)
                n_visited += len(cell_data)

                # Search budget logic (Inspired by FAISS max_codes)
                if n_visited >= self.max_codes:
                    break

        return np.array(candidate_indices, dtype=int)

    def save(self) -> None:
        """Save the indexer state to disk."""
        if not self.is_trained or self.centroids is None:
            return

        # Save centroids as binary
        np.save(self.centroids_path, self.centroids)

        # Save the rest as JSON
        serializable_cells = {str(k): v for k, v in self.cells.items()}
        state = {
            "n_cells": self.n_cells,
            "is_trained": self.is_trained,
            "cells": serializable_cells,
            "max_codes": self.max_codes,
        }
        with open(self.save_path, "w") as f:
            json.dump(state, f)

    def load(self) -> None:
        """Load the indexer state from disk."""
        if not os.path.exists(self.save_path) or not os.path.exists(
            self.centroids_path
        ):
            return

        try:
            self.centroids = np.load(self.centroids_path)
            self.centroid_norms = np.sum(self.centroids**2, axis=1)
            with open(self.save_path) as f:
                state = json.load(f)
                self.n_cells = state["n_cells"]
                self.is_trained = state["is_trained"]
                self.max_codes = state.get("max_codes", 50000)
                # Convert string keys back to int
                self.cells = {int(k): v for k, v in state["cells"].items()}
        except Exception:
            # If loading fails, just stay untrained
            self.is_trained = False

    def get_stats(self) -> dict[str, str | int | float]:
        """Return statistics about the index clusters."""
        if not self.is_trained:
            return {"status": "untrained"}

        cell_sizes = [len(v) for v in self.cells.values()]
        total = sum(cell_sizes)
        avg = total / len(cell_sizes) if cell_sizes else 0
        return {
            "status": "trained",
            "n_cells": self.n_cells,
            "total_indexed": total,
            "min_cell_size": min(cell_sizes) if cell_sizes else 0,
            "max_cell_size": max(cell_sizes) if cell_sizes else 0,
            "avg_cell_size": avg,
            "imbalance_factor": (max(cell_sizes) / avg) if (avg > 0) else 1.0,
            "max_codes": self.max_codes,
        }


# Global indexer instance
indexer = IVFIndex()
