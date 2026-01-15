"""
Search orchestration logic.
"""

import time

import numpy as np

from app.internal.distance import cosine_similarity, l2_distance
from app.models.schemas import SearchResult
from app.services.datastore import datastore
from app.services.indexer import indexer


class SearchService:
    """
    Orchestrates the vector search process.
    """

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metric: str = "l2",
        use_index: bool = True,
    ) -> tuple[list[SearchResult], float]:
        start_time = time.perf_counter()

        # Convert query to numpy
        query_np = np.array(query_vector, dtype="float32")

        # Get full vector pool
        all_vectors = datastore.get_vectors()
        if len(all_vectors) == 0:
            return [], (time.perf_counter() - start_time) * 1000

        # Optimization: Use IVF if available and requested
        candidate_indices: np.ndarray | None = None
        if use_index and indexer.is_trained:
            candidate_indices = indexer.search(
                query_np, nprobe=min(10, indexer.n_cells)
            )
            search_vectors = all_vectors[candidate_indices]
        else:
            search_vectors = all_vectors

        # Compute distances based on metric
        if metric == "l2":
            scores = l2_distance(query_np, search_vectors)
            # For L2, lower is better (ascending)
            relative_top_indices = np.argsort(scores)[:top_k]
        else:
            scores = cosine_similarity(query_np, search_vectors)
            # For Cosine, higher is better (descending)
            relative_top_indices = np.argsort(scores)[::-1][:top_k]

        # Map back to absolute indices if we used a subset
        if candidate_indices is not None:
            absolute_top_indices = candidate_indices[relative_top_indices]
        else:
            absolute_top_indices = relative_top_indices

        results = []
        if datastore.ids is not None:
            results = [
                SearchResult(id=int(datastore.ids[idx]), score=float(scores[rel_idx]))
                for rel_idx, idx in enumerate(absolute_top_indices)
            ]

        latency_ms = (time.perf_counter() - start_time) * 1000
        return results, latency_ms


search_service = SearchService()
