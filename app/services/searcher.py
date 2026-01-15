"""
Search orchestration logic.
"""

import time

import numpy as np

from app.internal.distance import cosine_similarity, l2_distance
from app.models.schemas import SearchResult
from app.services.datastore import datastore


class SearchService:
    """
    Orchestrates the vector search process.
    """

    def search(
        self, query_vector: list[float], top_k: int = 10, metric: str = "l2"
    ) -> tuple[list[SearchResult], float]:
        start_time = time.perf_counter()

        # Convert query to numpy
        query_np = np.array(query_vector, dtype="float32")

        # Get vectors from datastore
        indexed_vectors = datastore.get_vectors()

        if len(indexed_vectors) == 0:
            return [], (time.perf_counter() - start_time) * 1000

        # Compute distances based on metric
        if metric == "l2":
            scores = l2_distance(query_np, indexed_vectors)
            # For L2, lower is better (ascending)
            top_indices = np.argsort(scores)[:top_k]
        else:
            scores = cosine_similarity(query_np, indexed_vectors)
            # For Cosine, higher is better (descending)
            top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        if datastore.ids is not None:
            results = [
                SearchResult(id=int(datastore.ids[idx]), score=float(scores[idx]))
                for idx in top_indices
            ]

        latency_ms = (time.perf_counter() - start_time) * 1000
        return results, latency_ms


search_service = SearchService()
