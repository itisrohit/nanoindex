"""
Search orchestration logic with adaptive strategy selection.
"""

import time

import numpy as np

from app.internal.distance import cosine_similarity, l2_distance
from app.models.schemas import SearchResult
from app.services.adaptive import agent
from app.services.datastore import datastore
from app.services.indexer import indexer


class SearchService:
    """
    Orchestrates the vector search process with adaptive strategy selection.
    """

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        metric: str = "l2",
        use_index: bool = True,
        use_agent: bool = False,
    ) -> tuple[list[SearchResult], float, str | None]:
        """
        Perform vector search with optional adaptive strategy selection.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            metric: Distance metric ('l2' or 'cosine')
            use_index: Whether to use IVF index (ignored if use_agent=True)
            use_agent: Whether to use adaptive agent for strategy selection

        Returns:
            Tuple of (results, latency_ms, strategy_name)
        """
        start_time = time.perf_counter()

        # Convert query to numpy
        query_np = np.array(query_vector, dtype="float32")

        # Get full vector pool
        all_vectors = datastore.get_vectors()
        if len(all_vectors) == 0:
            return [], (time.perf_counter() - start_time) * 1000, None

        # Adaptive strategy selection
        strategy_name = None
        if use_agent:
            strategy = agent.select_arm()
            strategy_name = strategy.name
            use_index = strategy.use_index
            nprobe = strategy.nprobe
            max_codes = strategy.max_codes
        else:
            nprobe = 10
            max_codes = None

        # Optimization: Use IVF if available and requested
        candidate_indices: np.ndarray | None = None
        if use_index and indexer.is_trained:
            # Temporarily override indexer max_codes if strategy specifies
            original_max_codes = indexer.max_codes
            if max_codes is not None:
                indexer.max_codes = max_codes

            candidate_indices = indexer.search(
                query_np, nprobe=min(nprobe or 10, indexer.n_cells)
            )

            # Restore original max_codes
            indexer.max_codes = original_max_codes

            # If no candidates found (empty clusters), fallback to flat
            if len(candidate_indices) > 0:
                search_vectors = all_vectors[candidate_indices]
                search_norms = datastore.get_norms()[candidate_indices]
            else:
                search_vectors = all_vectors
                search_norms = datastore.get_norms()
                candidate_indices = None
        else:
            search_vectors = all_vectors
            search_norms = datastore.get_norms()

        # Compute distances based on metric
        if metric == "l2":
            # Pass cached squared norms for speed
            scores = l2_distance(query_np, search_vectors, v_sq=search_norms)
            # For L2, lower is better (ascending)
            relative_top_indices = np.argsort(scores)[:top_k]
        else:
            # For cosine, we would need L2 norms (sqrt of squared norms)
            v_norms = np.sqrt(search_norms) if len(search_norms) > 0 else None
            scores = cosine_similarity(query_np, search_vectors, v_norms=v_norms)
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

        # Update agent if used
        if use_agent and strategy_name:
            agent.update(strategy_name, latency_ms)

        return results, latency_ms, strategy_name


search_service = SearchService()
