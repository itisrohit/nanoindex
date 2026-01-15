"""
API endpoints for vector search.
"""

from fastapi import APIRouter

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.searcher import search_service

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def perform_search(
    request: SearchRequest,
) -> dict[str, str | list[SearchResult] | float | None]:
    """
    Perform a vector similarity search with optional adaptive strategy selection.
    """
    results, latency, strategy = search_service.search(
        query_vector=request.vector,
        top_k=request.top_k or 10,
        use_index=request.use_index,
        use_agent=request.use_agent,
    )

    return {
        "query_id": request.id or "default",
        "results": results,
        "latency_ms": latency,
        "strategy": strategy,
    }
