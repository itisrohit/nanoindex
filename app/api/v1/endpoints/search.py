from fastapi import APIRouter

from app.models.schemas import SearchRequest, SearchResponse, SearchResult

router = APIRouter()


@router.post("/", response_model=SearchResponse)
async def perform_search(
    request: SearchRequest,
) -> dict[str, str | list[SearchResult] | float]:
    """
    Perform a vector similarity search.
    """
    # Placeholder for actual search logic
    return {"query_id": request.id or "default", "results": [], "latency_ms": 0.0}
