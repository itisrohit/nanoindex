from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    vector: list[float] = Field(..., description="The query vector.")
    top_k: int | None = Field(10, description="Number of results to return.")
    id: str | None = None


class SearchResult(BaseModel):
    id: int
    score: float


class SearchResponse(BaseModel):
    query_id: str
    results: list[SearchResult]
    latency_ms: float
