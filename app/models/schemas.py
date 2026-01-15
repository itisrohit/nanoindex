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


class AddVectorsRequest(BaseModel):
    vectors: list[list[float]] = Field(..., description="List of vectors to add.")
    ids: list[int] | None = Field(
        None, description="Optional list of IDs for the vectors."
    )


class AddVectorsResponse(BaseModel):
    count: int
    total_count: int
    message: str
