from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    vector: list[float] = Field(..., description="The query vector.")
    top_k: int | None = Field(10, description="Number of results to return.")
    id: str | None = None
    use_index: bool = Field(
        True, description="Whether to use the IVF index if available."
    )
    use_agent: bool = Field(
        False, description="Whether to use adaptive agent for strategy selection."
    )


class SearchResult(BaseModel):
    id: int
    score: float


class SearchResponse(BaseModel):
    query_id: str
    results: list[SearchResult]
    latency_ms: float
    strategy: str | None = Field(None, description="Strategy used by adaptive agent.")


class AddVectorsRequest(BaseModel):
    vectors: list[list[float]] = Field(..., description="List of vectors to add.")
    ids: list[int] | None = Field(
        None, description="Optional list of IDs for the vectors."
    )


class AddVectorsResponse(BaseModel):
    count: int
    total_count: int
    message: str
