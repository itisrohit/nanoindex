import numpy as np
from fastapi import APIRouter, HTTPException

from app.models.schemas import AddVectorsRequest, AddVectorsResponse
from app.services.datastore import datastore

router = APIRouter()


@router.post("/add", response_model=AddVectorsResponse)
async def add_vectors(request: AddVectorsRequest) -> AddVectorsResponse:
    """
    Add vectors to the index.
    """
    try:
        vectors_np = np.array(request.vectors, dtype="float32")
        ids_np = np.array(request.ids, dtype="int64") if request.ids else None

        if len(vectors_np.shape) != 2:
            raise HTTPException(
                status_code=400, detail="Vectors must be a 2D array [n, d]"
            )

        datastore.add_vectors(vectors_np, ids_np)

        return AddVectorsResponse(
            count=vectors_np.shape[0],
            total_count=datastore.count,
            message="Vectors added successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/reset")
async def reset_index() -> dict[str, str]:
    """
    Reset the index by deleting all stored vectors.
    """
    try:
        datastore.reset()
        return {"message": "Index reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
