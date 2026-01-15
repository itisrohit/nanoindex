import numpy as np
from fastapi import APIRouter, HTTPException

from app.models.schemas import AddVectorsRequest, AddVectorsResponse
from app.services.datastore import datastore
from app.services.indexer import indexer

router = APIRouter()


@router.post("/add", response_model=AddVectorsResponse)
async def add_vectors(request: AddVectorsRequest) -> AddVectorsResponse:
    """
    Add vectors to the index.
    """
    try:
        current_count = datastore.count
        vectors_np = np.array(request.vectors, dtype="float32")
        ids_np = np.array(request.ids, dtype="int64") if request.ids else None

        if len(vectors_np.shape) != 2:
            raise HTTPException(
                status_code=400, detail="Vectors must be a 2D array [n, d]"
            )

        datastore.add_vectors(vectors_np, ids_np)

        # If IVF is trained, we must update it
        if indexer.is_trained:
            indexer.add_vectors(vectors_np, base_index=current_count)

        return AddVectorsResponse(
            count=vectors_np.shape[0],
            total_count=datastore.count,
            message="Vectors added successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/train")
async def train_index(n_cells: int = 100) -> dict[str, str]:
    """
    Train the IVF index using current vectors in the datastore.
    """
    try:
        vectors = datastore.get_vectors()
        if len(vectors) == 0:
            raise HTTPException(status_code=400, detail="No vectors to train on.")

        indexer.n_cells = n_cells
        indexer.train(vectors)
        return {"message": f"Index trained successfully with {n_cells} cells"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/reset")
async def reset_index() -> dict[str, str]:
    """
    Reset the index by deleting all stored vectors.
    """
    try:
        datastore.reset()
        # Also reset the IVF cell mapping
        indexer.cells = {}
        indexer.is_trained = False
        return {"message": "Index reset successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
