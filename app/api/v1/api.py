"""
API router for v1 endpoints.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import agent, index, search

api_router = APIRouter()
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(index.router, prefix="/index", tags=["index"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
