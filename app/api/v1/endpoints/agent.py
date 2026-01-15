"""
API endpoints for adaptive agent management.
"""

from typing import Any

from fastapi import APIRouter

from app.services.adaptive import agent

router = APIRouter()


@router.get("/stats")
async def get_agent_stats() -> dict[str, Any]:
    """
    Get current adaptive agent statistics.
    """
    return agent.get_stats()


@router.post("/reset")
async def reset_agent() -> dict[str, str]:
    """
    Reset the adaptive agent state (for re-exploration).
    """
    agent.reset()
    return {"message": "Agent state reset successfully"}
