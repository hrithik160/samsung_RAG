"""API routes. Thin layer — all decisions live in retrieval_controller/."""

from fastapi import APIRouter

from app.retrieval_controller.controller import RetrievalController
from app.shared.models import RetrievalDecision, StreamingInput

router = APIRouter()

# Singleton for MVP. Multi-process deployments must replace the
# InMemorySessionStore inside with Redis/DB (see state.py).
controller = RetrievalController()


@router.post(
    "/api/v1/retrieval-controller/decide",
    response_model=RetrievalDecision,
    tags=["retrieval-controller"],
    summary="Decide WHEN retrieval should happen",
)
def decide_endpoint(payload: StreamingInput) -> RetrievalDecision:
    """Accept a streaming transcript chunk, return WAIT/RETRIEVE/NO_RETRIEVAL."""
    return controller.decide(payload)
