"""FastAPI application entrypoint.

Member 1 (Retrieval Controller) owns this service in the MVP.
Members 2-4 integrate via HTTP contracts defined in app/shared/models.py.
Business logic lives in app/retrieval_controller/; routes stay thin.
"""

from fastapi import FastAPI

from app.api.routes import router as retrieval_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Streaming RAG — Retrieval Controller",
        version="0.1.0",
        description="Member 1: decides WHEN retrieval should happen (WAIT / RETRIEVE / NO_RETRIEVAL).",
    )
    app.include_router(retrieval_router)
    return app


app = create_app()


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "component": "retrieval-controller", "version": "0.1.0"}
