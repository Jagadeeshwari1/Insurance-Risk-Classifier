"""FastAPI application entrypoint.

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import model_is_ready
from api.routes import health, predict
from src.utils.logging import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model cache at startup so the first request isn't slow.
    if model_is_ready():
        logger.info("Model loaded and ready.")
    else:
        logger.warning("No model found — train one before serving predictions.")
    yield


app = FastAPI(
    title="Insurance Risk Classifier API",
    version="1.0.0",
    description="Calibrated risk scoring with per-decision explanations.",
    lifespan=lifespan,
)

# Allow the React dev server (and prod frontend) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {"service": "insurance-risk-classifier", "docs": "/docs"}
