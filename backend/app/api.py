from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .models import AskRequest, AskResponse, HealthResponse
from .service import StockNewsService, get_service

logger = logging.getLogger(__name__)
settings = get_settings()

## Launching FastAPI app with CORS middleware and endpoints for health check and question answering

app = FastAPI(
    title="Stock News RAG API",
    version="1.0.0",
    description="Closed-book RAG over a small stock-news corpus.",
)
## Adding CORS middleware to allow cross-origin requests from specified origins in settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

## starting the FastAPI application with endpoints for health check and question answering 
@app.get("/health", response_model=HealthResponse)
def health(service: StockNewsService = Depends(get_service)) -> HealthResponse:
    return service.health()

## this is an endpoint for handling user questions about the stock news.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, service: StockNewsService = Depends(get_service)) -> AskResponse:
    try:
        return service.ask(question=request.question, session_id=request.session_id, debug=request.debug)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled /ask failure")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to answer the question") from exc
