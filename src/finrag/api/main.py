"""FastAPI 應用與端點。"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from finrag import __version__
from finrag.api.errors import (
    INTERNAL_ERROR,
    INVALID_QUERY,
    VALIDATION_ERROR,
    ApiError,
)
from finrag.api.schema import (
    ChatRequest,
    ChatResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
)
from finrag.api.service import QueryService
from finrag.contracts import SearchRequest, SearchResponse
from finrag.generation.client import default_client
from finrag.logging import configure_logging
from finrag.retrieval.ontology import default_ontology
from finrag.retrieval.reranker import default_reranker
from finrag.storage.embedder import default_embedder
from finrag.storage.qdrant_client import build_client
from finrag.text import default_token_counter

logger = logging.getLogger(__name__)

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _service(request: Request) -> QueryService:
    service: QueryService | None = request.app.state.service
    if service is None:
        raise RuntimeError("服務尚未初始化")
    return service


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    client = None
    if app.state.service is None:
        client = build_client()
        app.state.service = QueryService(
            client,
            embedder=default_embedder(),
            reranker=default_reranker(),
            llm=default_client(),
            ontology=default_ontology(),
            counter=default_token_counter(),
        )
    try:
        yield
    finally:
        if client is not None:
            client.close()


def create_app(service: QueryService | None = None) -> FastAPI:
    """建立 FastAPI 應用；可注入既有的服務實例。"""
    app = FastAPI(title="FinRAG", version=__version__, lifespan=_lifespan)
    app.state.service = service

    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        return _error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return _error_response(422, VALIDATION_ERROR, "請求格式錯誤")

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("未預期的錯誤")
        return _error_response(500, INTERNAL_ERROR, "內部錯誤")

    @app.get("/health", response_model=HealthResponse)
    def health(
        response: Response, service: Annotated[QueryService, Depends(_service)]
    ) -> HealthResponse:
        report = service.health()
        if report.status != "healthy":
            response.status_code = 503
        return report

    @app.post(
        "/api/v1/search", response_model=SearchResponse, responses=_ERROR_RESPONSES
    )
    def search(
        payload: SearchRequest, service: Annotated[QueryService, Depends(_service)]
    ) -> SearchResponse:
        if not payload.query.strip():
            raise ApiError(INVALID_QUERY, "查詢字串不得為空", 400)
        return service.search(payload)

    @app.post("/api/v1/chat", response_model=ChatResponse, responses=_ERROR_RESPONSES)
    def chat(
        payload: ChatRequest, service: Annotated[QueryService, Depends(_service)]
    ) -> ChatResponse:
        if not payload.query.strip():
            raise ApiError(INVALID_QUERY, "查詢字串不得為空", 400)
        return service.chat(payload)

    return app


app = create_app()
