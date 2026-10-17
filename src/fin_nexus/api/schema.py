"""API 請求與回應模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from fin_nexus.contracts import SearchRequest

HealthState = Literal["loaded", "not_loaded"]


class ChatRequest(SearchRequest):
    """問答請求。"""

    abstain_threshold: float = Field(default=0.0, ge=0)
    stream: bool = False


class CitationOut(BaseModel):
    """引用來源。"""

    index: int
    doc_id: str | None = None
    chunk_id: str | None = None
    file_name: str | None = None
    source: str
    page: int | None = None
    table: str | None = None
    article_no: str | None = None
    score: float


class ChatResponse(BaseModel):
    """問答回應。"""

    query: str
    target_collection: str
    search_path_taken: str
    answer: str
    abstained: bool
    abstain_reason: str | None = None
    citations: list[CitationOut]
    latency_ms: int


class QdrantHealth(BaseModel):
    """Qdrant 健康狀態。"""

    reachable: bool
    collections: list[str] = Field(default_factory=list)


class LLMHealth(BaseModel):
    """生成模型狀態。"""

    model: str | None = None


class HealthResponse(BaseModel):
    """健康檢查回應。"""

    status: Literal["healthy", "unhealthy"]
    version: str
    qdrant: QdrantHealth
    embedding: HealthState
    reranker: HealthState
    llm: LLMHealth


class ErrorDetail(BaseModel):
    """錯誤內容。"""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """錯誤回應。"""

    error: ErrorDetail
