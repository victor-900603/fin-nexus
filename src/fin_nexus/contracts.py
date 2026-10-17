"""跨層共用契約：檢索請求、回應與命中結果。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fin_nexus.models import Quarter, ReportPeriod

SearchMode = Literal["auto", "dense", "filtered_hybrid", "hybrid"]
DocType = Literal["financial_report", "regulation"]

DEFAULT_TOP_K = 3
MAX_TOP_K = 100


class QueryFilters(BaseModel):
    """檢索過濾條件。"""

    model_config = ConfigDict(extra="forbid")

    ticker: str | None = None
    fiscal_year: int | None = None
    quarter: Quarter | None = None
    report_period: ReportPeriod | None = None
    reg_pcode: str | None = None
    article_no: str | None = None
    chapter: str | None = None
    doc_id: str | None = None
    doc_type: DocType | None = None


class SearchRequest(BaseModel):
    """檢索請求。"""

    query: str = Field(min_length=1)
    filters: QueryFilters | None = None
    search_mode: SearchMode = "auto"
    enable_rerank: bool = False
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class SearchResult(BaseModel):
    """單筆檢索結果。"""

    id: str
    score: float
    retrieval_score: float | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    file_name: str | None = None
    source: str
    page: int | None = None
    table: str | None = None
    article_no: str | None = None
    text: str


class SearchResponse(BaseModel):
    """檢索回應。"""

    query: str
    target_collection: str
    search_path_taken: str
    results: list[SearchResult]
    latency_ms: int


class SearchHit(BaseModel):
    """檢索命中的候選點。"""

    id: str
    score: float
    payload: dict[str, object]
    collection: str
    retrieval_score: float | None = None
