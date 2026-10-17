"""自適應檢索路由器：依查詢意圖決定模式、Collection 與過濾條件。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel
from qdrant_client import models

from fin_nexus.contracts import SearchMode
from fin_nexus.retrieval.expander import QueryAnalysis
from fin_nexus.storage.qdrant_client import REGULATIONS_COLLECTION, REPORTS_COLLECTION
from fin_nexus.storage.schema import regulation_filter, report_filter

ResolvedMode = Literal["dense", "filtered_hybrid", "hybrid"]

RETRIEVAL_POOL = 20


class RoutePlan(BaseModel):
    """檢索路由結果。"""

    collection: str
    mode: ResolvedMode
    query: str
    sparse_query: str
    filters: models.Filter | None = None
    top_k: int = RETRIEVAL_POOL
    use_sparse: bool


def _resolve_mode(
    analysis: QueryAnalysis, search_mode: SearchMode, filters: dict[str, Any] | None = None
) -> ResolvedMode:
    if search_mode != "auto":
        return search_mode
    values = filters or {}
    if analysis.reg_pcodes or values.get("reg_pcode"):
        return "hybrid"
    if (
        analysis.tickers
        or analysis.fiscal_years
        or analysis.quarters
        or analysis.report_period
        or values.get("ticker")
        or values.get("fiscal_year")
        or values.get("quarter")
        or values.get("report_period")
    ):
        return "filtered_hybrid"
    return "dense"


def _build_filters(
    analysis: QueryAnalysis, collection: str, filters: dict[str, Any] | None
) -> models.Filter | None:
    values = filters or {}
    if collection == REGULATIONS_COLLECTION:
        return regulation_filter(
            reg_pcode=values.get("reg_pcode") or analysis.reg_pcodes or None,
            article_no=values.get("article_no"),
            chapter=values.get("chapter"),
            doc_id=values.get("doc_id"),
        )
    report_period = values.get("report_period")
    if report_period is None and analysis.report_period is not None:
        report_period = analysis.report_period.value
    return report_filter(
        ticker=values.get("ticker") or analysis.tickers or None,
        fiscal_year=values.get("fiscal_year") or analysis.fiscal_years or None,
        quarter=values.get("quarter") or analysis.quarters or None,
        report_period=report_period,
        doc_id=values.get("doc_id"),
    )


def route(
    analysis: QueryAnalysis,
    *,
    filters: dict[str, Any] | None = None,
    search_mode: SearchMode = "auto",
    top_k: int = RETRIEVAL_POOL,
    collection: str | None = None,
) -> RoutePlan:
    """依查詢解析結果決定檢索路由；明示 Collection 優先於法規啟發式。"""
    mode = _resolve_mode(analysis, search_mode, filters)
    resolved_collection = collection or (
        REGULATIONS_COLLECTION if analysis.reg_pcodes else REPORTS_COLLECTION
    )
    return RoutePlan(
        collection=resolved_collection,
        mode=mode,
        query=analysis.expanded,
        sparse_query=analysis.sparse_query,
        filters=_build_filters(analysis, resolved_collection, filters),
        top_k=top_k,
        use_sparse=mode != "dense",
    )
