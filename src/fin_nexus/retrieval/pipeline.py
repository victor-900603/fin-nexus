"""檢索編排：查詢解析、路由、檢索、重排與父區塊擴展。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient

from fin_nexus.contracts import QueryFilters, SearchHit, SearchRequest
from fin_nexus.retrieval.expander import analyze
from fin_nexus.retrieval.hybrid import search
from fin_nexus.retrieval.ontology import Ontology
from fin_nexus.retrieval.parents import expand_parents
from fin_nexus.retrieval.reranker import BgeReranker, rerank
from fin_nexus.retrieval.router import RETRIEVAL_POOL, RoutePlan, route
from fin_nexus.storage.embedder import BgeM3Embedder
from fin_nexus.storage.qdrant_client import REGULATIONS_COLLECTION, REPORTS_COLLECTION
from fin_nexus.storage.schema import DOC_TYPE_REGULATIONS, DOC_TYPE_REPORTS

_COLLECTIONS = {
    DOC_TYPE_REPORTS: REPORTS_COLLECTION,
    DOC_TYPE_REGULATIONS: REGULATIONS_COLLECTION,
}


def _collection_for(filters: QueryFilters | None) -> str | None:
    if filters is None or filters.doc_type is None:
        return None
    return _COLLECTIONS[filters.doc_type]


def _filter_dict(filters: QueryFilters | None) -> dict[str, Any] | None:
    if filters is None:
        return None
    values = filters.model_dump(mode="json", exclude_none=True)
    values.pop("doc_type", None)
    return values or None


@dataclass(frozen=True)
class RetrievalResult:
    """檢索編排結果。"""

    plan: RoutePlan
    hits: list[SearchHit]


def retrieve(
    request: SearchRequest,
    client: QdrantClient,
    *,
    embedder: BgeM3Embedder,
    reranker: BgeReranker,
    ontology: Ontology,
    pool: int = RETRIEVAL_POOL,
) -> RetrievalResult:
    """解析查詢、路由、檢索、重排並擴展父區塊，回傳最終片段。"""
    analysis = analyze(request.query, ontology)
    plan = route(
        analysis,
        filters=_filter_dict(request.filters),
        search_mode=request.search_mode,
        top_k=max(pool, request.top_k),
        collection=_collection_for(request.filters),
    )
    hits = search(plan, client, embedder=embedder)
    if request.enable_rerank:
        hits = rerank(hits, request.query, reranker=reranker, top_k=None)
    contexts = expand_parents(hits, client)
    return RetrievalResult(plan=plan, hits=contexts[: request.top_k])
