"""服務健康狀態組裝。"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from fin_nexus import __version__
from fin_nexus.api.schema import HealthResponse, LLMHealth, QdrantHealth

logger = logging.getLogger(__name__)


def qdrant_health(client: QdrantClient) -> QdrantHealth:
    """查詢 Qdrant Collection 狀態。"""
    try:
        collections = [
            collection.name for collection in client.get_collections().collections
        ]
    except Exception:
        logger.exception("無法取得 Qdrant Collection 清單")
        return QdrantHealth(reachable=False)
    return QdrantHealth(reachable=True, collections=sorted(collections))


def build_health(
    qdrant: QdrantHealth,
    *,
    embedding_loaded: bool,
    reranker_loaded: bool,
    model: str | None,
) -> HealthResponse:
    """組裝服務健康回應。"""
    return HealthResponse(
        status="healthy" if qdrant.reachable else "unhealthy",
        version=__version__,
        qdrant=qdrant,
        embedding="loaded" if embedding_loaded else "not_loaded",
        reranker="loaded" if reranker_loaded else "not_loaded",
        llm=LLMHealth(model=model),
    )
