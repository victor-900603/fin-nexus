"""Dense 與 Sparse 混合檢索。"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from finrag.contracts import SearchHit
from finrag.retrieval.router import RoutePlan
from finrag.storage.embedder import Embedder
from finrag.storage.qdrant_client import DENSE_VECTOR, SPARSE_VECTOR

PREFETCH_FACTOR = 2


def _to_hits(points: list[models.ScoredPoint], collection: str) -> list[SearchHit]:
    return [
        SearchHit(
            id=str(point.id),
            score=point.score,
            payload=dict(point.payload or {}),
            collection=collection,
        )
        for point in points
    ]


def search(
    plan: RoutePlan,
    client: QdrantClient,
    *,
    embedder: Embedder,
    limit: int | None = None,
) -> list[SearchHit]:
    """依路由計畫執行檢索，回傳 Top-K 候選。"""
    top_k = limit if limit is not None else plan.top_k
    dense_list, sparse_list = embedder.embed([plan.query])
    dense = dense_list[0]
    sparse = sparse_list[0]

    if plan.use_sparse:
        prefetch_limit = top_k * PREFETCH_FACTOR
        response = client.query_points(
            collection_name=plan.collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE_VECTOR, limit=prefetch_limit),
                models.Prefetch(query=sparse, using=SPARSE_VECTOR, limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            query_filter=plan.filters,
        )
    else:
        response = client.query_points(
            collection_name=plan.collection,
            query=dense,
            using=DENSE_VECTOR,
            limit=top_k,
            query_filter=plan.filters,
        )
    return _to_hits(list(response.points), plan.collection)
