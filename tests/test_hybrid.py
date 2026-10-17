"""Dense 與 Sparse 混合檢索測試。"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from qdrant_client import models

from fin_nexus.contracts import SearchHit
from fin_nexus.retrieval.hybrid import search
from fin_nexus.retrieval.router import RoutePlan
from fin_nexus.storage.qdrant_client import DENSE_VECTOR, SPARSE_VECTOR


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        self.calls.append(list(texts))
        dense = [[0.1, 0.2] for _ in texts]
        sparse = [models.SparseVector(indices=[1], values=[0.5]) for _ in texts]
        return dense, sparse


class _FakeClient:
    def __init__(self, points: list[models.ScoredPoint]) -> None:
        self.points = points
        self.calls: list[dict[str, Any]] = []

    def query_points(self, collection_name: str, **kwargs: Any) -> Any:
        self.calls.append({"collection_name": collection_name, **kwargs})
        return SimpleNamespace(points=self.points)


def _points() -> list[models.ScoredPoint]:
    return [models.ScoredPoint(id="a", version=0, score=0.9, payload={"text": "x"})]


def _plan(**overrides: Any) -> RoutePlan:
    data: dict[str, Any] = {
        "collection": "fin_reports",
        "mode": "dense",
        "query": "台積電 2024 Q1 營收",
        "sparse_query": "營業收入淨額 營業收入 營收",
        "filters": None,
        "top_k": 20,
        "use_sparse": False,
    }
    data.update(overrides)
    return RoutePlan(**data)


def _hybrid_plan(**overrides: Any) -> RoutePlan:
    data: dict[str, Any] = {"mode": "hybrid", "use_sparse": True}
    data.update(overrides)
    return _plan(**data)


def test_dense_search_uses_single_query() -> None:
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()

    search(_plan(), client, embedder=embedder)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["collection_name"] == "fin_reports"
    assert call["query"] == [0.1, 0.2]
    assert call["using"] == DENSE_VECTOR
    assert call["limit"] == 20
    assert "prefetch" not in call


def test_hybrid_search_uses_prefetch_and_rrf() -> None:
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()

    search(_hybrid_plan(), client, embedder=embedder)

    call = client.calls[0]
    prefetch = call["prefetch"]
    assert [p.using for p in prefetch] == [DENSE_VECTOR, SPARSE_VECTOR]
    assert [p.limit for p in prefetch] == [40, 40]
    assert isinstance(call["query"], models.FusionQuery)
    assert call["query"].fusion == models.Fusion.RRF
    assert call["limit"] == 20


def test_filter_is_forwarded() -> None:
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()
    filters = models.Filter(
        must=[models.FieldCondition(key="ticker", match=models.MatchValue(value="2330"))]
    )

    search(_hybrid_plan(filters=filters), client, embedder=embedder)

    assert client.calls[0]["query_filter"] is filters


def test_embedder_called_once_with_query() -> None:
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()

    search(_plan(), client, embedder=embedder)

    assert embedder.calls == [["台積電 2024 Q1 營收"]]


def test_hybrid_search_embeds_sparse_query_separately() -> None:
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()

    search(_hybrid_plan(), client, embedder=embedder)

    assert embedder.calls == [["台積電 2024 Q1 營收", "營業收入淨額 營業收入 營收"]]


def test_hybrid_search_reuses_query_when_sparse_query_is_same() -> None:
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()

    search(_hybrid_plan(sparse_query="台積電 2024 Q1 營收"), client, embedder=embedder)

    assert embedder.calls == [["台積電 2024 Q1 營收"]]


def test_limit_overrides_top_k() -> None:
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()

    search(_plan(), client, embedder=embedder, limit=5)

    assert client.calls[0]["limit"] == 5


def test_returns_search_hits() -> None:
    client = _FakeClient(_points())
    embedder = _FakeEmbedder()

    hits = search(_plan(), client, embedder=embedder)

    assert hits == [SearchHit(id="a", score=0.9, payload={"text": "x"}, collection="fin_reports")]
