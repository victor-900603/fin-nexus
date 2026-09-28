"""API 端點單元測試。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from finrag import __version__
from finrag.api.main import create_app
from finrag.api.service import QueryService
from finrag.config import GenerationConfig
from finrag.contracts import SearchHit
from finrag.generation.client import ChatMessage
from finrag.retrieval.ontology import Ontology
from finrag.retrieval.reranker import BgeReranker
from finrag.storage.embedder import BgeM3Embedder


class _FakeLLM:
    def __init__(self, reply: str = "營業收入為 100[1]") -> None:
        self._reply = reply
        self.calls = 0

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls += 1
        return self._reply


class _FailingLLM:
    def chat(self, messages: Sequence[ChatMessage]) -> str:
        raise httpx.ConnectError("llm down")


class _BrokenClient:
    def get_collections(self) -> Any:
        raise RuntimeError("qdrant down")


class _ClosableClient:
    def __init__(self) -> None:
        self.closed = False
        self._inner = QdrantClient(":memory:")

    def get_collections(self) -> Any:
        return self._inner.get_collections()

    def close(self) -> None:
        self.closed = True
        self._inner.close()


def _raise_qdrant_error(
    plan: Any, client: Any, *, embedder: Any = None, limit: Any = None
) -> Any:
    raise ResponseHandlingException("connection refused")


def _raise_runtime_error(
    plan: Any, client: Any, *, embedder: Any = None, limit: Any = None
) -> Any:
    raise RuntimeError("boom")


def _report_hit() -> SearchHit:
    return SearchHit(
        id="point-1",
        score=0.9,
        retrieval_score=0.7,
        collection="fin_reports",
        payload={
            "doc_id": "2330_2024_Q2_consolidated",
            "chunk_id": "2330_2024_Q2_consolidated:p1:c1",
            "text": "營業收入 100",
            "page": 5,
            "table_title": "合併綜合損益表",
            "is_parent": True,
        },
    )


def _regulation_hit() -> SearchHit:
    return SearchHit(
        id="point-2",
        score=0.8,
        collection="regulations",
        payload={
            "doc_id": "law_G0400001",
            "chunk_id": "law_G0400001:p1:c1",
            "text": "第 1 條 為發展國民經濟",
            "article_no": "第 1 條",
            "is_parent": True,
        },
    )


def _make_service(
    monkeypatch: pytest.MonkeyPatch,
    hits: list[SearchHit],
    *,
    reply: str = "營業收入為 100[1]",
    model: str | None = "test-model",
    llm: Any | None = None,
    qdrant: Any | None = None,
    embedder: BgeM3Embedder | None = None,
    reranker: BgeReranker | None = None,
    ontology: Any | None = None,
) -> QueryService:
    monkeypatch.setattr(
        "finrag.retrieval.pipeline.search",
        lambda plan, client, *, embedder=None, limit=None: list(hits),
    )
    monkeypatch.setattr(
        "finrag.retrieval.pipeline.rerank",
        lambda hits, query, *, reranker=None, top_k=5: list(hits)[:top_k],
    )
    return QueryService(
        qdrant if qdrant is not None else QdrantClient(":memory:"),
        embedder=embedder if embedder is not None else BgeM3Embedder(),
        reranker=reranker if reranker is not None else BgeReranker(),
        llm=llm if llm is not None else _FakeLLM(reply),
        ontology=ontology if ontology is not None else Ontology.load(),
        generation_config=GenerationConfig(_env_file=None, model=model),
        counter=len,
    )


def _make_client(
    monkeypatch: pytest.MonkeyPatch, hits: list[SearchHit], **kwargs: Any
) -> TestClient:
    return TestClient(create_app(_make_service(monkeypatch, hits, **kwargs)))


def test_search_returns_results(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [_report_hit()])

    response = client.post(
        "/api/v1/search", json={"query": "台積電 2024 Q2 營收", "enable_rerank": False}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "台積電 2024 Q2 營收"
    assert body["target_collection"] == "fin_reports"
    assert body["search_path_taken"] in {"dense", "filtered_hybrid", "hybrid"}
    assert isinstance(body["latency_ms"], int)
    result = body["results"][0]
    assert result["file_name"] == "2330_2024_Q2_consolidated.pdf"
    assert result["table"] == "合併綜合損益表"
    assert result["page"] == 5
    assert result["source"] == "fin_reports"
    assert result["retrieval_score"] == 0.7


def test_search_uses_regulation_collection_from_doc_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, [_regulation_hit()])

    response = client.post(
        "/api/v1/search",
        json={
            "query": "證交法 第 1 條",
            "filters": {"doc_type": "regulation"},
            "enable_rerank": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_collection"] == "regulations"
    result = body["results"][0]
    assert result["file_name"] == "law_G0400001.html"
    assert result["article_no"] == "第 1 條"


def test_search_truncates_to_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [_report_hit(), _regulation_hit()])

    response = client.post(
        "/api/v1/search", json={"query": "查詢", "enable_rerank": True, "top_k": 1}
    )

    assert len(response.json()["results"]) == 1


def test_chat_returns_answer_and_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [_report_hit()])

    response = client.post(
        "/api/v1/chat", json={"query": "台積電 2024 Q2 營收", "enable_rerank": False}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "營業收入為 100[1]"
    assert body["abstained"] is False
    assert body["abstain_reason"] is None
    citation = body["citations"][0]
    assert citation["index"] == 1
    assert citation["file_name"] == "2330_2024_Q2_consolidated.pdf"
    assert citation["table"] == "合併綜合損益表"
    assert citation["score"] == 0.9


def test_chat_abstains_without_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [])

    response = client.post(
        "/api/v1/chat", json={"query": "查詢", "enable_rerank": False}
    )

    body = response.json()
    assert body["abstained"] is True
    assert body["abstain_reason"] == "no_hits"
    assert body["citations"] == []


def test_chat_abstains_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [_report_hit()])

    response = client.post(
        "/api/v1/chat",
        json={"query": "查詢", "enable_rerank": False, "abstain_threshold": 0.95},
    )

    body = response.json()
    assert body["abstained"] is True
    assert body["abstain_reason"] == "score_below_threshold"


def test_chat_returns_503_when_model_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, [_report_hit()], model=None)

    response = client.post(
        "/api/v1/chat", json={"query": "查詢", "enable_rerank": False}
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "LLM_NOT_CONFIGURED"


def test_chat_returns_502_when_llm_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [_report_hit()], llm=_FailingLLM())

    response = client.post(
        "/api/v1/chat", json={"query": "查詢", "enable_rerank": False}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_PROVIDER_ERROR"


def test_search_returns_503_when_vector_store_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, [])
    monkeypatch.setattr("finrag.retrieval.pipeline.search", _raise_qdrant_error)

    response = client.post(
        "/api/v1/search", json={"query": "查詢", "enable_rerank": False}
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "VECTOR_STORE_DISCONNECTED"


def test_search_rejects_blank_query(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [])

    response = client.post("/api/v1/search", json={"query": "   "})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_QUERY"


def test_chat_rejects_blank_query(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [])

    response = client.post("/api/v1/chat", json={"query": "   "})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_QUERY"


def test_search_without_doc_id_has_no_file_name(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = SearchHit(
        id="point-3",
        score=0.5,
        collection="fin_reports",
        payload={"chunk_id": "x:p0:c0", "text": "內容", "is_parent": True},
    )
    client = _make_client(monkeypatch, [hit])

    response = client.post(
        "/api/v1/search", json={"query": "查詢", "enable_rerank": False}
    )

    assert response.json()["results"][0]["file_name"] is None


def test_search_rejects_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [])

    response = client.post("/api/v1/search", json={"query": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_search_rejects_top_k_above_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [])

    response = client.post("/api/v1/search", json={"query": "查詢", "top_k": 101})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_search_rejects_unknown_filter_field(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [])

    response = client.post(
        "/api/v1/search", json={"query": "查詢", "filters": {"reg_pcod": "G0400001"}}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_search_rejects_invalid_quarter(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [])

    response = client.post(
        "/api/v1/search", json={"query": "查詢", "filters": {"quarter": "Q5"}}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_search_regulation_filters_use_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [_regulation_hit()])

    response = client.post(
        "/api/v1/search",
        json={
            "query": "公開說明書",
            "enable_rerank": False,
            "filters": {
                "doc_type": "regulation",
                "reg_pcode": "G0400001",
                "article_no": "第 1 條",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_collection"] == "regulations"
    assert body["search_path_taken"] == "hybrid"


def test_search_report_filters_use_filtered_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, [_report_hit()])

    response = client.post(
        "/api/v1/search",
        json={
            "query": "營業收入",
            "enable_rerank": False,
            "filters": {"ticker": "2330", "fiscal_year": 2023, "quarter": "Q1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["target_collection"] == "fin_reports"
    assert body["search_path_taken"] == "filtered_hybrid"


def test_search_without_filters_stays_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [_report_hit()])

    response = client.post(
        "/api/v1/search", json={"query": "營業收入", "enable_rerank": False}
    )

    assert response.status_code == 200
    assert response.json()["search_path_taken"] == "dense"


def test_health_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(monkeypatch, [])

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"] == __version__
    assert body["qdrant"]["reachable"] is True
    assert body["embedding"] == "not_loaded"
    assert body["reranker"] == "not_loaded"
    assert body["llm"]["model"] == "test-model"


def test_health_unhealthy_when_qdrant_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(monkeypatch, [], qdrant=_BrokenClient())

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["qdrant"]["reachable"] is False


def test_health_does_not_load_models(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = BgeM3Embedder()
    reranker = BgeReranker()
    service = _make_service(monkeypatch, [], embedder=embedder, reranker=reranker)
    client = TestClient(create_app(service))

    client.get("/health")

    assert embedder.is_loaded is False
    assert reranker.is_loaded is False


def test_lifespan_creates_and_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _ClosableClient()
    monkeypatch.setattr("finrag.api.main.build_client", lambda: stub)
    app = create_app()

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert app.state.service is not None

    assert stub.closed is True


def test_unexpected_error_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _make_service(monkeypatch, [])
    monkeypatch.setattr("finrag.retrieval.pipeline.search", _raise_runtime_error)
    client = TestClient(create_app(service), raise_server_exceptions=False)

    response = client.post("/api/v1/search", json={"query": "查詢"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"


def test_missing_service_returns_500() -> None:
    client = TestClient(create_app(), raise_server_exceptions=False)

    response = client.get("/health")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
