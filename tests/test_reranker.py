"""BGE-Reranker 交叉重排測試。"""

from __future__ import annotations

from fin_nexus.contracts import SearchHit
from fin_nexus.retrieval.reranker import BgeReranker, rerank


class _FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.pairs: list[list[str]] = []

    def compute_score(self, pairs, normalize=True):
        self.pairs = list(pairs)
        return list(self._scores)


def _hit(point_id: str, text: str, score: float) -> SearchHit:
    return SearchHit(id=point_id, score=score, payload={"text": text}, collection="fin_reports")


def test_rerank_sorts_by_score_and_truncates() -> None:
    hits = [_hit("a", "甲", 0.9), _hit("b", "乙", 0.8), _hit("c", "丙", 0.7)]
    reranker = BgeReranker(model=_FakeModel([0.1, 0.9, 0.5]))

    result = rerank(hits, "查詢", reranker=reranker, top_k=2)

    assert [hit.id for hit in result] == ["b", "c"]
    assert [hit.score for hit in result] == [0.9, 0.5]


def test_rerank_preserves_retrieval_score() -> None:
    hits = [_hit("a", "甲", 0.33)]
    reranker = BgeReranker(model=_FakeModel([0.77]))

    result = rerank(hits, "查詢", reranker=reranker)

    assert result[0].score == 0.77
    assert result[0].retrieval_score == 0.33


def test_rerank_uses_payload_text_and_query() -> None:
    hits = [_hit("a", "現金及約當現金", 0.5)]
    model = _FakeModel([0.5])

    rerank(hits, "現金", reranker=BgeReranker(model=model))

    assert model.pairs == [["現金", "現金及約當現金"]]


def test_rerank_default_top_k_is_five() -> None:
    hits = [_hit(str(index), f"文件{index}", 0.1) for index in range(7)]
    reranker = BgeReranker(model=_FakeModel([float(index) for index in range(7)]))

    result = rerank(hits, "查詢", reranker=reranker)

    assert len(result) == 5


def test_rerank_empty_returns_empty() -> None:
    assert rerank([], "查詢", reranker=BgeReranker()) == []


def test_bge_reranker_handles_scalar_score() -> None:
    class _ScalarModel:
        def compute_score(self, pairs, normalize=True):
            return 0.42

    hits = [_hit("a", "甲", 0.1)]

    result = rerank(hits, "查詢", reranker=BgeReranker(model=_ScalarModel()))

    assert result[0].score == 0.42
    assert result[0].retrieval_score == 0.1


def test_is_loaded_reflects_model_state() -> None:
    assert BgeReranker().is_loaded is False
    assert BgeReranker(model=_FakeModel([0.1])).is_loaded is True
