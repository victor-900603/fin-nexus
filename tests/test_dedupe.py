"""檢索結果去冗餘測試。"""

from __future__ import annotations

from fin_nexus.contracts import SearchHit
from fin_nexus.retrieval.dedupe import dedupe

COLLECTION = "fin_reports"


def _hit(chunk_id: str, text: str) -> SearchHit:
    return SearchHit(
        id=chunk_id,
        score=0.0,
        payload={"chunk_id": chunk_id, "text": text},
        collection=COLLECTION,
    )


def test_identical_texts_kept_once() -> None:
    hits = [_hit("a", "第一行\n第二行"), _hit("b", "第一行\n第二行")]

    assert [hit.id for hit in dedupe(hits)] == ["a"]


def test_high_overlap_removed() -> None:
    hits = [_hit("a", "一\n二\n三\n四"), _hit("b", "一\n二\n三\n四\n五")]

    assert [hit.id for hit in dedupe(hits)] == ["a"]


def test_low_overlap_kept() -> None:
    hits = [_hit("a", "一\n二\n三\n四"), _hit("b", "一\n五\n六\n七")]

    assert [hit.id for hit in dedupe(hits)] == ["a", "b"]


def test_threshold_override() -> None:
    hits = [_hit("a", "一\n二"), _hit("b", "一\n二\n三\n四")]

    assert [hit.id for hit in dedupe(hits, threshold=0.4)] == ["a"]
    assert [hit.id for hit in dedupe(hits, threshold=0.6)] == ["a", "b"]


def test_disjoint_all_kept() -> None:
    hits = [_hit("a", "一\n二"), _hit("b", "三\n四")]

    assert [hit.id for hit in dedupe(hits)] == ["a", "b"]


def test_empty_input() -> None:
    assert dedupe([]) == []
