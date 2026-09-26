"""父區塊擴展測試。"""

from __future__ import annotations

from types import SimpleNamespace

from finrag.contracts import SearchHit
from finrag.retrieval.parents import expand_parents
from finrag.storage.schema import point_id

COLLECTION = "fin_reports"


class _FakeClient:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._records = records
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        return self._records


def _child(
    chunk_id: str,
    parent_id: str | None,
    text: str,
    score: float,
    retrieval_score: float | None = None,
) -> SearchHit:
    return SearchHit(
        id=chunk_id,
        score=score,
        payload={"chunk_id": chunk_id, "parent_id": parent_id, "text": text},
        collection=COLLECTION,
        retrieval_score=retrieval_score,
    )


def _parent_record(parent_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=point_id(parent_id),
        payload={
            "chunk_id": parent_id,
            "parent_id": None,
            "is_parent": True,
            "text": text,
        },
    )


def test_child_replaced_by_parent() -> None:
    client = _FakeClient([_parent_record("doc:p1", "父區塊內容")])
    hits = [_child("doc:p1:c1", "doc:p1", "子切片內容", 0.9, 0.5)]

    result = expand_parents(hits, client)

    assert len(result) == 1
    assert result[0].id == point_id("doc:p1")
    assert result[0].payload["text"] == "父區塊內容"
    assert result[0].score == 0.9
    assert result[0].retrieval_score == 0.5
    assert result[0].collection == COLLECTION


def test_shared_parent_kept_once() -> None:
    client = _FakeClient([_parent_record("doc:p1", "父區塊內容")])
    hits = [
        _child("doc:p1:c1", "doc:p1", "子一", 0.9),
        _child("doc:p1:c2", "doc:p1", "子二", 0.8),
    ]

    result = expand_parents(hits, client)

    assert len(result) == 1
    assert result[0].score == 0.9


def test_parent_hit_passes_through() -> None:
    client = _FakeClient([])
    parent = SearchHit(
        id="p1",
        score=0.7,
        payload={"chunk_id": "doc:p1", "parent_id": None, "is_parent": True},
        collection=COLLECTION,
    )

    result = expand_parents([parent], client)

    assert result == [parent]
    assert client.calls == []


def test_child_without_parent_stays() -> None:
    client = _FakeClient([])
    hit = _child("doc:p1:c1", None, "子切片內容", 0.9)

    result = expand_parents([hit], client)

    assert result == [hit]
    assert client.calls == []


def test_missing_parent_falls_back_to_child() -> None:
    client = _FakeClient([])
    hits = [
        _child("doc:p1:c1", "doc:p1", "子一", 0.9),
        _child("doc:p1:c2", "doc:p1", "子二", 0.8),
    ]

    result = expand_parents(hits, client)

    assert len(result) == 1
    assert result[0].payload["text"] == "子一"


def test_parent_and_child_share_parent_kept_once() -> None:
    client = _FakeClient([_parent_record("doc:p1", "父區塊內容")])
    parent = SearchHit(
        id=point_id("doc:p1"),
        score=0.7,
        payload={"chunk_id": "doc:p1", "parent_id": None, "is_parent": True},
        collection=COLLECTION,
    )

    result = expand_parents([parent, _child("doc:p1:c1", "doc:p1", "子切片內容", 0.9)], client)

    assert len(result) == 1
    assert result[0].score == 0.7


def test_child_before_parent_kept_once() -> None:
    client = _FakeClient([_parent_record("doc:p1", "父區塊內容")])
    parent = SearchHit(
        id=point_id("doc:p1"),
        score=0.7,
        payload={"chunk_id": "doc:p1", "parent_id": None, "is_parent": True},
        collection=COLLECTION,
    )

    result = expand_parents([_child("doc:p1:c1", "doc:p1", "子切片內容", 0.9), parent], client)

    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].payload["text"] == "父區塊內容"


def test_retrieve_uses_point_ids() -> None:
    client = _FakeClient([_parent_record("doc:p1", "父區塊內容")])
    hits = [_child("doc:p1:c1", "doc:p1", "子切片內容", 0.9)]

    expand_parents(hits, client)

    assert client.calls == [
        {
            "collection_name": COLLECTION,
            "ids": [point_id("doc:p1")],
            "with_payload": True,
            "with_vectors": False,
        }
    ]
