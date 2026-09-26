"""父區塊擴展：以命中的子切片取回所屬父區塊。"""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient

from finrag.contracts import SearchHit
from finrag.storage.schema import point_id


def _parent_id(hit: SearchHit) -> str | None:
    if hit.payload.get("is_parent"):
        return None
    value = hit.payload.get("parent_id")
    return str(value) if value else None


def _load_parents(
    client: QdrantClient, grouped: dict[str, list[str]]
) -> dict[tuple[str, str], SearchHit]:
    parents: dict[tuple[str, str], SearchHit] = {}
    for collection, parent_ids in grouped.items():
        records = client.retrieve(
            collection_name=collection,
            ids=[point_id(parent_id) for parent_id in parent_ids],
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = dict(record.payload or {})
            chunk_id = str(payload.get("chunk_id", ""))
            parents[(collection, chunk_id)] = SearchHit(
                id=str(record.id),
                score=0.0,
                payload=payload,
                collection=collection,
            )
    return parents


def expand_parents(hits: Sequence[SearchHit], client: QdrantClient) -> list[SearchHit]:
    """將命中的子切片替換為所屬父區塊，同一父僅保留一次。"""
    grouped: dict[str, list[str]] = {}
    for hit in hits:
        parent_id = _parent_id(hit)
        if parent_id is None:
            continue
        parent_ids = grouped.setdefault(hit.collection, [])
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)

    parents = _load_parents(client, grouped) if grouped else {}

    result: list[SearchHit] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        parent_id = _parent_id(hit)
        chunk_id = str(hit.payload.get("chunk_id") or hit.id)
        key = (hit.collection, parent_id if parent_id is not None else chunk_id)
        if key in seen:
            continue
        seen.add(key)
        if parent_id is None:
            result.append(hit)
            continue
        parent = parents.get((hit.collection, parent_id))
        if parent is None:
            result.append(hit)
            continue
        result.append(
            parent.model_copy(
                update={"score": hit.score, "retrieval_score": hit.retrieval_score}
            )
        )
    return result
