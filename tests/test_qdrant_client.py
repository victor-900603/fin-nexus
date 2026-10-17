"""Qdrant 連線與 Collection 管理單元測試。"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from fin_nexus.config import QdrantConfig
from fin_nexus.storage.qdrant_client import (
    DENSE_SIZE,
    DENSE_VECTOR,
    REGULATIONS_COLLECTION,
    REPORTS_COLLECTION,
    SPARSE_VECTOR,
    build_client,
    ensure_collections,
    ensure_payload_indexes,
    ensure_storage,
)


class _RecordingClient:
    """記錄呼叫的 Qdrant client 替身。"""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.indexes: list[tuple[str, str, models.PayloadSchemaType]] = []

    def collection_exists(self, name: str) -> bool:
        return name in self.created

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: Any,
        sparse_vectors_config: Any,
    ) -> None:
        self.created.append(collection_name)

    def create_payload_index(
        self,
        *,
        collection_name: str,
        field_name: str,
        field_schema: Any,
        wait: bool = True,
    ) -> None:
        self.indexes.append((collection_name, field_name, field_schema))


def _memory_client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_build_client_returns_qdrant_client() -> None:
    client = build_client(QdrantConfig(url="http://localhost:6333", _env_file=None))

    assert isinstance(client, QdrantClient)


def test_ensure_collections_creates_both() -> None:
    client = _memory_client()

    ensure_collections(client)

    assert client.collection_exists(REPORTS_COLLECTION)
    assert client.collection_exists(REGULATIONS_COLLECTION)


def test_ensure_collections_configures_vectors() -> None:
    client = _memory_client()
    ensure_collections(client)

    info = client.get_collection(REPORTS_COLLECTION)
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    assert vectors[DENSE_VECTOR].size == DENSE_SIZE
    assert vectors[DENSE_VECTOR].distance == models.Distance.COSINE
    sparse = info.config.params.sparse_vectors
    assert sparse is not None
    assert SPARSE_VECTOR in sparse


def test_ensure_collections_is_idempotent() -> None:
    client = _memory_client()

    ensure_collections(client)
    ensure_collections(client)

    assert client.collection_exists(REPORTS_COLLECTION)


def test_ensure_payload_indexes_reports() -> None:
    client = _RecordingClient()

    ensure_payload_indexes(client)

    reports = [
        (field, schema)
        for name, field, schema in client.indexes
        if name == REPORTS_COLLECTION
    ]
    assert ("ticker", models.PayloadSchemaType.KEYWORD) in reports
    assert ("fiscal_year", models.PayloadSchemaType.INTEGER) in reports
    assert ("quarter", models.PayloadSchemaType.KEYWORD) in reports
    assert ("report_period", models.PayloadSchemaType.KEYWORD) in reports
    assert ("doc_id", models.PayloadSchemaType.KEYWORD) in reports


def test_ensure_payload_indexes_regulations() -> None:
    client = _RecordingClient()

    ensure_payload_indexes(client)

    regs = [
        (field, schema) for name, field, schema in client.indexes if name == REGULATIONS_COLLECTION
    ]
    assert ("reg_pcode", models.PayloadSchemaType.KEYWORD) in regs
    assert ("article_no", models.PayloadSchemaType.KEYWORD) in regs
    assert ("chapter", models.PayloadSchemaType.KEYWORD) in regs
    assert ("revision_date", models.PayloadSchemaType.KEYWORD) in regs
    assert ("doc_id", models.PayloadSchemaType.KEYWORD) in regs


def test_ensure_storage_creates_collections_and_indexes() -> None:
    client = _RecordingClient()

    ensure_storage(client)

    assert client.created == [REPORTS_COLLECTION, REGULATIONS_COLLECTION]
    assert len(client.indexes) == 10
