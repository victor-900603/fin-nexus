"""切片向量化並寫入 Qdrant。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from qdrant_client import QdrantClient, models

from finrag.config import (
    FINANCIAL_REPORTS,
    REGULATIONS,
    REGULATIONS_CATALOG,
    REPORTS_CATALOG,
    StoragePaths,
)
from finrag.models import (
    Chunk,
    RegulationCatalog,
    RegulationChunk,
    ReportCatalog,
)
from finrag.storage.embedder import Embedder
from finrag.storage.qdrant_client import (
    REGULATIONS_COLLECTION,
    REPORTS_COLLECTION,
    ensure_storage,
)
from finrag.storage.repository import read_catalog, read_jsonl, write_catalog
from finrag.storage.schema import regulation_point, report_point

DEFAULT_BATCH_SIZE = 64


def _upsert(
    client: QdrantClient,
    collection: str,
    points: Sequence[models.PointStruct],
    batch_size: int,
) -> None:
    for start in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection,
            points=list(points[start : start + batch_size]),
            wait=True,
        )


def _delete_document(client: QdrantClient, collection: str, doc_id: str) -> None:
    """移除同一文件既有的向量，避免重切後殘留舊切片。"""
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                ]
            )
        ),
        wait=True,
    )


def index_reports(
    doc_ids: Sequence[str],
    paths: StoragePaths,
    client: QdrantClient,
    *,
    embedder: Embedder,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ReportCatalog:
    """將財報切片向量化並寫入 Qdrant。"""
    ensure_storage(client)
    catalog_path = paths.catalog_path(REPORTS_CATALOG)
    catalog = read_catalog(catalog_path, ReportCatalog)
    for doc_id in doc_ids:
        entry = catalog.get(doc_id)
        if entry is None or not entry.status.chunked:
            continue
        if entry.status.indexed and not force:
            continue
        chunks = read_jsonl(paths.chunks_path(FINANCIAL_REPORTS, doc_id), Chunk)
        dense, sparse = embedder.embed([chunk.text for chunk in chunks])
        points = [
            report_point(chunk, vector, weights)
            for chunk, vector, weights in zip(chunks, dense, sparse, strict=True)
        ]
        _delete_document(client, REPORTS_COLLECTION, doc_id)
        _upsert(client, REPORTS_COLLECTION, points, batch_size)
        status = entry.status.model_copy(update={"indexed": True})
        catalog.upsert(
            doc_id,
            entry.model_copy(update={"status": status, "updated_at": datetime.now(UTC)}),
        )
        write_catalog(catalog_path, catalog)
    return catalog


def index_regulations(
    doc_ids: Sequence[str],
    paths: StoragePaths,
    client: QdrantClient,
    *,
    embedder: Embedder,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RegulationCatalog:
    """將法規切片向量化並寫入 Qdrant。"""
    ensure_storage(client)
    catalog_path = paths.catalog_path(REGULATIONS_CATALOG)
    catalog = read_catalog(catalog_path, RegulationCatalog)
    for doc_id in doc_ids:
        entry = catalog.get(doc_id)
        if entry is None or not entry.status.chunked:
            continue
        if entry.status.indexed and not force:
            continue
        chunks = read_jsonl(paths.chunks_path(REGULATIONS, doc_id), RegulationChunk)
        dense, sparse = embedder.embed([chunk.text for chunk in chunks])
        points = [
            regulation_point(chunk, vector, weights)
            for chunk, vector, weights in zip(chunks, dense, sparse, strict=True)
        ]
        _delete_document(client, REGULATIONS_COLLECTION, doc_id)
        _upsert(client, REGULATIONS_COLLECTION, points, batch_size)
        status = entry.status.model_copy(update={"indexed": True})
        catalog.upsert(
            doc_id,
            entry.model_copy(update={"status": status, "updated_at": datetime.now(UTC)}),
        )
        write_catalog(catalog_path, catalog)
    return catalog
