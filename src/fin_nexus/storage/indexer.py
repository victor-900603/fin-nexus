"""切片向量化並寫入 Qdrant。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

from fin_nexus.config import (
    FINANCIAL_REPORTS,
    REGULATIONS,
    REGULATIONS_CATALOG,
    REPORTS_CATALOG,
    StoragePaths,
)
from fin_nexus.models import (
    Catalog,
    Chunk,
    RegulationCatalog,
    RegulationChunk,
    ReportCatalog,
)
from fin_nexus.storage.embedder import Embedder
from fin_nexus.storage.qdrant_client import (
    REGULATIONS_COLLECTION,
    REPORTS_COLLECTION,
    ensure_storage,
)
from fin_nexus.storage.repository import read_catalog, read_jsonl, write_catalog
from fin_nexus.storage.schema import regulation_point, report_point

DEFAULT_BATCH_SIZE = 64

BuildPoints = Callable[[str], list[models.PointStruct]]


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


def _index_documents[CatalogT: Catalog](
    doc_ids: Sequence[str],
    catalog_path: Path,
    client: QdrantClient,
    *,
    collection: str,
    catalog_type: type[CatalogT],
    build_points: BuildPoints,
    force: bool,
    batch_size: int,
) -> CatalogT:
    """依 doc_id 向量化已切片文件並寫入 Qdrant，更新清單。"""
    ensure_storage(client)
    catalog = read_catalog(catalog_path, catalog_type)
    for doc_id in doc_ids:
        entry = catalog.get(doc_id)
        if entry is None or not entry.status.chunked:
            continue
        if entry.status.indexed and not force:
            continue
        points = build_points(doc_id)
        _delete_document(client, collection, doc_id)
        _upsert(client, collection, points, batch_size)
        status = entry.status.model_copy(update={"indexed": True})
        catalog.upsert(
            doc_id,
            entry.model_copy(update={"status": status, "updated_at": datetime.now(UTC)}),
        )
        write_catalog(catalog_path, catalog)
    return catalog


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

    def build_points(doc_id: str) -> list[models.PointStruct]:
        chunks = read_jsonl(paths.chunks_path(FINANCIAL_REPORTS, doc_id), Chunk)
        dense, sparse = embedder.embed([chunk.text for chunk in chunks])
        return [
            report_point(chunk, vector, weights)
            for chunk, vector, weights in zip(chunks, dense, sparse, strict=True)
        ]

    return _index_documents(
        doc_ids,
        paths.catalog_path(REPORTS_CATALOG),
        client,
        collection=REPORTS_COLLECTION,
        catalog_type=ReportCatalog,
        build_points=build_points,
        force=force,
        batch_size=batch_size,
    )


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

    def build_points(doc_id: str) -> list[models.PointStruct]:
        chunks = read_jsonl(paths.chunks_path(REGULATIONS, doc_id), RegulationChunk)
        dense, sparse = embedder.embed([chunk.text for chunk in chunks])
        return [
            regulation_point(chunk, vector, weights)
            for chunk, vector, weights in zip(chunks, dense, sparse, strict=True)
        ]

    return _index_documents(
        doc_ids,
        paths.catalog_path(REGULATIONS_CATALOG),
        client,
        collection=REGULATIONS_COLLECTION,
        catalog_type=RegulationCatalog,
        build_points=build_points,
        force=force,
        batch_size=batch_size,
    )
