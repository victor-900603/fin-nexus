"""Qdrant 連線、Collection 初始化與 Payload Index。"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from fin_nexus.config import QdrantConfig

REPORTS_COLLECTION = "fin_reports"
REGULATIONS_COLLECTION = "regulations"

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
DENSE_SIZE = 1024
DISTANCE = models.Distance.COSINE

_REPORTS_INDEXES: dict[str, models.PayloadSchemaType] = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
    "quarter": models.PayloadSchemaType.KEYWORD,
    "report_period": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
}

_REGULATIONS_INDEXES: dict[str, models.PayloadSchemaType] = {
    "reg_pcode": models.PayloadSchemaType.KEYWORD,
    "article_no": models.PayloadSchemaType.KEYWORD,
    "chapter": models.PayloadSchemaType.KEYWORD,
    "revision_date": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
}


def build_client(config: QdrantConfig | None = None) -> QdrantClient:
    """建立 Qdrant 連線。"""
    settings = config or QdrantConfig()
    return QdrantClient(url=settings.url, api_key=settings.api_key, timeout=settings.timeout)


def _vectors_config() -> dict[str, models.VectorParams]:
    return {DENSE_VECTOR: models.VectorParams(size=DENSE_SIZE, distance=DISTANCE)}


def _sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
    return {SPARSE_VECTOR: models.SparseVectorParams()}


def ensure_collections(client: QdrantClient) -> None:
    """建立尚不存在的 Collection。"""
    for name in (REPORTS_COLLECTION, REGULATIONS_COLLECTION):
        if client.collection_exists(name):
            continue
        client.create_collection(
            collection_name=name,
            vectors_config=_vectors_config(),
            sparse_vectors_config=_sparse_vectors_config(),
        )


def ensure_payload_indexes(client: QdrantClient) -> None:
    """建立各 Collection 的 Payload Index。"""
    for name, fields in (
        (REPORTS_COLLECTION, _REPORTS_INDEXES),
        (REGULATIONS_COLLECTION, _REGULATIONS_INDEXES),
    ):
        for field, schema in fields.items():
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
                wait=True,
            )


def ensure_storage(client: QdrantClient) -> None:
    """建立 Collection 與 Payload Index。"""
    ensure_collections(client)
    ensure_payload_indexes(client)
