"""Qdrant 索引結構：PointStruct 與 Filter 構建。"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import models

from finrag.models import Chunk, RegulationChunk
from finrag.storage.qdrant_client import DENSE_VECTOR, SPARSE_VECTOR

DOC_TYPE_REPORTS = "financial_report"
DOC_TYPE_REGULATIONS = "regulation"

_POINT_NAMESPACE = uuid.UUID("2f9e7c1a-6b3d-4a52-9c48-1d7e5a0b3f21")


def point_id(chunk_id: str) -> str:
    """由切片識別碼產生決定性的 Qdrant point id。"""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def report_payload(chunk: Chunk) -> dict[str, Any]:
    """財報切片的 Payload。"""
    meta = chunk.metadata
    return {
        "doc_id": meta.doc_id,
        "chunk_id": meta.chunk_id,
        "parent_id": meta.parent_id,
        "section": meta.section,
        "page": meta.page,
        "is_parent": meta.is_parent,
        "is_table": meta.is_table,
        "ticker": meta.ticker,
        "company_name": meta.company_name,
        "fiscal_year": meta.fiscal_year,
        "quarter": meta.quarter,
        "report_period": meta.report_period.value,
        "doc_type": DOC_TYPE_REPORTS,
        "text": chunk.text,
    }


def regulation_payload(chunk: RegulationChunk) -> dict[str, Any]:
    """法規切片的 Payload。"""
    meta = chunk.metadata
    return {
        "doc_id": meta.doc_id,
        "chunk_id": meta.chunk_id,
        "parent_id": meta.parent_id,
        "section": meta.section,
        "is_parent": meta.is_parent,
        "reg_name": meta.reg_name,
        "reg_pcode": meta.reg_pcode,
        "article_no": meta.article_no,
        "chapter": meta.chapter,
        "revision_date": meta.revision_date.isoformat(),
        "doc_type": DOC_TYPE_REGULATIONS,
        "text": chunk.text,
    }


def report_point(
    chunk: Chunk, dense: Sequence[float], sparse: models.SparseVector
) -> models.PointStruct:
    """財報切片轉為 PointStruct。"""
    return models.PointStruct(
        id=point_id(chunk.metadata.chunk_id),
        vector={DENSE_VECTOR: list(dense), SPARSE_VECTOR: sparse},
        payload=report_payload(chunk),
    )


def regulation_point(
    chunk: RegulationChunk, dense: Sequence[float], sparse: models.SparseVector
) -> models.PointStruct:
    """法規切片轉為 PointStruct。"""
    return models.PointStruct(
        id=point_id(chunk.metadata.chunk_id),
        vector={DENSE_VECTOR: list(dense), SPARSE_VECTOR: sparse},
        payload=regulation_payload(chunk),
    )


def _conditions(**fields: Any) -> list[models.Condition]:
    conditions: list[models.Condition] = []
    for key, value in fields.items():
        if value is not None:
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
    return conditions


def report_filter(
    *,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    quarter: str | None = None,
    report_period: str | None = None,
    doc_id: str | None = None,
) -> models.Filter | None:
    """財報查詢的 Payload Filter。"""
    conditions = _conditions(
        ticker=ticker,
        fiscal_year=fiscal_year,
        quarter=quarter,
        report_period=report_period,
        doc_id=doc_id,
    )
    return models.Filter(must=conditions) if conditions else None


def regulation_filter(
    *,
    reg_pcode: str | None = None,
    article_no: str | None = None,
    chapter: str | None = None,
    doc_id: str | None = None,
) -> models.Filter | None:
    """法規查詢的 Payload Filter。"""
    conditions = _conditions(
        reg_pcode=reg_pcode,
        article_no=article_no,
        chapter=chapter,
        doc_id=doc_id,
    )
    return models.Filter(must=conditions) if conditions else None
