"""Qdrant 索引結構單元測試。"""

from __future__ import annotations

from datetime import date

from qdrant_client import models

from finrag.models import (
    Chunk,
    ChunkMetadata,
    RegulationChunk,
    RegulationChunkMetadata,
    ReportPeriod,
)
from finrag.storage import schema


def _chunk() -> Chunk:
    return Chunk(
        text="現金及約當現金 100",
        metadata=ChunkMetadata(
            doc_id="2330_2024_Q1_consolidated",
            chunk_id="2330_2024_Q1_consolidated:p0:c0",
            parent_id="2330_2024_Q1_consolidated:p0",
            section="財務報表 > 資產負債表",
            page=3,
            is_parent=False,
            is_table=True,
            ticker="2330",
            company_name="台積電",
            fiscal_year=2024,
            quarter="Q1",
            report_period=ReportPeriod.QUARTERLY,
        ),
    )


def _regulation_chunk() -> RegulationChunk:
    return RegulationChunk(
        text="[證券交易法 > 第一章總則 > 第 1 條]\n為發展國民經濟。",
        metadata=RegulationChunkMetadata(
            doc_id="law_G0400001",
            chunk_id="law_G0400001:p0:c0",
            parent_id="law_G0400001:p0",
            section="第一章總則",
            is_parent=False,
            reg_name="證券交易法",
            reg_pcode="G0400001",
            article_no="第 1 條",
            chapter="第一章總則",
            revision_date=date(2024, 8, 7),
        ),
    )


def _sparse() -> models.SparseVector:
    return models.SparseVector(indices=[1, 2], values=[0.5, 0.6])


def test_point_id_is_deterministic() -> None:
    assert schema.point_id("a:p0:c0") == schema.point_id("a:p0:c0")
    assert schema.point_id("a:p0:c0") != schema.point_id("a:p0:c1")


def test_report_payload_fields() -> None:
    payload = schema.report_payload(_chunk())

    assert payload["doc_type"] == "financial_report"
    assert payload["ticker"] == "2330"
    assert payload["fiscal_year"] == 2024
    assert payload["quarter"] == "Q1"
    assert payload["report_period"] == "quarterly"
    assert payload["is_table"] is True
    assert payload["page"] == 3
    assert payload["text"] == "現金及約當現金 100"


def test_regulation_payload_fields() -> None:
    payload = schema.regulation_payload(_regulation_chunk())

    assert payload["doc_type"] == "regulation"
    assert payload["reg_pcode"] == "G0400001"
    assert payload["article_no"] == "第 1 條"
    assert payload["chapter"] == "第一章總則"
    assert payload["revision_date"] == "2024-08-07"
    assert payload["text"].startswith("[證券交易法")


def test_report_point_vectors_and_payload() -> None:
    chunk = _chunk()

    point = schema.report_point(chunk, [0.1, 0.2], _sparse())

    assert point.id == schema.point_id(chunk.metadata.chunk_id)
    assert isinstance(point.vector, dict)
    assert point.vector["dense"] == [0.1, 0.2]
    assert isinstance(point.vector["sparse"], models.SparseVector)
    assert point.payload is not None
    assert point.payload["chunk_id"] == chunk.metadata.chunk_id


def test_regulation_point_vectors_and_payload() -> None:
    chunk = _regulation_chunk()

    point = schema.regulation_point(chunk, [0.3, 0.4], _sparse())

    assert point.id == schema.point_id(chunk.metadata.chunk_id)
    assert isinstance(point.vector, dict)
    assert point.vector["dense"] == [0.3, 0.4]
    assert point.payload is not None
    assert point.payload["reg_pcode"] == "G0400001"


def test_report_filter_builds_conditions() -> None:
    result = schema.report_filter(ticker="2330", fiscal_year=2024, quarter="Q1")

    assert result is not None
    assert result.must is not None
    conditions = {condition.key: condition.match.value for condition in result.must}
    assert conditions == {"ticker": "2330", "fiscal_year": 2024, "quarter": "Q1"}


def test_report_filter_none_when_empty() -> None:
    assert schema.report_filter() is None


def test_regulation_filter_builds_conditions() -> None:
    result = schema.regulation_filter(reg_pcode="G0400001", article_no="第 1 條")

    assert result is not None
    assert result.must is not None
    conditions = {condition.key: condition.match.value for condition in result.must}
    assert conditions == {"reg_pcode": "G0400001", "article_no": "第 1 條"}


def test_regulation_filter_none_when_empty() -> None:
    assert schema.regulation_filter() is None
