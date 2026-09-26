"""文件切片模型。"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from finrag.models.document import Quarter, ReportPeriod


class ChunkMetadata(BaseModel):
    """切片中介資料。"""

    doc_id: str
    chunk_id: str
    parent_id: str | None = None
    section: str
    page: int | None = None
    is_parent: bool
    is_table: bool
    ticker: str
    company_name: str
    fiscal_year: int
    quarter: Quarter
    report_period: ReportPeriod
    table_title: str | None = None


class Chunk(BaseModel):
    """文件切片。"""

    text: str
    metadata: ChunkMetadata


class RegulationChunkMetadata(BaseModel):
    """法規切片中介資料。"""

    doc_id: str
    chunk_id: str
    parent_id: str | None = None
    section: str
    is_parent: bool
    reg_name: str
    reg_pcode: str
    article_no: str | None = None
    chapter: str | None = None
    revision_date: date


class RegulationChunk(BaseModel):
    """法規切片。"""

    text: str
    metadata: RegulationChunkMetadata
