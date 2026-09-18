"""文件切片模型。"""

from __future__ import annotations

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


class Chunk(BaseModel):
    """文件切片。"""

    text: str
    metadata: ChunkMetadata
