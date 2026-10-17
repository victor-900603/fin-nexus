"""文件清單模型。"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from fin_nexus.models.document import Quarter, ReportPeriod, ReportScope


class StageStatus(BaseModel):
    """文件在各階段是否完成。"""

    downloaded: bool = False
    parsed: bool = False
    chunked: bool = False
    indexed: bool = False


class CatalogEntry(BaseModel):
    """清單項目共用的狀態欄位。"""

    sha256: str | None = None
    status: StageStatus = Field(default_factory=StageStatus)
    chunk_count: int | None = None
    updated_at: datetime | None = None


class Catalog[EntryT: CatalogEntry](BaseModel):
    """以文件識別碼為鍵的清單。"""

    entries: dict[str, EntryT] = Field(default_factory=dict)

    def get(self, doc_id: str) -> EntryT | None:
        """取得文件的清單項目。"""
        return self.entries.get(doc_id)

    def upsert(self, doc_id: str, entry: EntryT) -> None:
        """新增或更新文件的清單項目。"""
        self.entries[doc_id] = entry


class ReportEntry(CatalogEntry):
    """單一財報在清單中的狀態。"""

    ticker: str
    fiscal_year: int
    quarter: Quarter
    report_period: ReportPeriod
    report_scope: ReportScope = ReportScope.CONSOLIDATED


class ReportCatalog(Catalog[ReportEntry]):
    """財報清單，以文件識別碼為鍵。"""


class RegulationEntry(CatalogEntry):
    """單一法規在清單中的狀態。"""

    reg_pcode: str
    reg_name: str
    revision_date: date | None = None


class RegulationCatalog(Catalog[RegulationEntry]):
    """法規清單，以文件識別碼為鍵。"""
