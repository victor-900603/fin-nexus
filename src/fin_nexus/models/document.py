"""財報與法規文件模型。"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SourceType(StrEnum):
    """資料來源。"""

    MOPS = "mops"
    MOJ_LAW = "moj_law"


class ReportPeriod(StrEnum):
    """財報期間別。"""

    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class ReportScope(StrEnum):
    """財報編製範圍。"""

    CONSOLIDATED = "consolidated"


Quarter = Literal["Q1", "Q2", "Q3", "Q4"]

QUARTERS: tuple[Quarter, ...] = ("Q1", "Q2", "Q3", "Q4")


def period_for_quarter(quarter: str) -> ReportPeriod:
    """由季度判斷財報期間別。"""
    return ReportPeriod.ANNUAL if quarter == "Q4" else ReportPeriod.QUARTERLY


def document_id(
    ticker: str,
    fiscal_year: int,
    quarter: str,
    scope: ReportScope = ReportScope.CONSOLIDATED,
) -> str:
    """文件識別碼，格式 {ticker}_{fiscal_year}_{quarter}_{scope}。"""
    return f"{ticker}_{fiscal_year}_{quarter}_{scope.value}"


def regulation_id(reg_pcode: str) -> str:
    """法規文件識別碼，格式 law_{reg_pcode}。"""
    return f"law_{reg_pcode}"


class BlockType(StrEnum):
    """解析後內容區塊種類。"""

    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"


class FinancialReportMeta(BaseModel):
    """財務報告書中介資料。"""

    ticker: str
    company_name: str
    fiscal_year: int
    quarter: Quarter
    report_period: ReportPeriod
    report_scope: ReportScope = ReportScope.CONSOLIDATED
    source: SourceType = SourceType.MOPS
    source_uri: str
    parse_engine: str | None = None

    @model_validator(mode="after")
    def _validate_period(self) -> FinancialReportMeta:
        expected = period_for_quarter(self.quarter)
        if self.report_period is not expected:
            raise ValueError(f"quarter {self.quarter} 的 report_period 應為 {expected.value}")
        return self

    @property
    def doc_id(self) -> str:
        """文件識別碼，格式 {ticker}_{fiscal_year}_{quarter}_{report_scope}。"""
        return document_id(self.ticker, self.fiscal_year, self.quarter, self.report_scope)


class RegulationMeta(BaseModel):
    """法規中介資料。"""

    reg_name: str
    reg_pcode: str
    revision_date: date
    source: SourceType = SourceType.MOJ_LAW
    source_uri: str
    parse_engine: str | None = None

    @property
    def doc_id(self) -> str:
        """文件識別碼，格式 law_{reg_pcode}。"""
        return regulation_id(self.reg_pcode)


class RegulationClause(BaseModel):
    """法規條文中的一項、款或目。"""

    level: int = Field(ge=0, le=2)
    text: str


class RegulationArticle(BaseModel):
    """法規的單一條文。"""

    article_no: str
    chapter: str | None = None
    section: str | None = None
    clauses: list[RegulationClause]


class ParsedRegulation(BaseModel):
    """一部法規的解析結果。"""

    meta: RegulationMeta
    articles: list[RegulationArticle]


class Block(BaseModel):
    """解析後的單一內容區塊。"""

    type: BlockType
    text: str
    level: int | None = None
    page: int | None = None


class ParsedDocument(BaseModel):
    """一份文件的解析結果。"""

    meta: FinancialReportMeta
    blocks: list[Block]
