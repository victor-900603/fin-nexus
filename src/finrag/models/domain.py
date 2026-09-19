"""領域資料：公司名稱與年度換算。"""

from __future__ import annotations

from dataclasses import dataclass

from finrag.models.document import (
    FinancialReportMeta,
    Quarter,
    ReportPeriod,
    document_id,
    period_for_quarter,
)

COMPANY_NAMES: dict[str, str] = {
    "2330": "台積電",
    "2303": "聯電",
    "2454": "聯發科",
}


def to_western_year(roc_year: int) -> int:
    """民國年轉西元年。"""
    return roc_year + 1911


def to_roc_year(fiscal_year: int) -> int:
    """西元年轉民國年。"""
    return fiscal_year - 1911


@dataclass(frozen=True)
class ReportTarget:
    """一份要處理的財報目標。"""

    ticker: str
    company_name: str
    fiscal_year: int
    quarter: Quarter

    @property
    def roc_year(self) -> int:
        """民國年度。"""
        return to_roc_year(self.fiscal_year)

    @property
    def report_period(self) -> ReportPeriod:
        """財報期間別。"""
        return period_for_quarter(self.quarter)

    @property
    def doc_id(self) -> str:
        """文件識別碼。"""
        return document_id(self.ticker, self.fiscal_year, self.quarter)

    def to_meta(self, source_uri: str) -> FinancialReportMeta:
        """建立對應的中介資料。"""
        return FinancialReportMeta(
            ticker=self.ticker,
            company_name=self.company_name,
            fiscal_year=self.fiscal_year,
            quarter=self.quarter,
            report_period=self.report_period,
            source_uri=source_uri,
        )
