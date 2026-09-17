"""文件模型測試。"""

from typing import Any

import pytest
from pydantic import ValidationError

from finrag.models import (
    FinancialReportMeta,
    ReportPeriod,
    ReportScope,
    SourceType,
)
from finrag.models.document import document_id, period_for_quarter


def _meta(**overrides: Any) -> FinancialReportMeta:
    data: dict[str, Any] = {
        "ticker": "2330",
        "company_name": "台積電",
        "fiscal_year": 2024,
        "quarter": "Q1",
        "report_period": ReportPeriod.QUARTERLY,
        "source_uri": "https://example.com/2330_2024_Q1.pdf",
    }
    data.update(overrides)
    return FinancialReportMeta(**data)


def test_enum_values() -> None:
    assert SourceType.MOPS == "mops"
    assert SourceType.MOJ_LAW == "moj_law"
    assert ReportPeriod.QUARTERLY == "quarterly"
    assert ReportPeriod.ANNUAL == "annual"
    assert ReportScope.CONSOLIDATED == "consolidated"


def test_full_construction() -> None:
    meta = _meta()
    assert meta.ticker == "2330"
    assert meta.company_name == "台積電"
    assert meta.fiscal_year == 2024
    assert meta.quarter == "Q1"
    assert meta.source_uri == "https://example.com/2330_2024_Q1.pdf"


def test_defaults() -> None:
    meta = _meta()
    assert meta.report_scope is ReportScope.CONSOLIDATED
    assert meta.source is SourceType.MOPS
    assert meta.parse_engine is None


def test_doc_id() -> None:
    assert _meta().doc_id == "2330_2024_Q1_consolidated"


def test_doc_id_annual() -> None:
    meta = _meta(quarter="Q4", report_period=ReportPeriod.ANNUAL)
    assert meta.doc_id == "2330_2024_Q4_consolidated"


def test_period_for_quarter() -> None:
    assert period_for_quarter("Q1") is ReportPeriod.QUARTERLY
    assert period_for_quarter("Q2") is ReportPeriod.QUARTERLY
    assert period_for_quarter("Q4") is ReportPeriod.ANNUAL


def test_document_id_helper() -> None:
    assert document_id("2330", 2024, "Q1") == "2330_2024_Q1_consolidated"
    assert document_id("2330", 2024, "Q4", ReportScope.CONSOLIDATED) == (
        "2330_2024_Q4_consolidated"
    )


@pytest.mark.parametrize("quarter", ["Q5", "Q0", "2"])
def test_invalid_quarter(quarter: str) -> None:
    with pytest.raises(ValidationError):
        _meta(quarter=quarter)


def test_q4_requires_annual() -> None:
    with pytest.raises(ValidationError):
        _meta(quarter="Q4", report_period=ReportPeriod.QUARTERLY)


def test_q1_rejects_annual() -> None:
    with pytest.raises(ValidationError):
        _meta(quarter="Q1", report_period=ReportPeriod.ANNUAL)


def test_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        FinancialReportMeta.model_validate({"ticker": "2330"})
