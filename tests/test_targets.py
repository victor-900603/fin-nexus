"""財報目標清單測試。"""

from __future__ import annotations

import pytest

from finrag.ingest.targets import build_targets, default_targets
from finrag.models import ReportPeriod
from finrag.models.domain import COMPANY_NAMES, ReportTarget


def test_default_targets_cover_all_companies() -> None:
    targets = default_targets()

    assert len(targets) == 36
    assert {target.ticker for target in targets} == set(COMPANY_NAMES)
    assert {target.fiscal_year for target in targets} == {2023, 2024, 2025}
    assert {target.roc_year for target in targets} == {112, 113, 114}
    assert {target.quarter for target in targets} == {"Q1", "Q2", "Q3", "Q4"}
    assert all(target.company_name == COMPANY_NAMES[target.ticker] for target in targets)


def test_roc_year() -> None:
    target = ReportTarget("2330", "台積電", 2023, "Q1")

    assert target.fiscal_year == 2023
    assert target.roc_year == 112


def test_report_period() -> None:
    assert ReportTarget("2330", "台積電", 2025, "Q4").report_period is ReportPeriod.ANNUAL
    assert ReportTarget("2330", "台積電", 2025, "Q1").report_period is ReportPeriod.QUARTERLY


def test_doc_id() -> None:
    target = ReportTarget("2330", "台積電", 2023, "Q1")

    assert target.doc_id == "2330_2023_Q1_consolidated"


def test_to_meta() -> None:
    target = ReportTarget("2330", "台積電", 2023, "Q1")

    meta = target.to_meta("https://example.com/list")

    assert meta.doc_id == target.doc_id
    assert meta.company_name == "台積電"
    assert meta.fiscal_year == 2023
    assert meta.report_period is ReportPeriod.QUARTERLY
    assert meta.source_uri == "https://example.com/list"


def test_build_targets_custom() -> None:
    targets = build_targets(["2303"], [2024], ["Q1", "Q2"])

    assert [(target.ticker, target.fiscal_year, target.quarter) for target in targets] == [
        ("2303", 2024, "Q1"),
        ("2303", 2024, "Q2"),
    ]
    assert all(target.company_name == "聯電" for target in targets)


def test_build_targets_unknown_ticker_uses_code() -> None:
    (target,) = build_targets(["1234"], [2024], ["Q1"])

    assert target.company_name == "1234"


def test_build_targets_rejects_bad_ticker() -> None:
    with pytest.raises(ValueError):
        build_targets(["233"], [2024], ["Q1"])


def test_build_targets_rejects_bad_quarter() -> None:
    with pytest.raises(ValueError):
        build_targets(["2330"], [2024], ["Q5"])


def test_company_names_known() -> None:
    assert COMPANY_NAMES["2330"] == "台積電"
