"""查詢擴展與實體提取測試。"""

from __future__ import annotations

from finrag.models.document import ReportPeriod
from finrag.retrieval.expander import analyze
from finrag.retrieval.ontology import Ontology

_ONTO = Ontology(
    tickers={"2330": ("台積電", "TSMC")},
    synonyms={"營業收入淨額": ("營收",)},
    terms={"CAPEX": "資本支出"},
)


def test_extracts_ticker_alias_and_period() -> None:
    analysis = analyze("台積電2024年Q1營收", _ONTO)

    assert analysis.tickers == ["2330"]
    assert analysis.fiscal_years == [2024]
    assert analysis.quarters == ["Q1"]
    assert analysis.report_period is ReportPeriod.QUARTERLY
    assert "營業收入淨額" in analysis.expanded


def test_extracts_ticker_code_and_abbreviation() -> None:
    analysis = analyze("2330 的 CAPEX 計畫", _ONTO)

    assert analysis.tickers == ["2330"]
    assert "資本支出" in analysis.expanded
    assert "CAPEX" in analysis.matches


def test_roc_year_and_chinese_quarter() -> None:
    analysis = analyze("台積電民國113年第2季", _ONTO)

    assert analysis.fiscal_years == [2024]
    assert analysis.quarters == ["Q2"]
    assert analysis.report_period is ReportPeriod.QUARTERLY


def test_q4_is_annual() -> None:
    analysis = analyze("台積電2024 Q4", _ONTO)

    assert analysis.report_period is ReportPeriod.ANNUAL


def test_annual_keyword() -> None:
    analysis = analyze("台積電2023年度", _ONTO)

    assert analysis.fiscal_years == [2023]
    assert analysis.quarters == []
    assert analysis.report_period is ReportPeriod.ANNUAL


def test_matches_regulation() -> None:
    analysis = analyze("證券交易法第14條", _ONTO)

    assert analysis.reg_pcodes == ["G0400001"]


def test_no_match_returns_query() -> None:
    analysis = analyze("今天天氣如何", _ONTO)

    assert analysis.tickers == []
    assert analysis.fiscal_years == []
    assert analysis.report_period is None
    assert analysis.reg_pcodes == []
    assert analysis.expanded == "今天天氣如何"
    assert analysis.sparse_query == "今天天氣如何"


def test_sparse_query_uses_subject_terms() -> None:
    analysis = analyze("台積電2024年Q1營收", _ONTO)

    assert analysis.sparse_query == "營業收入淨額 營收"


def test_sparse_query_keeps_canonical_with_synonyms() -> None:
    analysis = analyze("台積電的營業收入淨額", _ONTO)

    assert analysis.matches == ["營業收入淨額"]
    assert analysis.sparse_query == "營業收入淨額 營收"


def test_synonym_match_is_recorded() -> None:
    analysis = analyze("台積電營收表現", _ONTO)

    assert "營收" in analysis.matches
    assert "營業收入淨額" in analysis.expanded
