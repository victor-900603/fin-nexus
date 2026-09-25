"""查詢擴展與實體提取。"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from finrag.models.document import ReportPeriod
from finrag.models.domain import LAW_SPECS, to_western_year
from finrag.retrieval.ontology import Ontology

_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
_ROC_YEAR_RE = re.compile(r"民國\s*(\d{2,3})\s*年")
_QUARTER_RE = re.compile(r"Q\s*([1-4])", re.IGNORECASE)
_CN_QUARTER_RE = re.compile(r"第\s*([一二三四1234])\s*季")
_ANNUAL_RE = re.compile(r"年度|全年|年報")
_TICKER_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_CN_QUARTER_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "1": 1, "2": 2, "3": 3, "4": 4}


class QueryAnalysis(BaseModel):
    """查詢解析結果。"""

    query: str
    expanded: str
    sparse_query: str
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    quarters: list[str] = Field(default_factory=list)
    report_period: ReportPeriod | None = None
    reg_pcodes: list[str] = Field(default_factory=list)
    matches: list[str] = Field(default_factory=list)


def _match_tickers(text: str, ontology: Ontology) -> list[str]:
    haystack = text.lower()
    codes = set(_TICKER_RE.findall(text))
    found: list[str] = []
    for ticker, aliases in ontology.tickers.items():
        if ticker in codes or any(alias.lower() in haystack for alias in aliases):
            found.append(ticker)
    return found


def _match_years(text: str) -> list[int]:
    years = {int(year) for year in _YEAR_RE.findall(text)}
    years.update(to_western_year(int(year)) for year in _ROC_YEAR_RE.findall(text))
    return sorted(years)


def _match_quarters(text: str) -> list[str]:
    quarters = {f"Q{quarter}" for quarter in _QUARTER_RE.findall(text)}
    quarters.update(f"Q{_CN_QUARTER_DIGITS[quarter]}" for quarter in _CN_QUARTER_RE.findall(text))
    return sorted(quarters)


def _report_period(text: str, quarters: list[str]) -> ReportPeriod | None:
    if quarters:
        return ReportPeriod.ANNUAL if "Q4" in quarters else ReportPeriod.QUARTERLY
    if _ANNUAL_RE.search(text):
        return ReportPeriod.ANNUAL
    return None


def _match_regulations(text: str) -> list[str]:
    return [spec.pcode for spec in LAW_SPECS if spec.name in text]


def _expand(text: str, ontology: Ontology) -> tuple[str, str, list[str]]:
    """產生擴展查詢、稀疏查詢與命中的本體詞彙。

    擴展查詢保留原句並附加標準詞，供 Dense 使用；稀疏查詢僅取抽出的科目詞與其同義詞群組，
    供 Sparse 比對（口語名詞對應標準科目）；未命中任何科目時退回原句。
    """
    haystack = text.lower()
    extra: list[str] = []
    terms: list[str] = []
    matches: list[str] = []
    for abbrev, expansion in ontology.terms.items():
        if abbrev.lower() in haystack:
            extra.append(expansion)
            terms.extend((abbrev, expansion))
            matches.append(abbrev)
    for canonical, synonyms in ontology.synonyms.items():
        if canonical.lower() in haystack:
            extra.extend(synonyms)
            terms.extend((canonical, *synonyms))
            matches.append(canonical)
            continue
        for synonym in synonyms:
            if synonym.lower() in haystack:
                extra.append(canonical)
                terms.extend((canonical, *synonyms))
                matches.append(synonym)
                break
    expanded = text if not extra else f"{text} {' '.join(dict.fromkeys(extra))}"
    sparse_query = " ".join(dict.fromkeys(terms)) if terms else text
    return expanded, sparse_query, matches


def analyze(query: str, ontology: Ontology) -> QueryAnalysis:
    """解析查詢，提取公司、時間、法規與本體詞彙並產生擴展字串。"""
    onto = ontology
    quarters = _match_quarters(query)
    expanded, sparse_query, matches = _expand(query, onto)
    return QueryAnalysis(
        query=query,
        expanded=expanded,
        sparse_query=sparse_query,
        tickers=_match_tickers(query, onto),
        fiscal_years=_match_years(query),
        quarters=quarters,
        report_period=_report_period(query, quarters),
        reg_pcodes=_match_regulations(query),
        matches=matches,
    )
