"""全國法規資料庫法規條文結構化解析。"""

from __future__ import annotations

import re
from datetime import date

from bs4 import BeautifulSoup, Tag

from finrag.ingest.normalize import normalize_text
from finrag.ingest.regulations import law_url
from finrag.models.document import (
    ParsedRegulation,
    RegulationArticle,
    RegulationClause,
    RegulationMeta,
)
from finrag.models.domain import LawSpec

_PARSE_ENGINE = "moj_law_html"
_CONTENT_CLASS = "law-reg-content"
_CHAPTER_CLASS = "char-2"
_SECTION_CLASS = "char-3"
_CLAUSE_LEVELS: dict[str, int] = {"line-0000": 0, "line-0004": 1, "line-0006": 2}
_DATE_RE = re.compile(r"民國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日")
_ROC_OFFSET = 1911


def _text(value: str) -> str:
    return normalize_text(" ".join(value.split()))


def _classes(tag: Tag) -> list[str]:
    value = tag.get("class")
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    return list(value)


def parse_revision_date(html: str) -> date:
    """由法規頁面解析「修正日期」，回傳西元日期。"""
    soup = BeautifulSoup(html, "lxml")
    for header in soup.find_all("th"):
        if "修正日期" not in header.get_text():
            continue
        cell = header.find_next("td")
        if cell is None:
            break
        match = _DATE_RE.search(cell.get_text())
        if match is None:
            break
        year, month, day = (int(group) for group in match.groups())
        return date(year + _ROC_OFFSET, month, day)
    raise ValueError("找不到修正日期")


def _article_no(row: Tag) -> str | None:
    anchor = row.select_one("div.col-no a")
    if anchor is None:
        return None
    return _text(anchor.get_text()) or None


def _clauses(article: Tag) -> list[RegulationClause]:
    clauses: list[RegulationClause] = []
    for line in article.find_all("div", recursive=False):
        classes = _classes(line)
        for name, level in _CLAUSE_LEVELS.items():
            if name not in classes:
                continue
            text = _text(line.get_text())
            if text:
                clauses.append(RegulationClause(level=level, text=text))
            break
    return clauses


def parse_regulation(html: str, spec: LawSpec) -> ParsedRegulation:
    """將法規 HTML 解析為條／項／款結構。"""
    soup = BeautifulSoup(html, "lxml")
    container = soup.find("div", class_=_CONTENT_CLASS) or soup

    articles: list[RegulationArticle] = []
    chapter: str | None = None
    section: str | None = None
    for element in container.find_all(recursive=False):
        classes = _classes(element)
        if _CHAPTER_CLASS in classes:
            chapter = _text(element.get_text()) or None
            section = None
            continue
        if _SECTION_CLASS in classes:
            section = _text(element.get_text()) or None
            continue
        if element.name != "div" or "row" not in classes:
            continue
        article_no = _article_no(element)
        if article_no is None:
            continue
        body = element.select_one("div.law-article")
        clauses = _clauses(body) if body is not None else []
        articles.append(
            RegulationArticle(
                article_no=article_no,
                chapter=chapter,
                section=section,
                clauses=clauses,
            )
        )

    if not articles:
        raise ValueError(f"找不到法規條文：{spec.pcode}")

    meta = RegulationMeta(
        reg_name=spec.name,
        reg_pcode=spec.pcode,
        revision_date=parse_revision_date(html),
        source_uri=law_url(spec.pcode),
        parse_engine=_PARSE_ENGINE,
    )
    return ParsedRegulation(meta=meta, articles=articles)
