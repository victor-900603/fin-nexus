"""法規條文解析測試。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from finrag.ingest.law_parser import (
    parse_regulation,
    parse_revision_date,
)
from finrag.ingest.regulations import law_url
from finrag.models.document import ParsedRegulation
from finrag.models.domain import SECURITIES_LAW
from finrag.storage.repository import write_json

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def law_html() -> str:
    return (FIXTURES / "law_sample.html").read_text(encoding="utf-8")


def test_parse_revision_date(law_html: str) -> None:
    assert parse_revision_date(law_html) == date(2024, 8, 7)


def test_parse_revision_date_missing() -> None:
    with pytest.raises(ValueError):
        parse_revision_date("<html><body><p>無</p></body></html>")


def test_parse_regulation_meta(law_html: str) -> None:
    document = parse_regulation(law_html, SECURITIES_LAW)

    assert document.meta.reg_name == "證券交易法"
    assert document.meta.reg_pcode == "G0400001"
    assert document.meta.revision_date == date(2024, 8, 7)
    assert document.meta.source_uri == law_url("G0400001")
    assert document.meta.parse_engine == "moj_law_html"
    assert document.meta.doc_id == "law_G0400001"


def test_articles_in_document_order(law_html: str) -> None:
    document = parse_regulation(law_html, SECURITIES_LAW)

    assert [article.article_no for article in document.articles] == [
        "第 1 條",
        "第 4 條",
        "第 9 條",
        "第 14-1 條",
    ]


def test_chapter_and_section_attribution(law_html: str) -> None:
    document = parse_regulation(law_html, SECURITIES_LAW)
    first, *_, last = document.articles

    assert first.chapter == "第一章總則"
    assert first.section is None
    assert last.chapter == "第一章總則"
    assert last.section == "第一節有價證券之募集、發行及買賣"


def test_clause_levels(law_html: str) -> None:
    document = parse_regulation(law_html, SECURITIES_LAW)
    article = next(item for item in document.articles if item.article_no == "第 14-1 條")

    assert [clause.level for clause in article.clauses] == [0, 1, 1, 2, 2]
    assert article.clauses[0].text == "公開發行公司應建立財務、業務之內部控制制度。"
    assert article.clauses[1].text == "一、內部控制制度之設計。"
    assert article.clauses[3].text == "（一）營運之效果及效率。"


def test_deleted_article_keeps_marker(law_html: str) -> None:
    document = parse_regulation(law_html, SECURITIES_LAW)
    article = next(item for item in document.articles if item.article_no == "第 9 條")

    assert [clause.text for clause in article.clauses] == ["（刪除）"]


def test_missing_articles_raises() -> None:
    with pytest.raises(ValueError):
        parse_regulation("<html><body><p>無條文</p></body></html>", SECURITIES_LAW)


def test_write_structured_roundtrip(law_html: str, tmp_path: Path) -> None:
    document = parse_regulation(law_html, SECURITIES_LAW)

    path = write_json(document, tmp_path / "law_G0400001.json", indent=2)

    assert path == tmp_path / "law_G0400001.json"
    loaded = ParsedRegulation.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded == document
