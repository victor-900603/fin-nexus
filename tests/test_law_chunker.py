"""法規切片測試。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from finrag.ingest.chunker import ChunkConfig
from finrag.ingest.law_chunker import chunk_regulation
from finrag.models import (
    ParsedRegulation,
    RegulationArticle,
    RegulationChunk,
    RegulationClause,
    RegulationMeta,
)
from finrag.storage.repository import write_jsonl


def _meta() -> RegulationMeta:
    return RegulationMeta(
        reg_name="證券交易法",
        reg_pcode="G0400001",
        revision_date=date(2024, 8, 7),
        source_uri="https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=G0400001",
    )


def _article(
    article_no: str,
    chapter: str | None = "第一章總則",
    section: str | None = None,
    clauses: tuple[str, ...] = ("本條內容。",),
) -> RegulationArticle:
    return RegulationArticle(
        article_no=article_no,
        chapter=chapter,
        section=section,
        clauses=[RegulationClause(level=0, text=text) for text in clauses],
    )


def _document(articles: list[RegulationArticle]) -> ParsedRegulation:
    return ParsedRegulation(meta=_meta(), articles=articles)


def _parents(chunks: list[RegulationChunk]) -> list[RegulationChunk]:
    return [chunk for chunk in chunks if chunk.metadata.is_parent]


def _children(chunks: list[RegulationChunk]) -> list[RegulationChunk]:
    return [chunk for chunk in chunks if not chunk.metadata.is_parent]


def test_each_article_has_its_own_parent() -> None:
    chunks = chunk_regulation(_document([_article("第 1 條"), _article("第 2 條")]), counter=len)
    children = _children(chunks)
    parents = _parents(chunks)
    assert [child.metadata.article_no for child in children] == ["第 1 條", "第 2 條"]
    assert [parent.metadata.article_no for parent in parents] == ["第 1 條", "第 2 條"]
    assert [child.metadata.parent_id for child in children] == [
        parent.metadata.chunk_id for parent in parents
    ]


def test_chunk_ids_are_sequential() -> None:
    chunks = chunk_regulation(_document([_article("第 1 條")]), counter=len)
    parents = _parents(chunks)
    children = _children(chunks)
    assert parents[0].metadata.chunk_id == "law_G0400001:p0"
    assert children[0].metadata.chunk_id == "law_G0400001:p0:c0"


def test_child_prefix_includes_section() -> None:
    article = _article(
        "第 22 條",
        chapter="第一章總則",
        section="第一節有價證券之募集、發行及買賣",
    )
    child = _children(chunk_regulation(_document([article]), counter=len))[0]
    assert child.text.startswith(
        "[證券交易法 > 第一章總則 > 第一節有價證券之募集、發行及買賣 > 第 22 條]\n"
    )


def test_child_prefix_without_section() -> None:
    child = _children(chunk_regulation(_document([_article("第 1 條")]), counter=len))[0]
    assert child.text.startswith("[證券交易法 > 第一章總則 > 第 1 條]\n")


def test_parent_prefix_includes_article() -> None:
    chunks = chunk_regulation(_document([_article("第 1 條"), _article("第 2 條")]), counter=len)
    parents = _parents(chunks)
    assert parents[0].text.startswith("[證券交易法 > 第一章總則 > 第 1 條]\n")
    assert parents[1].text.startswith("[證券交易法 > 第一章總則 > 第 2 條]\n")
    assert "第 1 條" not in parents[1].text


def test_metadata_fields() -> None:
    child = _children(chunk_regulation(_document([_article("第 1 條")]), counter=len))[0]
    assert child.metadata.doc_id == "law_G0400001"
    assert child.metadata.reg_name == "證券交易法"
    assert child.metadata.reg_pcode == "G0400001"
    assert child.metadata.revision_date == date(2024, 8, 7)
    assert child.metadata.chapter == "第一章總則"
    assert child.metadata.section == "第一章總則"
    assert child.metadata.is_parent is False
    assert child.metadata.parent_id is not None


def test_parent_metadata_has_article() -> None:
    parent = _parents(chunk_regulation(_document([_article("第 1 條")]), counter=len))[0]
    assert parent.metadata.article_no == "第 1 條"
    assert parent.metadata.is_parent is True
    assert parent.metadata.parent_id is None


def test_oversized_article_is_kept_intact() -> None:
    article = _article("第 1 條", clauses=("長" * 500,))
    chunks = chunk_regulation(_document([article]), counter=len, config=ChunkConfig(child_max=10))
    children = _children(chunks)
    assert len(children) == 1
    assert len(children[0].text) > 10


def test_each_article_is_a_separate_parent() -> None:
    articles = [_article(f"第 {index} 條") for index in range(1, 6)]
    chunks = chunk_regulation(_document(articles), counter=len)

    assert len(_parents(chunks)) == 5
    assert len(_children(chunks)) == 5


def test_clauses_become_separate_children() -> None:
    article = _article("第 1 條", clauses=("第一款。", "第二款。"))
    chunks = chunk_regulation(
        _document([article]),
        counter=lambda text: text.count("。"),
        config=ChunkConfig(child_max=1),
    )

    children = _children(chunks)
    parent = _parents(chunks)[0]
    assert len(children) == 2
    assert {child.metadata.parent_id for child in children} == {parent.metadata.chunk_id}
    assert "第一款。" in children[0].text
    assert "第二款。" in children[1].text
    assert "第一款。" in parent.text
    assert "第二款。" in parent.text


def test_multiple_sections_separate_parents() -> None:
    articles = [
        _article("第 1 條", chapter="第一章總則", section="第一節 A"),
        _article("第 2 條", chapter="第一章總則", section="第二節 B"),
    ]
    parents = _parents(chunk_regulation(_document(articles), counter=len))
    assert [parent.metadata.section for parent in parents] == [
        "第一章總則 > 第一節 A",
        "第一章總則 > 第二節 B",
    ]


def test_empty_article_is_skipped() -> None:
    chunks = chunk_regulation(_document([_article("第 9 條", clauses=())]), counter=len)
    assert chunks == []


def test_write_chunks_jsonl(tmp_path: Path) -> None:
    chunks = chunk_regulation(_document([_article("第 1 條")]), counter=len)
    path = tmp_path / "law_G0400001.jsonl"
    write_jsonl(chunks, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)
    restored = [RegulationChunk.model_validate_json(line) for line in lines]
    assert restored[0].metadata.doc_id == "law_G0400001"
