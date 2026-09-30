"""chunker 單元測試。"""

from __future__ import annotations

import json
from pathlib import Path

from finrag.ingest.chunker import ChunkConfig, _table_header, chunk_document
from finrag.models import (
    Block,
    BlockType,
    FinancialReportMeta,
    ParsedDocument,
    ReportPeriod,
)
from finrag.storage.repository import write_jsonl

DOC_ID = "2330_2024_Q1_consolidated"


def _counter(text: str) -> int:
    return len(text)


def _meta() -> FinancialReportMeta:
    return FinancialReportMeta(
        ticker="2330",
        company_name="台積電",
        fiscal_year=2024,
        quarter="Q1",
        report_period=ReportPeriod.QUARTERLY,
        source_uri="https://example.com/report.pdf",
    )


def _document(blocks: list[Block]) -> ParsedDocument:
    return ParsedDocument(meta=_meta(), blocks=blocks)


def _simple_document() -> ParsedDocument:
    return _document(
        [
            Block(type=BlockType.HEADING, text="資產負債表", level=1),
            Block(type=BlockType.TEXT, text="現金。"),
        ]
    )


def test_creates_parent_and_children() -> None:
    blocks = [
        Block(type=BlockType.HEADING, text="資產負債表", level=2),
        Block(type=BlockType.TEXT, text="現金。應收帳款。", page=3),
    ]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=6, parent_max=100)
    )

    parents = [chunk for chunk in chunks if chunk.metadata.is_parent]
    children = [chunk for chunk in chunks if not chunk.metadata.is_parent]
    assert len(parents) == 1
    assert parents[0].metadata.chunk_id == f"{DOC_ID}:p0"
    assert parents[0].metadata.parent_id is None
    assert parents[0].metadata.section == "資產負債表"
    assert children
    assert all(child.metadata.parent_id == f"{DOC_ID}:p0" for child in children)


def test_child_ids_are_sequential() -> None:
    blocks = [
        Block(type=BlockType.HEADING, text="資產負債表", level=2),
        Block(type=BlockType.TEXT, text="現金。應收帳款。"),
    ]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=12, parent_max=100)
    )
    children = [chunk for chunk in chunks if not chunk.metadata.is_parent]
    assert [child.metadata.chunk_id for child in children] == [
        f"{DOC_ID}:p0:c0",
        f"{DOC_ID}:p0:c1",
    ]


def test_section_path_from_nested_headings() -> None:
    blocks = [
        Block(type=BlockType.HEADING, text="財務報表", level=1),
        Block(type=BlockType.HEADING, text="資產負債表", level=2),
        Block(type=BlockType.TEXT, text="現金。"),
    ]
    chunks = chunk_document(_document(blocks), counter=_counter)
    parent = next(chunk for chunk in chunks if chunk.metadata.is_parent)
    assert parent.metadata.section == "財務報表 > 資產負債表"
    assert parent.text.startswith("財務報表 > 資產負債表\n")


def test_multiple_sections_produce_multiple_parents() -> None:
    blocks = [
        Block(type=BlockType.HEADING, text="甲", level=1),
        Block(type=BlockType.TEXT, text="第一項。"),
        Block(type=BlockType.HEADING, text="乙", level=1),
        Block(type=BlockType.TEXT, text="第二項。"),
    ]
    chunks = chunk_document(_document(blocks), counter=_counter)
    parents = [chunk for chunk in chunks if chunk.metadata.is_parent]
    assert [parent.metadata.chunk_id for parent in parents] == [
        f"{DOC_ID}:p0",
        f"{DOC_ID}:p1",
    ]
    assert parents[0].metadata.section == "甲"
    assert parents[1].metadata.section == "乙"


def test_content_before_heading_has_empty_section() -> None:
    blocks = [Block(type=BlockType.TEXT, text="前言。")]
    chunks = chunk_document(_document(blocks), counter=_counter)
    parent = next(chunk for chunk in chunks if chunk.metadata.is_parent)
    assert parent.metadata.section == ""
    assert parent.text == "前言。"


def test_headings_only_produce_no_chunks() -> None:
    blocks = [Block(type=BlockType.HEADING, text="空章節", level=1)]
    assert chunk_document(_document(blocks), counter=_counter) == []


def test_packs_multiple_parents_when_exceeding_max() -> None:
    blocks = [Block(type=BlockType.TEXT, text="一。二。三。四。")]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=100, parent_max=4)
    )
    parents = [chunk for chunk in chunks if chunk.metadata.is_parent]
    assert [parent.metadata.chunk_id for parent in parents] == [
        f"{DOC_ID}:p0",
        f"{DOC_ID}:p1",
    ]


def test_table_split_repeats_header() -> None:
    table = (
        "| 項目 | 金額 |\n"
        "| --- | --- |\n"
        "| 現金 | 10 |\n"
        "| 應收帳款 | 20 |\n"
        "| 存貨 | 30 |"
    )
    blocks = [
        Block(type=BlockType.HEADING, text="資產負債表", level=1),
        Block(type=BlockType.TABLE, text=table, page=5),
    ]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=45, parent_max=1000)
    )
    tables = [chunk for chunk in chunks if not chunk.metadata.is_parent]
    assert len(tables) == 3
    assert all(chunk.metadata.is_table for chunk in tables)
    assert all("| 項目 |" in chunk.text for chunk in tables)
    assert all(chunk.metadata.page == 5 for chunk in tables)


def test_table_with_oversized_row_is_split() -> None:
    table = "| a | b |\n| --- | --- |\n| " + "x" * 40 + " | 1 |"
    blocks = [Block(type=BlockType.TABLE, text=table, page=2)]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=20, parent_max=1000)
    )
    children = [chunk for chunk in chunks if not chunk.metadata.is_parent]
    assert children
    assert all(len(child.text) <= 20 for child in children)


def test_long_sentence_is_hard_split() -> None:
    blocks = [Block(type=BlockType.TEXT, text="一二三四五六七八九十")]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=4, parent_max=100)
    )
    children = [chunk for chunk in chunks if not chunk.metadata.is_parent]
    assert [child.text for child in children] == ["一二三四", "五六七八", "九十"]


def test_metadata_copied_from_report() -> None:
    chunks = chunk_document(_simple_document(), counter=_counter)
    metadata = chunks[0].metadata
    assert metadata.doc_id == DOC_ID
    assert metadata.ticker == "2330"
    assert metadata.company_name == "台積電"
    assert metadata.fiscal_year == 2024
    assert metadata.quarter == "Q1"
    assert metadata.report_period is ReportPeriod.QUARTERLY


def test_write_chunks_jsonl(tmp_path: Path) -> None:
    chunks = chunk_document(_simple_document(), counter=_counter)
    path = tmp_path / "chunks" / f"{DOC_ID}.jsonl"

    write_jsonl(chunks, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(chunks)
    assert json.loads(lines[0])["metadata"]["doc_id"] == DOC_ID


def test_adjacent_tables_are_separated() -> None:
    first = "| a |\n| --- |\n| 1 |"
    second = "| b |\n| --- |\n| 2 |"
    blocks = [
        Block(type=BlockType.TABLE, text=first, page=1),
        Block(type=BlockType.TABLE, text=second, page=2),
    ]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=200, parent_max=400)
    )
    child = next(chunk for chunk in chunks if not chunk.metadata.is_parent)
    assert "\n\n| b |" in child.text


def test_oversized_row_split_keeps_header() -> None:
    cells = " | ".join(str(number) for number in range(1, 21))
    table = f"| h |\n| --- |\n| {cells} |"
    blocks = [Block(type=BlockType.TABLE, text=table, page=1)]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=30, parent_max=1000)
    )
    children = [chunk for chunk in chunks if not chunk.metadata.is_parent]
    assert len(children) > 1
    assert all(len(child.text) <= 30 for child in children)


def test_wide_row_split_when_header_pushes_over_budget() -> None:
    table = "| h1 | h2 |\n| --- | --- |\n| aaaaa | bbbb |"
    blocks = [Block(type=BlockType.TABLE, text=table, page=1)]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=40, parent_max=1000)
    )
    children = [chunk for chunk in chunks if not chunk.metadata.is_parent]
    assert len(children) > 1
    assert all(len(child.text) <= 40 for child in children)
    assert all("| h1 | h2 |" in child.text for child in children)


def test_table_title_from_caption() -> None:
    table = "**表一**\n\n| 項目 | 金額 |\n| --- | --- |\n| 現金 | 10 |"
    blocks = [
        Block(type=BlockType.HEADING, text="資產負債表", level=1),
        Block(type=BlockType.TABLE, text=table, page=5),
    ]
    chunks = chunk_document(_document(blocks), counter=_counter)

    child = next(chunk for chunk in chunks if not chunk.metadata.is_parent)
    assert child.metadata.table_title == "表一"


def test_table_title_falls_back_to_section() -> None:
    table = "| 項目 | 金額 |\n| --- | --- |\n| 現金 | 10 |"
    blocks = [
        Block(type=BlockType.HEADING, text="資產負債表", level=1),
        Block(type=BlockType.TABLE, text=table, page=5),
    ]
    chunks = chunk_document(_document(blocks), counter=_counter)

    child = next(chunk for chunk in chunks if not chunk.metadata.is_parent)
    assert child.metadata.table_title == "資產負債表"


def test_text_chunk_has_no_table_title() -> None:
    chunks = chunk_document(_simple_document(), counter=_counter)

    assert chunks
    assert all(chunk.metadata.table_title is None for chunk in chunks)


def test_table_fragments_keep_title_line() -> None:
    caption = "**表一**"
    rows = "".join(f"| 項目{i} | {i} |\n" for i in range(10))
    table = f"{caption}\n\n| 項目 | 金額 |\n| --- | --- |\n{rows}"
    blocks = [Block(type=BlockType.TABLE, text=table, page=1)]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=40, parent_max=1000)
    )
    children = [chunk for chunk in chunks if not chunk.metadata.is_parent]

    assert len(children) > 1
    assert all(caption in child.text for child in children)
    assert all(child.metadata.table_title == "表一" for child in children)


def test_multi_row_table_header_is_preserved() -> None:
    header = "| 項目 | 113 年 | 112 年 |\n| 名稱 | 金額 | 金額 |\n| --- | --- | --- |"
    rows = "".join(f"| 項目{i} | {i} | {i} |\n" for i in range(8))
    table = f"{header}\n{rows}"
    blocks = [Block(type=BlockType.TABLE, text=table, page=1)]
    chunks = chunk_document(
        _document(blocks), counter=_counter, config=ChunkConfig(child_max=80, parent_max=1000)
    )
    children = [chunk for chunk in chunks if not chunk.metadata.is_parent]

    assert len(children) > 1
    assert all("| 名稱 | 金額 | 金額 |" in child.text for child in children)
    assert all("| --- | --- | --- |" in child.text for child in children)


def test_dash_data_row_is_not_delimiter() -> None:
    lines = ["| 項目 | 金額 |", "| --- | --- |", "| - | - |", "| 1 | 2 |"]

    assert _table_header(lines) == ["| 項目 | 金額 |", "| --- | --- |"]


def test_table_header_ignores_blank_lines() -> None:
    lines = ["**表一**", "", "| 項目 | 金額 |", "| --- | --- |", "| 1 | 2 |"]

    assert _table_header(lines) == ["**表一**", "| 項目 | 金額 |", "| --- | --- |"]


def test_table_header_without_delimiter_falls_back_to_two_lines() -> None:
    assert _table_header(["| a |", "| b |", "| c |"]) == ["| a |", "| b |"]


def test_split_table_fragment_keeps_notes_and_drops_empty_pieces() -> None:
    from finrag.ingest.chunker import _split_table_fragment

    fragment = "| h |\n| --- |\n| " + "x" * 30 + " |\n\n<u>註</u>"
    pieces = _split_table_fragment(fragment, _counter, 20)

    assert "<u>註</u>" in pieces
    assert all(piece.strip() for piece in pieces)
    assert all(piece != "| h |\n| --- |" for piece in pieces)


_OUTLINE = (
    "| 項目 | 頁次 |\n"
    "| --- | --- |\n"
    "| 三、會計師核閱報告 | 3 |\n"
    "| 四、合併資產負債表 | 4 |\n"
    "| 五、合併綜合損益表 | 5 |\n"
)


def test_outline_fills_section_for_pages_without_headings() -> None:
    blocks = [
        Block(type=BlockType.TABLE, text=_OUTLINE, page=1),
        Block(type=BlockType.TABLE, text="| 1100 | 100 |", page=3),
        Block(type=BlockType.TABLE, text="| 4000 | 508 |", page=4),
    ]
    chunks = chunk_document(_document(blocks), counter=_counter)
    sections = [chunk.metadata.section for chunk in chunks if chunk.metadata.is_parent]

    assert [section for section in sections if section] == [
        "四、合併資產負債表",
        "五、合併綜合損益表",
    ]


def test_outline_boundary_resets_heading_stack() -> None:
    blocks = [
        Block(type=BlockType.TABLE, text=_OUTLINE, page=1),
        Block(type=BlockType.HEADING, text="結論", level=2, page=2),
        Block(type=BlockType.TEXT, text="核閱結論。", page=2),
        Block(type=BlockType.TABLE, text="| 1100 | 100 |", page=3),
    ]
    chunks = chunk_document(_document(blocks), counter=_counter)
    sections = [chunk.metadata.section for chunk in chunks if chunk.metadata.is_parent]

    assert [section for section in sections if section] == ["結論", "四、合併資產負債表"]


def test_outline_fills_gap_pages_from_previous_entry() -> None:
    outline = (
        "| 項目 | 頁次 |\n"
        "| --- | --- |\n"
        "| 六、合併權益變動表 | 8 |\n"
        "| 七、合併現金流量表 | 10 |\n"
    )
    blocks = [
        Block(type=BlockType.TABLE, text=outline, page=1),
        Block(type=BlockType.TABLE, text="| 附表一 | 100 |", page=8),
        Block(type=BlockType.TABLE, text="| 附表二 | 200 |", page=9),
    ]
    chunks = chunk_document(_document(blocks), counter=_counter)
    sections = [chunk.metadata.section for chunk in chunks if chunk.metadata.is_parent]

    assert [section for section in sections if section] == [
        "六、合併權益變動表",
        "七、合併現金流量表",
    ]


def test_outline_fills_pages_after_last_entry() -> None:
    outline = (
        "| 項目 | 頁次 |\n"
        "| --- | --- |\n"
        "| 二、目錄 | 1 |\n"
        "| 七、合併現金流量表 | 8 |\n"
    )
    blocks = [
        Block(type=BlockType.TABLE, text=outline, page=1),
        Block(type=BlockType.TABLE, text="| 附表一 | 100 |", page=8),
        Block(type=BlockType.TABLE, text="| 附表二 | 200 |", page=10),
    ]
    chunks = chunk_document(_document(blocks), counter=_counter)
    sections = [chunk.metadata.section for chunk in chunks if chunk.metadata.is_parent]

    assert sections == ["二、目錄", "七、合併現金流量表"]
