"""財報目錄對齊：由目錄取出章節名稱與頁次，並判斷標題是否可用。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from fin_nexus.models import Block, BlockType

_OUTLINE_COLUMNS = ("項目", "頁次")
_PAGE_SEPARATOR = re.compile(r"[、,，]")
_NUMBER = re.compile(r"\d+")
_PREFIX = (
    r"(?:[一二三四五六七八九十百]+、"
    r"|[（(]\s*[一二三四五六七八九十百]+\s*[）)]"
    r"|\d+\.\d+(?:\.\d+)*|\d+[.、]"
    r"|[A-Za-z][.、])"
)
_PAGES = r"\d+(?:[~-]\d+)?(?:[、,，]\s*\d+(?:[~-]\d+)?)*"
_ENTRY = re.compile(rf"({_PREFIX}[^\d（(]*?)\s*({_PAGES})(?=\s|$)")
_MIN_TEXT_ENTRIES = 5
_PAGE_CELL = re.compile(r"^[\d\s、,，~\-]*\d[\d\s、,，~\-]*$")
_NORMALIZE = re.compile(r"[\s\u3000()（）\[\]【】§、,，.。\-—_～~]")
_WHITESPACE = re.compile(r"[\s\u3000]+")
_HEADING = re.compile(rf"^{_PREFIX}")


@dataclass(frozen=True)
class OutlineEntry:
    """目錄中的一個章節項目，`pages` 為 0 基頁碼。"""

    name: str
    pages: tuple[int, ...]


def normalize_heading(text: str) -> str:
    """去除空白與標號符號，供標題比對使用。"""
    return _NORMALIZE.sub("", text)


def is_heading(text: str) -> bool:
    """判斷標題是否帶章節編號前綴；未帶編號者視為正文。"""
    return bool(_HEADING.match(_WHITESPACE.sub("", text)))


def _markdown_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
    return rows


def _entry_pages(text: str) -> tuple[int, ...]:
    """解析目錄頁次欄，回傳 0 基頁碼（支援 `5~7` 與 `56~57、58~91`）。"""
    pages: list[int] = []
    for segment in _PAGE_SEPARATOR.split(text.strip()):
        numbers = [int(value) for value in _NUMBER.findall(segment)]
        if not numbers:
            continue
        start = numbers[0] - 1
        end = numbers[-1] - 1
        pages.extend(range(start, max(start, end) + 1))
    return tuple(pages)


def _header_matches(header: Sequence[str]) -> bool:
    """表頭欄名可能被拆散（如「項」「目頁」「次」），以併接後比對。"""
    joined = normalize_heading("".join(header))
    return all(column in joined for column in _OUTLINE_COLUMNS)


def _page_cell_pages(cell: str) -> tuple[int, ...]:
    """僅由數字與分隔符組成的儲存格才視為頁次欄。"""
    if not _PAGE_CELL.match(cell):
        return ()
    return _entry_pages(cell)


def _row_pages(row: Sequence[str]) -> tuple[int, ...]:
    """頁次欄位置因表頭切分方式而異，取第一個像頁次的儲存格。"""
    for cell in row[1:]:
        pages = _page_cell_pages(cell)
        if pages:
            return pages
    return ()


def _table_outline(blocks: Sequence[Block]) -> list[OutlineEntry]:
    for block in blocks:
        if block.type is not BlockType.TABLE:
            continue
        rows = _markdown_rows(block.text)
        if len(rows) < 2 or not _header_matches(rows[0]):
            continue
        entries: list[OutlineEntry] = []
        for row in rows[1:]:
            if len(row) < 2 or not row[0]:
                continue
            pages = _row_pages(row)
            if pages:
                entries.append(OutlineEntry(name=row[0], pages=pages))
        if entries:
            return entries
    return []


def _text_outline(blocks: Sequence[Block]) -> list[OutlineEntry]:
    """目錄被輸出為單一文字區塊時，直接由文字取出章節與頁次。"""
    for block in blocks:
        if block.type is BlockType.TABLE:
            continue
        entries = [
            OutlineEntry(name=name.strip(), pages=pages)
            for name, page in _ENTRY.findall(block.text)
            if (pages := _entry_pages(page))
        ]
        if len(entries) >= _MIN_TEXT_ENTRIES:
            return entries
    return []


def extract_outline(blocks: Sequence[Block]) -> list[OutlineEntry]:
    """由目錄取出章節名稱與頁次對照，表格與文字兩種輸出皆支援。"""
    return _table_outline(blocks) or _text_outline(blocks)


def page_sections(outline: Sequence[OutlineEntry], last_page: int | None = None) -> dict[int, str]:
    """建立 0 基頁碼至章節名稱的對照，後出現的項目優先。

    目錄未列出的頁面沿用前一個項目；`last_page` 之後的頁面同樣沿用最後一個項目，
    使附錄等未編入目錄的頁面仍有章節名稱。
    """
    mapping: dict[int, str] = {}
    name: str | None = None
    covered = 0
    for entry in outline:
        for page in sorted(entry.pages):
            if page < 0:
                continue
            for gap in range(covered, page):
                if name is not None:
                    mapping[gap] = name
            mapping[page] = entry.name
            name = entry.name
            covered = max(covered, page + 1)
    if name is not None and last_page is not None:
        for page in range(covered, last_page + 1):
            mapping[page] = name
    return mapping
