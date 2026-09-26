"""財報文件父子切片。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from finrag.models import (
    Block,
    BlockType,
    Chunk,
    ChunkMetadata,
    FinancialReportMeta,
    ParsedDocument,
)
from finrag.text import TokenCounter, pack_by_tokens

_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])")
_TABLE_ROW = re.compile(r"^\s*\|")
_TABLE_TITLE = re.compile(r"^\*\*(.+)\*\*$")
_TABLE_DELIMITER_CELL = re.compile(r"^:?-{3,}:?$")


@dataclass(frozen=True)
class ChunkConfig:
    """切片參數。"""

    child_max: int = 250
    parent_max: int = 1200


@dataclass(frozen=True)
class _Unit:
    text: str
    is_table: bool
    page: int | None
    title: str | None = None


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_END.split(text)]
    return [part for part in parts if part]


def _hard_split(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for char in text:
        if current and counter(current + char) > max_tokens:
            pieces.append(current)
            current = char
        else:
            current += char
    if current:
        pieces.append(current)
    return pieces


def _split_by_cells(line: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    groups: list[list[str]] = []
    current: list[str] = []
    for cell in cells:
        trial = [*current, cell]
        if current and counter("| " + " | ".join(trial) + " |") > max_tokens:
            groups.append(current)
            current = [cell]
        else:
            current = trial
    if current:
        groups.append(current)
    return ["| " + " | ".join(group) + " |" for group in groups]


def _is_delimiter(line: str) -> bool:
    """判斷是否為 Markdown 表格的分隔列。"""
    stripped = line.strip()
    if not _TABLE_ROW.match(stripped):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return bool(cells) and all(_TABLE_DELIMITER_CELL.match(cell) for cell in cells)


def _table_header(lines: Sequence[str]) -> list[str]:
    """取表格的標題行（若存在）與表頭，範圍到分隔列為止。"""
    content = [line for line in lines if line.strip()]
    for index, line in enumerate(content):
        if _is_delimiter(line):
            return content[: index + 1]
    return content[:2]


def _table_title(text: str) -> str | None:
    """取表格文字開頭的標題行。"""
    lines = text.splitlines()
    if not lines:
        return None
    match = _TABLE_TITLE.match(lines[0].strip())
    return match.group(1) if match else None


def _split_table_fragment(
    fragment: str, counter: TokenCounter, max_tokens: int
) -> list[str]:
    lines = [line for line in fragment.splitlines() if line.strip()]
    header_lines = _table_header(lines)
    if len(header_lines) < 2:
        return _hard_split(fragment, counter, max_tokens)
    header = "\n".join(header_lines)
    budget = max(max_tokens - counter(header) - 1, 1)
    pieces: list[str] = []
    for line in lines[len(header_lines) :]:
        if not _TABLE_ROW.match(line):
            if counter(line) <= max_tokens:
                pieces.append(line)
            else:
                pieces.extend(_hard_split(line, counter, max_tokens))
            continue
        if counter(line) <= max_tokens:
            pieces.append(f"{header}\n{line}")
            continue
        for part in _split_by_cells(line, counter, budget):
            if not part.strip():
                continue
            piece = f"{header}\n{part}"
            if counter(piece) <= max_tokens:
                pieces.append(piece)
            else:
                pieces.extend(_hard_split(piece, counter, max_tokens))
    return pieces or _hard_split(fragment, counter, max_tokens)


def _table_fragments(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    if counter(text) <= max_tokens:
        return [text]
    lines = [line for line in text.splitlines() if line.strip()]
    header = _table_header(lines)
    if len(header) < 2 or not _is_delimiter(header[-1]):
        return _hard_split(text, counter, max_tokens)
    body = lines[len(header) :]
    pipe = [line for line in body if _TABLE_ROW.match(line)]
    notes = [line for line in body if not _TABLE_ROW.match(line)]
    fragments: list[str] = []
    current: list[str] = []
    for row in pipe:
        if current and counter("\n".join([*header, *current, row])) > max_tokens:
            fragments.append("\n".join([*header, *current]))
            current = []
        current.append(row)
    fragments.append("\n".join([*header, *current]))
    if notes:
        fragments[-1] = f"{fragments[-1]}\n\n{' '.join(notes)}"
    pieces: list[str] = []
    for fragment in fragments:
        if counter(fragment) <= max_tokens:
            pieces.append(fragment)
        else:
            pieces.extend(_split_table_fragment(fragment, counter, max_tokens))
    return pieces


def _block_units(block: Block, counter: TokenCounter, max_tokens: int) -> list[_Unit]:
    if block.type is BlockType.TABLE:
        title = _table_title(block.text)
        return [
            _Unit(text=part, is_table=True, page=block.page, title=title)
            for part in _table_fragments(block.text, counter, max_tokens)
        ]
    units: list[_Unit] = []
    for sentence in _split_sentences(block.text):
        pieces = (
            [sentence]
            if counter(sentence) <= max_tokens
            else _hard_split(sentence, counter, max_tokens)
        )
        units.extend(_Unit(text=piece, is_table=False, page=block.page) for piece in pieces)
    return units


def _pack(
    units: Sequence[_Unit], counter: TokenCounter, max_tokens: int, prefix_tokens: int
) -> list[list[_Unit]]:
    return pack_by_tokens(units, lambda unit: unit.text, counter, max_tokens, prefix_tokens)


def _sections(blocks: Sequence[Block]) -> list[tuple[list[str], list[Block]]]:
    """依標題切分章節。"""
    sections: list[tuple[list[str], list[Block]]] = []
    stack: list[tuple[int, str]] = []
    current: list[Block] = []

    def flush() -> None:
        if current:
            sections.append(([text for _, text in stack], list(current)))
            current.clear()

    for block in blocks:
        if block.type is BlockType.HEADING:
            flush()
            level = block.level if block.level is not None else len(stack) + 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text))
        else:
            current.append(block)
    flush()
    return sections


def _join_units(units: Sequence[_Unit]) -> str:
    if not units:
        return ""
    pieces = [units[0].text]
    for previous, unit in zip(units, units[1:], strict=False):
        pieces.append("\n\n" if previous.is_table or unit.is_table else "\n")
        pieces.append(unit.text)
    return "".join(pieces)


def _make_chunk(
    meta: FinancialReportMeta,
    chunk_id: str,
    parent_id: str | None,
    prefix: str,
    section: str,
    units: Sequence[_Unit],
) -> Chunk:
    text = prefix + _join_units(units)
    page = next((unit.page for unit in units if unit.page is not None), None)
    is_table = any(unit.is_table for unit in units)
    table_title = next((unit.title for unit in units if unit.title), None)
    if table_title is None and is_table:
        table_title = section or None
    metadata = ChunkMetadata(
        doc_id=meta.doc_id,
        chunk_id=chunk_id,
        parent_id=parent_id,
        section=section,
        page=page,
        is_parent=parent_id is None,
        is_table=is_table,
        ticker=meta.ticker,
        company_name=meta.company_name,
        fiscal_year=meta.fiscal_year,
        quarter=meta.quarter,
        report_period=meta.report_period,
        table_title=table_title,
    )
    return Chunk(text=text, metadata=metadata)


def chunk_document(
    document: ParsedDocument,
    counter: TokenCounter,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """將解析結果切成父子切片。"""
    config = config or ChunkConfig()
    meta = document.meta
    doc_id = meta.doc_id
    chunks: list[Chunk] = []
    parent_index = 0
    for path, blocks in _sections(document.blocks):
        section = " > ".join(path)
        prefix = f"{section}\n" if section else ""
        prefix_tokens = counter(prefix) if section else 0
        budget = max(config.child_max - prefix_tokens, 1)
        units = [unit for block in blocks for unit in _block_units(block, counter, budget)]
        if not units:
            continue
        for parent_units in _pack(units, counter, config.parent_max, prefix_tokens):
            parent_id = f"{doc_id}:p{parent_index}"
            chunks.append(_make_chunk(meta, parent_id, None, prefix, section, parent_units))
            for child_index, child_units in enumerate(
                _pack(parent_units, counter, config.child_max, prefix_tokens)
            ):
                child_id = f"{parent_id}:c{child_index}"
                chunks.append(_make_chunk(meta, child_id, parent_id, prefix, section, child_units))
            parent_index += 1
    return chunks
