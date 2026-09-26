"""上下文語意壓縮：去重、去分頁符並保護引用片段。"""

from __future__ import annotations

import re

from pydantic import BaseModel

from finrag.text import TokenCounter

_FORM_FEED = "\f"
_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$")
_ARTICLE = re.compile(r"第\s*\d+(?:-\d+)?\s*條")
_DIGIT = re.compile(r"\d")
_TABLE_ROW = re.compile(r"^\s*\|")
_TABLE_TITLE = re.compile(r"^\s*(\*\*.+\*\*|單位[：:])")


class CompactionResult(BaseModel):
    """壓縮後的上下文與統計。"""

    text: str
    original_tokens: int
    compacted_tokens: int
    removed_lines: int


def _protected(line: str) -> bool:
    return bool(
        _ARTICLE.search(line)
        or _DIGIT.search(line)
        or _TABLE_ROW.match(line)
        or _TABLE_TITLE.match(line)
    )


def compact_text(text: str, *, counter: TokenCounter) -> CompactionResult:
    """將單一字串壓縮為精簡上下文，保留含條號、數值與表格結構的引用片段。"""
    seen: set[str] = set()
    blocks: list[str] = []
    removed = 0
    for block in text.split("\n\n"):
        lines: list[str] = []
        for raw in block.replace(_FORM_FEED, "\n").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if _PAGE_NUMBER.match(raw):
                removed += 1
                continue
            if stripped in seen and not _protected(stripped):
                removed += 1
                continue
            seen.add(stripped)
            lines.append(stripped)
        if lines:
            blocks.append("\n".join(lines))

    compacted_text = "\n\n".join(blocks)
    return CompactionResult(
        text=compacted_text,
        original_tokens=counter(text),
        compacted_tokens=counter(compacted_text),
        removed_lines=removed,
    )
