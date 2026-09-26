"""MinerU middle_json 財報解析。"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finrag.config import DEFAULT_TIER, ParseConfig
from finrag.ingest.html_table import grid_to_markdown, table_rows
from finrag.ingest.normalize import normalize_text
from finrag.models import Block, BlockType, FinancialReportMeta, ParsedDocument

SKIP_BLOCK_TYPES = {"image", "page_number", "header", "footer", "aside_text"}
_TABLE_BODY = "table_body"
_TABLE_CAPTION = "table_caption"
_TABLE_FOOTNOTE = "table_footnote"
_PARSE_ENGINE = "mineru"
_MIDDLE_SCHEMA = "docvortex.middle"
_MAX_ERROR_OUTPUT = 2000
_API_KEY_FLAG = "--api-key"

Runner = Callable[[Sequence[str]], None]


def _redact(command: Sequence[str]) -> str:
    parts = list(command)
    for index, part in enumerate(parts):
        if part == _API_KEY_FLAG and index + 1 < len(parts):
            parts[index + 1] = "***"
        elif part.startswith(f"{_API_KEY_FLAG}="):
            parts[index] = f"{_API_KEY_FLAG}=***"
    return " ".join(parts)


def _default_runner(command: Sequence[str]) -> None:
    try:
        subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"找不到執行檔：{command[0]}") from error
    except subprocess.CalledProcessError as error:
        output = (error.stderr or error.stdout or "").strip()
        if len(output) > _MAX_ERROR_OUTPUT:
            output = output[-_MAX_ERROR_OUTPUT:]
        raise RuntimeError(
            f"指令失敗（exit {error.returncode}）：{_redact(command)}\n{output}"
        ) from error


def run_mineru(
    pdf: Path,
    out_dir: Path,
    tier: str = DEFAULT_TIER,
    runner: Runner | None = None,
    *,
    remote: bool = False,
    api_key: str | None = None,
    remote_url: str | None = None,
) -> None:
    """以 MinerU 解析 PDF，輸出 middle_json 至 out_dir。"""
    command = [
        "mineru-kit",
        "parse",
        str(pdf),
        "-o",
        str(out_dir),
        "-f",
        "middle_json",
    ]
    if remote:
        command.append("--remote")
        if api_key:
            command.extend([_API_KEY_FLAG, api_key])
        if remote_url:
            command.extend(["--remote-url", remote_url])
    else:
        command.extend(["--tier", tier])
    (runner or _default_runner)(command)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_extract_text(item) for item in value)
    if isinstance(value, dict):
        return _extract_text(value.get("content"))
    return ""


def _table_parts(content: Any) -> tuple[list[list[str]], list[str], list[str]]:
    rows: list[list[str]] = []
    captions: list[str] = []
    footnotes: list[str] = []
    if isinstance(content, str):
        text = _clean(content)
        if text:
            footnotes.append(text)
        return rows, captions, footnotes
    if not isinstance(content, list):
        return rows, captions, footnotes
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == _TABLE_BODY:
            rows.extend(table_rows(str(item.get("content", ""))))
        elif item_type == _TABLE_CAPTION:
            text = _clean(_extract_text(item.get("content")))
            if text:
                captions.append(text)
        elif item_type == _TABLE_FOOTNOTE:
            text = _clean(_extract_text(item.get("content")))
            if text:
                footnotes.append(text)
    return rows, captions, footnotes


def _render_table(rows: list[list[str]], caption: str | None, footnotes: list[str]) -> str:
    parts: list[str] = []
    if caption:
        parts.append(f"**{caption}**")
    markdown = grid_to_markdown(rows)
    if markdown:
        parts.append(markdown)
    parts.extend(footnotes)
    return "\n\n".join(parts)


def _merge_rows(prev: list[list[str]], new: list[list[str]]) -> list[list[str]]:
    if not new:
        return prev
    if not prev:
        return new
    width = max(len(prev[0]), len(new[0]))
    tail = new[1:] if prev[0] == new[0] else new
    return [row + [""] * (width - len(row)) for row in [*prev, *tail]]


def _continues_prev(raw: dict[str, Any]) -> bool:
    return raw.get("continues_prev") is True


def _text_block(block_type: Any, raw: dict[str, Any], page: int | None) -> Block | None:
    text = _clean(_extract_text(raw.get("content")))
    if not text:
        return None
    if block_type == "paragraph_title":
        level = raw.get("level")
        return Block(
            type=BlockType.HEADING,
            text=text,
            level=level if isinstance(level, int) else 2,
            page=page,
        )
    return Block(type=BlockType.TEXT, text=text, page=page)


def _iter_raw_blocks(data: dict[str, Any]) -> Iterator[tuple[dict[str, Any], int | None]]:
    pages = data.get("pages")
    if not isinstance(pages, list):
        return
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_idx = page.get("page_idx")
        page_number = page_idx if isinstance(page_idx, int) else None
        raw_blocks = page.get("blocks")
        if not isinstance(raw_blocks, list):
            continue
        for raw in raw_blocks:
            if not isinstance(raw, dict):
                continue
            block_type = raw.get("type")
            if block_type in SKIP_BLOCK_TYPES:
                continue
            yield raw, page_number


@dataclass
class _BlockBuilder:
    blocks: list[Block] = field(default_factory=list)
    _table_grid: list[list[str]] | None = None
    _table_caption: str | None = None
    _table_footnotes: list[str] = field(default_factory=list)

    def add(self, raw: dict[str, Any], page: int | None) -> None:
        if raw.get("type") == "table":
            self._add_table(raw, page)
        else:
            self._add_text(raw, page)

    def _add_table(self, raw: dict[str, Any], page: int | None) -> None:
        rows, captions, footnotes = _table_parts(raw.get("content"))
        if not rows and not captions and not footnotes:
            return
        previous = self.blocks[-1] if self.blocks else None
        if (
            _continues_prev(raw)
            and previous is not None
            and previous.type is BlockType.TABLE
            and self._table_grid is not None
        ):
            self._table_grid = _merge_rows(self._table_grid, rows)
            self._table_footnotes = [*self._table_footnotes, *footnotes]
            self.blocks[-1] = Block(
                type=BlockType.TABLE,
                text=_render_table(
                    self._table_grid, self._table_caption, self._table_footnotes
                ),
                page=previous.page,
            )
            return
        caption = captions[0] if captions else None
        self._table_grid = rows
        self._table_caption = caption
        self._table_footnotes = list(footnotes)
        self.blocks.append(
            Block(type=BlockType.TABLE, text=_render_table(rows, caption, footnotes), page=page)
        )

    def _add_text(self, raw: dict[str, Any], page: int | None) -> None:
        self._table_grid = None
        self._table_caption = None
        self._table_footnotes = []
        block = _text_block(raw.get("type"), raw, page)
        if block is None:
            return
        previous = self.blocks[-1] if self.blocks else None
        if (
            _continues_prev(raw)
            and block.type is BlockType.TEXT
            and previous is not None
            and previous.type is BlockType.TEXT
        ):
            self.blocks[-1] = Block(
                type=BlockType.TEXT,
                text=f"{previous.text}\n{block.text}",
                page=previous.page,
            )
        else:
            self.blocks.append(block)


def middle_json_to_blocks(data: dict[str, Any]) -> list[Block]:
    """將 MinerU middle_json 轉為內容區塊序列，合併跨頁續接的區塊。"""
    builder = _BlockBuilder()
    for raw, page in _iter_raw_blocks(data):
        builder.add(raw, page)
    return [
        block.model_copy(update={"text": normalize_text(block.text)})
        for block in builder.blocks
    ]


def _is_middle_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and data.get("schema") == _MIDDLE_SCHEMA


def _find_middle_json(out_dir: Path, stem: str) -> Path | None:
    candidates = sorted(out_dir.rglob("*.json"))
    for path in candidates:
        if path.stem == stem:
            return path
    for path in candidates:
        if _is_middle_json(path):
            return path
    return None


def parse_pdf(
    pdf: Path,
    out_dir: Path,
    meta: FinancialReportMeta,
    config: ParseConfig | None = None,
    runner: Runner | None = None,
) -> ParsedDocument:
    """解析單一財報 PDF，回傳解析結果。"""
    resolved = config or ParseConfig()
    if resolved.mode == "remote" and not resolved.api_key:
        raise ValueError("remote 模式需提供 API 金鑰（FINRAG_MINERU_API_KEY）")
    run_mineru(
        pdf,
        out_dir,
        tier=resolved.tier,
        runner=runner,
        remote=resolved.mode == "remote",
        api_key=resolved.api_key,
        remote_url=resolved.remote_url,
    )
    middle = _find_middle_json(out_dir, pdf.stem)
    if middle is None:
        raise FileNotFoundError(f"找不到 MinerU middle_json：{out_dir}")
    data = json.loads(middle.read_text(encoding="utf-8"))
    blocks = middle_json_to_blocks(data)
    updated = meta.model_copy(update={"parse_engine": _PARSE_ENGINE})
    return ParsedDocument(meta=updated, blocks=blocks)
