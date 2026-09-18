"""HTML 表格轉 Markdown。"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag


def _cell_text(cell: Tag) -> str:
    text = " ".join(cell.get_text(" ", strip=True).split())
    return text.replace("|", "\\|")


def _span(cell: Tag, name: str) -> int:
    raw = cell.get(name)
    if raw is None:
        return 1
    try:
        value = int(str(raw))
    except ValueError:
        return 1
    return value if value > 0 else 1


def _rows(table: Tag) -> list[Tag]:
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


def _expand(table: Tag) -> list[list[str]]:
    rows = _rows(table)
    grid: list[list[str]] = []
    for row_index, tr in enumerate(rows):
        while len(grid) <= row_index:
            grid.append([])
        column = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while column < len(grid[row_index]) and grid[row_index][column] != "":
                column += 1
            colspan = _span(cell, "colspan")
            rowspan = _span(cell, "rowspan")
            text = _cell_text(cell)
            for offset_row in range(rowspan):
                target = row_index + offset_row
                while len(grid) <= target:
                    grid.append([])
                for offset_column in range(colspan):
                    col = column + offset_column
                    while len(grid[target]) <= col:
                        grid[target].append("")
                    grid[target][col] = text
            column += colspan
    return grid


def _trim(grid: list[list[str]]) -> list[list[str]]:
    if not grid:
        return grid
    width = max(len(row) for row in grid)
    padded = [row + [""] * (width - len(row)) for row in grid]
    kept = [index for index in range(width) if any(row[index] != "" for row in padded)]
    return [[row[index] for index in kept] for row in padded]


def _format_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def table_rows(html: str) -> list[list[str]]:
    """將 HTML 表格展開為列欄網格，找不到表格回空序列。"""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not isinstance(table, Tag):
        return []
    return _trim(_expand(table))


def grid_to_markdown(rows: list[list[str]]) -> str:
    """將列欄網格轉為 Markdown，首列作表頭。"""
    if not rows:
        return ""
    lines = [_format_row(rows[0]), _format_row(["---"] * len(rows[0]))]
    lines.extend(_format_row(row) for row in rows[1:])
    return "\n".join(lines)
