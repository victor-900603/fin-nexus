"""html_table 單元測試。"""

from __future__ import annotations

from fin_nexus.ingest.html_table import grid_to_markdown, table_rows


def _markdown(html: str) -> str:
    return grid_to_markdown(table_rows(html))


def test_markdown_basic() -> None:
    html = (
        "<table><tr><th>項目</th><th>金額</th></tr>"
        "<tr><td>現金及約當現金</td><td>100</td></tr></table>"
    )
    markdown = _markdown(html)
    assert markdown == "| 項目 | 金額 |\n| --- | --- |\n| 現金及約當現金 | 100 |"


def test_markdown_expands_colspan() -> None:
    html = '<table><tr><th colspan="2">標題</th></tr><tr><td>a</td><td>b</td></tr></table>'
    markdown = _markdown(html)
    assert markdown.splitlines()[0] == "| 標題 | 標題 |"


def test_markdown_expands_rowspan() -> None:
    html = (
        "<table><tr><th>h1</th><th>h2</th></tr>"
        '<tr><td rowspan="2">a</td><td>1</td></tr>'
        "<tr><td>2</td></tr></table>"
    )
    markdown = _markdown(html)
    assert markdown.splitlines()[2] == "| a | 1 |"
    assert markdown.splitlines()[3] == "|  | 2 |"


def test_markdown_escapes_pipe() -> None:
    html = "<table><tr><th>a</th></tr><tr><td>x|y</td></tr></table>"
    markdown = _markdown(html)
    assert "x\\|y" in markdown


def test_markdown_trims_empty_columns() -> None:
    html = "<table><tr><th>a</th><th></th></tr><tr><td>1</td><td></td></tr></table>"
    markdown = _markdown(html)
    assert markdown.splitlines()[0] == "| a |"


def test_markdown_handles_ragged_rows() -> None:
    html = (
        "<table><tr><th>h1</th><th>h2</th></tr>"
        '<tr><td rowspan="2">a</td><td>1</td></tr>'
        "<tr></tr></table>"
    )
    markdown = _markdown(html)
    assert markdown.splitlines()[3] == "|  |  |"


def test_markdown_rowspan_continuation_is_empty() -> None:
    html = (
        "<table><tr><th>h1</th><th>h2</th><th>h3</th></tr>"
        '<tr><td rowspan="2">a</td><td>1</td><td>2</td></tr>'
        "<tr><td>3</td><td>4</td></tr></table>"
    )
    markdown = _markdown(html)
    assert markdown.splitlines()[2] == "| a | 1 | 2 |"
    assert markdown.splitlines()[3] == "|  | 3 | 4 |"


def test_markdown_uniform_columns() -> None:
    html = (
        "<table><tr><th>h1</th><th>h2</th><th>h3</th></tr>"
        '<tr><td rowspan="2">a</td><td colspan="2">1</td></tr>'
        "<tr><td>3</td><td>4</td></tr></table>"
    )
    lines = _markdown(html).splitlines()
    assert len({line.count("|") for line in lines}) == 1


def test_table_rows_without_table_returns_empty() -> None:
    assert table_rows("<p>hi</p>") == []


def test_markdown_empty_table() -> None:
    assert _markdown("<table></table>") == ""
    assert grid_to_markdown([]) == ""
