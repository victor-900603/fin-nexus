"""上下文語意壓縮單元測試。"""

from __future__ import annotations

from fin_nexus.retrieval.compactor import compact_text


def _counter(text: str) -> int:
    return len(text)


def test_deduplicates_identical_lines_across_blocks() -> None:
    text = "營業收入\n其他\n\n營業收入\n備註"

    result = compact_text(text, counter=_counter)

    assert result.text.count("營業收入") == 1
    assert result.removed_lines == 1


def test_removes_page_numbers_and_form_feed() -> None:
    result = compact_text("第一段\f12\n第二段", counter=_counter)

    assert result.text == "第一段\n第二段"
    assert result.removed_lines == 1


def test_keeps_protected_lines_when_duplicated() -> None:
    text = "第 1 條 為發展國民經濟\n\n第 1 條 為發展國民經濟"

    result = compact_text(text, counter=_counter)

    assert result.text.count("第 1 條") == 2
    assert result.removed_lines == 0


def test_table_structure_is_preserved() -> None:
    table_a = "| 項目 | 金額 |\n| --- | --- |\n| 現金 | 100 |"
    table_b = "| 項目 | 金額 |\n| --- | --- |\n| 存貨 | 200 |"

    result = compact_text(f"{table_a}\n\n{table_b}", counter=_counter)

    assert result.text.count("| 項目 | 金額 |") == 2
    assert result.text.count("| --- | --- |") == 2
    assert "| 現金 | 100 |" in result.text
    assert "| 存貨 | 200 |" in result.text
    assert result.removed_lines == 0


def test_reports_token_counts() -> None:
    result = compact_text("營業收入\n營業收入", counter=_counter)

    assert result.original_tokens > result.compacted_tokens
    assert result.compacted_tokens == len(result.text)


def test_empty_input_returns_empty() -> None:
    result = compact_text("", counter=_counter)

    assert result.text == ""
    assert result.original_tokens == 0
    assert result.compacted_tokens == 0
    assert result.removed_lines == 0


def test_deduplicates_and_keeps_numbered_blocks() -> None:
    text = "[1] 2330 | 第 3 頁\n營業收入\n\n[2] 2330 | 第 4 頁\n營業收入\n備註"

    result = compact_text(text, counter=_counter)

    assert result.text.count("營業收入") == 1
    assert "[1] 2330 | 第 3 頁" in result.text
    assert "[2] 2330 | 第 4 頁" in result.text
    assert result.removed_lines == 1
    assert result.original_tokens > result.compacted_tokens
