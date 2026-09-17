"""normalize 單元測試。"""

from __future__ import annotations

from finrag.ingest.normalize import normalize_text


def test_removes_space_between_cjk() -> None:
    assert normalize_text("台灣 積體 電路") == "台灣積體電路"


def test_removes_ideographic_space_between_cjk() -> None:
    assert normalize_text("台灣\u3000積體") == "台灣積體"


def test_preserves_newline_between_cjk() -> None:
    assert normalize_text("台灣\n積體") == "台灣\n積體"


def test_collapses_space_around_punctuation() -> None:
    assert normalize_text("台灣積體電路 （ 股 ） 公司") == "台灣積體電路（股）公司"


def test_keeps_space_next_to_digits() -> None:
    assert normalize_text("現金 100 元") == "現金 100 元"


def test_keeps_space_between_fullwidth_digits() -> None:
    assert normalize_text("１０ ２０") == "１０ ２０"


def test_keeps_space_next_to_ascii() -> None:
    assert normalize_text("EPS 每股盈餘") == "EPS 每股盈餘"


def test_converts_simplified_to_traditional() -> None:
    assert normalize_text("综计现员") == "綜計現員"


def test_keeps_shared_characters() -> None:
    assert normalize_text("台灣 公里 干涉") == "台灣公里干涉"


def test_traditional_text_unchanged() -> None:
    assert normalize_text("財務報表") == "財務報表"


def test_idempotent() -> None:
    once = normalize_text("台灣 積體 電路 综计")
    assert normalize_text(once) == once
