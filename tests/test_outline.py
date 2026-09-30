"""outline 單元測試。"""

from __future__ import annotations

from finrag.ingest.outline import (
    OutlineEntry,
    extract_outline,
    is_heading,
    normalize_heading,
    page_sections,
)
from finrag.models import Block, BlockType

_OUTLINE_TEXT = "\n".join(
    [
        "| 項目 | 頁次 | 附註編號 |",
        "| --- | --- | --- |",
        "| 三、會計師核閱報告 | 3 |  |",
        "| 四、合併資產負債表 | 4 |  |",
        "| 五、合併綜合損益表 | 5~7 |  |",
        "| (十三)附註揭露事項 |  |  |",
        "| 1.重大交易事項相關資訊 | 56~57、58~91 |  |",
        "| (十)重大之災害損失 | - |  |",
    ]
)


def _table_block(text: str, page: int = 1) -> Block:
    return Block(type=BlockType.TABLE, text=text, page=page)


def test_extract_outline_reads_names_and_pages() -> None:
    entries = extract_outline([_table_block(_OUTLINE_TEXT)])
    assert entries[0] == OutlineEntry(name="三、會計師核閱報告", pages=(2,))
    assert entries[1] == OutlineEntry(name="四、合併資產負債表", pages=(3,))
    assert entries[2] == OutlineEntry(name="五、合併綜合損益表", pages=(4, 5, 6))


def test_extract_outline_expands_multiple_segments() -> None:
    entries = extract_outline([_table_block(_OUTLINE_TEXT)])
    entry = next(item for item in entries if item.name.startswith("1."))
    assert entry.pages == tuple(range(55, 91))


def test_extract_outline_skips_entries_without_pages() -> None:
    names = [entry.name for entry in extract_outline([_table_block(_OUTLINE_TEXT)])]
    assert "(十三)附註揭露事項" not in names
    assert "(十)重大之災害損失" not in names


def test_extract_outline_ignores_other_tables() -> None:
    block = _table_block("| 代碼 | 資產 | 金額 |\n| --- | --- | --- |\n| 1100 | 現金 | 100 |")
    assert extract_outline([block]) == []


_SPLIT_HEADER_TEXT = "\n".join(
    [
        "| 項 | 目頁 | 次 | 附註 | 編號 |",
        "| --- | --- | --- | --- | --- |",
        "| 一、封面 | 1 |  | - |  |",
        "| 二、目錄 | 2 |  | - |  |",
        "| 三、會計師核閱報告 | 3 |  | - |  |",
        "| 四、合併資產負債表 | 4 |  | - |  |",
        "| 五、合併綜合損益表 | 5 |  | - |  |",
    ]
)

_SHIFTED_PAGE_COLUMN_TEXT = "\n".join(
    [
        "| 項 | 目 | 頁 | 次 | 財務報告附註編號 |",
        "| --- | --- | --- | --- | --- |",
        "| 一、封面 |  | 1 |  | - |",
        "| 二、目錄 |  | 2 |  | - |",
        "| 三、會計師核閱報告 |  | 3 |  | - |",
        "| 四、合併資產負債表 |  | 4 |  | - |",
        "| 五、合併綜合損益表 |  | 5 |  | - |",
    ]
)

_FLAT_TEXT_OUTLINE = (
    "項目一、封面 1 二、目錄 2 三、會計師核閱報告 3-4 四、合併資產負債表 5-6 "
    "五、合併綜合損益表 7 六、合併權益變動表 8"
)


def test_extract_outline_reads_split_header() -> None:
    entries = extract_outline([_table_block(_SPLIT_HEADER_TEXT)])
    assert entries[0] == OutlineEntry(name="一、封面", pages=(0,))
    assert entries[2] == OutlineEntry(name="三、會計師核閱報告", pages=(2,))


def test_extract_outline_finds_page_column_at_any_position() -> None:
    entries = extract_outline([_table_block(_SHIFTED_PAGE_COLUMN_TEXT)])
    assert entries[0] == OutlineEntry(name="一、封面", pages=(0,))
    assert entries[3] == OutlineEntry(name="四、合併資產負債表", pages=(3,))


def test_extract_outline_reads_flat_text_block() -> None:
    block = Block(type=BlockType.TEXT, text=_FLAT_TEXT_OUTLINE, page=1)
    entries = extract_outline([block])
    assert entries[0] == OutlineEntry(name="一、封面", pages=(0,))
    assert entries[2] == OutlineEntry(name="三、會計師核閱報告", pages=(2, 3))
    assert entries[3] == OutlineEntry(name="四、合併資產負債表", pages=(4, 5))


def test_extract_outline_prefers_table_over_text() -> None:
    text_block = Block(type=BlockType.TEXT, text=_FLAT_TEXT_OUTLINE, page=1)
    entries = extract_outline([text_block, _table_block(_OUTLINE_TEXT)])
    assert entries[0].name == "三、會計師核閱報告"


def test_extract_outline_ignores_text_with_few_entries() -> None:
    block = Block(type=BlockType.TEXT, text="一、封面 1 二、目錄 2", page=1)
    assert extract_outline([block]) == []


def test_page_sections_maps_every_page() -> None:
    mapping = page_sections(
        [OutlineEntry(name="甲", pages=(1, 2)), OutlineEntry(name="乙", pages=(2, 3))]
    )
    assert mapping == {1: "甲", 2: "乙", 3: "乙"}


def test_normalize_heading_removes_punctuation() -> None:
    assert normalize_heading("(一) 公司沿革") == normalize_heading("一、公司沿革")


def test_is_heading_accepts_numbered_titles() -> None:
    assert is_heading("十三、採用權益法之投資")
    assert is_heading("十二 、 存貨")
    assert is_heading("三一 、 關係 人 交易")
    assert is_heading("(一) 公司沿革")
    assert is_heading("（十三）附註揭露事項")
    assert is_heading("1. 合併報告編製原則")
    assert is_heading("2.3轉投資事業相關資訊")
    assert is_heading("A. 現金及約當現金")


def test_is_heading_rejects_unnumbered_text() -> None:
    assert not is_heading("（承前頁）")
    assert not is_heading("§目錄§")
    assert not is_heading("單位：新台幣仟元")
    assert not is_heading("民國 112 年 1月 1日至 3月 31日")
    assert not is_heading("台灣積體電路製造股份有限公司及子公司")
    assert not is_heading("(金額均以新臺幣千元或外幣千元為單位)")
    assert not is_heading("利率風險")
    assert not is_heading("1100 現金及約當現金")
    assert not is_heading("   ")


_FLAT_WITH_SUB_ENTRIES = (
    "項目一、封面 1 二、目錄 2 三、會計師核閱報告 3-4 四、合併資產負債表 5-6 "
    "五、合併綜合損益表 7 六、合併權益變動表 8 （十三）附註揭露事項 "
    "1.重大交易事項及轉投資事業相關資訊 59-60、62-73 2.大陸投資資訊 74-76 3.主要股東資訊 81"
)


def test_extract_outline_reads_arabic_sub_entries() -> None:
    block = Block(type=BlockType.TEXT, text=_FLAT_WITH_SUB_ENTRIES, page=1)
    entries = extract_outline([block])

    assert OutlineEntry(
        name="1.重大交易事項及轉投資事業相關資訊", pages=(58, 59, *range(61, 73))
    ) in entries
    assert OutlineEntry(name="2.大陸投資資訊", pages=(73, 74, 75)) in entries
    assert all(entry.name != "（十三）附註揭露事項" for entry in entries)


def test_page_sections_fills_pages_missing_from_outline() -> None:
    mapping = page_sections(
        [OutlineEntry(name="甲", pages=(1,)), OutlineEntry(name="乙", pages=(4,))]
    )
    assert mapping == {1: "甲", 2: "甲", 3: "甲", 4: "乙"}


def test_page_sections_fills_pages_after_last_entry() -> None:
    mapping = page_sections(
        [OutlineEntry(name="甲", pages=(1,)), OutlineEntry(name="乙", pages=(4,))], last_page=6
    )
    assert mapping == {1: "甲", 2: "甲", 3: "甲", 4: "乙", 5: "乙", 6: "乙"}


def test_page_sections_without_last_page_keeps_tail_empty() -> None:
    mapping = page_sections([OutlineEntry(name="甲", pages=(1,))])
    assert mapping == {1: "甲"}
