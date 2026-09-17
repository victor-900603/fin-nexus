"""領域資料與年度換算。"""

from __future__ import annotations


def to_western_year(roc_year: int) -> int:
    """民國年轉西元年。"""
    return roc_year + 1911
