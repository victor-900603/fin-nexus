"""財報目標清單。"""

from __future__ import annotations

import re
from collections.abc import Iterable

from finrag.models.document import QUARTERS
from finrag.models.domain import COMPANY_NAMES, ReportTarget

DEFAULT_TICKERS: tuple[str, ...] = tuple(COMPANY_NAMES)
DEFAULT_YEARS: tuple[int, ...] = (2023, 2024, 2025)
DEFAULT_QUARTERS: tuple[str, ...] = QUARTERS

_TICKER_RE = re.compile(r"^\d{4}$")


def build_targets(
    tickers: Iterable[str],
    years: Iterable[int],
    quarters: Iterable[str],
) -> list[ReportTarget]:
    """依公司、年度與季度組合出目標清單。"""
    targets: list[ReportTarget] = []
    for ticker in tickers:
        if not _TICKER_RE.match(ticker):
            raise ValueError(f"公司代號格式不正確：{ticker}")
        company_name = COMPANY_NAMES.get(ticker, ticker)
        for year in years:
            for quarter in quarters:
                if quarter not in QUARTERS:
                    raise ValueError(f"不支援的季度：{quarter}")
                targets.append(ReportTarget(ticker, company_name, year, quarter))
    return targets


def default_targets() -> list[ReportTarget]:
    """首波預設目標：台積電 2023–2025 年四季。"""
    return build_targets(DEFAULT_TICKERS, DEFAULT_YEARS, DEFAULT_QUARTERS)
