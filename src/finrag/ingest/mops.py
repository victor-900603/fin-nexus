"""公開資訊觀測站財務報告書擷取。"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

from finrag.http import get_bytes, get_text, post_text
from finrag.models.document import QUARTERS
from finrag.models.domain import to_western_year

DOC_HOST = "https://doc.twse.com.tw"
MOPS_DOC_URL = f"{DOC_HOST}/server-java/t57sb01"
REPORT_TYPE = "A"
REPORT_KIND = "AI1"

_FILE_RE = re.compile(r'readfile2\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
_PDF_RE = re.compile(r"href=['\"]([^'\"]+\.pdf)['\"]")
_QUARTER_NUM: dict[str, int] = {quarter: index + 1 for index, quarter in enumerate(QUARTERS)}


def report_filename(ticker: str, fiscal_year: int, quarter: str) -> str:
    """本地檔名，格式 {ticker}_{fiscal_year}_{quarter}_consolidated.pdf。"""
    return f"{ticker}_{fiscal_year}_{quarter}_consolidated.pdf"


def mops_report_name(ticker: str, fiscal_year: int, quarter: str) -> str:
    """MOPS 清單中的檔名，格式 {fiscal_year}{quarter:02d}_{ticker}_{kind}.pdf。"""
    try:
        number = _QUARTER_NUM[quarter]
    except KeyError as error:
        raise ValueError(f"不支援的季度：{quarter}") from error
    return f"{fiscal_year}{number:02d}_{ticker}_{REPORT_KIND}.pdf"


def list_url(ticker: str, roc_year: int) -> str:
    """MOPS 財務報告書清單網址。"""
    return (
        f"{MOPS_DOC_URL}?step=1&colorchg=1&co_id={ticker}"
        f"&year={roc_year}&seamon=&mtype={REPORT_TYPE}&"
    )


def list_report_files(client: httpx.Client, ticker: str, roc_year: int) -> list[str]:
    """取得 MOPS 財務報告書清單的檔名。"""
    html = get_text(client, list_url(ticker, roc_year), encoding="big5")
    return [
        name
        for kind, co_id, name in _FILE_RE.findall(html)
        if kind == REPORT_TYPE and co_id == ticker
    ]


def fetch_financial_report(
    client: httpx.Client,
    ticker: str,
    roc_year: int,
    quarter: str,
    dest_dir: Path,
) -> Path:
    """下載單一季度的合併財務報告書，回傳本地路徑。"""
    fiscal_year = to_western_year(roc_year)
    name = mops_report_name(ticker, fiscal_year, quarter)
    if name not in list_report_files(client, ticker, roc_year):
        raise FileNotFoundError(f"MOPS 財務報告書清單缺少 {name}")

    data = {
        "step": "9",
        "kind": REPORT_TYPE,
        "co_id": ticker,
        "filename": name,
        "colorchg": "1",
    }
    html = post_text(client, MOPS_DOC_URL, data, referer=MOPS_DOC_URL, encoding="big5")
    match = _PDF_RE.search(html)
    if match is None:
        raise FileNotFoundError(f"MOPS 未回傳 {name} 的 PDF 連結")

    path = match.group(1)
    pdf_url = f"{DOC_HOST}{path}" if path.startswith("/") else path
    content = get_bytes(client, pdf_url, referer=MOPS_DOC_URL)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / report_filename(ticker, fiscal_year, quarter)
    dest.write_bytes(content)
    return dest
