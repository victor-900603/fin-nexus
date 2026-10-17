"""全國法規資料庫法規全文擷取。"""

from __future__ import annotations

from pathlib import Path

import httpx

from fin_nexus.http import get_text
from fin_nexus.models.domain import LawSpec

LAW_BASE_URL = "https://law.moj.gov.tw/LawClass/LawAll.aspx"


def law_url(pcode: str) -> str:
    """法規全文頁網址。"""
    return f"{LAW_BASE_URL}?pcode={pcode}"


def regulation_filename(pcode: str) -> str:
    """本地檔名，格式 law_{pcode}.html。"""
    return f"law_{pcode}.html"


def fetch_regulation(client: httpx.Client, spec: LawSpec, dest_dir: Path) -> Path:
    """下載單一法規全文 HTML，回傳本地路徑。"""
    html = get_text(client, law_url(spec.pcode))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / regulation_filename(spec.pcode)
    dest.write_text(html, encoding="utf-8")
    return dest
