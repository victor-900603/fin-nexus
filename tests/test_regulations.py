"""全國法規資料庫法規擷取測試。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from finrag.ingest.regulations import (
    fetch_regulation,
    law_url,
    regulation_filename,
)
from finrag.models.domain import (
    ACQUISITION_DISPOSAL_LAW,
    INTERNAL_CONTROL_LAW,
    LAW_SPECS,
    SECURITIES_LAW,
    LawSpec,
)

LAW_HTML = "<div class='law-reg-content'><div class='row'>第 1 條</div></div>"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_law_url() -> None:
    assert law_url("G0400001") == "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=G0400001"


def test_regulation_filename() -> None:
    assert regulation_filename("G0400001") == "law_G0400001.html"


def test_law_specs() -> None:
    assert LAW_SPECS == (SECURITIES_LAW, ACQUISITION_DISPOSAL_LAW, INTERNAL_CONTROL_LAW)
    assert LawSpec("G0400001", "證券交易法") == SECURITIES_LAW
    assert ACQUISITION_DISPOSAL_LAW.name == "公開發行公司取得或處分資產處理準則"
    assert [spec.pcode for spec in LAW_SPECS] == ["G0400001", "G0400069", "G0400045"]


def test_fetch_regulation_writes_file(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=LAW_HTML)

    with _client(handler) as client:
        path = fetch_regulation(client, SECURITIES_LAW, tmp_path)

    assert path == tmp_path / "law_G0400001.html"
    assert "law-reg-content" in path.read_text(encoding="utf-8")
