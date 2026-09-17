"""MOPS 財務報告書擷取測試。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from finrag.ingest.mops import (
    fetch_financial_report,
    list_report_files,
    mops_report_name,
    report_filename,
)
from finrag.models.domain import to_western_year

LIST_HTML = (
    'readfile2("A","2330","202501_2330_AI1.pdf");'
    'readfile2("A","2330","202504_2330_AI1.pdf");'
)
PDF_STEP_HTML = "<a href='/pdf/202501_2330_AI1_x.pdf'>下載</a>"
PDF_BYTES = b"%PDF-1.7 fake"


def _handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if request.method == "GET" and "t57sb01" in url and "step=1" in url:
        return httpx.Response(200, content=LIST_HTML.encode("big5"))
    if request.method == "POST":
        return httpx.Response(200, content=PDF_STEP_HTML.encode("big5"))
    if request.method == "GET" and url.endswith(".pdf"):
        return httpx.Response(200, content=PDF_BYTES)
    return httpx.Response(404)


def _client(handler: Callable[[httpx.Request], httpx.Response] = _handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_to_western_year() -> None:
    assert to_western_year(114) == 2025


def test_report_filename() -> None:
    assert report_filename("2330", 2025, "Q1") == "2330_2025_Q1_consolidated.pdf"


def test_mops_report_name() -> None:
    assert mops_report_name("2330", 2025, "Q1") == "202501_2330_AI1.pdf"
    assert mops_report_name("2330", 2025, "Q4") == "202504_2330_AI1.pdf"


def test_mops_report_name_rejects_unknown_quarter() -> None:
    with pytest.raises(ValueError):
        mops_report_name("2330", 2025, "Q5")


def test_list_report_files() -> None:
    with _client() as client:
        files = list_report_files(client, "2330", 114)
    assert files == ["202501_2330_AI1.pdf", "202504_2330_AI1.pdf"]


def test_fetch_financial_report_writes_file(tmp_path: Path) -> None:
    with _client() as client:
        path = fetch_financial_report(client, "2330", 114, "Q1", tmp_path)
    assert path == tmp_path / "2330_2025_Q1_consolidated.pdf"
    assert path.read_bytes() == PDF_BYTES


def test_fetch_financial_report_missing_file(tmp_path: Path) -> None:
    with _client() as client, pytest.raises(FileNotFoundError):
        fetch_financial_report(client, "2330", 114, "Q3", tmp_path)


def test_fetch_financial_report_missing_link(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "step=1" in str(request.url):
            return httpx.Response(200, content=LIST_HTML.encode("big5"))
        if request.method == "POST":
            return httpx.Response(200, content="<html>無</html>".encode("big5"))
        return httpx.Response(404)

    with _client(handler) as client, pytest.raises(FileNotFoundError):
        fetch_financial_report(client, "2330", 114, "Q1", tmp_path)
