"""HTTP 工具測試。"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from finrag.http import (
    build_client,
    get_bytes,
    get_text,
    post_text,
    request_with_retry,
)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_build_client_sets_user_agent() -> None:
    client = build_client(user_agent="test-agent", timeout=5.0)
    try:
        assert client.headers["User-Agent"] == "test-agent"
    finally:
        client.close()


def test_get_text_utf8() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="財務")

    with _client(handler) as client:
        assert get_text(client, "https://example.com/") == "財務"


def test_get_text_decodes_big5() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="財務報告書".encode("big5"))

    with _client(handler) as client:
        assert get_text(client, "https://example.com/", encoding="big5") == "財務報告書"


def test_get_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-1.7")

    with _client(handler) as client:
        assert get_bytes(client, "https://example.com/a.pdf") == b"%PDF-1.7"


def test_post_text_sends_referer_and_data() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["referer"] = request.headers.get("referer", "")
        captured["body"] = request.content.decode()
        return httpx.Response(200, text="ok")

    with _client(handler) as client:
        result = post_text(
            client,
            "https://example.com/",
            {"step": "9"},
            referer="https://ref.example/",
        )
    assert result == "ok"
    assert captured["referer"] == "https://ref.example/"
    assert captured["body"] == "step=9"


def test_request_with_retry_succeeds_after_failures() -> None:
    calls = 0
    delays: list[float] = []

    def request() -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("boom")
        return httpx.Response(200)

    response = request_with_retry(request, attempts=3, base_delay=0.0, sleep=delays.append)
    assert response.status_code == 200
    assert calls == 3
    assert delays == [0.0, 0.0]


def test_request_with_retry_raises_after_attempts() -> None:
    def request() -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(httpx.ConnectError):
        request_with_retry(request, attempts=2, base_delay=0.0, sleep=lambda _: None)


def test_request_with_retry_does_not_retry_client_error() -> None:
    calls = 0
    delays: list[float] = []

    def request() -> httpx.Response:
        nonlocal calls
        calls += 1
        response = httpx.Response(404, request=httpx.Request("GET", "https://example.com/"))
        response.raise_for_status()
        return response

    with pytest.raises(httpx.HTTPStatusError):
        request_with_retry(request, attempts=3, base_delay=0.0, sleep=delays.append)
    assert calls == 1
    assert delays == []


def test_request_with_retry_retries_server_error() -> None:
    calls = 0

    def request() -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            response = httpx.Response(503, request=httpx.Request("GET", "https://example.com/"))
            response.raise_for_status()
        return httpx.Response(200, request=httpx.Request("GET", "https://example.com/"))

    response = request_with_retry(request, attempts=3, base_delay=0.0, sleep=lambda _: None)
    assert response.status_code == 200
    assert calls == 3
