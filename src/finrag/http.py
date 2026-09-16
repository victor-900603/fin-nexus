"""HTTP 請求工具。"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import httpx

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 30.0


def build_client(
    user_agent: str = BROWSER_USER_AGENT,
    timeout: float = DEFAULT_TIMEOUT,
) -> httpx.Client:
    """建立帶預設標頭的 HTTP 客戶端。"""
    return httpx.Client(
        headers={"User-Agent": user_agent},
        follow_redirects=True,
        timeout=timeout,
    )


def _headers(referer: str | None) -> dict[str, str]:
    return {"Referer": referer} if referer else {}


def _decode(response: httpx.Response, encoding: str | None) -> str:
    if encoding is not None:
        return response.content.decode(encoding, errors="replace")
    return response.text


def request_with_retry(
    request: Callable[[], httpx.Response],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """執行請求，失敗時以指數退避加上隨機抖動重試。"""
    last_error: httpx.HTTPError | None = None
    for index in range(attempts):
        try:
            return request()
        except httpx.HTTPError as error:
            last_error = error
            client_error = (
                isinstance(error, httpx.HTTPStatusError)
                and error.response.status_code < 500
            )
            if client_error or index == attempts - 1:
                break
            delay = base_delay * (2**index)
            sleep(delay + random.uniform(0.0, delay))
    assert last_error is not None
    raise last_error


def _send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: dict[str, str] | None = None,
    json: Any | None = None,
) -> httpx.Response:
    def request() -> httpx.Response:
        response = client.request(method, url, headers=headers, data=data, json=json)
        response.raise_for_status()
        return response

    return request_with_retry(request)


def get_text(
    client: httpx.Client,
    url: str,
    *,
    referer: str | None = None,
    encoding: str | None = None,
) -> str:
    """GET 網址並回傳文字。"""
    response = _send(client, "GET", url, headers=_headers(referer))
    return _decode(response, encoding)


def post_text(
    client: httpx.Client,
    url: str,
    data: dict[str, str],
    *,
    referer: str | None = None,
    encoding: str | None = None,
) -> str:
    """POST 表單並回傳文字。"""
    response = _send(client, "POST", url, headers=_headers(referer), data=data)
    return _decode(response, encoding)


def get_bytes(
    client: httpx.Client,
    url: str,
    *,
    referer: str | None = None,
) -> bytes:
    """GET 網址並回傳位元組。"""
    response = _send(client, "GET", url, headers=_headers(referer))
    return response.content
