"""OpenAI 相容 Chat Completions 客戶端。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from fin_nexus.config import GenerationConfig
from fin_nexus.http import post_json, stream_lines

_CHAT_PATH = "/chat/completions"
_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class ChatMessage(BaseModel):
    """一則對話訊息。"""

    role: str
    content: str


class LLMClient(Protocol):
    """語言模型客戶端。"""

    def chat(self, messages: Sequence[ChatMessage]) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]: ...


class ChatCompletionsClient:
    """以 OpenAI 相容端點產生回答。"""

    def __init__(self, config: GenerationConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _payload(self, messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {
            "model": self._config.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """送出對話並回傳回答文字。"""
        url = f"{self._config.base_url}{_CHAT_PATH}"
        payload = self._payload(messages)
        headers = self._headers()
        if self._client is not None:
            return _parse(post_json(self._client, url, payload, headers=headers))
        with httpx.Client(timeout=self._config.timeout) as client:
            return _parse(post_json(client, url, payload, headers=headers))

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        """以串流送出對話並逐段回傳文字。"""
        url = f"{self._config.base_url}{_CHAT_PATH}"
        payload = self._payload(messages)
        payload["stream"] = True
        headers = self._headers()
        if self._client is not None:
            yield from _iter_deltas(
                stream_lines(self._client, url, payload, headers=headers)
            )
            return
        with httpx.Client(timeout=self._config.timeout) as client:
            yield from _iter_deltas(stream_lines(client, url, payload, headers=headers))


def _parse(data: Any) -> str:
    return str(data["choices"][0]["message"]["content"])


def _delta_content(data: str) -> str:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return ""
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _iter_deltas(lines: Iterator[str]) -> Iterator[str]:
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(_SSE_DATA_PREFIX):
            continue
        data = stripped[len(_SSE_DATA_PREFIX) :].strip()
        if data == _SSE_DONE:
            return
        content = _delta_content(data)
        if content:
            yield content


@lru_cache(maxsize=1)
def default_client() -> ChatCompletionsClient:
    """預設客戶端。"""
    return ChatCompletionsClient(GenerationConfig())
