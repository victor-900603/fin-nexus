"""生成層單元測試。"""

from __future__ import annotations

import json

import httpx

from finrag.config import GenerationConfig
from finrag.contracts import SearchHit
from finrag.generation.client import ChatCompletionsClient, ChatMessage, default_client
from finrag.generation.generator import (
    abstain_reason,
    build_citations,
    build_messages,
    finalize_answer,
)
from finrag.generation.prompt import ABSTAIN_TEXT, build_context, build_user_prompt


def _hit(score: float = 0.9) -> SearchHit:
    return SearchHit(
        id="point-1",
        score=score,
        collection="fin_reports",
        payload={
            "doc_id": "2330_2024_Q1_consolidated",
            "chunk_id": "2330_2024_Q1_consolidated:p0:c0",
            "text": "現金及約當現金 100",
            "page": 3,
            "table_title": "合併資產負債表",
        },
    )


def test_build_citations_from_hits() -> None:
    citations = build_citations([_hit()])

    assert len(citations) == 1
    citation = citations[0]
    assert citation.index == 1
    assert citation.doc_id == "2330_2024_Q1_consolidated"
    assert citation.chunk_id == "2330_2024_Q1_consolidated:p0:c0"
    assert citation.source == "fin_reports"
    assert citation.page == 3
    assert citation.table_title == "合併資產負債表"
    assert citation.article_no is None
    assert citation.score == 0.9


def test_build_context_labels_and_indexes() -> None:
    context = build_context([_hit(), _hit()])

    assert context.startswith("[1] 2330_2024_Q1_consolidated | 合併資產負債表 | 第 3 頁")
    assert "[2]" in context
    assert "現金及約當現金 100" in context


def test_build_user_prompt_includes_question() -> None:
    prompt = build_user_prompt("現金多少？", "[1] 文件\n現金 100")

    assert "現金多少？" in prompt
    assert "[1] 文件" in prompt


def test_build_messages_includes_system_and_user() -> None:
    messages = build_messages("現金多少？", "[1] 文件\n現金 100")

    assert [message.role for message in messages] == ["system", "user"]
    assert "現金多少？" in messages[1].content


def test_abstain_reason_returns_expected_values() -> None:
    assert abstain_reason([], 0.0) == "no_hits"
    assert abstain_reason([_hit(score=0.5)], 0.8) == "score_below_threshold"
    assert abstain_reason([_hit(score=0.9)], 0.8) is None


def test_finalize_answer_marks_not_disclosed() -> None:
    answer = finalize_answer(ABSTAIN_TEXT, [_hit()])

    assert answer.abstained is True
    assert answer.abstain_reason == "not_disclosed"
    assert len(answer.citations) == 1


def test_finalize_answer_keeps_disclosed_answer() -> None:
    answer = finalize_answer("現金為 100 [1]", [_hit()])

    assert answer.abstained is False
    assert answer.abstain_reason is None
    assert len(answer.citations) == 1


def test_chat_completions_client_posts_and_parses() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "答案 [1]"}}]})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = ChatCompletionsClient(
            GenerationConfig(
                base_url="http://example.com/v1",
                api_key="secret",
                model="test-model",
                _env_file=None,
            ),
            client=http_client,
        )
        result = client.chat([ChatMessage(role="user", content="問題")])

    assert result == "答案 [1]"
    assert captured["url"] == "http://example.com/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "test-model"
    assert body["messages"] == [{"role": "user", "content": "問題"}]


def test_chat_completions_client_streams_deltas() -> None:
    captured: dict[str, object] = {}
    body = (
        'data: {"choices": [{"delta": {"content": "現金"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "為 100"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=body.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = ChatCompletionsClient(
            GenerationConfig(
                base_url="http://example.com/v1",
                model="test-model",
                _env_file=None,
            ),
            client=http_client,
        )
        deltas = list(client.stream([ChatMessage(role="user", content="問題")]))

    assert deltas == ["現金", "為 100"]
    payload = captured["body"]
    assert isinstance(payload, dict)
    assert payload["stream"] is True


def test_chat_completions_client_stream_ignores_malformed_lines() -> None:
    body = (
        "event: message\n"
        ": keep-alive\n\n"
        "data: not-json\n\n"
        'data: {"choices": []}\n\n'
        'data: {"choices": [{"delta": {"content": 5}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = ChatCompletionsClient(
            GenerationConfig(
                base_url="http://example.com/v1",
                model="test-model",
                _env_file=None,
            ),
            client=http_client,
        )
        deltas = list(client.stream([ChatMessage(role="user", content="問題")]))

    assert deltas == ["ok"]


def test_default_client_is_cached() -> None:
    assert default_client() is default_client()
