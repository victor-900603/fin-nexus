"""嚴格引用生成：依檢索內容產生帶引用的回答。"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from finrag.contracts import SearchHit
from finrag.generation.client import ChatMessage
from finrag.generation.prompt import ABSTAIN_TEXT, SYSTEM_PROMPT, build_user_prompt

ABSTAIN_NO_HITS = "no_hits"
ABSTAIN_LOW_SCORE = "score_below_threshold"
ABSTAIN_NOT_DISCLOSED = "not_disclosed"


class Citation(BaseModel):
    """引用來源。"""

    index: int
    doc_id: str
    chunk_id: str
    source: str
    page: int | None = None
    table_title: str | None = None
    article_no: str | None = None
    score: float


class Answer(BaseModel):
    """生成結果。"""

    answer: str
    citations: list[Citation]
    abstained: bool
    abstain_reason: str | None = None


def _text_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def build_citations(hits: Sequence[SearchHit]) -> list[Citation]:
    """由檢索結果組裝引用清單。"""
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        payload = hit.payload
        page = payload.get("page")
        citations.append(
            Citation(
                index=index,
                doc_id=str(payload.get("doc_id", "")),
                chunk_id=str(payload.get("chunk_id", "")),
                source=hit.collection,
                page=page if isinstance(page, int) else None,
                table_title=_text_or_none(payload.get("table_title")),
                article_no=_text_or_none(payload.get("article_no")),
                score=hit.score,
            )
        )
    return citations


def abstain_reason(hits: Sequence[SearchHit], min_score: float) -> str | None:
    """依檢索結果判斷是否拒答。"""
    if not hits:
        return ABSTAIN_NO_HITS
    if max(hit.score for hit in hits) < min_score:
        return ABSTAIN_LOW_SCORE
    return None


def build_messages(question: str, context: str) -> list[ChatMessage]:
    """組裝送給模型的訊息。"""
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_user_prompt(question, context)),
    ]


def abstained_answer(reason: str) -> Answer:
    """建立拒答結果。"""
    return Answer(answer=ABSTAIN_TEXT, citations=[], abstained=True, abstain_reason=reason)


def finalize_answer(answer: str, hits: Sequence[SearchHit]) -> Answer:
    """由模型輸出與檢索結果組裝答案，並判斷是否屬未揭露。"""
    citations = build_citations(hits)
    if ABSTAIN_TEXT in answer:
        return Answer(
            answer=answer,
            citations=citations,
            abstained=True,
            abstain_reason=ABSTAIN_NOT_DISCLOSED,
        )
    return Answer(answer=answer, citations=citations, abstained=False)
