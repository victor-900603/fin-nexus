"""檢索與問答服務。"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from finrag.api.errors import (
    LLM_NOT_CONFIGURED,
    LLM_PROVIDER_ERROR,
    VECTOR_STORE_DISCONNECTED,
    ApiError,
)
from finrag.api.health import build_health, qdrant_health
from finrag.api.schema import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    HealthResponse,
)
from finrag.config import GenerationConfig
from finrag.contracts import SearchHit, SearchRequest, SearchResponse, SearchResult
from finrag.generation.client import LLMClient
from finrag.generation.generator import (
    Answer,
    Citation,
    abstain_reason,
    abstained_answer,
    build_messages,
    finalize_answer,
)
from finrag.generation.prompt import build_context
from finrag.retrieval.compactor import compact_text
from finrag.retrieval.dedupe import dedupe
from finrag.retrieval.ontology import Ontology
from finrag.retrieval.pipeline import RetrievalResult, retrieve
from finrag.retrieval.reranker import BgeReranker
from finrag.retrieval.router import RoutePlan
from finrag.storage.embedder import BgeM3Embedder
from finrag.storage.qdrant_client import REGULATIONS_COLLECTION, REPORTS_COLLECTION
from finrag.text import TokenCounter

logger = logging.getLogger(__name__)

_QDRANT_ERRORS = (ResponseHandlingException, UnexpectedResponse, httpx.HTTPError)
_FILE_EXTENSIONS = {
    REPORTS_COLLECTION: ".pdf",
    REGULATIONS_COLLECTION: ".html",
}


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _page(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _file_name(doc_id: object, collection: str) -> str | None:
    if not isinstance(doc_id, str):
        return None
    return f"{doc_id}{_FILE_EXTENSIONS.get(collection, '')}"


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


@dataclass(frozen=True)
class ChatPlan:
    """問答的前置作業結果。"""

    route_plan: RoutePlan
    contexts: list[SearchHit]
    context: str
    config: GenerationConfig
    reason: str | None
    started: float


def _to_result(hit: SearchHit) -> SearchResult:
    payload = hit.payload
    return SearchResult(
        id=hit.id,
        score=hit.score,
        retrieval_score=hit.retrieval_score,
        doc_id=_text(payload.get("doc_id")),
        chunk_id=_text(payload.get("chunk_id")),
        file_name=_file_name(payload.get("doc_id"), hit.collection),
        source=hit.collection,
        page=_page(payload.get("page")),
        table=_text(payload.get("table_title")),
        article_no=_text(payload.get("article_no")),
        text=str(payload.get("text", "")),
    )


def _to_citation(citation: Citation) -> CitationOut:
    return CitationOut(
        index=citation.index,
        doc_id=citation.doc_id,
        chunk_id=citation.chunk_id,
        file_name=_file_name(citation.doc_id, citation.source),
        source=citation.source,
        page=citation.page,
        table=citation.table_title,
        article_no=citation.article_no,
        score=citation.score,
    )


class QueryService:
    """串接查詢解析、路由、檢索、重排、父區塊擴展、壓縮與生成。"""

    def __init__(
        self,
        qdrant: QdrantClient,
        *,
        embedder: BgeM3Embedder,
        reranker: BgeReranker,
        llm: LLMClient,
        ontology: Ontology,
        counter: TokenCounter,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._reranker = reranker
        self._llm = llm
        self._ontology = ontology
        self._generation_config = generation_config or GenerationConfig()
        self._counter = counter

    def search(self, request: SearchRequest) -> SearchResponse:
        """執行檢索並回傳結構化片段。"""
        started = time.perf_counter()
        result = self._retrieve(request)
        results = [_to_result(hit) for hit in result.hits]
        latency = _elapsed_ms(started)
        logger.info(
            "檢索 query=%r mode=%s collection=%s results=%d latency_ms=%d",
            request.query,
            result.plan.mode,
            result.plan.collection,
            len(results),
            latency,
        )
        return SearchResponse(
            query=request.query,
            target_collection=result.plan.collection,
            search_path_taken=result.plan.mode,
            results=results,
            latency_ms=latency,
        )

    def prepare_chat(self, request: ChatRequest) -> ChatPlan:
        """完成檢索、去冗餘與壓縮，供問答與串流問答共用。"""
        started = time.perf_counter()
        result = self._retrieve(request)
        contexts = dedupe(result.hits)
        compaction = compact_text(build_context(contexts), counter=self._counter)
        config = self._generation_config.model_copy(
            update={"min_score": request.abstain_threshold}
        )
        reason = abstain_reason(contexts, config.min_score)
        if reason is None and not config.model:
            raise ApiError(LLM_NOT_CONFIGURED, "需設定生成模型（FINRAG_LLM_MODEL）", 503)
        logger.info(
            "問答前置 query=%r mode=%s collection=%s contexts=%d reason=%s tokens=%d->%d",
            request.query,
            result.plan.mode,
            result.plan.collection,
            len(contexts),
            reason,
            compaction.original_tokens,
            compaction.compacted_tokens,
        )
        return ChatPlan(
            route_plan=result.plan,
            contexts=contexts,
            context=compaction.text,
            config=config,
            reason=reason,
            started=started,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """執行檢索並生成嚴格引用答案。"""
        plan = self.prepare_chat(request)
        answer = self._answer(request.query, plan)
        response = self._build_response(
            request, plan, answer.answer, answer.abstain_reason, answer.citations
        )
        logger.info(
            "問答 query=%r abstained=%s reason=%s citations=%d latency_ms=%d",
            request.query,
            answer.abstained,
            answer.abstain_reason,
            len(response.citations),
            response.latency_ms,
        )
        return response

    def _answer(self, question: str, plan: ChatPlan) -> Answer:
        if plan.reason is not None:
            return abstained_answer(plan.reason)
        try:
            text = self._llm.chat(build_messages(question, plan.context))
        except httpx.HTTPError as error:
            raise ApiError(LLM_PROVIDER_ERROR, str(error), 502) from error
        return finalize_answer(text, plan.contexts)

    def _build_response(
        self,
        request: ChatRequest,
        plan: ChatPlan,
        answer: str,
        reason: str | None,
        citations: Sequence[Citation],
    ) -> ChatResponse:
        return ChatResponse(
            query=request.query,
            target_collection=plan.route_plan.collection,
            search_path_taken=plan.route_plan.mode,
            answer=answer,
            abstained=reason is not None,
            abstain_reason=reason,
            citations=[_to_citation(citation) for citation in citations],
            latency_ms=_elapsed_ms(plan.started),
        )

    def health(self) -> HealthResponse:
        """回報服務狀態；僅讀記憶體狀態，不觸發模型載入。"""
        return build_health(
            qdrant_health(self._qdrant),
            embedding_loaded=self._embedder.is_loaded,
            reranker_loaded=self._reranker.is_loaded,
            model=self._generation_config.model,
        )

    def _retrieve(self, request: SearchRequest) -> RetrievalResult:
        try:
            return retrieve(
                request,
                self._qdrant,
                embedder=self._embedder,
                reranker=self._reranker,
                ontology=self._ontology,
            )
        except _QDRANT_ERRORS as error:
            raise ApiError(VECTOR_STORE_DISCONNECTED, str(error), 503) from error
