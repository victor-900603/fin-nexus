"""BGE-Reranker 交叉重排。"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

from finrag.contracts import SearchHit

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 5


class Reranker(Protocol):
    """交叉重排器。"""

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...


class BgeReranker:
    """以 BGE-Reranker 計算查詢與各文件的相關分數。"""

    def __init__(self, model_name: str = RERANK_MODEL, model: Any | None = None) -> None:
        self._model_name = model_name
        self._model = model

    @property
    def is_loaded(self) -> bool:
        """模型是否已載入（不觸發載入）。"""
        return self._model is not None

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from FlagEmbedding import FlagReranker

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = FlagReranker(
                self._model_name, use_fp16=device == "cuda", devices=device
            )
        return self._model

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """回傳每份文件相對查詢的分數。"""
        model = self._load()
        pairs = [[query, document] for document in documents]
        raw = model.compute_score(pairs, normalize=True)
        if isinstance(raw, (int, float)):
            return [float(raw)]
        return [float(score) for score in raw]


@lru_cache(maxsize=1)
def default_reranker() -> BgeReranker:
    """預設重排器。"""
    return BgeReranker()


def rerank(
    hits: Sequence[SearchHit],
    query: str,
    *,
    reranker: Reranker | None = None,
    top_k: int = RERANK_TOP_K,
) -> list[SearchHit]:
    """對候選重新評分並取前 top_k，原檢索分數保留於 retrieval_score。"""
    if not hits:
        return []
    resolved = reranker or default_reranker()
    documents = [str(hit.payload.get("text", "")) for hit in hits]
    scores = resolved.rerank(query, documents)
    ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [
        hit.model_copy(update={"score": score, "retrieval_score": hit.score})
        for hit, score in ranked[:top_k]
    ]
