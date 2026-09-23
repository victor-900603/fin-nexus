"""BGE-M3 向量化。"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

from qdrant_client import models

EMBEDDING_MODEL = "BAAI/bge-m3"
ENCODE_BATCH_SIZE = 16


class Embedder(Protocol):
    """文字轉向量介面。"""

    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        """將文字轉為 dense 與 sparse 向量。"""
        ...


def _to_sparse(weights: dict[Any, Any]) -> models.SparseVector:
    indices = [int(index) for index in weights]
    values = [float(value) for value in weights.values()]
    return models.SparseVector(indices=indices, values=values)


class BgeM3Embedder:
    """以 FlagEmbedding 載入 BGE-M3 產生 dense 與 sparse 向量。"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, model: Any | None = None) -> None:
        self._model_name = model_name
        self._model = model

    @property
    def is_loaded(self) -> bool:
        """模型是否已載入（不觸發載入）。"""
        return self._model is not None

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=device == "cuda",
                devices=device,
            )
        return self._model

    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        output = self._load().encode(
            list(texts),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=ENCODE_BATCH_SIZE,
        )
        dense = [[float(value) for value in vector] for vector in output["dense_vecs"]]
        sparse = [_to_sparse(weights) for weights in output["lexical_weights"]]
        return dense, sparse


@lru_cache(maxsize=1)
def default_embedder() -> BgeM3Embedder:
    """回傳預設的 BGE-M3 向量器。"""
    return BgeM3Embedder()
