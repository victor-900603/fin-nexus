"""BGE-M3 向量器單元測試。"""

from __future__ import annotations

from typing import Any

from finrag.storage.embedder import BgeM3Embedder, _to_sparse


class _FakeModel:
    def encode(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        return {
            "dense_vecs": [[0.1, 0.2] for _ in texts],
            "lexical_weights": [{1: 0.5, 2: 1.0} for _ in texts],
        }


def test_to_sparse_converts_weights() -> None:
    sparse = _to_sparse({1: 0.5, 2: 1.0})

    assert sparse.indices == [1, 2]
    assert sparse.values == [0.5, 1.0]


def test_embed_returns_dense_and_sparse() -> None:
    embedder = BgeM3Embedder(model=_FakeModel())

    dense, sparse = embedder.embed(["甲", "乙"])

    assert dense == [[0.1, 0.2], [0.1, 0.2]]
    assert sparse[0].indices == [1, 2]
    assert sparse[1].values == [0.5, 1.0]


def test_is_loaded_reflects_model_state() -> None:
    assert BgeM3Embedder().is_loaded is False
    assert BgeM3Embedder(model=_FakeModel()).is_loaded is True
