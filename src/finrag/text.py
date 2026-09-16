"""共用文字工具：token 計數與依 token 分組。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

TokenCounter = Callable[[str], int]

_TOKENIZER_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_TOKENIZER_NAME)


def default_token_counter() -> TokenCounter:
    """以 BGE-M3 tokenizer 計算 token 數。"""
    tokenizer = _tokenizer()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def pack_by_tokens[Item](
    items: Sequence[Item],
    text_of: Callable[[Item], str],
    counter: TokenCounter,
    max_tokens: int,
    prefix_tokens: int,
) -> list[list[Item]]:
    """依 token 上限將項目分組。"""
    groups: list[list[Item]] = []
    current: list[Item] = []
    total = prefix_tokens
    for item in items:
        tokens = counter(text_of(item))
        if current and total + tokens > max_tokens:
            groups.append(current)
            current = []
            total = prefix_tokens
        current.append(item)
        total += tokens
    if current:
        groups.append(current)
    return groups
