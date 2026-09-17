"""解析後文字正規化。"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

_CJK = "\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f"
_PUNCT = "\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff5e"
_SPACE = re.compile(rf"(?<=[{_CJK}{_PUNCT}])[ \t\u3000]+(?=[{_CJK}{_PUNCT}])")
_DICT = ("opencc", "dictionary", "STCharacters.txt")


@lru_cache(maxsize=1)
def _table() -> dict[int, str]:
    text = resources.files(_DICT[0]).joinpath(_DICT[1], _DICT[2]).read_text(encoding="utf-8")
    mapping: dict[int, str] = {}
    for line in text.splitlines():
        key, _, values = line.partition("\t")
        targets = values.split()
        if not key or key.startswith("#") or not targets:
            continue
        if key not in targets:
            mapping[ord(key)] = targets[0]
    return str.maketrans(mapping)


def normalize_text(text: str) -> str:
    """去除中日韓字與標點之間的空白，並將簡體字轉為繁體。"""
    return _SPACE.sub("", text).translate(_table())
