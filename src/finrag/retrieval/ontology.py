"""金融本體庫載入。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from finrag.config import StoragePaths

TICKER_MAP = "ticker_map.json"
IFRS_SYNONYMS = "ifrs_synonyms.json"
FINANCIAL_TERMS = "financial_terms.json"

_ONTOLOGY_PACKAGE = "finrag"
_ONTOLOGY_DIR = "ontology"
_FILES = (TICKER_MAP, IFRS_SYNONYMS, FINANCIAL_TERMS)


def _read(directory: Path | None, name: str) -> dict[str, Any]:
    if directory is not None:
        text = (directory / name).read_text(encoding="utf-8")
    else:
        text = files(_ONTOLOGY_PACKAGE).joinpath(_ONTOLOGY_DIR, name).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"本體檔格式錯誤：{name}")
    return data


@dataclass(frozen=True)
class Ontology:
    """金融本體知識：公司別名、科目同義詞與縮寫。"""

    tickers: dict[str, tuple[str, ...]]
    synonyms: dict[str, tuple[str, ...]]
    terms: dict[str, str]

    @classmethod
    def load(cls, directory: Path | None = None) -> Ontology:
        """載入本體庫；directory 有值時讀取該目錄，否則讀套件內建。"""
        tickers = _read(directory, TICKER_MAP)
        synonyms = _read(directory, IFRS_SYNONYMS)
        terms = _read(directory, FINANCIAL_TERMS)
        return cls(
            tickers={key: tuple(value) for key, value in tickers.items()},
            synonyms={key: tuple(value) for key, value in synonyms.items()},
            terms={key: str(value) for key, value in terms.items()},
        )


def _is_complete(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in _FILES)


@lru_cache(maxsize=1)
def default_ontology() -> Ontology:
    """預設本體庫；`data/ontology/` 具備同名檔時優先使用。"""
    override = StoragePaths().ontology_dir()
    if _is_complete(override):
        return Ontology.load(override)
    return Ontology.load()
