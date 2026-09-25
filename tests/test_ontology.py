"""本體庫載入測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finrag.retrieval.ontology import (
    FINANCIAL_TERMS,
    IFRS_SYNONYMS,
    TICKER_MAP,
    Ontology,
    default_ontology,
)

_TICKERS = {"2330": ["台積電", "TSMC"]}
_SYNONYMS = {"營業收入淨額": ["營收"]}
_TERMS = {"CAPEX": "資本支出"}


def _write(
    directory: Path,
    tickers: dict[str, list[str]] = _TICKERS,
    synonyms: dict[str, list[str]] = _SYNONYMS,
    terms: dict[str, str] = _TERMS,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TICKER_MAP).write_text(json.dumps(tickers), encoding="utf-8")
    (directory / IFRS_SYNONYMS).write_text(json.dumps(synonyms), encoding="utf-8")
    (directory / FINANCIAL_TERMS).write_text(json.dumps(terms), encoding="utf-8")


def test_load_from_directory(tmp_path: Path) -> None:
    _write(tmp_path)

    ontology = Ontology.load(tmp_path)

    assert ontology.tickers["2330"] == ("台積電", "TSMC")
    assert ontology.synonyms["營業收入淨額"] == ("營收",)
    assert ontology.terms["CAPEX"] == "資本支出"


def test_default_ontology_reads_package() -> None:
    default_ontology.cache_clear()

    ontology = default_ontology()

    assert "2330" in ontology.tickers
    assert ontology.terms["CAPEX"] == "資本支出"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Ontology.load(tmp_path)


def test_default_ontology_prefers_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINRAG_DATA_DIR", str(tmp_path))
    _write(tmp_path / "ontology", tickers={"9999": ["測試公司"]})
    default_ontology.cache_clear()

    ontology = default_ontology()

    assert "9999" in ontology.tickers
    assert "2330" not in ontology.tickers
    default_ontology.cache_clear()
