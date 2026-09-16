"""解析設定單元測試。"""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag.config import (
    FINANCIAL_REPORTS,
    REGULATIONS,
    REPORTS_CATALOG,
    StoragePaths,
)

_FINRAG_ENV = ("FINRAG_DATA_DIR",)


def _clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _FINRAG_ENV:
        monkeypatch.delenv(key, raising=False)


def test_storage_defaults_use_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_environment(monkeypatch)

    assert StoragePaths(_env_file=None).data_dir == Path("data")


def test_storage_composes_document_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = StoragePaths(data_dir=Path("base"), _env_file=None)

    assert paths.raw_dir(FINANCIAL_REPORTS) == Path("base/storage/financial_reports/raw")
    assert paths.parsed_path(FINANCIAL_REPORTS, "doc1") == Path(
        "base/storage/financial_reports/parsed/doc1.json"
    )
    assert paths.chunks_path(FINANCIAL_REPORTS, "doc1") == Path(
        "base/storage/financial_reports/chunks/doc1.jsonl"
    )
    assert paths.mineru_dir("doc1") == Path("base/interim/mineru/doc1")


def test_storage_composes_catalog_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = StoragePaths(data_dir=Path("base"), _env_file=None)

    assert paths.catalogs_dir() == Path("base/catalogs")
    assert paths.catalog_path(REPORTS_CATALOG) == Path("base/catalogs/reports_catalog.json")


def test_storage_composes_structured_path(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = StoragePaths(data_dir=Path("base"), _env_file=None)

    assert paths.structured_dir(REGULATIONS) == Path("base/storage/regulations/structured")
    assert paths.structured_path(REGULATIONS, "law_G0400001") == Path(
        "base/storage/regulations/structured/law_G0400001.json"
    )


def test_storage_reads_data_dir_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINRAG_DATA_DIR", "  custom  ")

    assert StoragePaths(_env_file=None).data_dir == Path("custom")


def test_storage_blank_data_dir_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINRAG_DATA_DIR", "  ")

    assert StoragePaths(_env_file=None).data_dir == Path("data")
