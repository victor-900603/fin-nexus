"""文件清單模型測試。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finrag.models import (
    ReportCatalog,
    ReportEntry,
    ReportPeriod,
    ReportScope,
    StageStatus,
)
from finrag.storage.repository import read_catalog, write_catalog

DOC_ID = "2330_2024_Q1_consolidated"


def _entry(**overrides: Any) -> ReportEntry:
    data: dict[str, Any] = {
        "ticker": "2330",
        "fiscal_year": 2024,
        "quarter": "Q1",
        "report_period": ReportPeriod.QUARTERLY,
    }
    data.update(overrides)
    return ReportEntry(**data)


def test_stage_status_defaults() -> None:
    status = StageStatus()

    assert status.downloaded is False
    assert status.parsed is False
    assert status.chunked is False
    assert status.indexed is False


def test_entry_defaults() -> None:
    entry = _entry()

    assert entry.report_scope is ReportScope.CONSOLIDATED
    assert entry.sha256 is None
    assert entry.chunk_count is None
    assert entry.updated_at is None


def test_catalog_upsert_and_get() -> None:
    catalog = ReportCatalog()
    entry = _entry()
    catalog.upsert(DOC_ID, entry)

    assert catalog.get(DOC_ID) is entry
    assert catalog.get("missing") is None


def test_catalog_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "catalogs" / "reports_catalog.json"
    entry = _entry(
        sha256="abc123",
        status=StageStatus(downloaded=True, parsed=True),
        chunk_count=86,
        updated_at=datetime(2026, 10, 8, 10, 15, 30, tzinfo=UTC),
    )
    catalog = ReportCatalog()
    catalog.upsert(DOC_ID, entry)
    write_catalog(path, catalog)

    loaded = read_catalog(path, ReportCatalog)

    assert loaded == catalog
    assert loaded.get(DOC_ID) is not None
    assert loaded.entries[DOC_ID].chunk_count == 86


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    catalog = read_catalog(tmp_path / "nope.json", ReportCatalog)

    assert catalog.entries == {}
