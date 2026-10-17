"""文件清單模型測試。"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fin_nexus.models import (
    RegulationCatalog,
    RegulationEntry,
    ReportCatalog,
    ReportEntry,
    ReportPeriod,
    ReportScope,
    StageStatus,
)
from fin_nexus.storage.repository import read_catalog, write_catalog

DOC_ID = "2330_2024_Q1_consolidated"
LAW_DOC_ID = "law_G0400001"


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


def _regulation_entry(**overrides: Any) -> RegulationEntry:
    data: dict[str, Any] = {
        "reg_pcode": "G0400001",
        "reg_name": "證券交易法",
    }
    data.update(overrides)
    return RegulationEntry(**data)


def test_regulation_entry_defaults() -> None:
    entry = _regulation_entry()

    assert entry.revision_date is None
    assert entry.sha256 is None
    assert entry.chunk_count is None
    assert entry.updated_at is None
    assert entry.status.downloaded is False


def test_regulation_catalog_upsert_and_get() -> None:
    catalog = RegulationCatalog()
    entry = _regulation_entry()
    catalog.upsert(LAW_DOC_ID, entry)

    assert catalog.get(LAW_DOC_ID) is entry
    assert catalog.get("missing") is None


def test_regulation_catalog_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "catalogs" / "regulations_catalog.json"
    entry = _regulation_entry(
        revision_date=date(2024, 8, 7),
        sha256="def456",
        status=StageStatus(downloaded=True, parsed=True, chunked=True),
        chunk_count=265,
        updated_at=datetime(2026, 10, 12, 9, 30, 0, tzinfo=UTC),
    )
    catalog = RegulationCatalog()
    catalog.upsert(LAW_DOC_ID, entry)
    write_catalog(path, catalog)

    loaded = read_catalog(path, RegulationCatalog)

    assert loaded == catalog
    assert loaded.get(LAW_DOC_ID) is not None
    assert loaded.entries[LAW_DOC_ID].revision_date == date(2024, 8, 7)
    assert loaded.entries[LAW_DOC_ID].chunk_count == 265


def test_regulation_catalog_load_missing_file_returns_empty(tmp_path: Path) -> None:
    catalog = read_catalog(tmp_path / "nope.json", RegulationCatalog)

    assert catalog.entries == {}
