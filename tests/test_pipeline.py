"""資料編排單元測試。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fin_nexus.config import (
    FINANCIAL_REPORTS,
    REGULATIONS,
    REGULATIONS_CATALOG,
    REPORTS_CATALOG,
    StoragePaths,
)
from fin_nexus.ingest import pipeline
from fin_nexus.ingest.targets import build_targets
from fin_nexus.models import (
    Block,
    BlockType,
    ParsedDocument,
    ParsedRegulation,
    RegulationArticle,
    RegulationClause,
    RegulationMeta,
    ReportCatalog,
)
from fin_nexus.models.domain import (
    ACQUISITION_DISPOSAL_LAW,
    SECURITIES_LAW,
    LawSpec,
    ReportTarget,
)
from fin_nexus.storage.repository import read_catalog, write_catalog

_TARGETS = build_targets(["2330"], [2023], ["Q1", "Q2"])
_SPECS = (SECURITIES_LAW, ACQUISITION_DISPOSAL_LAW)


def _paths(tmp_path: Path) -> StoragePaths:
    return StoragePaths(data_dir=tmp_path, _env_file=None)


def _catalog(paths: StoragePaths) -> ReportCatalog:
    return read_catalog(paths.catalog_path(REPORTS_CATALOG), ReportCatalog)


def _fake_fetch(calls: list[tuple[str, str, int, str]]) -> object:
    def fetch(client: object, ticker: str, roc_year: int, quarter: str, dest_dir: Path) -> Path:
        calls.append((ticker, "fetch", roc_year, quarter))
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{ticker}_{roc_year + 1911}_{quarter}_consolidated.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        return path

    return fetch


def _fake_parse(calls: list[str]) -> object:
    def parse(
        pdf: Path,
        out_dir: Path,
        meta: object,
        config: object = None,
        runner: object = None,
    ) -> ParsedDocument:
        calls.append(pdf.name)
        return ParsedDocument.model_validate(
            {
                "meta": meta.model_copy(update={"parse_engine": "mineru"}).model_dump(),
                "blocks": [
                    Block(type=BlockType.TEXT, text="現金及約當現金 100", page=0).model_dump()
                ],
            }
        )

    return parse


def test_fetch_reports_records_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    calls: list[tuple[str, str, int, str]] = []
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch(calls))

    catalog = pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)

    assert len(calls) == 2
    entry = catalog.get(_TARGETS[0].doc_id)
    assert entry is not None
    assert entry.status.downloaded is True
    assert entry.sha256 is not None
    assert entry.updated_at is not None
    assert (paths.raw_dir(FINANCIAL_REPORTS) / "2330_2023_Q1_consolidated.pdf").is_file()


def test_fetch_reports_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    calls: list[tuple[str, str, int, str]] = []
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch(calls))

    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)
    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)

    assert len(calls) == 2


def test_fetch_reports_force_refetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    calls: list[tuple[str, str, int, str]] = []
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch(calls))

    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)
    pipeline.fetch_reports(_TARGETS, paths, None, force=True, sleep=lambda _: None)

    assert len(calls) == 4


def test_fetch_reports_sleeps_between_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch([]))
    delays: list[float] = []

    pipeline.fetch_reports(_TARGETS, paths, None, delay=5.0, sleep=delays.append)

    assert delays == [5.0]


def test_parse_documents_writes_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch([]))
    parsed_calls: list[str] = []
    monkeypatch.setattr(pipeline, "parse_pdf", _fake_parse(parsed_calls))
    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)

    catalog = pipeline.parse_documents(_TARGETS, paths)

    assert len(parsed_calls) == 2
    entry = catalog.get(_TARGETS[0].doc_id)
    assert entry is not None
    assert entry.status.parsed is True
    assert paths.parsed_path(FINANCIAL_REPORTS, _TARGETS[0].doc_id).is_file()


def test_parse_documents_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch([]))
    parsed_calls: list[str] = []
    monkeypatch.setattr(pipeline, "parse_pdf", _fake_parse(parsed_calls))
    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)

    pipeline.parse_documents(_TARGETS, paths)
    pipeline.parse_documents(_TARGETS, paths)

    assert len(parsed_calls) == 2


def test_parse_documents_skips_undownloaded(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    catalog = pipeline.parse_documents(_TARGETS, paths)

    assert catalog.get(_TARGETS[0].doc_id) is None
    assert not paths.parsed_path(FINANCIAL_REPORTS, _TARGETS[0].doc_id).is_file()


def test_parse_documents_requires_raw_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch([]))
    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)
    raw = paths.raw_dir(FINANCIAL_REPORTS) / "2330_2023_Q1_consolidated.pdf"
    raw.unlink()

    with pytest.raises(FileNotFoundError, match="找不到財報 PDF"):
        pipeline.parse_documents(_TARGETS, paths)


def test_chunk_documents_writes_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch([]))
    monkeypatch.setattr(pipeline, "parse_pdf", _fake_parse([]))
    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)
    pipeline.parse_documents(_TARGETS, paths)

    catalog = pipeline.chunk_documents(_TARGETS, paths, counter=len)

    entry = catalog.get(_TARGETS[0].doc_id)
    assert entry is not None
    assert entry.status.chunked is True
    assert entry.chunk_count is not None and entry.chunk_count > 0
    assert paths.chunks_path(FINANCIAL_REPORTS, _TARGETS[0].doc_id).is_file()


def test_chunk_documents_resets_indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch([]))
    monkeypatch.setattr(pipeline, "parse_pdf", _fake_parse([]))
    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)
    pipeline.parse_documents(_TARGETS, paths)
    doc_id = _TARGETS[0].doc_id
    catalog = pipeline.chunk_documents(_TARGETS, paths, counter=len)
    entry = catalog.get(doc_id)
    assert entry is not None
    catalog.upsert(
        doc_id,
        entry.model_copy(update={"status": entry.status.model_copy(update={"indexed": True})}),
    )
    write_catalog(paths.catalog_path(REPORTS_CATALOG), catalog)

    catalog = pipeline.chunk_documents(_TARGETS, paths, counter=len, force=True)

    entry = catalog.get(doc_id)
    assert entry is not None
    assert entry.status.chunked is True
    assert entry.status.indexed is False


def test_chunk_documents_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch([]))
    monkeypatch.setattr(pipeline, "parse_pdf", _fake_parse([]))
    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)
    pipeline.parse_documents(_TARGETS, paths)
    pipeline.chunk_documents(_TARGETS, paths, counter=len)
    chunks_file = paths.chunks_path(FINANCIAL_REPORTS, _TARGETS[0].doc_id)
    before = chunks_file.read_bytes()

    pipeline.chunk_documents(_TARGETS, paths, counter=len)

    assert chunks_file.read_bytes() == before


def test_chunk_documents_skips_unparsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_financial_report", _fake_fetch([]))
    pipeline.fetch_reports(_TARGETS, paths, None, sleep=lambda _: None)

    catalog = pipeline.chunk_documents(_TARGETS, paths, counter=len)

    entry = catalog.get(_TARGETS[0].doc_id)
    assert entry is not None
    assert entry.status.chunked is False
    assert not paths.chunks_path(FINANCIAL_REPORTS, _TARGETS[0].doc_id).is_file()


def test_target_meta_uses_list_url() -> None:
    target = ReportTarget("2330", "台積電", 2023, "Q1")

    meta = target.to_meta("https://example.com")

    assert meta.ticker == "2330"


def _fake_fetch_regulation(calls: list[str]) -> object:
    def fetch(client: object, spec: LawSpec, dest_dir: Path) -> Path:
        calls.append(spec.pcode)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"law_{spec.pcode}.html"
        path.write_text("<html></html>", encoding="utf-8")
        return path

    return fetch


def _fake_parse_regulation(calls: list[str]) -> object:
    def parse(html: str, spec: LawSpec) -> ParsedRegulation:
        calls.append(spec.pcode)
        return ParsedRegulation(
            meta=RegulationMeta(
                reg_name=spec.name,
                reg_pcode=spec.pcode,
                revision_date=date(2024, 8, 7),
                source_uri="https://example.com",
                parse_engine="moj_law_html",
            ),
            articles=[
                RegulationArticle(
                    article_no="第 1 條",
                    chapter="第一章總則",
                    clauses=[RegulationClause(level=0, text="為發展國民經濟，並保障投資。")],
                )
            ],
        )

    return parse


def test_fetch_regulations_records_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation(calls))

    catalog = pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)

    assert calls == ["G0400001", "G0400069"]
    entry = catalog.get("law_G0400001")
    assert entry is not None
    assert entry.status.downloaded is True
    assert entry.sha256 is not None
    assert entry.updated_at is not None
    assert (paths.raw_dir(REGULATIONS) / "law_G0400001.html").is_file()


def test_fetch_regulations_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation(calls))

    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)
    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)

    assert len(calls) == 2


def test_fetch_regulations_force_refetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation(calls))

    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)
    pipeline.fetch_regulations(_SPECS, paths, None, force=True, sleep=lambda _: None)

    assert len(calls) == 4


def test_fetch_regulations_sleeps_between_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation([]))
    delays: list[float] = []

    pipeline.fetch_regulations(_SPECS, paths, None, delay=2.0, sleep=delays.append)

    assert delays == [2.0]


def test_parse_regulations_writes_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation([]))
    parsed_calls: list[str] = []
    monkeypatch.setattr(pipeline, "parse_regulation", _fake_parse_regulation(parsed_calls))
    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)

    catalog = pipeline.parse_regulations(_SPECS, paths)

    assert parsed_calls == ["G0400001", "G0400069"]
    entry = catalog.get("law_G0400001")
    assert entry is not None
    assert entry.status.parsed is True
    assert entry.revision_date == date(2024, 8, 7)
    assert paths.structured_path(REGULATIONS, "law_G0400001").is_file()


def test_parse_regulations_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation([]))
    parsed_calls: list[str] = []
    monkeypatch.setattr(pipeline, "parse_regulation", _fake_parse_regulation(parsed_calls))
    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)

    pipeline.parse_regulations(_SPECS, paths)
    pipeline.parse_regulations(_SPECS, paths)

    assert len(parsed_calls) == 2


def test_parse_regulations_skips_undownloaded(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    catalog = pipeline.parse_regulations(_SPECS, paths)

    assert catalog.get("law_G0400001") is None
    assert not paths.structured_path(REGULATIONS, "law_G0400001").is_file()


def test_parse_regulations_requires_raw_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation([]))
    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)
    raw = paths.raw_dir(REGULATIONS) / "law_G0400001.html"
    raw.unlink()

    with pytest.raises(FileNotFoundError, match="找不到法規 HTML"):
        pipeline.parse_regulations(_SPECS, paths)


def test_chunk_regulations_writes_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation([]))
    monkeypatch.setattr(pipeline, "parse_regulation", _fake_parse_regulation([]))
    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)
    pipeline.parse_regulations(_SPECS, paths)

    catalog = pipeline.chunk_regulations(_SPECS, paths, counter=len)

    entry = catalog.get("law_G0400001")
    assert entry is not None
    assert entry.status.chunked is True
    assert entry.chunk_count is not None and entry.chunk_count > 0
    assert paths.chunks_path(REGULATIONS, "law_G0400001").is_file()


def test_chunk_regulations_resets_indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation([]))
    monkeypatch.setattr(pipeline, "parse_regulation", _fake_parse_regulation([]))
    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)
    pipeline.parse_regulations(_SPECS, paths)
    doc_id = "law_G0400001"
    catalog = pipeline.chunk_regulations(_SPECS, paths, counter=len)
    entry = catalog.get(doc_id)
    assert entry is not None
    catalog.upsert(
        doc_id,
        entry.model_copy(update={"status": entry.status.model_copy(update={"indexed": True})}),
    )
    write_catalog(paths.catalog_path(REGULATIONS_CATALOG), catalog)

    catalog = pipeline.chunk_regulations(_SPECS, paths, counter=len, force=True)

    entry = catalog.get(doc_id)
    assert entry is not None
    assert entry.status.chunked is True
    assert entry.status.indexed is False


def test_chunk_regulations_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation([]))
    monkeypatch.setattr(pipeline, "parse_regulation", _fake_parse_regulation([]))
    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)
    pipeline.parse_regulations(_SPECS, paths)
    pipeline.chunk_regulations(_SPECS, paths, counter=len)
    chunks_file = paths.chunks_path(REGULATIONS, "law_G0400001")
    before = chunks_file.read_bytes()

    pipeline.chunk_regulations(_SPECS, paths, counter=len)

    assert chunks_file.read_bytes() == before


def test_chunk_regulations_skips_unparsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline, "fetch_regulation", _fake_fetch_regulation([]))
    pipeline.fetch_regulations(_SPECS, paths, None, sleep=lambda _: None)

    catalog = pipeline.chunk_regulations(_SPECS, paths, counter=len)

    entry = catalog.get("law_G0400001")
    assert entry is not None
    assert entry.status.chunked is False
    assert not paths.chunks_path(REGULATIONS, "law_G0400001").is_file()
