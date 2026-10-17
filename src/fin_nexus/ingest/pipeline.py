"""資料編排：以清單冪等地執行財報與法規的擷取、解析與切片。"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from fin_nexus.config import (
    FINANCIAL_REPORTS,
    REGULATIONS,
    REGULATIONS_CATALOG,
    REPORTS_CATALOG,
    ParseConfig,
    StoragePaths,
)
from fin_nexus.ingest.chunker import ChunkConfig, chunk_document
from fin_nexus.ingest.law_chunker import chunk_regulation
from fin_nexus.ingest.law_parser import parse_regulation
from fin_nexus.ingest.mops import fetch_financial_report, list_url, report_filename
from fin_nexus.ingest.parser import Runner, parse_pdf
from fin_nexus.ingest.regulations import fetch_regulation, regulation_filename
from fin_nexus.models import (
    ParsedDocument,
    ParsedRegulation,
    RegulationCatalog,
    RegulationEntry,
    ReportCatalog,
    ReportEntry,
    StageStatus,
)
from fin_nexus.models.document import regulation_id
from fin_nexus.models.domain import LawSpec, ReportTarget
from fin_nexus.storage.repository import (
    read_catalog,
    read_json,
    write_catalog,
    write_json,
    write_jsonl,
)
from fin_nexus.text import TokenCounter


def _catalog_path(paths: StoragePaths) -> Path:
    return paths.catalog_path(REPORTS_CATALOG)


def _regulation_catalog_path(paths: StoragePaths) -> Path:
    return paths.catalog_path(REGULATIONS_CATALOG)


def _now() -> datetime:
    return datetime.now(UTC)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(
    target: ReportTarget,
    status: StageStatus,
    *,
    sha256: str | None = None,
    chunk_count: int | None = None,
) -> ReportEntry:
    return ReportEntry(
        ticker=target.ticker,
        fiscal_year=target.fiscal_year,
        quarter=target.quarter,
        report_period=target.report_period,
        sha256=sha256,
        status=status,
        chunk_count=chunk_count,
        updated_at=_now(),
    )


def _regulation_entry(
    spec: LawSpec,
    status: StageStatus,
    *,
    revision_date: date | None = None,
    sha256: str | None = None,
    chunk_count: int | None = None,
) -> RegulationEntry:
    return RegulationEntry(
        reg_pcode=spec.pcode,
        reg_name=spec.name,
        revision_date=revision_date,
        sha256=sha256,
        status=status,
        chunk_count=chunk_count,
        updated_at=_now(),
    )


def fetch_reports(
    targets: Sequence[ReportTarget],
    paths: StoragePaths,
    client: httpx.Client,
    *,
    force: bool = False,
    delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> ReportCatalog:
    """下載尚未取得的財報 PDF，並更新清單。"""
    catalog = read_catalog(_catalog_path(paths), ReportCatalog)
    raw_dir = paths.raw_dir(FINANCIAL_REPORTS)
    fetched = False
    for target in targets:
        entry = catalog.get(target.doc_id)
        if entry is not None and entry.status.downloaded and not force:
            continue
        if fetched:
            sleep(delay)
        path = fetch_financial_report(
            client, target.ticker, target.roc_year, target.quarter, raw_dir
        )
        status = StageStatus(downloaded=True)
        catalog.upsert(target.doc_id, _entry(target, status, sha256=_sha256(path)))
        write_catalog(_catalog_path(paths), catalog)
        fetched = True
    return catalog


def parse_documents(
    targets: Sequence[ReportTarget],
    paths: StoragePaths,
    *,
    config: ParseConfig | None = None,
    runner: Runner | None = None,
    force: bool = False,
) -> ReportCatalog:
    """解析尚未解析的財報 PDF，並更新清單。"""
    catalog = read_catalog(_catalog_path(paths), ReportCatalog)
    raw_dir = paths.raw_dir(FINANCIAL_REPORTS)
    for target in targets:
        entry = catalog.get(target.doc_id)
        if entry is None or not entry.status.downloaded:
            continue
        if entry.status.parsed and not force:
            continue
        pdf = raw_dir / report_filename(target.ticker, target.fiscal_year, target.quarter)
        if not pdf.is_file():
            raise FileNotFoundError(f"找不到財報 PDF：{pdf}")
        meta = target.to_meta(list_url(target.ticker, target.roc_year))
        document = parse_pdf(pdf, paths.mineru_dir(target.doc_id), meta, config, runner)
        out = paths.parsed_path(FINANCIAL_REPORTS, target.doc_id)
        write_json(document, out)
        status = entry.status.model_copy(update={"parsed": True})
        catalog.upsert(
            target.doc_id,
            entry.model_copy(update={"status": status, "updated_at": _now()}),
        )
        write_catalog(_catalog_path(paths), catalog)
    return catalog


def chunk_documents(
    targets: Sequence[ReportTarget],
    paths: StoragePaths,
    *,
    counter: TokenCounter,
    config: ChunkConfig | None = None,
    force: bool = False,
) -> ReportCatalog:
    """切片尚未切片的財報解析結果，並更新清單。"""
    catalog = read_catalog(_catalog_path(paths), ReportCatalog)
    for target in targets:
        entry = catalog.get(target.doc_id)
        if entry is None or not entry.status.parsed:
            continue
        if entry.status.chunked and not force:
            continue
        parsed = paths.parsed_path(FINANCIAL_REPORTS, target.doc_id)
        document = read_json(parsed, ParsedDocument)
        chunks = chunk_document(document, counter, config)
        write_jsonl(chunks, paths.chunks_path(FINANCIAL_REPORTS, target.doc_id))
        status = entry.status.model_copy(update={"chunked": True, "indexed": False})
        catalog.upsert(
            target.doc_id,
            entry.model_copy(
                update={
                    "status": status,
                    "chunk_count": len(chunks),
                    "updated_at": _now(),
                }
            ),
        )
        write_catalog(_catalog_path(paths), catalog)
    return catalog


def fetch_regulations(
    specs: Sequence[LawSpec],
    paths: StoragePaths,
    client: httpx.Client,
    *,
    force: bool = False,
    delay: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
) -> RegulationCatalog:
    """下載尚未取得的法規全文，並更新清單。"""
    catalog = read_catalog(_regulation_catalog_path(paths), RegulationCatalog)
    raw_dir = paths.raw_dir(REGULATIONS)
    fetched = False
    for spec in specs:
        doc_id = regulation_id(spec.pcode)
        entry = catalog.get(doc_id)
        if entry is not None and entry.status.downloaded and not force:
            continue
        if fetched:
            sleep(delay)
        path = fetch_regulation(client, spec, raw_dir)
        status = StageStatus(downloaded=True)
        catalog.upsert(doc_id, _regulation_entry(spec, status, sha256=_sha256(path)))
        write_catalog(_regulation_catalog_path(paths), catalog)
        fetched = True
    return catalog


def parse_regulations(
    specs: Sequence[LawSpec],
    paths: StoragePaths,
    *,
    force: bool = False,
) -> RegulationCatalog:
    """解析尚未解析的法規全文，並更新清單。"""
    catalog = read_catalog(_regulation_catalog_path(paths), RegulationCatalog)
    raw_dir = paths.raw_dir(REGULATIONS)
    for spec in specs:
        doc_id = regulation_id(spec.pcode)
        entry = catalog.get(doc_id)
        if entry is None or not entry.status.downloaded:
            continue
        if entry.status.parsed and not force:
            continue
        raw = raw_dir / regulation_filename(spec.pcode)
        if not raw.is_file():
            raise FileNotFoundError(f"找不到法規 HTML：{raw}")
        document = parse_regulation(raw.read_text(encoding="utf-8"), spec)
        write_json(document, paths.structured_path(REGULATIONS, doc_id), indent=2)
        status = entry.status.model_copy(update={"parsed": True})
        catalog.upsert(
            doc_id,
            entry.model_copy(
                update={
                    "status": status,
                    "revision_date": document.meta.revision_date,
                    "updated_at": _now(),
                }
            ),
        )
        write_catalog(_regulation_catalog_path(paths), catalog)
    return catalog


def chunk_regulations(
    specs: Sequence[LawSpec],
    paths: StoragePaths,
    *,
    counter: TokenCounter,
    config: ChunkConfig | None = None,
    force: bool = False,
) -> RegulationCatalog:
    """切片尚未切片的法規結構，並更新清單。"""
    catalog = read_catalog(_regulation_catalog_path(paths), RegulationCatalog)
    for spec in specs:
        doc_id = regulation_id(spec.pcode)
        entry = catalog.get(doc_id)
        if entry is None or not entry.status.parsed:
            continue
        if entry.status.chunked and not force:
            continue
        structured = paths.structured_path(REGULATIONS, doc_id)
        document = read_json(structured, ParsedRegulation)
        chunks = chunk_regulation(document, counter, config)
        write_jsonl(chunks, paths.chunks_path(REGULATIONS, doc_id))
        status = entry.status.model_copy(update={"chunked": True, "indexed": False})
        catalog.upsert(
            doc_id,
            entry.model_copy(
                update={
                    "status": status,
                    "chunk_count": len(chunks),
                    "updated_at": _now(),
                }
            ),
        )
        write_catalog(_regulation_catalog_path(paths), catalog)
    return catalog
