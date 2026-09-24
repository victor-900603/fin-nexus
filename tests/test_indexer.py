"""入庫單元測試。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

from qdrant_client import QdrantClient, models

from finrag.config import (
    FINANCIAL_REPORTS,
    REGULATIONS,
    REGULATIONS_CATALOG,
    REPORTS_CATALOG,
    StoragePaths,
)
from finrag.models import (
    Chunk,
    ChunkMetadata,
    RegulationCatalog,
    RegulationChunk,
    RegulationChunkMetadata,
    RegulationEntry,
    ReportCatalog,
    ReportEntry,
    ReportPeriod,
    StageStatus,
)
from finrag.storage import indexer
from finrag.storage.qdrant_client import (
    DENSE_SIZE,
    REGULATIONS_COLLECTION,
    REPORTS_COLLECTION,
)
from finrag.storage.repository import read_jsonl, write_catalog

_DOC_ID = "2330_2023_Q1_consolidated"
_LAW_DOC_ID = "law_G0400001"


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        self.calls += 1
        dense = [[0.0] * DENSE_SIZE for _ in texts]
        sparse = [models.SparseVector(indices=[1], values=[0.5]) for _ in texts]
        return dense, sparse


def _paths(tmp_path: Path) -> StoragePaths:
    return StoragePaths(data_dir=tmp_path, _env_file=None)


def _report_chunk(chunk_id: str, parent_id: str | None = None) -> Chunk:
    return Chunk(
        text="現金及約當現金 100",
        metadata=ChunkMetadata(
            doc_id=_DOC_ID,
            chunk_id=chunk_id,
            parent_id=parent_id,
            section="資產負債表",
            is_parent=parent_id is None,
            is_table=False,
            ticker="2330",
            company_name="台積電",
            fiscal_year=2023,
            quarter="Q1",
            report_period=ReportPeriod.QUARTERLY,
        ),
    )


def _write_reports(paths: StoragePaths, chunks: list[Chunk]) -> None:
    path = paths.chunks_path(FINANCIAL_REPORTS, _DOC_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunk.model_dump_json() for chunk in chunks), encoding="utf-8")
    catalog = ReportCatalog()
    catalog.upsert(
        _DOC_ID,
        ReportEntry(
            ticker="2330",
            fiscal_year=2023,
            quarter="Q1",
            report_period=ReportPeriod.QUARTERLY,
            status=StageStatus(downloaded=True, parsed=True, chunked=True),
            chunk_count=len(chunks),
        ),
    )
    write_catalog(paths.catalog_path(REPORTS_CATALOG), catalog)


def _regulation_chunk(chunk_id: str, parent_id: str | None = None) -> RegulationChunk:
    return RegulationChunk(
        text="[證券交易法 > 第一章總則 > 第 1 條]\n為發展國民經濟。",
        metadata=RegulationChunkMetadata(
            doc_id=_LAW_DOC_ID,
            chunk_id=chunk_id,
            parent_id=parent_id,
            section="第一章總則",
            is_parent=parent_id is None,
            reg_name="證券交易法",
            reg_pcode="G0400001",
            article_no=None if parent_id is None else "第 1 條",
            chapter="第一章總則",
            revision_date=date(2024, 8, 7),
        ),
    )


def _write_regulations(paths: StoragePaths, chunks: list[RegulationChunk]) -> None:
    path = paths.chunks_path(REGULATIONS, _LAW_DOC_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunk.model_dump_json() for chunk in chunks), encoding="utf-8")
    catalog = RegulationCatalog()
    catalog.upsert(
        _LAW_DOC_ID,
        RegulationEntry(
            reg_pcode="G0400001",
            reg_name="證券交易法",
            revision_date=date(2024, 8, 7),
            status=StageStatus(downloaded=True, parsed=True, chunked=True),
            chunk_count=len(chunks),
        ),
    )
    write_catalog(paths.catalog_path(REGULATIONS_CATALOG), catalog)


def test_load_chunks_reads_jsonl(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_reports(paths, [_report_chunk("p0"), _report_chunk("p0:c0", "p0")])

    chunks = read_jsonl(paths.chunks_path(FINANCIAL_REPORTS, _DOC_ID), Chunk)

    assert [chunk.metadata.chunk_id for chunk in chunks] == ["p0", "p0:c0"]


def test_index_reports_writes_points(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_reports(paths, [_report_chunk("p0"), _report_chunk("p0:c0", "p0")])
    client = QdrantClient(":memory:")

    catalog =     indexer.index_reports([_DOC_ID], paths, client, embedder=_FakeEmbedder())

    assert client.count(REPORTS_COLLECTION).count == 2
    entry = catalog.get(_DOC_ID)
    assert entry is not None
    assert entry.status.indexed is True
    assert entry.updated_at is not None
    records, _ = client.scroll(REPORTS_COLLECTION, limit=10, with_payload=True)
    payload = records[0].payload or {}
    assert payload["text"] == "現金及約當現金 100"
    assert payload["doc_type"] == "financial_report"


def test_index_reports_is_idempotent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_reports(paths, [_report_chunk("p0")])
    client = QdrantClient(":memory:")
    embedder = _FakeEmbedder()

    indexer.index_reports([_DOC_ID], paths, client, embedder=embedder)
    indexer.index_reports([_DOC_ID], paths, client, embedder=embedder)

    assert embedder.calls == 1
    assert client.count(REPORTS_COLLECTION).count == 1


def test_index_reports_force_reindexes(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_reports(paths, [_report_chunk("p0")])
    client = QdrantClient(":memory:")
    embedder = _FakeEmbedder()

    indexer.index_reports([_DOC_ID], paths, client, embedder=embedder)
    indexer.index_reports([_DOC_ID], paths, client, embedder=embedder, force=True)

    assert embedder.calls == 2


def test_index_reports_skips_unchunked(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    client = QdrantClient(":memory:")
    embedder = _FakeEmbedder()

    indexer.index_reports([_DOC_ID], paths, client, embedder=embedder)

    assert embedder.calls == 0
    assert client.count(REPORTS_COLLECTION).count == 0


def test_index_reports_batches(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_reports(paths, [_report_chunk("p0"), _report_chunk("p0:c0", "p0")])
    client = QdrantClient(":memory:")

    indexer.index_reports([_DOC_ID], paths, client, embedder=_FakeEmbedder(), batch_size=1)

    assert client.count(REPORTS_COLLECTION).count == 2


def test_index_reports_removes_stale_points(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_reports(paths, [_report_chunk("p0"), _report_chunk("p0:c0", "p0")])
    client = QdrantClient(":memory:")
    embedder = _FakeEmbedder()

    indexer.index_reports([_DOC_ID], paths, client, embedder=embedder)
    assert client.count(REPORTS_COLLECTION).count == 2

    _write_reports(paths, [_report_chunk("p0")])
    indexer.index_reports([_DOC_ID], paths, client, embedder=embedder, force=True)

    assert client.count(REPORTS_COLLECTION).count == 1
    records, _ = client.scroll(REPORTS_COLLECTION, limit=10, with_payload=True)
    assert (records[0].payload or {})["chunk_id"] == "p0"


def test_index_regulations_writes_points(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_regulations(paths, [_regulation_chunk("p0"), _regulation_chunk("p0:c0", "p0")])
    client = QdrantClient(":memory:")

    catalog = indexer.index_regulations(
        [_LAW_DOC_ID], paths, client, embedder=_FakeEmbedder()
    )

    assert client.count(REGULATIONS_COLLECTION).count == 2
    entry = catalog.get(_LAW_DOC_ID)
    assert entry is not None
    assert entry.status.indexed is True
    records, _ = client.scroll(REGULATIONS_COLLECTION, limit=10, with_payload=True)
    payload = records[0].payload or {}
    assert payload["reg_pcode"] == "G0400001"
    assert payload["doc_type"] == "regulation"


def test_index_regulations_skips_unchunked(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    client = QdrantClient(":memory:")
    embedder = _FakeEmbedder()

    indexer.index_regulations([_LAW_DOC_ID], paths, client, embedder=embedder)

    assert embedder.calls == 0
    assert client.count(REGULATIONS_COLLECTION).count == 0


def test_index_regulations_removes_stale_points(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_regulations(paths, [_regulation_chunk("p0"), _regulation_chunk("p0:c0", "p0")])
    client = QdrantClient(":memory:")
    embedder = _FakeEmbedder()

    indexer.index_regulations([_LAW_DOC_ID], paths, client, embedder=embedder)
    assert client.count(REGULATIONS_COLLECTION).count == 2

    _write_regulations(paths, [_regulation_chunk("p0")])
    indexer.index_regulations([_LAW_DOC_ID], paths, client, embedder=embedder, force=True)

    assert client.count(REGULATIONS_COLLECTION).count == 1
    records, _ = client.scroll(REGULATIONS_COLLECTION, limit=10, with_payload=True)
    assert (records[0].payload or {})["chunk_id"] == "p0"
