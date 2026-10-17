"""切片模型單元測試。"""

from __future__ import annotations

from fin_nexus.models import Chunk, ChunkMetadata, ReportPeriod


def _metadata(**overrides: object) -> ChunkMetadata:
    base: dict[str, object] = {
        "doc_id": "2330_2024_Q1_consolidated",
        "chunk_id": "2330_2024_Q1_consolidated:p0",
        "section": "合併資產負債表",
        "is_parent": True,
        "is_table": False,
        "ticker": "2330",
        "company_name": "台積電",
        "fiscal_year": 2024,
        "quarter": "Q1",
        "report_period": ReportPeriod.QUARTERLY,
    }
    base.update(overrides)
    return ChunkMetadata(**base)


def test_chunk_metadata_defaults() -> None:
    metadata = _metadata()
    assert metadata.parent_id is None
    assert metadata.page is None


def test_chunk_holds_text_and_metadata() -> None:
    chunk = Chunk(text="內容", metadata=_metadata())
    assert chunk.text == "內容"
    assert chunk.metadata.chunk_id.endswith(":p0")
