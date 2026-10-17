"""解析設定單元測試。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from fin_nexus.config import (
    FINANCIAL_REPORTS,
    REGULATIONS,
    REPORTS_CATALOG,
    LogConfig,
    ParseConfig,
    QdrantConfig,
    StoragePaths,
)
from fin_nexus.logging import configure_logging

_FIN_NEXUS_ENV = (
    "FIN_NEXUS_MINERU_MODE",
    "FIN_NEXUS_MINERU_TIER",
    "FIN_NEXUS_MINERU_API_KEY",
    "FIN_NEXUS_MINERU_REMOTE_URL",
    "FIN_NEXUS_DATA_DIR",
    "FIN_NEXUS_QDRANT_URL",
    "FIN_NEXUS_QDRANT_API_KEY",
    "FIN_NEXUS_QDRANT_TIMEOUT",
    "FIN_NEXUS_LOG_LEVEL",
)


def _clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _FIN_NEXUS_ENV:
        monkeypatch.delenv(key, raising=False)


def test_defaults_use_remote_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_environment(monkeypatch)

    config = ParseConfig(_env_file=None)

    assert config.mode == "remote"
    assert config.tier == "basic"
    assert config.api_key is None
    assert config.remote_url is None


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIN_NEXUS_MINERU_MODE", "local")
    monkeypatch.setenv("FIN_NEXUS_MINERU_TIER", "standard")
    monkeypatch.setenv("FIN_NEXUS_MINERU_API_KEY", " secret-token ")
    monkeypatch.setenv("FIN_NEXUS_MINERU_REMOTE_URL", "https://example.com/api")

    config = ParseConfig(_env_file=None)

    assert config.mode == "local"
    assert config.tier == "standard"
    assert config.api_key == "secret-token"
    assert config.remote_url == "https://example.com/api"


def test_blank_environment_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIN_NEXUS_MINERU_MODE", "  ")
    monkeypatch.setenv("FIN_NEXUS_MINERU_TIER", "  ")
    monkeypatch.setenv("FIN_NEXUS_MINERU_API_KEY", "")

    config = ParseConfig(_env_file=None)

    assert config.mode == "remote"
    assert config.tier == "basic"
    assert config.api_key is None


def test_rejects_unknown_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIN_NEXUS_MINERU_MODE", "cloud")

    with pytest.raises(ValueError):
        ParseConfig(_env_file=None)


def test_reads_dotenv_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "# 註解\n"
        "FIN_NEXUS_MINERU_MODE=local\n"
        "FIN_NEXUS_MINERU_TIER='standard'\n"
        'FIN_NEXUS_MINERU_API_KEY="secret-token"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = ParseConfig()

    assert config.mode == "local"
    assert config.tier == "standard"
    assert config.api_key == "secret-token"


def test_environment_overrides_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("FIN_NEXUS_MINERU_TIER=standard\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FIN_NEXUS_MINERU_TIER", "advanced")

    assert ParseConfig().tier == "advanced"


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
    monkeypatch.setenv("FIN_NEXUS_DATA_DIR", "  custom  ")

    assert StoragePaths(_env_file=None).data_dir == Path("custom")


def test_storage_blank_data_dir_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIN_NEXUS_DATA_DIR", "  ")

    assert StoragePaths(_env_file=None).data_dir == Path("data")


def test_qdrant_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_environment(monkeypatch)

    config = QdrantConfig(_env_file=None)

    assert config.url == "http://localhost:6333"
    assert config.api_key is None
    assert config.timeout == 10


def test_qdrant_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIN_NEXUS_QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("FIN_NEXUS_QDRANT_API_KEY", " secret ")
    monkeypatch.setenv("FIN_NEXUS_QDRANT_TIMEOUT", "30")

    config = QdrantConfig(_env_file=None)

    assert config.url == "http://qdrant:6333"
    assert config.api_key == "secret"
    assert config.timeout == 30


def test_qdrant_blank_values_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIN_NEXUS_QDRANT_URL", "  ")
    monkeypatch.setenv("FIN_NEXUS_QDRANT_API_KEY", "")

    config = QdrantConfig(_env_file=None)

    assert config.url == "http://localhost:6333"
    assert config.api_key is None


def test_log_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_environment(monkeypatch)

    assert LogConfig(_env_file=None).log_level == "INFO"


def test_log_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIN_NEXUS_LOG_LEVEL", " debug ")

    assert LogConfig(_env_file=None).log_level == "DEBUG"


def test_log_config_blank_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIN_NEXUS_LOG_LEVEL", "  ")

    assert LogConfig(_env_file=None).log_level == "INFO"


def test_configure_logging_sets_root_level() -> None:
    configure_logging("WARNING")

    assert logging.getLogger().level == logging.WARNING
