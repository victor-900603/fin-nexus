"""解析設定與資料路徑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ParseMode = Literal["remote", "local"]

MODES: tuple[ParseMode, ...] = ("remote", "local")
DEFAULT_MODE: ParseMode = "remote"
DEFAULT_TIER = "basic"

FINANCIAL_REPORTS = "financial_reports"
REGULATIONS = "regulations"
REPORTS_CATALOG = "reports_catalog.json"
REGULATIONS_CATALOG = "regulations_catalog.json"

DEFAULT_DATA_DIR = Path("data")

DEFAULT_LOG_LEVEL = "INFO"


class ParseConfig(BaseSettings):
    """MinerU 解析設定：預設遠端，可切換本機。"""

    model_config = SettingsConfigDict(
        env_prefix="FINRAG_MINERU_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: ParseMode = DEFAULT_MODE
    tier: str = DEFAULT_TIER
    api_key: str | None = None
    remote_url: str | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return DEFAULT_MODE
            if text not in MODES:
                raise ValueError(f"不支援的解析模式：{text}")
            return text
        return value

    @field_validator("tier", mode="before")
    @classmethod
    def _normalize_tier(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or DEFAULT_TIER
        return value

    @field_validator("api_key", "remote_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value


class LogConfig(BaseSettings):
    """日誌設定。"""

    model_config = SettingsConfigDict(
        env_prefix="FINRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = DEFAULT_LOG_LEVEL

    @field_validator("log_level", mode="before")
    @classmethod
    def _default_log_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper() or DEFAULT_LOG_LEVEL
        return value


class StoragePaths(BaseSettings):
    """資料落盤路徑：raw、parsed、chunks 與 MinerU 工作目錄。"""

    model_config = SettingsConfigDict(
        env_prefix="FINRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = DEFAULT_DATA_DIR

    @field_validator("data_dir", mode="before")
    @classmethod
    def _default_data_dir(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or DEFAULT_DATA_DIR
        return value

    def raw_dir(self, kind: str) -> Path:
        """原始檔案目錄。"""
        return self.data_dir / "storage" / kind / "raw"

    def parsed_dir(self, kind: str) -> Path:
        """解析成品目錄。"""
        return self.data_dir / "storage" / kind / "parsed"

    def chunks_dir(self, kind: str) -> Path:
        """切片成品目錄。"""
        return self.data_dir / "storage" / kind / "chunks"

    def parsed_path(self, kind: str, doc_id: str) -> Path:
        """單一文件的解析成品檔。"""
        return self.parsed_dir(kind) / f"{doc_id}.json"

    def chunks_path(self, kind: str, doc_id: str) -> Path:
        """單一文件的切片成品檔。"""
        return self.chunks_dir(kind) / f"{doc_id}.jsonl"

    def structured_dir(self, kind: str) -> Path:
        """結構化成品目錄。"""
        return self.data_dir / "storage" / kind / "structured"

    def structured_path(self, kind: str, doc_id: str) -> Path:
        """單一文件的結構化成品檔。"""
        return self.structured_dir(kind) / f"{doc_id}.json"

    def mineru_dir(self, doc_id: str) -> Path:
        """MinerU 原始輸出目錄。"""
        return self.data_dir / "interim" / "mineru" / doc_id

    def ontology_dir(self) -> Path:
        """本體庫目錄。"""
        return self.data_dir / "ontology"

    def catalogs_dir(self) -> Path:
        """清單目錄。"""
        return self.data_dir / "catalogs"

    def catalog_path(self, name: str) -> Path:
        """清單檔路徑。"""
        return self.catalogs_dir() / name
