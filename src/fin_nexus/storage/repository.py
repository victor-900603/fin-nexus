"""檔案儲存：清單、切片與結構化文件的讀寫。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel


def read_catalog[CatalogT: BaseModel](path: Path, model: type[CatalogT]) -> CatalogT:
    """讀取清單檔，檔案不存在時回傳空清單。"""
    if not path.is_file():
        return model()
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def write_catalog(path: Path, catalog: BaseModel) -> None:
    """寫入清單檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")


def read_json[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    """讀取 JSON 檔。"""
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def write_json(model: BaseModel, path: Path, *, indent: int | None = None) -> Path:
    """寫入 JSON 檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=indent), encoding="utf-8")
    return path


def read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    """讀取 JSONL 檔。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line]


def write_jsonl(records: Sequence[BaseModel], path: Path) -> None:
    """寫入 JSONL 檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")
