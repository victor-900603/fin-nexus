# FinRAG

金融財報與法規的混合檢索增強檢索系統。

## 環境需求

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## 安裝

```powershell
uv sync
```

## 設定

資料落盤路徑可透過 `FINRAG_DATA_DIR` 覆寫，預設為 `data/`。

## 開發指令

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
```
