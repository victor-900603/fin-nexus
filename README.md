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

## MinerU 解析

預設以遠端模式呼叫 MinerU 解析財報 PDF，可透過下列環境變數設定：

- `FINRAG_MINERU_MODE`：`remote`（預設）或 `local`
- `FINRAG_MINERU_TIER`：本機模式使用的模型階層，預設 `basic`
- `FINRAG_MINERU_API_KEY`：遠端模式所需的 API 金鑰
- `FINRAG_MINERU_REMOTE_URL`：遠端服務位址

## 開發指令

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
```
