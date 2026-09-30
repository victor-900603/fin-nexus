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

以環境變數設定，也可寫入工作目錄的 `.env` 檔（`.env` 已被 gitignore，金鑰不會進版控）；`os.environ` 的設定會覆蓋 `.env`。可複製 `.env.example` 作為範本：

- `FINRAG_MINERU_MODE`：解析模式，`remote`（預設）或 `local`。
- `FINRAG_MINERU_TIER`：本機模式使用的 tier（預設 `basic`）。
- `FINRAG_MINERU_API_KEY`：遠端模式所需的 API 金鑰。
- `FINRAG_MINERU_REMOTE_URL`：遠端服務 URL（選填）。
- `FINRAG_DATA_DIR`：資料根目錄（預設 `data`）。

## Qdrant

索引層使用 Qdrant（`qdrant/qdrant:v1.19.1`）。以 Docker 啟動：

```powershell
docker compose up -d qdrant
```

連線設定（見 `.env.example`）：

- `FINRAG_QDRANT_URL`：Qdrant URL（預設 `http://localhost:6333`）。
- `FINRAG_QDRANT_API_KEY`：API 金鑰（選填）。
- `FINRAG_QDRANT_TIMEOUT`：逾時秒數（預設 `10`）。

Collection 為 `fin_reports`（財報）與 `regulations`（法規），各含 `dense`（維度 1024、Cosine）與 `sparse` 向量。

索引資料持久化於 Docker named volume `finrag_qdrant_storage`（存於 WSL2 的 ext4，避免 Windows bind mount 造成 Qdrant optimizer 的 rename 失敗）。備份方式：

```powershell
docker run --rm -v finrag_qdrant_storage:/data -v ${PWD}:/backup alpine tar czf /backup/qdrant_storage.tgz -C /data .
```


## 資料路徑

```text
data/
  storage/
    financial_reports/
      raw/     {doc_id}.pdf
      parsed/  {doc_id}.json
      chunks/  {doc_id}.jsonl
    regulations/
      raw/         law_{pcode}.html
      structured/  law_{pcode}.json
      chunks/      law_{pcode}.jsonl
  ontology/  ticker_map.json / ifrs_synonyms.json / financial_terms.json（選填）
  interim/
    mineru/  {doc_id}/
```

`samples/` 內含少量已解析與切片的成品範例（`{doc_id}.json`、`{doc_id}.chunks.jsonl`）。

## 本體庫

查詢擴展使用金融本體知識（公司別名、科目同義詞、縮寫）。預設隨套件提供於 `src/finrag/ontology/`；若 `data/ontology/` 內具備同名檔，則優先採用。

## 檢索路由

依查詢解析結果自適應選擇檢索路徑：命中法規條號時採雙路混合（Dense + Sparse，Qdrant 原生 RRF，k=60）；命中公司、年度或季度時採預過濾混合；其餘採純 Dense。混合模式對 Dense 與 Sparse 各取候選（上限為 Top-K 的兩倍）後以 RRF 融合，並套用路由產生的過濾條件；Dense 使用完整查詢，Sparse 使用查詢解析抽出的科目詞（含本體同義詞）。啟用重排時以 `BAAI/bge-reranker-v2-m3` 對候選與查詢交叉比對，並將重排分數記為 `score`、原檢索分數保留於 `retrieval_score`。重排後依命中切片的 `parent_id` 取回所屬父區塊作為生成上下文，並以行集合重疊率門檻去除高度重疊的區塊。壓縮階段去除重複語句、無效分頁符與重複表頭，並保留含條號、數值與表格標題的引用片段。

重排預設關閉：`bge-reranker-v2-m3` 對 markdown 數值表格切片幾乎不給分，開啟後會把檢索前段的正確切片壓到後段，使財報數值題的命中率下降。

財報切片於 Payload 記錄表格標題 `table_title`（取自 MinerU 表格標題，無則退回章節路徑），供生成階段標註引用。

## 生成層

以 OpenAI 相容的 Chat Completions 端點生成嚴格引用答案（設定見 `.env.example`）：

- `FINRAG_LLM_BASE_URL`：端點 URL（預設 `http://localhost:8000/v1`）。
- `FINRAG_LLM_API_KEY`：API 金鑰（選填）。
- `FINRAG_LLM_MODEL`：模型名稱（必填，未設定時生成會拋錯）。
- `FINRAG_LLM_TIMEOUT`：逾時秒數（預設 `60`）。
- `FINRAG_LLM_MAX_TOKENS`：回應 token 上限（預設 `1024`）。
- `FINRAG_LLM_TEMPERATURE`：取樣溫度（預設 `0`）。
- `FINRAG_LLM_MIN_SCORE`：拒答分數門檻（預設 `0`，即不啟用）。

生成僅依檢索內容作答並附上引用；檢索結果為空或最高分低於 `FINRAG_LLM_MIN_SCORE` 時回覆「文檔未揭露」。

常見端點範例：

- Groq：`FINRAG_LLM_BASE_URL=https://api.groq.com/openai/v1`、`FINRAG_LLM_API_KEY=gsk_...`、`FINRAG_LLM_MODEL=openai/gpt-oss-120b`。
- Ollama（本機）：`FINRAG_LLM_BASE_URL=http://localhost:11434/v1`、`FINRAG_LLM_API_KEY` 留空、`FINRAG_LLM_MODEL=llama3.1`。

`FINRAG_LLM_BASE_URL` 需包含 `/v1`；`FINRAG_LLM_API_KEY` 留空時不送出 `Authorization` 標頭。

## API

以 FastAPI 提供端點，可透過內建指令啟動：

```powershell
uv run finrag serve --host 127.0.0.1 --port 8000
```

也可直接執行 `uvicorn finrag.api.main:app`；需要其他 uvicorn 選項時請直接使用 `uvicorn`。

- `POST /api/v1/search`：純檢索，回傳結構化片段。
- `POST /api/v1/chat`：檢索後生成嚴格引用答案。
- `GET /health`：健康檢查（僅讀記憶體狀態，不觸發模型載入）。

`search` 與 `chat` 共用的請求欄位：`query`（必填）、`filters`（`ticker`、`fiscal_year`、`quarter`、`report_period`、`reg_pcode`、`article_no`、`chapter`、`doc_id`、`doc_type`）、`search_mode`（`auto`／`dense`／`filtered_hybrid`／`hybrid`，預設 `auto`）、`enable_rerank`（預設 `false`）、`top_k`（最終回傳切片數，預設 `3`）；`chat` 另有 `abstain_threshold`（預設 `0`）。

`search` 回應 `query`、`target_collection`、`search_path_taken`、`results`、`latency_ms`；`chat` 回應 `answer`、`abstained`、`abstain_reason`（`no_hits`／`score_below_threshold`／`not_disclosed`）、`citations`、`latency_ms`。錯誤以 `{"error": {"code", "message"}}` 回傳，代碼為 `INVALID_QUERY`（400）、`VALIDATION_ERROR`（422）、`LLM_PROVIDER_ERROR`（502）、`LLM_NOT_CONFIGURED`（503）、`VECTOR_STORE_DISCONNECTED`（503）、`SERVICE_UNAVAILABLE`（503）。

`chat` 另可設 `stream`（預設 `false`）；設為 `true` 時以 `text/event-stream` 回傳，依序送出 `delta`（`{"text": "..."}`）與 `done`（完整問答回應）事件；串流中途失敗則改送 `error`（`{"code", "message"}`）事件。

## 文件入庫

以 `finrag` 指令將財報與法規入庫（`uv run finrag <command>`）：

```powershell
uv run finrag fetch --ticker 2330 --years 2023 --quarters Q1
uv run finrag parse
uv run finrag chunk
uv run finrag index
uv run finrag all --source all
```

- `fetch`：下載資料（財報 PDF 自公開資訊觀測站；法規 HTML 自全國法規資料庫）。
- `parse`：以 MinerU 解析財報 PDF；法規解析為條／項／款結構。
- `chunk`：產生切片（財報父子切片；法規以條為單位）。
- `index`：以 BGE-M3 產生向量並寫入 Qdrant（首次執行會下載模型）。
- `all`：依序執行 `fetch`、`parse`、`chunk`（不含 `index`）。

共同選項：`--source`（`reports`、`laws`、`all`，預設 `reports`）、`--ticker`（預設 `2330 2303 2454`）、`--years`（西元，預設 `2023 2024 2025`）、`--quarters`（預設 `Q1 Q2 Q3 Q4`）、`--data-dir`、`--force`；`fetch` 另有 `--delay`（財報下載間隔，預設 20 秒）；`index` 另有 `--batch-size`（每批寫入點數，預設 64）。

向量化以 BGE-M3 於 GPU 執行（`torch` 由 PyTorch cu128 索引安裝）；無可用 CUDA 時自動退回 CPU，速度較慢。

## 評估

以 `src/finrag/eval/dataset.json` 的黃金測試集逐題檢索並計算指標：

```powershell
uv run finrag eval
uv run finrag eval --rerank --output report.json
```

輸出 Hit Rate@3、MRR、fact_recall 與拒答正確率摘要，並可將完整報告寫成 JSON。選項：`--dataset`（測試集路徑，預設套件內建）、`--top-k`（預設 `3`）、`--rerank`（啟用重排，預設關閉）、`--output`（報告輸出路徑）。

## 開發指令

```powershell
uv run ruff check .
uv run mypy src
uv run lint-imports
uv run pytest
```
