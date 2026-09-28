"""生成層提示詞組裝。"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.contracts import SearchHit

SYSTEM_PROMPT = (
    "你是金融文件問答助理。僅能依據提供的檢索內容回答，不得自行推測或補充。"
    "每一項數據或法規結論都必須標註來源編號，格式為 [n]。"
    "若提供的內容不足以回答，必須回覆「文檔未揭露」。"
    "回答使用繁體中文，精簡且直接。"
)

ABSTAIN_TEXT = "文檔未揭露"


def _source_label(hit: SearchHit) -> str:
    payload = hit.payload
    parts: list[str] = []
    doc_id = payload.get("doc_id")
    if doc_id is not None:
        parts.append(str(doc_id))
    table_title = payload.get("table_title")
    if table_title is not None:
        parts.append(str(table_title))
    article_no = payload.get("article_no")
    if article_no is not None:
        parts.append(str(article_no))
    page = payload.get("page")
    if isinstance(page, int):
        parts.append(f"第 {page} 頁")
    return " | ".join(parts)


def build_context(hits: Sequence[SearchHit]) -> str:
    """將檢索結果編號並附上中介資料。"""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        text = str(hit.payload.get("text", ""))
        blocks.append(f"[{index}] {_source_label(hit)}\n{text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    """組裝使用者提示。"""
    return f"檢索內容：\n{context}\n\n問題：{question}"
