"""Server-Sent Events 訊息編碼。"""

from __future__ import annotations

import json
from typing import Any

EVENT_DELTA = "delta"
EVENT_DONE = "done"
EVENT_ERROR = "error"


def sse_event(event: str, data: Any) -> str:
    """將事件名稱與資料編碼為單一 SSE 區塊。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
