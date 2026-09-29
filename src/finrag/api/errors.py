"""API 錯誤型別與錯誤代碼。"""

from __future__ import annotations

INVALID_QUERY = "INVALID_QUERY"
VALIDATION_ERROR = "VALIDATION_ERROR"
LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
LLM_NOT_CONFIGURED = "LLM_NOT_CONFIGURED"
VECTOR_STORE_DISCONNECTED = "VECTOR_STORE_DISCONNECTED"
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiError(Exception):
    """帶 HTTP 狀態碼與錯誤代碼的 API 錯誤。"""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
