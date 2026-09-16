"""日誌設定。"""

from __future__ import annotations

import logging

from finrag.config import LogConfig


def configure_logging(level: str | None = None) -> None:
    """設定根日誌等級與輸出格式。"""
    resolved = (level or LogConfig().log_level).upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger().setLevel(resolved)
