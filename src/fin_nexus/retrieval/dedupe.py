"""檢索結果去冗餘：以行集合重疊率門檻去除高度重疊的區塊。"""

from __future__ import annotations

from collections.abc import Sequence

from fin_nexus.contracts import SearchHit

DEFAULT_THRESHOLD = 0.8


def _lines(hit: SearchHit) -> set[str]:
    text = str(hit.payload.get("text", ""))
    return {line.strip() for line in text.splitlines() if line.strip()}


def dedupe(
    hits: Sequence[SearchHit], *, threshold: float = DEFAULT_THRESHOLD
) -> list[SearchHit]:
    """去除與已保留區塊高度重疊的候選，維持原順序。"""
    kept: list[SearchHit] = []
    kept_lines: list[set[str]] = []
    for hit in hits:
        lines = _lines(hit)
        if lines and any(
            len(lines & previous) / len(lines) >= threshold for previous in kept_lines
        ):
            continue
        kept.append(hit)
        kept_lines.append(lines)
    return kept
