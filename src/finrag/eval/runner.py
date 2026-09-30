"""黃金測試集評估：檢索指標計算與報告產出。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from finrag.contracts import (
    DEFAULT_TOP_K,
    QueryFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

DATASET_NAME = "dataset.json"
NO_ANSWER_CATEGORY = "no_answer"


class Expect(BaseModel):
    """單一題目的預期檢索結果；未填欄位不納入判定。"""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = None
    article_no: str | None = None
    section: str | None = None
    facts: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    """黃金測試集的一題。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    expect: Expect


class EvalDataset(BaseModel):
    """黃金測試集。"""

    model_config = ConfigDict(extra="forbid")

    version: str
    cases: list[EvalCase]


class CaseResult(BaseModel):
    """單一題目的評估結果。"""

    id: str
    category: str
    passed: bool
    rank: int | None
    matched_facts: list[str]
    missing_facts: list[str]
    top_doc_ids: list[str | None]
    top_sections: list[str]
    latency_ms: int


class EvalReport(BaseModel):
    """整體評估報告。"""

    dataset: str
    top_k: int
    enable_rerank: bool
    total: int
    passed: int
    hit_rate: float
    hit_rate_at_3: float
    mrr: float
    fact_recall: float
    no_answer_accuracy: float
    results: list[CaseResult]


class Searcher(Protocol):
    """評估所需的最小檢索介面。"""

    def search(self, request: SearchRequest) -> SearchResponse: ...


def load_dataset(path: Path | None = None) -> EvalDataset:
    """載入黃金測試集；未指定路徑時讀取套件內建檔案。"""
    target = path or Path(__file__).with_name(DATASET_NAME)
    return EvalDataset.model_validate_json(target.read_text(encoding="utf-8"))


def section_of(result: SearchResult) -> str:
    """取回切片文字首行，即切片建立時寫入的章節前綴。"""
    return result.text.split("\n", 1)[0].strip()


def _normalize(text: str) -> str:
    """移除所有空白，用於比對語料中不規則的標題空白。"""
    return "".join(text.split())


def matches(result: SearchResult, expect: Expect) -> bool:
    """判斷檢索結果是否符合預期。"""
    if expect.doc_id is not None and result.doc_id != expect.doc_id:
        return False
    if expect.article_no is not None and result.article_no != expect.article_no:
        return False
    if expect.section is not None and not _normalize(section_of(result)).endswith(
        _normalize(expect.section)
    ):
        return False
    return all(fact in result.text for fact in expect.facts)


def _missing_facts(results: Sequence[SearchResult], expect: Expect) -> list[str]:
    return [fact for fact in expect.facts if not any(fact in item.text for item in results)]


def _forbidden_seen(results: Sequence[SearchResult], expect: Expect) -> bool:
    return any(fact in item.text for item in results for fact in expect.forbid)


def _evaluate_case(case: EvalCase, response: SearchResponse) -> CaseResult:
    results = response.results
    missing = _missing_facts(results, case.expect)
    rank = next(
        (index for index, item in enumerate(results, start=1) if matches(item, case.expect)),
        None,
    )
    if case.category == NO_ANSWER_CATEGORY:
        passed = not _forbidden_seen(results, case.expect)
    else:
        passed = rank is not None
    return CaseResult(
        id=case.id,
        category=case.category,
        passed=passed,
        rank=rank,
        matched_facts=[fact for fact in case.expect.facts if fact not in missing],
        missing_facts=missing,
        top_doc_ids=[item.doc_id for item in results],
        top_sections=[section_of(item) for item in results],
        latency_ms=response.latency_ms,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_eval(
    dataset: EvalDataset,
    searcher: Searcher,
    *,
    top_k: int = DEFAULT_TOP_K,
    enable_rerank: bool = False,
    dataset_name: str = DATASET_NAME,
) -> EvalReport:
    """逐題執行檢索並計算指標。"""
    results: list[CaseResult] = []
    for case in dataset.cases:
        request = SearchRequest(
            query=case.query,
            filters=QueryFilters.model_validate(case.filters) if case.filters else None,
            top_k=top_k,
            enable_rerank=enable_rerank,
        )
        results.append(_evaluate_case(case, searcher.search(request)))

    answerable = [item for item in results if item.category != NO_ANSWER_CATEGORY]
    unanswerable = [item for item in results if item.category == NO_ANSWER_CATEGORY]
    hits = [item for item in answerable if item.rank is not None]
    facts = [item for item in answerable if item.matched_facts or item.missing_facts]
    recall = [
        len(item.matched_facts) / (len(item.matched_facts) + len(item.missing_facts))
        for item in facts
    ]
    return EvalReport(
        dataset=dataset_name,
        top_k=top_k,
        enable_rerank=enable_rerank,
        total=len(results),
        passed=sum(1 for item in results if item.passed),
        hit_rate=len(hits) / len(answerable) if answerable else 0.0,
        hit_rate_at_3=_mean(
            [1.0 if item.rank is not None and item.rank <= 3 else 0.0 for item in answerable]
        ),
        mrr=_mean([1.0 / item.rank if item.rank is not None else 0.0 for item in answerable]),
        fact_recall=_mean(recall),
        no_answer_accuracy=(
            sum(1 for item in unanswerable if item.passed) / len(unanswerable)
            if unanswerable
            else 0.0
        ),
        results=results,
    )


def write_report(report: EvalReport, path: Path) -> None:
    """將評估報告寫入 JSON 檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize(report: EvalReport) -> str:
    """產出可讀的摘要文字。"""
    lines = [
        f"dataset={report.dataset} top_k={report.top_k} rerank={report.enable_rerank}",
        f"total={report.total} passed={report.passed}",
        f"hit_rate={report.hit_rate:.3f} hit_rate@3={report.hit_rate_at_3:.3f} "
        f"mrr={report.mrr:.3f} fact_recall={report.fact_recall:.3f} "
        f"no_answer_accuracy={report.no_answer_accuracy:.3f}",
    ]
    for item in report.results:
        flag = "PASS" if item.passed else "FAIL"
        rank = "-" if item.rank is None else str(item.rank)
        lines.append(f"[{flag}] {item.id} rank={rank} missing={item.missing_facts}")
    return "\n".join(lines)
