"""評估層測試。"""

from __future__ import annotations

import json
from pathlib import Path

from finrag.contracts import SearchRequest, SearchResponse, SearchResult
from finrag.eval.runner import (
    EvalCase,
    EvalDataset,
    Expect,
    load_dataset,
    matches,
    run_eval,
    section_of,
    summarize,
    write_report,
)


def _result(
    text: str,
    *,
    doc_id: str = "doc",
    article_no: str | None = None,
) -> SearchResult:
    return SearchResult(
        id="point",
        score=1.0,
        source="fin_reports",
        doc_id=doc_id,
        article_no=article_no,
        text=text,
    )


class _FakeSearcher:
    def __init__(self, results: dict[str, list[SearchResult]]) -> None:
        self._results = results

    def search(self, request: SearchRequest) -> SearchResponse:
        return SearchResponse(
            query=request.query,
            target_collection="fin_reports",
            search_path_taken="filtered_hybrid",
            results=self._results.get(request.query, []),
            latency_ms=12,
        )


def _dataset(*cases: EvalCase) -> EvalDataset:
    return EvalDataset(version="test", cases=list(cases))


def test_load_dataset_reads_shipped_cases() -> None:
    dataset = load_dataset()

    assert dataset.version == "2"
    ids = [case.id for case in dataset.cases]
    assert len(ids) == len(set(ids))
    assert len(ids) == 50
    categories = {case.category for case in dataset.cases}
    assert categories == {"single_table_value", "multi_hop_note", "regulation", "no_answer"}


def test_section_of_returns_first_line() -> None:
    assert section_of(_result("六、存貨\n製成品 50,416,137")) == "六、存貨"


def test_matches_requires_doc_id() -> None:
    assert not matches(_result("x", doc_id="other"), Expect(doc_id="doc"))


def test_matches_requires_every_fact() -> None:
    result = _result("五、合併綜合損益表\n營業收入淨額 508,632,973")

    assert matches(result, Expect(facts=["508,632,973"]))
    assert not matches(result, Expect(facts=["508,632,973", "111,111"]))


def test_matches_accepts_nested_section_path() -> None:
    result = _result("六、重要會計項目之說明 > (一) 普通股股本\n額定股本 280,500,000")

    assert matches(result, Expect(section="(一) 普通股股本"))


def test_matches_ignores_section_whitespace() -> None:
    result = _result("(一 ) 普通股股本\n額定股本 280,500,000")

    assert matches(result, Expect(section="(一) 普通股股本"))


def test_matches_rejects_article_mismatch() -> None:
    result = _result("[證券交易法 > 第 30 條]\n公開說明書", article_no="第 30 條")

    assert matches(result, Expect(article_no="第 30 條"))
    assert not matches(result, Expect(article_no="第 31 條"))


def test_matches_without_expectations_accepts_any_result() -> None:
    assert matches(_result("anything"), Expect())


def test_run_eval_computes_hit_rate_and_mrr() -> None:
    dataset = _dataset(
        EvalCase(
            id="hit",
            category="single_table_value",
            query="q1",
            expect=Expect(doc_id="doc"),
        ),
        EvalCase(
            id="miss",
            category="single_table_value",
            query="q2",
            expect=Expect(doc_id="doc"),
        ),
    )
    searcher = _FakeSearcher({"q1": [_result("a", doc_id="doc")]})

    report = run_eval(dataset, searcher, top_k=3)

    assert report.total == 2
    assert report.passed == 1
    assert report.hit_rate == 0.5
    assert report.hit_rate_at_3 == 0.5
    assert report.mrr == 0.5
    assert [item.rank for item in report.results] == [1, None]


def test_run_eval_ranks_beyond_third_result() -> None:
    dataset = _dataset(
        EvalCase(id="late", category="regulation", query="q", expect=Expect(doc_id="doc"))
    )
    results = [_result(f"filler {index}", doc_id="other") for index in range(3)]
    results.append(_result("target", doc_id="doc"))
    searcher = _FakeSearcher({"q": results})

    report = run_eval(dataset, searcher, top_k=5)

    assert report.results[0].rank == 4
    assert report.hit_rate == 1.0
    assert report.hit_rate_at_3 == 0.0
    assert report.mrr == 0.25


def test_run_eval_computes_fact_recall_and_requires_all_facts() -> None:
    dataset = _dataset(
        EvalCase(
            id="full",
            category="multi_hop_note",
            query="q1",
            expect=Expect(doc_id="doc", facts=["1,000", "2,000"]),
        ),
        EvalCase(
            id="partial",
            category="multi_hop_note",
            query="q2",
            expect=Expect(doc_id="doc", facts=["1,000", "2,000"]),
        ),
    )
    searcher = _FakeSearcher(
        {
            "q1": [_result("存貨 1,000 製成品 2,000", doc_id="doc")],
            "q2": [_result("存貨 1,000", doc_id="doc")],
        }
    )

    report = run_eval(dataset, searcher)

    assert report.fact_recall == 0.75
    assert [item.passed for item in report.results] == [True, False]
    assert [item.rank for item in report.results] == [1, None]
    assert report.results[1].matched_facts == ["1,000"]
    assert report.results[1].missing_facts == ["2,000"]


def test_run_eval_scores_no_answer_cases() -> None:
    dataset = _dataset(
        EvalCase(
            id="clean",
            category="no_answer",
            query="q1",
            expect=Expect(forbid=["加密貨幣"]),
        ),
        EvalCase(
            id="leaked",
            category="no_answer",
            query="q2",
            expect=Expect(forbid=["加密貨幣"]),
        ),
    )
    searcher = _FakeSearcher(
        {
            "q1": [_result("營業收入 1,000")],
            "q2": [_result("持有加密貨幣 5")],
        }
    )

    report = run_eval(dataset, searcher)

    assert report.no_answer_accuracy == 0.5
    assert [item.passed for item in report.results] == [True, False]
    assert report.hit_rate == 0.0


def test_summarize_lists_every_case() -> None:
    dataset = _dataset(
        EvalCase(id="only", category="regulation", query="q", expect=Expect(doc_id="doc"))
    )

    text = summarize(run_eval(dataset, _FakeSearcher({"q": [_result("t", doc_id="doc")]})))

    assert "hit_rate=1.000" in text
    assert "[PASS] only" in text


def test_write_report_creates_json(tmp_path: Path) -> None:
    dataset = _dataset(
        EvalCase(id="only", category="regulation", query="q", expect=Expect(doc_id="doc"))
    )
    report = run_eval(dataset, _FakeSearcher({"q": [_result("t", doc_id="doc")]}))
    target = tmp_path / "nested" / "report.json"

    write_report(report, target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["total"] == 1
    assert payload["results"][0]["id"] == "only"
