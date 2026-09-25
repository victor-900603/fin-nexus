"""自適應檢索路由器測試。"""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from finrag.retrieval.expander import QueryAnalysis
from finrag.retrieval.router import route
from finrag.storage.qdrant_client import REGULATIONS_COLLECTION, REPORTS_COLLECTION


def _analysis(**overrides: Any) -> QueryAnalysis:
    data: dict[str, Any] = {
        "query": "台積電 2024 Q1 營收",
        "expanded": "台積電 2024 Q1 營收 營業收入淨額",
        "sparse_query": "營業收入淨額 營業收入 營收",
        "tickers": [],
        "fiscal_years": [],
        "quarters": [],
        "report_period": None,
        "reg_pcodes": [],
        "matches": [],
    }
    data.update(overrides)
    return QueryAnalysis(**data)


def test_auto_dense_without_entities() -> None:
    plan = route(_analysis())

    assert plan.mode == "dense"
    assert plan.collection == REPORTS_COLLECTION
    assert plan.use_sparse is False
    assert plan.filters is None


def test_auto_filtered_hybrid_with_report_entities() -> None:
    plan = route(_analysis(tickers=["2330"], fiscal_years=[2024], quarters=["Q1"]))

    assert plan.mode == "filtered_hybrid"
    assert plan.collection == REPORTS_COLLECTION
    assert plan.use_sparse is True
    assert plan.query == "台積電 2024 Q1 營收 營業收入淨額"


def test_auto_hybrid_for_regulation() -> None:
    plan = route(_analysis(reg_pcodes=["G0400001"]))

    assert plan.mode == "hybrid"
    assert plan.collection == REGULATIONS_COLLECTION


def test_explicit_mode_overrides_auto() -> None:
    plan = route(_analysis(tickers=["2330"]), search_mode="dense")

    assert plan.mode == "dense"
    assert plan.collection == REPORTS_COLLECTION
    assert plan.use_sparse is False


def test_top_k_passed_through() -> None:
    plan = route(_analysis(), top_k=5)

    assert plan.top_k == 5


def test_filters_single_value() -> None:
    plan = route(_analysis(tickers=["2330"]))

    assert plan.filters is not None
    condition = plan.filters.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.match == models.MatchValue(value="2330")


def test_filters_multi_value_same_field_uses_match_any() -> None:
    plan = route(_analysis(tickers=["2330", "2303"]))

    assert plan.filters is not None
    condition = plan.filters.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.match == models.MatchAny(any=["2330", "2303"])


def test_filters_cross_field_are_and() -> None:
    plan = route(_analysis(tickers=["2330"], fiscal_years=[2024], quarters=["Q1"]))

    assert plan.filters is not None
    keys = [c.key for c in plan.filters.must if isinstance(c, models.FieldCondition)]
    assert keys == ["ticker", "fiscal_year", "quarter"]


def test_explicit_filters_override_analysis() -> None:
    plan = route(_analysis(tickers=["2330"]), filters={"ticker": "2454"})

    assert plan.filters is not None
    condition = plan.filters.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.match == models.MatchValue(value="2454")


def test_regulation_filter_uses_reg_pcode() -> None:
    plan = route(_analysis(reg_pcodes=["G0400001", "G0400069"]))

    assert plan.filters is not None
    condition = plan.filters.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.key == "reg_pcode"
    assert condition.match == models.MatchAny(any=["G0400001", "G0400069"])


def test_explicit_collection_overrides_regulation_heuristic() -> None:
    plan = route(_analysis(reg_pcodes=["G0400001"]), collection=REPORTS_COLLECTION)

    assert plan.collection == REPORTS_COLLECTION
    assert plan.filters is None


def test_explicit_regulation_collection_builds_regulation_filter() -> None:
    plan = route(
        _analysis(tickers=["2330"]),
        collection=REGULATIONS_COLLECTION,
        filters={"reg_pcode": "G0400001"},
    )

    assert plan.collection == REGULATIONS_COLLECTION
    assert plan.filters is not None
    condition = plan.filters.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.key == "reg_pcode"
    assert condition.match == models.MatchValue(value="G0400001")


def test_filters_ticker_switches_to_filtered_hybrid() -> None:
    plan = route(_analysis(), filters={"ticker": "2330"})

    assert plan.mode == "filtered_hybrid"
    assert plan.use_sparse is True


def test_filters_fiscal_year_switches_to_filtered_hybrid() -> None:
    plan = route(_analysis(), filters={"fiscal_year": 2024})

    assert plan.mode == "filtered_hybrid"


def test_filters_quarter_switches_to_filtered_hybrid() -> None:
    plan = route(_analysis(), filters={"quarter": "Q1"})

    assert plan.mode == "filtered_hybrid"


def test_filters_reg_pcode_switches_to_hybrid() -> None:
    plan = route(
        _analysis(), collection=REGULATIONS_COLLECTION, filters={"reg_pcode": "G0400001"}
    )

    assert plan.mode == "hybrid"
    assert plan.collection == REGULATIONS_COLLECTION


def test_filters_without_entity_keeps_dense() -> None:
    plan = route(_analysis(), filters={"doc_id": "2330_2023_Q1_consolidated"})

    assert plan.mode == "dense"


def test_explicit_mode_overrides_filters() -> None:
    plan = route(_analysis(), filters={"ticker": "2330"}, search_mode="dense")

    assert plan.mode == "dense"
    assert plan.use_sparse is False
