"""資料入庫指令列介面。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from finrag.api.service import QueryService
from finrag.config import StoragePaths
from finrag.contracts import DEFAULT_TOP_K
from finrag.eval.runner import load_dataset, run_eval, summarize, write_report
from finrag.generation.client import default_client
from finrag.http import build_client
from finrag.ingest.pipeline import (
    chunk_documents,
    chunk_regulations,
    fetch_regulations,
    fetch_reports,
    parse_documents,
    parse_regulations,
)
from finrag.ingest.targets import (
    DEFAULT_QUARTERS,
    DEFAULT_TICKERS,
    DEFAULT_YEARS,
    build_targets,
)
from finrag.logging import configure_logging
from finrag.models.domain import LAW_SPECS, LawSpec, ReportTarget
from finrag.retrieval.ontology import default_ontology
from finrag.retrieval.reranker import default_reranker
from finrag.storage.embedder import default_embedder
from finrag.storage.indexer import index_regulations, index_reports
from finrag.storage.qdrant_client import build_client as build_qdrant_client
from finrag.text import default_token_counter

SOURCES: tuple[str, ...] = ("reports", "laws", "all")


def build_parser() -> argparse.ArgumentParser:
    """建立指令列解析器。"""
    parser = argparse.ArgumentParser(prog="finrag", description="財報與法規入庫")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--source", choices=SOURCES, default="reports")
        sub.add_argument("--ticker", nargs="+", default=list(DEFAULT_TICKERS))
        sub.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
        sub.add_argument("--quarters", nargs="+", default=list(DEFAULT_QUARTERS))
        sub.add_argument("--data-dir", default=None)
        sub.add_argument("--force", action="store_true")

    for name in ("fetch", "parse", "chunk", "index", "all"):
        sub = subparsers.add_parser(name)
        add_common(sub)
        if name in ("fetch", "all"):
            sub.add_argument("--delay", type=float, default=20.0)
        if name == "index":
            sub.add_argument("--batch-size", type=int, default=64)

    serve = subparsers.add_parser("serve", help="啟動 API 服務")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    evaluate = subparsers.add_parser("eval", help="執行黃金測試集評估")
    evaluate.add_argument("--dataset", default=None)
    evaluate.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    evaluate.add_argument("--rerank", action="store_true")
    evaluate.add_argument("--output", default=None)

    return parser


def _paths(args: argparse.Namespace) -> StoragePaths:
    if args.data_dir:
        return StoragePaths(data_dir=args.data_dir)
    return StoragePaths()


def _sources(source: str) -> tuple[str, ...]:
    if source == "all":
        return ("reports", "laws")
    return (source,)


def _dispatch_reports(
    command: str,
    args: argparse.Namespace,
    paths: StoragePaths,
    targets: Sequence[ReportTarget],
) -> None:
    if command == "fetch":
        with build_client() as client:
            fetch_reports(targets, paths, client, force=args.force, delay=args.delay)
    elif command == "parse":
        parse_documents(targets, paths, force=args.force)
    elif command == "chunk":
        chunk_documents(targets, paths, counter=default_token_counter(), force=args.force)
    else:
        qdrant = build_qdrant_client()
        try:
            index_reports(
                [target.doc_id for target in targets],
                paths,
                qdrant,
                embedder=default_embedder(),
                force=args.force,
                batch_size=args.batch_size,
            )
        finally:
            qdrant.close()


def _dispatch_regulations(
    command: str,
    args: argparse.Namespace,
    paths: StoragePaths,
    specs: Sequence[LawSpec],
) -> None:
    if command == "fetch":
        with build_client() as client:
            fetch_regulations(specs, paths, client, force=args.force)
    elif command == "parse":
        parse_regulations(specs, paths, force=args.force)
    elif command == "chunk":
        chunk_regulations(specs, paths, counter=default_token_counter(), force=args.force)
    else:
        qdrant = build_qdrant_client()
        try:
            index_regulations(
                [spec.doc_id for spec in specs],
                paths,
                qdrant,
                embedder=default_embedder(),
                force=args.force,
                batch_size=args.batch_size,
            )
        finally:
            qdrant.close()


def _run_serve(args: argparse.Namespace) -> int:
    configure_logging()
    uvicorn.run(
        "finrag.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    dataset = load_dataset(Path(args.dataset) if args.dataset else None)
    qdrant = build_qdrant_client()
    try:
        report = run_eval(
            dataset,
            QueryService(
                qdrant,
                embedder=default_embedder(),
                reranker=default_reranker(),
                llm=default_client(),
                ontology=default_ontology(),
                counter=default_token_counter(),
            ),
            top_k=args.top_k,
            enable_rerank=args.rerank,
        )
    finally:
        qdrant.close()
    print(summarize(report))
    if args.output:
        write_report(report, Path(args.output))
        print(f"report: {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """執行指令列入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _run_serve(args)
    if args.command == "eval":
        return _run_eval(args)

    paths = _paths(args)
    sources = _sources(args.source)

    targets: list[ReportTarget] = []
    if "reports" in sources:
        try:
            targets = build_targets(args.ticker, args.years, args.quarters)
        except ValueError as error:
            parser.error(str(error))

    commands = ("fetch", "parse", "chunk") if args.command == "all" else (args.command,)
    for command in commands:
        if "reports" in sources:
            _dispatch_reports(command, args, paths, targets)
            print(f"{command}: {len(targets)} target(s)")
        if "laws" in sources:
            _dispatch_regulations(command, args, paths, LAW_SPECS)
            print(f"{command}: {len(LAW_SPECS)} regulation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
