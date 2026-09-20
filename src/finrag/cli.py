"""資料入庫指令列介面。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from finrag.config import StoragePaths
from finrag.http import build_client
from finrag.ingest.pipeline import chunk_documents, fetch_reports, parse_documents
from finrag.ingest.targets import (
    DEFAULT_QUARTERS,
    DEFAULT_TICKERS,
    DEFAULT_YEARS,
    build_targets,
)
from finrag.models.domain import ReportTarget
from finrag.text import default_token_counter


def build_parser() -> argparse.ArgumentParser:
    """建立指令列解析器。"""
    parser = argparse.ArgumentParser(prog="finrag", description="財報入庫")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--ticker", nargs="+", default=list(DEFAULT_TICKERS))
        sub.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
        sub.add_argument("--quarters", nargs="+", default=list(DEFAULT_QUARTERS))
        sub.add_argument("--data-dir", default=None)
        sub.add_argument("--force", action="store_true")

    for name in ("fetch", "parse", "chunk", "all"):
        sub = subparsers.add_parser(name)
        add_common(sub)
        if name in ("fetch", "all"):
            sub.add_argument("--delay", type=float, default=20.0)

    return parser


def _paths(args: argparse.Namespace) -> StoragePaths:
    if args.data_dir:
        return StoragePaths(data_dir=args.data_dir)
    return StoragePaths()


def _dispatch(
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
    else:
        chunk_documents(targets, paths, counter=default_token_counter(), force=args.force)


def main(argv: Sequence[str] | None = None) -> int:
    """執行指令列入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = _paths(args)
    try:
        targets = build_targets(args.ticker, args.years, args.quarters)
    except ValueError as error:
        parser.error(str(error))

    commands = ("fetch", "parse", "chunk") if args.command == "all" else (args.command,)
    for command in commands:
        _dispatch(command, args, paths, targets)
        print(f"{command}: {len(targets)} target(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
