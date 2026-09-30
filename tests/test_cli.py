"""資料入庫指令測試。"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest

from finrag import cli


def _fake(monkeypatch: pytest.MonkeyPatch, name: str, calls: list[tuple[str, Any]]) -> None:
    def fake(*args: Any, **kwargs: Any) -> None:
        calls.append((name, (args, kwargs)))

    monkeypatch.setattr(cli, name, fake)


def _null_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_client", lambda *args, **kwargs: contextlib.nullcontext(None))


def test_parser_has_subcommands() -> None:
    parser = cli.build_parser()

    for command in ("fetch", "parse", "chunk", "index", "all", "serve"):
        assert parser.parse_args([command]).command == command


def test_default_arguments() -> None:
    args = cli.build_parser().parse_args(["fetch"])

    assert args.ticker == ["2330", "2303", "2454"]
    assert args.years == [2023, 2024, 2025]
    assert args.quarters == ["Q1", "Q2", "Q3", "Q4"]
    assert args.source == "reports"
    assert args.force is False
    assert args.delay == 20.0


def test_custom_arguments() -> None:
    args = cli.build_parser().parse_args(
        [
            "fetch",
            "--ticker",
            "2303",
            "2454",
            "--years",
            "2024",
            "--quarters",
            "Q1",
            "Q2",
            "--data-dir",
            "custom",
            "--force",
            "--delay",
            "5",
        ]
    )

    assert args.ticker == ["2303", "2454"]
    assert args.years == [2024]
    assert args.quarters == ["Q1", "Q2"]
    assert args.data_dir == "custom"
    assert args.force is True
    assert args.delay == 5.0


def test_parse_has_no_delay_option() -> None:
    assert not hasattr(cli.build_parser().parse_args(["parse"]), "delay")


def test_main_fetch_calls_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Any]] = []
    _fake(monkeypatch, "fetch_reports", calls)
    _null_client(monkeypatch)

    code = cli.main(
        [
            "fetch",
            "--ticker",
            "2330",
            "--years",
            "2023",
            "--quarters",
            "Q1",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    (name, (args, kwargs)), = calls
    targets, paths, _client = args
    assert name == "fetch_reports"
    assert [target.doc_id for target in targets] == ["2330_2023_Q1_consolidated"]
    assert paths.data_dir == tmp_path
    assert kwargs == {"force": False, "delay": 20.0}


def test_main_force_and_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []
    _fake(monkeypatch, "fetch_reports", calls)
    _null_client(monkeypatch)

    cli.main(["fetch", "--force", "--delay", "3"])

    _, (_, kwargs) = calls[0]
    assert kwargs == {"force": True, "delay": 3.0}


def test_main_all_runs_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    for name in ("fetch_reports", "parse_documents", "chunk_documents"):
        monkeypatch.setattr(
            cli,
            name,
            (lambda label: lambda *args, **kwargs: order.append(label))(name),
        )
    _null_client(monkeypatch)

    code = cli.main(["all", "--ticker", "2330", "--years", "2023", "--quarters", "Q1"])

    assert code == 0
    assert order == ["fetch_reports", "parse_documents", "chunk_documents"]


def test_main_invalid_ticker_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        cli.main(["fetch", "--ticker", "233"])


def test_main_reports_target_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake(monkeypatch, "chunk_documents", [])
    monkeypatch.setattr(cli, "build_client", lambda *args, **kwargs: contextlib.nullcontext(None))

    cli.main(["chunk", "--ticker", "2330", "--years", "2023", "--quarters", "Q1", "Q2"])

    assert capsys.readouterr().out.strip() == "chunk: 2 target(s)"


def test_default_source_is_reports() -> None:
    assert cli.build_parser().parse_args(["fetch"]).source == "reports"


def test_source_laws_runs_regulations(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    for name in ("fetch_regulations", "parse_regulations", "chunk_regulations"):
        monkeypatch.setattr(
            cli,
            name,
            (lambda label: lambda *args, **kwargs: order.append(label))(name),
        )
    _null_client(monkeypatch)

    code = cli.main(["all", "--source", "laws"])

    assert code == 0
    assert order == ["fetch_regulations", "parse_regulations", "chunk_regulations"]
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [
        "fetch: 3 regulation(s)",
        "parse: 3 regulation(s)",
        "chunk: 3 regulation(s)",
    ]


def test_source_all_runs_both(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    names = (
        "fetch_reports",
        "fetch_regulations",
        "parse_documents",
        "parse_regulations",
        "chunk_documents",
        "chunk_regulations",
    )
    for name in names:
        monkeypatch.setattr(
            cli,
            name,
            (lambda label: lambda *args, **kwargs: order.append(label))(name),
        )
    _null_client(monkeypatch)

    cli.main(
        ["all", "--source", "all", "--ticker", "2330", "--years", "2023", "--quarters", "Q1"]
    )

    assert order == list(names)


def test_source_laws_fetch_uses_default_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []
    _fake(monkeypatch, "fetch_regulations", calls)
    _null_client(monkeypatch)

    cli.main(["fetch", "--source", "laws"])

    (name, (_, kwargs)), = calls
    assert name == "fetch_regulations"
    assert kwargs == {"force": False}


def test_source_laws_skips_ticker_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, "chunk_regulations", [])
    _null_client(monkeypatch)

    code = cli.main(["chunk", "--source", "laws", "--ticker", "233"])

    assert code == 0


def test_index_subcommand_default_batch_size() -> None:
    assert cli.build_parser().parse_args(["index"]).batch_size == 64


def test_index_reports_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Any]] = []
    _fake(monkeypatch, "index_reports", calls)
    _null_client(monkeypatch)

    code = cli.main(
        [
            "index",
            "--ticker",
            "2330",
            "--years",
            "2023",
            "--quarters",
            "Q1",
            "--batch-size",
            "8",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    (name, (args, kwargs)), = calls
    doc_ids, paths, _client = args
    assert name == "index_reports"
    assert doc_ids == ["2330_2023_Q1_consolidated"]
    assert paths.data_dir == tmp_path
    assert set(kwargs) == {"embedder", "force", "batch_size"}
    assert kwargs["force"] is False
    assert kwargs["batch_size"] == 8


def test_index_laws_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []
    _fake(monkeypatch, "index_regulations", calls)
    _null_client(monkeypatch)

    code = cli.main(["index", "--source", "laws"])

    assert code == 0
    (name, (_, kwargs)), = calls
    assert name == "index_regulations"
    assert set(kwargs) == {"embedder", "force", "batch_size"}
    assert kwargs["force"] is False
    assert kwargs["batch_size"] == 64


def test_index_reports_target_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake(monkeypatch, "index_reports", [])
    _null_client(monkeypatch)

    cli.main(["index", "--ticker", "2330", "--years", "2023", "--quarters", "Q1", "Q2"])

    assert capsys.readouterr().out.strip() == "index: 2 target(s)"


def test_serve_subcommand_defaults() -> None:
    args = cli.build_parser().parse_args(["serve"])

    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.reload is False
    assert not hasattr(args, "source")


def test_serve_subcommand_custom_arguments() -> None:
    args = cli.build_parser().parse_args(
        ["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"]
    )

    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.reload is True


def test_main_serve_runs_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []
    _fake(monkeypatch, "configure_logging", calls)
    run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run(*args: Any, **kwargs: Any) -> None:
        run_calls.append((args, kwargs))

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    code = cli.main(["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])

    assert code == 0
    assert [name for name, _ in calls] == ["configure_logging"]
    assert run_calls == [
        (("finrag.api.main:app",), {"host": "0.0.0.0", "port": 9000, "reload": True})
    ]
