"""parser 單元測試。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fin_nexus.config import ParseConfig
from fin_nexus.ingest.parser import _default_runner, middle_json_to_blocks, parse_pdf, run_mineru
from fin_nexus.models import BlockType, FinancialReportMeta, ReportPeriod

FIXTURES = Path(__file__).parent / "fixtures"
_LOCAL = ParseConfig(mode="local", _env_file=None)


def _meta() -> FinancialReportMeta:
    return FinancialReportMeta(
        ticker="2330",
        company_name="台積電",
        fiscal_year=2024,
        quarter="Q1",
        report_period=ReportPeriod.QUARTERLY,
        source_uri="https://example.com/report.pdf",
    )


def _parse_local(tmp_path: Path, runner):
    return parse_pdf(Path("a.pdf"), tmp_path, _meta(), config=_LOCAL, runner=runner)


def test_middle_json_to_blocks() -> None:
    data = json.loads((FIXTURES / "middle_json.json").read_text(encoding="utf-8"))
    blocks = middle_json_to_blocks(data)

    headings = [block for block in blocks if block.type is BlockType.HEADING]
    assert headings[0].text == "四、合併資產負債表"
    assert headings[0].level == 2

    tables = [block for block in blocks if block.type is BlockType.TABLE]
    assert tables[0].text.startswith("**合併資產負債表（節錄）**")
    assert "現金及約當現金" in tables[0].text
    assert "註：單位為仟元。" in tables[0].text
    assert tables[0].page == 1

    assert any(block.type is BlockType.TEXT for block in blocks)
    assert all("page_number" not in block.text for block in blocks)


def test_unnumbered_title_becomes_text() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "paragraph_title",
                        "bbox": [0.1, 0.4, 0.9, 0.44],
                        "content": "期末持有有價證券情形",
                    },
                    {
                        "type": "doc_title",
                        "bbox": [0.1, 0.5, 0.9, 0.54],
                        "content": "取得不動產之金額達新台幣三億元或實收資本額百分之二十以上",
                    },
                ],
            }
        ]
    }
    blocks = middle_json_to_blocks(data)

    assert [block.type for block in blocks] == [BlockType.TEXT, BlockType.TEXT]


def test_titles_are_classified_by_numbering_prefix() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "paragraph_title",
                        "bbox": [0.1, 0.05, 0.9, 0.09],
                        "content": "聯華電子股份有限公司及子公司合併財務報告附註(續)",
                    },
                    {
                        "type": "doc_title",
                        "bbox": [0.1, 0.11, 0.9, 0.14],
                        "content": "台灣積體電路製造股份有限公司及子公司",
                    },
                    {
                        "type": "paragraph_title",
                        "bbox": [0.1, 0.3, 0.9, 0.34],
                        "content": "十三、採用權益法之投資",
                    },
                ],
            }
        ]
    }
    blocks = middle_json_to_blocks(data)

    assert [block.type for block in blocks] == [
        BlockType.TEXT,
        BlockType.TEXT,
        BlockType.HEADING,
    ]


def test_table_caption_is_rendered_first() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "table",
                        "content": [
                            {
                                "type": "table_caption",
                                "content": [{"type": "text", "content": "表一"}],
                            },
                            {
                                "type": "table_body",
                                "content": (
                                    "<table><tr><th>項目</th></tr>"
                                    "<tr><td>現金</td></tr></table>"
                                ),
                            },
                        ],
                    }
                ],
            }
        ]
    }
    blocks = middle_json_to_blocks(data)

    assert blocks[0].text.startswith("**表一**")


def test_continued_table_keeps_first_caption() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "table",
                        "content": [
                            {
                                "type": "table_caption",
                                "content": [{"type": "text", "content": "第一頁標題"}],
                            },
                            {
                                "type": "table_body",
                                "content": (
                                    "<table><tr><th>項目</th></tr>"
                                    "<tr><td>現金</td></tr></table>"
                                ),
                            },
                        ],
                    }
                ],
            },
            {
                "page_idx": 1,
                "blocks": [
                    {
                        "type": "table",
                        "continues_prev": True,
                        "content": [
                            {
                                "type": "table_caption",
                                "content": [{"type": "text", "content": "第二頁標題"}],
                            },
                            {
                                "type": "table_body",
                                "content": "<table><tr><td>存貨</td></tr></table>",
                            },
                        ],
                    }
                ],
            },
        ]
    }
    blocks = middle_json_to_blocks(data)

    assert len(blocks) == 1
    assert blocks[0].text.startswith("**第一頁標題**")
    assert "第二頁標題" not in blocks[0].text


def test_middle_json_to_blocks_handles_missing_pages() -> None:
    assert middle_json_to_blocks({}) == []


def test_run_mineru_builds_command() -> None:
    captured: list[list[str]] = []
    run_mineru(
        Path("a.pdf"),
        Path("out"),
        tier="basic",
        runner=lambda command: captured.append(list(command)),
    )
    command = captured[0]
    assert command[:2] == ["mineru-kit", "parse"]
    assert "--tier" in command
    assert command[command.index("--tier") + 1] == "basic"
    assert "-f" in command
    assert command[command.index("-f") + 1] == "middle_json"


def test_run_mineru_remote_command() -> None:
    captured: list[list[str]] = []
    run_mineru(
        Path("a.pdf"),
        Path("out"),
        runner=lambda command: captured.append(list(command)),
        remote=True,
        api_key="secret-token",
    )
    command = captured[0]
    assert "--remote" in command
    assert "--tier" not in command
    assert command[command.index("--api-key") + 1] == "secret-token"


def test_run_mineru_remote_without_api_key() -> None:
    captured: list[list[str]] = []
    run_mineru(
        Path("a.pdf"),
        Path("out"),
        runner=lambda command: captured.append(list(command)),
        remote=True,
    )
    command = captured[0]
    assert "--remote" in command
    assert "--api-key" not in command


def test_parse_pdf_reads_middle_json(tmp_path: Path) -> None:
    source = (FIXTURES / "middle_json.json").read_text(encoding="utf-8")

    def runner(command: list[str]) -> None:
        (tmp_path / "report.json").write_text(source, encoding="utf-8")

    document = _parse_local(tmp_path, runner)
    assert document.meta.parse_engine == "mineru"
    assert document.blocks


def test_parse_pdf_missing_middle_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _parse_local(tmp_path, lambda command: None)


def test_parse_pdf_prefers_stem_match(tmp_path: Path) -> None:
    target = {
        "schema": "docvortex.middle",
        "pages": [{"page_idx": 0, "blocks": [{"type": "text", "content": "目標文件"}]}],
    }
    other = {
        "schema": "docvortex.middle",
        "pages": [{"page_idx": 0, "blocks": [{"type": "text", "content": "其他文件"}]}],
    }
    (tmp_path / "a.json").write_text(json.dumps(target), encoding="utf-8")
    (tmp_path / "0.json").write_text(json.dumps(other), encoding="utf-8")

    document = _parse_local(tmp_path, lambda command: None)
    assert [block.text for block in document.blocks] == ["目標文件"]


def test_parse_pdf_finds_middle_by_schema(tmp_path: Path) -> None:
    (tmp_path / "notes.json").write_text('{"schema": "other"}', encoding="utf-8")
    real = {
        "schema": "docvortex.middle",
        "pages": [{"page_idx": 0, "blocks": [{"type": "text", "content": "真正內容"}]}],
    }
    (tmp_path / "report.json").write_text(json.dumps(real), encoding="utf-8")

    document = _parse_local(tmp_path, lambda command: None)
    assert [block.text for block in document.blocks] == ["真正內容"]


def test_parse_pdf_ignores_non_middle_json(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text('{"schema": "other"}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        _parse_local(tmp_path, lambda command: None)


def test_default_runner_reports_failure_output() -> None:
    command = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]
    with pytest.raises(RuntimeError) as excinfo:
        _default_runner(command)
    message = str(excinfo.value)
    assert "exit 3" in message
    assert "boom" in message


def test_default_runner_redacts_api_key() -> None:
    command = [sys.executable, "-c", "import sys; sys.exit(3)", "--api-key", "super-secret"]
    with pytest.raises(RuntimeError) as excinfo:
        _default_runner(command)
    message = str(excinfo.value)
    assert "super-secret" not in message
    assert "--api-key ***" in message


def test_default_runner_missing_executable() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        _default_runner(["definitely-not-a-real-command-xyz"])
    assert "找不到執行檔" in str(excinfo.value)


def test_merges_continued_table_with_repeated_header() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "table",
                        "content": [
                            {
                                "type": "table_body",
                                "content": (
                                    "<table><tr><th>項目</th><th>金額</th></tr>"
                                    "<tr><td>現金</td><td>10</td></tr></table>"
                                ),
                            }
                        ],
                    }
                ],
            },
            {
                "page_idx": 1,
                "blocks": [
                    {
                        "type": "table",
                        "continues_prev": True,
                        "content": [
                            {
                                "type": "table_body",
                                "content": (
                                    "<table><tr><th>項目</th><th>金額</th></tr>"
                                    "<tr><td>應收帳款</td><td>20</td></tr></table>"
                                ),
                            }
                        ],
                    }
                ],
            },
        ]
    }
    blocks = middle_json_to_blocks(data)

    assert len(blocks) == 1
    assert blocks[0].type is BlockType.TABLE
    assert blocks[0].page == 0
    assert "現金" in blocks[0].text
    assert "應收帳款" in blocks[0].text
    assert blocks[0].text.count("項目") == 1


def test_merges_continued_table_without_repeated_header() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "table",
                        "content": [
                            {
                                "type": "table_body",
                                "content": (
                                    "<table><tr><th>項目</th><th>金額</th></tr>"
                                    "<tr><td>現金</td><td>10</td></tr></table>"
                                ),
                            }
                        ],
                    }
                ],
            },
            {
                "page_idx": 1,
                "blocks": [
                    {
                        "type": "table",
                        "continues_prev": True,
                        "content": [
                            {
                                "type": "table_body",
                                "content": "<table><tr><td>存貨</td><td>30</td></tr></table>",
                            }
                        ],
                    }
                ],
            },
        ]
    }
    blocks = middle_json_to_blocks(data)

    assert len(blocks) == 1
    assert "存貨" in blocks[0].text
    assert blocks[0].text.count("項目") == 1


def test_merges_continued_text() -> None:
    data = {
        "pages": [
            {"page_idx": 0, "blocks": [{"type": "text", "content": "前半段"}]},
            {
                "page_idx": 1,
                "blocks": [{"type": "text", "continues_prev": True, "content": "後半段"}],
            },
        ]
    }
    blocks = middle_json_to_blocks(data)

    assert len(blocks) == 1
    assert blocks[0].type is BlockType.TEXT
    assert blocks[0].text == "前半段\n後半段"


def test_middle_json_to_blocks_normalizes_text() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [{"type": "text", "content": "台灣 積體 電路 综计"}],
            }
        ]
    }
    blocks = middle_json_to_blocks(data)

    assert blocks[0].text == "台灣積體電路綜計"


def test_run_mineru_remote_url() -> None:
    captured: list[list[str]] = []
    run_mineru(
        Path("a.pdf"),
        Path("out"),
        runner=lambda command: captured.append(list(command)),
        remote=True,
        api_key="secret-token",
        remote_url="https://example.com/api",
    )
    command = captured[0]
    assert command[command.index("--remote-url") + 1] == "https://example.com/api"


def test_parse_pdf_remote_requires_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FIN_NEXUS_MINERU_API_KEY", raising=False)
    with pytest.raises(ValueError):
        parse_pdf(
            Path("a.pdf"),
            tmp_path,
            _meta(),
            config=ParseConfig(mode="remote", _env_file=None),
            runner=lambda command: None,
        )


def test_parse_pdf_remote_passes_key(tmp_path: Path) -> None:
    source = (FIXTURES / "middle_json.json").read_text(encoding="utf-8")
    captured: list[list[str]] = []

    def runner(command: list[str]) -> None:
        captured.append(list(command))
        (tmp_path / "report.json").write_text(source, encoding="utf-8")

    parse_pdf(
        Path("a.pdf"),
        tmp_path,
        _meta(),
        config=ParseConfig(mode="remote", api_key="secret-token", _env_file=None),
        runner=runner,
    )
    command = captured[0]
    assert "--remote" in command
    assert command[command.index("--api-key") + 1] == "secret-token"
    assert "--tier" not in command


def test_empty_text_block_resets_table_state() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "blocks": [
                    {
                        "type": "table",
                        "content": [
                            {
                                "type": "table_body",
                                "content": (
                                    "<table><tr><th>項目</th><th>金額</th></tr>"
                                    "<tr><td>現金</td><td>10</td></tr></table>"
                                ),
                            }
                        ],
                    }
                ],
            },
            {
                "page_idx": 1,
                "blocks": [
                    {"type": "text", "content": []},
                    {
                        "type": "table",
                        "continues_prev": True,
                        "content": [
                            {
                                "type": "table_body",
                                "content": "<table><tr><td>存貨</td><td>30</td></tr></table>",
                            }
                        ],
                    },
                ],
            },
        ]
    }
    blocks = middle_json_to_blocks(data)

    tables = [block for block in blocks if block.type is BlockType.TABLE]
    assert len(tables) == 2
