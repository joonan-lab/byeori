"""question_agent: the research agent reads a paper's stored supplementary tables as a tool."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from byeori import question_agent
from byeori.question_agent import SUPPLEMENTARY_RESULT_CHARS, WikiTools
from test_supplementary_reader import DEG, STEM, workbook, world


def tools(files, **kw):
    return WikiTools(world(files, **kw), "bucket", search=None, reread="auto")


def test_the_tool_is_offered_and_the_prompt_says_when_to_use_it():
    names = [tool["toolSpec"]["name"] for tool in question_agent.TOOLS]
    assert "read_supplementary" in names
    assert "read_supplementary" in question_agent.SYSTEM and "Supplementary Files" in question_agent.SYSTEM


def test_stem_alone_returns_the_guide_and_the_kept_files():
    runtime = tools({"deg.xlsx": workbook({"Data 3": DEG}), "rs.pdf": b"%PDF"}, kept={"rs.pdf": False})

    result = runtime.call("read_supplementary", {"stem": STEM})

    assert result["text"].startswith("# Supplementary files") and [f["file"] for f in result["files"]] == ["deg.xlsx"]
    assert runtime.supplementary_reads == [{"stem": STEM, "file": None, "sheet": None, "find": None, "matches": None}]


def test_a_table_read_and_a_paper_search_return_matching_rows():
    runtime = tools({"deg.xlsx": workbook({"Data 3": DEG})})

    table = runtime.call("read_supplementary", {"stem": STEM, "file": "deg.xlsx", "find": "GFAP"})
    search = runtime.call("read_supplementary", {"stem": STEM, "find": "SNCA"})

    assert table["parts"][0]["rows"] == [[7, ["GFAP", "Astro", "3.1", "1e-30"]]]
    assert search["files_with_matches"][0]["matches_total"] == 2
    assert [r["matches"] for r in runtime.supplementary_reads] == [1, None]


def test_an_archive_alone_lists_its_members():
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("S1.csv", "gene\nTH\n")
    runtime = tools({"t.zip": inner.getvalue()})

    result = runtime.call("read_supplementary", {"stem": STEM, "file": "t.zip"})

    assert result["members"] == [{"member": "t.zip::S1.csv", "bytes": 8, "readable": True}]


def test_refusals_reach_the_model_as_tool_errors():
    runtime = tools({"deg.xlsx": workbook({"Data 3": DEG}), "si.pdf": b"%PDF"})

    with pytest.raises(ValueError, match="PDF text is not extracted"):
        runtime.call("read_supplementary", {"stem": STEM, "file": "si.pdf"})
    with pytest.raises(ValueError, match="not among"):
        runtime.call("read_supplementary", {"stem": STEM, "file": "../../wiki/index.md"})
    with pytest.raises(ValueError, match="exact paper stem"):
        runtime.call("read_supplementary", {"stem": "../x", "file": "deg.xlsx"})


def test_a_large_result_is_cut_to_the_read_window():
    rows = [["gene", "note"]] + [[f"G{i}", "x" * 190] for i in range(400)]
    runtime = tools({"big.xlsx": workbook({"S": rows})})

    result = runtime.call("read_supplementary", {"stem": STEM, "file": "big.xlsx", "max_rows": 200})

    assert len(json.dumps(result, ensure_ascii=False)) <= SUPPLEMENTARY_RESULT_CHARS
    assert result["parts"][0]["rows_cut_to_fit"] is True
