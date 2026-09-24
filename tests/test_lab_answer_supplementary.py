"""The answer worker reading a paper's stored supplementary tables in its one lookup.

Since 2026-09-23 the kept supplementary files of 304 papers sit at ``papers/{stem}/supplementary/``
and each note says which file holds what. A value a question turns on -- one gene's fold change in
one cell type -- is only in the table, so the worker may ask for the rows it needs.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

from byeori import lab_answer
from byeori.lab_answer import answer_job
from byeori.lab_jobs import Member, claim, intake, queue
from byeori.lab_policy import LEASE_SECONDS, PACKET_LIMITS, SUPPLEMENTARY_MAX_CHARS, SUPPLEMENTARY_MAX_READS
from byeori.lab_store import ReceiptWriter
from lab_fakes import FakeConverse, MemoryTable, index_connection, member, source_note, tool_use, wiki_with_index
from test_supplementary_reader import DEG, workbook

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
INDEX_KEY = "index/wiki-index-v2.sqlite3"
MODEL = "global.anthropic.claude-opus-5"
BUCKET = "bucket"
QUESTION = "Was regional inheritance stable in the cohort, and how strongly was SNCA changed?"
NOTE_KEY = "wiki/sources/paper-one.md"
OTHER_KEY = "wiki/sources/paper-two.md"
PREFIX = "papers/paper-one/supplementary/"
ANSWER = {
    "answer": "SNCA was higher in dopaminergic neurons (avg_logFC 1.25; Supplementary Data 3, row 4).",
    "citations": [{"key": NOTE_KEY, "section": "Supplementary Files"}],
    "limitations": [], "evidence_state": "sufficient", "unresolved_items": [],
    "maintenance_hint": {"kind": "none", "target_keys": [], "note": ""},
}
USAGE = {"inputTokens": 1200, "outputTokens": 300, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}


class World:
    """One claimed job over two notes; only paper one has stored supplementary files."""

    def __init__(self):
        pages = {NOTE_KEY: source_note(), OTHER_KEY: source_note("Paper two", stem="paper-two")}
        self.s3 = wiki_with_index(pages, INDEX_KEY)
        data = workbook({"Data 3": DEG})
        manifest = {"stem": "paper-one", "guide": PREFIX + "README.md",
                    "files": [{"file": "deg.xlsx", "bytes": len(data), "sha256": "a" * 64, "kind": "data_table",
                               "decision": "upload", "label": "Supplementary Data 3", "uses": ["gene_sets"]},
                              {"file": "rs.pdf", "bytes": 4, "kind": "reporting_summary", "decision": "skip"}]}
        self.s3._store(PREFIX + "deg.xlsx", data)
        self.s3._store(PREFIX + "manifest.json", json.dumps(manifest).encode())
        self.s3._store(PREFIX + "README.md", b"# Supplementary files: paper one\n\nDEGs per cell type.\n")
        self.table = MemoryTable()
        self.receipts = ReceiptWriter(self.s3, BUCKET)
        member(self.table, "m1")
        job = intake(self.table, self.receipts, Member("m1"), {"request_id": "r1", "question": QUESTION}, NOW)
        job = queue(self.table, job["job_id"], period="2026-09", now=NOW)
        self.job = claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, NOW)
        self.job_id = self.job["job_id"]
        self.index = (index_connection(self.s3.objects[INDEX_KEY]), self.s3.etag(INDEX_KEY))

    def run(self, model):
        return answer_job(self.job, table=self.table, receipts=self.receipts, s3=self.s3, bucket=BUCKET,
                          index=self.index, model=model, model_id=MODEL, reasoning="medium", now=NOW,
                          limits=PACKET_LIMITS, remaining_ms=lambda: 850_000)

    def answer(self):
        return self.s3.json(f"runs/lab-questions/{self.job_id}/answer.json")


def sent(model, call):
    return json.loads(model.requests[call - 1]["messages"][-1]["content"][0]["text"])


def run_lookup(*reads):
    w = World()
    model = FakeConverse([tool_use("request_lookup", {"read_supplementary": list(reads)}, usage=USAGE),
                          tool_use("submit_answer", ANSWER, tool_use_id="call-2", usage=USAGE)])
    result = w.run(model)
    return w, model, result


def test_the_packet_says_which_notes_have_stored_supplementary_files():
    w = World()
    model = FakeConverse([tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    documents = {d["key"]: d["has_supplementary"] for d in sent(model, 1)["evidence"]["documents"]}
    assert documents == {NOTE_KEY: True, OTHER_KEY: False}


def test_a_named_table_and_find_return_the_header_and_matching_rows():
    w, model, result = run_lookup({"key": NOTE_KEY, "file": "deg.xlsx", "find": "SNCA"})

    payload = sent(model, 2)
    [read] = payload["supplementary"]
    assert read["note_key"] == NOTE_KEY and read["mode"] == "table" and read["matches_total"] == 2
    part = read["result"]["parts"][0]
    assert [row[0] for row in part["rows"]] == [4, 6] and part["rows"][0][1][:3] == ["SNCA", "DaN", "1.25"]
    assert read["chars"] <= SUPPLEMENTARY_MAX_CHARS
    assert payload["lookup"]["supplementary_read"] == [
        {"note_key": NOTE_KEY, "file": "deg.xlsx", "sheet": None, "find": "SNCA", "mode": "table",
         "matches_total": 2, "chars": read["chars"], "error": None}]
    assert w.answer()["lookup"]["supplementary_read"][0]["matches_total"] == 2
    assert result["status"] == "completed"


def test_find_alone_searches_the_papers_tables_and_nothing_names_the_file_guide():
    _, model, _ = run_lookup({"key": NOTE_KEY, "find": "TH"}, {"key": NOTE_KEY})

    search, guide = sent(model, 2)["supplementary"]
    assert search["mode"] == "search" and search["matches_total"] == 1
    assert [f["file"] for f in search["result"]["files_with_matches"]] == ["deg.xlsx"]
    assert guide["mode"] == "guide" and guide["result"]["text"].startswith("# Supplementary files: paper one")
    assert [f["file"] for f in guide["result"]["files"]] == ["deg.xlsx"]


def test_notes_without_supplementary_files_or_outside_the_packet_are_not_read():
    w, model, _ = run_lookup({"key": OTHER_KEY, "file": "deg.xlsx"}, {"key": "wiki/sources/not-there.md"},
                             {"key": "papers/paper-one/supplementary/deg.xlsx"})

    assert "supplementary" not in sent(model, 2)
    unresolved = w.answer()["unresolved_items"]
    assert any(OTHER_KEY in item for item in unresolved)
    assert any("not-there" in item for item in unresolved)
    assert not any(key.startswith("papers/paper-two/") for key in w.s3.reads)


def test_a_failed_read_is_kept_with_its_reason_and_the_answer_still_completes():
    w, model, result = run_lookup({"key": NOTE_KEY, "file": "rs.pdf"}, {"key": NOTE_KEY, "file": "deg.xlsx",
                                                                        "sheet": "nope", "find": "TH"})

    first, second = sent(model, 2)["supplementary"]
    assert "was not kept" in first["error"] and "no sheet named" in second["error"]
    assert result["status"] == "completed"
    assert sum("supplementary read" in item for item in w.answer()["unresolved_items"]) == 2


def test_at_most_the_policy_number_of_reads_run():
    reads = [{"key": NOTE_KEY, "file": "deg.xlsx", "find": g} for g in ("SNCA", "TH", "GFAP", "APP", "MAPT", "LRRK2")]
    _, model, _ = run_lookup(*reads)

    assert len(sent(model, 2)["supplementary"]) == SUPPLEMENTARY_MAX_READS


def test_the_tool_and_the_prompt_offer_the_read():
    lookup_tool = lab_answer.TOOLS[1]["toolSpec"]
    assert "read_supplementary" in lookup_tool["inputSchema"]["json"]["properties"]
    assert "has_supplementary" in lab_answer.SYSTEM and "read_supplementary" in lab_answer.SYSTEM


def test_the_read_bounds_leave_room_for_the_notes():
    """Four 8,000-character windows, like the originals: a third of the packet's own budget."""
    assert (SUPPLEMENTARY_MAX_READS, SUPPLEMENTARY_MAX_CHARS) == (4, 8_000)
    assert SUPPLEMENTARY_MAX_READS * SUPPLEMENTARY_MAX_CHARS < PACKET_LIMITS.total_bytes / 2
