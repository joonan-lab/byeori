"""The answer worker opening the paper itself when a note does not settle a point.

The user's rule of 2026-09-20: answer from the wiki, and when the wiki has nothing, read the
original. A source note is a summary, so a sample size in a table or a value in a figure caption
is often only in the paper. Since 2026-09-22 the worker may ask for that text, and for the figure
and table text uploaded beside it, in the one supplemental call it already had.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from byeori import evidence_packet, lab_answer
from byeori.lab_answer import answer_job
from byeori.lab_jobs import Member, claim, intake, queue
from byeori.lab_policy import ASSET_MAX_CHARS, LEASE_SECONDS, ORIGINAL_MAX_CHARS, ORIGINAL_MAX_PAPERS, PACKET_LIMITS
from byeori.lab_store import ReceiptWriter, keys
from lab_fakes import (
    FakeConverse,
    MemoryTable,
    index_connection,
    member,
    paper_assets,
    paper_text,
    source_note,
    tool_use,
    wiki_with_index,
)

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
INDEX_KEY = "index/wiki-index-v2.sqlite3"
MODEL = "global.anthropic.claude-opus-5"
BUCKET = "bucket"
QUESTION = "Was regional inheritance stable in the cohort, and in how many probands?"
NOTE_KEY = "wiki/sources/paper-one.md"
PDF = f"s3://{BUCKET}/papers/paper-one/original.pdf"
CLEAN = "papers/paper-one/clean.md"
ASSETS = "papers/paper-one/assets/assets.md"
ANSWER = {
    "answer": "Regional inheritance was stable across 312 probands (OR 2.4).",
    "citations": [{"key": NOTE_KEY, "section": "Results"}],
    "limitations": [], "evidence_state": "sufficient", "unresolved_items": [],
    "maintenance_hint": {"kind": "none", "target_keys": [], "note": ""},
}
USAGE = {"inputTokens": 1200, "outputTokens": 300, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}


class World:
    """One claimed job over a wiki whose single note points at a stored paper with assets."""

    def __init__(self, *, with_paper: bool = True, with_assets: bool = True, question: str = QUESTION):
        pages = {NOTE_KEY: source_note(pdf_path=PDF),
                 "wiki/sources/paper-two.md": source_note("Paper two", stem="paper-two")}
        self.s3 = wiki_with_index(pages, INDEX_KEY)
        if with_paper:
            self.s3._store(CLEAN, paper_text().encode("utf-8"))
        if with_assets:
            self.s3._store(ASSETS, paper_assets().encode("utf-8"))
        self.table = MemoryTable()
        self.receipts = ReceiptWriter(self.s3, BUCKET)
        member(self.table, "m1")
        job = intake(self.table, self.receipts, Member("m1"), {"request_id": "r1", "question": question}, NOW)
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

    def packet_sent(self, model, call: int):
        """The JSON packet the model saw on the given call."""
        import json
        message = model.requests[call - 1]["messages"][-1]
        return json.loads(message["content"][0]["text"])


def lookup(**arguments):
    return tool_use("request_lookup", arguments, usage=USAGE)


# ---------------------------------------------------------------------------------------------
# The paper reaches the model
# ---------------------------------------------------------------------------------------------

def test_a_named_paper_arrives_with_its_text_and_its_figure_captions():
    w = World()
    model = FakeConverse([lookup(read_originals=[{"key": NOTE_KEY}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    result = w.run(model)

    assert result["status"] == "completed"
    packet = w.packet_sent(model, 2)
    originals = packet["originals"]
    assert [o["note_key"] for o in originals] == [NOTE_KEY]
    assert originals[0]["source_key"] == CLEAN
    # The number the note does not carry is in the paper, and the figure text came with it.
    assert "312 probands" in originals[0]["text"]
    assert "Odds ratio by ancestry group" in originals[0]["assets"]["text"]
    assert "n = 118, 97 and 97" in originals[0]["assets"]["text"]


def test_a_named_section_returns_that_section_rather_than_the_outline():
    w = World()
    model = FakeConverse([lookup(read_originals=[{"key": NOTE_KEY, "section": "Results"}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    original = w.packet_sent(model, 2)["originals"][0]
    assert original["mode"] == "section" and original["section"] == "Results"
    assert "312 probands" in original["text"] and "Recruitment, sequencing" not in original["text"]


def test_the_answer_records_which_papers_were_opened():
    w = World()
    model = FakeConverse([lookup(read_originals=[{"key": NOTE_KEY}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    read = w.answer()["lookup"]["originals_read"]
    assert [r["note_key"] for r in read] == [NOTE_KEY]
    assert read[0]["source_key"] == CLEAN and read[0]["chars"] > 0 and read[0]["assets_chars"] > 0


def test_a_paper_with_no_stored_figures_still_returns_its_text():
    w = World(with_assets=False)
    model = FakeConverse([lookup(read_originals=[{"key": NOTE_KEY}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    original = w.packet_sent(model, 2)["originals"][0]
    assert "312 probands" in original["text"] and "assets" not in original


# ---------------------------------------------------------------------------------------------
# What may be opened
# ---------------------------------------------------------------------------------------------

def test_only_a_paper_already_in_the_packet_may_be_opened():
    """The model names notes it was given; it cannot reach an arbitrary key."""
    w = World()
    model = FakeConverse([lookup(read_originals=[{"key": "wiki/sources/never-retrieved.md"}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    packet = w.packet_sent(model, 2)
    assert "originals" not in packet
    assert any("not in the packet" in item for item in w.answer()["unresolved_items"])


@pytest.mark.parametrize("bad", ["papers/paper-one/clean.md", "../../etc/passwd", "index/wiki-index-v2.sqlite3", ""])
def test_a_key_that_is_not_a_wiki_page_never_reaches_s3(bad):
    w = World()
    before = list(w.s3.reads)
    model = FakeConverse([lookup(read_originals=[{"key": bad}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    assert "originals" not in w.packet_sent(model, 2)
    assert bad not in w.s3.reads[len(before):]


def test_a_note_with_no_stored_paper_is_reported_rather_than_skipped_silently():
    w = World(with_paper=False)
    model = FakeConverse([lookup(read_originals=[{"key": NOTE_KEY}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    assert "originals" not in w.packet_sent(model, 2)
    assert any("could not read the paper" in item or "no stored full text" in item
               for item in w.answer()["unresolved_items"])


def test_the_same_paper_named_twice_is_opened_once():
    w = World()
    model = FakeConverse([lookup(read_originals=[{"key": NOTE_KEY}, {"key": NOTE_KEY}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    assert len(w.packet_sent(model, 2)["originals"]) == 1


def test_more_papers_than_the_bound_are_cut_to_it():
    w = World()
    many = [{"key": f"wiki/sources/paper-{n}.md"} for n in range(ORIGINAL_MAX_PAPERS + 3)]
    model = FakeConverse([lookup(read_originals=many), tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    # Only the bound is considered at all; the rest are never looked at, named or reported.
    reported = [item for item in w.answer()["unresolved_items"] if "not in the packet" in item]
    assert len(reported) <= ORIGINAL_MAX_PAPERS


# ---------------------------------------------------------------------------------------------
# Bounds and the tool contract
# ---------------------------------------------------------------------------------------------

def test_the_tool_offers_read_originals_and_the_prompt_says_when_to_use_it():
    schema = lab_answer.TOOLS[1]["toolSpec"]["inputSchema"]["json"]["properties"]
    assert set(schema) == {"english_query", "read", "read_originals", "read_supplementary"}
    assert "read_originals" in lab_answer.SYSTEM
    assert "a note is a summary" in lab_answer.SYSTEM.lower()


def test_the_read_bounds_are_the_policy_values():
    """One read window is the same 8,000 characters any excerpt is capped at."""
    assert (ORIGINAL_MAX_PAPERS, ORIGINAL_MAX_CHARS, ASSET_MAX_CHARS) == (4, 8_000, 6_000)
    assert ORIGINAL_MAX_PAPERS * ORIGINAL_MAX_CHARS < PACKET_LIMITS.total_bytes / 2


def test_both_ingest_routes_find_their_papers_figures():
    """The crops go to papers/{stem}/assets/ whichever route stored the paper's text."""
    assert evidence_packet.assets_key_for_source(CLEAN) == ASSETS
    assert evidence_packet.assets_key_for_source("sources/W123.md") == "papers/W123/assets/assets.md"


def test_nothing_that_is_not_an_extraction_has_figures():
    for other in ("wiki/sources/paper-one.md", "papers/x/original.pdf", "sources/nested/x.md",
                  "index/documents.json", None, 7):
        assert evidence_packet.assets_key_for_source(other) is None


def test_reading_a_paper_writes_nothing():
    w = World()
    before = len(w.s3.writes)
    model = FakeConverse([lookup(read_originals=[{"key": NOTE_KEY}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    written = [key for key, _ in w.s3.writes[before:]]
    assert all(key.startswith(("runs/lab-questions/", "wiki/lab-questions/")) for key in written), written
    assert CLEAN not in written and ASSETS not in written


def test_the_evidence_says_which_notes_have_a_paper_behind_them():
    """Without this the model cannot know a paper is openable, and a live question showed it asking
    for another search instead (2026-09-22)."""
    w = World()
    model = FakeConverse([tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    documents = w.packet_sent(model, 1)["evidence"]["documents"]
    by_key = {d["key"]: d for d in documents}
    assert by_key[NOTE_KEY]["has_original"] is True
    assert by_key["wiki/sources/paper-two.md"]["has_original"] is False


def test_the_prompt_tells_the_model_to_open_a_paper_alongside_any_search():
    """A live run spent its one lookup on a search and said so itself (2026-09-22), so the tool and
    the prompt both have to say the fields combine rather than compete."""
    system = lab_answer.SYSTEM
    assert "has_original: true" in system
    assert "put read_originals and any search in the same call" in system
    assert "read_originals is required, not optional" in system
    description = lab_answer.TOOLS[1]["toolSpec"]["description"]
    assert "not alternatives" in description and "in the same call is the normal use" in description
    # The field the model reaches for least is the one it sees first.
    assert list(lab_answer.TOOLS[1]["toolSpec"]["inputSchema"]["json"]["properties"])[0] == "read_originals"


def test_a_heading_the_paper_does_not_have_returns_the_paper_instead_of_losing_it():
    """A live run named the note's heading for the paper and lost two of four papers (2026-09-22).

    There is no second lookup to correct it, so the paper comes back with its own headings and the
    substitution is recorded rather than the read failing.
    """
    w = World()
    model = FakeConverse([lookup(read_originals=[{"key": NOTE_KEY, "section": "4. Key Results and Benchmarks"}]),
                          tool_use("submit_answer", ANSWER, usage=USAGE)])

    w.run(model)

    original = w.packet_sent(model, 2)["originals"][0]
    assert original["mode"] == "outline" and "312 probands" in original["text"]
    assert any("has no section named" in item for item in w.answer()["unresolved_items"])
    read = w.answer()["lookup"]["originals_read"][0]
    assert read["chars"] > 0


def test_the_prompt_warns_that_a_paper_has_its_own_headings():
    assert "never a note heading" in lab_answer.SYSTEM
