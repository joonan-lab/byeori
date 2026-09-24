"""byeori.gap_ingest: an approved gap's candidates become readable wiki pages, or say why not."""
from __future__ import annotations

import pytest

from byeori import gap_ingest
from byeori.gap_ingest import candidates_for_gap, ingest_for_gap, why_not


def candidate(work_id: str, source: str, *, job: str = "j1", licence: str = "cc-by",
              open_access: bool = True, hosted: bool = True, issn=None):
    record = {"work_id": work_id, "source": source, "is_open_access": open_access,
              "oa_license": licence, "openalex_pdf_url": "https://content.openalex.org/x.pdf" if hosted else None,
              "grobid_xml_url": "https://content.openalex.org/x.xml" if hosted else None,
              "corpus": {"query_ids": [f"gap:{job}"], "scope": "collection"}}
    if issn:
        record["source_issn"] = issn
    return {"work_id": work_id, "stem": f"stem-{work_id.lower()}", "title": f"Paper {work_id}",
            "record": record}


class Run:
    """Records what was asked of AWS without reaching it."""

    def __init__(self, *, ingest_fails=(), note_state="reviewed", note_fails=()):
        self.ingested, self.notes, self.rebuilds = [], [], 0
        self.ingest_fails, self.note_fails = set(ingest_fails), set(note_fails)
        self.note_state = note_state

    def ingest(self, candidate):
        if candidate["work_id"] in self.ingest_fails:
            raise RuntimeError("the PDF could not be fetched")
        self.ingested.append(candidate["work_id"])
        return {"status": "fulltext_ready"}

    def write_note(self, stem):
        if stem in self.note_fails:
            raise RuntimeError("Bedrock refused")
        self.notes.append(stem)
        return {"state": self.note_state}

    def rebuild(self):
        self.rebuilds += 1
        return {"documents": 11_941}


# ---------------------------------------------------------------------------------------------
# Which candidates belong to the gap
# ---------------------------------------------------------------------------------------------

def test_only_the_candidates_this_gaps_search_saved_are_considered():
    rows = [candidate("W1", "Nature Genetics"), candidate("W2", "Neuron", job="other"),
            {"work_id": "W3", "record": {"corpus": {}}}, {"work_id": "W4"}]

    assert [c["work_id"] for c in candidates_for_gap(rows, "j1")] == ["W1"]


# ---------------------------------------------------------------------------------------------
# What may be brought in, and why the rest may not
# ---------------------------------------------------------------------------------------------

def test_a_listed_journal_with_hosted_open_full_text_may_come_in():
    assert why_not(candidate("W1", "Nature Genetics")) is None
    assert why_not(candidate("W2", "Science", issn=["0036-8075"])) is None


@pytest.mark.parametrize("source", ["Cell Reports", "Scientific Reports", "Nutrients", "Brain"])
def test_a_journal_off_the_collection_list_is_refused_with_that_reason(source):
    assert why_not(candidate("W1", source)) == gap_ingest.OFF_LIST


@pytest.mark.parametrize("kwargs", [
    {"licence": "all-rights-reserved"}, {"licence": ""}, {"open_access": False}, {"hosted": False},
])
def test_a_paper_with_no_usable_full_text_is_refused_with_a_different_reason(kwargs):
    """An off-list journal and an unusable licence are not the same refusal."""
    assert why_not(candidate("W1", "Nature Genetics", **kwargs)) == gap_ingest.INELIGIBLE_LICENCE


def test_the_two_refusals_are_reported_separately():
    run = Run()
    rows = [candidate("W1", "Nature Genetics"), candidate("W2", "Cell Reports"),
            candidate("W3", "Neuron", licence="all-rights-reserved")]

    report = ingest_for_gap("j1", rows, ingest=run.ingest, write_note=run.write_note,
                            rebuild_index=run.rebuild, dry_run=False)

    assert [i["work_id"] for i in report["ingested"]] == ["W1"]
    assert {(s["work_id"], s["reason"]) for s in report["skipped"]} == {
        ("W2", gap_ingest.OFF_LIST), ("W3", gap_ingest.INELIGIBLE_LICENCE)}


# ---------------------------------------------------------------------------------------------
# The paper stays: ingest, note, and one index rebuild
# ---------------------------------------------------------------------------------------------

def test_each_paper_gets_its_evidence_note_and_the_index_is_rebuilt_once():
    """A note nobody can search for is not in the wiki in any useful sense."""
    run = Run()
    rows = [candidate("W1", "Nature Genetics"), candidate("W2", "Neuron"), candidate("W3", "Cell")]

    report = ingest_for_gap("j1", rows, ingest=run.ingest, write_note=run.write_note,
                            rebuild_index=run.rebuild, dry_run=False)

    assert run.ingested == ["W1", "W2", "W3"]
    assert run.notes == ["W1", "W2", "W3"], "the catalogue keys an OpenAlex paper by its work id"
    assert run.rebuilds == 1 and report["index_rebuilt"] is True
    assert [i["note_key"] for i in report["ingested"]] == [
        "wiki/sources/W1.md", "wiki/sources/W2.md", "wiki/sources/W3.md"]


def test_nothing_is_fetched_or_written_without_apply():
    run = Run()
    rows = [candidate("W1", "Nature Genetics")]

    report = ingest_for_gap("j1", rows, ingest=run.ingest, write_note=run.write_note,
                            rebuild_index=run.rebuild, dry_run=True)

    assert (run.ingested, run.notes, run.rebuilds) == ([], [], 0)
    assert report["ingested_count"] == 1 and report["ingested"][0]["stored"] is False


def test_a_paper_already_in_the_wiki_is_not_fetched_again():
    run = Run()
    rows = [candidate("W1", "Nature Genetics"), candidate("W2", "Neuron")]

    report = ingest_for_gap("j1", rows, ingest=run.ingest, write_note=run.write_note,
                            rebuild_index=run.rebuild, dry_run=False,
                            already_here=lambda catalog_id: catalog_id == "W1")

    assert run.ingested == ["W2"]
    assert [(s["work_id"], s["reason"]) for s in report["skipped"]] == [("W1", gap_ingest.ALREADY_HERE)]


# ---------------------------------------------------------------------------------------------
# One failure never loses the rest
# ---------------------------------------------------------------------------------------------

def test_a_paper_whose_pdf_cannot_be_fetched_is_recorded_and_the_others_continue():
    run = Run(ingest_fails={"W2"})
    rows = [candidate("W1", "Nature Genetics"), candidate("W2", "Neuron"), candidate("W3", "Cell")]

    report = ingest_for_gap("j1", rows, ingest=run.ingest, write_note=run.write_note,
                            rebuild_index=run.rebuild, dry_run=False)

    assert run.ingested == ["W1", "W3"] and run.notes == ["W1", "W3"]
    assert [(e["work_id"], e["stage"]) for e in report["errors"]] == [("W2", "ingest")]
    assert report["ingested_count"] == 2 and run.rebuilds == 1


def test_a_note_that_will_not_write_leaves_the_pdf_stored_and_says_so():
    run = Run(note_fails={"W1"})
    rows = [candidate("W1", "Nature Genetics"), candidate("W2", "Neuron")]

    report = ingest_for_gap("j1", rows, ingest=run.ingest, write_note=run.write_note,
                            rebuild_index=run.rebuild, dry_run=False)

    assert run.ingested == ["W1", "W2"], "the PDF is in S3; only the note failed"
    assert [(e["work_id"], e["stage"]) for e in report["errors"]] == [("W1", "source_note")]
    assert [i["work_id"] for i in report["ingested"]] == ["W2"]


def test_a_note_that_comes_back_unready_is_an_error_not_a_success():
    run = Run(note_state="source_failed")
    rows = [candidate("W1", "Nature Genetics")]

    report = ingest_for_gap("j1", rows, ingest=run.ingest, write_note=run.write_note,
                            rebuild_index=run.rebuild, dry_run=False)

    assert report["ingested"] == [] and report["errors"][0]["error"] == "not_ready"
    assert run.rebuilds == 0, "nothing was added, so there is nothing to reindex"


def test_an_index_rebuild_that_fails_does_not_undo_the_papers():
    def rebuild():
        raise RuntimeError("the index build timed out")

    run = Run()
    report = ingest_for_gap("j1", [candidate("W1", "Nature Genetics")], ingest=run.ingest,
                            write_note=run.write_note, rebuild_index=rebuild, dry_run=False)

    assert report["ingested_count"] == 1 and report["index_rebuilt"] is False
    assert report["errors"][0]["stage"] == "build_index"


@pytest.mark.parametrize("limit", [0, -1, gap_ingest.MAX_PAPERS + 1])
def test_the_limit_is_bounded(limit):
    with pytest.raises(ValueError):
        ingest_for_gap("j1", [], ingest=Run().ingest, write_note=Run().write_note, limit=limit)


# ---------------------------------------------------------------------------------------------
# The papers automatic collection cannot bring in
# ---------------------------------------------------------------------------------------------

def test_the_blocked_list_names_only_the_papers_a_licence_stops():
    """Measured 2026-09-22: six of eight candidates for one gap, the needed trial report among them."""
    rows = [candidate("W1", "New England Journal of Medicine", licence="all-rights-reserved"),
            candidate("W2", "Cell Reports"),
            candidate("W3", "Nature Genetics")]
    rows[0]["doi"] = "10.1056/NEJMoa2107454"
    rows[0]["record"]["publication_year"] = 2021
    rows[0]["record"]["landing_page_url"] = "https://www.nejm.org/doi/full/10.1056/NEJMoa2107454"
    skipped = [{"work_id": "W1", "source": "New England Journal of Medicine",
                "reason": gap_ingest.INELIGIBLE_LICENCE},
               {"work_id": "W2", "source": "Cell Reports", "reason": gap_ingest.OFF_LIST}]

    blocked = gap_ingest.blocked_list("j1", "crispr base editing transthyretin", skipped, rows)

    assert blocked["count"] == 1 and blocked["reason"] == gap_ingest.INELIGIBLE_LICENCE
    paper = blocked["papers"][0]
    assert paper["doi"] == "10.1056/NEJMoa2107454" and paper["journal"] == "New England Journal of Medicine"
    assert paper["publication_year"] == 2021 and paper["landing_page_url"].startswith("https://")
    assert "to-s3" in blocked["note"] and "nothing here downloads" in blocked["note"]


def test_an_off_list_journal_is_not_a_paper_to_go_and_fetch():
    """Refusing a journal is a decision; a licence is an obstacle. Only the second goes on the list."""
    skipped = [{"work_id": "W2", "source": "Scientific Reports", "reason": gap_ingest.OFF_LIST},
               {"work_id": "W3", "source": "Nature Genetics", "reason": gap_ingest.ALREADY_HERE}]

    assert gap_ingest.blocked_list("j1", None, skipped, [])["papers"] == []


def test_a_gap_where_everything_could_be_ingested_has_an_empty_list():
    assert gap_ingest.blocked_list("j1", "q", [], [])["count"] == 0


@pytest.mark.parametrize("state", ["reviewed", "source_ready"])
def test_both_ingest_routes_words_for_a_written_note_count_as_ready(state):
    """run_paper says "reviewed" and run_stem says "source_ready"; both mean the note exists."""
    run = Run(note_state=state)
    report = ingest_for_gap("j1", [candidate("W1", "Nature Genetics")], ingest=run.ingest,
                            write_note=run.write_note, rebuild_index=run.rebuild, dry_run=False)
    assert report["ingested_count"] == 1 and report["errors"] == []
