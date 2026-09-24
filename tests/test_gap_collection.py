"""byeori.gap_collection: an approved gap becomes candidates, and only ones that may be."""
from __future__ import annotations

import pytest

from byeori import gap_collection
from byeori.gap_collection import collect_for_query


def work(work_id: str, source: str, *, title: str = "Autism de novo variants", **extra):
    return {"work_id": work_id, "doi": f"10.1000/{work_id.lower()}", "title": title, "source": source,
            "publication_year": 2025, "authors": ["A"], "topics": [], "is_open_access": True,
            "journal_scope_verdict": "allowed", "journal_warning": None, **extra}


class Search:
    """Scripted search reports per scope, recording how it was called."""

    def __init__(self, listed=(), wide=(), refused=(), wide_refused=()):
        self.listed, self.wide = list(listed), list(wide)
        self.refused, self.wide_refused = list(refused), list(wide_refused)
        self.calls: list[tuple[str, int, str]] = []

    def __call__(self, query, *, limit, journal_scope):
        self.calls.append((query, limit, journal_scope))
        if journal_scope == gap_collection.COLLECTION_SCOPE:
            return {"results": list(self.listed), "refused_count": len(self.refused),
                    "refused_journals": list(self.refused)}
        return {"results": list(self.wide), "refused_count": len(self.wide_refused),
                "refused_journals": list(self.wide_refused)}

    @property
    def scopes(self) -> list[str]:
        return [scope for _q, _l, scope in self.calls]


class Save:
    def __init__(self, fail_on: set[str] | None = None):
        self.saved: list[dict] = []
        self.fail_on = fail_on or set()

    def __call__(self, annotated):
        if annotated["work_id"] in self.fail_on:
            raise ValueError(f"DOI is already stored as another work for {annotated['work_id']}")
        self.saved.append(annotated)
        return {"work_id": annotated["work_id"], "title": annotated["title"], "stem": "stem",
                "status": "candidate"}


# ---------------------------------------------------------------------------------------------
# What gets saved
# ---------------------------------------------------------------------------------------------

def test_the_lab_journals_are_searched_and_their_works_are_saved_as_candidates():
    search = Search(listed=[work("W1", "Nature Genetics"), work("W2", "Neuron")])
    save = Save()

    report = collect_for_query("CHD8 brain overgrowth", search=search, save=save, job_id="j1")

    assert search.scopes == [gap_collection.COLLECTION_SCOPE], "no wide search when the list answered"
    assert report["saved_count"] == 2 and report["errors"] == [] and report["skipped"] == []
    assert [c["work_id"] for c in report["saved"]] == ["W1", "W2"]
    assert all(c["stored"] is True for c in report["saved"])
    corpus = save.saved[0]["corpus"]
    assert corpus["journal_verdict"] == "include"
    assert corpus["scope"] == gap_collection.SCOPE_TAG and corpus["query_ids"] == ["gap:j1"]


def test_nothing_is_saved_without_apply():
    search = Search(listed=[work("W1", "Nature Genetics")])
    save = Save()

    report = collect_for_query("x", search=search, save=save, dry_run=True)

    assert save.saved == [] and report["dry_run"] is True
    assert report["saved_count"] == 1 and report["saved"][0]["stored"] is False


def test_a_work_the_policy_would_not_include_is_skipped_rather_than_saved():
    """The list search should not return one, but a work with no journal name can arrive."""
    search = Search(listed=[work("W1", "Nature Genetics"), work("W5", "Cell Reports"),
                            work("W6", None)])
    save = Save()

    report = collect_for_query("x", search=search, save=save)

    assert [c["work_id"] for c in report["saved"]] == ["W1"]
    assert {s["work_id"] for s in report["skipped"]} == {"W5", "W6"}
    assert [s["reason"] for s in report["skipped"] if s["work_id"] == "W6"] == ["journal metadata missing"]
    assert [saved["work_id"] for saved in save.saved] == ["W1"]


def test_one_work_that_cannot_be_saved_never_loses_the_others():
    search = Search(listed=[work("W1", "Nature Genetics"), work("W2", "Neuron"), work("W3", "Cell")])
    save = Save(fail_on={"W2"})

    report = collect_for_query("x", search=search, save=save)

    assert [c["work_id"] for c in report["saved"]] == ["W1", "W3"]
    assert [e["work_id"] for e in report["errors"]] == ["W2"]
    assert report["errors"][0]["error"] == "ValueError"


# ---------------------------------------------------------------------------------------------
# What is only reported
# ---------------------------------------------------------------------------------------------

def test_an_empty_list_search_looks_wider_and_reports_without_saving():
    """The professor learns the subject exists elsewhere instead of hearing there is nothing."""
    outside = work("W9", "Cell Reports", journal_scope_verdict="outside_list",
                   journal_warning="Cell Reports is not on the lab's journal list; ...")
    search = Search(listed=[], wide=[outside],
                    wide_refused=["Frontiers in Neuroscience", "Scientific Reports"])
    save = Save()

    report = collect_for_query("a subject the lab has nothing on", search=search, save=save)

    assert search.scopes == [gap_collection.COLLECTION_SCOPE, gap_collection.REPORT_SCOPE]
    assert report["saved_count"] == 0 and save.saved == []
    wider = report["outside_list"]
    assert wider["found"] == 1 and wider["works"][0]["source"] == "Cell Reports"
    assert wider["works"][0]["warning"]
    assert wider["refused_journals"] == ["Frontiers in Neuroscience", "Scientific Reports"]
    assert "not saved" in wider["note"]


def test_a_refused_house_is_reported_by_the_search_and_never_reaches_a_candidate():
    search = Search(listed=[work("W1", "Nature Genetics")], refused=["Nutrients", "Scientific Reports"])
    save = Save()

    report = collect_for_query("x", search=search, save=save)

    assert report["refused_count"] == 2 and report["refused_journals"] == ["Nutrients", "Scientific Reports"]
    assert [saved["work_id"] for saved in save.saved] == ["W1"]


def test_a_wide_search_is_never_run_when_the_list_had_anything():
    search = Search(listed=[work("W1", "Nature Genetics")], wide=[work("W9", "Cell Reports")])
    collect_for_query("x", search=search, save=Save())
    assert gap_collection.REPORT_SCOPE not in search.scopes


# ---------------------------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["", "   ", None])
def test_a_run_without_a_query_is_refused(query):
    with pytest.raises(ValueError):
        collect_for_query(query, search=Search(), save=Save())


@pytest.mark.parametrize("limit", [0, -1, gap_collection.MAX_LIMIT + 1])
def test_the_limit_is_bounded(limit):
    with pytest.raises(ValueError):
        collect_for_query("x", search=Search(), save=Save(), limit=limit)


def test_the_limit_reaches_the_search():
    search = Search(listed=[work("W1", "Nature Genetics")])
    collect_for_query("x", search=search, save=Save(), limit=7)
    assert search.calls[0][1] == 7
