"""Exercise document selection against the deployed index's actual SQLite schema."""
from __future__ import annotations

import sqlite3

import pytest

from byeori.wiki_search import search_index


@pytest.fixture
def index():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE docs (doc_type TEXT, doc_id TEXT, title TEXT, path TEXT, year TEXT, "
        "journal TEXT, doi TEXT, work_ids TEXT, category TEXT, s3_key TEXT, "
        "PRIMARY KEY (doc_type, doc_id));"
        "CREATE VIRTUAL TABLE sections USING fts5(title, section, content, "
        "tokenize='porter unicode61', content='');"
        "CREATE TABLE section_map (rowid INTEGER PRIMARY KEY, doc_type TEXT, doc_id TEXT, section TEXT);"
    )
    yield connection
    connection.close()


def add_page(connection, ident, sections, *, doc_type="note", title=None, category="asd-ndd"):
    title = title or ident
    folder = {"note": "sources", "question": "questions", "overview": "overviews",
              "concept": "concepts", "paper": "asd-ndd"}[doc_type]
    connection.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?)", (
        doc_type, ident, title, f"data/wiki/{folder}/{ident}.md", "2026", "Nature",
        f"10.1/{ident}", "", category, f"wiki/{folder}/{ident}.md",
    ))
    for heading, content in sections:
        cursor = connection.execute(
            "INSERT INTO sections(title, section, content) VALUES (?,?,?)", (title, heading, content),
        )
        connection.execute("INSERT INTO section_map VALUES (?,?,?,?)", (
            cursor.lastrowid, doc_type, ident, heading,
        ))


def test_many_sections_cannot_consume_other_documents_result_slots(index):
    add_page(index, "many", [(f"Section {i}", "CHD8 chromatin") for i in range(100)], title="CHD8")
    add_page(index, "second", [("Results", "background " * 200 + "CHD8")])
    add_page(index, "third", [("Results", "background " * 300 + "CHD8")])
    crowded_rows = index.execute(
        "SELECT m.doc_id FROM sections s JOIN section_map m ON s.rowid = m.rowid "
        "WHERE sections MATCH 'CHD8' ORDER BY bm25(sections, 5.0, 2.0, 1.0) LIMIT 24",
    ).fetchall()
    assert {row[0] for row in crowded_rows} == {"many"}, "The old 3 * 8 section cap would return one document"

    results = search_index(index, "CHD8", 3)

    assert len(results) == 3
    assert results[0]["doc_id"] == "many"
    assert {row["doc_id"] for row in results} == {"many", "second", "third"}
    assert index.execute("SELECT count(*) FROM docs").fetchone()[0] == 3


def test_selects_best_section_and_preserves_result_contract(index):
    add_page(index, "evidence", [
        ("Unrelated", "background " * 200),
        ("Direct evidence", "RFWD2 " * 8),
        ("Incidental", "background " * 200 + "RFWD2"),
    ])

    result = search_index(index, "RFWD2", 10)[0]

    assert result["section"] == "Direct evidence"
    assert set(result) == {"doc_type", "doc_id", "title", "section", "score", "path", "year",
                           "journal", "doi", "category", "s3_key"}
    assert result["s3_key"] == "wiki/sources/evidence.md"
    assert result["doi"] == "10.1/evidence"
    assert "content" not in result


def test_questions_are_excluded_by_default_even_when_the_title_matches(index):
    add_page(index, "self-answer", [("Question", "RFWD2 dosage")],
             doc_type="question", title="RFWD2 dosage")
    add_page(index, "source", [("Results", "RFWD2 dosage")])
    add_page(index, "synthesis", [("Summary", "RFWD2 dosage")], doc_type="overview")

    results = search_index(index, "RFWD2 dosage", 10)

    assert {row["doc_id"] for row in results} == {"source", "synthesis"}


def test_explicit_question_type_is_available(index):
    add_page(index, "self-answer", [("Question", "RFWD2 dosage")], doc_type="question")
    add_page(index, "source", [("Results", "RFWD2 dosage")])

    results = search_index(index, "RFWD2 dosage", 10, doc_type="question")

    assert [row["doc_id"] for row in results] == ["self-answer"]


def test_type_and_category_filters_apply_before_document_selection(index):
    add_page(index, "source", [("Results", "CHD8")])
    add_page(index, "other-category", [("Results", "CHD8")], category="cancer")
    add_page(index, "concept", [("Definition", "CHD8")], doc_type="concept")

    results = search_index(index, "CHD8", 1, doc_type="note", category="asd-ndd")

    assert [row["doc_id"] for row in results] == ["source"]


@pytest.mark.parametrize("query", ["E3", "X", "10"])
def test_short_identifiers_survive_when_no_content_word_remains(index, query):
    add_page(index, "short-term", [("Results", f"An identifier {query} occurs here.")])
    add_page(index, "unrelated", [("Results", "Different biology.")])

    assert [row["doc_id"] for row in search_index(index, query, 10)] == ["short-term"]


def test_content_terms_are_ored_instead_of_requiring_every_word(index):
    add_page(index, "rfwd2", [("Results", "RFWD2 dosage")])
    add_page(index, "ube3a", [("Results", "UBE3A synapses")])

    assert {row["doc_id"] for row in search_index(index, "Does RFWD2 converge with UBE3A?", 10)} == {
        "rfwd2", "ube3a",
    }


def test_nonmatching_query_returns_empty_result(index):
    add_page(index, "source", [("Results", "RFWD2")])
    assert search_index(index, "Schwarz", 10) == []


@pytest.mark.parametrize("kwargs", [
    {"query": ""}, {"query": "   "}, {"query": None},
    {"limit": 0}, {"limit": 101}, {"limit": True}, {"limit": "10"},
    {"doc_type": "source"}, {"doc_type": []},
    {"category": "../sources"}, {"category": "ASD"},
])
def test_invalid_input_is_rejected(index, kwargs):
    options = {"query": "RFWD2", "limit": 10, **kwargs}
    with pytest.raises(ValueError):
        search_index(index, **options)
