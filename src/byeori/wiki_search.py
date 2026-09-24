"""Document-ranked search of the AWS wiki's contentless SQLite index."""
from __future__ import annotations

import re
import sqlite3
from typing import Any


QUESTION_STOPWORDS = set(
    "a an the of and or to in for with by on from is are was were be been does do did what which "
    "how why when where who whom whose can could would should may might will shall than then that "
    "this these those it its as at into onto over under between among across about versus vs not no "
    "any all some more most less much many such same other another each per via".split()
)
DOC_TYPES = {"note", "paper", "overview", "question", "concept"}
RESULT_FIELDS = (
    "doc_type", "doc_id", "title", "section", "score", "path", "year", "journal", "doi",
    "category", "s3_key",
)


def search_index(
    connection: sqlite3.Connection,
    query: str,
    limit: int,
    doc_type: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Return at most ``limit`` distinct documents, with their best matching section.

    Questions are excluded unless explicitly requested. Content words are OR'd for BM25
    ranking; when none survive the stopword/length filter, original terms preserve short
    identifiers such as E3. The caller owns the connection and fetches snippets from S3.
    This helper does not read or return page bodies.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    if doc_type is not None and (not isinstance(doc_type, str) or doc_type not in DOC_TYPES):
        raise ValueError("doc_type must be note, paper, overview, question, or concept")
    if category is not None and (
        not isinstance(category, str) or not re.fullmatch(r"[a-z0-9-]+", category)
    ):
        raise ValueError("category must be a lowercase slug such as asd-ndd or long-read")

    words = [
        word for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", query.lower())
        if len(word) > 2 and word not in QUESTION_STOPWORDS
    ] or query.lower().split()
    match = " OR ".join('"' + word.replace('"', '""') + '"' for word in dict.fromkeys(words))
    sql = (
        "SELECT m.doc_type, m.doc_id, d.title, m.section, "
        "bm25(sections, 5.0, 2.0, 1.0) AS score, "
        "d.path, d.year, d.journal, d.doi, d.category, d.s3_key "
        "FROM sections s JOIN section_map m ON m.rowid = s.rowid "
        "JOIN docs d ON d.doc_type = m.doc_type AND d.doc_id = m.doc_id "
        "WHERE sections MATCH ?"
    )
    params: list[Any] = [match]
    if doc_type is None:
        sql += " AND m.doc_type != 'question'"
    else:
        sql += " AND m.doc_type = ?"
        params.append(doc_type)
    if category is not None:
        sql += " AND d.category = ?"
        params.append(category)
    sql += " ORDER BY score, m.doc_type, m.doc_id, m.section"

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    cursor = connection.execute(sql, params)
    try:
        # Apply the limit after document selection, not to an arbitrary multiple of section
        # rows. The first ranked section is the document's best, regardless of its length.
        for row in cursor:
            identity = (row[0], row[1])
            if identity in seen:
                continue
            seen.add(identity)
            item = dict(zip(RESULT_FIELDS, row))
            item["score"] = round(-row[4], 3)
            results.append(item)
            if len(results) >= limit:
                break
    finally:
        cursor.close()
    return results
