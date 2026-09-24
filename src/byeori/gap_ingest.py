"""Bring an approved gap's candidates into the wiki, so the next question can find them.

``gap_collection`` saves candidates for a question the wiki could not answer. Saving is not
reading: a candidate is a row of metadata, and nothing can be cited from it. The user's decision
of 2026-09-22 is that when Byeori decides a paper has to be read, it is ingested and kept, so the
paper is there for every later question rather than opened once and dropped.

What one paper needs, all of it existing machinery:

1. ``pipeline.ingest_step`` puts the OpenAlex-hosted PDF and its GROBID extraction in S3. It
   refuses a journal off the collection list, and only an open-access paper with a Creative
   Commons or public-domain licence has a PDF to fetch at all.
2. ``pipeline.run_stem`` writes the evidence note under ``wiki/sources/`` from that extraction.
3. The search index is rebuilt once at the end, not per paper, because that is what makes the
   notes findable and it reads the whole bucket each time.

Nothing here decides what is worth collecting; the professor already did that by approving the
gap. This module decides only which of the saved candidates can be brought in at all, and says
plainly why each of the others cannot: an ineligible licence and an off-list journal are
different refusals and are reported as such.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from byeori.corpus import eligible_fulltext
from byeori.journal_policy import INCLUDE, journal_verdict

__all__ = ["INELIGIBLE_LICENCE", "OFF_LIST", "blocked_list", "candidates_for_gap", "ingest_for_gap",
           "why_not"]

OFF_LIST = "journal is not on the collection list"
INELIGIBLE_LICENCE = "no OpenAlex-hosted full text under a Creative Commons or public-domain licence"
ALREADY_HERE = "already in the wiki"
MAX_PAPERS = 25


def candidates_for_gap(rows: Sequence[Mapping[str, Any]], job_id: str) -> list[dict[str, Any]]:
    """The saved candidates that came from this gap's search, newest first is not guaranteed."""
    marker = f"gap:{job_id}"
    chosen = []
    for row in rows:
        record = row.get("record") if isinstance(row.get("record"), Mapping) else row
        corpus = record.get("corpus") if isinstance(record.get("corpus"), Mapping) else {}
        if marker in (corpus.get("query_ids") or []):
            chosen.append(dict(row))
    return chosen


def why_not(candidate: Mapping[str, Any]) -> str | None:
    """Why this candidate cannot be brought in, or ``None`` when it can."""
    record = candidate.get("record") if isinstance(candidate.get("record"), Mapping) else candidate
    if journal_verdict(record.get("source"), record.get("source_issn"),
                       record.get("source_id"))["verdict"] != INCLUDE:
        return OFF_LIST
    if not eligible_fulltext(record):
        return INELIGIBLE_LICENCE
    return None


def blocked_list(job_id: str, query: str | None, skipped: Sequence[Mapping[str, Any]],
                 candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The papers this gap needs that automatic collection cannot bring in.

    Serverless ingest takes only OpenAlex-hosted content under a Creative Commons or public-domain
    licence (``.kiro/steering/ingest-policy.md``), so a paper in NEJM, Cell or a Nature Reviews
    title is saved as a candidate and stops there. Measured on 2026-09-22: six of eight candidates
    for one gap, including the very trial report the answer was missing.

    This is the handover to the route that does work for them -- the Optimus scout's legal PDF
    retrieval into ``to-s3/``, or the user fetching one themselves. It carries what somebody needs
    to find the paper: the DOI, the title, the journal and the landing page. It fetches nothing.
    """
    by_id = {c.get("work_id"): c for c in candidates}
    wanted = []
    for item in skipped:
        if item.get("reason") != INELIGIBLE_LICENCE:
            continue                      # an off-list journal is a policy decision, not a gap to fill
        candidate = by_id.get(item.get("work_id")) or {}
        record = candidate.get("record") if isinstance(candidate.get("record"), Mapping) else {}
        wanted.append({
            "work_id": item.get("work_id"),
            "doi": candidate.get("doi") or record.get("doi"),
            "title": candidate.get("title") or record.get("title"),
            "journal": item.get("source"),
            "publication_year": record.get("publication_year"),
            "landing_page_url": record.get("landing_page_url"),
            "stem": candidate.get("stem"),
        })
    return {
        "job_id": job_id,
        "query": query,
        "count": len(wanted),
        "reason": INELIGIBLE_LICENCE,
        "papers": wanted,
        "note": ("These are on the lab's journal list and the wiki needs them, but serverless ingest "
                 "cannot fetch them. Obtain each legally (publisher, PMC, author or institutional "
                 "copy) and place it in the shared to-s3 folder; nothing here downloads anything."),
    }


def ingest_for_gap(
    job_id: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    ingest: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    write_note: Callable[[str], Mapping[str, Any]],
    rebuild_index: Callable[[], Mapping[str, Any]] | None = None,
    already_here: Callable[[str], bool] | None = None,
    limit: int = MAX_PAPERS,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Ingest what may be ingested and write each one's evidence note.

    Every dependency is injected so the decisions are testable without AWS and the caller chooses
    which account is reached. One paper that fails never stops the others: its error is recorded
    against it and the run continues, because a gap with three of five papers read is better than
    a gap with none.
    """
    if not 1 <= int(limit) <= MAX_PAPERS:
        raise ValueError(f"limit must be from 1 to {MAX_PAPERS}")
    report: dict[str, Any] = {"job_id": job_id, "considered": len(candidates), "dry_run": bool(dry_run),
                              "ingested": [], "skipped": [], "errors": [], "index_rebuilt": False}
    for candidate in list(candidates)[: int(limit)]:
        work_id = candidate.get("work_id")
        stem = candidate.get("stem")
        # The catalogue keys an OpenAlex paper by its work id and an uploaded one by its stem, and
        # the note is named after whichever it is. A live run on 2026-09-22 passed the stem for an
        # OpenAlex paper, found no row, and lost two notes whose PDFs were already stored.
        catalog_id = work_id or stem
        record = candidate.get("record") if isinstance(candidate.get("record"), Mapping) else candidate
        source = record.get("source")
        reason = why_not(candidate)
        if reason is not None:
            report["skipped"].append({"work_id": work_id, "source": source, "reason": reason})
            continue
        if already_here is not None and catalog_id and already_here(catalog_id):
            report["skipped"].append({"work_id": work_id, "source": source, "reason": ALREADY_HERE})
            continue
        if dry_run:
            report["ingested"].append({"work_id": work_id, "stem": stem, "source": source,
                                       "title": candidate.get("title"), "stored": False})
            continue
        try:
            ingest(candidate)
        except Exception as exc:  # noqa: BLE001 - one paper never stops the rest
            report["errors"].append({"work_id": work_id, "stage": "ingest", "error": type(exc).__name__,
                                     "message": str(exc)[:300]})
            continue
        note: Mapping[str, Any] = {}
        try:
            note = write_note(catalog_id)
        except Exception as exc:  # noqa: BLE001 - the PDF is stored; the note can be retried
            report["errors"].append({"work_id": work_id, "stem": stem, "stage": "source_note",
                                     "error": type(exc).__name__, "message": str(exc)[:300]})
            continue
        state = note.get("state") or note.get("status")
        # "reviewed" is what run_paper returns when draft and promote published the note;
        # "source_ready" is the uploaded-PDF route's word for the same thing.
        if state not in {"source_ready", "reviewed"}:
            report["errors"].append({"work_id": work_id, "stem": stem, "stage": "source_note",
                                     "error": "not_ready", "message": str(state)[:120],
                                     "problems": note.get("problems")})
            continue
        report["ingested"].append({"work_id": work_id, "stem": stem, "source": source,
                                   "title": candidate.get("title"), "stored": True,
                                   "note_key": f"wiki/sources/{catalog_id}.md"})

    stored = [item for item in report["ingested"] if item.get("stored")]
    if stored and rebuild_index is not None and not dry_run:
        # Once, at the end: a rebuild reads the whole bucket, and the notes are only findable after
        # it runs. Without this the papers are in the wiki and no question can retrieve them.
        try:
            report["index"] = dict(rebuild_index())
            report["index_rebuilt"] = True
        except Exception as exc:  # noqa: BLE001 - the notes are written; the index can be rebuilt again
            report["errors"].append({"stage": "build_index", "error": type(exc).__name__,
                                     "message": str(exc)[:300]})
    report["ingested_count"] = len(stored) if not dry_run else len(report["ingested"])
    return report
