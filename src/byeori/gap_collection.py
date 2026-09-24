"""Turn an approved gap into saved candidates: search the lab's journals, save, report the rest.

This is the step after ``lab_collection``. A question the wiki could not answer has been recorded,
a member may have asked for papers on it, and the professor has approved collecting for it. What
happens then, by the user's decisions of 2026-09-22:

- **The search is held to the lab's 65 journals.** Anything outside them cannot be ingested, so
  saving it as a candidate would only fill the catalogue with papers nothing may read.
- **A wide search runs only when the list returns nothing, and only to report.** The professor
  should know that the subject exists outside the lab's journals rather than being told there is
  no literature at all. Those works are never saved, and the refused houses and titles never
  appear at all.
- **Saving stops at a candidate.** No PDF is fetched and no note is written. Ingest stays its own
  step with its own checks (``pipeline.require_allowlisted_journal``).

The search and the save are passed in rather than built here, so the whole decision is testable
without AWS and so the caller decides which account it reaches. ``collect_for_query`` never
raises for one bad work: a candidate that cannot be saved is counted and named, and the rest are.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from byeori.corpus import annotate
from byeori.journal_policy import INCLUDE

__all__ = ["COLLECTION_SCOPE", "DEFAULT_LIMIT", "MAX_LIMIT", "REPORT_SCOPE", "collect_for_query"]

COLLECTION_SCOPE = "list"   # what may be saved: the lab's journals
REPORT_SCOPE = "wide"       # what is only shown: anything else the policy does not refuse
DEFAULT_LIMIT = 25
MAX_LIMIT = 100
SCOPE_TAG = "collection"    # the corpus scope recorded on a candidate saved this way


def collect_for_query(
    query: str,
    *,
    search: Callable[..., Mapping[str, Any]],
    save: Callable[[dict[str, Any]], Mapping[str, Any]],
    limit: int = DEFAULT_LIMIT,
    dry_run: bool = False,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Search for ``query`` and save what the lab may collect; report what it may not.

    ``search(query, limit=..., journal_scope=...)`` returns the report
    ``AwsStore.search_openalex_report`` produces. ``save(work)`` stores one annotated work as a
    candidate and is not called at all when ``dry_run``.
    """
    text = (query or "").strip()
    if not text:
        raise ValueError("a collection run needs a query")
    if not 1 <= int(limit) <= MAX_LIMIT:
        raise ValueError(f"limit must be from 1 to {MAX_LIMIT}")

    found = search(text, limit=int(limit), journal_scope=COLLECTION_SCOPE)
    works = list(found.get("results") or [])
    report: dict[str, Any] = {
        "query": text,
        "job_id": job_id,
        "scope": COLLECTION_SCOPE,
        "found": len(works),
        "saved": [],
        "skipped": [],
        "errors": [],
        "dry_run": bool(dry_run),
        "refused_count": found.get("refused_count", 0),
        "refused_journals": found.get("refused_journals") or [],
        "outside_list": None,
    }

    for work in works:
        annotated = annotate(dict(work), SCOPE_TAG, [f"gap:{job_id}"] if job_id else ["gap"])
        verdict = (annotated.get("corpus") or {}).get("journal_verdict")
        if verdict != INCLUDE:
            # The list search should not return these; a work whose journal metadata is missing can.
            report["skipped"].append({"work_id": work.get("work_id"), "source": work.get("source"),
                                      "reason": (annotated.get("corpus") or {}).get("journal_reason")})
            continue
        if dry_run:
            report["saved"].append({"work_id": work.get("work_id"), "title": work.get("title"),
                                    "source": work.get("source"), "stored": False})
            continue
        try:
            candidate = save(annotated)
        except Exception as exc:  # noqa: BLE001 - one unsavable work never loses the others
            report["errors"].append({"work_id": work.get("work_id"), "error": type(exc).__name__,
                                     "message": str(exc)[:300]})
            continue
        report["saved"].append({"work_id": candidate.get("work_id"), "title": candidate.get("title"),
                                "source": work.get("source"), "stem": candidate.get("stem"),
                                "status": candidate.get("status"), "stored": True})

    if not works:
        # Nothing on the list. Say whether the subject exists elsewhere instead of implying it does
        # not exist at all; none of this is saved.
        wide = search(text, limit=int(limit), journal_scope=REPORT_SCOPE)
        report["outside_list"] = {
            "scope": REPORT_SCOPE,
            "found": len(wide.get("results") or []),
            "refused_count": wide.get("refused_count", 0),
            "refused_journals": wide.get("refused_journals") or [],
            "works": [{"work_id": w.get("work_id"), "title": w.get("title"), "source": w.get("source"),
                       "doi": w.get("doi"), "warning": w.get("journal_warning")}
                      for w in (wide.get("results") or [])],
            "note": ("Outside the lab's journal list and therefore not saved. Ingest refuses these; "
                     "ask the user before moving a title onto the list."),
        }
    report["saved_count"] = len(report["saved"])
    return report
