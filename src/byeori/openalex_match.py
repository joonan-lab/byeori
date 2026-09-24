"""Submit AWS matching and share DOI lookup/metadata helpers with the worker.

OpenAlex answers up to 50 DOIs in one filtered request. The AWS worker uses this unkeyed
batch endpoint and persists upstream rate-limit waits in its AWS continuation.

Everything OpenAlex returns is stored under an ``openalex_`` prefix. The catalogue's own title,
year, journal and DOI came from the user's llm-wiki frontmatter and the PDF; OpenAlex is a
metadata source, not evidence, and it does not get to overwrite them. ``pmid`` and ``pmcid`` are
the exception: they are identifiers with one correct value, so a missing one is filled in.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from .config import Settings
from .contact import contact_email, user_agent

BATCH = 50
FIELDS = ("id,doi,ids,display_name,publication_year,type,primary_location,authorships,"
          "referenced_works,open_access")

# The unkeyed pool is metered in credits, and a request costs one credit whether it carries one
# DOI or fifty - which is the whole reason this matcher batches. Measured 2026-09-19: a 50-DOI
# request reported credits_used 1 against a limit of 1000. The corpus is ~230 requests, so one
# run fits inside a single window, but the budget is read back from every response rather than
# assumed, because a limit learned from a header can change without notice.
RATE: dict[str, Any] = {}
CREDIT_FLOOR = 40          # pause here rather than spend the last of the window


class OpenAlexRequestError(RuntimeError):
    def __init__(self, code: int):
        super().__init__(f"OpenAlex HTTP {code}")
        self.code = code
        self.retryable = code in (429, 500, 502, 503, 504)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", str(value).strip().lower())
    return doi or None


def normalize_openalex_id(value: Any) -> str | None:
    work_id = str(value or "").strip().rstrip("/").rsplit("/", 1)[-1].upper()
    return work_id if re.fullmatch(r"W\d+", work_id) else None


def _read_rate(response: Any) -> None:
    headers = {k.lower(): v for k, v in response.headers.items()}
    for header, key in (("x-ratelimit-remaining", "remaining"), ("x-ratelimit-limit", "limit"),
                        ("x-ratelimit-reset", "reset"), ("retry-after", "retry_after")):
        try:
            RATE[key] = int(headers[header])
        except (KeyError, TypeError, ValueError):
            if key == "retry_after" and headers.get(header):
                try:
                    RATE[key] = max(1, int(parsedate_to_datetime(headers[header]).timestamp() - time.time()) + 1)
                except (TypeError, ValueError, OverflowError):
                    pass


def fetch_one(doi: str, *, attempts: int = 3) -> dict[str, Any] | None:
    """Singleton lookup, for a DOI a batch did not return. None when OpenAlex does not have it."""
    mailto = f"&mailto={contact_email()}" if contact_email() else ""
    url = ("https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="/")
           + f"?select={FIELDS}{mailto}")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent("byeori/0.1")})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                _read_rate(response)
                return json.load(response)
        except urllib.error.HTTPError as exc:
            _read_rate(exc)
            if exc.code == 404:
                return None
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts:
                raise OpenAlexRequestError(exc.code) from None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)
    return None


def fetch_batch(dois: list[str], *, attempts: int = 4) -> dict[str, dict[str, Any]]:
    """One request for up to 50 DOIs. Returns normalized DOI -> work, missing DOIs simply absent.

    ``per-page`` is the page size, not the DOI count: one DOI can match more than one OpenAlex
    record - a preprint and its published version can carry the same DOI - so asking for
    ``len(dois)`` results silently drops DOIs off the end of the page. 200 is the maximum page
    size and cannot be reached by 50 DOIs.
    """
    if not dois:
        return {}
    mailto = f"&mailto={contact_email()}" if contact_email() else ""
    url = ("https://api.openalex.org/works?filter=doi:"
           + urllib.parse.quote("|".join(dois), safe="|/.")
           + f"&per-page=200&select={FIELDS}{mailto}")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent("byeori/0.1")})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                _read_rate(response)
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            _read_rate(exc)
            # 429 and 5xx are the polite pool asking us to slow down, not a bad DOI.
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts:
                raise OpenAlexRequestError(exc.code) from None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)
    found: dict[str, dict[str, Any]] = {}
    for work in payload.get("results", []):
        key = normalize_doi(work.get("doi"))
        if key:
            previous = found.get(key)
            if previous:
                previous_ids = previous.get("_ambiguous_openalex_ids") or [
                    normalize_openalex_id(previous.get("id")) or str(previous.get("id", ""))]
                work_id = normalize_openalex_id(work.get("id")) or str(work.get("id", ""))
                identities = sorted(set(previous_ids + [work_id]))
                if len(identities) > 1:
                    found[key] = {"doi": key, "_ambiguous_openalex_ids": identities}
                    continue
            found[key] = work
    return found


def work_fields(work: dict[str, Any]) -> dict[str, Any]:
    """The attributes worth keeping, named so they cannot collide with the catalogue's own."""
    ids = work.get("ids") or {}
    source = (work.get("primary_location") or {}).get("source") or {}
    authors = [a.get("author", {}).get("display_name") for a in (work.get("authorships") or [])]
    authors = [author for author in authors if author]
    # Title, year and authors are kept as a cross-check (user, 2026-09-19): they agree with the
    # catalogue 98% and 96% of the time, and it is the remaining few per cent that are worth
    # seeing - a disagreement means either the DOI matched the wrong work or the frontmatter is
    # wrong. They are a second reading of the same fact, never the fact itself; the catalogue's
    # own title, year and authors stay authoritative. Citation count and OpenAlex topics are
    # still out: a citation count is not evidence, and OpenAlex's topics are a second taxonomy
    # competing with the user's categories rather than a check on them.
    fields: dict[str, Any] = {
        "openalex_id": normalize_openalex_id(work.get("id")),
        "openalex_title": work.get("display_name"),
        "openalex_year": work.get("publication_year"),
        # A review has no Methods or Results of its own; knowing which papers are reviews is
        # something the catalogue cannot derive for itself.
        "openalex_type": work.get("type"),
        # Not a duplicate: 42% of the catalogue has no journal, and the ingest allowlist is
        # written in terms of this field.
        "openalex_venue": source.get("display_name"),
        "openalex_authors": authors[:30],
        "openalex_authors_returned_count": len(authors),
        "openalex_authors_truncated": len(authors) > 30,
        "openalex_is_oa": bool((work.get("open_access") or {}).get("is_oa")),
        # Kept whole: the overlap between these and our own openalex_ids is the citation graph
        # across the corpus, and refetching 11,500 papers to get it later would be wasteful.
        "openalex_referenced_works": [w.rsplit("/", 1)[-1] for w in (work.get("referenced_works") or [])],
        "openalex_matched_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "openalex_status": "matched",
    }
    if ids.get("pmid"):
        fields["pmid_openalex"] = ids["pmid"].rsplit("/", 1)[-1]
    if ids.get("pmcid"):
        fields["pmcid_openalex"] = ids["pmcid"].rsplit("/", 1)[-1].upper()
    return {k: v for k, v in fields.items() if v not in (None, [], "")}


# Greek letters are not accented Latin ones, so NFKD leaves them whole and stripping to ASCII
# would delete them: "Nrxn1a" and "Nrxn1\u03b1" are the same gene written two ways, and without
# this map the first would read as a disagreement. Biology writes these constantly.
GREEK = str.maketrans({"\u03b1": "a", "\u03b2": "b", "\u03b3": "g", "\u03b4": "d", "\u03b5": "e", "\u03b6": "z",
                       "\u03b7": "e", "\u03b8": "th", "\u03b9": "i", "\u03ba": "k", "\u03bb": "l", "\u03bc": "u",
                       "\u03bd": "n", "\u03be": "x", "\u03bf": "o", "\u03c0": "p", "\u03c1": "r", "\u03c3": "s",
                       "\u03c2": "s", "\u03c4": "t", "\u03c5": "y", "\u03c6": "f", "\u03c7": "x", "\u03c8": "ps",
                       "\u03c9": "w", "\u00b5": "u"})


def _squash(value: Any) -> str:
    """Compare on content, so a way of writing a character is not reported as a disagreement."""
    import unicodedata
    text = str(value or "").lower().translate(GREEK)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text)


def match_openalex(settings: Settings, *, limit: int = 0, refresh: bool = False, dry_run: bool = False,
                   batch: int = BATCH) -> dict[str, Any]:
    """Submit AWS matching; catalogue selection and all upstream requests stay in AWS."""
    if not 1 <= batch <= BATCH or limit < 0:
        raise ValueError("batch must be 1..50 and limit must be nonnegative")
    if not (settings.aws_bucket and settings.aws_table and settings.aws_ingest_function):
        raise ValueError("AWS bucket, table and ingest function must be explicitly configured")
    payload = {"limit": limit, "refresh": refresh, "batch": batch}
    if dry_run:
        from .aws_store import AwsStore
        return AwsStore(settings)._invoke({"action": "openalex_match_plan", **payload})
    from .runs import start_run
    return start_run(settings, "OpenAlexMatchStateMachineArn", payload)
