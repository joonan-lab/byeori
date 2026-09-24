"""Bounded evidence packets over the shared BM25 index (docs/LAB-QUESTION-WORKFLOW.md section 5).

The student answer worker must not read whole pages or the top heading alone. This module opens
the AWS index read-only, ranks candidates with the campaign's ``wiki_search.search_index``, reads
each chosen page once and pins the version it read (ETag, VersionId, sha256), then selects the
best-matching section together with results, methods or interpretation and always a limitations
section when one exists. Managed backlink and catalog blocks are stripped before outlining, so
section offsets are character offsets into ``page_body(text)``. Every drop or cut is recorded.

Page text is untrusted input: pages above ``MAX_PAGE_BYTES`` are refused, and every scan over a
page (headings, list markers, link markup, managed blocks) is linear in its length, so no page
that a search can reach or a model can request stalls the answer worker. A single unreadable or
undecodable page is recorded in the packet's ``omitted`` list and never aborts the packet.

Nothing here writes to S3. Reads returned to clients keep the 8,000-character cap.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from botocore.exceptions import ClientError

from .lab_policy import PACKET_LIMITS, READ_MAX_CHARS, PacketLimits
from .wiki_connections import BACKLINK_END, BACKLINK_START, CATALOG_END, CATALOG_START
from .wiki_search import QUESTION_STOPWORDS, search_index

__all__ = ["PacketLimits", "PACKET_LIMITS", "PageVersion", "Section", "open_index", "clean_query", "search",
           "read_page", "page_body", "strip_managed_blocks", "outline", "classify", "select_sections",
           "build_packet", "read_excerpt", "backlinks"]

EVIDENCE_STATES = ("sufficient", "links_only", "truncated_decisive", "insufficient")
SECTION_KINDS = ("results", "methods", "interpretation", "limitations", "links", "other")
DECISIVE_KINDS = frozenset({"results", "limitations"})
# A note's headings are fixed, so its kinds are read off them. A synthesis page names its sections
# in its own words - "Five constructions of the axis", "Which models exist here, and which do not" -
# and the kind patterns call all of them `other`. On 2026-09-24 that left a liver question with 2 of
# an overview's 8 sections, and the answer said so: it had to reconstruct the taxonomy indirectly
# because the sections describing it were never delivered. A synthesis page has no background prose
# to hold back; it is the cross-paper conclusion, so all of it is evidence.
SYNTHESIS_TYPES = frozenset({"concept", "overview"})

# Folder mapping shared with the campaign's question agent; paper pages keep their indexed path.
TYPE_FOLDERS = {"note": "sources", "concept": "concepts", "overview": "overviews", "question": "questions"}
FOLDER_TYPES = {folder: doc_type for doc_type, folder in TYPE_FOLDERS.items()}

MISSING_CODES = {"NoSuchKey", "404", "NotFound"}
MAX_PAGE_BYTES = 512_000  # a wiki page is a Markdown note, never a data dump; refuse anything larger
MAX_KEY_BYTES = 512  # S3 allows 1,024; published wiki keys are far shorter
TITLE_MAX_CHARS = 300  # page-derived strings outside the byte budget are bounded so a page cannot smuggle text
NAME_MAX_CHARS = 200
OPENING = "(opening)"
LINK_SECTION_NAMES = frozenset({"linked pages", "connected wiki pages"})
LINK_TEXT_RATIO = 0.10

# Every pattern applied to page text must run in linear time on adversarial input: no nested or
# adjacent quantifiers over overlapping character classes, and every repeated class excludes the
# character that opens the construct (a run of ``[`` or ``(`` is otherwise rescanned from each
# position) as well as the newline. Heading names are stripped afterwards.
HEADING = re.compile(r"^(#{2,3}) (.*)$", re.M)
TITLE_LINE = re.compile(r"\s*# [^\n]*\n?")
LINK_MARKUP = re.compile(r"\[\[[^\[\]\n]*\]\]|\[[^\[\]\n]*\]\([^()\n]*\)|https?://\S+")
LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]*", re.M)
WORD = re.compile(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣\-']+")
SAFE_KEY = re.compile(r"[A-Za-z0-9._/-]+")
# Control characters removed from search queries; newline and tab are kept as word separators.
QUERY_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# Case-insensitive heading keywords, English and Korean. Limitations are checked first because
# that is the section the selection must never lose.
KIND_PATTERNS = (
    ("limitations", re.compile(r"limitation|caveat|counter|contradict|open question|untested|한계|반례|미해결", re.I)),
    ("results", re.compile(r"result|finding|결과", re.I)),
    ("methods", re.compile(r"method|design|cohort|sample|방법|설계", re.I)),
    ("interpretation", re.compile(r"discussion|interpretation|synthesis|conclusion|해석|논의|종합", re.I)),
)
METADATA_FIELDS = ("title", "doi", "journal", "year", "publication_year", "review_status", "category")


@dataclass(frozen=True)
class PageVersion:
    """One page exactly as read: the text and the version identifiers that pin it."""

    key: str
    text: str
    etag: str
    version_id: str
    sha256: str


@dataclass(frozen=True)
class Section:
    """A heading's content with its offsets into ``page_body`` and its evidence kind."""

    name: str
    order: int
    kind: str
    start: int
    end: int
    text: str


# ---------------------------------------------------------------------------------------------
# Index access and search
# ---------------------------------------------------------------------------------------------

def open_index(s3, bucket: str, key: str, cache_dir) -> tuple[sqlite3.Connection, str]:
    """Open the shared index read-only, downloading only when the S3 ETag changed."""
    head = s3.head_object(Bucket=bucket, Key=key)
    etag = str(head.get("ETag", ""))
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    local = directory / Path(key).name
    marker = directory / (local.name + ".etag")
    if not local.exists() or not marker.exists() or marker.read_text() != etag:
        s3.download_file(bucket, key, str(local))
        marker.write_text(etag)
    return sqlite3.connect(f"file:{quote(str(local))}?mode=ro", uri=True), etag


def _connection(index) -> tuple[sqlite3.Connection, str | None]:
    """Accept a connection or the ``(connection, etag)`` pair ``open_index`` returns."""
    if isinstance(index, sqlite3.Connection):
        return index, None
    connection, etag = index
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("index must be a sqlite3.Connection or an (connection, etag) pair")
    return connection, None if etag is None else str(etag)


def hit_key(hit: Mapping[str, Any]) -> str:
    """The S3 key of a search hit, with the campaign's folder mapping."""
    folder = TYPE_FOLDERS.get(str(hit.get("doc_type")))
    if folder:
        return f"wiki/{folder}/{hit['doc_id']}.md"
    return str(hit.get("path") or "").removeprefix("data/")


def clean_query(query: Any) -> str:
    """The query without control characters (newline and tab kept); ``ValueError`` when nothing is left.

    A NUL byte inside an FTS5 quoted phrase terminates the string and raises
    ``sqlite3.OperationalError`` from the index, so it is removed before the query reaches SQLite.
    """
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    cleaned = QUERY_CONTROL.sub("", query)
    if not cleaned.strip():
        raise ValueError("query must not be empty")
    return cleaned


def search(connection: sqlite3.Connection, query: str, limit: int, doc_type: str | None = None) -> list[dict[str, Any]]:
    """BM25 document hits with their S3 ``key``; questions only when ``doc_type`` asks for them.

    Control characters are stripped from ``query`` first; a query that is empty afterwards
    raises ``ValueError`` like an empty query does.
    """
    hits = []
    for row in search_index(connection, clean_query(query), limit, doc_type):
        hits.append({"key": hit_key(row), "doc_type": row["doc_type"], "doc_id": row["doc_id"], "title": row["title"],
                     "section": row["section"], "score": float(row["score"]), "category": row["category"],
                     "year": row["year"], "journal": row["journal"], "doi": row["doi"]})
    return hits


def doc_identity(key: str) -> tuple[str, str]:
    """``(doc_type, doc_id)`` of a wiki key, mirroring the index builder."""
    parts = page_key(key).split("/")
    doc_type = FOLDER_TYPES.get(parts[1], "paper")
    return doc_type, "/".join(parts[2:]).removesuffix(".md")


# ---------------------------------------------------------------------------------------------
# Page reads
# ---------------------------------------------------------------------------------------------

def page_key(key: Any) -> str:
    """A published wiki Markdown key: ``wiki/{folder}/...md`` without drafts, failures or dots.

    Keys longer than ``MAX_KEY_BYTES`` are rejected before any S3 call, so an over-long key a
    model requests can never surface as an S3 error inside a packet build.
    """
    if (not isinstance(key, str) or len(key.encode("utf-8")) > MAX_KEY_BYTES or not SAFE_KEY.fullmatch(key)
            or any(part in {"", ".", ".."} for part in key.split("/"))
            or not key.startswith("wiki/") or not key.endswith(".md") or len(key.split("/")) < 3
            or key.startswith("wiki/drafts/") or "/failed/" in key):
        raise ValueError("Expected a published Markdown key under wiki/")
    return key


SOURCE_TEXT_KEY = re.compile(r"^(papers/[A-Za-z0-9._-]+/clean\.md|sources/[A-Za-z0-9._-]+\.md)$")
MAX_SOURCE_BYTES = 2_000_000  # a GROBID extraction of one paper; the PDF itself is never read here


def source_key(key: Any) -> str:
    """The stored full-text extraction of one paper: ``papers/{stem}/clean.md`` or ``sources/{work_id}.md``."""
    if not isinstance(key, str) or len(key.encode("utf-8")) > MAX_KEY_BYTES or not SOURCE_TEXT_KEY.fullmatch(key):
        raise ValueError("Expected a stored extraction key: papers/{stem}/clean.md or sources/{work_id}.md")
    return key


def assets_key_for_source(key: str) -> str | None:
    """The figure and table text stored beside an extraction, or ``None`` when there can be none.

    Uploaded on 2026-09-22 for 11,260 papers as ``papers/{stem}/assets/assets.md``: the caption,
    the panel labels, every sentence in the paper that mentions the figure, and what was cited
    alongside it. A paper arriving now is cut by the asset worker as it lands.

    Both ingest routes answer here, because both now produce that folder. An uploaded original
    keeps its extraction at ``papers/{stem}/clean.md`` and an OpenAlex-hosted one at
    ``sources/{work_id}.md``, but the crops go to ``papers/{stem}/assets/`` either way, so a
    paper's figures are in one place whichever route brought it in.
    """
    if not isinstance(key, str):
        return None
    if key.startswith("papers/") and key.endswith("/clean.md"):
        return key[: -len("clean.md")] + "assets/assets.md"
    if key.startswith("sources/") and key.endswith(".md"):
        stem = key[len("sources/"): -len(".md")]
        return f"papers/{stem}/assets/assets.md" if stem and "/" not in stem else None
    return None


def read_assets_excerpt(s3, bucket: str, key: str, *, max_chars: int = READ_MAX_CHARS) -> dict[str, Any] | None:
    """The head of a paper's figure and table text; ``None`` when the paper has none stored."""
    assets = assets_key_for_source(key)
    if assets is None:
        return None
    try:
        page = _read_object(s3, bucket, assets, MAX_SOURCE_BYTES)
    except (FileNotFoundError, PageTooLarge):
        return None
    body = page.text
    return {"key": assets, "etag": page.etag, "version_id": page.version_id, "sha256": page.sha256,
            "text": body[:max_chars], "chars": len(body),
            "truncated": len(body) > max_chars}


def source_key_for_note(fields: Mapping[str, str], bucket: str) -> str | None:
    """The extraction key a source note points at, or ``None`` when it records no stored full text.

    OpenAlex-path notes carry ``source_key`` (``sources/{work_id}.md``); user-supplied PDFs carry
    ``pdf_path`` ``s3://{bucket}/papers/{stem}/original.pdf`` whose extraction is ``clean.md``.
    """
    declared = str(fields.get("source_key") or "").strip()
    if declared and SOURCE_TEXT_KEY.fullmatch(declared):
        return declared
    pdf_path = str(fields.get("pdf_path") or "").strip()
    prefix = f"s3://{bucket}/papers/"
    if pdf_path.startswith(prefix) and pdf_path.endswith("/original.pdf"):
        stem = pdf_path[len(prefix):-len("/original.pdf")]
        candidate = f"papers/{stem}/clean.md"
        if SOURCE_TEXT_KEY.fullmatch(candidate):
            return candidate
    return None


class PageTooLarge(ValueError):
    """A page above MAX_PAGE_BYTES is never read whole; it is recorded and skipped."""


def read_page(s3, bucket: str, key: str) -> PageVersion:
    """Read one published page whole, pinning the version read.

    Raises ``ValueError`` for an unpublishable key, ``FileNotFoundError`` for a missing object,
    ``PageTooLarge`` above ``MAX_PAGE_BYTES``, ``UnicodeDecodeError`` for a non-UTF-8 body and
    re-raises any other ``botocore`` ``ClientError`` unchanged.
    """
    return _read_object(s3, bucket, page_key(key), MAX_PAGE_BYTES)


def read_source_text(s3, bucket: str, key: str) -> PageVersion:
    """Read one stored extraction whole (``source_key``), up to ``MAX_SOURCE_BYTES``."""
    return _read_object(s3, bucket, source_key(key), MAX_SOURCE_BYTES)


def _read_object(s3, bucket: str, key: str, limit: int) -> PageVersion:
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in MISSING_CODES:
            raise FileNotFoundError(f"S3 page not found: s3://{bucket}/{key}; the search index may be stale") from exc
        raise
    body = response["Body"]
    try:
        declared = response.get("ContentLength")
        if isinstance(declared, int) and declared > limit:
            raise PageTooLarge(f"page too large: {key} ({declared} bytes)")
        raw = body.read(limit + 1)
        if len(raw) > limit:
            raise PageTooLarge(f"page too large: {key}")
    finally:
        body.close()
    return PageVersion(key=key, text=raw.decode("utf-8"), etag=str(response.get("ETag", "")),
                       version_id=str(response.get("VersionId", "")), sha256=hashlib.sha256(raw).hexdigest())


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    fields: dict[str, str] = {}
    if not text.startswith("---\n"):
        return fields, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return fields, text
    for line in text[4:end].splitlines():
        name, sep, value = line.partition(":")
        if sep and not line.startswith(" "):
            fields[name.strip()] = value.strip().strip('"').strip("'")
    return fields, text[end + 5:]


def _strip_block(text: str, start_marker: str, end_marker: str) -> str:
    """Remove every ``start_marker ... end_marker`` span, shortest match first, in one pass.

    This is the non-greedy ``START.*?END`` substitution of ``wiki_connections`` written with
    ``str.find`` so that a page holding thousands of start markers without an end marker costs
    one scan instead of one scan per marker.
    """
    pieces: list[str] = []
    position = 0
    while True:
        start = text.find(start_marker, position)
        if start < 0:
            break
        end = text.find(end_marker, start + len(start_marker))
        if end < 0:
            break
        pieces.append(text[position:start])
        position = end + len(end_marker)
    pieces.append(text[position:])
    return "".join(pieces)


def strip_managed_blocks(text: str) -> str:
    """Remove the byeori backlink and catalog blocks; they are navigation, not evidence."""
    return _strip_block(_strip_block(text, BACKLINK_START, BACKLINK_END), CATALOG_START, CATALOG_END)


def page_body(text: str) -> str:
    """The text that section offsets index into: frontmatter and managed blocks removed."""
    return strip_managed_blocks(split_frontmatter(text)[1])


# ---------------------------------------------------------------------------------------------
# Outline and classification
# ---------------------------------------------------------------------------------------------

def _non_link_ratio(text: str) -> float:
    remainder = LIST_MARKER.sub("", LINK_MARKUP.sub("", text))
    remainder = re.sub(r"\s+", "", remainder)
    return len(remainder) / max(len(text), 1)


def classify(name: str, text: str) -> str:
    """Evidence kind of a section from its heading, or ``links`` when its text is a link list."""
    if name.strip().casefold() in LINK_SECTION_NAMES:
        return "links"
    if text.strip() and _non_link_ratio(text) < LINK_TEXT_RATIO:
        return "links"
    for kind, pattern in KIND_PATTERNS:
        if pattern.search(name):
            return kind
    return "other"


def _trimmed(body: str, start: int, end: int) -> tuple[int, int]:
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    return start, end


def outline(text: str) -> list[Section]:
    """Sections of a page split on ``## `` headings; ``### Linked pages`` blocks stand alone.

    The text before the first heading, minus the ``# `` title line, is the ``(opening)`` section.
    Empty sections are dropped, so ``order`` counts the sections that carry text.
    """
    body = page_body(text)
    headings: list[tuple[int, int, str]] = []
    for match in HEADING.finditer(body):
        name = match.group(2).strip()
        if len(match.group(1)) == 3 and name.casefold() not in LINK_SECTION_NAMES:
            continue
        headings.append((match.start(), match.end(), name))
    first = headings[0][0] if headings else len(body)
    title = TITLE_LINE.match(body[:first])
    spans: list[tuple[int, int, str | None]] = [(title.end() if title else 0, first, None)]
    for position, (_, content_start, name) in enumerate(headings):
        content_end = headings[position + 1][0] if position + 1 < len(headings) else len(body)
        spans.append((content_start, content_end, name))
    names = {name.casefold() for _, _, name in headings}
    sections: list[Section] = []
    for raw_start, raw_end, name in spans:
        start, end = _trimmed(body, raw_start, raw_end)
        if start >= end:
            continue
        if name is None:
            name = OPENING
            while name.casefold() in names:
                name = "(" + name + ")"
        content = body[start:end]
        sections.append(Section(name=name, order=len(sections), kind=classify(name, content),
                                start=start, end=end, text=content))
    return sections


def _find_section(sections: Sequence[Section], name: str | None) -> Section | None:
    if name is None:
        return None
    wanted = name.strip().casefold()
    if wanted == "":
        return next((s for s in sections if s.name == OPENING), None)
    return next((s for s in sections if s.name.casefold() == wanted), None)


# ---------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------

def _content_words(text: str) -> set[str]:
    return {w for w in (m.group(0).lower() for m in WORD.finditer(text)) if len(w) > 2 and w not in QUESTION_STOPWORDS}


def _overlap(words: set[str], section: Section) -> int:
    return len(words & _content_words(section.text))


def select_sections(question: str, outline: Sequence[Section], best_section_name: str | None, limits: PacketLimits,
                    *, max_sections: int | None = None, all_sections: bool = False) -> list[Section]:
    """Sections to read from one document, in selection priority order.

    The best BM25 section comes first, then one limitations section whenever the page has one,
    then one results section and one methods or interpretation section, until the section
    budget (``limits.max_sections`` or the smaller ``max_sections`` still free in the packet) is
    used. Among several sections of a kind the one sharing most content words with the question
    wins; ties keep document order.
    """
    budget = limits.max_sections if max_sections is None else max_sections
    if budget <= 0 or not outline:
        return []
    words = _content_words(question)
    chosen: list[Section] = []
    best = _find_section(outline, best_section_name)
    if best is None:
        prose = [s for s in outline if s.kind != "links"] or list(outline)
        best = max(prose, key=lambda s: (_overlap(words, s), -s.order))
    chosen.append(best)
    covered = {best.kind}
    for group in ({"limitations"}, {"results"}, {"methods", "interpretation"}):
        if covered & group:
            continue
        candidates = [s for s in outline if s.kind in group and s not in chosen]
        if not candidates:
            continue
        pick = max(candidates, key=lambda s: (_overlap(words, s), -s.order))
        chosen.append(pick)
        covered.add(pick.kind)
    if all_sections:
        rest = [s for s in outline if s.kind != "links" and s not in chosen]
        chosen += sorted(rest, key=lambda s: (-_overlap(words, s), s.order))
    return chosen[:budget]


# ---------------------------------------------------------------------------------------------
# Packet
# ---------------------------------------------------------------------------------------------

def _candidates(queries: Sequence[dict[str, Any]], extra_reads: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Documents to read in rank order: explicit reads first, then hits by best score.

    A requested read with an unusable key is returned in the second list instead of raising:
    the model chooses those keys after reading page text, so a page must not be able to abort
    the packet through them.
    """
    ordered: dict[str, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    for item in extra_reads:
        try:
            key, section = (item, None) if isinstance(item, str) else (item.get("key"), item.get("section"))
            key = page_key(key)
        except (ValueError, AttributeError, TypeError):
            invalid.append({"key": str(item)[:NAME_MAX_CHARS], "reason": "invalid_key"})
            continue
        if section is not None and not isinstance(section, str):
            section = None
        doc_type, doc_id = doc_identity(key)
        ordered.setdefault(key, {"key": key, "doc_type": doc_type, "doc_id": doc_id, "title": doc_id,
                                 "section": section, "score": None, "source": "requested"})
    by_score: dict[str, dict[str, Any]] = {}
    for query in queries:
        for hit in query["hits"]:
            current = by_score.get(hit["key"])
            if current is None or hit["score"] > current["score"]:
                by_score[hit["key"]] = {"key": hit["key"], "doc_type": hit["doc_type"], "doc_id": hit["doc_id"],
                                        "title": hit["title"], "section": hit["section"], "score": hit["score"],
                                        "source": "search"}
    for key, candidate in sorted(by_score.items(), key=lambda kv: (-kv[1]["score"], kv[0])):
        ordered.setdefault(key, candidate)
    return list(ordered.values()), invalid


def build_packet(question: str, *, index, s3, bucket: str, limits: PacketLimits = PACKET_LIMITS,
                 extra_queries: Sequence[str] = (), extra_reads: Sequence[Any] = ()) -> dict[str, Any]:
    """Search, read and select bounded evidence for one question.

    ``index`` is the ``(connection, etag)`` pair from ``open_index``. ``extra_queries`` are
    supplemental searches (each counts against ``limits.searches``); ``extra_reads`` are keys or
    ``{"key", "section"}`` requests read before search hits. Sections are admitted in document
    rank and selection priority order, cut to ``limits.section_chars`` when they are decisive
    (results, limitations) and to the smaller ``limits.context_chars`` otherwise.

    The byte budget is shared, not raced for. Each document may add what is left divided by the
    document slots that could still be filled, so a page read early cannot take the room the
    pages behind it need; whatever a document leaves unused returns to the pages after it. A
    document that wins a slot always contributes its best section when the packet as a whole can
    still hold it, so a question that matches two pages is not cut to a share computed for eight.
    Nothing is written.

    A query that is empty once control characters are removed is recorded as
    ``{"query", "hits": [], "error": "invalid_query"}``; a page that cannot be read or decoded is
    recorded under ``omitted`` (``read_error:<Code>``, ``not_utf8``). Only a non-string or blank
    ``question`` and too many searches raise.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    query_texts = [question, *extra_queries]
    if len(query_texts) > limits.searches:
        raise ValueError(f"{len(query_texts)} searches requested but the packet allows {limits.searches}")
    connection, index_etag = _connection(index)
    queries: list[dict[str, Any]] = []
    for text in query_texts:
        try:
            queries.append({"query": text, "hits": search(connection, text, limits.candidates)})
        except ValueError:
            # Recorded verbatim, so the receipt shows what was asked; an unusable query leaves
            # the packet insufficient rather than aborting the job.
            queries.append({"query": text, "hits": [], "error": "invalid_query"})
    candidates, invalid_reads = _candidates(queries, extra_reads)

    documents: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = list(invalid_reads)
    truncated: list[dict[str, str]] = []
    notes = [f"{len(candidates)} candidate documents from {len(queries)} searches and {len(extra_reads)} requested reads"]
    remaining_bytes = limits.total_bytes
    remaining_sections = limits.max_sections
    for position, candidate in enumerate(candidates):
        key = candidate["key"]
        if len(documents) >= limits.max_documents:
            omitted.append({"key": key, "reason": "max_documents"})
            continue
        if remaining_sections <= 0:
            omitted.append({"key": key, "reason": "max_sections"})
            continue
        if remaining_bytes <= 0:
            omitted.append({"key": key, "reason": "byte_budget"})
            continue
        try:
            page = read_page(s3, bucket, key)
        except PageTooLarge:
            omitted.append({"key": key, "reason": "too_large"})
            continue
        except UnicodeDecodeError:
            omitted.append({"key": key, "reason": "not_utf8"})
            continue
        except ValueError:
            omitted.append({"key": key, "reason": "invalid_key"})  # a draft or non-wiki path in the index
            continue
        except FileNotFoundError:
            omitted.append({"key": key, "reason": "not_found"})
            continue
        except ClientError as exc:
            # AccessDenied, SlowDown, KeyTooLongError and the like: one unreadable page is
            # recorded and skipped; it must not leave the job running past its lease.
            code = str(exc.response.get("Error", {}).get("Code") or "unknown")[:64]
            omitted.append({"key": key, "reason": f"read_error:{code}"})
            continue
        sections = outline(page.text)
        if not sections:
            omitted.append({"key": key, "reason": "empty"})
            continue
        synthesis = candidate["doc_type"] in SYNTHESIS_TYPES
        selected = select_sections(question, sections, candidate["section"], limits,
                                   max_sections=remaining_sections, all_sections=synthesis)
        slots = min(limits.max_documents - len(documents), len(candidates) - position)
        share = remaining_bytes if slots <= 1 else max(remaining_bytes // slots, 1)
        kept: list[dict[str, Any]] = []
        used = 0
        for section in selected:
            text = section.text[:_section_limit(section, limits, synthesis=synthesis)]
            cut = len(text) < len(section.text)
            size = len(text.encode("utf-8"))
            # The first section of a document answers to the packet budget alone; the rest also
            # answer to this document's share, so one page cannot spend what the others need.
            room = remaining_bytes if not kept else min(remaining_bytes, share - used)
            if size > room:
                if kept:
                    omitted.append({"key": key, "section": section.name[:NAME_MAX_CHARS], "reason": "byte_budget"})
                else:
                    omitted.append({"key": key, "reason": "byte_budget"})
                break
            entry = {"name": section.name[:NAME_MAX_CHARS], "order": section.order, "kind": section.kind,
                     "start": section.start, "end": section.start + len(text), "text": text, "truncated": cut}
            if cut:
                entry["full_end"] = section.end
                truncated.append({"key": key, "section": section.name[:NAME_MAX_CHARS]})
            kept.append(entry)
            remaining_bytes -= size
            remaining_sections -= 1
            used += size
        if not kept:
            continue
        fields, _ = split_frontmatter(page.text)
        title = (fields.get("title") or _h1(page.text) or candidate["title"])[:TITLE_MAX_CHARS]
        kept.sort(key=lambda s: s["order"])
        documents.append({"key": key, "doc_type": candidate["doc_type"], "title": title, "etag": page.etag,
                          "version_id": page.version_id, "sha256": page.sha256,
                          "body_sha256": hashlib.sha256(page_body(page.text).encode("utf-8")).hexdigest(),
                          # Where the paper behind this note is stored, so a reader that needs a
                          # number the note does not carry can open it without re-reading the note.
                          "source_key": source_key_for_note(fields, bucket),
                          "sections": kept})
        names = ", ".join(f"{s['name']} ({s['kind']})" for s in kept)
        notes.append(f"{key}: kept {names}")
        if not any(s.kind == "limitations" for s in sections):
            notes.append(f"{key}: no limitations section on the page")

    total_bytes = sum(len(s["text"].encode("utf-8")) for d in documents for s in d["sections"])
    return {"question": question, "queries": queries, "index_etag": index_etag, "documents": documents,
            "omitted": omitted, "truncated": truncated, "total_bytes": total_bytes,
            "evidence_state": evidence_state(documents), "selection_notes": notes,
            "limits": asdict(limits)}


def _section_limit(section: Section, limits: PacketLimits, *, synthesis: bool = False) -> int:
    """Characters kept from one section: the full cut for a decisive kind, the shorter one else.

    Results and limitations carry the numbers and the counter-evidence an answer is judged on.
    Methods, interpretation and the rest are context, and before 2026-09-22 a methods section
    that happened to be the best BM25 match took 6,000 characters of a 30,000-byte packet ahead
    of every results section behind it.
    """
    if synthesis or section.kind in DECISIVE_KINDS:
        return limits.section_chars
    return limits.context_chars


def _h1(text: str) -> str:
    match = re.search(r"^# (.+)$", split_frontmatter(text)[1], re.M)
    return match.group(1).strip() if match else ""


def evidence_state(documents: Sequence[Mapping[str, Any]]) -> str:
    """``insufficient`` without documents, ``links_only`` for link lists and question pages,
    ``truncated_decisive`` when a results or limitations section was cut, else ``sufficient``."""
    if not documents:
        return "insufficient"
    kept = [(d, s) for d in documents for s in d["sections"]]
    if all(d["doc_type"] == "question" or s["kind"] == "links" for d, s in kept):
        return "links_only"
    if any(s["truncated"] and s["kind"] in DECISIVE_KINDS for _, s in kept):
        return "truncated_decisive"
    return "sufficient"


# ---------------------------------------------------------------------------------------------
# Client reads
# ---------------------------------------------------------------------------------------------

def read_excerpt(s3, bucket: str, key: str, *, section: str | None = None, start: int = 0,
                 max_chars: int = 4000) -> dict[str, Any]:
    """The page outline by default (``text`` empty); one section window when ``section`` is given.

    ``max_chars`` above ``READ_MAX_CHARS`` (8,000) is refused, so a client can never pull a whole
    page through this call.
    """
    _check_window(start, max_chars)
    return _excerpt(read_page(s3, bucket, key), section=section, start=start, max_chars=max_chars)


def read_source_excerpt(s3, bucket: str, key: str, *, section: str | None = None, start: int = 0,
                        max_chars: int = 4000) -> dict[str, Any]:
    """``read_excerpt`` for a stored extraction: the outline of the paper's text, or one section window."""
    _check_window(start, max_chars)
    return _excerpt(read_source_text(s3, bucket, key), section=section, start=start, max_chars=max_chars)


def _check_window(start: Any, max_chars: Any) -> None:
    if type(start) is not int or start < 0 or type(max_chars) is not int or not 1 <= max_chars <= READ_MAX_CHARS:
        raise ValueError(f"start must be a non-negative integer and max_chars an integer from 1 to {READ_MAX_CHARS}")


def _excerpt(page: PageVersion, *, section: str | None, start: int, max_chars: int) -> dict[str, Any]:
    fields, _ = split_frontmatter(page.text)
    sections = outline(page.text)
    result = {"key": page.key, "etag": page.etag, "version_id": page.version_id, "sha256": page.sha256,
              "metadata": {name: fields[name][:1000] for name in METADATA_FIELDS if name in fields},
              "sections": [{"name": s.name[:200], "order": s.order, "kind": s.kind, "chars": len(s.text)}
                           for s in sections[:128]]}
    if section is None:
        return {**result, "mode": "outline", "text": ""}
    if not isinstance(section, str) or not section.strip() or len(section) > 200:
        raise ValueError("section must name a heading from the outline")
    found = _find_section(sections, section)
    if found is None:
        raise ValueError(f"Section {section!r} not found; request the page outline first")
    if start > len(found.text):
        raise ValueError("start exceeds the selected section length")
    end = min(start + max_chars, len(found.text))
    return {**result, "mode": "section", "section": found.name, "kind": found.kind, "text": found.text[start:end],
            "start": start, "next_start": end if end < len(found.text) else None,
            "has_more": end < len(found.text), "total_chars": len(found.text)}


def backlinks(index, key: str) -> dict[str, Any]:
    """Pages whose wikilinks point at ``key``, from the index's ``links`` table."""
    connection, index_etag = _connection(index)
    doc_type, doc_id = doc_identity(key)
    links_table = True
    try:
        rows = connection.execute(
            "SELECT DISTINCT l.from_type, l.from_id, d.title, d.path FROM links l "
            "JOIN docs d ON d.doc_type = l.from_type AND d.doc_id = l.from_id "
            "WHERE l.to_type = ? AND l.to_id = ? ORDER BY l.from_type, l.from_id", (doc_type, doc_id)).fetchall()
    except sqlite3.OperationalError:
        rows, links_table = [], False  # an index built before the links table existed
    citing = [{"key": hit_key({"doc_type": r[0], "doc_id": r[1], "path": r[3]}), "doc_type": r[0], "doc_id": r[1],
               "title": r[2]} for r in rows]
    result = {"key": key, "doc_type": doc_type, "doc_id": doc_id, "backlinks": citing, "links_table": links_table}
    if index_etag is not None:
        result["index_etag"] = index_etag
    return result
