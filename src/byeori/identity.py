"""Which paper is this PDF, and does the catalogue's answer hold?

The judgement a paper's identity turns on, with nothing that reaches AWS. The uploads intake asks
it three ways and all three live here: the file name against a record (`compare_identity`), the
title and authors GROBID read out of the PDF against a record (`judge`), and, for a paper with no
DOI that OpenAlex does not hold, the record the PDF itself yields (`record_from_pdf`).

It was two local scripts until 2026-09-23, when the user asked for the whole intake to run in AWS
("AWS에 올려서 자동으로 하세요"). The reading and writing stayed with the callers -- the scripts,
and `ingest_lambda.resolve_identity` -- because they differ in where S3 and OpenAlex are reached;
what they share is every rule below, so a paper is judged the same way whoever asks.
"""
from __future__ import annotations

import difflib
import html
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from typing import Any

from .contact import user_agent
from .journal_policy import apply_upload_policy
from .openalex import eligible_fulltext, normalize_work

SOURCE = "to-s3"
CORPUS_ID = "lab-shared"
# The uploads are one intake among several; they are not the autism-genomics discovery corpus, and
# giving them that id would silently enlarge a corpus the user defined by a different rule.
TEI_HEAD_BYTES = 80_000
MIN_TITLE_OVERLAP = 0.6
# A paper, not the code or the data it came with. Zenodo holds software deposits that carry the
# paper's exact title, and one of them was accepted as the paper itself before this existed.
PAPER_TYPES = ("article", "review", "preprint", "book-chapter", "letter", "editorial")
# OpenAlex fields that are large, derivable, or of no use to a catalog row. `push_candidate` drops
# `raw` for the same reason; `abstract` is dropped as well, since no page may be written from one.
DROPPED_RECORD_FIELDS = ("raw", "referenced_works", "abstract")
# The extraction worker parks a paper here until its journal has been checked. Releasing it is the
# one change this job may make to `ingest_status`, and `promote_to_ready` is the only way it does.
UNCLASSIFIED = "fulltext_ready_unclassified"
# A thesis or a commentary a person looked at and turned away (user, 2026-09-23: "학위논문은 벼리에
# 올리지 맙시다", "이건 논평이라 ingest하면 안되는거네"). No run re-judges it, `--force` included.
NOT_A_PAPER = "not_a_paper"

TITLE_RATIO = 0.9       # difflib ratio between normalised titles
TITLE_JACCARD = 0.8     # or word-set overlap, for a subtitle or a dropped article
TEXT_HEAD_BYTES = 4_000   # the opening of the extraction, where a paper prints its own title
MIN_TITLE_WORDS_IN_TEXT = 4
SEARCH_FILE_NAME_COVERAGE = 0.8
MIN_SHARED_AUTHORS = 3    # a consortium paper's file name names no person; its PDF's authors do
VENUE_TEXT_BYTES = 200_000  # a proceedings footnote can sit well into the text (Stromme 2023: 27,843)
# A record by one of these names is not a paper even when its title goes on to match: a PDF's DOI
# led `dachtler-2015` to APA's "Supplemental Material for Heterozygous Deletion of α-Neurexin I...",
# and two others to records titled only "Abstract" (2026-09-23).
NOT_THE_PAPER = re.compile(r"^\s*(supplement(al|ary)\s+(material|information|data)\b|abstract\s*\.?\s*$"
                           r"|erratum\b|corrigendum\b|correction\s+to\b)", re.I)


# ---------------------------------------------------------------- name comparison

_LETTERS = str.maketrans({"ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d", "ð": "d", "ı": "i",
                          "þ": "th", "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe"})


def fold(text: str | None) -> str:
    """Lowercase ASCII letters and digits only, so `Rühle` and `ruhle` compare equal."""
    decomposed = unicodedata.normalize("NFKD", (text or "").translate(_LETTERS))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", stripped.lower())


def words(text: str | None) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", (text or "").translate(_LETTERS))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return [token for token in re.split(r"[^a-z0-9]+", stripped.lower()) if token]


def split_stem(stem: str) -> tuple[str, str, str]:
    """Split `{author}-{year}-{title-words}` at the year, so a hyphenated surname stays whole."""
    parts = stem.split("-")
    for index, part in enumerate(parts):
        if re.fullmatch(r"(19|20)\d{2}", part):
            return "-".join(parts[:index]), part, "-".join(parts[index + 1:])
    return (parts[0] if parts else ""), "", "-".join(parts[1:])


def name_words(text: str | None) -> list[str]:
    """The words of a personal name, an apostrophe joining rather than splitting: `O'Brien` is `obrien`."""
    return words(re.sub(r"['‘’ʼ`]", "", text or ""))


def spells_surname(surname: list[str], name: list[str]) -> bool:
    """Is `surname` a run of whole words in `name`, spelled out or folded into one?

    `saenz-de-santa-maria` is four words of "Inés Sáenz de Santa María", and `vanderbilt` is
    "Van der Bilt" run together. Initials never join a run, so "L. I. Smith" does not spell `li`.
    """
    folded = "".join(surname)
    for start in range(len(name)):
        if name[start:start + len(surname)] == surname:
            return True
        joined = ""
        for word in name[start:]:
            if len(word) < 2 or len(joined) >= len(folded):
                break
            joined += word
            if joined == folded:
                return True
    return False


def author_agreement(stem_author: str, authors: list[str]) -> bool | None:
    """True, False, or None when no name can be compared at all.

    A surname is compared as whole words, never as a string inside another name. On 2026-09-23 the
    file name `li-2026-...` was found inside "Elizabeth A. Pattie", and a Rett syndrome paper took
    the identity of a hippocampus atlas by other authors.

    Matching runs both ways. A file name may carry a compound surname that OpenAlex shortened
    (`hofstatter-azambuja` against `Juliana H. Azambuja`), so an author's own surname that is a
    whole word of the file name's counts too. OpenAlex may hold a name in a script the file name
    romanised, which folds to nothing and is reported as uncomparable rather than as a disagreement.
    """
    surname = name_words(stem_author)
    # The file name is built from the first author, so it is that name which decides whether a
    # comparison is possible at all; later authors only widen what counts as a match.
    if not surname or not authors or not fold(authors[0]):
        return None
    names = [name_words(author) for author in authors]
    if any(spells_surname(surname, name) for name in names if name):
        return True
    for name in names:
        pieces = [piece for piece in name if len(piece) >= 4]
        if pieces and pieces[-1] in surname:
            return True
    return False


def compare_identity(stem: str, record: dict[str, Any]) -> dict[str, Any]:
    """Does the file name agree with the paper OpenAlex returned? Report the comparison, not a verdict alone."""
    stem_author, stem_year, stem_title = split_stem(stem)
    authors = record.get("authors") or []
    author_ok = author_agreement(stem_author, authors)
    stem_tokens = words(stem_title)
    title_tokens = set(words(record.get("title")))
    overlap = (len(set(stem_tokens) & title_tokens) / len(stem_tokens)) if stem_tokens else 0.0
    openalex_year = record.get("publication_year")
    try:
        year_gap = abs(int(stem_year) - int(openalex_year))
    except (TypeError, ValueError):
        year_gap = None
    # A preprint the lab downloaded in one year and OpenAlex dates to the previous one is still the
    # same paper, so one year of slack is allowed and the exact figures are kept for the reader.
    year_ok = year_gap is not None and year_gap <= 1
    title_ok = overlap >= MIN_TITLE_OVERLAP
    # With no comparable author name, the title has to carry the whole claim: every word of the file
    # name present in the paper's title, and the year exactly right.
    agrees = (bool(author_ok) and year_ok and title_ok) if author_ok is not None \
        else (overlap == 1.0 and year_gap == 0)
    return {
        "stem_author": stem_author, "openalex_authors": authors[:3],
        "stem_year": stem_year, "openalex_year": openalex_year, "year_gap": year_gap,
        "stem_title_words": stem_title, "openalex_title": record.get("title"),
        "title_overlap": round(overlap, 3),
        "author_agrees": author_ok, "year_agrees": year_ok, "title_agrees": title_ok,
        "basis": "author_year_title" if author_ok is not None else "year_title_only",
        "agrees": agrees,
    }


# ---------------------------------------------------------------- what the PDF itself says


def parse_header(text: str) -> dict[str, Any]:
    head = text.split("</teiHeader>")[0]
    title = re.search(r'<title level="a" type="main">(.*?)</title>', head, re.S)
    analytic = re.search(r"<analytic>(.*?)</analytic>", head, re.S)
    people = re.findall(r"<persName[^>]*>(.*?)</persName>", analytic.group(1) if analytic else "", re.S)
    surnames = re.findall(r"<surname>(.*?)</surname>", analytic.group(1) if analytic else "", re.S)
    year = re.search(r'<date type="published" when="(\d{4})', head)
    # The TEI escapes what it quotes: "one cell&apos;s junk" searched as "apos" (2026-09-23).
    return {"title": html.unescape(re.sub(r"\s+", " ", title.group(1)).strip()) if title else None,
            "surnames": [html.unescape(re.sub(r"<[^>]+>", "", s).strip()) for s in surnames],
            "authors": [html.unescape(" ".join(re.sub(r"<[^>]+>", "", part).strip()
                                               for part in re.findall(r"<(?:forename|surname)[^>]*>(.*?)</(?:forename|surname)>", person, re.S)))
                        for person in people],
            "year": int(year.group(1)) if year else None}


def dehyphenate(title: str | None) -> str:
    """Join a word the PDF broke across a line: GROBID read ICLR's "SNAP-SHOTS" and "OPTI-MAL"."""
    return re.sub(r"(\w)-\s*(\w)", r"\1\2", title or "")


def title_agrees(pdf_title: str | None, record_title: str | None) -> tuple[bool, float]:
    b = words(record_title)
    best = (False, 0.0)
    for variant in dict.fromkeys([pdf_title or "", dehyphenate(pdf_title)]):
        a = words(variant)
        if not a or not b:
            continue
        ratio = difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
        jaccard = len(set(a) & set(b)) / len(set(a) | set(b))
        found = (ratio >= TITLE_RATIO or jaccard >= TITLE_JACCARD, round(max(ratio, jaccard), 3))
        best = max(best, found)
    return best


def title_in_text(record_title: str | None, text_head: str | None) -> bool:
    """Does the record's title stand, word for word, in the opening of the paper's own text?

    GROBID sometimes leaves the header title empty while the title is the first heading of the body
    (`## An integrated map of genetic variation from 1,092 human genomes`).
    """
    title, text = words(record_title), words(text_head)
    if len(title) < MIN_TITLE_WORDS_IN_TEXT or not text:
        return False
    return f" {' '.join(title)} " in f" {' '.join(text)} "


def file_name_coverage(stem: str, record: dict[str, Any]) -> float:
    """Share of the file name's title words found in the record's title, a word cut short counting.

    `samocha-2014-framework-interpretati` stops mid-word; its `interpretati` is the start of the
    title's `interpretation`, which the backfill's exact-word overlap counted as missing.
    """
    tokens = words(split_stem(stem)[2])
    title = words(record.get("title"))
    if not tokens or not title:
        return 0.0
    present = [t for t in tokens if t in title or (len(t) >= 4 and any(w.startswith(t) for w in title))]
    return len(present) / len(tokens)


def shared_authors(surnames: list[str], authors: list[str]) -> int:
    """How many of the PDF's author surnames stand as whole words among the record's authors.

    `consortium-2013-genotype-tissue-expression` names the GTEx Consortium, which no record carries
    as an author, but its PDF lists Lonsdale, Salvatore, Sullivan and others that Nature Genetics'
    record lists too.
    """
    names = {word for name in authors for word in words(name)}
    shared = set()
    for surname in surnames:
        parts = tuple(words(surname))
        if parts and len("".join(parts)) >= 3 and set(parts) <= names:
            shared.add(parts)
    return len(shared)


def file_name_agrees(stem: str, record: dict[str, Any], threshold: float = 0.6, shared_authors: int = 0) -> bool:
    """The backfill's file-name rule (0.6 of the words, one year of slack, the author) with cut words.

    0.6 is the backfill's bar for a record the PDF's own DOI led to. A record found by searching has
    to clear more: on 2026-09-23 `andersson-2009-studying-phenotypic-evolution-in-domestic` met 0.6
    with the same author's coat-colour paper of that year, while its own title covers every word.
    """
    stem_author, stem_year, _ = split_stem(stem)
    try:
        year_ok = abs(int(stem_year) - int(record.get("publication_year"))) <= 1
    except (TypeError, ValueError):
        year_ok = False
    author_ok = (author_agreement(stem_author, record.get("authors") or []) is True
                 or shared_authors >= MIN_SHARED_AUTHORS)
    return file_name_coverage(stem, record) >= threshold and year_ok and author_ok


def first_author_agrees(surnames: list[str], authors: list[str]) -> bool | None:
    if not surnames or not authors:
        return None
    first = set(words(surnames[0]))
    return any(first & set(words(name)) for name in authors)


# What a publisher's band is made of besides the journal's own name. A band carries the section it
# is in and the state of the article, and nothing about the paper; a real title carries words of its
# own, which is what separates the two.
BANNER_TERMS = frozenset("""
research article articles review reviews original open access communication communications report
reports brief letter letters perspective correspondence editorial comment commentary in press online
early advance first accepted published article's the a an and of for on
""".split())


def looks_like_journal_banner(title: str | None, journal: str | None) -> bool:
    """Is what GROBID called the title the journal's running header rather than the paper's title?

    A publisher's first page often prints a band - "RESEARCH | Article in Press Genome Medicine" -
    above the title, and GROBID reads the band. Then the PDF has, as far as the rules can see, no
    title of its own, and the file name and the authors are what is left to decide by. A band is the
    journal's name plus publishing boilerplate and nothing else, so a title that merely mentions its
    journal ("Nature versus nurture in the developing cortex") is not one (santoro-2026 and
    beer-wells-2026, 2026-09-23/24).
    """
    if not title or not journal:
        return False
    band, name = words(title), set(words(journal))
    if not name or not band or not name.issubset(set(band)):
        return False
    return all(word in name or word in BANNER_TERMS for word in band)


def judge(header: dict[str, Any], record: dict[str, Any], *, text_head: str | None = None,
          stem: str | None = None, doi_from_pdf: bool = False) -> dict[str, Any]:
    """Is `record` the paper in this PDF?

    The title must agree in one of three ways: with the header GROBID read, word for word in the
    opening of the extracted text, or, only when the PDF printed no title, with the file name by
    the backfill's rule. When the DOI came off the PDF itself, the PDF's title agreeing is enough,
    because consortium papers name no first author a record can carry. Otherwise the PDF's first
    author must not be missing from the record; a header with no author leaves the title to decide.
    """
    banner = looks_like_journal_banner(header.get("title"), record.get("source"))
    no_title = bool(header.get("from_file_name")) or banner
    header_agrees, score = ((False, 0.0) if no_title
                            else title_agrees(header.get("title"), record.get("title")))
    in_text = title_in_text(record.get("title"), text_head)
    shared = shared_authors(header.get("surnames") or [], record.get("authors") or [])
    by_name = bool(stem) and file_name_agrees(stem, record, 0.6 if doi_from_pdf else SEARCH_FILE_NAME_COVERAGE,
                                              shared_authors=shared)
    # The file name speaks only when the PDF printed no title of its own. A title that is there and
    # disagrees outweighs it: `arora-2025-healthbench` met the file-name rule with the same team's
    # "HealthBench Professional" of 2026, while its PDF says "HealthBench: Evaluating Large Language
    # Models Towards Improved Human Health" (2026-09-23).
    title_ok = header_agrees or in_text or (by_name and no_title)
    author = first_author_agrees(header.get("surnames") or [], record.get("authors") or [])
    not_the_paper = bool(NOT_THE_PAPER.search(record.get("title") or ""))
    agrees = title_ok and (doi_from_pdf or by_name or author is not False) and not not_the_paper
    return {"pdf_title": header.get("title"), "pdf_first_author": (header.get("surnames") or [None])[0],
            "record_title": record.get("title"), "title_score": score, "title_agrees": header_agrees,
            "title_in_text": in_text, "file_name_agrees": by_name, "first_author_agrees": author,
            "journal_banner": banner,
            "shared_authors": shared, "not_the_paper": not_the_paper, "agrees": agrees}


def _query(title: str, limit: int = 14) -> str:
    """The title as written, punctuation dropped, letters kept as they are.

    Folding to ASCII lost the paper: "schrodinger bridges" brought back AlphaFold, while "Schrödinger
    Bridges" found Hong et al. at the top (2026-09-23).
    """
    return " ".join(re.findall(r"\w+", title or "")[:limit])


def queries(header: dict[str, Any], stem_title: str) -> list[str]:
    """The PDF's title, then that title with its line-break hyphens joined, then the file name's words.

    GROBID misreads a title now and then ("Denning" for "Defining" in Jacobson and Truax 1991), and
    the file name a person typed then finds what the header cannot.
    """
    titles = [header.get("title") or "", dehyphenate(header.get("title"))] if not header.get("from_file_name") else []
    return [q for q in dict.fromkeys([*titles, stem_title.replace("-", " ")]) if _query(q)]


def venue_in_text(venue: str, text: str | None) -> bool:
    """Does the paper's own text name the venue, word for word?"""
    wanted, haystack = " ".join(words(venue)), " ".join(words(text))
    return bool(wanted) and wanted in haystack


# ---------------------------------------------------------------- DOI recovery


def repair_doi(doi: str) -> str:
    """Undo the two ways GROBID mangles a DOI it lifts off a publisher's PDF banner."""
    cleaned = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip(), flags=re.I)
    # `10.1158/0008-5472.CAN-26-1023/3814737/can-26-4971.pdfbygueston`: the download path and the
    # "by guest on ..." footer are appended to the real DOI.
    cleaned = re.sub(r"/\d{5,}/[^/]*$", "", cleaned)
    cleaned = re.sub(r"\.pdf(by|$).*$", "", cleaned, flags=re.I)
    return cleaned.rstrip(".,;)")


def looks_truncated(doi: str) -> bool:
    """`10.1186/s1305` is the first five characters of a Genome Biology DOI, not a DOI."""
    suffix = doi.split("/", 1)[1] if "/" in doi else ""
    return len(suffix) < 6


def crossref_by_doi(doi: str) -> dict[str, Any] | None:
    """The DOI registration agency's own record, used only where OpenAlex cannot answer.

    Two cases need it. OpenAlex sometimes holds a paper with repository locations alone, so its
    `primary_location` is `PubMed` or a university archive and the journal is nowhere in the record.
    And a paper published days ago may not be indexed at all, while its DOI is registered the day it
    appears. Crossref is where the journal name is registered, so it settles both.
    """
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent("byeori/0.1")})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            message = json.load(response).get("message") or {}
    except Exception:
        return None
    time.sleep(0.1)
    container = (message.get("container-title") or [None])[0]
    issued = ((message.get("issued") or {}).get("date-parts") or [[None]])[0]
    authors = [" ".join(part for part in (person.get("given"), person.get("family")) if part)
               for person in (message.get("author") or [])]
    return {"journal": container, "title": (message.get("title") or [None])[0],
            "authors": [name for name in authors if name],
            "publication_year": issued[0] if issued else None, "type": message.get("type")}
# Where a copy sits rather than where the paper was published: PubMed, PubMed Central, Zenodo,
# university archives, and aggregators such as DOAJ. Anything else, including a source whose type
# OpenAlex left out, is taken as the publication.
NON_JOURNAL_SOURCE_TYPES = ("repository", "metadata")


def journal_name(work: dict[str, Any]) -> tuple[str | None, str]:
    """The journal this paper appeared in, and where that name came from.

    OpenAlex ranks a repository copy first often enough that reading `primary_location` alone turns
    a Nature Genetics paper into `ArTS Archivio della ricerca di Trieste`. A repository is where a
    copy sits, never the journal, so a journal-typed location is preferred and Crossref answers when
    OpenAlex records no journal at all.
    """
    for location in ([work.get("primary_location")] + list(work.get("locations") or [])):
        source = (location or {}).get("source") or {}
        if source.get("display_name") and source.get("type") not in NON_JOURNAL_SOURCE_TYPES:
            return source["display_name"], "openalex_location"
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", str(work.get("doi") or ""), flags=re.I)
    if doi:
        registered = crossref_by_doi(doi)
        if registered and registered.get("journal"):
            return registered["journal"], "crossref"
    primary = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    return (primary, "openalex_repository_only") if primary else (None, "not_recorded")


# ---------------------------------------------------------------- the catalogue record


def _corpus_block(record: dict[str, Any]) -> dict[str, Any]:
    topics = [topic for topic in (record.get("topics") or []) if topic][:3]
    record["corpus"] = {
        "id": CORPUS_ID, "scope": "lab_shared_pdf", "intake": SOURCE,
        "tags": topics or ["Unclassified"], "tag_basis": "openalex_topics",
        "screening": "pdf_uploaded", "fulltext_eligible": eligible_fulltext(record),
        "fulltext_stored": True,
    }
    record["catalog_search_text"] = " ".join([
        record.get("title") or "", record.get("doi") or "", record.get("source") or "",
        " ".join(record.get("authors") or []), " ".join(record.get("topics") or []),
    ]).lower()
    # A person chose every PDF this reads, so only a refused house or title keeps it out; the
    # discovery list is the OpenAlex route's gate, not this one (user, 2026-09-23).
    return apply_upload_policy(record)


def record_from_crossref(doi: str, registered: dict[str, Any]) -> dict[str, Any]:
    """A catalog record for a paper OpenAlex has not indexed yet, built from its DOI registration.

    It carries no OpenAlex id, no topics and no citation count, because Crossref holds none of
    those. It does carry the journal, which is what the allowlist needs, and `journal_source_basis`
    says where that came from so nobody mistakes it for an OpenAlex answer.
    """
    record = {
        "work_id": None, "doi": doi, "title": registered.get("title"),
        "publication_year": registered.get("publication_year"), "publication_date": None,
        "type": "article" if registered.get("type") == "journal-article" else registered.get("type"),
        "authors": registered.get("authors") or [], "source": registered.get("journal"),
        "cited_by_count": 0, "is_open_access": False, "oa_license": None,
        "landing_page_url": f"https://doi.org/{doi}", "pdf_url": None,
        "openalex_pdf_url": None, "grobid_xml_url": None, "topics": [],
        "journal_source_basis": "crossref",
    }
    return _corpus_block(record)


def build_record(work: dict[str, Any]) -> dict[str, Any]:
    record = {key: value for key, value in normalize_work(work).items()
              if key not in DROPPED_RECORD_FIELDS}
    # `normalize_work` reports `primary_location`, which may be a repository copy; the verdict has
    # to be taken on the journal.
    source, basis = journal_name(work)
    record["source"], record["journal_source_basis"] = source, basis
    return _corpus_block(record)


def record_from_pdf(header: dict[str, Any], *, venue: str, year: int | None) -> dict[str, Any]:
    """A catalog record for a paper that has no DOI and no OpenAlex work, built from its own PDF.

    Like `record_from_crossref` it carries no OpenAlex id, topics or citation count, and
    `journal_source_basis` says where it came from.
    """
    record = {
        "work_id": None, "doi": None, "title": header.get("title"), "publication_year": year,
        "publication_date": None, "type": "article",
        "authors": header.get("authors") or header.get("surnames") or [], "source": venue,
        "cited_by_count": 0, "is_open_access": False, "oa_license": None, "landing_page_url": None,
        "pdf_url": None, "openalex_pdf_url": None, "grobid_xml_url": None, "topics": [],
        "journal_source_basis": "pdf",
    }
    return _corpus_block(record)
