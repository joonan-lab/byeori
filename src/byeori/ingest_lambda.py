import gzip
import hashlib
import random
import time
import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
import urllib.error
import urllib.parse
import urllib.request
import shutil
import uuid
from pathlib import Path
from xml.etree import ElementTree

import sqlite3
from concurrent.futures import ThreadPoolExecutor
import boto3
import botocore.config
import botocore.exceptions
from boto3.dynamodb.conditions import Attr

from byeori import journal_policy

BUCKET_NAME = os.environ["BUCKET_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
OPENALEX_API_KEY_PARAMETER = os.environ["OPENALEX_API_KEY_PARAMETER"]
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
USER_AGENT = "byeori-workshop/0.2"
WORK_ID_PATTERN = re.compile(r"W[0-9]+")
SEARCH_SELECT_FIELDS = (
    "id,doi,display_name,publication_year,publication_date,type,authorships,"
    "primary_location,cited_by_count,open_access,best_oa_location,content_urls,topics"
)
# What a topic search may be held to. "list" keeps OpenAlex inside the lab's journals by filtering
# on their source ids; "wide" lets it answer from anywhere and leaves the refusing and the warning
# to journal_policy.search_verdict (user, 2026-09-22). 65 ids fit OpenAlex's cap of 100 values per
# filter; the same journals' 102 ISSNs do not, which is why the id is what is sent.
SEARCH_SCOPES = ("list", "wide")
MAX_SEARCH_AUTHORS = 25
MAX_SEARCH_TOPICS = 10
DRAFT_MODEL_ID = os.environ.get("DRAFT_MODEL_ID", "global.anthropic.claude-opus-5")
# Evidence notes have their own model because the user moved them to Opus 5.5 on 2026-09-23 while
# answers and synthesis stay on Opus 5. Opus 5.5 runs a biology classifier Opus 5 does not, and it
# declined 8 of 14 new papers and 9 of 40 noted ones that day with stopReason content_filtered, a
# second try rescuing 1 in 8 and 1 in 9. A declined paper is therefore written by the fallback at
# once instead of being retried (state/model-trial-opus55-refusal-sample-20260923.json).
NOTE_MODEL_ID = os.environ.get("NOTE_MODEL_ID") or DRAFT_MODEL_ID
NOTE_FALLBACK_MODEL_ID = os.environ.get("NOTE_FALLBACK_MODEL_ID", "")
DRAFT_MAX_INPUT_CHARS = 400_000
DRAFT_MAX_OUTPUT_TOKENS = 8_000
# Adaptive thinking and the answer share one maxTokens budget, so it has to cover the reasoning
# before any output is written. A flat 24,000 was not enough at xhigh: four of the 464 questions on
# 2026-09-20 returned an empty page with stopReason max_tokens, one after only three output tokens.
# Opus 5 accepts up to 128,000 output tokens (its 1M figure is the context window, a different
# limit), so the headroom scales with the effort that decides how much thinking happens.
THINKING_HEADROOM = {"low": 8_000, "medium": 16_000, "high": 32_000, "xhigh": 56_000, "max": 96_000}
DRAFT_SECTIONS = ("## Citation", "## Methods", "## Results", "## Limitations", "## Evidence boundary", "## Related pages")

# A prompt that ends in the paper's own text reads as a document to continue. On 2026-09-22, 80 of
# 521 evidence notes failed the section check, and re-running one showed what it produces instead:
# stopReason end_turn, 7,281 output tokens, 36 level-2 headings, none of them the ones asked for.
# It had written the Nature reporting checklist, which was nowhere in the extraction. Other failures
# opened on an author list, a copyright page or mid-paper prose. So the instruction is repeated
# after the text, where it is the last thing the model reads.
WRITE_NOW = (
    "\n\n---\n\nThe extracted text above is material to read, not a document to continue. Write the page "
    "for it now, in Markdown, with exactly the level-2 sections your instructions list, in that order, "
    "beginning with {first}. Do not reproduce the paper, its front matter or any reporting checklist."
)

# A Nature-family PDF can carry the reporting summary, a standards form with no findings in it. It
# reached about one extraction in forty on 2026-09-22 and took 30% of the one that had it. Cutting
# it at read time rather than at extraction reaches the papers already stored.
REPORTING_SUMMARY_MARKERS = (
    "nature portfolio reporting summary",
    "nature research reporting summary",
    "reporting summary",
    "field-specific reporting",
    "life sciences study design",
    "reporting for specific materials, systems and methods",
)
REPORTING_SUMMARY_MIN_OFFSET = 2_000   # never cut a paper down to its opening pages


def _strip_reporting_summary(text):
    """The extraction without its trailing reporting checklist, when one is recognisable."""
    lowered = text.lower()
    cuts = [lowered.find(marker) for marker in REPORTING_SUMMARY_MARKERS]
    cuts = [cut for cut in cuts if cut >= REPORTING_SUMMARY_MIN_OFFSET]
    if not cuts:
        return text
    return text[:min(cuts)].rstrip()
# Self-reference is "As an AI, I ...", not the words themselves: a bare "as an ai" matched "was an
# AI or a human" and a cited title "GPT-4 as an AI chatbot", failing two notes on AI papers (2026-09-23).
DRAFT_FORBIDDEN = re.compile(r"\[(?:TODO|TBD|placeholder|insert|citation needed)[^\]]*\]|lorem ipsum|"
                             r"\bas an ai(?: language model| assistant)?,? i\b", re.I)
DRAFT_SYSTEM = (
    "You write evidence notes for a scientific literature wiki. You receive the metadata of one "
    "paper and its full text as extracted from the PDF by GROBID. Write one Markdown note in English "
    "with exactly these level-2 sections in this order: Citation, Methods, Results, Limitations, "
    "Evidence boundary, Related pages.\n\n"
    "Rules:\n"
    "- Report only what the extracted text states. Do not add knowledge from outside the text. "
    "Do not speculate about mechanisms or clinical implications the authors did not state.\n"
    "- Copy the specific values the paper reports, exactly as written: sample and cohort sizes, "
    "effect sizes and confidence intervals, p-values and error rates, benchmark scores, doses and "
    "concentrations, throughput and runtime, and the identifiers the field uses (gene or protein "
    "symbols, variants, cell lines, strains, datasets, model or software names with versions). "
    "When a number appears only in a figure or table that the extraction did not capture, say so "
    "instead of estimating it.\n"
    "- Methods: what was studied (cohort, samples, cell lines, datasets, or corpora), the data types, the "
    "key analyses or model architecture, and the statistical or evaluation approach, as stated.\n"
    "- Results: the main findings as claims, each tied to the section or figure the text cites.\n"
    "- Limitations: those the authors state, marked as theirs, then at most three that follow "
    "directly from the described design, marked as reviewer notes.\n"
    "- Evidence boundary: what in this note was verified against the extracted text, what depends "
    "on figures or tables that must be checked in the PDF, and any place the extraction looked "
    "corrupted or truncated.\n"
    "- Related pages: two to five plain-text bullet points naming the topics this paper belongs to "
    "(for example: de novo variants in autism). No links.\n"
    "- Citation: authors (first three, then et al.), year, title, journal, DOI, from the metadata.\n"
    "- Do not write frontmatter, a level-1 title, a preamble, placeholders, or notes to the reader. "
    "Start directly with '## Citation'."
)
TOPIC_SECTIONS = ("## Scope", "## Synthesis", "## Open questions", "## Related papers")
MAX_TOPIC_NOTES = 20
# v2 is contentless FTS5: the index carries no page text, so it is about a third of the size and a
# cold Lambda spends that much less time fetching it. Section text is read from S3 per hit instead.
WIKI_INDEX_KEY = "index/wiki-index-v2.sqlite3"
# Answered student questions live here as Markdown and stay out of the index (user, 2026-09-22).
# The value is the one in byeori.lab_store.ANSWER_PAGE_PREFIX; it is repeated rather than
# imported because the campaign Lambda never imports a lab module, and a test compares the two.
LAB_QUESTION_PREFIX = "wiki/lab-questions/"
WIKI_INDEX_LOCAL = Path("/tmp/wiki-index-v2.sqlite3")
TOPIC_SYSTEM = (
    "You write overview pages (encyclopedic topic pages) for a scientific literature wiki. You receive a topic title and several "
    "reviewed evidence notes, each labelled with its OpenAlex work ID and citation. Write one Markdown "
    "page in English with exactly these level-2 sections in this order: Scope, Synthesis, Open questions, "
    "Related papers.\n\n"
    "Rules:\n"
    "- Use only what the evidence notes state. Do not add findings, papers, or background from outside "
    "the notes. If the notes disagree, say so and attribute each position.\n"
    "- Scope: what this topic covers and which papers the page rests on, in one paragraph.\n"
    "- Synthesis: compare the papers' designs and results. Attribute every claim to a paper by citing "
    "its first author and year in parentheses, exactly as in the note's Citation. Keep numbers as written.\n"
    "- Open questions: only questions that follow from gaps or disagreements the notes themselves show.\n"
    "- Related papers: one bullet per note, formatted as 'W<id>: first author (year). title.'\n"
    "- Do not write frontmatter, a level-1 title, a preamble, placeholders, or notes to the reader. "
    "Start directly with '## Scope'."
)
SOURCE_NOTE_SECTIONS = ("## One-line Summary", "## 2. Key Contributions", "## 3. Methodology and Architecture",
                        "## 4. Key Results and Benchmarks", "## 5. Limitations and Future Work", "## 6. Related Work", "## 7. Glossary")
INGEST_HARNESS = os.environ.get("INGEST_HARNESS", "aws-bedrock")
INGEST_REASONING = os.environ.get("INGEST_REASONING", "default")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
INGEST_AGENT = "byeori-ingest"
# A model trial writes a note for comparison and nothing else: not the wiki, not the catalogue, not
# the index. The name becomes one folder under runs/model-trials/, so it may not carry a path.
TRIAL_RUN_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
INGEST_AGENT_VERSION = "v1"
SOURCE_NOTE_SYSTEM = (
    "You write the source page of a scientific literature wiki from the full text of one paper, "
    "extracted from the PDF by GROBID, plus its bibliographic metadata. Write Markdown in English with "
    "exactly these level-2 sections in this order: 'One-line Summary', '2. Key Contributions', "
    "'3. Methodology and Architecture', '4. Key Results and Benchmarks', '5. Limitations and Future Work', "
    "'6. Related Work', '7. Glossary'. Section 1 (Document Information) is written by the system; do not write it.\n\n"
    "Rules:\n"
    "- Report only what the text states; no outside knowledge, no speculation.\n"
    "- One-line Summary: one sentence with the main finding and its scale (cohort size, effect size).\n"
    "- Key Contributions: bullets, each a specific contribution as the authors frame it.\n"
    "- Methodology and Architecture: samples or cohort, data types, key analyses, statistics, as stated.\n"
    "- Key Results and Benchmarks: bullets carrying the values above exactly as written, whichever kinds "
    "this paper reports; name the table or figure the text cites. If a number lives only in a figure the "
    "extraction did not capture, say so.\n"
    "- Limitations and Future Work: the authors' stated limitations first, marked as theirs; then at most three that "
    "follow from the design, marked as reviewer notes.\n"
    "- Related Work: how the authors position the paper against prior work they cite, with author and year.\n"
    "- Glossary: five to twelve terms central to this paper, one line each - the genes, proteins, cell "
    "types, assays, architectures, metrics or datasets a reader from a neighbouring field would not know.\n"
    "- No frontmatter, no level-1 title, no preamble, no placeholders. Start directly with '## One-line Summary'."
)
s3 = boto3.client("s3")
ssm = boto3.client("ssm")
# Opus 5 takes minutes per paper: botocore's 60-second default read timeout would fail the
# call and retry it, running (and billing) the model again. Wait up to 15 minutes, never retry.
bedrock = boto3.client("bedrock-runtime", config=botocore.config.Config(read_timeout=880, connect_timeout=10, retries={"max_attempts": 1}))
table = boto3.resource("dynamodb").Table(TABLE_NAME)

def _with_api_key(url, api_key):
    if not api_key:
        return url
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key == "api_key" for key, _ in query):
        query.append(("api_key", api_key))
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )

def _request(url, api_key=None):
    return urllib.request.Request(
        _with_api_key(url, api_key),
        headers={"User-Agent": USER_AGENT},
    )

def _fetch_json(url, api_key):
    try:
        with urllib.request.urlopen(_request(url, api_key), timeout=30) as response:
            return json.load(response)
    except (TimeoutError, urllib.error.URLError):
        raise RuntimeError("OpenAlex metadata request failed") from None

def _require_openalex_content_url(url):
    parsed = urllib.parse.urlsplit(url or "")
    if parsed.scheme != "https" or parsed.hostname != "content.openalex.org":
        raise ValueError("full text must use an OpenAlex content URL")

def _finalize_download(raw_path, target, content_encoding=""):
    """Store the decoded bytes at ``target``; return (sha256, size, content_type_hint).

    OpenAlex serves some pre-compressed objects with ``Content-Encoding: gzip`` and
    urllib does not decode them, so gzip is detected by header or magic bytes.
    """
    with raw_path.open("rb") as handle:
        head = handle.read(2)
    if head == b"\x1f\x8b" or "gzip" in (content_encoding or "").lower():
        with gzip.open(raw_path, "rb") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
    else:
        shutil.move(str(raw_path), str(target))
    digest = hashlib.sha256()
    total = 0
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total

def _download(url, target, api_key):
    _require_openalex_content_url(url)
    raw_path = target.with_suffix(target.suffix + ".raw")
    total = 0
    try:
        with urllib.request.urlopen(_request(url, api_key), timeout=60) as response:
            content_encoding = response.headers.get("Content-Encoding", "")
            content_type = response.headers.get("Content-Type", "")
            with raw_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError("OpenAlex content exceeds 100 MB workshop limit")
                    output.write(chunk)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read(500).decode("utf-8", "replace")
        except Exception:
            detail = ""
        raise RuntimeError(f"OpenAlex full-text download failed with HTTP {exc.code}: {detail[:300]}") from None
    except (TimeoutError, urllib.error.URLError):
        raise RuntimeError("OpenAlex full-text download failed (no HTTP response)") from None
    digest, size = _finalize_download(raw_path, target, content_encoding)
    return digest, size, content_type

def _node_text(node):
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())

def _grobid_markdown(xml_path, work):
    root = ElementTree.parse(xml_path).getroot()
    title = _node_text(root.find(".//{*}titleStmt/{*}title")) or work.get("display_name")
    lines = [f"# {title}", ""]
    abstract = root.find(".//{*}profileDesc/{*}abstract")
    abstract_text = _node_text(abstract)
    if abstract_text:
        lines.extend(["## Abstract", "", abstract_text, ""])
    body = root.find(".//{*}text/{*}body")
    if body is None:
        raise ValueError("OpenAlex GROBID XML has no body")
    paragraph_count = 0
    for node in body.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        text = _node_text(node)
        if not text:
            continue
        if tag == "head":
            lines.extend([f"## {text}", ""])
        elif tag == "p":
            lines.extend([text, ""])
            paragraph_count += 1
    if paragraph_count == 0:
        raise ValueError("OpenAlex GROBID XML has no body paragraphs")
    lines.extend(
        [
            "## Extraction note",
            "",
            "This text was parsed from OpenAlex GROBID XML and must be checked against the stored PDF before scientific claims are finalized.",
            "",
        ]
    )
    return "\n".join(lines)

def _api_key():
    return ssm.get_parameter(
        Name=OPENALEX_API_KEY_PARAMETER,
        WithDecryption=True,
    )["Parameter"]["Value"]

def _compact_search_work(work):
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    best_oa = work.get("best_oa_location") or {}
    authorships = []
    for authorship in (work.get("authorships") or [])[:MAX_SEARCH_AUTHORS]:
        name = (authorship.get("author") or {}).get("display_name")
        if name:
            authorships.append({"author": {"display_name": name}})
    topics = [
        {"display_name": topic.get("display_name")}
        for topic in (work.get("topics") or [])[:MAX_SEARCH_TOPICS]
        if topic.get("display_name")
    ]
    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "display_name": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "authorships": authorships,
        "primary_location": {
            "landing_page_url": primary.get("landing_page_url"),
            # The id, the ISSNs and the publishing house decide the journal policy. A display name
            # drifts ("Science (New York, N.Y.)") and a house-wide refusal needs the house.
            "source": {
                "id": source.get("id"),
                "display_name": source.get("display_name"),
                "issn": [issn for issn in (source.get("issn") or []) if issn],
                "host_organization_name": source.get("host_organization_name"),
            },
        },
        "cited_by_count": work.get("cited_by_count", 0),
        "open_access": {
            "is_oa": bool((work.get("open_access") or {}).get("is_oa"))
        },
        "best_oa_location": {
            "license": best_oa.get("license"),
            "landing_page_url": best_oa.get("landing_page_url"),
            "pdf_url": best_oa.get("pdf_url"),
        },
        "content_urls": work.get("content_urls") or {},
        "topics": topics,
    }

def _search(event, api_key):
    """Search OpenAlex, holding the result to the lab's journal policy.

    ``journal_scope`` is ``list`` (the default) or ``wide``. ``list`` asks OpenAlex for the lab's
    65 journals only. ``wide`` asks for everything and then applies the policy here: a refused
    publisher or title is dropped and counted, and anything outside the 65 comes back carrying the
    warning the caller must show. A paper being searchable never makes it ingestable; that stays
    ``corpus.journal_verdict``.
    """
    query = str(event.get("query", "")).strip()
    limit = int(event.get("limit", 10))
    if not query or not 1 <= limit <= 100:
        raise ValueError("search needs a query and a limit from 1 to 100")
    scope = str(event.get("journal_scope") or "list")
    if scope not in SEARCH_SCOPES:
        raise ValueError(f"journal_scope must be one of {', '.join(SEARCH_SCOPES)}")
    filters = []
    if scope == "list":
        filters.append("primary_location.source.id:" + "|".join(journal_policy.SOURCE_IDS))
    if event.get("from_year") is not None:
        filters.append(f"from_publication_date:{int(event['from_year'])}-01-01")
    if event.get("to_year") is not None:
        filters.append(f"to_publication_date:{int(event['to_year'])}-12-31")
    if event.get("oa_only"):
        filters.append("is_oa:true")
    if event.get("fulltext_only"):
        filters.append("has_content.grobid_xml:true")
    params = {
        "search": query,
        # A wide search is thinned by the refusals below, so ask for more than the caller wants.
        "per_page": min(100, limit * 2) if scope == "wide" else limit,
        "select": SEARCH_SELECT_FIELDS,
    }
    if filters:
        params["filter"] = ",".join(filters)
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    payload = _fetch_json(url, api_key)

    results, refused, outside = [], [], 0
    for work in payload.get("results") or []:
        compact = _compact_search_work(work)
        source = (compact.get("primary_location") or {}).get("source") or {}
        verdict = journal_policy.search_verdict(
            source.get("display_name"), source.get("issn"),
            source.get("host_organization_name"), source.get("id"))
        if verdict["verdict"] == journal_policy.FORBIDDEN:
            refused.append(source.get("display_name") or source.get("host_organization_name") or "")
            continue
        if verdict["verdict"] == journal_policy.OUTSIDE_LIST:
            outside += 1
        compact["journal_scope_verdict"] = verdict["verdict"]
        compact["journal_warning"] = verdict["warning"]
        results.append(compact)
        if len(results) >= limit:
            break
    return {
        "results": results,
        "journal_scope": scope,
        "outside_list_count": outside,
        "refused_count": len(refused),
        "refused_journals": sorted({name for name in refused if name}),
    }

def _get(event, api_key):
    doi = str(event.get("doi") or "").strip().lower()
    if doi:
        doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)
        if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
            raise ValueError("doi must look like 10.xxxx/yyyy")
        return {"work": _fetch_json("https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="/"), api_key)}
    work_id = str(event.get("work_id", "")).upper()
    if not WORK_ID_PATTERN.fullmatch(work_id):
        raise ValueError("work_id must be an OpenAlex ID such as W2741809807")
    return {
        "work": _fetch_json(f"https://api.openalex.org/works/{work_id}", api_key)
    }

def _resolve_identity(event, api_key):
    """Which paper the upload just extracted is, and whether the wiki may read it.

    The rules are `identity`; the reading and writing are `identity_resolve`; what is here is the
    two ways this Lambda reaches OpenAlex, handed in as functions. The event is either a stem asked
    for by hand or the object-created event for `papers/{stem}/clean.md`, which is what makes the
    intake run by itself (user, 2026-09-23).
    """
    from . import identity_resolve
    stem = str(event.get("stem") or "").strip()
    if not stem:
        key = (((event.get("detail") or {}).get("object") or {}).get("key")) or event.get("key") or ""
        stem = identity_resolve.stem_of(str(key)) or ""
    if not stem or "/" in stem:
        raise ValueError("resolve_identity needs a stem, or the key of a papers/{stem}/clean.md object")

    def fetch_work(doi):
        doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", str(doi or "").strip().lower())
        if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
            return None
        try:
            return _fetch_json("https://api.openalex.org/works/https://doi.org/"
                               + urllib.parse.quote(doi, safe="/"), api_key)
        except Exception:      # noqa: BLE001 - a DOI OpenAlex does not hold is an answer, not a failure
            return None

    def search(query, year):
        if not query:
            return []
        request = {"query": query, "limit": 10, "journal_scope": "wide"}
        if year:
            request |= {"from_year": year - 1, "to_year": year + 1}
        return _search(request, api_key).get("results") or []

    return identity_resolve.resolve(stem, s3=s3, table=table, bucket=BUCKET_NAME,
                                    fetch_work=fetch_work, search=search,
                                    apply=bool(event.get("apply", True)))


def _ingest(event, api_key):
    started = datetime.now(timezone.utc)
    work_id = str(event.get("work_id", "")).upper()
    if not WORK_ID_PATTERN.fullmatch(work_id):
        raise ValueError("work_id must be an OpenAlex ID such as W2741809807")
    work = _fetch_json(f"https://api.openalex.org/works/{work_id}", api_key)
    if not (work.get("open_access") or {}).get("is_oa"):
        raise ValueError("work is not marked open access by OpenAlex")
    best_oa = work.get("best_oa_location") or {}
    license_name = best_oa.get("license") or ""
    if not (license_name.startswith("cc-") or license_name in {"cc0", "public-domain"}):
        raise ValueError("work does not have a supported Creative Commons or public-domain license")

    content_urls = work.get("content_urls") or {}
    pdf_url = content_urls.get("pdf")
    grobid_url = content_urls.get("grobid_xml")
    if not pdf_url or not grobid_url:
        raise ValueError("work needs both OpenAlex PDF and GROBID content URLs")

    pdf_path = Path("/tmp/paper.pdf")
    xml_path = Path("/tmp/paper.tei.xml")
    pdf_sha256, pdf_bytes, pdf_type = _download(pdf_url, pdf_path, api_key)
    with pdf_path.open("rb") as pdf_handle:
        if pdf_handle.read(5) != b"%PDF-":
            raise ValueError(f"downloaded content is not a PDF (content-type {pdf_type!r})")
    xml_sha256, xml_bytes, xml_type = _download(grobid_url, xml_path, api_key)
    with xml_path.open("rb") as xml_handle:
        xml_head = xml_handle.read(80)
    if b"<" not in xml_head:
        raise ValueError(
            f"downloaded GROBID content is not XML (content-type {xml_type!r}, starts with {xml_head[:40]!r})"
        )
    markdown = _grobid_markdown(xml_path, work)

    pdf_key = f"papers/{work_id}.pdf"
    xml_key = f"sources/{work_id}.tei.xml"
    markdown_key = f"sources/{work_id}.md"
    metadata_key = f"candidates/{work_id}.json"
    s3.upload_file(str(pdf_path), BUCKET_NAME, pdf_key)
    s3.upload_file(str(xml_path), BUCKET_NAME, xml_key)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=markdown_key,
        Body=markdown.encode("utf-8"),
        ContentType="text/markdown; charset=utf-8",
    )
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=metadata_key,
        Body=(json.dumps(work, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json",
    )
    table.update_item(
        Key={"work_id": work_id},
        UpdateExpression=(
            "SET ingest_status = :status, pdf_key = :pdf, source_key = :source, "
            "pdf_sha256 = :pdf_hash, grobid_sha256 = :xml_hash"
        ),
        ExpressionAttributeValues={
            ":status": "fulltext_ready",
            ":pdf": pdf_key,
            ":source": markdown_key,
            ":pdf_hash": pdf_sha256,
            ":xml_hash": xml_sha256,
        },
    )
    return {
        "work_id": work_id,
        "status": "fulltext_ready",
        "license": license_name,
        "pdf_s3_uri": f"s3://{BUCKET_NAME}/{pdf_key}",
        "source_s3_uri": f"s3://{BUCKET_NAME}/{markdown_key}",
        "pdf_sha256": pdf_sha256,
        "pdf_bytes": pdf_bytes,
        "grobid_bytes": xml_bytes,
        "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }

def _validate_sections(text, sections, first):
    problems = []
    for section in sections:
        if section not in text:
            problems.append(f"missing section {section}")
    # Count headings at line start only: papers use "##" inline as a significance marker
    # ("# P < 0.001 and ## P < 0.0001"), and a bare count of the substring rejects a good note.
    if len(re.findall(r"^## ", text, re.M)) != len(sections):
        problems.append("unexpected number of level-2 sections")
    if not text.lstrip().startswith(first):
        problems.append(f"note must start with {first}")
    elif not re.search(r"^" + re.escape(first) + r"\s*$", text, re.M):
        problems.append(f"{first} must be a heading on its own line")
    if "---" in text.split(first)[0]:
        problems.append("frontmatter must not be model-written")
    if DRAFT_FORBIDDEN.search(text):
        problems.append("placeholder or filler text present")
    if len(text) < 1500:
        problems.append("note is too short to be a full evidence note")
    return problems

def _validate_draft(text):
    return _validate_sections(text, DRAFT_SECTIONS, "## Citation")
def _yaml_scalar(value):
    return json.dumps("" if value is None else str(value), ensure_ascii=False)

def _draft_frontmatter(work_id, work, item, model_id, usage, drafted_at, status, problems):
    authors = [(a.get("author") or {}).get("display_name") for a in (work.get("authorships") or [])]
    lines = [
        "---",
        f"work_id: {_yaml_scalar(work_id)}",
        f"doi: {_yaml_scalar(work.get('doi'))}",
        f"title: {_yaml_scalar(work.get('display_name'))}",
        f"authors: {json.dumps([a for a in authors if a][:25], ensure_ascii=False)}",
        f"publication_year: {work.get('publication_year') or ''}",
        f"journal: {_yaml_scalar(((work.get('primary_location') or {}).get('source') or {}).get('display_name'))}",
        f"pdf_path: {_yaml_scalar('s3://' + BUCKET_NAME + '/' + item.get('pdf_key', ''))}",
        f"pdf_sha256: {_yaml_scalar(item.get('pdf_sha256'))}",
        f"source_format: pdf",
        f"text_extractor: openalex-grobid",
        f"source_key: {_yaml_scalar(item.get('source_key'))}",
        f"grobid_sha256: {_yaml_scalar(item.get('grobid_sha256'))}",
        f"ingest_harness: aws-lambda-bedrock",
        f"ingest_model: {_yaml_scalar(model_id)}",
        f"ingest_input_tokens: {usage.get('inputTokens', 0)}",
        f"ingest_output_tokens: {usage.get('outputTokens', 0)}",
        f"drafted_at: {_yaml_scalar(drafted_at)}",
        f"draft_status: {status}",
        f"draft_problems: {json.dumps(problems, ensure_ascii=False)}",
        "review_status: unreviewed",
        "---",
        "",
    ]
    return "\n".join(lines)

def _converse(model_id, prompt):
    request = {
        "modelId": model_id,
        "system": [{"text": DRAFT_SYSTEM}, {"cachePoint": {"type": "default"}}],
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": DRAFT_MAX_OUTPUT_TOKENS},
    }
    try:
        return bedrock.converse(**request)
    except bedrock.exceptions.ValidationException as exc:
        if "cachePoint" not in str(exc):
            raise
        request["system"] = [{"text": DRAFT_SYSTEM}]
        return bedrock.converse(**request)

def _draft(event):
    work_id = str(event.get("work_id", "")).upper()
    if not WORK_ID_PATTERN.fullmatch(work_id):
        raise ValueError("work_id must be an OpenAlex ID such as W2741809807")
    model_id = str(event.get("model_id") or DRAFT_MODEL_ID)
    if not re.fullmatch(r"[a-z0-9.:-]+", model_id) or "anthropic" not in model_id:
        raise ValueError("model_id must be an Anthropic Bedrock model or inference profile id")
    item = table.get_item(Key={"work_id": work_id}).get("Item") or {}
    if item.get("ingest_status") not in {"fulltext_ready", "model_draft", "draft_failed"}:
        raise ValueError("draft requires a completed full-text ingest for this work")
    work = json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=f"candidates/{work_id}.json")["Body"].read())
    source = _strip_reporting_summary(
        s3.get_object(Bucket=BUCKET_NAME, Key=item["source_key"])["Body"].read().decode("utf-8"))
    if len(source) > DRAFT_MAX_INPUT_CHARS:
        raise ValueError("extracted text exceeds the draft input limit")
    prompt = (
        "Metadata (from OpenAlex):\n"
        + json.dumps({
            "title": work.get("display_name"), "doi": work.get("doi"),
            "publication_year": work.get("publication_year"),
            "journal": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "authors": [(a.get("author") or {}).get("display_name") for a in (work.get("authorships") or [])][:25],
        }, ensure_ascii=False, indent=2)
        + "\n\nExtracted full text (GROBID):\n\n" + source
        + WRITE_NOW.format(first=DRAFT_SECTIONS[0])
    )
    started = datetime.now(timezone.utc)
    response = _converse(model_id, prompt)
    text = "".join(block.get("text", "") for block in response["output"]["message"]["content"]).strip()
    usage = response.get("usage") or {}
    problems = _validate_draft(text)
    if response.get("stopReason") != "end_turn":
        problems.append(f"stop reason {response.get('stopReason')}")
    drafted_at = started.replace(microsecond=0).isoformat()
    seconds = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    status = "draft_failed" if problems else "model_draft"
    key = f"wiki/drafts/{'failed/' if problems else ''}{work_id}.md"
    page = _draft_frontmatter(work_id, work, item, model_id, usage, drafted_at, status, problems) + text + "\n"
    digest = hashlib.sha256(page.encode("utf-8")).hexdigest()
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=page.encode("utf-8"), ContentType="text/markdown; charset=utf-8")
    table.update_item(
        Key={"work_id": work_id},
        UpdateExpression=(
            "SET ingest_status = :status, draft_key = :key, draft_model = :model, "
            "draft_input_tokens = :inp, draft_output_tokens = :out, draft_sha256 = :digest, "
            "drafted_at = :at, draft_problems = :problems, review_status = :review, "
            "draft_seconds = :seconds, draft_cache_read_tokens = :cache_read, "
            "draft_cache_write_tokens = :cache_write"
        ),
        ExpressionAttributeValues={
            ":status": status, ":key": key, ":model": model_id,
            ":inp": int(usage.get("inputTokens", 0)), ":out": int(usage.get("outputTokens", 0)),
            ":digest": digest, ":at": drafted_at, ":problems": problems, ":review": "unreviewed",
            ":seconds": Decimal(str(seconds)),
            ":cache_read": int(usage.get("cacheReadInputTokens", 0)),
            ":cache_write": int(usage.get("cacheWriteInputTokens", 0)),
        },
    )
    return {
        "work_id": work_id, "status": status, "draft_s3_uri": f"s3://{BUCKET_NAME}/{key}",
        "model_id": model_id, "usage": {k: int(v) for k, v in usage.items() if isinstance(v, int)},
        "stop_reason": response.get("stopReason"), "problems": problems,
        "seconds": seconds, "draft_sha256": digest,
    }

def _reviewed_note(work_id):
    """Return (item, note text, key) for a work whose evidence note passed review."""
    item = table.get_item(Key={"work_id": work_id}).get("Item") or {}
    if item.get("review_status") != "reviewed" or not item.get("reviewed_key"):
        raise ValueError(f"{work_id} has no reviewed evidence note; promote its draft first")
    note = s3.get_object(Bucket=BUCKET_NAME, Key=item["reviewed_key"])["Body"].read().decode("utf-8")
    if hashlib.sha256(note.encode("utf-8")).hexdigest() != item.get("reviewed_sha256"):
        raise ValueError(f"{work_id}: the reviewed note in S3 does not match the hash recorded at promotion")
    return item, note, item["reviewed_key"]

# A throttled Bedrock call is rejected before any inference runs, so retrying costs nothing and
# bills nothing twice. A read timeout is the opposite: the model may still be generating, and a
# retry would run it, and bill it, again. Only the pre-inference rejections are retried here.
THROTTLE_CODES = ("ThrottlingException", "TooManyRequestsException",
                  "ServiceUnavailableException", "ModelNotReadyException")
THROTTLE_MAX_WAIT_SECONDS = 150


def _converse_with_backoff(client, request, *, budget=THROTTLE_MAX_WAIT_SECONDS):
    """Call Bedrock, waiting out throttling. Returns (response, seconds waited, attempts)."""
    delay, waited, attempts = 2.0, 0.0, 0
    while True:
        attempts += 1
        try:
            return client.converse(**request), round(waited, 1), attempts
        except botocore.exceptions.ClientError as exc:
            code = (exc.response.get("Error") or {}).get("Code")
            if code not in THROTTLE_CODES or waited + delay > budget:
                raise
            pause = delay + random.uniform(0, 1)
            time.sleep(pause)
            waited += pause
            delay = min(delay * 2, 30.0)


def _generate(system, prompt, sections, first, model_id):
    """Run one Bedrock call and validate the section structure. Returns a dict, never raises on content."""
    started = datetime.now(timezone.utc)
    thinking = INGEST_REASONING in EFFORT_LEVELS
    request = {
        "modelId": model_id,
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": DRAFT_MAX_OUTPUT_TOKENS
                            + (THINKING_HEADROOM.get(INGEST_REASONING, 24_000) if thinking else 0)},
    }
    if thinking:
        # Claude 5 on Bedrock: adaptive thinking steered by output_config.effort (low..max). The
        # reasoning blocks carry no "text" and are never written to pages.
        request["additionalModelRequestFields"] = {"thinking": {"type": "adaptive"}, "output_config": {"effort": INGEST_REASONING}}
    response, throttled_seconds, attempts = _converse_with_backoff(bedrock, request)
    text = "".join(block.get("text", "") for block in response["output"]["message"]["content"]).strip()
    usage = response.get("usage") or {}
    problems = _validate_sections(text, sections, first)
    if response.get("stopReason") != "end_turn":
        problems.append(f"stop reason {response.get('stopReason')}")
    return {
        "text": text, "usage": usage, "problems": problems, "stop_reason": response.get("stopReason"),
        "throttled_seconds": throttled_seconds, "attempts": attempts,
        "generated_at": started.replace(microsecond=0).isoformat(),
        "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }


CLASSIFY_SYSTEM = (
    "You file scientific papers into the folders of an existing literature wiki. You are given the "
    "folders the wiki already uses and a numbered list of paper titles. Reply with one line per "
    "paper, `number: folder`, and nothing else.\n\n"
    "Rules:\n"
    "- Use only a folder from the list, spelled exactly as given.\n"
    "- Choose where a reader of that field would look for the paper first. One folder per paper.\n"
    "- A paper about a method belongs to the biology it is used on when the title names one, and to "
    "the method's own folder when it does not.\n"
    "- Answer `other` only when no folder on the list fits; a rough fit is better than `other`.\n"
)
CLASSIFY_BATCH = 50
CLASSIFY_MAX = 400


def _note_titles(stems):
    """Each note's title as its own frontmatter states it, for the stems that have a note."""
    titles = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for stem, (text, etag) in zip(stems, pool.map(
                lambda s: _read_published(f"wiki/sources/{s}.md"), stems)):
            if etag is None:
                continue
            match = re.search(r'^title:\s*"?(.+?)"?\s*$', text.split("---")[1] if "---" in text else "", re.M)
            titles[stem] = (match.group(1).strip() if match else stem)[:300]
    return titles


# A page's own kind is not a field. 47 notes carry `category: "note"`, which is enough to pass the
# 20-note floor, so the classifier was being offered `note` as somewhere to file a paper
# (2026-09-24). None of these words can name a field, whatever a note's frontmatter says.
NON_FIELDS = frozenset({"note", "notes", "source", "sources", "overview", "overviews",
                        "question", "questions", "concept", "concepts", "index", "indexes",
                        "paper", "papers", "wiki"})


def _known_categories(index):
    """The folders the wiki already uses, learned from the notes rather than from a fixed list."""
    con, _etag = index()
    try:
        rows = con.execute("SELECT category, count(*) FROM docs WHERE doc_type = 'note' "
                           "GROUP BY category ORDER BY 2 DESC").fetchall()
    finally:
        con.close()
    return [c for c, n in rows if c and c != "other" and c not in NON_FIELDS and n >= 20]


def _set_catalogue_category(stem, category):
    """Put the field on the catalogue item too, because that is what the synthesis planner reads.

    A note carries its field in three places: its own frontmatter, `papers/{stem}/meta.json`, and
    the catalogue item DynamoDB holds. Search, the catalogs and a student's answer all come from the
    index, which is built from the frontmatter, so two of the three were enough for everything a
    person sees. The synthesis planner is the exception: `_ready_notes` scans the catalogue. Leaving
    it behind planned molecular-validation and spatial-seq as empty and partitioned single-cell-dl
    from a membership 178 notes out of date (2026-09-24).
    """
    try:
        table.update_item(Key={"work_id": stem}, UpdateExpression="SET category = :c",
                          ExpressionAttributeValues={":c": category},
                          ConditionExpression=Attr("work_id").exists())
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise
    return True


def _set_note_category(stem, category):
    """Rewrite the note's own `category:` line and meta.json, each conditional on what was read."""
    key = f"wiki/sources/{stem}.md"
    text, etag = _read_published(key)
    if etag is None:
        return "no_note"
    head, sep, body = text.partition("\n---\n")
    if not sep or not re.search(r"^category:", head, re.M):
        return "no_category_line"
    new_head = re.sub(r'^category:.*$', f'category: "{category}"', head, count=1, flags=re.M)
    if new_head == head:
        return "unchanged"
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=(new_head + sep + body).encode("utf-8"),
                      ContentType="text/markdown; charset=utf-8", IfMatch=etag)
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412"):
            return "raced"
        raise
    meta_key = f"papers/{stem}/meta.json"
    meta_text, meta_etag = _read_published(meta_key)
    if meta_etag is not None:
        try:
            meta = json.loads(meta_text)
            meta["category"] = category
            s3.put_object(Bucket=BUCKET_NAME, Key=meta_key, ContentType="application/json",
                          Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"), IfMatch=meta_etag)
        except (ValueError, botocore.exceptions.ClientError):
            return "note_written_meta_stale"
    if not _set_catalogue_category(stem, category):
        return "note_written_catalogue_missing"
    return "filed"


OPEN_FIELDS_KEY = "wiki/indexes/open-fields.json"


def _open_fields():
    """The fields a person opened, as `{slug: scope}`, whatever their note count."""
    text, _etag = _read_published(OPEN_FIELDS_KEY)
    if text is None:
        return {}
    try:
        stored = json.loads(text).get("fields") or {}
    except ValueError:
        return {}
    return {name: str(row.get("scope") or "") for name, row in stored.items() if isinstance(row, dict)}


def _remember_open_fields(folders):
    """Keep an opened field open, so the next paper for it is filed there rather than nearby.

    Without this a field of fewer than 20 notes disappears from the folder list the moment the call
    that opened it ends, and the next paper lands in whichever older field is closest - which is how
    the field would stay under 20 for ever.
    """
    text, etag = _read_published(OPEN_FIELDS_KEY)
    stored = {}
    if text is not None:
        try:
            stored = json.loads(text).get("fields") or {}
        except ValueError:
            stored = {}
    today = datetime.now(timezone.utc).date().isoformat()
    for name, scope in folders.items():
        row = stored.get(name) if isinstance(stored.get(name), dict) else {}
        stored[name] = {"scope": scope or row.get("scope", ""), "opened": row.get("opened", today)}
    body = json.dumps({"fields": stored}, ensure_ascii=False, indent=2).encode("utf-8")
    condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=OPEN_FIELDS_KEY, ContentType="application/json",
                      Body=body, **condition)
    except botocore.exceptions.ClientError:
        return False
    return True


def _sync_note_categories(event, index):
    """Make the catalogue agree with the notes about which field each one is in.

    The index is built from the notes' own frontmatter, so it is what a reader sees; the catalogue
    is what the synthesis planner partitions. Every note filed before `_set_catalogue_category`
    existed moved in the first and not the second, which is 1,336 notes from 2026-09-23 and 738
    more from 2026-09-24. Nothing here reads a model, and a note whose two records already agree is
    not written.
    """
    con, etag = index()
    try:
        wanted = {row[0]: row[1] for row in con.execute(
            "SELECT doc_id, category FROM docs WHERE doc_type = 'note' AND category IS NOT NULL "
            "AND category != ''").fetchall()}
    finally:
        con.close()
    items = _scan_ready_notes()
    apply = bool(event.get("apply"))
    differ, missing, changed = [], 0, {}
    for item in items:
        stem = item["work_id"]
        target = wanted.get(stem)
        if target is None or target in NON_FIELDS:
            missing += 1
            continue
        if item.get("category") == target:
            continue
        differ.append(stem)
        changed[target] = changed.get(target, 0) + 1
        if apply:
            _set_catalogue_category(stem, target)
    return {"catalogue_notes": len(items), "index_notes": len(wanted), "not_in_index": missing,
            "differ": len(differ), "applied": apply, "into": dict(sorted(changed.items())),
            "index_etag": etag, "execution": "aws"}


def _scan_ready_notes():
    """Every paper with a validated note, as the synthesis planner counts them."""
    items, request = [], {
        "FilterExpression": Attr("id_kind").eq("stem") & Attr("source_note_status").eq("source_ready"),
        "ProjectionExpression": "work_id, category"}
    while True:
        response = table.scan(**request)
        items.extend(response.get("Items", []))
        if not response.get("LastEvaluatedKey"):
            return items
        request["ExclusiveStartKey"] = response["LastEvaluatedKey"]


def _fields_action(event):
    """Read the fields a person opened, and open or close one.

    Closing matters as much as opening: a field that turns out to belong under another one, the way
    stochastic differential equations belong under ai-math (user, 2026-09-24), has to stop being
    offered to the classifier, or papers keep landing in a folder that is no longer a field. Closing
    a field does not move the notes that carry it; `file_notes` does that.
    """
    closing = [str(name) for name in (event.get("close") or [])]
    opening = _new_folders({"new_folders": event.get("open") or []})
    if opening:
        _remember_open_fields(opening)
    if closing:
        text, etag = _read_published(OPEN_FIELDS_KEY)
        stored = {}
        if text is not None:
            try:
                stored = json.loads(text).get("fields") or {}
            except ValueError:
                stored = {}
        kept = {name: row for name, row in stored.items() if name not in closing}
        body = json.dumps({"fields": kept}, ensure_ascii=False, indent=2).encode("utf-8")
        s3.put_object(Bucket=BUCKET_NAME, Key=OPEN_FIELDS_KEY, ContentType="application/json",
                      Body=body, **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}))
    return {"fields": _open_fields(), "opened": sorted(opening), "closed": sorted(closing),
            "execution": "aws"}


def _new_folders(event):
    """Folders a caller is opening, which no note carries yet, as `{slug: scope}`.

    `_known_categories` learns the folder list from the notes and keeps only folders of 20 or more,
    so a field nobody has filed into is invisible to the model and can never start. A person naming
    a field the lab works on is the one thing that breaks that circle (user, 2026-09-24), and the
    scope line is what separates two fields a title alone would not, such as spatial assays from
    the foundation models trained on them.
    """
    folders = {}
    for row in (event.get("new_folders") or []):
        name = str(row.get("name") or "").strip() if isinstance(row, dict) else str(row).strip()
        scope = str(row.get("scope") or "").strip()[:200] if isinstance(row, dict) else ""
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError(f"a new folder name must be a lowercase slug, not {name!r}")
        folders[name] = scope
    if len(folders) > 12:
        raise ValueError("open at most 12 new folders in one call")
    return folders


def _file_notes(event, index):
    """Put named notes in a named field, because a person said so. No model is called.

    The classifier reads a title and guesses; a person naming both the papers and the field is not
    guessing, and asking a model to agree would only add a way to be overruled. This is what moves
    the few papers a field boundary cuts across - the synthetic-enhancer papers sitting in
    immunology or chromosome-biology that belong under genomic-dl (user, 2026-09-24).

    The field must already exist, as a folder the notes use or one a person opened, so a typo
    cannot scatter notes into a folder nothing will ever look in.
    """
    stems = [str(s) for s in (event.get("stems") or [])]
    if not stems or len(stems) > CLASSIFY_MAX or len(set(stems)) != len(stems):
        raise ValueError(f"stems must list 1 to {CLASSIFY_MAX} distinct stems")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", s) for s in stems):
        raise ValueError("each stem must be a paper stem")
    category = str(event.get("category") or "").strip()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", category):
        raise ValueError(f"category must be a lowercase slug, not {category!r}")
    known = (set(_known_categories(index)) | set(_open_fields()) | {"other"}) - NON_FIELDS
    if category not in known:
        raise ValueError(f"{category} is not a field of this wiki; open it first with new_folders")
    apply = bool(event.get("apply"))
    results = [{"stem": s, "category": category,
                "state": "would_file" if not apply else _set_note_category(s, category)} for s in stems]
    counts = {}
    for row in results:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {"stems": len(stems), "category": category, "applied": apply, "states": counts,
            "papers": results, "execution": "aws"}


def _classify_notes(event, index):
    """File notes the intake left in `other` into the folders the wiki already uses.

    A paper uploaded to the shared folder has no llm-wiki folder to inherit, so every one of them
    landed in `other`: 1,336 notes on 2026-09-23, 1,123 of them cited by no synthesis. `other` is
    not a field, so a per-field catalog built over it is a catalog of nothing. The title alone
    decides the folder here - it is what a person reads to file a paper - which keeps the call small
    enough that the whole backlog costs about a dollar. Thinking is off: this is a labelling task.

    `new_folders` adds fields the wiki does not have yet, so a person can open one; see
    `_new_folders`.
    """
    stems = [str(s) for s in (event.get("stems") or [])]
    if not stems or len(stems) > CLASSIFY_MAX or len(set(stems)) != len(stems):
        raise ValueError(f"stems must list 1 to {CLASSIFY_MAX} distinct stems")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", s) for s in stems):
        raise ValueError("each stem must be a paper stem")
    apply = bool(event.get("apply"))
    model_id = _model_id(event)
    only_new = bool(event.get("only_new_folders"))
    only_into = {str(c) for c in (event.get("only_into") or [])}
    opening = _new_folders(event)
    already_open = _open_fields()
    if apply and opening:
        _remember_open_fields(opening)
    opening = {**already_open, **opening}
    categories = _known_categories(index)
    if not categories and not opening:
        raise ValueError("The index holds no categories to file into")
    categories = categories + [c for c in opening if c not in categories]
    folder_lines = "\n".join(f"- {c}: {opening[c]}" if opening.get(c) else f"- {c}" for c in categories)
    titles = _note_titles(stems)
    decided, usage, calls = {}, {}, 0
    for start in range(0, len(stems), CLASSIFY_BATCH):
        batch = [s for s in stems[start:start + CLASSIFY_BATCH] if s in titles]
        if not batch:
            continue
        prompt = ("Folders:\n" + folder_lines + "\n\nPapers:\n"
                  + "\n".join(f"{i}. {titles[s]}" for i, s in enumerate(batch, 1)))
        request = {"modelId": model_id, "system": [{"text": CLASSIFY_SYSTEM}],
                   "messages": [{"role": "user", "content": [{"text": prompt}]}],
                   "inferenceConfig": {"maxTokens": 4000}}
        response, _throttled, _attempts = _converse_with_backoff(bedrock, request)
        calls += 1
        for key, value in (response.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
        text = "".join(b.get("text", "") for b in response["output"]["message"]["content"])
        allowed = set(categories)
        for line in text.splitlines():
            match = re.match(r"\s*(\d+)\s*[:.]\s*([a-z0-9-]+)\s*$", line)
            if not match:
                continue
            number, category = int(match.group(1)), match.group(2)
            if 1 <= number <= len(batch) and category in allowed:
                decided[batch[number - 1]] = category
    # The title is what the model read, not a result; returning 1,336 of them is what pushed one
    # run past the 128 KiB response budget (2026-09-23).
    if only_new or only_into:
        # Adding a field should not re-file the rest of the wiki: a note the model moves between two
        # older folders would leave that folder's planned subtopics naming a note it no longer holds.
        # `only_into` says the same for an existing field, which is how a subtopic is gathered into
        # one the wiki already has, such as the maths under `statistics` (user, 2026-09-24).
        keep = set(only_into) | (set(opening) if only_new else set())
        decided = {s: c for s, c in decided.items() if c in keep}
    results = [{"stem": s, "category": decided.get(s),
                "state": ("would_file" if not apply else _set_note_category(s, decided[s])) if s in decided
                         else ("no_note" if s not in titles else "undecided")}
               for s in stems]
    counts = {}
    for row in results:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {"stems": len(stems), "decided": len(decided), "applied": apply, "model_id": model_id,
            "categories": len(categories), "open_fields": sorted(opening), "only_new_folders": only_new,
            "only_into": sorted(only_into),
            "model_calls": calls, "usage": usage,
            "states": counts, "papers": results,
            "execution": "aws"}


CATEGORY_CATALOG_PREFIX = "wiki/indexes/categories/"
CATALOG_SUMMARY_CHARS = 260


def _category_catalog_text(category, rows, coverage):
    """One field's catalog: every note in it, one line, with what a reader needs to choose.

    llm-wiki keeps one of these per category (`indexes/liver.md`, 287 papers, 132 KB) and it is the
    browse path its agents use before searching. Byeori had one flat `sources.md` of 12,878 lines
    over every field at once, which nothing can read (user, 2026-09-23). The summary comes from the
    note's own One-line Summary, so no model is called to build this.
    """
    orphans = coverage["notes"] - coverage["connected"]
    lines = [f"# Index - {category}", "",
             "> Per-category catalog of evidence notes. Generated by `aws-build-category-catalogs`.",
             "> Parent: [[indexes/categories]]", "",
             f"- Notes: {coverage['notes']}",
             f"- Synthesis coverage: {coverage['connected']}/{coverage['notes']} connected "
             f"({coverage['coverage']:.0%}), {orphans} orphans",
             "- An orphan is a note no concept or overview cites yet.", ""]
    for stem, title, year, journal, doi, summary, connected in rows:
        facts = " · ".join(p for p in (year, journal, doi) if p)
        mark = "" if connected else " *(orphan)*"
        line = f"- [[sources/{stem}|{title or stem}]]{mark}"
        if facts:
            line += f" — {facts}"
        if summary:
            line += f" — {summary[:CATALOG_SUMMARY_CHARS]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _categories_root_text(coverages):
    lines = ["# Index - categories", "",
             "> Every field the evidence notes are filed under, with how much synthesis reaches them.",
             "> Generated by `aws-build-category-catalogs`.", "",
             "| Category | Notes | Synthesis coverage | Catalog |", "|---|---:|---|---|"]
    for c in coverages:
        orphans = c["notes"] - c["connected"]
        lines.append(f"| {c['category']} | {c['notes']} | {c['coverage']:.0%} ({orphans} orphan) | "
                     f"[[indexes/categories/{c['category']}]] |")
    notes = sum(c["notes"] for c in coverages)
    connected = sum(c["connected"] for c in coverages)
    lines += ["", f"- Notes: {notes}, connected {connected}, orphans {notes - connected} "
                  f"({connected / notes:.0%} coverage)" if notes else ""]
    return "\n".join(lines) + "\n"


def _build_category_catalogs(event, index):
    """Write one catalog per field, and the root table of fields, from the index alone."""
    minimum = event.get("min_notes")
    minimum = 1 if minimum is None else minimum
    if type(minimum) is not int or minimum < 1:
        raise ValueError("min_notes must be a positive integer")
    con, etag = index()
    try:
        cited = {r[0] for r in con.execute(
            "SELECT DISTINCT to_id FROM links WHERE to_type = 'note' AND from_type IN ('concept', 'overview')")}
        notes = con.execute("SELECT doc_id, title, year, journal, doi, category, summary FROM docs "
                            "WHERE doc_type = 'note' ORDER BY title, doc_id").fetchall()
    finally:
        con.close()
    by_category = {}
    for stem, title, year, journal, doi, category, summary in notes:
        by_category.setdefault(category or "other", []).append(
            (stem, title, year, journal, doi, summary or "", stem in cited))
    written, errors = [], []
    coverages = []
    for category, rows in sorted(by_category.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        connected = sum(1 for r in rows if r[6])
        coverage = {"category": category, "notes": len(rows), "connected": connected,
                    "coverage": connected / len(rows) if rows else 0.0}
        coverages.append(coverage)
        if len(rows) < minimum:
            continue
        key = f"{CATEGORY_CATALOG_PREFIX}{category}.md"
        try:
            s3.put_object(Bucket=BUCKET_NAME, Key=key, ContentType="text/markdown; charset=utf-8",
                          Body=_category_catalog_text(category, rows, coverage).encode("utf-8"))
            written.append({"key": key, "notes": len(rows), "orphans": len(rows) - connected})
        except Exception as exc:  # noqa: BLE001 - one field must not stop the rest
            errors.append({"key": key, "error": str(exc)[:300]})
    root = f"{CATEGORY_CATALOG_PREFIX.rstrip('/')}.md"
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=root, ContentType="text/markdown; charset=utf-8",
                      Body=_categories_root_text(coverages).encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append({"key": root, "error": str(exc)[:300]})
    return {"catalogs": len(written), "root": root, "categories": len(coverages),
            "notes": sum(c["notes"] for c in coverages),
            "connected": sum(c["connected"] for c in coverages),
            "written": written, "errors": errors, "index_etag": etag, "execution": "aws"}

def _model_id(event):
    model_id = str(event.get("model_id") or DRAFT_MODEL_ID)
    if not re.fullmatch(r"[a-z0-9.:-]+", model_id) or "anthropic" not in model_id:
        raise ValueError("model_id must be an Anthropic Bedrock model or inference profile id")
    return model_id


def _synthesize(event):
    slug = str(event.get("topic", ""))
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        raise ValueError("topic must be a lowercase slug such as de-novo-variants-in-autism")
    title = str(event.get("title") or "").strip()
    if not title:
        raise ValueError("title is required")
    work_ids = [str(w).upper() for w in (event.get("work_ids") or [])]
    if not work_ids or len(work_ids) > MAX_TOPIC_NOTES or len(set(work_ids)) != len(work_ids):
        raise ValueError(f"work_ids must list 1 to {MAX_TOPIC_NOTES} distinct OpenAlex IDs")
    if not all(WORK_ID_PATTERN.fullmatch(w) for w in work_ids):
        raise ValueError("work_ids must be OpenAlex IDs such as W2741809807")
    existing, expected_etag = _read_published(f"wiki/overviews/{slug}.md")
    model_id = _model_id(event)
    notes = []
    sources = []
    for work_id in work_ids:
        item, note, note_key = _reviewed_note(work_id)
        notes.append(f"=== Evidence note {work_id} ===\n{note}\n")
        sources.append({"work_id": work_id, "key": note_key, "sha256": item.get("reviewed_sha256")})
    prompt = f"Topic: {title}\n\n" + "\n".join(notes)
    if len(prompt) > DRAFT_MAX_INPUT_CHARS:
        raise ValueError("the selected notes exceed the synthesis input limit")
    if existing:
        prompt += "\n\nExisting overview: preserve unrelated material and revise its scientific argument with the supplied evidence.\n" + existing
    result = _generate(TOPIC_SYSTEM, prompt, TOPIC_SECTIONS, "## Scope", model_id)
    status = "topic_failed" if result["problems"] else "model_topic"
    key = f"wiki/overviews/{'failed/' if result['problems'] else ''}{slug}.md"
    usage = result["usage"]
    frontmatter = "\n".join([
        "---",
        f"topic: {_yaml_scalar(slug)}",
        f"title: {_yaml_scalar(title)}",
        f"work_ids: {json.dumps(work_ids)}",
        f"source_notes: {json.dumps(sources, ensure_ascii=False)}",
        "ingest_harness: aws-lambda-bedrock",
        f"ingest_model: {_yaml_scalar(model_id)}",
        f"ingest_input_tokens: {usage.get('inputTokens', 0)}",
        f"ingest_output_tokens: {usage.get('outputTokens', 0)}",
        f"generated_at: {_yaml_scalar(result['generated_at'])}",
        f"topic_status: {status}",
        f"topic_problems: {json.dumps(result['problems'], ensure_ascii=False)}",
        "---", "",
    ])
    page = frontmatter + f"# {title}\n\n" + result["text"] + "\n"
    digest = hashlib.sha256(page.encode("utf-8")).hexdigest()
    publication = None
    if result["problems"]:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=page.encode("utf-8"), ContentType="text/markdown; charset=utf-8")
    else:
        from byeori.wiki_connections import publish_page
        publication = publish_page(s3, BUCKET_NAME, key, page, expected_etag=expected_etag,
                                   create_only=expected_etag is None)
        digest = publication["sha256"]

    return {
        "topic": slug, "title": title, "work_ids": work_ids, "status": status,
        "overview_s3_uri": f"s3://{BUCKET_NAME}/{key}", "overview_key": key, "model_id": model_id,
        "publication": publication,
        "usage": {k: int(v) for k, v in usage.items() if isinstance(v, int)},
        "stop_reason": result["stop_reason"], "problems": result["problems"],
        "seconds": result["seconds"], "topic_sha256": digest,
    }
def _wiki_index():
    """Keep the BM25 index in /tmp and refresh it only when the S3 object changed."""
    head = s3.head_object(Bucket=BUCKET_NAME, Key=WIKI_INDEX_KEY)
    etag = head.get("ETag", "")
    marker = WIKI_INDEX_LOCAL.with_suffix(".etag")
    if not WIKI_INDEX_LOCAL.exists() or not marker.exists() or marker.read_text() != etag:
        s3.download_file(BUCKET_NAME, WIKI_INDEX_KEY, str(WIKI_INDEX_LOCAL))
        marker.write_text(etag)
    return sqlite3.connect(f"file:{WIKI_INDEX_LOCAL}?mode=ro", uri=True), etag

def _section_texts(rows):
    """Fetch the text behind contentless hits: {(s3_key, section): text}.

    The index holds no page text, so a hit's body is read from its page in S3. Hits that share a
    page share one read, and the reads run together, so a search costs a few object reads rather
    than one per hit. Nothing is written and no page reaches the caller whole.
    """
    keys = list(dict.fromkeys(r["s3_key"] for r in rows if r.get("s3_key")))
    if not keys:
        return {}

    def read(key):
        try:
            body = s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read().decode("utf-8", "replace")
        except Exception:
            return key, {}
        if body.startswith("---\n"):
            end = body.find("\n---\n", 4)
            if end > 0:
                body = body[end + 5:]
        return key, {name: content for name, content in _split_sections(body)}

    with ThreadPoolExecutor(max_workers=min(16, len(keys))) as pool:
        pages = dict(pool.map(read, keys))
    return {(key, name): text for key, sections in pages.items() for name, text in sections.items()}


def _snippet(text, terms, window=320):
    """Bracket the query terms in a window around the first match, as snippet() used to do."""
    if not text:
        return ""
    lowered = text.lower()
    hit = min((lowered.find(t.lower()) for t in terms if t and lowered.find(t.lower()) >= 0), default=-1)
    start = max(0, hit - window // 3) if hit >= 0 else 0
    excerpt = text[start:start + window]
    for term in sorted({t for t in terms if len(t) > 2}, key=len, reverse=True):
        excerpt = re.sub(f"({re.escape(term)})", r"[\1]", excerpt, flags=re.I)
    return ("… " if start else "") + excerpt.strip().replace("\n", " ") + (" …" if start + window < len(text) else "")


def _wiki_search(event):
    query = str(event.get("query") or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    limit = int(event.get("limit") or 10)
    if not 1 <= limit <= 100:
        raise ValueError("limit must be 1 to 100")
    doc_type = event.get("doc_type")
    if doc_type not in (None, "note", "paper", "overview", "question", "concept"):
        raise ValueError("doc_type must be note, paper, overview, question, or concept")
    category = event.get("category")
    if category is not None and not re.fullmatch(r"[a-z0-9-]+", str(category)):
        raise ValueError("category must be a lowercase slug such as asd-ndd or long-read")
    from byeori.wiki_search import search_index
    terms = [t for t in re.split(r"\s+", query) if t]
    con, etag = _wiki_index()
    try:
        results = search_index(con, query, limit, doc_type=doc_type, category=category)
    finally:
        con.close()
    texts = _section_texts(results)
    for item in results:
        item["snippet"] = _snippet(texts.get((item.pop("s3_key"), item["section"]), ""), terms)
    return {"query": query, "index": f"s3://{BUCKET_NAME}/{WIKI_INDEX_KEY}", "index_etag": etag, "results": results}
def _stem_item(stem):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,200}", stem):
        raise ValueError("stem must be a lowercase llm-wiki stem such as abdi-2023-genomic-architecture-of-autism-spectrum")
    item = table.get_item(Key={"work_id": stem}).get("Item") or {}
    if item.get("id_kind") != "stem":
        raise ValueError(f"{stem} is not an uploaded paper")
    meta = json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=f"papers/{stem}/meta.json")["Body"].read())
    return item, meta


def _model_family(model_id):
    m = re.search(r"claude-([a-z]+)-(\d+)(?:-(\d+))?", model_id)
    if not m:
        return model_id, ""
    return m.group(1), m.group(2) + ("." + m.group(3) if m.group(3) else "")


def _source_collection(meta):
    """Where this paper came from, without asserting a provenance we were not told.

    `llm-wiki` stays the last resort because that is what every upload was until the shared folder
    existed, but a paper whose `meta.json` names its intake says so: the flat-folder uploader
    records `to-s3`, and reporting those as `llm-wiki` put a false line in 446 notes.
    """
    return meta.get("source_collection") or meta.get("source") or "llm-wiki"


def _llm_wiki_frontmatter(stem, item, meta, model_id, extra, *, harness=None, agent=None,
                          agent_version=None, reasoning=None):
    """Frontmatter in llm-wiki's schema (docs/WIKI-SCHEMA.md); values are JSON-quoted strings.

    The four provenance values default to this deployment's, and a caller that did not write the
    note here passes its own: a note written in a Claude Code session says so rather than carrying
    `aws-bedrock` (user, 2026-09-24).
    """
    family, version = _model_family(model_id)
    fields = [
        ("title", meta.get("title") or stem), ("authors", meta.get("authors") or ""), ("year", str(meta.get("year") or "")),
        ("doi", meta.get("doi") or ""),
    ]
    fields += extra
    fields += [
        ("category", meta.get("category") or "other"), ("pdf_path", f"s3://{BUCKET_NAME}/papers/{stem}/original.pdf"),
        ("pdf_filename", f"{stem}.pdf"), ("source_collection", _source_collection(meta)),
        ("source_format", "pdf"), ("text_extractor", item.get("text_extractor") or "grobid"),
        ("text_extracted_date", item.get("text_extracted_date") or ""),
        ("ingest_harness", harness or INGEST_HARNESS), ("ingest_agent", agent or INGEST_AGENT),
        ("ingest_agent_version", agent_version or INGEST_AGENT_VERSION),
        ("ingest_model", family), ("ingest_model_version", version),
        ("ingest_reasoning", reasoning or INGEST_REASONING),
        ("ingest_model_id", model_id),
        ("pdf_sha256", meta.get("pdf_sha256") or ""),
    ]
    for key in ("journal", "pmid", "pmcid"):
        if meta.get(key):
            fields.append((key, str(meta[key])))
    fields.append(("created", datetime.now(timezone.utc).date().isoformat()))
    return "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in fields) + "\n---\n\n"


def _document_information(meta):
    rows = [("Title", meta.get("title")), ("Authors", meta.get("authors")), ("Year", meta.get("year")),
            ("Venue", meta.get("journal")), ("DOI", meta.get("doi")), ("PMID", meta.get("pmid")),
            ("Category", meta.get("category")), ("Source collection", _source_collection(meta))]
    lines = ["## 1. Document Information", "", "| Field | Details |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows if v]
    return "\n".join(lines) + "\n"


def _read_published(key):
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return response["Body"].read().decode("utf-8"), response["ETag"]
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return "", None
        raise


def _model_slug(model_id):
    return re.sub(r"[^a-z0-9-]+", "-", model_id.rsplit("anthropic.", 1)[-1].lower()).strip("-")


def _note_models(event):
    """The model that writes a note, and the one that takes over if its output is filtered.

    A model the caller names is the one the caller wants, and a trial exists to see what one model
    does, so neither is swapped for the fallback.
    """
    if event.get("model_id"):
        return _model_id(event), None
    fallback = NOTE_FALLBACK_MODEL_ID if NOTE_FALLBACK_MODEL_ID and NOTE_FALLBACK_MODEL_ID != NOTE_MODEL_ID else None
    return NOTE_MODEL_ID, None if event.get("trial_run") else fallback


def _source_note(event):
    stem = str(event.get("stem", ""))
    model_id, fallback = _note_models(event)
    trial = event.get("trial_run")
    if trial is not None and not TRIAL_RUN_PATTERN.fullmatch(str(trial)):
        raise ValueError("trial_run must be a lowercase slug such as opus-5-5-vs-5-20260923")
    item, meta = _stem_item(stem)
    if item.get("ingest_status") not in ("fulltext_ready", "model_draft", "draft_failed"):
        raise ValueError(f"{stem} has no extracted text yet (ingest_status {item.get('ingest_status')})")
    # Models compared on one paper have to read the same thing, and the wiki's note is not part of it.
    existing, expected_etag = ("", None) if trial else _read_published(f"wiki/sources/{stem}.md")
    text = _strip_reporting_summary(
        s3.get_object(Bucket=BUCKET_NAME, Key=item["source_key"])["Body"].read().decode("utf-8"))
    if len(text) > DRAFT_MAX_INPUT_CHARS:
        text = text[:DRAFT_MAX_INPUT_CHARS]
    # A scan read by OCR misspells words and can misread digits. The model is told, so that a value
    # which looks garbled is flagged in the note rather than silently corrected or copied as fact.
    label = ("Extracted full text (GROBID, from OCR of a scanned PDF; characters and digits may be misread. "
             "Where a value or word looks garbled, flag it in the note and do not correct it from memory):"
             if item.get("text_preprocess") == "ocr" else "Extracted full text (GROBID):")
    prompt = "Metadata:\n" + json.dumps({k: meta.get(k) for k in ("title", "authors", "year", "journal", "doi", "category")},
                                          ensure_ascii=False, indent=2) + f"\n\n{label}\n\n" + text
    if existing:
        prompt += ("\n\nExisting source note: preserve its verified evidence and scientific connections unless "
                   "the original justifies correcting them. Preserve unrelated material.\n\n" + existing)
    prompt += WRITE_NOW.format(first="## One-line Summary")
    result = _generate(SOURCE_NOTE_SYSTEM, prompt, SOURCE_NOTE_SECTIONS, "## One-line Summary", model_id)
    filtered = None
    if fallback and result["stop_reason"] == "content_filtered":
        # The declined attempt is kept for its cost; the note, its frontmatter and the catalogue
        # name the model that actually wrote it.
        filtered = {"model_id": model_id, "stop_reason": result["stop_reason"],
                    "usage": {k: int(v) for k, v in result["usage"].items() if isinstance(v, int)},
                    "seconds": result["seconds"]}
        model_id = fallback
        result = _generate(SOURCE_NOTE_SYSTEM, prompt, SOURCE_NOTE_SECTIONS, "## One-line Summary", model_id)
    body = result["text"]
    # Insert the code-written Document Information table after the one-line summary.
    marker = "## 2. Key Contributions"
    if marker in body:
        head, tail = body.split(marker, 1)
        body = head.rstrip() + "\n\n" + _document_information(meta) + "\n" + marker + tail
    status = "source_failed" if result["problems"] else "source_ready"
    if trial:
        key = f"runs/model-trials/{trial}/{_model_slug(model_id)}/{stem}.md"
        page = _llm_wiki_frontmatter(stem, item, meta, model_id, []) + body + "\n"
        # Create-only: a trial that is run again under the same name refuses rather than replacing
        # the result someone may already have compared.
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=page.encode("utf-8"), ContentType="text/markdown; charset=utf-8",
                      IfNoneMatch="*")
        return {"stem": stem, "status": status, "key": key, "model_id": model_id, "trial_run": trial,
                "usage": {k: int(v) for k, v in result["usage"].items() if isinstance(v, int)},
                "problems": result["problems"], "seconds": result["seconds"], "stop_reason": result["stop_reason"],
                "sha256": hashlib.sha256(page.encode("utf-8")).hexdigest()}
    key = f"wiki/sources/{'failed/' if result['problems'] else ''}{stem}.md"
    page = _llm_wiki_frontmatter(stem, item, meta, model_id, []) + body + "\n"
    digest = hashlib.sha256(page.encode("utf-8")).hexdigest()
    publication = None
    if result["problems"]:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=page.encode("utf-8"), ContentType="text/markdown; charset=utf-8")
    else:
        from byeori.wiki_connections import publish_page
        publication = publish_page(s3, BUCKET_NAME, key, page, expected_etag=expected_etag,
                                   create_only=expected_etag is None)
        digest = publication["sha256"]

    usage = result["usage"]
    expression = ("SET source_note_status = :st, source_note_key = :key, source_note_sha256 = :digest, "
                  "source_note_model = :model, source_note_input_tokens = :inp, source_note_output_tokens = :out, "
                  "source_note_seconds = :sec, source_note_at = :at, source_note_problems = :problems")
    values = {":st": status, ":key": key, ":digest": digest, ":model": model_id,
              ":inp": int(usage.get("inputTokens", 0)), ":out": int(usage.get("outputTokens", 0)),
              ":sec": Decimal(str(result["seconds"])), ":at": result["generated_at"], ":problems": result["problems"]}
    if filtered:
        expression += (", source_note_filtered_model = :fmodel, source_note_filtered_input_tokens = :finp, "
                       "source_note_filtered_output_tokens = :fout")
        values |= {":fmodel": filtered["model_id"], ":finp": filtered["usage"].get("inputTokens", 0),
                   ":fout": filtered["usage"].get("outputTokens", 0)}
    else:
        # A decline recorded by an earlier generation of this note no longer describes it.
        expression += (" REMOVE source_note_filtered_model, source_note_filtered_input_tokens, "
                       "source_note_filtered_output_tokens")
    table.update_item(Key={"work_id": stem}, UpdateExpression=expression, ExpressionAttributeValues=values)
    return {"stem": stem, "status": status, "source_note_key": key, "model_id": model_id,
            "filtered_attempt": filtered, "publication": publication,
            "usage": {k: int(v) for k, v in usage.items() if isinstance(v, int)}, "problems": result["problems"],
            "seconds": result["seconds"], "source_note_sha256": digest, "stop_reason": result["stop_reason"]}



LOCAL_NOTE_PROVENANCE = {"harness": "claude-code", "agent": "byeori-note-local",
                         "agent_version": "v1", "reasoning": "default"}
EXTRACTION_MAX_CHARS = 40_000


def _read_extraction(event):
    """A window of one paper's stored extraction, for a note written outside AWS.

    The client can otherwise reach no `papers/` key: `AwsStore.get_text` allows only candidate and
    run JSON, and `wiki_ops._key` only `wiki/` and `sources/`. Writing a note means reading the
    paper, so the narrowest opening is this one - a window of `clean.md`, the same text the AWS note
    writer reads, and nothing else under `papers/` (user, 2026-09-24: write the note locally when
    asked, for a paper that has just arrived).
    """
    stem = str(event.get("stem", ""))
    item, _meta = _stem_item(stem)
    if item.get("ingest_status") not in ("fulltext_ready", "model_draft", "draft_failed",
                                         "fulltext_ready_unclassified"):
        raise ValueError(f"{stem} has no extracted text yet (ingest_status {item.get('ingest_status')})")
    start = event.get("start") or 0
    max_chars = event.get("max_chars") or EXTRACTION_MAX_CHARS
    if type(start) is not int or start < 0 or type(max_chars) is not int or not 1 <= max_chars <= EXTRACTION_MAX_CHARS:
        raise ValueError(f"start must be a nonnegative integer and max_chars 1 to {EXTRACTION_MAX_CHARS}")
    text = _strip_reporting_summary(
        s3.get_object(Bucket=BUCKET_NAME, Key=item["source_key"])["Body"].read().decode("utf-8"))
    window = text[start:start + max_chars]
    return {"stem": stem, "key": item["source_key"], "text": window, "start": start,
            "next_start": start + len(window) if start + len(window) < len(text) else None,
            "total_chars": len(text), "text_extractor": item.get("text_extractor"),
            "preprocess": item.get("text_preprocess"), "execution": "aws"}


def _publish_source_note(event):
    """Publish an evidence note whose body was written outside AWS, as if it had been written here.

    The body is the seven sections and nothing else; the frontmatter, the Document Information table,
    the section check, the catalog entry and the reciprocal links are all built here, so a locally
    written note is the same object as an AWS one and is reachable the same way. Only a paper with no
    note yet: replacing one would leave the first writer's provenance in place, because
    `wiki_connections._scientific_text` keeps the existing frontmatter.
    """
    stem = str(event.get("stem", ""))
    body = str(event.get("markdown") or "").strip()
    if not body:
        raise ValueError("markdown must be the note's seven sections")
    model_id = str(event.get("model_id") or "").strip()
    if not re.fullmatch(r"[a-z0-9.-]+", model_id):
        raise ValueError("model_id must name the model that actually wrote this note, e.g. claude-opus-5")
    item, meta = _stem_item(stem)
    if item.get("ingest_status") not in ("fulltext_ready", "model_draft", "draft_failed"):
        raise ValueError(f"{stem} has no extracted text yet (ingest_status {item.get('ingest_status')})")
    if item.get("source_note_key") and item.get("source_note_status") == "source_ready":
        raise ValueError(f"{stem} already has a note; this path only writes a paper's first note")
    problems = _validate_sections(body, SOURCE_NOTE_SECTIONS, "## One-line Summary")
    if problems:
        return {"stem": stem, "status": "source_failed", "problems": problems, "published": False,
                "execution": "aws"}
    marker = "## 2. Key Contributions"
    if marker in body:
        head, tail = body.split(marker, 1)
        body = head.rstrip() + "\n\n" + _document_information(meta) + "\n" + marker + tail
    key = f"wiki/sources/{stem}.md"
    page = _llm_wiki_frontmatter(stem, item, meta, model_id, [], **LOCAL_NOTE_PROVENANCE) + body + "\n"
    from byeori.wiki_connections import publish_page
    publication = publish_page(s3, BUCKET_NAME, key, page, create_only=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    # No token counts and no seconds: this note was not generated here, and a Bedrock price on it
    # would be a fabricated cost. Any recorded by an earlier attempt no longer describes it.
    table.update_item(
        Key={"work_id": stem},
        UpdateExpression=("SET source_note_status = :st, source_note_key = :key, source_note_sha256 = :digest, "
                          "source_note_model = :model, source_note_at = :at, source_note_problems = :problems, "
                          "source_note_written_by = :by "
                          "REMOVE source_note_input_tokens, source_note_output_tokens, source_note_seconds, "
                          "source_note_filtered_model, source_note_filtered_input_tokens, "
                          "source_note_filtered_output_tokens"),
        ExpressionAttributeValues={":st": "source_ready", ":key": key, ":digest": publication["sha256"],
                                   ":model": model_id, ":at": now, ":problems": [],
                                   ":by": LOCAL_NOTE_PROVENANCE["harness"]})
    return {"stem": stem, "status": "source_ready", "source_note_key": key, "model_id": model_id,
            "source_note_sha256": publication["sha256"], "publication": publication,
            "written_by": LOCAL_NOTE_PROVENANCE["harness"], "problems": [], "published": True,
            "execution": "aws"}

def _answer_question(event, context=None):
    from byeori.question_agent import run_answer
    remaining = getattr(context, "get_remaining_time_in_millis", None)

    def converse(client, request):
        seconds = remaining() / 1000 if remaining else 900
        final_call = "toolConfig" not in request
        # The text-only answer can take longer than a research turn. Wait within the
        # invocation deadline, retaining time to publish it and record the result.
        read_timeout = (max(5, int(seconds - 40)) if final_call
                        else max(5, min(240, int(seconds - 20))))
        # Do not let a later model call outlive the Lambda and lose its completed writes.
        bounded = boto3.client("bedrock-runtime", config=botocore.config.Config(
            read_timeout=read_timeout, connect_timeout=5,
            retries={"total_max_attempts": 1}))
        # A final call uses its available wait once; a backoff must not consume the
        # time reserved for saving its answer after this client timeout is computed.
        budget = 0 if final_call else min(30, max(0, seconds - 260))
        return _converse_with_backoff(bounded, request, budget=budget)

    agent_s3 = boto3.client("s3", config=botocore.config.Config(
        read_timeout=5, connect_timeout=3, retries={"total_max_attempts": 2}))
    return run_answer(event, s3=agent_s3, bucket=BUCKET_NAME, model_client=bedrock,
                      model_id=_model_id(event), reasoning=INGEST_REASONING,
                      converse=converse, search=_wiki_search, remaining_ms=remaining)


def _connect_wiki_pages(event):
    """Repair links/catalog entries for explicitly named existing pages, entirely in AWS."""
    from byeori.wiki_connections import publish_page
    keys = event.get("keys") or []
    if not isinstance(keys, list) or not 1 <= len(keys) <= 100:
        raise ValueError("keys must name 1 to 100 existing wiki pages")
    results, errors = [], []
    for key in dict.fromkeys(keys):
        try:
            if not isinstance(key, str) or not key.startswith("wiki/") or not key.endswith(".md"):
                raise ValueError("Expected a wiki Markdown key")
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            text = response["Body"].read().decode("utf-8")
            results.append(publish_page(s3, BUCKET_NAME, key, text, expected_etag=response["ETag"]))
        except Exception as exc:
            errors.append({"key": key, "error": str(exc)})
    return {"pages": results, "errors": errors, "execution": "aws"}


def _inspect_wiki_connections(event):
    """Inspect fresh S3 graph edges without sending full wiki pages to the client."""
    from byeori.wiki_connections import _page_key, _outgoing, _linked_key, WIKILINK
    key = _page_key(event.get("key"))
    text, etag = _read_published(key)
    if etag is None:
        raise ValueError(f"Page not found: {key}")
    targets = []
    for target in sorted(_outgoing(text) - {key}):
        other, other_etag = _read_published(target)
        links = {_linked_key(value) for value in WIKILINK.findall(other)}
        targets.append({"key": target, "exists": other_etag is not None, "reciprocal": key in links})
    folder = key.split("/")[1]
    catalog_key = f"wiki/indexes/{folder}.md"
    catalog, _ = _read_published(catalog_key)
    root, _ = _read_published("wiki/index.md")
    return {"key": key, "etag": etag, "chars": len(text), "targets": targets,
            "catalog_key": catalog_key,
            "catalog_registered": key in {_linked_key(value) for value in WIKILINK.findall(catalog)},
            "root_catalog_link": f"[[indexes/{folder}|" in root or f"[[indexes/{folder}]]" in root,
            "execution": "aws"}


def _split_sections(body):
    """Split a page body into (heading, text) the one way the index and the readers both use.

    The lead paragraph keeps an empty heading. Indexing and retrieval must agree exactly: a
    contentless index stores only the heading, so a drift here loses the text behind a hit.
    """
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    preamble = re.sub(r"^# .+\n?", "", parts[0].strip()).strip()
    rows = ([("", preamble)] if preamble else []) + list(zip(parts[1::2], parts[2::2]))
    return [(name.strip(), content.strip()) for name, content in rows if content.strip()]


def _index_documents(keys_and_bodies):
    """Yield (meta, sections, links) rows the same way the local builder does."""
    link_types = {"sources": "note", "concepts": "concept", "overviews": "overview", "questions": "question", "papers": "paper"}
    for key, text in keys_and_bodies:
        if key == "wiki/index.md" or key.startswith("wiki/indexes/"):
            continue
        parts = key.split("/")
        folder = parts[1] if len(parts) > 2 else "papers"
        # Nested pages keep their path in the id: overviews/asd-ndd/de-novo-variants, overviews/asd-ndd/index.
        doc_id = ("/".join(parts[2:]) if len(parts) > 2 else parts[-1])[:-3]
        doc_type = {"sources": "note", "overviews": "overview", "questions": "question", "concepts": "concept"}.get(folder, "paper")
        fields = {}
        body = text
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end > 0:
                for line in text[4:end].splitlines():
                    k, sep, v = line.partition(":")
                    if sep and not line.startswith(" "):
                        fields[k.strip()] = v.strip().strip('"')
                body = text[end + 5:]
        # An evidence note keeps the category its frontmatter states, so one filter reaches
        # both the note and the paper page of the same work. A subtopic or category page
        # keeps the category in its path; a concept spans categories and keeps its folder.
        if folder == "sources":
            category = fields.get("category") or "note"
        elif folder == "overviews" and len(parts) > 3:
            category = parts[2]
        else:
            category = folder
        title = fields.get("title") or ""
        if not title:
            head = re.search(r"^# (.+)$", body, re.M)
            title = head.group(1).strip() if head else doc_id
        work_ids = fields.get("work_ids", "")
        rows = _split_sections(body)
        # `sections` is a contentless FTS5 table, so the text is searchable but not readable back.
        # The per-field catalogs need one sentence per note, and this is where it is free to take.
        summary = next((" ".join(content.split())[:CATALOG_SUMMARY_CHARS]
                        for name, content in rows if name.startswith("One-line Summary")), "")
        meta = (doc_type, doc_id, title, f"data/{'sources' if folder == 'sources' else 'wiki/' + folder}/{doc_id}.md",
                str(fields.get("publication_year") or fields.get("year") or ""), fields.get("journal", ""),
                fields.get("doi", ""), work_ids if isinstance(work_ids, str) else "", category, key, summary)
        links = []
        if doc_type in ("note", "paper", "concept", "overview", "question"):
            for kind, ident in re.findall(r"\[\[(?:wiki/)?([a-z]+)/([^\]|#]+)(?:[|#][^\]]*)?\]\]", body):
                if kind in link_types:
                    links.append((link_types[kind], ident.strip().removesuffix(".md")))
        yield meta, rows, links


def _plan_notes(event):
    """List the papers still needing a note and leave the list in S3 for the state machine.

    This used to run on a laptop: scan the table, build the list, upload it, start the
    run. That made the first step of every run depend on a particular machine being
    open. Here a phone can start a run, because starting one carries no arguments.
    """
    only_failed = bool(event.get("only_failed"))
    limit = int(event.get("limit") or 0)
    scan = {"FilterExpression": Attr("id_kind").eq("stem"),
            "ProjectionExpression": "work_id, ingest_status, source_note_status"}
    items = []
    while True:
        page = table.scan(**scan)
        items.extend(page.get("Items", []))
        if not page.get("LastEvaluatedKey"):
            break
        scan["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    ready = ("fulltext_ready", "model_draft", "draft_failed")
    stems = []
    for item in sorted(items, key=lambda i: i["work_id"]):
        if item.get("ingest_status") not in ready:
            continue
        note = item.get("source_note_status")
        if note == "source_ready":
            continue
        # A paper never attempted has no status at all. On a retry pass that is not a
        # failure and is left to an ordinary run; on an ordinary run it is the work.
        if only_failed and not note:
            continue
        stems.append(item["work_id"])
    if limit:
        stems = stems[:limit]
    key = ("runs/manifests/notes-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
           + ("-failed" if only_failed else "") + ".json")
    s3.put_object(Bucket=BUCKET_NAME, Key=key, ContentType="application/json",
                  Body=json.dumps(stems).encode("utf-8"))
    return {"status": "planned", "manifest": {"bucket": BUCKET_NAME, "key": key},
            "count": len(stems), "only_failed": only_failed,
            "catalogued": len(items)}


def _index_excluded_notes():
    """Explicitly failed/stale catalogue notes cannot be made searchable by a retained S3 page."""
    request = {"FilterExpression": Attr("id_kind").eq("stem") & Attr("source_note_status").exists(),
               "ProjectionExpression": "work_id, source_note_status"}
    excluded = set()
    while True:
        response = table.scan(**request)
        excluded.update(item["work_id"] for item in response.get("Items", [])
                        if item.get("source_note_status") and item["source_note_status"] != "source_ready")
        if not response.get("LastEvaluatedKey"):
            return excluded
        request["ExclusiveStartKey"] = response["LastEvaluatedKey"]


def _build_wiki_index(event):
    """Build the BM25 index from the pages in S3 and publish it, with no local working copy.

    Synthesis pages carry [[links]]; those go into a links table, which is what serves
    backlinks (a note's "cited by") and the orphan report: notes no concept or subtopic cites.

    ``LAB_QUESTION_PREFIX`` is skipped (user, 2026-09-22). Every answered student question is kept
    as Markdown there so a reader can follow it, but at the lab's volume those pages would
    outnumber the source notes within a month and take the result slots the notes need. They are
    reached by link instead: each cited page carries one standing line to its own hub of questions.
    """
    from botocore.exceptions import ClientError

    started = datetime.now(timezone.utc)
    # Capture the publication version before taking the wiki snapshot. A slower build
    # must not replace an index another build publishes while this one is reading pages.
    try:
        index_etag = s3.head_object(Bucket=BUCKET_NAME, Key=WIKI_INDEX_KEY)["ETag"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
        index_etag = None
    paginator = s3.get_paginator("list_objects_v2")
    keys = [o["Key"] for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="wiki/")
            for o in page.get("Contents", [])
            if o["Key"].endswith(".md") and not o["Key"].startswith("wiki/drafts/") and "/failed/" not in o["Key"]
            and not o["Key"].startswith("wiki/indexes/") and not o["Key"].startswith(LAB_QUESTION_PREFIX)
            and len(o["Key"].split("/")) > 2]
    excluded = _index_excluded_notes()
    excluded_keys = {f"wiki/sources/{stem}.md" for stem in excluded}
    excluded_count = sum(key in excluded_keys for key in keys)
    keys = [key for key in keys if key not in excluded_keys]
    if not keys:
        raise ValueError("no wiki pages found in S3")

    def read(key):
        return key, s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")

    with ThreadPoolExecutor(max_workers=32) as pool:
        bodies = list(pool.map(read, keys))
    path = Path("/tmp/wiki-index.building")
    path.unlink(missing_ok=True)
    con = None
    try:
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE docs (doc_type TEXT, doc_id TEXT, title TEXT, path TEXT, year TEXT, journal TEXT, "
                    "doi TEXT, work_ids TEXT, category TEXT, s3_key TEXT, summary TEXT, PRIMARY KEY (doc_type, doc_id))")
        # content='' keeps the page text out of the file. Measured on the 13,151-page llm-wiki corpus,
        # storing it costs 1.99x the text and leaving it out costs 0.70x. detail stays at its default:
        # detail=column and detail=none shrink the file further but make bm25() return 0 for every row,
        # which removes ranking altogether.
        con.execute("CREATE VIRTUAL TABLE sections USING fts5(title, section, content, "
                    "tokenize='porter unicode61', content='')")
        con.execute("CREATE TABLE section_map (rowid INTEGER PRIMARY KEY, doc_type TEXT, doc_id TEXT, section TEXT)")
        con.execute("CREATE INDEX section_map_doc ON section_map (doc_type, doc_id)")
        con.execute("CREATE TABLE links (from_type TEXT, from_id TEXT, to_type TEXT, to_id TEXT)")
        con.execute("CREATE INDEX links_to ON links (to_type, to_id)")
        counts, rows, edges = {}, 0, 0
        catalog_documents = []
        for meta, sections, links in _index_documents(bodies):
            try:
                con.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?)", meta)
            except sqlite3.IntegrityError:
                continue  # the same id in two folders: keep the first
            counts[meta[0]] = counts.get(meta[0], 0) + 1
            catalog_documents.append((meta[9], meta[2]))
            for name, content in sections:
                cur = con.execute("INSERT INTO sections(title, section, content) VALUES (?,?,?)",
                                  (meta[2], name, content))
                con.execute("INSERT INTO section_map VALUES (?,?,?,?)", (cur.lastrowid, meta[0], meta[1], name))
            unique_links = list(dict.fromkeys(links))
            con.executemany("INSERT INTO links VALUES (?,?,?,?)", [(meta[0], meta[1], t, i) for t, i in unique_links])
            rows += len(sections)
            edges += len(unique_links)
        con.commit()
        orphans = [r[0] for r in con.execute(
            "SELECT doc_id FROM docs WHERE doc_type = 'note' AND doc_id NOT IN "
            "(SELECT to_id FROM links WHERE to_type = 'note' AND from_type IN ('concept', 'overview')) ORDER BY doc_id")]
        con.close()
        size = path.stat().st_size
        conditions = {"IfMatch": index_etag} if index_etag is not None else {"IfNoneMatch": "*"}
        try:
            with path.open("rb") as body:
                s3.put_object(Bucket=BUCKET_NAME, Key=WIKI_INDEX_KEY, Body=body,
                              ContentType="application/vnd.sqlite3", **conditions)
        except ClientError as exc:
            error = exc.response.get("Error", {}).get("Code")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error not in {"PreconditionFailed", "ConditionalRequestConflict", "412", "409"} and status not in {409, 412}:
                raise
            return {"status": "index_superseded", "published": False, "pages": len(keys),
                    "index_s3_uri": f"s3://{BUCKET_NAME}/{WIKI_INDEX_KEY}",
                    "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1)}
        s3.put_object(Bucket=BUCKET_NAME, Key="runs/synthesis/orphans.json", ContentType="application/json",
                      Body=json.dumps({"built_at": started.replace(microsecond=0).isoformat(), "count": len(orphans),
                                       "stems": orphans}).encode("utf-8"))
        WIKI_INDEX_LOCAL.unlink(missing_ok=True)
        WIKI_INDEX_LOCAL.with_suffix(".etag").unlink(missing_ok=True)
        from byeori.wiki_connections import rebuild_catalogs
        catalogs = rebuild_catalogs(s3, BUCKET_NAME, catalog_documents)
        return {"pages": len(keys), "documents": counts, "catalogs": catalogs, "sections": rows, "links": edges, "orphans": len(orphans),
                "excluded_nonready_notes": excluded_count,
                "index_bytes": size, "index_s3_uri": f"s3://{BUCKET_NAME}/{WIKI_INDEX_KEY}",
                "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1)}
    finally:
        if con is not None:
            con.close()
        path.unlink(missing_ok=True)

def _wiki_backlinks(event):
    """Pages that link to the given page, from the index's links table."""
    doc_type = str(event.get("doc_type") or "")
    doc_id = str(event.get("doc_id") or "")
    if doc_type not in ("note", "paper", "overview", "question", "concept") or not re.fullmatch(r"[A-Za-z0-9._/-]+", doc_id) or ".." in doc_id:
        raise ValueError("doc_type must be note, paper, overview, question, or concept and doc_id a stem or slug")
    con, etag = _wiki_index()
    links_table = True
    try:
        rows = con.execute("SELECT DISTINCT l.from_type, l.from_id, d.title FROM links l JOIN docs d ON d.doc_type = l.from_type "
                           "AND d.doc_id = l.from_id WHERE l.to_type = ? AND l.to_id = ? ORDER BY l.from_type, l.from_id",
                           (doc_type, doc_id)).fetchall()
    except sqlite3.OperationalError:
        rows, links_table = [], False  # an index built before the links table existed; rebuild it
    con.close()
    return {"doc_type": doc_type, "doc_id": doc_id, "index_etag": etag, "links_table": links_table,
            "cited_by": [{"doc_type": r[0], "doc_id": r[1], "title": r[2]} for r in rows]}
# Byeori is used through MCP from whatever machine a lab member is at, so a log kept on any one of
# those machines records only that person's share and nothing at all once 25 people are using it.
# The request and what came back are written here, in AWS, where every caller's traffic lands.
# Verbatim: the prompt is what every other record has to be read against (user, 2026-09-20).
LOGGED_ACTIONS = ("answer_question", "wiki_search", "wiki_read", "wiki_backlinks", "save_question",
                  "draft", "synthesize", "source_note", "connect_wiki_pages")
REQUEST_LOG_MAX = 40_000


def _log_request(event, result, started, error=None):
    """Append one object per call under runs/requests/{date}/. Never fails the call."""
    try:
        action = str(event.get("action") or "ingest")
        if action not in LOGGED_ACTIONS:
            return
        now = datetime.now(timezone.utc)
        caller = ""
        try:
            caller = boto3.client("sts").get_caller_identity().get("Arn", "")
        except Exception:
            pass
        request = {k: v for k, v in event.items() if k != "action"}
        record = {
            "at": now.replace(microsecond=0).isoformat(),
            "action": action,
            "caller": caller,
            "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 2),
            "request": json.loads(json.dumps(request, ensure_ascii=False, default=str)[:REQUEST_LOG_MAX]
                                  or "{}") if len(json.dumps(request, default=str)) <= REQUEST_LOG_MAX else request,
            "error": str(error)[:2000] if error else None,
        }
        if isinstance(result, dict):
            record["response"] = {k: v for k, v in result.items()
                                  if k in ("status", "question_key", "key", "sha256", "problems", "model_id",
                                           "model", "usage", "seconds", "answer_passes",
                                           "reread", "skipped_reason", "index_etag", "results", "count",
                                           "pages_written", "page_errors", "connections", "connection_errors",
                                           "trace_key", "tool_calls", "estimated_usd", "budget_usd")}
        key = f"runs/requests/{now:%Y-%m-%d}/{now:%H%M%S}-{action}-{uuid.uuid4().hex[:8]}.json"
        s3.put_object(Bucket=BUCKET_NAME, Key=key,
                      Body=(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8"),
                      ContentType="application/json")
    except Exception as exc:  # noqa: BLE001
        print(f"_log_request: {exc}")


def handler(event, _context):
    started = datetime.now(timezone.utc)
    try:
        result = _handle(event, _context)
    except Exception as exc:
        _log_request(event, None, started, error=exc)
        raise
    _log_request(event, result, started)
    return result


def _handle(event, _context):
    from byeori.wiki_ops import ACTIONS, dispatch
    if event.get("action") in ACTIONS:
        return dispatch(event, s3=s3, table=table, bucket=BUCKET_NAME, index=_wiki_index)
    action = event.get("action") or "ingest"
    if action == "metadata_review_plan":
        from byeori.metadata_review_plan import plan_metadata_review
        return plan_metadata_review(event, table=table, s3=s3, bucket=BUCKET_NAME)
    if action in ("metadata_review_plan_summary", "metadata_review_plan_inspect"):
        from byeori.metadata_review_plan import inspect_metadata_review, metadata_review_summary
        operation = inspect_metadata_review if action.endswith("inspect") else metadata_review_summary
        return operation(event, s3=s3, bucket=BUCKET_NAME)
    if action in ("metadata_review_progress", "metadata_review_results"):
        from byeori.metadata_review_worker import review_progress, review_results
        operation = review_results if action.endswith("results") else review_progress
        return operation(event, s3=s3, bucket=BUCKET_NAME)
    if action == "openalex_audit":
        from byeori.openalex_audit import audit_catalogue
        return audit_catalogue(table=table, s3=s3, bucket=BUCKET_NAME,
                               run_id=event.get("run_id"), sample_limit=event.get("sample_limit", 3))
    if action == "openalex_match_plan":
        from byeori.openalex_match_lambda import plan_match
        return plan_match(event, table=table)
    if action == "openalex_match_batch":
        from byeori.openalex_match_lambda import handle_match_batch
        return handle_match_batch(event, table=table, s3=s3, bucket=BUCKET_NAME, context=_context)
    if action == "draft":
        return _draft(event)
    if action == "synthesize":
        return _synthesize(event)
    if action == "wiki_search":
        return _wiki_search(event)
    if action == "wiki_backlinks":
        return _wiki_backlinks(event)
    if action == "source_note":
        return _source_note(event)
    if action == "answer_question":
        return _answer_question(event, _context)
    if action == "question_campaign_plan":
        from byeori.question_campaign import plan_batch
        return plan_batch(event, s3=s3, bucket=BUCKET_NAME)
    if action == "question_campaign_progress":
        from byeori.question_campaign import progress
        return progress(event, s3=s3, bucket=BUCKET_NAME)
    if action == "question_campaign_answer":
        from byeori.question_campaign import run_one
        return run_one(event, s3=s3, bucket=BUCKET_NAME,
                       answer_callback=lambda request: handler(request, _context))
    if action == "connect_wiki_pages":
        return _connect_wiki_pages(event)
    if action == "inspect_wiki_connections":
        return _inspect_wiki_connections(event)
    if action == "build_index":
        return _build_wiki_index(event)
    if action == "plan_notes":
        return _plan_notes(event)
    api_key = _api_key()
    if action == "search":
        return _search(event, api_key)
    if action == "get":
        return _get(event, api_key)
    if action == "ingest":
        return _ingest(event, api_key)
    if action == "resolve_identity":
        return _resolve_identity(event, api_key)
    if action == "classify_notes":
        return _classify_notes(event, _wiki_index)
    if action == "file_notes":
        return _file_notes(event, _wiki_index)
    if action == "fields":
        return _fields_action(event)
    if action == "sync_note_categories":
        return _sync_note_categories(event, _wiki_index)
    if action == "build_category_catalogs":
        return _build_category_catalogs(event, _wiki_index)
    if action == "read_extraction":
        return _read_extraction(event)
    if action == "publish_source_note":
        return _publish_source_note(event)
    raise ValueError("action must be search, get, ingest, resolve_identity, draft, synthesize, wiki_search, wiki_backlinks, source_note, answer_question, build_index, or plan_notes")
