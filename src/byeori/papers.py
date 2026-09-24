"""User-supplied PDFs: copy them from the local llm-wiki into S3, one folder per paper.

Layout (per paper, keyed by the llm-wiki stem ``{author}-{year}-{title-5-tokens}``):

    papers/{stem}/original.pdf      the PDF as it exists locally (sha256 recorded)
    papers/{stem}/meta.json         frontmatter of the local source note plus stem and hashes
    papers/{stem}/grobid.tei.xml    written later by the extraction worker
    papers/{stem}/clean.md          written later by the extraction worker (LLM input)

The DynamoDB item uses the stem as ``work_id`` with ``id_kind: stem``; DOI, PMID, and, when
resolved, the OpenAlex ID are attributes. Nothing local is moved or modified.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .config import Settings
from .contact import contact_email, user_agent

AUTISM_CATEGORIES = {"asd-ndd", "asd-models"}
# Document types this wiki does not ingest (user, 2026-09-18). A doctoral thesis extracts to
# hundreds of thousands of characters and is truncated mid-chapter, so the model writes from a
# fragment; a short report or a correspondence piece carries too little to write an evidence note
# from. Both produced the only two papers that failed three times in the autism run.
EXCLUDED_DOCUMENT_TYPES = {"thesis", "doctoral-thesis", "masters-thesis", "dissertation",
                           "short-report", "report", "technical-report", "research-briefing",
                           "editorial", "comment", "news-and-views", "correction", "erratum",
                           "addendum", "corrigendum", "retraction"}
# Publishers title these as "Addendum: <the original title>", and llm-wiki's stem keeps the word
# after the year. The paper they amend is in the corpus on its own; the notice is a page of prose
# that GROBID often cannot parse at all (user, 2026-09-18: these PDFs are deletion targets).
NOTICE_TITLE = re.compile(r"^\s*(addendum|erratum|corrigendum|retraction|author correction|"
                          r"publisher correction|editorial expression of concern)\s*[:\-]", re.I)
# "retraction" is left out of the stem rule on purpose: a stem carries no colon, and a paper
# titled "Retraction of neurite outgrowth by Sema3A" is ordinary biology. The title rule catches
# the notice form, "Retraction: <original title>".
NOTICE_STEM = re.compile(r"^[a-z\-]+-\d{4}-(addendum|erratum|corrigendum)-", re.I)


def is_amendment_notice(stem: str, fields: dict[str, str]) -> bool:
    """A notice that amends another paper rather than reporting work of its own."""
    return bool(NOTICE_TITLE.match(fields.get("title", "")) or NOTICE_STEM.match(stem))
FRONTMATTER_KEYS = ("title", "authors", "year", "doi", "pmid", "pmcid", "category", "journal", "pdf_filename",
                    "source_collection", "document_type")
_thread_local = threading.local()


def read_frontmatter(path: Path) -> dict[str, str]:
    """Flat ``key: value`` frontmatter of a llm-wiki page; values are unquoted strings."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        key, sep, value = line.partition(":")
        if sep and not line.startswith(" ") and key.strip() in FRONTMATTER_KEYS:
            fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_papers(llm_wiki: Path, *, select: str = "autism", stems: list[str] | None = None) -> list[dict[str, Any]]:
    """Papers to upload: llm-wiki source notes that have a PDF, filtered by ``select``."""
    pdfs = {p.stem: p for p in (llm_wiki / "papers").rglob("*.pdf")}
    chosen: list[dict[str, Any]] = []
    for note in sorted((llm_wiki / "sources").glob("*.md")):
        if stems and note.stem not in stems:
            continue
        pdf = pdfs.get(note.stem)
        if pdf is None:
            continue
        fields = read_frontmatter(note)
        document_type = (fields.get("document_type") or "").strip().lower()
        if document_type in EXCLUDED_DOCUMENT_TYPES or is_amendment_notice(note.stem, fields):
            continue
        if stems:
            chosen.append({"stem": note.stem, "pdf": pdf, "source_note": note, **fields})
            continue
        if select == "autism":
            head = note.read_text(encoding="utf-8", errors="ignore")[:1500].lower()
            if fields.get("category") not in AUTISM_CATEGORIES and "autis" not in head:
                continue
        elif select != "all":
            raise ValueError("select must be autism or all")
        chosen.append({"stem": note.stem, "pdf": pdf, "source_note": note, **fields})
    return chosen


def _s3():
    client = getattr(_thread_local, "s3", None)
    if client is None:
        client = boto3.Session().client("s3")
        _thread_local.s3 = client
    return client


def _table(settings: Settings):
    table = getattr(_thread_local, "table", None)
    if table is None:
        table = boto3.Session(region_name=settings.aws_region).resource("dynamodb").Table(settings.aws_table)
        _thread_local.table = table
    return table


def upload_one(settings: Settings, paper: dict[str, Any]) -> dict[str, Any]:
    stem = paper["stem"]
    pdf: Path = paper["pdf"]
    digest = sha256_file(pdf)
    pdf_key = f"papers/{stem}/original.pdf"
    meta_key = f"papers/{stem}/meta.json"
    s3 = _s3()
    state = "uploaded"
    try:
        head = s3.head_object(Bucket=settings.aws_bucket, Key=pdf_key)
        if head.get("Metadata", {}).get("sha256") == digest:
            state = "already_present"
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
            raise
    if state == "uploaded":
        s3.upload_file(str(pdf), settings.aws_bucket, pdf_key,
                       ExtraArgs={"ContentType": "application/pdf", "Metadata": {"sha256": digest, "stem": stem}})
    meta = {key: paper.get(key) for key in FRONTMATTER_KEYS if paper.get(key)}
    meta.update({"stem": stem, "pdf_key": pdf_key, "pdf_sha256": digest, "pdf_bytes": pdf.stat().st_size,
                 "local_pdf": str(pdf), "local_source_note": str(paper["source_note"]),
                 "uploaded_at": datetime.now(UTC).replace(microsecond=0).isoformat(), "id_kind": "stem"})
    s3.put_object(Bucket=settings.aws_bucket, Key=meta_key, Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
                  ContentType="application/json")
    if settings.aws_table:
        item = {k: v for k, v in meta.items() if k not in ("local_pdf", "local_source_note")}
        names = {f"#f{i}": k for i, k in enumerate(item)}
        values = {f":v{i}": v for i, v in enumerate(item.values())}
        values[":status"] = "pdf_uploaded"
        names["#st"] = "ingest_status"
        # A paper already extracted or drafted keeps its status; a fresh one becomes pdf_uploaded.
        _table(settings).update_item(
            Key={"work_id": stem},
            UpdateExpression="SET " + ", ".join(f"{n} = :v{n[2:]}" for n in names if n != "#st") + ", #st = if_not_exists(#st, :status)",
            ExpressionAttributeNames=names, ExpressionAttributeValues=values)
    return {"stem": stem, "state": state, "bytes": meta["pdf_bytes"], "pdf_sha256": digest}


def _log(message: str) -> None:
    print(f"{datetime.now(UTC).replace(microsecond=0).isoformat()} {message}", file=sys.stderr, flush=True)


def upload_papers(settings: Settings, llm_wiki: Path, *, select: str = "autism", stems: list[str] | None = None,
                  limit: int = 0, concurrency: int = 8, dry_run: bool = False) -> dict[str, Any]:
    if not settings.aws_bucket:
        raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured")
    papers = select_papers(llm_wiki, select=select, stems=stems)
    if limit:
        papers = papers[:limit]
    total_bytes = sum(p["pdf"].stat().st_size for p in papers)
    run: dict[str, Any] = {"started_at": datetime.now(UTC).replace(microsecond=0).isoformat(), "llm_wiki": str(llm_wiki),
                           "select": select, "selected": len(papers), "total_bytes": total_bytes, "dry_run": dry_run,
                           "results": [], "errors": []}
    if dry_run:
        run["stems"] = [p["stem"] for p in papers]
        return run
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(upload_one, settings, p): p["stem"] for p in papers}
        for index, future in enumerate(as_completed(futures), 1):
            stem = futures[future]
            try:
                result = future.result()
                run["results"].append(result)
                if index % 25 == 0 or index == len(papers):
                    done_bytes = sum(r["bytes"] for r in run["results"])
                    _log(f"[{index}/{len(papers)}] {done_bytes/1e9:.2f} GB, {time.monotonic()-started:.0f}s")
            except Exception as exc:
                run["errors"].append({"stem": stem, "error": str(exc)})
                _log(f"[{index}/{len(papers)}] {stem} error: {exc}")
    run["finished_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    run["seconds"] = round(time.monotonic() - started, 1)
    run["uploaded"] = sum(1 for r in run["results"] if r["state"] == "uploaded")
    run["already_present"] = sum(1 for r in run["results"] if r["state"] == "already_present")
    report = settings.state_dir / f"upload-{run['started_at'].replace(':', '')}.json"
    report.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    run["report"] = str(report)
    run["results"] = len(run["results"])
    return run


def _ids_from_work(work: dict[str, Any]) -> dict[str, str]:
    ids = work.get("ids") or {}
    out: dict[str, str] = {}
    if ids.get("openalex"):
        out["openalex_id"] = ids["openalex"].rsplit("/", 1)[-1]
    if ids.get("pmid"):
        out["pmid"] = ids["pmid"].rsplit("/", 1)[-1]
    if ids.get("pmcid"):
        out["pmcid"] = ids["pmcid"].rsplit("/", 1)[-1].upper()
    return out


# The three fields `ingest_lambda._llm_wiki_frontmatter` reads from meta.json, writing the stem and
# two blanks when they are missing.
BIBLIOGRAPHIC_FIELDS = ("title", "authors", "year")
NOTE_AUTHORS_IN_FULL = 25


def bibliographic_meta(record: dict[str, Any]) -> dict[str, str]:
    """A catalog record's title, authors and year, in the shape llm-wiki's notes carry them.

    An llm-wiki upload copies the three from the user's own source note. A paper from the shared
    folder has only the OpenAlex or Crossref record its identity was checked against, so they come
    from there: authors as one comma-separated string, the year as text. A consortium paper can list
    3,896 names; past 25 the first 20 and the last four are kept around "...", the way llm-wiki
    abbreviates, since the last names are usually the senior authors.
    """
    names = [name for name in (record.get("authors") or []) if name]
    if len(names) > NOTE_AUTHORS_IN_FULL:
        names = names[:20] + ["..."] + names[-4:]
    year = record.get("publication_year")
    fields = {"title": (record.get("title") or "").strip(), "authors": ", ".join(names),
              "year": str(int(year)) if year is not None else ""}
    return {key: value for key, value in fields.items() if value}


def merge_bibliography(meta: dict[str, Any], record: dict[str, Any], *, verified: bool) -> dict[str, Any]:
    """`meta` with the record's title, authors and year, for a verified identity only.

    A value already in meta.json came from the user's own note and is kept. One this merge wrote is
    listed in `bibliographic_fields`, so it is this merge's to correct, and to take back when the
    identity no longer holds: a disputed paper falls back to its file name rather than showing the
    title of a paper it may not be.
    """
    merged = dict(meta)
    for key in merged.pop("bibliographic_fields", None) or []:
        merged.pop(key, None)
    merged.pop("bibliographic_basis", None)
    if not verified:
        return merged
    offered = bibliographic_meta(record)
    written = [key for key in offered if not merged.get(key)]
    if written:
        merged.update({key: offered[key] for key in written})
        merged["bibliographic_fields"] = written
        # A record with no OpenAlex id came from Crossref, or from the PDF itself for a paper with no DOI.
        merged["bibliographic_basis"] = ("openalex" if record.get("work_id")
                                         else "pdf" if record.get("journal_source_basis") == "pdf" else "crossref")
    return merged


def _openalex_by_doi_direct(doi: str) -> dict[str, Any] | None:
    """Unkeyed OpenAlex singleton lookup from this machine (free; polite rate). None when 404."""
    import urllib.error
    import urllib.parse
    import urllib.request
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip().lower())
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent("byeori/0.1")})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"OpenAlex HTTP {exc.code}: {exc.read(300).decode('utf-8', 'replace')}") from None


NCBI_IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
NCBI_BATCH = 200


def _ncbi_ids_for_dois(dois: list[str], *, email: str | None = None) -> dict[str, dict[str, str]]:
    """DOI -> {pmid, pmcid} through NCBI's PMC ID converter. Free, batched, no OpenAlex budget.

    Only articles deposited in PMC resolve here; a DOI that is in PubMed but not PMC comes back
    as not found, and a DOI in neither is simply absent from the result. ``email`` defaults to the
    configured contact address (possibly empty; NCBI does not require it).
    """
    import urllib.error
    import urllib.parse
    import urllib.request
    address = email if email is not None else contact_email()
    found: dict[str, dict[str, str]] = {}

    def fetch(batch: list[str]) -> None:
        query = urllib.parse.urlencode({"format": "json", "tool": "byeori", "email": address, "ids": ",".join(batch)})
        request = urllib.request.Request(f"{NCBI_IDCONV}?{query}", headers={"User-Agent": "byeori/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            # One malformed identifier fails the whole batch; split until the bad one is alone.
            if exc.code == 400 and len(batch) > 1:
                fetch(batch[: len(batch) // 2])
                fetch(batch[len(batch) // 2:])
                return
            if exc.code == 400:
                return
            raise
        for record in payload.get("records") or []:
            doi = (record.get("doi") or "").lower()
            ids = {}
            if record.get("pmid"):
                ids["pmid"] = str(record["pmid"])
            if record.get("pmcid"):
                ids["pmcid"] = str(record["pmcid"])
            if doi and ids:
                found[doi] = ids
        time.sleep(0.4)  # NCBI asks for at most 3 requests per second without an API key

    clean = [d for d in dois if re.fullmatch(r"10\.\d{4,9}/[^\s,]+", d)]
    for start in range(0, len(clean), NCBI_BATCH):
        fetch(clean[start:start + NCBI_BATCH])
    return found


def resolve_ids_ncbi(settings: Settings, llm_wiki: Path, *, select: str = "autism", limit: int = 0) -> dict[str, Any]:
    """Fill pmid/pmcid from NCBI for uploaded papers that lack them. Free; no OpenAlex request."""
    if not settings.aws_bucket:
        raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured")
    papers = select_papers(llm_wiki, select=select)
    if limit:
        papers = papers[:limit]
    table = _table(settings) if settings.aws_table else None
    wanted: dict[str, str] = {}
    already = 0
    for paper in papers:
        stem, doi = paper["stem"], (paper.get("doi") or "").strip().lower()
        if not doi:
            continue
        item = table.get_item(Key={"work_id": stem}, ProjectionExpression="pmid").get("Item") if table else None
        if item and item.get("pmid"):
            already += 1
            continue
        wanted[re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)] = stem
    _log(f"{len(wanted)} papers need a PMID ({already} already have one); querying NCBI in batches of {NCBI_BATCH}")
    found = _ncbi_ids_for_dois(sorted(wanted)) if wanted else {}
    s3 = _s3()
    updated = 0
    for doi, ids in found.items():
        stem = wanted.get(doi)
        if not stem:
            continue
        meta_key = f"papers/{stem}/meta.json"
        try:
            meta = json.loads(s3.get_object(Bucket=settings.aws_bucket, Key=meta_key)["Body"].read())
        except ClientError:
            meta = {"stem": stem}
        meta.update(ids)
        s3.put_object(Bucket=settings.aws_bucket, Key=meta_key, Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
                      ContentType="application/json")
        if table:
            names = {f"#f{i}": k for i, k in enumerate(ids)}
            values = {f":v{i}": v for i, v in enumerate(ids.values())}
            table.update_item(Key={"work_id": stem}, UpdateExpression="SET " + ", ".join(f"{n} = :v{n[2:]}" for n in names),
                              ExpressionAttributeNames=names, ExpressionAttributeValues=values)
        updated += 1
    run = {"source": "ncbi-idconv", "selected": len(papers), "already_had_pmid": already, "queried": len(wanted),
           "resolved": updated, "not_in_pmc": len(wanted) - updated, "cost_usd": 0.0}
    report = settings.state_dir / f"resolve-ids-ncbi-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    report.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    run["report"] = str(report)
    return run


def resolve_ids(settings: Settings, llm_wiki: Path, *, select: str = "autism", stems: list[str] | None = None,
                limit: int = 0, concurrency: int = 4, direct: bool = False) -> dict[str, Any]:
    """Look each uploaded paper up in OpenAlex by DOI (through the Lambda, which holds the API key)
    and record openalex_id, pmid, pmcid in meta.json and DynamoDB. Papers without a DOI are listed
    for a title lookup. Lookups are OpenAlex singletons, which cost nothing."""
    from .aws_store import AwsStore
    if not settings.aws_bucket or not settings.aws_ingest_function:
        raise RuntimeError("bucket and ingest function must be configured")
    papers = select_papers(llm_wiki, select=select, stems=stems)
    if limit:
        papers = papers[:limit]
    run: dict[str, Any] = {"started_at": datetime.now(UTC).replace(microsecond=0).isoformat(), "selected": len(papers),
                           "resolved": 0, "already_resolved": 0, "with_pmid": 0, "not_found": [], "no_doi": [], "errors": []}
    local = threading.local()

    def store() -> AwsStore:
        st = getattr(local, "store", None)
        if st is None:
            st = AwsStore(settings)
            local.store = st
        return st

    def one(paper: dict[str, Any]) -> dict[str, Any]:
        stem = paper["stem"]
        doi = (paper.get("doi") or "").strip()
        if not doi:
            return {"stem": stem, "state": "no_doi"}
        if settings.aws_table:
            existing = _table(settings).get_item(Key={"work_id": stem}, ProjectionExpression="openalex_id, pmid").get("Item") or {}
            if existing.get("openalex_id"):
                return {"stem": stem, "state": "resolved", "openalex_id": existing["openalex_id"], "pmid": existing.get("pmid"), "cached": True}
        if direct:
            work = _openalex_by_doi_direct(doi)
            if work is None:
                return {"stem": stem, "state": "not_found", "doi": doi}
        else:
            try:
                work = store().get_openalex_work_by_doi(doi)
            except RuntimeError as exc:
                if "metadata request failed" in str(exc):
                    return {"stem": stem, "state": "not_found", "doi": doi}
                raise
        ids = _ids_from_work(work)
        s3 = _s3()
        meta_key = f"papers/{stem}/meta.json"
        try:
            meta = json.loads(s3.get_object(Bucket=settings.aws_bucket, Key=meta_key)["Body"].read())
        except ClientError:
            meta = {"stem": stem}
        meta.update(ids)
        meta["openalex_title"] = work.get("display_name")
        s3.put_object(Bucket=settings.aws_bucket, Key=meta_key, Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
                      ContentType="application/json")
        if settings.aws_table and ids:
            names = {f"#f{i}": k for i, k in enumerate(ids)}
            values = {f":v{i}": v for i, v in enumerate(ids.values())}
            _table(settings).update_item(Key={"work_id": stem},
                                         UpdateExpression="SET " + ", ".join(f"{n} = :v{n[2:]}" for n in names),
                                         ExpressionAttributeNames=names, ExpressionAttributeValues=values)
        return {"stem": stem, "state": "resolved", **ids}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, p): p["stem"] for p in papers}
        for index, future in enumerate(as_completed(futures), 1):
            stem = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                run["errors"].append({"stem": stem, "error": str(exc)})
                continue
            if result["state"] == "resolved":
                run["resolved"] += 1
                run["already_resolved"] += 1 if result.get("cached") else 0
                run["with_pmid"] += 1 if result.get("pmid") else 0
            elif result["state"] == "not_found":
                run["not_found"].append({"stem": stem, "doi": result["doi"]})
            else:
                run["no_doi"].append(stem)
            if index % 100 == 0 or index == len(papers):
                _log(f"[{index}/{len(papers)}] resolved {run['resolved']}, pmid {run['with_pmid']}")
    run["finished_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    report = settings.state_dir / f"resolve-ids-{run['started_at'].replace(':', '')}.json"
    report.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    run["report"] = str(report)
    return run
