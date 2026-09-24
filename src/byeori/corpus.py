from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Any

from .aws_store import AwsStore
from .catalog import AwsCatalog as Catalog, utc_now
from .config import Settings
from .journal_policy import allowed_verdicts, apply_journal_policy
from .openalex import OpenAlexClient, eligible_fulltext, normalize_doi  # noqa: F401 - re-exported


CORPUS_ID = "autism-genomics"
QUERIES = (
    "autism genomics", "autism exome sequencing", "autism whole genome sequencing",
    "autism de novo mutation", "autism copy number variation",
    "autism genome wide association", "autism rare variants", "autism polygenic",
)
TAG_PATTERNS = {
    "WES": r"\bexom|\bwes\b",
    "WGS": r"whole[ -]genome|genome[ -]sequenc|\bwgs\b",
    "CNV/SV": r"copy[ -]number|structural vari|\bcnvs?\b|chromosom|microdelet|microduplic",
    "De novo": r"de[ -]novo",
    "GWAS/polygenic": r"genome[ -]wide association|\bgwas\b|polygenic|common (?:genetic )?variant|heritab",
    "Rare variants": r"rare (?:coding |genetic |protein.truncating )?variant|rare mutation",
}
GENETICS = re.compile(
    r"genom|genetic|exom|variant|mutation|copy[ -]number|polygenic|heritab|"
    r"chromosom|sequenc|haploinsuff|mosaic|genotyp|\bgene[ s-]", re.I
)


def date_range() -> tuple[str, str]:
    today = date.today()
    try:
        start = today.replace(year=today.year - 15)
    except ValueError:
        start = today.replace(year=today.year - 15, day=28)
    return start.isoformat(), today.isoformat()


def screen_work(work: dict[str, Any], start: str, end: str) -> tuple[str | None, str]:
    """Heuristic discovery screening only; never a scientific or human-study assessment."""
    try:
        published = date.fromisoformat(work.get("publication_date") or "")
    except (ValueError, TypeError):
        return None, "missing_publication_date"
    if not date.fromisoformat(start) <= published <= date.fromisoformat(end):
        return None, "outside_date_range"
    if work.get("type") not in {"article", "review", "preprint"}:
        return None, "other_publication_type"
    title = work.get("title") or ""
    topics = " ".join(work.get("topics") or [])
    title_autism = bool(re.search(r"autis", title, re.I))
    topic_autism = bool(re.search(r"autis", topics, re.I))
    if not (title_autism or topic_autism):
        return None, "no_autism_title_or_topic"
    if not (GENETICS.search(title) or (title_autism and GENETICS.search(topics))):
        return None, "no_genetics_title_or_topic"
    return ("autism_title" if title_autism else "related_topic"), "candidate"


def annotate(work: dict[str, Any], scope: str, query_ids: list[str]) -> dict[str, Any]:
    title = work.get("title") or ""
    tags = [name for name, pattern in TAG_PATTERNS.items() if re.search(pattern, title, re.I)]
    annotated = dict(work)
    annotated["corpus"] = {
        "id": CORPUS_ID, "scope": scope, "tags": tags or ["Genetics/general"],
        "screening": "metadata_candidate", "tag_basis": "title_keywords",
        "query_ids": sorted(set(query_ids)), "fulltext_eligible": eligible_fulltext(work),
    }
    annotated["catalog_search_text"] = " ".join([
        title, work.get("doi") or "", work.get("source") or "",
        " ".join(work.get("authors") or []), " ".join(work.get("topics") or []),
    ]).lower()
    return apply_journal_policy(annotated)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(
    settings: Settings, *, start: str, end: str, per_query_limit: int = 100,
    publish_aws: bool = False, resume: Path | None = None,
) -> dict[str, Any]:
    if not all((settings.aws_bucket, settings.aws_table, settings.aws_ingest_function)):
        raise RuntimeError("Candidate collection requires the configured AWS bucket, table, and Lambda")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or not 1 <= per_query_limit <= 100:
        raise ValueError("date range must be ordered and per-query limit must be 1 to 100")
    config = {"start": start, "end": end, "per_query_limit": per_query_limit,
              "queries": list(QUERIES), "provider": "aws-lambda" if settings.aws_ingest_function else "openalex-api"}
    if resume:
        run_dir = resume.resolve()
        if json.loads((run_dir / "config.json").read_text()) != config:
            raise ValueError("resume parameters differ from the saved collection configuration")
    else:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = settings.state_dir / "corpus" / CORPUS_ID / "runs" / run_id
        write_json(run_dir / "config.json", config)
    store = AwsStore(settings) if settings.aws_ingest_function or publish_aws else None
    client = OpenAlexClient(settings.openalex_api_key) if not settings.aws_ingest_function else None
    manifest: dict[str, Any] = {
        "corpus": CORPUS_ID, "configuration": config, "run_dir": str(run_dir),
        "started_at": utc_now(), "queries": [], "errors": [],
        "coverage": "Bounded keyword discovery by year, not an exhaustive or systematically reviewed corpus.",
    }
    selected: dict[str, dict[str, Any]] = {}
    doi_to_id: dict[str, str] = {}
    work_to_id: dict[str, str] = {}
    excluded: Counter[str] = Counter()
    occurrences = 0
    duplicate_hits = 0
    try:
        for year in range(first.year, last.year + 1):
            for index, query in enumerate(QUERIES):
                query_id = f"{year}-{index + 1:02d}"
                cache_key = f"runs/discovery/{run_dir.name}/{query_id}.json"
                try:
                    from botocore.exceptions import ClientError
                    try:
                        works = json.loads(store.get_text(cache_key))["works"]
                    except ClientError as exc:
                        if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
                            raise
                        works = store.search_openalex(query, limit=per_query_limit, from_year=year, to_year=year)
                        store.upload_json({"query": query, "year": year, "fetched_at": utc_now(), "works": works}, cache_key)
                except Exception as exc:
                    # Preserve completed requests; a subsequent --resume retries failed requests only.
                    manifest["errors"].append({"query_id": query_id, "error_type": type(exc).__name__})
                    print(f"{query_id}: request failed ({type(exc).__name__})", file=sys.stderr, flush=True)
                    continue
                manifest["queries"].append({"id": query_id, "query": query, "year": year,
                                            "returned": len(works), "at_limit": len(works) == per_query_limit})
                for work in works:
                    occurrences += 1
                    # Retain canonical DOI suffixes, including punctuation, in older cached responses.
                    raw_doi = (work.get("raw") or {}).get("doi")
                    if raw_doi:
                        work["doi"] = normalize_doi(raw_doi)
                    scope, reason = screen_work(work, start, end)
                    if scope is None or not work.get("work_id"):
                        excluded[reason if scope is None else "missing_work_id"] += 1
                        continue
                    doi = normalize_doi(work.get("doi"))
                    known_id = work_to_id.get(work["work_id"])
                    doi_id = doi_to_id.get(doi) if doi else None
                    if known_id and doi_id and known_id != doi_id:
                        # A later DOI can connect two records previously seen under separate IDs.
                        previous_hit = selected.pop(known_id)
                        selected[doi_id]["corpus"]["query_ids"].extend(previous_hit["corpus"]["query_ids"])
                        for aliases in (work_to_id, doi_to_id):
                            for alias, target in list(aliases.items()):
                                if target == known_id:
                                    aliases[alias] = doi_id
                        duplicate_hits += 1
                        known_id = doi_id
                    canonical_id = doi_id or known_id or work["work_id"]
                    if canonical_id in selected:
                        duplicate_hits += 1
                        selected[canonical_id]["corpus"]["query_ids"].append(query_id)
                        if doi and not selected[canonical_id].get("doi"):
                            selected[canonical_id]["doi"] = doi
                            selected[canonical_id]["catalog_search_text"] += " " + doi
                    else:
                        selected[canonical_id] = annotate(work, scope, [query_id])
                    work_to_id[work["work_id"]] = canonical_id
                    if doi:
                        doi_to_id[doi] = canonical_id
                print(f"{query_id}: {len(works)} hits, {len(selected)} unique candidates", file=sys.stderr, flush=True)
    finally:
        if client:
            client.close()
    saved = []
    with Catalog(settings) as catalog:
        for work in selected.values():
            identifier = work.get("doi") or work["work_id"]
            try:
                previous = catalog.get_candidate(identifier)
            except KeyError:
                previous = None
            if previous:
                # Keep the existing canonical identity, richer metadata, and local curation.
                old_record = previous["record"]
                merged = old_record | {key: value for key, value in work.items() if value is not None}
                if old_record.get("abstract"):
                    merged["abstract"] = old_record["abstract"]
                if old_record.get("raw") and len(json.dumps(old_record["raw"])) > len(json.dumps(work.get("raw", {}))):
                    merged["raw"] = old_record["raw"]
                merged["work_id"] = previous["work_id"]
                work = merged
            work["corpus"]["query_ids"] = sorted(set(work["corpus"]["query_ids"]))
            saved.append(catalog.save_candidate(work))
    manifest.update({
        "completed_at": utc_now(), "returned_occurrences": occurrences,
        "excluded_occurrences": dict(excluded), "duplicate_occurrences": duplicate_hits,
        "candidate_count": len(saved), "work_ids": [item["work_id"] for item in saved],
        "at_limit_queries": sum(item["at_limit"] for item in manifest["queries"]),
        "by_year": dict(sorted(Counter(item["publication_year"] for item in saved).items())),
        "by_scope": dict(Counter(item["record"]["corpus"]["scope"] for item in saved)),
        "fulltext_eligible": sum(item["record"]["corpus"]["fulltext_eligible"] for item in saved),
        "aws_published": len(saved),
    })
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    if publish_aws:
        bundle = {"manifest": manifest, "records": [
            {key: value for key, value in item["record"].items() if key != "raw"} for item in saved
        ]}
        try:
            manifest["s3_export"] = store.upload_json(bundle, f"candidates/{CORPUS_ID}/{run_dir.name}.json")
        except Exception as exc:
            manifest["errors"].append({"stage": "s3_export", "error_type": type(exc).__name__})
        write_json(manifest_path, manifest)
    write_json(settings.state_dir / "corpus" / CORPUS_ID / "latest-run.json", {"run_dir": str(run_dir)})
    return manifest


def compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    work = item["record"]
    corpus = work.get("corpus", {})
    return {
        "work_id": item["work_id"], "doi": item.get("doi"), "title": item["title"],
        "publication_year": item.get("publication_year"), "publication_date": work.get("publication_date"),
        "authors": work.get("authors", []), "source": work.get("source"), "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count", 0), "is_open_access": work.get("is_open_access", False),
        "tags": corpus.get("tags", []), "scope": corpus.get("scope"),
        "fulltext_eligible": corpus.get("fulltext_eligible", False),
        "journal_verdict": corpus.get("journal_verdict", "unclassified"),
        "journal_family": corpus.get("journal_family"),
        # Which intake a hit came from, and how far it got, so a cross-corpus result is readable.
        "corpus": corpus.get("id", "unclassified"), "screening": corpus.get("screening"),
        "identity_status": item.get("identity_status"),
        "ingest_status": item.get("ingest_status", "not_recorded"),
        "review_status": "metadata_candidate", "search_text": work.get("catalog_search_text", ""),
    }


def search_corpus(settings: Settings, *, backend: str = "aws", limit: int = 50,
                  journal_policy: str = "include", corpus: str = "all",
                  **filters: Any) -> dict[str, Any]:
    """Search stored metadata. By default only allowlisted journals are returned.

    ``corpus`` is an intake id or ``all``. It searches every intake by default because the catalog
    holds more than the autism-genomics pilot it started as, and a search that silently covered only
    one of them reported papers as absent when they were merely in another.
    """
    if not 1 <= limit <= 10000:
        raise ValueError("limit must be 1 to 10000")
    if filters.get("from_year") and filters.get("to_year") and filters["from_year"] > filters["to_year"]:
        raise ValueError("from-year must not exceed to-year")
    verdicts = allowed_verdicts(journal_policy)
    if backend == "aws":
        response = AwsStore(settings).search_corpus(
            journal_verdicts=verdicts, corpus_id=None if corpus == "all" else corpus, **filters)
        items = response.pop("items")
        details = response | {"method": "DynamoDB paginated Scan with metadata filters"}
    else:
        raise ValueError("The local corpus backend was removed; use aws")
    items.sort(key=lambda item: (int(item["record"].get("cited_by_count") or 0),
                                 item.get("publication_year") or 0, item["work_id"]), reverse=True)
    return {"backend": backend, "journal_policy": journal_policy, "corpus": corpus,
            "total": len(items), "returned": min(limit, len(items)), **details,
            "results": [compact_candidate(item) for item in items[:limit]]}


def apply_policy_to_catalog(settings: Settings, *, publish_aws: bool = False) -> dict[str, Any]:
    """Re-classify stored candidate journals in AWS without local metadata copies."""
    verdicts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    review_journals: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    published = 0
    superseded = 0
    with Catalog(settings) as catalog:
        candidates = catalog.list_candidates()
        for item in candidates:
            if item["record"].get("corpus", {}).get("id") != CORPUS_ID:
                continue
            if item.get("superseded_by"):
                # Retired as a duplicate. Its DOI belongs to the row that superseded it, so
                # `save_candidate` would refuse it, and reclassifying it would mean nothing.
                superseded += 1
                continue
            record = apply_journal_policy(item["record"])
            corpus = record["corpus"]
            verdicts[corpus["journal_verdict"]] += 1
            if corpus["journal_verdict"] == "include":
                families[corpus["journal_family"]] += 1
            if corpus["journal_verdict"] == "review":
                review_journals[record.get("source") or "<missing>"] += 1
            try:
                catalog.save_candidate(record)
                published += 1
            except Exception as exc:
                errors.append({"work_id": item["work_id"], "error_type": type(exc).__name__})
    return {"classified": sum(verdicts.values()), "verdicts": dict(verdicts),
            "included_by_family": dict(families), "review_journals": dict(review_journals),
            "aws_published": published, "superseded_skipped": superseded, "errors": errors}


def export_report(settings: Settings, backend: str = "aws") -> str:
    # The published report is the autism-genomics pilot catalog, at `reports/{CORPUS_ID}/`.
    response = search_corpus(settings, backend=backend, limit=10000, journal_policy="all",
                             corpus=CORPUS_ID)
    if response["total"] > response["returned"]:
        raise RuntimeError("report exceeds the pilot limit of 10000 records")
    pointer = settings.state_dir / "corpus" / CORPUS_ID / "latest-run.json"
    manifest = {}
    if pointer.exists():
        run_dir = Path(json.loads(pointer.read_text())["run_dir"])
        manifest = json.loads((run_dir / "manifest.json").read_text())
    dates = sorted(item["publication_date"] for item in response["results"] if item.get("publication_date"))
    public_manifest = {key: manifest[key] for key in (
        "configuration", "queries", "errors", "at_limit_queries", "candidate_count", "by_year", "by_scope"
    ) if key in manifest}
    payload = {"generated_at": utc_now(), "backend": backend, "manifest": public_manifest,
               "catalog_date_range": {"start": dates[0] if dates else None, "end": dates[-1] if dates else None},
               "papers": response["results"]}
    template = (Path(__file__).parent / "corpus_view.html").read_text(encoding="utf-8")
    # JSON inside a script element must not contain a literal closing script tag.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("\u00b7", ",")
    key = f"reports/{CORPUS_ID}/index.html"
    AwsStore(settings).put_text(key, template.replace("__CORPUS_DATA__", data),
                               content_type="text/html; charset=utf-8")
    return f"s3://{settings.aws_bucket}/{key}"
