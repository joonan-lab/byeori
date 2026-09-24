"""The per-paper steps and the batch that runs them over the allowlisted OA corpus.

Steps: ingest (Lambda downloads PDF + GROBID) -> draft (Bedrock evidence note) -> promote
(structural validator accepted it; recorded as review_method ``validator``). A paper's one page is
its note. ``run_pipeline`` walks the corpus, skips work already done according
to DynamoDB, records every step in the cost ledger, and never stops on one paper's failure.
Topic pages are synthesized per corpus tag from the reviewed notes at the end.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import costs
from .aws_store import AwsStore
from .catalog import AwsCatalog as Catalog
from .config import Settings
from .corpus import CORPUS_ID, search_corpus
from .journal_policy import INCLUDE, journal_verdict
from .promote import promote_draft

VALIDATOR_REVIEWER = "aws-lambda-bedrock validator"
DEFAULT_CONCURRENCY = 6
# Worth retrying: Bedrock throttling, and transient failures of the credential-refresh endpoint.
THROTTLE_MARKERS = ("ThrottlingException", "TooManyRequests", "Rate exceeded", "429",
                    "Could not connect to the endpoint URL", "EndpointConnectionError",
                    "Read timeout on endpoint URL", "ConnectTimeoutError")
_thread_local = threading.local()
_shared_session = None
_session_lock = threading.Lock()


def _store_for_thread(settings: Settings) -> AwsStore:
    """One AwsStore per worker thread, all sharing a single boto3 Session.

    Clients are not thread-safe, so each thread keeps its own store (and its own clients). The
    Session is shared on purpose: it owns the process credentials, and resolving them once under
    a lock replaces N threads resolving them at the same time. That contention is what made the
    signin endpoint drop connections mid-run on 2026-09-17, back when the pipeline still ran on
    a short-lived browser session; the project has used a long-lived IAM key since 2026-09-18.
    """
    global _shared_session
    store = getattr(_thread_local, "store", None)
    if store is None:
        with _session_lock:
            if _shared_session is None:
                import boto3
                _shared_session = boto3.Session(region_name=settings.aws_region)
            store = AwsStore(settings, session=_shared_session)
        _thread_local.store = store
    return store
DRAFTABLE = {"fulltext_ready", "model_draft", "draft_failed"}
MAX_TOPIC_NOTES = 20


def require_allowlisted_journal(candidate: dict[str, Any]) -> None:
    """Refuse full-text ingest for any journal outside the test allowlist."""
    verdict = journal_verdict(candidate["record"].get("source"))
    if verdict["verdict"] != INCLUDE:
        raise ValueError(
            f"{candidate['work_id']} is in '{candidate['record'].get('source') or 'unknown journal'}' "
            f"({verdict['verdict']}: {verdict['reason']}); the test corpus ingests allowlisted journals only"
        )


def ingest_step(settings: Settings, store: AwsStore, candidate: dict[str, Any]) -> dict[str, Any]:
    require_allowlisted_journal(candidate)
    store.push_candidate(candidate)
    if settings.aws_bucket:
        store.upload_json(candidate["record"], f"candidates/{candidate['work_id']}.json")
    started = time.monotonic()
    result = store.ingest_work(candidate["work_id"])
    result["ledger"] = costs.record(settings.state_dir, {
        "step": "ingest", "work_id": candidate["work_id"], "seconds": round(time.monotonic() - started, 1),
        "lambda_seconds": result.get("seconds"), "pdf_bytes": result.get("pdf_bytes"),
        "estimated_usd": costs.OPENALEX_FULLTEXT_USD, "basis": "OpenAlex full-text budget estimate",
    })
    return result


def _bedrock_step(settings: Settings, step: str, work_id: str, result: dict[str, Any], started: float) -> dict[str, Any]:
    declined = result.get("filtered_attempt")
    if declined:
        # The note model's declined call is paid for too, so it is a ledger row of its own.
        spent = declined.get("usage") or {}
        costs.record(settings.state_dir, {
            "step": f"{step}_filtered", "work_id": work_id, "status": declined.get("stop_reason"),
            "model_id": declined.get("model_id"), "lambda_seconds": declined.get("seconds"),
            "input_tokens": spent.get("inputTokens"), "output_tokens": spent.get("outputTokens"),
            "estimated_usd": costs.estimate_draft_usd(declined.get("model_id", ""), spent),
            "basis": "Anthropic list prices applied to reported token counts; not reconciled against the AWS bill",
        })
    usage = result.get("usage") or {}
    result["estimated_usd"] = costs.estimate_draft_usd(result.get("model_id", ""), usage)
    result["ledger"] = costs.record(settings.state_dir, {
        "step": step, "work_id": work_id, "status": result.get("status"),
        "model_id": result.get("model_id"), "seconds": round(time.monotonic() - started, 1),
        "lambda_seconds": result.get("seconds"), "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"), "cache_read_tokens": usage.get("cacheReadInputTokens"),
        "cache_write_tokens": usage.get("cacheWriteInputTokens"), "estimated_usd": result["estimated_usd"],
        "basis": "Anthropic list prices applied to reported token counts; not reconciled against the AWS bill",
    })
    return result


def draft_step(settings: Settings, store: AwsStore, candidate: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    require_allowlisted_journal(candidate)
    started = time.monotonic()
    return _bedrock_step(settings, "draft", candidate["work_id"], store.draft_work(candidate["work_id"], model_id), started)


def synthesize_step(settings: Settings, store: AwsStore, slug: str, title: str, work_ids: list[str],
                    model_id: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    result = _bedrock_step(settings, "synthesize", slug, store.synthesize_topic(slug, title, work_ids, model_id), started)
    return result


def select_candidates(settings: Settings, *, limit: int = 0, types: tuple[str, ...] = ("article",)) -> list[dict[str, Any]]:
    """Allowlisted, OpenAlex-hosted, CC-licensed candidates of the wanted types, most cited first."""
    # Scoped to the discovery corpus on purpose: this picks papers to fetch and extract, and an
    # intake whose PDFs are already stored has nothing to fetch.
    found = search_corpus(settings, backend="aws", limit=10000, fulltext_only=True,
                          journal_policy="include", corpus=CORPUS_ID)
    selected = [item for item in found["results"] if item.get("type") in types]
    return selected[:limit] if limit else selected


def tag_slug(tag: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")


def _log(message: str) -> None:
    print(f"{datetime.now(UTC).replace(microsecond=0).isoformat()} {message}", file=sys.stderr, flush=True)


def run_paper(settings: Settings, store: AwsStore, work_id: str, *, model_id: str | None = None) -> dict[str, Any]:
    """Bring one paper to a reviewed evidence note; returns the stages that ran and the final state.

    A paper gets one page, its note (2026-09-18); the reader-facing paper page that used to follow
    was retired with that decision, and its generator with it on 2026-09-23.
    """
    stages: list[str] = []
    item = store.get_item(work_id)
    if item.get("review_status") != "reviewed":
        if item.get("ingest_status") not in DRAFTABLE:
            with Catalog(settings) as catalog:
                candidate = catalog.get_candidate(work_id)
            ingest_step(settings, store, candidate)
            stages.append("ingest")
            item = store.get_item(work_id)
        if item.get("ingest_status") != "model_draft":
            with Catalog(settings) as catalog:
                candidate = catalog.get_candidate(work_id)
            result = draft_step(settings, store, candidate, model_id)
            stages.append("draft")
            if result.get("status") != "model_draft":
                return {"work_id": work_id, "stages": stages, "state": "draft_failed", "problems": result.get("problems")}
        promote_draft(settings, work_id, reviewer=VALIDATOR_REVIEWER, method="validator", store=store)
        stages.append("promote")
    return {"work_id": work_id, "stages": stages, "state": "reviewed"}


def _run_paper_with_retry(settings: Settings, work_id: str, *, model_id: str | None) -> dict[str, Any]:
    store = _store_for_thread(settings)
    for attempt in (1, 2, 3):
        try:
            return run_paper(settings, store, work_id, model_id=model_id)
        except Exception as exc:
            if attempt == 3 or not any(marker in str(exc) for marker in THROTTLE_MARKERS):
                raise
            _log(f"    {work_id} throttled; waiting {30 * attempt}s before retry {attempt + 1}")
            time.sleep(30 * attempt)
    raise AssertionError("unreachable")


def run_pipeline(settings: Settings, *, limit: int = 0, dry_run: bool = False,
                 skip_topics: bool = False, model_id: str | None = None,
                 concurrency: int = DEFAULT_CONCURRENCY) -> dict[str, Any]:
    if not 1 <= concurrency <= 16:
        raise ValueError("concurrency must be 1 to 16")
    store = AwsStore(settings)
    candidates = select_candidates(settings, limit=limit)
    run = {"started_at": datetime.now(UTC).replace(microsecond=0).isoformat(), "selected": len(candidates),
           "dry_run": dry_run, "concurrency": concurrency, "papers": [], "topics": [], "errors": []}
    if dry_run:
        run["papers"] = [{"work_id": c["work_id"], "title": c["title"], "source": c["source"],
                          "cited_by_count": c["cited_by_count"]} for c in candidates]
        return run
    titles = {c["work_id"]: c["title"] for c in candidates}
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_paper_with_retry, settings, c["work_id"], model_id=model_id): c["work_id"]
                   for c in candidates}
        for future in as_completed(futures):
            work_id = futures[future]
            done += 1
            try:
                outcome = future.result()
                run["papers"].append(outcome)
                _log(f"[{done}/{len(candidates)}] {work_id} {outcome['state']} after "
                     f"{', '.join(outcome['stages']) or 'nothing'}: {titles[work_id][:60]}")
            except Exception as exc:  # one paper's failure must not stop the corpus
                run["errors"].append({"work_id": work_id, "error": str(exc)})
                _log(f"[{done}/{len(candidates)}] {work_id} error: {exc}")
    run["papers"].sort(key=lambda p: list(titles).index(p["work_id"]))
    if not skip_topics:
        reviewed = {p["work_id"] for p in run["papers"] if p["state"] in {"page_ready", "reviewed"}}
        by_tag: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            if candidate["work_id"] in reviewed:
                for tag in candidate.get("tags") or []:
                    by_tag.setdefault(tag, []).append(candidate)
        for tag, members in sorted(by_tag.items()):
            slug = tag_slug(tag)
            work_ids = [m["work_id"] for m in members[:MAX_TOPIC_NOTES]]
            _log(f"topic {slug}: {len(work_ids)} of {len(members)} notes")
            try:
                result = synthesize_step(settings, store, slug, f"{tag} in autism genomics", work_ids, model_id)
                run["topics"].append({"topic": slug, "status": result.get("status"), "work_ids": work_ids,
                                      "problems": result.get("problems")})
            except Exception as exc:
                run["errors"].append({"topic": slug, "error": str(exc)})
                _log(f"    error: {exc}")
    run["finished_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    run["ledger"] = costs.summarize(settings.state_dir)
    report = settings.state_dir / f"pipeline-{run['started_at'].replace(':', '')}.json"
    report.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    run["report"] = str(report)
    return run


# ---- llm-wiki-schema pipeline over uploaded PDFs (stem-keyed) --------------------------------

def _stems_with_status(settings: Settings, **wanted: Any) -> list[dict[str, Any]]:
    """Uploaded papers (id_kind stem) matching the given attribute values, sorted by stem."""
    from boto3.dynamodb.conditions import Attr
    import boto3
    table = boto3.Session(region_name=settings.aws_region).resource("dynamodb").Table(settings.aws_table)
    condition = Attr("id_kind").eq("stem")
    for key, value in wanted.items():
        condition &= Attr(key).eq(value) if not isinstance(value, (list, tuple)) else Attr(key).is_in(list(value))
    request: dict[str, Any] = {"FilterExpression": condition,
                               "ProjectionExpression": "work_id, ingest_status, source_note_status, source_note_problems, "
                                                       "page_status, page_problems, category"}
    items: list[dict[str, Any]] = []
    while True:
        page = table.scan(**request)
        items.extend(page.get("Items", []))
        if not page.get("LastEvaluatedKey"):
            break
        request["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    return sorted(items, key=lambda i: i["work_id"])


def failed_stems(settings: Settings) -> list[dict[str, Any]]:
    """Read the AWS-selected failed attempts, without downloading the whole catalogue."""
    store = AwsStore(settings)
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = store.pipeline_failures(verbose=True, offset=offset)
        rows.extend(response["papers"])
        offset = response.get("next_offset")
        if offset is None:
            return rows


def source_note_step(settings: Settings, store: AwsStore, stem: str, model_id: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    result = _bedrock_step(settings, "source_note", stem, store.source_note(stem, model_id), started)
    return result


def run_stem(settings: Settings, stem: str, *, model_id: str | None = None) -> dict[str, Any]:
    store = _store_for_thread(settings)
    stages: list[str] = []
    item = store.get_item(stem)
    if item.get("ingest_status") not in ("fulltext_ready", "model_draft", "draft_failed"):
        return {"stem": stem, "stages": stages, "state": f"waiting:{item.get('ingest_status')}"}
    if item.get("source_note_status") != "source_ready":
        result = source_note_step(settings, store, stem, model_id)
        stages.append("source_note")
        if result.get("status") != "source_ready":
            return {"stem": stem, "stages": stages, "state": "source_failed", "problems": result.get("problems")}
    return {"stem": stem, "stages": stages, "state": "source_ready"}


def run_stem_pipeline(settings: Settings, *, stems: list[str] | None = None, limit: int = 0, dry_run: bool = False,
                      model_id: str | None = None, only_failed: bool = False,
                      concurrency: int = DEFAULT_CONCURRENCY, skip_index: bool = False) -> dict[str, Any]:
    """The evidence note (llm-wiki schema) for every extracted upload that has none.

    Without `model_id` the Lambda's NoteModelId writes the note and NoteFallbackModelId takes over a
    filtered one; a named model is used as given, with no fallback.
    """
    if not 1 <= concurrency <= 16:
        raise ValueError("concurrency must be 1 to 16")
    if stems:
        selected = sorted(set(stems))
    elif only_failed:
        failures = failed_stems(settings)
        run_failures = failures
        selected = [f["stem"] for f in failures]
    else:
        items = _stems_with_status(settings, ingest_status=("fulltext_ready", "model_draft", "draft_failed"))
        selected = [i["work_id"] for i in items
                    if i.get("source_note_status") != "source_ready"]
    if limit:
        selected = selected[:limit]
    run = {"started_at": datetime.now(UTC).replace(microsecond=0).isoformat(), "selected": len(selected),
           "dry_run": dry_run, "only_failed": only_failed, "concurrency": concurrency, "papers": [], "errors": []}
    if only_failed and not stems:
        run["previous_failures"] = run_failures
    if dry_run:
        run["stems"] = selected
        return run
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_retry_throttle, run_stem, settings, stem, model_id=model_id): stem
                   for stem in selected}
        for future in as_completed(futures):
            stem = futures[future]
            done += 1
            try:
                outcome = future.result()
                run["papers"].append(outcome)
                _log(f"[{done}/{len(selected)}] {stem} {outcome['state']} after {', '.join(outcome['stages']) or 'nothing'}")
            except Exception as exc:
                run["errors"].append({"stem": stem, "error": str(exc)})
                _log(f"[{done}/{len(selected)}] {stem} error: {exc}")
    # The index is rebuilt here, in AWS, so nothing outside this command has to be scheduled.
    if any(p["state"] == "source_ready" for p in run["papers"]) and not skip_index:
        try:
            run["index"] = AwsStore(settings).build_wiki_index()
            _log(f"index rebuilt: {run['index']['documents']}, {run['index']['sections']} sections, "
                 f"{run['index']['seconds']}s")
        except Exception as exc:
            run["errors"].append({"step": "build_index", "error": str(exc)})
            _log(f"index rebuild failed: {exc}")
    run["finished_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    run["ledger"] = costs.summarize(settings.state_dir)
    report = settings.state_dir / f"pipeline-stems-{run['started_at'].replace(':', '')}.json"
    report.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    run["report"] = str(report)
    return run


def _retry_throttle(func, settings: Settings, key: str, **kwargs: Any) -> dict[str, Any]:
    for attempt in (1, 2, 3):
        try:
            return func(settings, key, **kwargs)
        except Exception as exc:
            if attempt == 3 or not any(marker in str(exc) for marker in THROTTLE_MARKERS):
                raise
            _log(f"    {key} throttled; waiting {30 * attempt}s before retry {attempt + 1}")
            time.sleep(30 * attempt)
    raise AssertionError("unreachable")
