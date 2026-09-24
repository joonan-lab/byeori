"""Authenticated Function URL gateway for the student question workflow (docs/LAB-QUESTION-WORKFLOW.md, sections 3-4).

One POST carries one action. The gateway parses the payload-format-2.0 event, matches the
caller's IAM identity against the member registry, checks the action's role rule and hands the
body to the module that owns the record: ``lab_jobs`` for intake and queueing, ``lab_offers`` for
the student's decision on a synthesis offer, ``evidence_packet`` for read-only wiki access and
``lab_review`` for the professor's records. Identity is never read from a request body; ``author``,
``member_id`` and ``role`` fields play no part in authentication.

Every response, success or failure, is the ``{"ok": ...}`` envelope the MCP client relies on.
Failures carry a stable ``error`` code and a bounded message; they never carry a traceback, a
principal ARN or id, a secret, or another member's records. The only S3 writes on this path are
the receipts the owning modules write through ``lab_store.ReceiptWriter``; reads never write.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError

from byeori import evidence_packet, lab_collection, lab_jobs, lab_offers, lab_review, lab_usage
from byeori.lab_budget import BudgetExceeded
from byeori.lab_jobs import Member
from byeori.lab_policy import POLICY_REVISION
from byeori.lab_store import ConditionFailed, Put, ReceiptWriter, StoreError, TablePort, keys, new_item, now_iso

__all__ = [
    "ACTIONS", "GatewayDeps", "GatewayError", "INTERNAL_MESSAGE", "MAX_BODY_BYTES", "POLL_AFTER_SECONDS",
    "QUERY_MAX_CHARS", "STATUS_BY_CODE", "failure", "handle", "parse_event", "verify_identity",
]

POLL_AFTER_SECONDS = 5
SEARCH_DEFAULT_LIMIT = 10
SEARCH_MAX_LIMIT = 30
QUERY_MAX_CHARS = 1000
READ_DEFAULT_CHARS = 4000
MAX_BODY_BYTES = 262_144            # a question is at most 8,000 characters and its context 4,000
MESSAGE_MAX_CHARS = 300
CONTENT_TYPE = "application/json"
DOC_TYPES = frozenset({"note", "paper", "overview", "question", "concept"})
STUDENT_AND_ADMIN = frozenset({"student", "admin"})
ADMIN_ONLY = frozenset({"admin"})
INTERNAL_MESSAGE = "The request could not be completed."

# The answer receipt fields a student receives (design section 4: answer, citations, limitations).
ANSWER_FIELDS = ("answer", "citations", "limitations", "evidence_state", "unresolved_items", "hold_reason")
# Job statuses the MCP client keeps polling on; the response names the next poll for them only.
ACTIVE_STATUSES = frozenset({lab_jobs.RECEIVED, lab_jobs.QUEUED, lab_jobs.RUNNING})
TRIAGE_VIEW = {"pending": "triage_pending"}

# StoreError codes the gateway maps to HTTP statuses; any other store code is an internal failure.
STATUS_BY_CODE = {
    "idempotency_conflict": 409, "revision_conflict": 409, "expired": 409, "not_found": 404, "forbidden": 403,
    "invalid_transition": 409, "lease_held": 409, "invalid_outbox": 409, "budget_exceeded": 409, "conflict": 409,
}
MESSAGES = {
    "method_not_allowed": "Only POST is accepted.",
    "invalid_request": "The request is not valid.",
    "unknown_action": "Unknown action.",
    "unauthenticated": "The caller is not authenticated.",
    "wrong_account": "The caller belongs to another AWS account.",
    "forbidden": "The caller may not perform this action.",
    "inactive_member": "The member is not active.",
    "not_found": "No such record for this caller.",
    "idempotency_conflict": "This request_id was already used with a different payload.",
    "revision_conflict": "The revision or hash does not name the current proposal.",
    "expired": "The offer has expired.",
    "invalid_transition": "The record is not in a state that allows this action.",
    "lease_held": "The job is being processed; ask again later.",
    "invalid_outbox": "The delivery record does not belong to this job.",
    "budget_exceeded": "The budget for this period is exhausted.",
    "conflict": "The record changed concurrently; read it again and retry.",
    "internal": INTERNAL_MESSAGE,
}

_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class GatewayError(Exception):
    """A refusal with a stable ``code`` and HTTP ``status``; ``message`` is safe to return."""

    def __init__(self, code: str, status: int, message: str | None = None):
        self.code, self.status = code, int(status)
        self.message = message or MESSAGES.get(code, code)
        super().__init__(self.message)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class GatewayDeps:
    """What one gateway invocation works with; ``lab_lambda`` binds the AWS clients.

    ``index`` is the ``(connection, etag)`` pair from ``evidence_packet.open_index``; it may be
    ``None`` until the first read action opens it through ``index_opener`` and caches it here.
    ``now`` is called once per request. ``queue_sender(url, body)`` hands one message to SQS.
    """

    table: TablePort
    receipts: ReceiptWriter
    s3: Any
    bucket: str
    account_id: str
    answer_queue_url: str
    queue_sender: Callable[[str, dict[str, Any]], Any]
    index: tuple[sqlite3.Connection, str | None] | None = None
    index_opener: Callable[[], tuple[sqlite3.Connection, str | None]] | None = None
    now: Callable[[], datetime] = _utc_now
    policy_revision: str = POLICY_REVISION


# ---------------------------------------------------------------------------------------------
# Event parsing and identity
# ---------------------------------------------------------------------------------------------

def parse_event(event: Any) -> tuple[str, dict[str, Any]]:
    """``(method, body)`` of a Function URL payload-2.0 event; POST with a JSON object body only.

    Any other method is ``method_not_allowed`` (405). A missing, oversized, undecodable or
    non-object body is ``invalid_request`` (400). ``isBase64Encoded`` bodies are decoded first.
    """
    if not isinstance(event, Mapping):
        raise GatewayError("invalid_request", 400, "The event is not a Function URL payload.")
    context = event.get("requestContext")
    http = context.get("http") if isinstance(context, Mapping) else None
    method = http.get("method") if isinstance(http, Mapping) else None
    if not isinstance(method, str) or method.upper() != "POST":
        raise GatewayError("method_not_allowed", 405)
    raw = event.get("body")
    if raw is None:
        raise GatewayError("invalid_request", 400, "A JSON object body is required.")
    if event.get("isBase64Encoded"):
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise GatewayError("invalid_request", 400, "The body is not valid base64.") from exc
    elif isinstance(raw, str):
        data = raw.encode("utf-8")
    elif isinstance(raw, (bytes, bytearray)):
        data = bytes(raw)
    else:
        raise GatewayError("invalid_request", 400, "A JSON object body is required.")
    if len(data) > MAX_BODY_BYTES:
        raise GatewayError("invalid_request", 400, f"The body exceeds {MAX_BODY_BYTES} bytes.")
    try:
        body = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise GatewayError("invalid_request", 400, "The body is not valid JSON.") from exc
    if not isinstance(body, dict):
        raise GatewayError("invalid_request", 400, "A JSON object body is required.")
    return "POST", body


def verify_identity(event: Any, table: TablePort, *, account_id: str) -> Member:
    """The registry member behind ``requestContext.authorizer.iam`` (design section 3).

    The caller must be authenticated by the Function URL (``userId`` and ``userArn`` present),
    belong to the stack's account, and match a registered member on **both** the principal id
    and the ARN, so a re-created IAM user with the old name is refused. An inactive member is
    refused after the match. Nothing in the request body takes part in this check.
    """
    context = event.get("requestContext") if isinstance(event, Mapping) else None
    authorizer = context.get("authorizer") if isinstance(context, Mapping) else None
    iam = authorizer.get("iam") if isinstance(authorizer, Mapping) else None
    if not isinstance(iam, Mapping) or not iam.get("userId") or not iam.get("userArn"):
        raise GatewayError("unauthenticated", 401)
    user_id, user_arn = iam.get("userId"), iam.get("userArn")
    if not isinstance(user_id, str) or not isinstance(user_arn, str):
        raise GatewayError("unauthenticated", 401)
    if iam.get("accountId") != account_id:
        raise GatewayError("wrong_account", 403)
    pointer = table.get(*keys.principal(user_id))
    member_id = pointer.get("member_id") if isinstance(pointer, Mapping) else None
    profile = table.get(*keys.member(member_id)) if isinstance(member_id, str) and member_id else None
    if (not profile or profile.get("principal_arn") != user_arn or profile.get("principal_id") != user_id
            or profile.get("member_id") != member_id or profile.get("role") not in lab_jobs.ROLES):
        raise GatewayError("forbidden", 403)
    if profile.get("active") is not True:
        raise GatewayError("inactive_member", 403)
    return Member(member_id=member_id, role=profile["role"],
                  policy_revision=profile.get("policy_revision") or POLICY_REVISION)


# ---------------------------------------------------------------------------------------------
# Dispatch and the envelope
# ---------------------------------------------------------------------------------------------

def handle(event: Any, *, deps: GatewayDeps) -> dict[str, Any]:
    """Serve one Function URL event: ``{"statusCode", "headers", "body"}`` with the envelope in ``body``.

    Order: method and body (405/400), identity (401/403), action and role rule (400/403), then
    the action itself with ``deps.now()`` read once. Every exception becomes a failure envelope;
    nothing raised here reaches the Lambda runtime.
    """
    try:
        _method, body = parse_event(event)
        member = verify_identity(event, deps.table, account_id=deps.account_id)
        action = body.get("action")
        if not isinstance(action, str) or action not in ACTIONS:
            raise GatewayError("unknown_action", 400)
        roles, handler = ACTIONS[action]
        if member.role not in roles:
            raise GatewayError("forbidden", 403, "This action is limited to administrators.")
        moment = _moment(deps.now())
        result = handler(body, member, deps, moment)
        return _response(200, {"ok": True, "action": action, **result})
    except Exception as exc:  # noqa: BLE001 - every failure must become the envelope
        status, payload = failure(exc)
        return _response(status, payload)


def failure(exc: BaseException) -> tuple[int, dict[str, Any]]:
    """The HTTP status and failure envelope for ``exc``; only validation text is echoed.

    ``GatewayError`` carries its own code. ``lab_budget.BudgetExceeded`` is ``budget_exceeded``
    (409) and is checked before its ``ConditionFailed`` base, which is a plain ``conflict`` (409).
    ``StoreError`` maps by ``code`` through ``STATUS_BY_CODE``; ``ValueError`` is a bad request
    whose message is the validation text. Anything else is ``internal`` (500) with a fixed message.
    """
    if isinstance(exc, GatewayError):
        return exc.status, _failure(exc.code, exc.message)
    if isinstance(exc, BudgetExceeded):
        return 409, _failure("budget_exceeded")
    if isinstance(exc, StoreError):
        status = STATUS_BY_CODE.get(exc.code)
        if status is None:
            return 500, _failure("internal")
        return status, _failure(exc.code)
    if isinstance(exc, ConditionFailed):
        return 409, _failure("conflict")
    if isinstance(exc, ValueError):
        return 400, _failure("invalid_request", str(exc)[:MESSAGE_MAX_CHARS] or MESSAGES["invalid_request"])
    return 500, _failure("internal")


def _failure(code: str, message: str | None = None) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": message or MESSAGES.get(code, code)}


def _response(status: int, payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        status, text = 500, json.dumps(_failure("internal"))
    return {"statusCode": int(status), "headers": {"content-type": CONTENT_TYPE}, "body": text}


# ---------------------------------------------------------------------------------------------
# Student actions
# ---------------------------------------------------------------------------------------------

def ask_byeori(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """Record the question, queue it with its reservation and outbox row, and hand the message to SQS.

    A ``request_id`` the member already used returns that job. Nothing limits how many questions
    a member may start: the user's decision is that the corpus and the agent decide the result,
    not a counter. ``lab_jobs.queue`` runs only for a job still
    ``received``; a ``rejected_budget`` outcome is returned as the job status with no message
    sent. The SQS send is best effort: the outbox row stays ``pending`` for the relay when it
    fails, and ``delivery`` reports ``sent``, ``pending`` or the row's later state.
    """
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not _IDENTIFIER.match(request_id):
        raise ValueError("request_id must be a simple identifier of at most 128 characters")
    job = lab_jobs.intake(deps.table, deps.receipts, member, body, moment, policy_revision=deps.policy_revision)
    if job["status"] == lab_jobs.RECEIVED:
        job = lab_jobs.queue(deps.table, job["job_id"], now=moment)
    delivery = None
    if job["status"] == lab_jobs.QUEUED and job.get("outbox_id"):
        delivery = _deliver(deps, job, moment)
    return {"job_id": job["job_id"], "session_id": job.get("session_id"), "turn": job.get("turn"),
            "status": job["status"], "poll_after_seconds": POLL_AFTER_SECONDS, "delivery": delivery}


def _deliver(deps: GatewayDeps, job: Mapping[str, Any], moment: datetime) -> str | None:
    """Send the job's pending outbox message and mark the row ``sent``; report the row's state.

    A row that already left ``pending`` (the relay or an earlier call sent it) is not re-sent.
    A failed send leaves the row ``pending``: the outbox relay re-sends it, so the job stays
    queued rather than failing the student's request. A concurrent ``mark_sent`` is harmless.
    """
    outbox_id = job["outbox_id"]
    row = deps.table.get(*keys.outbox(outbox_id))
    if row is None:
        return None
    if row.get("status") != lab_jobs.OUTBOX_PENDING:
        return row.get("status")
    message = {"outbox_id": outbox_id, "job_id": job["job_id"], "kind": lab_jobs.KIND_ANSWER}
    try:
        deps.queue_sender(deps.answer_queue_url, message)
    except Exception:  # noqa: BLE001 - the outbox row is the durable record; the relay retries
        return lab_jobs.OUTBOX_PENDING
    try:
        lab_jobs.mark_sent(deps.table, outbox_id, row["revision"], now=moment)
    except (ConditionFailed, StoreError):
        pass
    return lab_jobs.OUTBOX_SENT


def get_byeori_answer(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """The job's status, its answer fields once completed, the triage state, the offer and its research.

    ``lab_jobs.read_job`` admits the owner and administrators and raises ``NotFound`` for anyone
    else. The answer fields come from the answer receipt; the triage state is the job's
    ``triage_status`` with ``pending`` shown as ``triage_pending``. An offer is shown through
    ``lab_offers.offer_view`` and an ``offered`` one past its TTL is shown as ``expired`` without
    recording anything. ``research`` names the execution an accepted offer started.
    """
    job_id = _identifier(body.get("job_id"), "job_id")
    job = lab_jobs.read_job(deps.table, job_id, member)
    status = job["status"]
    view: dict[str, Any] = {
        "job_id": job["job_id"], "kind": job.get("kind"), "session_id": job.get("session_id"), "turn": job.get("turn"),
        "parent_job_id": job.get("parent_job_id"), "question": job.get("question"), "status": status,
        "created_at": job.get("created_at"), "completed_at": job.get("completed_at"),
        "usage": job.get("usage"), "usd_micros": job.get("usd_micros"),
        "triage_status": TRIAGE_VIEW.get(job.get("triage_status"), job.get("triage_status")),
        "poll_after_seconds": POLL_AFTER_SECONDS if status in ACTIVE_STATUSES else None,
    }
    if status in (lab_jobs.COMPLETED, lab_jobs.PARTIAL):
        view.update(_answer_fields(deps.receipts, job))
    elif status == lab_jobs.REJECTED_BUDGET:
        view["reason"] = job.get("reason")
    elif status in (lab_jobs.FAILED, lab_jobs.OUTCOME_UNKNOWN):
        view["error_code"] = job.get("error_code")
    offer = lab_offers.offer_for_job(deps.table, job_id)
    if offer is not None:
        shown = lab_offers.offer_view(offer)
        if shown.get("status") == lab_offers.OFFERED and lab_offers.is_expired(offer, moment):
            shown["status"] = lab_offers.EXPIRED
        view["synthesis_offer"] = shown
        execution_id = offer.get("execution_id")
        if execution_id:
            research = deps.table.get(*keys.job(execution_id))
            view["research"] = {"execution_id": execution_id, "status": research.get("status") if research else None}
    # The wiki fell short on this question: the member may ask for papers to be collected, and the
    # professor sees it either way (user, 2026-09-22). Accepting starts nothing by itself.
    collection = lab_collection.offer_view(lab_collection.gap_for_job(deps.table, job_id))
    if collection is not None:
        view["collection_offer"] = collection
    return view


def _answer_fields(receipts: ReceiptWriter, job: Mapping[str, Any]) -> dict[str, Any]:
    """The student-facing fields of the answer receipt; a missing receipt yields ``None`` values."""
    empty = {name: None for name in ANSWER_FIELDS}
    key = job.get("receipt_key")
    if not isinstance(key, str) or not key:
        return {**empty, "answer_receipt": "unavailable"}
    try:
        record = receipts.get_json(key)
    except (ClientError, ValueError):
        return {**empty, "answer_receipt": "unavailable"}
    if not isinstance(record, Mapping):
        return {**empty, "answer_receipt": "unavailable"}
    return {name: record.get(name) for name in ANSWER_FIELDS}


def respond_to_synthesis_offer(body: Mapping[str, Any], member: Member, deps: GatewayDeps,
                               moment: datetime) -> dict[str, Any]:
    """Record the owner's ``accept`` or ``decline`` through ``lab_offers.respond``."""
    result = lab_offers.respond(deps.table, deps.receipts, member, body, moment)
    return {"offer_id": result["offer_id"], "status": result["status"], "execution_id": result.get("execution_id"),
            "approval_id": result.get("approval_id"), "research_status": result.get("research_status")}


def search_wiki(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """BM25 hits for a cleaned, bounded query; ``index_etag`` names the index version searched."""
    query = _query(body.get("query"))
    limit = _bounded_int(body.get("limit"), "limit", SEARCH_DEFAULT_LIMIT, 1, SEARCH_MAX_LIMIT)
    doc_type = body.get("doc_type")
    if doc_type is not None and (not isinstance(doc_type, str) or doc_type not in DOC_TYPES):
        raise ValueError(f"doc_type must be one of {sorted(DOC_TYPES)}")
    connection, etag = _index(deps)
    try:
        hits = evidence_packet.search(connection, query, limit, doc_type)
    except sqlite3.OperationalError as exc:
        raise GatewayError("invalid_request", 400, "The query could not be searched; simplify it.") from exc
    return {"query": query, "results": hits, "index_etag": etag}


def read_wiki_page(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """A page outline, or one section window of at most 8,000 characters (``evidence_packet.read_excerpt``)."""
    key = evidence_packet.page_key(body.get("key"))
    section = body.get("section")
    if section is not None and not isinstance(section, str):
        raise ValueError("section must name a heading from the outline")
    start = body.get("start")
    max_chars = body.get("max_chars")
    try:
        return evidence_packet.read_excerpt(deps.s3, deps.bucket, key, section=section,
                                            start=0 if start is None else start,
                                            max_chars=READ_DEFAULT_CHARS if max_chars is None else max_chars)
    except FileNotFoundError as exc:
        raise GatewayError("not_found", 404, "No page at that key.") from exc


def wiki_backlinks(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """Pages whose wikilinks point at the key, from the index's links table."""
    key = evidence_packet.page_key(body.get("key"))
    return evidence_packet.backlinks(_index(deps), key)


def read_source(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """The stored full text behind a source note: its outline, or one section window of at most 8,000 characters.

    ``key`` is the ``wiki/sources/`` note. The note's frontmatter names the extraction
    (``source_key`` or the ``papers/{stem}/original.pdf`` path whose text is ``clean.md``); the PDF
    itself is never returned. A note without a stored extraction is ``not_found``.
    """
    note_key = evidence_packet.page_key(body.get("key"))
    if not note_key.startswith("wiki/sources/"):
        raise ValueError("key must be a wiki/sources/ note; read_wiki_page reads other pages")
    section = body.get("section")
    if section is not None and not isinstance(section, str):
        raise ValueError("section must name a heading from the outline")
    start, max_chars = body.get("start"), body.get("max_chars")
    try:
        note = evidence_packet.read_page(deps.s3, deps.bucket, note_key)
    except FileNotFoundError as exc:
        raise GatewayError("not_found", 404, "No page at that key.") from exc
    fields, _ = evidence_packet.split_frontmatter(note.text)
    source_key = evidence_packet.source_key_for_note(fields, deps.bucket)
    if source_key is None:
        raise GatewayError("not_found", 404, "This note records no stored full text.")
    try:
        excerpt = evidence_packet.read_source_excerpt(deps.s3, deps.bucket, source_key, section=section,
                                                      start=0 if start is None else start,
                                                      max_chars=READ_DEFAULT_CHARS if max_chars is None else max_chars)
    except FileNotFoundError as exc:
        raise GatewayError("not_found", 404, "The stored full text for this note is missing.") from exc
    return {**excerpt, "note_key": note_key, "source_key": source_key,
            "note_metadata": {name: fields[name][:1000] for name in evidence_packet.METADATA_FIELDS if name in fields}}


PAPER_REQUEST_REASON_MAX = 1000
PAPER_REQUEST_TITLE_MAX = 300
PAPER_REQUEST_STATUSES = frozenset({"requested", "accepted", "declined", "ingested"})


def normalized_doi(value: Any) -> str | None:
    """``10.xxxx/...`` in lower case without a resolver prefix, or ``None`` for an empty value."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("doi must be a string")
    doi = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    if not doi:
        return None
    if not re.fullmatch(r"10\.\d{4,9}/[^\s]+", doi) or len(doi) > 200:
        raise ValueError("doi must look like 10.xxxx/suffix")
    return doi


def request_paper(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """Record a member's request to add a paper to the wiki; the administrator ingests it later.

    The wiki is checked first: a DOI already indexed returns the notes that carry it and records
    nothing. A DOI already requested returns that request (``duplicate``). Nothing here fetches
    the paper, calls OpenAlex or writes to the wiki.
    """
    doi = normalized_doi(body.get("doi"))
    title = body.get("title")
    if title is not None and (not isinstance(title, str) or not title.strip() or len(title) > PAPER_REQUEST_TITLE_MAX):
        raise ValueError(f"title must be a non-empty string of at most {PAPER_REQUEST_TITLE_MAX} characters")
    reason = body.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > PAPER_REQUEST_REASON_MAX:
        raise ValueError(f"reason must say why the paper is needed, in at most {PAPER_REQUEST_REASON_MAX} characters")
    if doi is None and title is None:
        raise ValueError("give the doi or the title of the paper")
    if doi is not None:
        found = _notes_with_doi(deps, doi)
        if found:
            return {"status": "already_in_wiki", "doi": doi, "notes": found, "request_id": None, "duplicate": False}
        pointer = deps.table.get(*keys.paper_request_doi(doi))
        if pointer is not None:
            existing = deps.table.get(*keys.paper_request(str(pointer.get("request_id"))))
            if existing is not None:
                return {"request_id": existing["request_id"], "status": existing["status"], "doi": doi,
                        "duplicate": True, "requested_by_you": existing.get("member_id") == member.member_id}
    request_id = uuid.uuid4().hex
    stamp = now_iso(moment)
    record = new_item(*keys.paper_request(request_id), stamp, request_id=request_id, member_id=member.member_id,
                      doi=doi, title=title.strip() if title else None, reason=reason.strip(), status="requested",
                      decision_note=None, decided_at=None, decided_by=None)
    operations = [Put(record), Put(new_item(*keys.paper_request_pointer(stamp, request_id), stamp, request_id=request_id,
                                           member_id=member.member_id, doi=doi, status="requested"))]
    if doi is not None:
        operations.append(Put(new_item(*keys.paper_request_doi(doi), stamp, request_id=request_id)))
    deps.table.transact(operations)
    return {"request_id": request_id, "status": "requested", "doi": doi, "duplicate": False}


def _notes_with_doi(deps: GatewayDeps, doi: str) -> list[dict[str, Any]]:
    connection, _etag = _index(deps)
    rows = connection.execute(
        "SELECT doc_type, doc_id, title, doi FROM docs WHERE lower(doi) LIKE ? OR lower(doi) LIKE ? LIMIT 5",
        (f"%{doi}", f"%{doi}/")).fetchall()
    notes = []
    for row in rows:
        notes.append({"key": evidence_packet.hit_key({"doc_type": row[0], "doc_id": row[1]}), "title": row[2]})
    return notes


def usage_report(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """Per-member and lab spend for one month, read from the budget ledger.

    ``period`` is ``YYYY-MM`` and defaults to the month of this request; ``member_id`` limits the
    report to one member. No cap is enforced anywhere: this is accounting, not a limit.
    """
    period = body.get("period")
    if period is None:
        period = lab_usage.period_of(moment)
    elif not isinstance(period, str):
        raise ValueError("period must look like YYYY-MM")
    who = body.get("member_id")
    if who is not None and (not isinstance(who, str) or not _IDENTIFIER.match(who)):
        raise ValueError("member_id must be a simple identifier")
    return lab_usage.monthly_usage(deps.table, period, member_id=who)


def list_paper_requests(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """Administrator view of paper requests, newest first, optionally one status; ``cursor`` continues."""
    status = body.get("status")
    if status is not None and status not in PAPER_REQUEST_STATUSES:
        raise ValueError(f"status must be one of {sorted(PAPER_REQUEST_STATUSES)}")
    cursor = body.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise ValueError("cursor must be the string returned by the previous page")
    pointers, next_cursor = deps.table.query(keys.paper_request_pointer("", "")[0], sk_prefix="REQ#", limit=50,
                                             start_after=cursor, ascending=False)
    requests = []
    for pointer in pointers:
        record = deps.table.get(*keys.paper_request(str(pointer.get("request_id"))))
        if record is None or (status and record.get("status") != status):
            continue
        requests.append({name: record.get(name) for name in ("request_id", "member_id", "doi", "title", "reason", "status",
                                                             "created_at", "decided_at", "decision_note")})
    return {"requests": requests, "cursor": next_cursor}


def decide_paper_request(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """Administrator marks a request ``accepted``, ``declined`` or ``ingested`` with an optional note."""
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not _IDENTIFIER.match(request_id):
        raise ValueError("request_id must be a simple identifier")
    decision = body.get("decision")
    if decision not in {"accepted", "declined", "ingested"}:
        raise ValueError("decision must be accepted, declined or ingested")
    note = body.get("note")
    if note is not None and (not isinstance(note, str) or len(note) > PAPER_REQUEST_REASON_MAX):
        raise ValueError("note must be a string of at most 1000 characters")
    record = deps.table.get(*keys.paper_request(request_id))
    if record is None:
        raise GatewayError("not_found", 404, "No such paper request.")
    stamp = now_iso(moment)
    changes = {"status": decision, "decision_note": note, "decided_at": stamp, "decided_by": member.member_id}
    updated = deps.table.update(*keys.paper_request(request_id), record["revision"], changes)
    pointer_key = keys.paper_request_pointer(record["created_at"], request_id)
    pointer = deps.table.get(*pointer_key)
    if pointer is not None:
        deps.table.update(*pointer_key, pointer["revision"], {"status": decision})
    return {"request_id": request_id, "status": updated["status"], "decided_at": stamp}


# ---------------------------------------------------------------------------------------------
# Administrator actions
# ---------------------------------------------------------------------------------------------

def list_question_records(body: Mapping[str, Any], member: Member, deps: GatewayDeps,
                          moment: datetime) -> dict[str, Any]:
    """Question records of a day window with the professor's filters (``lab_review.list_question_records``)."""
    options = _present(body, "status", "min_probability", "max_probability", "member_id", "include_unscored",
                       "cursor", "limit")
    return lab_review.list_question_records(deps.table, from_day=_first(body, "from", "from_day"),
                                            to_day=_first(body, "to", "to_day"), **options)


def propose_research_from_records(body: Mapping[str, Any], member: Member, deps: GatewayDeps,
                                  moment: datetime) -> dict[str, Any]:
    """Turn selected question jobs into a research candidate; no research runs here."""
    candidate = lab_review.propose_research_from_records(deps.table, deps.receipts, member, body, moment)
    return {"candidate_id": candidate["candidate_id"], "status": candidate.get("status"),
            "proposal_revision": candidate.get("proposal_revision"), "proposal_hash": candidate.get("proposal_hash"),
            "linked_offer_ids": candidate.get("linked_offer_ids"),
            "linked_execution_ids": candidate.get("linked_execution_ids"), "candidate": _public(candidate)}


def list_research_candidates(body: Mapping[str, Any], member: Member, deps: GatewayDeps,
                             moment: datetime) -> dict[str, Any]:
    """Research candidates oldest first, optionally filtered by status."""
    return lab_review.list_research_candidates(deps.table, **_present(body, "status", "cursor", "limit"))


def decide_research_candidate(body: Mapping[str, Any], member: Member, deps: GatewayDeps,
                              moment: datetime) -> dict[str, Any]:
    """Approve or reject exactly the proposal revision and hash the professor reviewed."""
    candidate = lab_review.decide_research_candidate(deps.table, deps.receipts, member, body, moment)
    return {"candidate_id": candidate["candidate_id"], "status": candidate.get("status"),
            "execution_id": candidate.get("execution_id"), "approval_id": candidate.get("approval_id"),
            "candidate": _public(candidate)}


def get_research_job(body: Mapping[str, Any], member: Member, deps: GatewayDeps, moment: datetime) -> dict[str, Any]:
    """A research job with the approval that authorised it; an answer job is not found here."""
    job_id = _identifier(body.get("job_id"), "job_id")
    job = deps.table.get(*keys.job(job_id))
    if job is None or job.get("kind") != lab_jobs.KIND_RESEARCH:
        raise lab_jobs.NotFound(f"no research job {job_id}")
    approval_id = job.get("approval_id")
    approval = deps.table.get(*keys.approval(approval_id)) if isinstance(approval_id, str) and approval_id else None
    return {"job_id": job_id, "execution_id": job_id, "status": job.get("status"), "parent_job_id": job.get("parent_job_id"),
            "member_id": job.get("member_id"), "approval_id": approval_id, "job": _public(job),
            "approval": _public(approval) if approval is not None else None}


def respond_to_collection_offer(body: Mapping[str, Any], member: Member, deps: GatewayDeps,
                                moment: datetime) -> dict[str, Any]:
    """Accept or decline collecting papers for a question the wiki could not answer.

    An accept records a request and nothing more: no search runs, no candidate is saved and no
    money is spent, because collection is the professor's to approve (user, 2026-09-22). Another
    member's gap raises ``NotFound``, never ``Forbidden``, so its existence does not leak.
    """
    job_id = _identifier(body.get("job_id"), "job_id")
    decision = body.get("decision")
    if decision not in lab_collection.DECISIONS:
        raise ValueError(f"decision must be one of {', '.join(sorted(lab_collection.DECISIONS))}")
    record = lab_collection.respond(deps.table, job_id, member.member_id, decision, now=moment)
    return {"job_id": job_id, "status": record["status"], "decision": decision,
            "note": lab_collection.OFFER_NOTE}


def list_collection_gaps(body: Mapping[str, Any], member: Member, deps: GatewayDeps,
                         moment: datetime) -> dict[str, Any]:
    """Every question the wiki could not answer, requested or not."""
    status = body.get("status")
    if status is not None and status not in lab_collection.GAP_STATUSES:
        raise ValueError(f"status must be one of {', '.join(sorted(lab_collection.GAP_STATUSES))}")
    limit = int(body.get("limit") or 50)
    if not 1 <= limit <= 100:
        raise ValueError("limit must be from 1 to 100")
    result = lab_collection.list_gaps(deps.table, status=status, limit=limit,
                                      start_after=body.get("cursor"))
    return {"gaps": [_gap_view(gap) for gap in result["gaps"]], "counted": result["counted"],
            "next_cursor": result["next_cursor"], "status": status}


def decide_collection_gap(body: Mapping[str, Any], member: Member, deps: GatewayDeps,
                          moment: datetime) -> dict[str, Any]:
    """The professor approves or rejects collecting for one question."""
    job_id = _identifier(body.get("job_id"), "job_id")
    decision = body.get("decision")
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if decision == "approve":
        record = lab_collection.approve(deps.table, job_id, member.member_id,
                                        query=body.get("query"), now=moment)
    else:
        record = lab_collection.reject(deps.table, job_id, member.member_id,
                                       reason=body.get("reason"), now=moment)
    return {"job_id": job_id, "status": record["status"], "query": record.get("query")}


def _gap_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """What an administrator sees. The member id is kept: the professor may ask who was blocked."""
    return {name: record.get(name) for name in
            ("job_id", "member_id", "question", "query", "reason", "status", "evidence_state",
             "created_at", "decision", "decided_at", "approved_by", "candidates_saved")}


# Routing table: action name -> (roles allowed, handler). Ownership checks live in the owning modules.
ACTIONS: dict[str, tuple[frozenset[str], Callable[..., dict[str, Any]]]] = {
    "ask_byeori": (STUDENT_AND_ADMIN, ask_byeori),
    "get_byeori_answer": (STUDENT_AND_ADMIN, get_byeori_answer),
    "respond_to_synthesis_offer": (STUDENT_AND_ADMIN, respond_to_synthesis_offer),
    "respond_to_collection_offer": (STUDENT_AND_ADMIN, respond_to_collection_offer),
    "search_wiki": (STUDENT_AND_ADMIN, search_wiki),
    "read_wiki_page": (STUDENT_AND_ADMIN, read_wiki_page),
    "wiki_backlinks": (STUDENT_AND_ADMIN, wiki_backlinks),
    "read_source": (STUDENT_AND_ADMIN, read_source),
    "request_paper": (STUDENT_AND_ADMIN, request_paper),
    "list_paper_requests": (ADMIN_ONLY, list_paper_requests),
    "usage_report": (ADMIN_ONLY, usage_report),
    "decide_paper_request": (ADMIN_ONLY, decide_paper_request),
    "list_question_records": (ADMIN_ONLY, list_question_records),
    "propose_research_from_records": (ADMIN_ONLY, propose_research_from_records),
    "list_research_candidates": (ADMIN_ONLY, list_research_candidates),
    "decide_research_candidate": (ADMIN_ONLY, decide_research_candidate),
    "get_research_job": (ADMIN_ONLY, get_research_job),
    "list_collection_gaps": (ADMIN_ONLY, list_collection_gaps),
    "decide_collection_gap": (ADMIN_ONLY, decide_collection_gap),
}


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def _index(deps: GatewayDeps) -> tuple[sqlite3.Connection, str | None]:
    """The cached ``(connection, etag)`` pair, opened through ``index_opener`` on first use."""
    if deps.index is None:
        if deps.index_opener is None:
            raise RuntimeError("the gateway has no index and no index_opener")
        deps.index = deps.index_opener()
    connection, etag = deps.index
    if not isinstance(connection, sqlite3.Connection):
        raise RuntimeError("index must be the (connection, etag) pair from evidence_packet.open_index")
    return connection, etag


def _moment(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RuntimeError("deps.now must return a timezone-aware datetime")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ValueError(f"{name} must be a simple identifier of at most 128 characters")
    return value


def _query(value: Any) -> str:
    """The search text with NUL and other control characters removed, whitespace collapsed, bounded."""
    if not isinstance(value, str):
        raise ValueError("query must be a string")
    cleaned = " ".join(_CONTROL.sub("", value).split())[:QUERY_MAX_CHARS].strip()
    if not cleaned:
        raise ValueError("query must not be empty")
    return cleaned


def _bounded_int(value: Any, name: str, default: int, low: int, high: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return value


def _first(body: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if body.get(name) is not None:
            return body[name]
    return None


def _present(body: Mapping[str, Any], *names: str) -> dict[str, Any]:
    """The named body fields that are present and not null, so the callee applies its own defaults."""
    return {name: body[name] for name in names if body.get(name) is not None}


def _public(record: Mapping[str, Any]) -> dict[str, Any]:
    """A stored record without its table keys."""
    return {name: value for name, value in record.items() if name not in {"pk", "sk"}}
