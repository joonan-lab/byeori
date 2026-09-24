"""Asynchronous Jev triage of one finished answer job (docs/LAB-QUESTION-WORKFLOW.md, P3).

The triage worker runs after the answer was delivered, so nothing here can delay or fail the
answer. For a ``completed`` or ``partial`` answer job it re-reads the job and its ``answer.json``
and ``evidence.json`` receipts, decides whether the exchange may leave AWS at all, reuses an
identical earlier verdict when one exists, otherwise sends the question, its context, the answer,
the cited excerpts, the existing synthesis passages, the limitations and the maintenance hint to
Jev exactly once, and records the three-state verdict with its raw probabilities.

Server-side rules decide before any HTTP call: a job flagged ``private_material``, an empty or
held answer, evidence the packet itself marked ``insufficient``, ``links_only`` or
``truncated_decisive``, and an answer whose own ``evidence_state`` is ``insufficient`` are recorded
as ``skipped`` with ``probabilities`` ``null`` and a reason. A failed call is recorded as
``unavailable`` with its error code and is never retried. A valid verdict stores all three
probabilities, the confidence, the 0.99 cutoff and whether the raw ``review_candidate`` value
passed it. Only a passing verdict with a concrete maintenance hint is a ``review_candidate``; it
then gets a wiki-scope check (concept, overview and explicit question searches with the index
version, queries and pages recorded) and an offer through ``lab_offers.issue``:
``supplement_existing`` when validated targets exist, ``new_synthesis`` otherwise. When the same
session already holds an equivalent offer the student declined or has not answered, ``issue``
returns ``None`` and the verdict records ``offer_suppressed`` naming that offer instead, so the
candidate stays visible to the professor without a repeated prompt (design section 6).
Everything else is ``unconfirmed_candidate``, ``answer_only`` or ``needs_lookup``.

Records: ``JOB#{job_id}/TRIAGE`` (the verdict), ``VERDICT#{input_hash}/META`` (reuse pointer for a
fresh verdict) and the job's ``triage_status`` through ``lab_jobs.set_triage_status``. The only S3
write is ``runs/lab-questions/{job_id}/triage.json`` through ``lab_store.ReceiptWriter``, written
once before the verdict transaction; a redelivery finds the verdict or the receipt and finishes
without a second Jev call. The Jev secret is read through the caller's ``secret_reader``, handed
to ``jev_post`` and never stored, logged or placed in a reason.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError

from byeori import jev_client, lab_jobs, lab_offers
from byeori.evidence_packet import doc_identity, page_key, search
from byeori.jev_client import JevError
from byeori.lab_jobs import TRANSACTION_ATTEMPTS, InvalidTransition, NotFound
from byeori.lab_policy import (
    JEV_MODEL,
    PACKET_LIMITS,
    POLICY_REVISION,
    REVIEW_CANDIDATE_CUTOFF,
    SCOPE_MATCH_SCORE,
    passes_cutoff,
)
from byeori.lab_store import (
    ConditionFailed,
    Operation,
    Put,
    ReceiptWriter,
    TablePort,
    Update,
    digest,
    digest_bytes,
    keys,
    new_item,
    now_iso,
    receipt_key,
)

__all__ = [
    "CANDIDATE_STATUSES", "CONCRETE_HINT_KINDS", "SCOPE_DOC_TYPES", "SERVER_NEEDS_LOOKUP_STATES", "TriagePolicy",
    "VERDICT_STATUSES", "candidate_status", "input_hash", "scope_check", "triage_job",
]

COMPLETE, UNAVAILABLE, SKIPPED, REUSED = "complete", "unavailable", "skipped", "reused"
VERDICT_STATUSES = frozenset({COMPLETE, UNAVAILABLE, SKIPPED, REUSED})
# Job triage_status recorded for each verdict status (lab_jobs.TRIAGE_STATUSES).
_TRIAGE_STATUS_OF = {COMPLETE: "complete", REUSED: "complete", UNAVAILABLE: "unavailable", SKIPPED: "skipped"}

ANSWER_ONLY, NEEDS_LOOKUP, REVIEW_CANDIDATE, UNCONFIRMED = ("answer_only", "needs_lookup", "review_candidate",
                                                           "unconfirmed_candidate")
CANDIDATE_STATUSES = frozenset({ANSWER_ONLY, NEEDS_LOOKUP, REVIEW_CANDIDATE, UNCONFIRMED})

# Packet states the server treats as needs_lookup without asking Jev (design section 6).
SERVER_NEEDS_LOOKUP_STATES = frozenset({"insufficient", "links_only", "truncated_decisive"})
TRIAGED_JOB_STATUSES = frozenset({lab_jobs.COMPLETED, lab_jobs.PARTIAL})

# A maintenance hint is concrete when it names one of these kinds and carries a note.
CONCRETE_HINT_KINDS = frozenset({"supplement_existing", "new_synthesis", "correction"})
HINT_KINDS = frozenset({"none", *CONCRETE_HINT_KINDS})
NO_HINT = {"kind": "none", "target_keys": [], "note": ""}

# Scope check: synthesis layers whose hits may become supplement targets, plus explicit questions.
SYNTHESIS_DOC_TYPES = ("concept", "overview")
SCOPE_DOC_TYPES = (*SYNTHESIS_DOC_TYPES, "question")
SCOPE_QUERY_MAX_CHARS = 1000

REASON_PRIVATE_MATERIAL = "private_material"
REASON_EMPTY_ANSWER = "empty_answer"
REASON_HELD_ANSWER = "held_answer"
REASON_ANSWER_INSUFFICIENT = "answer_insufficient"   # the model declared its evidence insufficient
REASON_ANSWER_MISSING = "answer_receipt_missing"
REASON_INPUT_TOO_LARGE = "input_too_large"
REASON_EVIDENCE_PREFIX = "evidence_"       # followed by the packet's evidence_state
REASON_JEV_PREFIX = "jev_error:"           # followed by the JevError code only

MAX_TEXT_CHARS = 20_000
MAX_ITEMS = 50
MAX_ITEM_CHARS = 2_000
MISSING_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
_STORE_FIELDS = frozenset({"pk", "sk", "revision", "updated_at", "receipt_key", "receipt_sha256"})
_META_FIELDS = ("input_hash", "choice", "probabilities", "confidence", "cutoff", "passed_cutoff", "model",
                "policy_revision", "usage", "usd_micros")


@dataclass(frozen=True)
class TriagePolicy:
    """Adjustable triage settings; the cutoff and the Jev model are fixed by ``lab_policy``.

    ``policy_revision`` is recorded on every verdict and offer; ``scope_match_score`` is the BM25
    score at or above which an existing concept or overview counts as the scope of a supplement;
    ``scope_search_limit`` bounds each of the three scope searches.
    """

    policy_revision: str = POLICY_REVISION
    scope_match_score: float = SCOPE_MATCH_SCORE
    scope_search_limit: int = PACKET_LIMITS.candidates

    def __post_init__(self) -> None:
        if not isinstance(self.policy_revision, str) or not self.policy_revision:
            raise ValueError("policy_revision must be a non-empty string")
        if isinstance(self.scope_match_score, bool) or not isinstance(self.scope_match_score, (int, float)):
            raise ValueError("scope_match_score must be a number")
        if type(self.scope_search_limit) is not int or not 1 <= self.scope_search_limit <= 100:
            raise ValueError("scope_search_limit must be an integer from 1 to 100")


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def triage_job(job: Mapping[str, Any], *, table: TablePort, receipts: ReceiptWriter, s3: Any, bucket: str, index: Any,
               jev_post: Callable[[bytes, str], bytes], secret_reader: Callable[[], str], now: datetime | None,
               policy: TriagePolicy | None = None) -> dict[str, Any]:
    """Triage one delivered answer job and return its stored ``JOB#{job_id}/TRIAGE`` verdict.

    ``job`` needs ``job_id``; the record is re-read from the table and must be an answer job in
    status ``completed`` or ``partial`` (``InvalidTransition`` otherwise, ``NotFound`` when it does
    not exist). ``index`` is the ``(connection, etag)`` pair from ``evidence_packet.open_index``.
    ``jev_post(payload, secret) -> raw`` and ``secret_reader() -> secret`` are the only paths to
    the provider and the secret; a ``JevError`` from either records ``unavailable`` and is not
    retried. A redelivery returns the existing verdict and finishes any step left undone
    (offer, ``triage_status``) without another Jev call. ``s3`` and ``bucket`` are accepted for the
    worker's uniform signature; every read and write goes through ``receipts``.
    """
    settings = policy or TriagePolicy()
    if not isinstance(settings, TriagePolicy):
        raise ValueError("policy must be a TriagePolicy")
    moment = _moment(now)
    job_id = _job_id_of(job)
    stored = _triageable_job(table, job_id)
    verdict = lab_jobs.read_verdict(table, job_id)
    if verdict is None:
        verdict = _record_verdict(stored, table=table, receipts=receipts, index=index, jev_post=jev_post,
                                  secret_reader=secret_reader, moment=moment, settings=settings)
    return _finalise(table, receipts, stored, verdict, moment, settings)


# ---------------------------------------------------------------------------------------------
# Deciding and recording
# ---------------------------------------------------------------------------------------------

def _record_verdict(stored: dict[str, Any], *, table: TablePort, receipts: ReceiptWriter, index: Any, jev_post: Any,
                    secret_reader: Any, moment: datetime, settings: TriagePolicy) -> dict[str, Any]:
    """Judge the job (or recover a receipt an earlier attempt left), then store the verdict once."""
    job_id = stored["job_id"]
    stamp = now_iso(moment)
    triage_key = receipt_key(job_id, "triage.json")
    body = _read_receipt(receipts, triage_key)
    if body is None:
        body = _judge(stored, table=table, receipts=receipts, index=index, jev_post=jev_post,
                      secret_reader=secret_reader, stamp=stamp, settings=settings)
        if not _write_once(receipts, triage_key, body):
            body = receipts.get_json(triage_key)  # a concurrent worker wrote it first; its verdict stands
    record = new_item(*keys.verdict(job_id), stamp, **body, receipt_key=triage_key, receipt_sha256=digest(body))
    operations: list[Operation] = [Put(record)]
    if body["status"] == COMPLETE:
        operations.append(Put(new_item(*keys.verdict_by_input(body["input_hash"]), stamp, job_id=job_id,
                                       **{name: body[name] for name in _META_FIELDS})))
    try:
        table.transact(operations)
    except ConditionFailed:
        current = lab_jobs.read_verdict(table, job_id)
        if current is not None:
            return current  # another worker recorded this job's verdict first
        table.transact([Put(record)])  # the reuse pointer was claimed by another job; this verdict still stands
    return lab_jobs.read_verdict(table, job_id)  # type: ignore[return-value]


def _judge(stored: dict[str, Any], *, table: TablePort, receipts: ReceiptWriter, index: Any, jev_post: Any,
           secret_reader: Any, stamp: str, settings: TriagePolicy) -> dict[str, Any]:
    """The verdict body for the receipt and the ``TRIAGE`` record; calls Jev at most once."""
    job_id = stored["job_id"]
    answer_key = stored.get("receipt_key")
    answer = _read_receipt(receipts, answer_key) if isinstance(answer_key, str) else None
    evidence_key = stored.get("evidence_key") or (answer or {}).get("evidence_key")
    packet = _read_receipt(receipts, evidence_key) if isinstance(evidence_key, str) else None
    answer = answer if isinstance(answer, Mapping) else None
    packet = packet if isinstance(packet, Mapping) else None

    question = stored["question"]
    scope_question = stored.get("standalone_question") or question
    answer_text = _bounded_text((answer or {}).get("answer"))
    hint = _hint((answer or {}).get("maintenance_hint"))
    evidence_state = _evidence_state(packet, answer)
    evidence_sha256 = digest(packet) if packet is not None else (answer or {}).get("evidence_sha256")
    answer_sha256 = digest_bytes(answer_text.encode("utf-8"))
    context_hash = stored.get("context_hash") or (answer or {}).get("context_hash") or ""
    hashed = input_hash(question, context_hash, answer_sha256, evidence_sha256, settings.policy_revision, JEV_MODEL)
    body: dict[str, Any] = {
        "job_id": job_id, "member_id": stored.get("member_id"), "question_hash": stored.get("question_hash"),
        "answer_key": answer_key, "evidence_key": evidence_key, "answer_sha256": answer_sha256,
        "evidence_sha256": evidence_sha256, "evidence_state": evidence_state, "maintenance_hint": hint,
        "input_hash": hashed, "model": JEV_MODEL, "policy_revision": settings.policy_revision,
        "cutoff": REVIEW_CANDIDATE_CUTOFF, "triaged_at": stamp, "jev_called": False, "payload_bytes": None,
        "dropped": [], "scope_check": None, "offer": None,
    }

    if stored.get("private_material") is True:
        return _skipped(body, REASON_PRIVATE_MATERIAL, NEEDS_LOOKUP)
    if answer is None:
        return _skipped(body, REASON_ANSWER_MISSING, NEEDS_LOOKUP)
    if not answer_text:
        return _skipped(body, REASON_EMPTY_ANSWER, ANSWER_ONLY)
    if answer.get("hold_reason"):
        return _skipped(body, REASON_HELD_ANSWER, ANSWER_ONLY)
    if evidence_state != "sufficient":
        return _skipped(body, f"{REASON_EVIDENCE_PREFIX}{evidence_state or 'missing'}", NEEDS_LOOKUP)
    if answer.get("evidence_state") == "insufficient":
        # The packet had enough text, but the model said it could not answer from it; Jev is not
        # asked to turn a declared gap into a synthesis candidate (design section 5).
        return _skipped(body, REASON_ANSWER_INSUFFICIENT, NEEDS_LOOKUP)

    earlier = table.get(*keys.verdict_by_input(hashed))
    if earlier is not None and earlier.get("job_id") != job_id and isinstance(earlier.get("probabilities"), Mapping):
        return _reused(body, earlier, scope_question, hint, index, settings)

    excerpts, passages = _excerpts(packet)
    payload = jev_client.build_payload(question, _context(answer.get("context")), answer_text, excerpts,
                                       passages, _strings(answer.get("limitations")), dict(hint))
    if payload is None:
        return _skipped(body, REASON_INPUT_TOO_LARGE, NEEDS_LOOKUP)
    body["payload_bytes"] = len(payload)
    body["dropped"] = jev_client.payload_state(payload).get("dropped", [])

    # The secret and the provider body live only in these locals; nothing below stores either.
    body["jev_called"] = True
    try:
        result = jev_client.validate(jev_post(payload, secret_reader()))
    except JevError as exc:
        return _unavailable(body, exc.code)
    return _complete(body, result, scope_question, hint, index, settings)


def _skipped(body: dict[str, Any], reason: str, status: str) -> dict[str, Any]:
    return {**body, "status": SKIPPED, "choice": None, "probabilities": None, "confidence": None,
            "passed_cutoff": False, "usage": None, "usd_micros": 0, "error_code": None, "reason": reason,
            "reused_from_job_id": None, "candidate_status": status}


def _unavailable(body: dict[str, Any], code: str) -> dict[str, Any]:
    return {**body, "status": UNAVAILABLE, "choice": None, "probabilities": None, "confidence": None,
            "passed_cutoff": False, "usage": None, "usd_micros": 0, "error_code": code,
            "reason": f"{REASON_JEV_PREFIX}{code}", "reused_from_job_id": None, "candidate_status": NEEDS_LOOKUP}


def _complete(body: dict[str, Any], result: Mapping[str, Any], question: str, hint: Mapping[str, Any], index: Any,
              settings: TriagePolicy) -> dict[str, Any]:
    probabilities = dict(result["probabilities"])
    passed = passes_cutoff(probabilities)
    verdict = {**body, "status": COMPLETE, "choice": result["choice"], "probabilities": probabilities,
               "confidence": result["confidence"], "passed_cutoff": passed, "usage": dict(result["usage"]),
               "usd_micros": int(result["estimated_usd_micros"]), "error_code": None, "reason": None,
               "reused_from_job_id": None, "candidate_status": candidate_status(result["choice"], passed, hint)}
    return _with_scope(verdict, question, hint, index, settings)


def _reused(body: dict[str, Any], earlier: Mapping[str, Any], question: str, hint: Mapping[str, Any], index: Any,
            settings: TriagePolicy) -> dict[str, Any]:
    """An identical input already has a verdict: copy it, pay nothing, run this job's own scope check."""
    probabilities = dict(earlier["probabilities"])
    passed = passes_cutoff(probabilities)
    verdict = {**body, "status": REUSED, "choice": earlier.get("choice"), "probabilities": probabilities,
               "confidence": earlier.get("confidence"), "passed_cutoff": passed, "usage": None, "usd_micros": 0,
               "error_code": None, "reason": None, "reused_from_job_id": earlier.get("job_id"),
               "model": earlier.get("model", body["model"]),
               "candidate_status": candidate_status(earlier.get("choice"), passed, hint)}
    return _with_scope(verdict, question, hint, index, settings)


def _with_scope(verdict: dict[str, Any], question: str, hint: Mapping[str, Any], index: Any,
                settings: TriagePolicy) -> dict[str, Any]:
    """A review candidate gets the wiki-scope check and the offer proposal it implies."""
    if verdict["candidate_status"] != REVIEW_CANDIDATE:
        return verdict
    check = scope_check(index, question, hint, settings)
    return {**verdict, "scope_check": check, "offer": {"kind": check["kind"], "targets": list(check["targets"])}}


# ---------------------------------------------------------------------------------------------
# Finishing: offer and job status
# ---------------------------------------------------------------------------------------------

def _finalise(table: TablePort, receipts: ReceiptWriter, stored: dict[str, Any], verdict: dict[str, Any],
              moment: datetime, settings: TriagePolicy) -> dict[str, Any]:
    """Issue the offer a review candidate earned and record the job's ``triage_status``; idempotent.

    ``lab_offers.issue`` returns ``None`` when the same session already holds an equivalent offer
    (declined or unanswered); the stored verdict then takes ``offer`` ``None`` and
    ``offer_suppressed`` naming that offer, so a redelivery issues nothing and the professor's
    record still shows the candidate.
    """
    job_id = stored["job_id"]
    offer_id = None
    proposal = verdict.get("offer")
    if verdict.get("candidate_status") == REVIEW_CANDIDATE and isinstance(proposal, Mapping):
        kind, targets = proposal["kind"], list(proposal.get("targets") or [])
        offer = lab_offers.issue(table, receipts, stored, verdict, kind, targets, verdict.get("scope_check") or {}, moment,
                                 policy_revision=settings.policy_revision)
        if offer is None:
            verdict = _record_suppression(table, stored, verdict, kind, targets, moment)
        else:
            offer_id = offer["offer_id"]
    status = _TRIAGE_STATUS_OF[verdict["status"]]
    last_error: ConditionFailed | None = None
    for _attempt in range(TRANSACTION_ATTEMPTS):
        job = _triageable_job(table, job_id)
        outbox_id = job.get("triage_outbox_id")
        outbox = table.get(*keys.outbox(outbox_id)) if isinstance(outbox_id, str) and outbox_id else None
        outbox_done = outbox is None or outbox.get("status") == lab_jobs.OUTBOX_DONE
        if job.get("triage_status") == status and outbox_done and (offer_id is None or job.get("offer_id") == offer_id):
            return verdict
        extra = {"offer_id": offer_id} if offer_id and job.get("offer_id") != offer_id else {}
        try:
            lab_jobs.set_triage_status(table, job_id, job["revision"], status, outbox_id=None if outbox_done else outbox_id,
                                       now=moment, **extra)
        except ConditionFailed as exc:
            last_error = exc
            continue
        return verdict
    raise last_error  # type: ignore[misc]


def _record_suppression(table: TablePort, stored: dict[str, Any], verdict: dict[str, Any], kind: str,
                        targets: list[str], moment: datetime) -> dict[str, Any]:
    """Replace the verdict's offer proposal with ``offer_suppressed`` naming the session's earlier offer.

    The ``TRIAGE`` record is updated at its revision; the receipt already written stands as the
    proposal the check produced. A concurrent worker that recorded the same suppression first is
    fine. Returns the stored verdict.
    """
    earlier = lab_offers.suppressing_offer(table, stored.get("session_id"), kind, targets, verdict.get("input_hash"),
                                           now=moment)
    suppressed = {"duplicate_of": earlier["offer_id"] if isinstance(earlier, Mapping) else None}
    changes = {"offer": None, "offer_suppressed": suppressed}
    last_error: ConditionFailed | None = None
    for _attempt in range(TRANSACTION_ATTEMPTS):
        current = lab_jobs.read_verdict(table, stored["job_id"]) or verdict
        if current.get("offer") is None and current.get("offer_suppressed") == suppressed:
            return current
        try:
            table.transact([Update(*keys.verdict(stored["job_id"]), current["revision"], changes)])
        except ConditionFailed as exc:
            last_error = exc
            continue
        return lab_jobs.read_verdict(table, stored["job_id"])  # type: ignore[return-value]
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------------------------

def input_hash(question: str, context_hash: str, answer_sha256: str, evidence_sha256: str | None, policy_revision: str,
               model: str) -> str:
    """The reuse key: same question, context, answer text, evidence version, policy and Jev model."""
    return digest({"question": question, "context_hash": context_hash, "answer_sha256": answer_sha256,
                   "evidence_sha256": evidence_sha256, "policy_revision": policy_revision, "model": model})


def candidate_status(choice: Any, passed_cutoff: bool, hint: Mapping[str, Any]) -> str:
    """``answer_only``/``needs_lookup`` as chosen; ``review_candidate`` only past the cutoff with a concrete hint."""
    if choice in (ANSWER_ONLY, NEEDS_LOOKUP):
        return str(choice)
    if choice == REVIEW_CANDIDATE and passed_cutoff is True and _concrete(hint):
        return REVIEW_CANDIDATE
    return UNCONFIRMED


def _concrete(hint: Mapping[str, Any]) -> bool:
    note = hint.get("note")
    return hint.get("kind") in CONCRETE_HINT_KINDS and isinstance(note, str) and bool(note.strip())


def scope_check(index: Any, question: str, hint: Mapping[str, Any], settings: TriagePolicy | None = None) -> dict[str, Any]:
    """Search concepts, overviews and explicit questions for the question's existing wiki scope.

    Records ``index_etag``, the three ``queries`` with their hits, ``pages_checked`` and the
    supplement ``targets``: the hint's ``target_keys`` that are published page keys present in the
    index ``docs`` table (rejected ones are listed with a reason) plus concept or overview hits
    scoring at least ``scope_match_score``. ``kind`` is ``supplement_existing`` when any target
    survived, ``new_synthesis`` otherwise. Nothing is written.
    """
    settings = settings or TriagePolicy()
    connection, etag = _connection(index)
    query = _scope_query(question)
    queries: list[dict[str, Any]] = []
    pages_checked: list[str] = []
    matched: list[dict[str, Any]] = []
    for doc_type in SCOPE_DOC_TYPES:
        hits, error = _search(connection, query, settings.scope_search_limit, doc_type)
        entry: dict[str, Any] = {"doc_type": doc_type, "query": query,
                                 "hits": [{"key": h["key"], "title": str(h.get("title") or "")[:MAX_ITEM_CHARS],
                                           "score": h["score"]} for h in hits]}
        if error:
            entry["error"] = error
        queries.append(entry)
        for hit in hits:
            if hit["key"] not in pages_checked:
                pages_checked.append(hit["key"])
            if doc_type in SYNTHESIS_DOC_TYPES and hit["score"] >= settings.scope_match_score and _published(hit["key"]):
                matched.append({"key": hit["key"], "doc_type": doc_type, "score": hit["score"]})
    targets: list[str] = []
    rejected: list[dict[str, str]] = []
    for raw in list(hint.get("target_keys") or [])[:MAX_ITEMS]:
        try:
            key = page_key(raw)
        except ValueError:
            rejected.append({"key": str(raw)[:MAX_ITEM_CHARS], "reason": "invalid_key"})
            continue
        if not _indexed(connection, key):
            rejected.append({"key": key, "reason": "not_indexed"})
            continue
        if key not in targets:
            targets.append(key)
    for hit in matched:
        if hit["key"] not in targets:
            targets.append(hit["key"])
    targets = targets[:lab_offers.MAX_TARGETS]
    return {"index_etag": etag, "question": query, "queries": queries, "pages_checked": pages_checked,
            "matched": matched, "targets": targets, "rejected_targets": rejected,
            "kind": lab_offers.SUPPLEMENT_EXISTING if targets else lab_offers.NEW_SYNTHESIS,
            "scope_match_score": settings.scope_match_score, "hint_kind": hint.get("kind")}


def _search(connection: sqlite3.Connection, query: str, limit: int, doc_type: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return search(connection, query, limit, doc_type), None
    except (ValueError, sqlite3.OperationalError) as exc:
        return [], type(exc).__name__  # a page or question cannot abort the check; the failure is recorded


def _indexed(connection: sqlite3.Connection, key: str) -> bool:
    try:
        row = connection.execute("SELECT 1 FROM docs WHERE s3_key = ? LIMIT 1", (key,)).fetchone()
    except sqlite3.OperationalError:
        doc_type, doc_id = doc_identity(key)  # an index built before docs.s3_key existed
        row = connection.execute("SELECT 1 FROM docs WHERE doc_type = ? AND doc_id = ? LIMIT 1", (doc_type, doc_id)).fetchone()
    return row is not None


def _published(key: Any) -> bool:
    try:
        page_key(key)
    except ValueError:
        return False
    return True


def _scope_query(question: str) -> str:
    text = "".join(ch for ch in str(question) if ch != "\x00" and (ch.isprintable() or ch.isspace())).strip()
    return text[:SCOPE_QUERY_MAX_CHARS] or "-"


def _connection(index: Any) -> tuple[sqlite3.Connection, str | None]:
    if isinstance(index, sqlite3.Connection):
        return index, None
    try:
        connection, etag = index
    except (TypeError, ValueError):
        raise TypeError("index must be a sqlite3.Connection or the (connection, etag) pair from open_index") from None
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("index must be a sqlite3.Connection or the (connection, etag) pair from open_index")
    return connection, None if etag is None else str(etag)


# ---------------------------------------------------------------------------------------------
# Inputs from the receipts
# ---------------------------------------------------------------------------------------------

def _evidence_state(packet: Mapping[str, Any] | None, answer: Mapping[str, Any] | None) -> str | None:
    state = packet.get("evidence_state") if packet is not None else None
    if not isinstance(state, str) and answer is not None:
        state = answer.get("packet_evidence_state")
    return state if isinstance(state, str) else None


def _excerpts(packet: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Packet sections as Jev inputs: synthesis pages are existing passages, the rest evidence excerpts."""
    excerpts: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    documents = packet.get("documents") if packet is not None else None
    for document in documents if isinstance(documents, list) else []:
        if not isinstance(document, Mapping) or not isinstance(document.get("key"), str):
            continue
        for section in document.get("sections") if isinstance(document.get("sections"), list) else []:
            if not isinstance(section, Mapping) or not isinstance(section.get("text"), str) or not section["text"]:
                continue
            item = {"key": document["key"], "title": str(document.get("title") or "")[:MAX_ITEM_CHARS],
                    "doc_type": document.get("doc_type"), "section": str(section.get("name") or "")[:MAX_ITEM_CHARS],
                    "text": section["text"], "truncated": bool(section.get("truncated"))}
            if document.get("doc_type") in SYNTHESIS_DOC_TYPES:
                passages.append(item)
            else:
                excerpts.append({**item, "kind": section.get("kind")})
    return excerpts, passages


def _context(value: Any) -> list[dict[str, str]]:
    turns = []
    for item in (value if isinstance(value, list) else [])[:lab_jobs.CONTEXT_MAX_ITEMS]:
        if isinstance(item, Mapping) and isinstance(item.get("role"), str) and isinstance(item.get("text"), str):
            turns.append({"role": item["role"], "text": item["text"][:lab_jobs.CONTEXT_MAX_CHARS]})
    return turns


def _hint(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("kind") not in HINT_KINDS:
        return dict(NO_HINT)
    note = value.get("note")
    return {"kind": value["kind"], "target_keys": _strings(value.get("target_keys")),
            "note": note[:MAX_ITEM_CHARS] if isinstance(note, str) else ""}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item[:MAX_ITEM_CHARS] for item in value[:MAX_ITEMS] if isinstance(item, str) and item.strip()]


def _bounded_text(value: Any) -> str:
    return value.strip()[:MAX_TEXT_CHARS] if isinstance(value, str) else ""


# ---------------------------------------------------------------------------------------------
# Receipts and records
# ---------------------------------------------------------------------------------------------

def _read_receipt(receipts: ReceiptWriter, key: str) -> Any:
    """A receipt's JSON, or ``None`` when the object does not exist; other S3 failures propagate."""
    try:
        return receipts.get_json(key)
    except ClientError as exc:
        response = exc.response if hasattr(exc, "response") else {}
        code = response.get("Error", {}).get("Code")
        if code in MISSING_CODES or response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return None
        raise


def _write_once(receipts: ReceiptWriter, key: str, body: Any) -> bool:
    """Write a receipt with IfNoneMatch; ``False`` when an object already stood there (412)."""
    try:
        receipts.put_json(key, body)
    except ClientError as exc:
        response = exc.response if hasattr(exc, "response") else {}
        if (response.get("Error", {}).get("Code") == "PreconditionFailed"
                or response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 412):
            return False
        raise
    return True


def _triageable_job(table: TablePort, job_id: str) -> dict[str, Any]:
    job = table.get(*keys.job(job_id))
    if job is None:
        raise NotFound(f"no job {job_id}")
    if job.get("kind") != lab_jobs.KIND_ANSWER:
        raise ValueError("triage runs for answer jobs only")
    if job.get("status") not in TRIAGED_JOB_STATUSES:
        raise InvalidTransition(f"job {job_id} is {job.get('status')}, not completed or partial")
    return job


def _job_id_of(job: Any) -> str:
    if not isinstance(job, Mapping):
        raise ValueError("job must be the answer job record (or a mapping with its job_id)")
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or not job_id or "/" in job_id or len(job_id) > 128:
        raise ValueError("job_id must be a simple identifier")
    return job_id


def _moment(now: datetime | None) -> datetime:
    moment = now or datetime.now(UTC)
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    return moment
