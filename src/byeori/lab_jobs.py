"""Job records for the student question workflow (docs/LAB-QUESTION-WORKFLOW.md, P1).

A question becomes a job in two deliberately separate steps. ``intake`` records the request
idempotently under the caller's verified identity, then ``queue`` moves it onto the paid path in
one transaction: the status change, the job's budget reservation from ``lab_budget``, the outbox
row a relay sends to SQS and that row's pending pointer. Workers ``claim`` a job under a lease
and close it with ``complete``, ``fail`` or ``mark_unknown``; each close settles or flags the job
reservation in the same transaction, so a period scope never keeps a finished job's hold.

Everything here goes through ``lab_store.TablePort`` operations and ``ReceiptWriter``; the only
S3 keys written are ``runs/lab-questions/{job_id}/request.json``. Identity is never read from a
request body: ``intake`` takes the verified member as an argument and drops ``author``,
``member_id`` and ``role`` fields before hashing.

Every composed transaction touches the period scopes ``BUDGET#lab:{period}`` and
``BUDGET#member:{id}:{period}``, which all queueing and every close contend for. Under
``ConditionFailed`` (including ``TransactionConflict``) the loops here re-read, re-plan and
commit again up to ``TRANSACTION_ATTEMPTS`` times with a jittered pause from
``backoff_seconds`` between attempts; ``_sleep`` is module level so tests can record it.
"""
from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from botocore.exceptions import ClientError

from byeori import lab_budget
from byeori.lab_budget import BudgetExceeded, InvalidTransition, NotFound
from byeori.lab_policy import ANSWER_JOB_CAP_MICROS, LEASE_SECONDS, POLICY_REVISION
from byeori.lab_store import (
    RECEIPT_PREFIX,
    ConditionFailed,
    Delete,
    Operation,
    Put,
    ReceiptWriter,
    StoreError,
    TablePort,
    Update,
    context_hash,
    day_for,
    digest,
    keys,
    new_id,
    new_item,
    now_iso,
    period_for,
    question_hash,
    receipt_key,
)

__all__ = [
    "CLAIMABLE_STATUSES", "Forbidden", "IdempotencyConflict", "InvalidTransition", "JobPlan", "KIND_ANSWER",
    "KIND_RESEARCH", "Member", "NotFound", "OUTBOX_STATUSES", "STATUSES", "TERMINAL_STATUSES", "TRANSACTION_ATTEMPTS",
    "TRIAGE_STATUSES", "backoff_seconds", "claim", "complete", "create_research_job", "existing_execution",
    "expired_leases", "fail", "intake", "mark_sent", "mark_unknown", "normalise_request", "pending_outbox",
    "plan_research_job", "queue", "read_job", "read_verdict", "set_triage_status", "sweep_expired_leases",
]

KIND_ANSWER, KIND_RESEARCH, KIND_TRIAGE = "answer", "research", "triage"
OUTBOX_KINDS = frozenset({KIND_ANSWER, KIND_TRIAGE, KIND_RESEARCH})

RECEIVED, QUEUED, REJECTED_BUDGET, RUNNING = "received", "queued", "rejected_budget", "running"
COMPLETED, PARTIAL, FAILED, OUTCOME_UNKNOWN = "completed", "partial", "failed", "outcome_unknown"
PAUSED_BUDGET, PAUSED_RESUME = "paused_budget", "paused_resume"
STATUSES = frozenset({RECEIVED, QUEUED, REJECTED_BUDGET, RUNNING, COMPLETED, PARTIAL, FAILED, OUTCOME_UNKNOWN,
                      PAUSED_BUDGET, PAUSED_RESUME})
TERMINAL_STATUSES = frozenset({REJECTED_BUDGET, COMPLETED, PARTIAL, FAILED, OUTCOME_UNKNOWN})
CLAIMABLE_STATUSES = frozenset({QUEUED, PAUSED_RESUME})
TRIAGE_STATUSES = frozenset({"pending", "complete", "unavailable", "skipped"})

OUTBOX_PENDING, OUTBOX_SENT, OUTBOX_DONE, OUTBOX_FAILED = "pending", "sent", "done", "failed"
OUTBOX_STATUSES = frozenset({OUTBOX_PENDING, OUTBOX_SENT, OUTBOX_DONE, OUTBOX_FAILED})

ROLES = frozenset({"student", "admin"})
CONTEXT_MAX_ITEMS = 8
CONTEXT_MAX_CHARS = 4000
CONTEXT_ROLES = frozenset({"user", "assistant"})
QUESTION_MAX_CHARS = 8000          # storage bound for one question; the packet, not the question, is the model input
TRANSACTION_ATTEMPTS = 8           # re-plans under ConditionFailed from contention on the period scopes
BACKOFF_BASE_SECONDS = 0.02        # first pause ceiling; doubles per attempt up to BACKOFF_MAX_SECONDS
BACKOFF_MAX_SECONDS = 0.5
OUTBOX_SCAN_PAGES = 10             # pending_outbox reads at most this many 100-row pointer pages
LEASE_SWEEP_DAYS = 2               # expired_leases walks the day pointers of today and yesterday
LEASE_SCAN_PAGES = 20              # per day, 500 pointers each
EXECUTION_SK = "EXECUTION"         # JOB#{parent}/EXECUTION: the one research execution a parent job may have
ORPHAN_REASON = "previous attempt left a sent call unresolved"
LEASE_EXPIRED_REASON = "lease_expired"
IDENTITY_FIELDS = ("author", "member_id", "role")
TRANSPORT_FIELDS = ("action", "request_id")

_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# C0 and C1 control characters except newline and tab; they never carry meaning in a question.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

_sleep = time.sleep                # injectable: tests replace it with a recorder


def backoff_seconds(attempt: int) -> float:
    """A jittered pause before retry number ``attempt`` (0 for the first retry).

    Full jitter over an exponential ceiling: ``uniform(0, min(MAX, BASE * 2**attempt))``. The
    period scopes are hot items every queueing and every close contend for, so a fixed pause
    would let the same competitors collide again.
    """
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("attempt must be a non-negative integer")
    ceiling = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** attempt))
    return random.uniform(0, ceiling)


def _pause_before(attempt: int) -> None:
    """Sleep between transaction attempts; ``attempt`` is the loop index of the attempt about to run."""
    if attempt > 0:
        _sleep(backoff_seconds(attempt - 1))


class IdempotencyConflict(StoreError):
    """The same ``request_id`` arrived again with a different payload; nothing was written."""

    code = "idempotency_conflict"


class Forbidden(StoreError):
    """The caller's role does not allow the action.

    A session or parent job that belongs to another member is reported as ``NotFound`` instead,
    so a foreign id is indistinguishable from a missing one and never confirms existence.
    """

    code = "forbidden"


@dataclass(frozen=True)
class Member:
    """The verified caller. The gateway builds it from the registry, never from a request body."""

    member_id: str
    role: str = "student"
    policy_revision: str = POLICY_REVISION


@dataclass(frozen=True)
class JobPlan:
    """A research job with its outbox row, reservation and request receipt; nothing is written yet.

    ``lab_offers`` and ``lab_review`` add their own offer, approval or candidate operations to
    ``operations``, commit one transaction, then write ``request`` to ``job["request_key"]``.
    ``outbox`` and ``reservation`` are ``None`` for a plan made with ``reserve=False``.
    """

    job: dict[str, Any]
    outbox: dict[str, Any] | None
    reservation: dict[str, Any] | None
    request: dict[str, Any]
    operations: tuple[Operation, ...]


# ---------------------------------------------------------------------------------------------
# Request normalisation
# ---------------------------------------------------------------------------------------------

def normalise_request(body: Mapping[str, Any]) -> dict[str, Any]:
    """The request fields intake records, with identity and transport fields dropped.

    ``author``, ``member_id`` and ``role`` are never identity here and never reach the payload
    hash; ``action`` and ``request_id`` are transport. Absent optional fields are ``None`` so the
    hash of a retry does not depend on which keys the client omitted. Control characters other
    than newline and tab are removed from the question and the context texts (a NUL byte would
    otherwise reach the FTS5 tokenizer); the words themselves are kept verbatim, since
    negations, populations, numbers and conditions are part of the record.
    """
    if not isinstance(body, Mapping):
        raise ValueError("the request body must be a JSON object")
    question = body.get("question")
    if not isinstance(question, str):
        raise ValueError("question must be a non-empty string")
    question = _CONTROL.sub("", question)
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    if len(question) > QUESTION_MAX_CHARS:
        raise ValueError(f"question holds at most {QUESTION_MAX_CHARS} characters")
    private = body.get("private_material", False)
    if not isinstance(private, bool):
        raise ValueError("private_material must be true or false when present")
    return {
        "question": question,
        "session_id": _optional_identifier(body.get("session_id"), "session_id"),
        "parent_job_id": _optional_identifier(body.get("parent_job_id"), "parent_job_id"),
        "context": _normalised_context(body.get("context")),
        "private_material": private,
    }


def _normalised_context(context: Any) -> list[dict[str, str]]:
    if context is None:
        return []
    if not isinstance(context, list) or len(context) > CONTEXT_MAX_ITEMS:
        raise ValueError(f"context holds at most {CONTEXT_MAX_ITEMS} prior turns")
    cleaned, total = [], 0
    for item in context:
        if not isinstance(item, Mapping) or not isinstance(item.get("role"), str) or not isinstance(item.get("text"), str):
            raise ValueError("each context item is {'role': 'user'|'assistant', 'text': str}")
        if item["role"] not in CONTEXT_ROLES:
            raise ValueError("context roles are user or assistant; instructions are not conversation context")
        text = _CONTROL.sub("", item["text"])
        total += len(text)
        cleaned.append({"role": item["role"], "text": text})
    if total > CONTEXT_MAX_CHARS:
        raise ValueError(f"context holds at most {CONTEXT_MAX_CHARS} characters in total")
    return cleaned


def _optional_identifier(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ValueError(f"{name} must be a simple identifier of at most 128 characters")
    return value


def _identity(member: Any) -> tuple[str, str]:
    if isinstance(member, Mapping):
        member_id, role = member.get("member_id"), member.get("role")
    else:
        member_id, role = getattr(member, "member_id", None), getattr(member, "role", None)
    if not isinstance(member_id, str) or not _IDENTIFIER.match(member_id) or role not in ROLES:
        raise ValueError("member must be the verified registry member with member_id and role")
    return member_id, role


# ---------------------------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------------------------

def intake(table: TablePort, receipts: ReceiptWriter, member: Any, body: Mapping[str, Any],
           now: datetime | None = None, *, policy_revision: str = POLICY_REVISION) -> dict[str, Any]:
    """Record a question under the verified ``member`` and return its job record (status ``received``).

    One transaction writes the idempotency key, the job, its member and day pointers and the
    session (new, or the caller's own with ``last_turn`` advanced), then ``request.json`` goes to
    S3. The same ``request_id`` with the same payload returns the existing job without writing;
    a different payload raises ``IdempotencyConflict``. A session or parent job that is missing
    or belongs to another member raises ``NotFound`` either way, so ids never leak.
    """
    member_id, _role = _identity(member)
    request_id = _identifier(body.get("request_id") if isinstance(body, Mapping) else None, "request_id")
    request = normalise_request(body)
    payload_hash = digest(request)
    existing = _existing_job(table, member_id, KIND_ANSWER, request_id, payload_hash)
    if existing is not None:
        return _returned_existing(receipts, existing, request)
    moment = _moment(now)
    stamp = now_iso(moment)
    job_id = new_id()
    last_error: ConditionFailed | None = None
    for attempt in range(TRANSACTION_ATTEMPTS):
        _pause_before(attempt)
        _parent_of(table, member_id, request["parent_job_id"])
        session_id, turn, session_op = _session_operation(table, member_id, request["session_id"], job_id, stamp)
        job = _answer_job(job_id, member_id, request, request_id, payload_hash, session_id, turn, stamp, policy_revision)
        idem = new_item(*keys.idempotency(member_id, KIND_ANSWER, request_id), stamp, payload_hash=payload_hash,
                        job_id=job_id, kind=KIND_ANSWER)
        operations = [Put(idem), Put(job), *_pointer_puts(job, stamp), session_op]
        try:
            table.transact(operations)
        except ConditionFailed as exc:
            last_error = exc
            existing = _existing_job(table, member_id, KIND_ANSWER, request_id, payload_hash)
            if existing is not None:
                return _returned_existing(receipts, existing, request)
            continue
        receipts.put_json(job["request_key"], _request_receipt(job, request))
        return job
    raise last_error  # type: ignore[misc]


def _existing_job(table: TablePort, member_id: str, kind: str, request_id: str, payload_hash: str) -> dict[str, Any] | None:
    idem = table.get(*keys.idempotency(member_id, kind, request_id))
    if idem is None:
        return None
    if idem.get("payload_hash") != payload_hash:
        raise IdempotencyConflict(f"request {request_id} was already used with a different payload")
    job = table.get(*keys.job(idem["job_id"]))
    if job is None:
        raise NotFound(f"idempotency key for {request_id} points at missing job {idem['job_id']}")
    return job


def _returned_existing(receipts: ReceiptWriter, job: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """A duplicate whose first intake stopped before its receipt landed gets the receipt now, once."""
    if job["status"] == RECEIVED and job["kind"] == KIND_ANSWER:
        try:
            receipts.put_json(job["request_key"], _request_receipt(job, request))
        except ClientError as exc:
            if not _already_exists(exc):
                raise
    return job


def _already_exists(exc: ClientError) -> bool:
    error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") if hasattr(exc, "response") else None
    return error.get("Code") == "PreconditionFailed" or status == 412


def _parent_of(table: TablePort, member_id: str, parent_job_id: str | None) -> dict[str, Any] | None:
    """The caller's own parent job; a missing or foreign one is ``NotFound`` alike (no existence oracle)."""
    if parent_job_id is None:
        return None
    parent = table.get(*keys.job(parent_job_id))
    if parent is None or parent.get("member_id") != member_id:
        raise NotFound(f"no job {parent_job_id}")
    return parent


def _session_operation(table: TablePort, member_id: str, session_id: str | None, job_id: str,
                       stamp: str) -> tuple[str, int, Operation]:
    if session_id is None:
        session_id = new_id()
        item = new_item(*keys.session(session_id), stamp, session_id=session_id, member_id=member_id, last_turn=1,
                        last_job_id=job_id)
        return session_id, 1, Put(item)
    session = table.get(*keys.session(session_id))
    if session is None or session.get("member_id") != member_id:
        raise NotFound(f"no session {session_id}")
    turn = int(session.get("last_turn", 0)) + 1
    return session_id, turn, Update(*keys.session(session_id), session["revision"], {"last_turn": turn, "last_job_id": job_id})


def _answer_job(job_id: str, member_id: str, request: dict[str, Any], request_id: str, payload_hash: str,
                session_id: str, turn: int, stamp: str, policy_revision: str) -> dict[str, Any]:
    ctx_hash = context_hash(request["context"])
    follow_up = bool(request["context"]) or request["parent_job_id"] is not None
    return new_item(
        *keys.job(job_id), stamp,
        job_id=job_id, kind=KIND_ANSWER, member_id=member_id, session_id=session_id, turn=turn,
        parent_job_id=request["parent_job_id"], question=request["question"],
        standalone_question=None if follow_up else request["question"],
        context_hash=ctx_hash, question_hash=question_hash(request["question"], ctx_hash),
        request_id=request_id, request_hash=payload_hash, policy_revision=policy_revision,
        private_material=request["private_material"],
        status=RECEIVED, attempt=0, lease_until=None, period=None, reservation_id=None, outbox_id=None,
        request_key=receipt_key(job_id, "request.json"), receipt_key=None, evidence_key=None,
        triage_status="pending", triage_outbox_id=None, offer_id=None, completed_at=None, usage=None, usd_micros=None,
    )


def _request_receipt(job: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"], "kind": job["kind"], "member_id": job["member_id"], "session_id": job["session_id"],
        "turn": job["turn"], "parent_job_id": job["parent_job_id"], "request_id": job["request_id"],
        "request_hash": job["request_hash"], "question": request["question"], "context": request["context"],
        "context_hash": job["context_hash"], "question_hash": job["question_hash"],
        "private_material": request["private_material"], "policy_revision": job["policy_revision"],
        "received_at": job["created_at"],
    }


def _pointer_puts(job: dict[str, Any], stamp: str) -> list[Operation]:
    member_pk, member_sk = keys.member_job(job["member_id"], job["created_at"], job["job_id"])
    day_pk, day_sk = keys.day_record(_day(job["created_at"]), job["created_at"], job["job_id"])
    return [
        Put(new_item(member_pk, member_sk, stamp, job_id=job["job_id"], kind=job["kind"], status=job["status"])),
        Put(new_item(day_pk, day_sk, stamp, job_id=job["job_id"], member_id=job["member_id"], kind=job["kind"],
                     status=job["status"])),
    ]


# ---------------------------------------------------------------------------------------------
# Queueing: status change + reservation + outbox in one transaction
# ---------------------------------------------------------------------------------------------

def queue(table: TablePort, job_id: str, *, period: str | None = None, cap: int | None = ANSWER_JOB_CAP_MICROS,
          now: datetime | None = None) -> dict[str, Any]:
    """Move a ``received`` job to ``queued`` with its reservation, outbox row and pending pointer.

    ``cap`` is ``None`` by default, so the job's reservation records what it spends and refuses
    nothing. The ``rejected_budget`` path stays because a cap an administrator sets on a period
    scope still applies: when one would be exceeded the job carries the reason and no reservation
    or outbox row exists. A job that already left ``received`` is returned unchanged, so a retried
    gateway call after a crash is harmless. Contention on the period scopes is re-planned with a
    backoff up to ``TRANSACTION_ATTEMPTS`` times.
    """
    moment = _moment(now)
    stamp = now_iso(moment)
    period = period or period_for(moment)
    last_error: ConditionFailed | None = None
    for attempt in range(TRANSACTION_ATTEMPTS):
        _pause_before(attempt)
        job = _require_job(table, job_id)
        if job["status"] != RECEIVED:
            return job
        try:
            plan = lab_budget.plan_job_reservation(table, job_id, job["member_id"], period, cap, now=moment)
        except BudgetExceeded as exc:
            changes = {"status": REJECTED_BUDGET, "reason": str(exc), "rejected_scope": exc.scope,
                       "requested_micros": exc.requested, "available_micros": exc.available, "rejected_at": stamp,
                       "triage_status": "skipped"}
            operations = [Update(*keys.job(job_id), job["revision"], changes),
                          *_pointer_status_updates(table, job, REJECTED_BUDGET)]
        else:
            outbox_id = new_id()
            changes = {"status": QUEUED, "reservation_id": plan.record["reservation_id"], "outbox_id": outbox_id,
                       "period": period, "queued_at": stamp}
            operations = [Update(*keys.job(job_id), job["revision"], changes),
                          *_pointer_status_updates(table, job, QUEUED),
                          *plan.operations,
                          *_outbox_puts(outbox_id, KIND_ANSWER, job_id, stamp)]
        try:
            table.transact(operations)
        except ConditionFailed as exc:
            last_error = exc
            continue
        return _require_job(table, job_id)
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# Worker transitions
# ---------------------------------------------------------------------------------------------

def claim(table: TablePort, job_id: str, outbox_id: str, lease_seconds: int = LEASE_SECONDS,
          now: datetime | None = None) -> dict[str, Any]:
    """Take a ``queued`` (or ``paused_resume``) job to ``running`` under a lease, ``attempt + 1``.

    A live lease refuses with ``InvalidTransition`` whose ``code`` is ``lease_held`` (retriable);
    an expired lease is re-claimed. A terminal or never-queued job refuses with the plain
    ``invalid_transition`` code, so a duplicate queue delivery is dropped, not retried.

    An expired lease whose job scope still holds micros (an attempt reservation of the earlier
    attempt was never settled, released or resolved: the worker died with a model call in
    flight) is not re-run. The job is closed as ``outcome_unknown`` with ``ORPHAN_REASON``, its
    job reservation is flagged unknown so the period scopes keep the hold, and the plain
    ``invalid_transition`` code tells the worker to drop the message.
    """
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be a positive integer")
    moment = _moment(now)
    stamp = now_iso(moment)
    job = _require_job(table, job_id)
    status = job["status"]
    if status == RUNNING:
        lease_until = job.get("lease_until")
        if lease_until and datetime.fromisoformat(lease_until) > moment:
            raise InvalidTransition(f"job {job_id} is running under a lease until {lease_until}", code="lease_held")
        if _orphaned_micros(table, job) > 0:
            _close_orphaned(table, job, stamp, moment)
            raise InvalidTransition(f"job {job_id} closed as {OUTCOME_UNKNOWN}: {ORPHAN_REASON}")
    elif status not in CLAIMABLE_STATUSES:
        raise InvalidTransition(f"job {job_id} is {status}; only queued or paused_resume jobs can be claimed")
    outbox = table.get(*keys.outbox(_identifier(outbox_id, "outbox_id")))
    if outbox is None or outbox.get("job_id") != job_id:
        raise InvalidTransition(f"outbox {outbox_id} does not deliver job {job_id}", code="invalid_outbox")
    changes = {"status": RUNNING, "attempt": int(job.get("attempt", 0)) + 1,
               "lease_until": now_iso(moment + timedelta(seconds=lease_seconds)), "claimed_at": stamp,
               "claimed_outbox_id": outbox_id}
    table.transact([Update(*keys.job(job_id), job["revision"], changes), *_pointer_status_updates(table, job, RUNNING)])
    return _require_job(table, job_id)


def _close_orphaned(table: TablePort, job: dict[str, Any], stamp: str, moment: datetime) -> None:
    """Close a re-delivered job whose earlier attempt left a call unresolved; a concurrent close is fine."""
    try:
        table.transact(_unknown_operations(table, job, ORPHAN_REASON, stamp, moment))
    except ConditionFailed:
        if _require_job(table, job["job_id"])["status"] == RUNNING:
            raise


def complete(table: TablePort, job_id: str, expected_revision: int, *, receipt_key: str, evidence_key: str | None,
             usage: Mapping[str, Any] | None, usd_micros: int, status: str, hold_reason: str | None = None,
             now: datetime | None = None, attempt: int | None = None,
             claimed_outbox_id: str | None = None) -> dict[str, Any]:
    """Close a ``running`` job as ``completed`` or ``partial`` in one transaction.

    The transaction writes the terminal state, marks the answer outbox ``done`` (removing its
    pending pointer), creates the triage outbox for answer jobs (``triage_status`` stays
    ``pending``) and settles the job reservation with what the job ledger shows as billed,
    returning the rest to the period scopes. Attempt reservations are the worker's to settle
    before calling this; when one is still held or unknown (``reserved_micros > 0`` on the job
    scope) the job reservation is flagged unknown instead of settled, the period scopes keep the
    hold and the job records ``reservation_status`` ``unknown`` with ``orphaned_reserved_micros``.

    ``expected_revision`` is honoured on the first read. Should the transaction lose to another
    writer (a period scope moved), the job is re-read and re-planned; from then on ownership is
    the lease itself: the job must still carry the ``attempt`` and ``claimed_outbox_id`` of the
    first read, or the ones passed explicitly, so a revision moved by a concurrent pointer or
    budget update does not fail the close while a re-claim by another worker does.
    """
    if status not in {COMPLETED, PARTIAL}:
        raise ValueError(f"completion status is completed or partial, not {status!r}")
    amount = _amount(usd_micros, "usd_micros")
    moment = _moment(now)
    stamp = now_iso(moment)
    _running_job(table, job_id)  # a job that is not running is refused before its keys are judged
    _receipt_under_job(receipt_key, job_id, "receipt_key")
    if evidence_key is not None:
        _receipt_under_job(evidence_key, job_id, "evidence_key")

    def plan(job: dict[str, Any]) -> list[Operation]:
        changes: dict[str, Any] = {"status": status, "receipt_key": receipt_key, "evidence_key": evidence_key,
                                   "usage": dict(usage) if usage is not None else None, "usd_micros": amount,
                                   "completed_at": stamp, "lease_until": None}
        if hold_reason is not None:
            changes["hold_reason"] = str(hold_reason)
        operations: list[Operation] = []
        if job["kind"] == KIND_ANSWER:
            triage_outbox_id = new_id()
            changes.update(triage_status="pending", triage_outbox_id=triage_outbox_id)
            operations.extend(_outbox_puts(triage_outbox_id, KIND_TRIAGE, job_id, stamp))
        reservation_ops, reservation_changes = _close_job_reservation(table, job, moment)
        changes.update(reservation_changes)
        return [Update(*keys.job(job_id), job["revision"], changes),
                *_pointer_status_updates(table, job, status),
                *_outbox_status_updates(table, job.get("outbox_id"), OUTBOX_DONE, stamp, job_id=job_id),
                *operations,
                *reservation_ops]

    return _close_running_job(table, job_id, expected_revision, plan, attempt=attempt, claimed_outbox_id=claimed_outbox_id)


def fail(table: TablePort, job_id: str, expected_revision: int, *, reason: str, error_code: str | None = None,
         usage: Mapping[str, Any] | None = None, usd_micros: int | None = None,
         now: datetime | None = None, attempt: int | None = None,
         claimed_outbox_id: str | None = None) -> dict[str, Any]:
    """Close a ``running`` job as ``failed``: a definite failure whose bill is known (usually zero).

    The job reservation is settled with the ledger's billed amount, so unused micros return to
    the period scopes; an attempt reservation still held flags it unknown instead (see
    ``complete``). No triage outbox is created; ``triage_status`` becomes ``skipped``. Retries
    and ownership follow ``complete``.
    """
    given = None if usd_micros is None else _amount(usd_micros, "usd_micros")
    moment = _moment(now)
    stamp = now_iso(moment)

    def plan(job: dict[str, Any]) -> list[Operation]:
        billed = _billed(table, job)
        changes = {"status": FAILED, "reason": str(reason), "error_code": error_code,
                   "usage": dict(usage) if usage is not None else None,
                   "usd_micros": billed if given is None else given,
                   "completed_at": stamp, "lease_until": None, "triage_status": "skipped"}
        reservation_ops, reservation_changes = _close_job_reservation(table, job, moment, billed=billed)
        changes.update(reservation_changes)
        return [Update(*keys.job(job_id), job["revision"], changes),
                *_pointer_status_updates(table, job, FAILED),
                *_outbox_status_updates(table, job.get("outbox_id"), OUTBOX_DONE, stamp, job_id=job_id),
                *reservation_ops]

    return _close_running_job(table, job_id, expected_revision, plan, attempt=attempt, claimed_outbox_id=claimed_outbox_id)


def mark_unknown(table: TablePort, job_id: str, expected_revision: int, *, reason: str,
                 now: datetime | None = None, attempt: int | None = None,
                 claimed_outbox_id: str | None = None) -> dict[str, Any]:
    """Close a ``running`` job as ``outcome_unknown``: the bill is not known, so nothing is released.

    The job reservation is flagged ``unknown`` and keeps its micros reserved on the period
    scopes until an operator settles it. This is never treated as cost zero or re-run. Retries
    and ownership follow ``complete``.
    """
    moment = _moment(now)
    stamp = now_iso(moment)

    def plan(job: dict[str, Any]) -> list[Operation]:
        return _unknown_operations(table, job, str(reason), stamp, moment)

    return _close_running_job(table, job_id, expected_revision, plan, attempt=attempt, claimed_outbox_id=claimed_outbox_id)


def _unknown_operations(table: TablePort, job: dict[str, Any], reason: str, stamp: str,
                        moment: datetime) -> list[Operation]:
    changes = {"status": OUTCOME_UNKNOWN, "reason": reason, "completed_at": stamp, "unknown_at": stamp,
               "lease_until": None, "triage_status": "skipped"}
    operations: list[Operation] = [Update(*keys.job(job["job_id"]), job["revision"], changes),
                                   *_pointer_status_updates(table, job, OUTCOME_UNKNOWN),
                                   *_outbox_status_updates(table, job.get("outbox_id"), OUTBOX_DONE, stamp,
                                                           job_id=job["job_id"])]
    if job.get("reservation_id"):
        plan = lab_budget.plan_mark_unknown(table, job["reservation_id"], reason=reason, now=moment)
        operations.extend(plan.operations)
    return operations


def _close_running_job(table: TablePort, job_id: str, expected_revision: int,
                       plan: Callable[[dict[str, Any]], list[Operation]], *, attempt: int | None,
                       claimed_outbox_id: str | None) -> dict[str, Any]:
    """Re-read, plan and commit a close of the running job under a bounded, backed-off retry.

    The first read must show ``expected_revision`` unless the caller proves ownership with the
    ``attempt`` and ``claimed_outbox_id`` of its claim; every later read must show the same
    pair. A job that another worker claimed again, or that is no longer running, ends the loop
    with ``ConditionFailed`` or ``InvalidTransition`` respectively.
    """
    if (attempt is None) != (claimed_outbox_id is None):
        raise ValueError("attempt and claimed_outbox_id prove ownership together; pass both or neither")
    proof = None if attempt is None else (int(attempt), claimed_outbox_id)
    last_error: ConditionFailed | None = None
    for number in range(TRANSACTION_ATTEMPTS):
        _pause_before(number)
        job = _running_job(table, job_id)
        lease = (int(job.get("attempt", 0)), job.get("claimed_outbox_id"))
        if proof is None:
            if job["revision"] != int(expected_revision):
                raise ConditionFailed(f"job {job_id} is at revision {job['revision']}, not {expected_revision}")
            proof = lease
        elif lease != proof:
            raise ConditionFailed(f"job {job_id} was claimed again (attempt {lease[0]}); this close is stale")
        try:
            table.transact(plan(job))
        except ConditionFailed as exc:
            last_error = exc
            continue
        return _require_job(table, job_id)
    raise last_error  # type: ignore[misc]


def set_triage_status(table: TablePort, job_id: str, expected_revision: int, status: str, *,
                      outbox_id: str | None = None, now: datetime | None = None, **extra: Any) -> dict[str, Any]:
    """Record the triage worker's outcome on the job (``pending``/``complete``/``unavailable``/``skipped``).

    ``extra`` attributes such as ``offer_id`` are stored with it. With ``outbox_id`` the same
    transaction marks that triage outbox row ``done`` and removes its pending pointer. The job
    status itself never changes here. ``expected_revision`` is honoured on the first read; when
    the transaction loses to another writer the job is re-read and, unless the outcome is
    already recorded, re-planned at its current revision.
    """
    if status not in TRIAGE_STATUSES:
        raise ValueError(f"triage status must be one of {sorted(TRIAGE_STATUSES)}, not {status!r}")
    reserved = {"pk", "sk", "revision", "updated_at", "status", "triage_status", "job_id", "kind", "member_id"}
    if reserved & set(extra):
        raise ValueError(f"set_triage_status cannot change {sorted(reserved & set(extra))}")
    stamp = now_iso(_moment(now))
    last_error: ConditionFailed | None = None
    for number in range(TRANSACTION_ATTEMPTS):
        _pause_before(number)
        job = _require_job(table, job_id)
        if number == 0 and job["revision"] != int(expected_revision):
            raise ConditionFailed(f"job {job_id} is at revision {job['revision']}, not {expected_revision}")
        outbox_ops = _outbox_status_updates(table, outbox_id, OUTBOX_DONE, stamp, job_id=job_id) if outbox_id else []
        recorded = job.get("triage_status") == status and all(job.get(k) == v for k, v in extra.items())
        if number and recorded and not outbox_ops:
            return job
        operations: list[Operation] = [Update(*keys.job(job_id), job["revision"], {"triage_status": status, **extra}),
                                       *outbox_ops]
        try:
            table.transact(operations)
        except ConditionFailed as exc:
            last_error = exc
            continue
        return _require_job(table, job_id)
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------------------------

def read_job(table: TablePort, job_id: str, member: Any) -> dict[str, Any]:
    """The job for its owner or an admin; anyone else gets ``NotFound`` so existence never leaks."""
    member_id, role = _identity(member)
    job = table.get(*keys.job(job_id)) if isinstance(job_id, str) and job_id else None
    if job is None or (role != "admin" and job.get("member_id") != member_id):
        raise NotFound(f"no job {job_id}")
    return job


def read_verdict(table: TablePort, job_id: str) -> dict[str, Any] | None:
    """The job's ``TRIAGE`` record, or ``None`` while triage has not recorded one."""
    return table.get(*keys.verdict(job_id))


def existing_execution(table: TablePort, parent_job_id: str) -> dict[str, Any] | None:
    """The ``JOB#{parent}/EXECUTION`` guard naming the one research execution of ``parent_job_id``, or ``None``.

    ``plan_research_job`` puts this item once, so a student's consent and a professor's
    approval for the same question cannot both start a research run: the loser's transaction
    fails with ``ConditionFailed`` and reads the winner here.
    """
    if not isinstance(parent_job_id, str) or not parent_job_id:
        return None
    return table.get(*_execution_key(parent_job_id))


def _execution_key(parent_job_id: str) -> tuple[str, str]:
    return keys.job(parent_job_id)[0], EXECUTION_SK


# ---------------------------------------------------------------------------------------------
# Lease recovery (administrator reconciliation; docs/LAB-QUESTION-WORKFLOW.md section 8)
# ---------------------------------------------------------------------------------------------

def expired_leases(table: TablePort, now: datetime | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """``running`` jobs whose lease has ended, oldest first, at most ``limit``.

    Walks the ``RECORDS#{day}`` pointers of the last ``LEASE_SWEEP_DAYS`` days (a lease lasts
    minutes, so an older running pointer is a job the worker never closed and this sweep would
    have caught the day before). A lease counts as ended exactly when ``claim`` would re-claim
    it: ``lease_until`` is not after ``now``.
    """
    moment = _moment(now)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    found: list[dict[str, Any]] = []
    for offset in range(LEASE_SWEEP_DAYS - 1, -1, -1):
        day_pk = keys.day_record(day_for(moment - timedelta(days=offset)), "", "")[0]
        cursor: str | None = None
        for _page in range(LEASE_SCAN_PAGES):
            pointers, cursor = table.query(day_pk, limit=500, start_after=cursor)
            for pointer in pointers:
                if pointer.get("status") != RUNNING or not pointer.get("job_id"):
                    continue
                job = table.get(*keys.job(pointer["job_id"]))
                if job is None or job.get("status") != RUNNING or not job.get("lease_until"):
                    continue
                if datetime.fromisoformat(job["lease_until"]) <= moment:
                    found.append(job)
                    if len(found) >= limit:
                        return found
            if cursor is None:
                break
    return found


def sweep_expired_leases(table: TablePort, now: datetime | None = None, limit: int = 100) -> dict[str, Any]:
    """Close every expired running job as ``outcome_unknown`` (reason ``lease_expired``) and count the work.

    The bill of a job whose worker vanished is unknown, so each job reservation is flagged
    unknown and its micros stay held for an operator to settle against the provider's record.
    A job that was closed or claimed again between listing and closing is counted as skipped.
    """
    moment = _moment(now)
    jobs = expired_leases(table, moment, limit)
    counts: dict[str, Any] = {"expired": len(jobs), "closed": 0, "skipped": 0, "job_ids": []}
    for job in jobs:
        try:
            mark_unknown(table, job["job_id"], job["revision"], reason=LEASE_EXPIRED_REASON, now=moment)
        except (ConditionFailed, StoreError):
            counts["skipped"] += 1
            continue
        counts["closed"] += 1
        counts["job_ids"].append(job["job_id"])
    return counts


# ---------------------------------------------------------------------------------------------
# Outbox relay bookkeeping
# ---------------------------------------------------------------------------------------------

def pending_outbox(table: TablePort, kind: str, limit: int = 25, *,
                   include_sent_before: datetime | str | None = None) -> list[dict[str, Any]]:
    """Outbox rows of ``kind`` still ``pending``, oldest first, at most ``limit``.

    ``OUTBOX/PENDING#`` pointers exist only for live rows: ``mark_sent`` and every close remove
    the pointer in the same transaction that moves the row, so the scan (at most
    ``OUTBOX_SCAN_PAGES`` pages of 100) never walks finished deliveries. With
    ``include_sent_before`` a row a pointer still references that was marked ``sent`` before
    that moment is returned as well. A message lost after ``mark_sent`` committed is SQS's to
    redeliver (at-least-once); the relay only re-sends rows whose pointer survived.
    """
    if kind not in OUTBOX_KINDS:
        raise ValueError(f"outbox kind must be one of {sorted(OUTBOX_KINDS)}, not {kind!r}")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    stale_before = now_iso(include_sent_before) if isinstance(include_sent_before, datetime) else include_sent_before
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    for _page in range(OUTBOX_SCAN_PAGES):
        pointers, cursor = table.query("OUTBOX", sk_prefix="PENDING#", limit=100, start_after=cursor)
        for pointer in pointers:
            if pointer.get("kind") != kind or not pointer.get("outbox_id"):
                continue
            row = table.get(*keys.outbox(pointer["outbox_id"]))
            if row is not None and _relayable(row, stale_before):
                rows.append(row)
                if len(rows) >= limit:
                    return rows
        if cursor is None:
            break
    return rows


def _relayable(row: Mapping[str, Any], stale_before: str | None) -> bool:
    status = row.get("status")
    if status == OUTBOX_PENDING:
        return True
    return status == OUTBOX_SENT and stale_before is not None and (row.get("sent_at") or "") < stale_before


def mark_sent(table: TablePort, outbox_id: str, expected_revision: int, *, now: datetime | None = None) -> dict[str, Any]:
    """Record that the row's message was handed to SQS: ``sent``, ``sent_at``, ``attempts + 1``.

    The row's pending pointer is removed in the same transaction, so the relay's next scan does
    not see it again; the row itself stays as the delivery record.
    """
    stamp = now_iso(_moment(now))
    row = table.get(*keys.outbox(outbox_id))
    if row is None:
        raise NotFound(f"no outbox row {outbox_id}")
    if row["status"] not in {OUTBOX_PENDING, OUTBOX_SENT}:
        raise InvalidTransition(f"outbox {outbox_id} is {row['status']}; only pending or sent rows are sent")
    operations: list[Operation] = [Update(*keys.outbox(outbox_id), int(expected_revision),
                                          {"status": OUTBOX_SENT, "sent_at": stamp, "attempts": int(row.get("attempts", 0)) + 1})]
    pointer = table.get(*keys.outbox_pending(row["created_at"], outbox_id))
    if pointer is not None:
        operations.append(Delete(*keys.outbox_pending(row["created_at"], outbox_id), pointer["revision"]))
    table.transact(operations)
    return table.get(*keys.outbox(outbox_id))


# ---------------------------------------------------------------------------------------------
# Research jobs (created by a student's consent or a professor's approval)
# ---------------------------------------------------------------------------------------------

def plan_research_job(table: TablePort, *, parent_job: Mapping[str, Any], member_id: str, approval_id: str,
                      scope: Mapping[str, Any], budget_usd_micros: int, now: datetime | None = None,
                      policy_revision: str = POLICY_REVISION, reserve: bool = True,
                      status: str = QUEUED) -> JobPlan:
    """Plan a research job for ``parent_job`` with its reservation, outbox, receipt and execution guard.

    The idempotency key is ``(member_id, "research", approval_id)`` and its payload hash is the
    digest of ``scope``. ``BudgetExceeded`` propagates: the caller decides how to record it.
    The plan also puts ``JOB#{parent}/EXECUTION`` once, so only one research execution can
    exist per parent answer job whichever path (student consent, professor approval) commits
    first; the other path's transaction fails and ``existing_execution`` names the winner.

    With ``reserve=False`` the plan carries no budget operations and no outbox row and the job
    takes the given ``status`` (``paused_budget`` records a consent the period cap could not
    hold); a ``queued`` job always reserves.
    """
    member_id = _identifier(member_id, "member_id")
    approval_id = _identifier(approval_id, "approval_id")
    if not isinstance(scope, Mapping):
        raise ValueError("scope must be a JSON object naming the question, targets and new pages")
    if not isinstance(parent_job, Mapping) or not parent_job.get("job_id"):
        raise ValueError("parent_job must be the parent job record")
    if status not in STATUSES:
        raise ValueError(f"research job status must be one of {sorted(STATUSES)}, not {status!r}")
    if not isinstance(reserve, bool):
        raise ValueError("reserve must be true or false")
    if (status == QUEUED) != reserve:
        raise ValueError("a queued research job reserves its budget and outbox row; any other status plans without them")
    question = scope.get("question") or parent_job.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("the research scope or its parent job must carry the question")
    moment = _moment(now)
    stamp = now_iso(moment)
    period = period_for(moment)
    job_id, outbox_id = new_id(), new_id()
    if reserve:
        budget = lab_budget.plan_job_reservation(table, job_id, member_id, period, budget_usd_micros, now=moment)
        reservation, budget_ops = budget.record, list(budget.operations)
        outbox_ops = _outbox_puts(outbox_id, KIND_RESEARCH, job_id, stamp)
    else:
        if isinstance(budget_usd_micros, bool) or not isinstance(budget_usd_micros, int) or budget_usd_micros <= 0:
            raise ValueError("budget_usd_micros must be a positive integer number of micro-USD")
        reservation, budget_ops, outbox_ops = None, [], []
    scope_record = dict(scope)
    scope_hash = digest(scope_record)
    job = new_item(
        *keys.job(job_id), stamp,
        job_id=job_id, kind=KIND_RESEARCH, member_id=member_id, session_id=parent_job.get("session_id"),
        turn=parent_job.get("turn"), parent_job_id=parent_job["job_id"], question=question,
        standalone_question=parent_job.get("standalone_question"), context_hash=parent_job.get("context_hash"),
        question_hash=parent_job.get("question_hash"), request_id=approval_id, request_hash=scope_hash,
        approval_id=approval_id, scope=scope_record, budget_usd_micros=budget_usd_micros,
        policy_revision=policy_revision, private_material=bool(parent_job.get("private_material", False)),
        status=status, attempt=0, lease_until=None, period=period,
        reservation_id=reservation["reservation_id"] if reservation else None,
        outbox_id=outbox_id if reserve else None, request_key=receipt_key(job_id, "request.json"),
        receipt_key=None, evidence_key=None, triage_status="skipped", triage_outbox_id=None, offer_id=None,
        completed_at=None, usage=None, usd_micros=None, queued_at=stamp if reserve else None,
    )
    idem = new_item(*keys.idempotency(member_id, KIND_RESEARCH, approval_id), stamp, payload_hash=scope_hash,
                    job_id=job_id, kind=KIND_RESEARCH)
    guard = new_item(*_execution_key(parent_job["job_id"]), stamp, execution_id=job_id, approval_id=approval_id,
                     parent_job_id=parent_job["job_id"], member_id=member_id)
    request = {"job_id": job_id, "kind": KIND_RESEARCH, "member_id": member_id, "parent_job_id": parent_job["job_id"],
               "session_id": job["session_id"], "turn": job["turn"], "approval_id": approval_id, "scope": scope_record,
               "scope_hash": scope_hash, "budget_usd_micros": budget_usd_micros, "question": question,
               "policy_revision": policy_revision, "created_at": stamp}
    operations = (Put(idem), Put(job), Put(guard), *_pointer_puts(job, stamp), *budget_ops, *outbox_ops)
    return JobPlan(job=job, outbox=outbox_ops[0].item if outbox_ops else None,  # type: ignore[union-attr]
                   reservation=reservation, request=request, operations=operations)


def create_research_job(table: TablePort, receipts: ReceiptWriter, *, parent_job: Mapping[str, Any], member_id: str,
                        approval_id: str, scope: Mapping[str, Any], budget_usd_micros: int,
                        now: datetime | None = None, policy_revision: str = POLICY_REVISION) -> dict[str, Any]:
    """Commit ``plan_research_job`` in one transaction and write its ``request.json``; idempotent per approval."""
    member_id = _identifier(member_id, "member_id")
    approval_id = _identifier(approval_id, "approval_id")
    if not isinstance(scope, Mapping):
        raise ValueError("scope must be a JSON object naming the question, targets and new pages")
    scope_hash = digest(dict(scope))
    existing = _existing_job(table, member_id, KIND_RESEARCH, approval_id, scope_hash)
    if existing is not None:
        return existing
    last_error: ConditionFailed | None = None
    for attempt in range(TRANSACTION_ATTEMPTS):
        _pause_before(attempt)
        plan = plan_research_job(table, parent_job=parent_job, member_id=member_id, approval_id=approval_id, scope=scope,
                                 budget_usd_micros=budget_usd_micros, now=now, policy_revision=policy_revision)
        try:
            table.transact(list(plan.operations))
        except ConditionFailed as exc:
            last_error = exc
            existing = _existing_job(table, member_id, KIND_RESEARCH, approval_id, scope_hash)
            if existing is not None:
                return existing
            continue
        receipts.put_json(plan.job["request_key"], plan.request)
        return plan.job
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------

def _moment(now: datetime | None) -> datetime:
    moment = now or datetime.now(UTC)
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    return moment


def _day(created_at: str) -> str:
    return created_at[:10]


def _amount(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer number of micro-USD, got {value!r}")
    return value


def _require_job(table: TablePort, job_id: str) -> dict[str, Any]:
    job = table.get(*keys.job(job_id)) if isinstance(job_id, str) and job_id else None
    if job is None:
        raise NotFound(f"no job {job_id}")
    return job


def _running_job(table: TablePort, job_id: str) -> dict[str, Any]:
    job = _require_job(table, job_id)
    if job["status"] != RUNNING:
        raise InvalidTransition(f"job {job_id} is {job['status']}, not running")
    return job


def _receipt_under_job(key: Any, job_id: str, name: str) -> None:
    prefix = f"{RECEIPT_PREFIX}{job_id}/"
    if not isinstance(key, str) or not key.startswith(prefix) or ".." in key:
        raise ValueError(f"{name} must be a receipt under {prefix}")


def _billed(table: TablePort, job: Mapping[str, Any]) -> int:
    if not job.get("reservation_id"):
        return 0
    return int(lab_budget.job_balance(table, job["job_id"])["settled_micros"])


def _orphaned_micros(table: TablePort, job: Mapping[str, Any]) -> int:
    """Micros still reserved inside the job scope: attempt reservations no attempt resolved."""
    if not job.get("reservation_id"):
        return 0
    return int(lab_budget.job_balance(table, job["job_id"])["reserved_micros"])


def _close_job_reservation(table: TablePort, job: Mapping[str, Any], moment: datetime, *,
                           billed: int | None = None) -> tuple[list[Operation], dict[str, Any]]:
    """Settle the job reservation with the ledger's bill, or flag it unknown when an attempt is still held.

    Returns the ledger operations and the attributes the job record takes with them:
    ``reservation_status`` (``settled`` or ``unknown``) and, for an orphaned attempt,
    ``orphaned_reserved_micros``. The period scopes keep an unknown reservation's hold until an
    operator settles it against the provider's record.
    """
    reservation_id = job.get("reservation_id")
    if not reservation_id:
        return [], {}
    balance = lab_budget.job_balance(table, job["job_id"])
    orphaned = int(balance["reserved_micros"])
    if orphaned > 0:
        plan = lab_budget.plan_mark_unknown(table, reservation_id, reason=ORPHAN_REASON, now=moment)
        return list(plan.operations), {"reservation_status": "unknown", "orphaned_reserved_micros": orphaned}
    amount = int(balance["settled_micros"]) if billed is None else billed
    plan = lab_budget.plan_settle(table, reservation_id, amount, now=moment)
    return list(plan.operations), {"reservation_status": "settled"}


def _pointer_status_updates(table: TablePort, job: Mapping[str, Any], status: str) -> list[Operation]:
    updates: list[Operation] = []
    for pk, sk in (keys.member_job(job["member_id"], job["created_at"], job["job_id"]),
                   keys.day_record(_day(job["created_at"]), job["created_at"], job["job_id"])):
        pointer = table.get(pk, sk)
        if pointer is not None and pointer.get("status") != status:
            updates.append(Update(pk, sk, pointer["revision"], {"status": status}))
    return updates


def _outbox_puts(outbox_id: str, kind: str, job_id: str, stamp: str) -> list[Operation]:
    row = new_item(*keys.outbox(outbox_id), stamp, outbox_id=outbox_id, kind=kind, job_id=job_id,
                   status=OUTBOX_PENDING, attempts=0, sent_at=None)
    pointer = new_item(*keys.outbox_pending(stamp, outbox_id), stamp, outbox_id=outbox_id, kind=kind, job_id=job_id,
                       status=OUTBOX_PENDING)
    return [Put(row), Put(pointer)]


def _outbox_status_updates(table: TablePort, outbox_id: str | None, status: str, stamp: str, *,
                           job_id: str | None = None) -> list[Operation]:
    """Move the outbox row to ``status`` and remove its pending pointer, so ``PENDING#`` holds live rows only."""
    if not outbox_id:
        return []
    row = table.get(*keys.outbox(outbox_id))
    if row is None:
        raise NotFound(f"no outbox row {outbox_id}")
    if job_id is not None and row.get("job_id") != job_id:
        raise InvalidTransition(f"outbox {outbox_id} does not deliver job {job_id}", code="invalid_outbox")
    operations: list[Operation] = []
    if row.get("status") != status:
        operations.append(Update(*keys.outbox(outbox_id), row["revision"], {"status": status, f"{status}_at": stamp}))
    pointer = table.get(*keys.outbox_pending(row["created_at"], outbox_id))
    if pointer is not None:
        operations.append(Delete(*keys.outbox_pending(row["created_at"], outbox_id), pointer["revision"]))
    return operations
