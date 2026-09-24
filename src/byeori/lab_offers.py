"""Synthesis offers and the student's consent record (docs/LAB-QUESTION-WORKFLOW.md, P3).

The triage worker calls ``issue`` when a verdict passed the 0.99 cutoff and the wiki-scope check
found a concrete supplement or gap. The offer is a fixed proposal: its ``hash`` covers the offer
id, the job, the kind, the targets, the Korean message, the policy revision and the expiry, and
the student's client must echo ``revision`` and ``hash`` back, so a decision can never apply to a
proposal the student did not see. The offer also pins the server research profile and the policy
revision it was issued under; an accept applies those pinned values, never the module constants
current at accept time, and an offer whose policy revision is no longer current is recorded as
``stale`` and refused (design section 8: the values and the price-table revision are fixed on the
issued offer).

One session is not asked the same thing twice (design section 6). ``issue`` keeps a
``SESSION#{session_id}/OFFER#{created_at}#{offer_id}`` pointer per offer and, before creating a new
one, looks the session's offers up: an ``offered`` or ``declined`` one with the same kind and
targets, or the same verdict input hash, suppresses the new offer and ``issue`` returns ``None``
without writing; ``suppressing_offer`` names it.

``respond`` is the student's only action on an offer. Ownership is checked before anything else
and a non-owner gets ``NotFound``, never ``Forbidden``, so the existence of another student's
offer does not leak. A decline records the decision and nothing else. An accept is one
transaction: the offer becomes ``accepted`` with its ``execution_id``, an ``APPROVAL`` of kind
``student_consent`` is created under the pinned research profile, and the research job with its
budget reservation and outbox row from ``lab_jobs.plan_research_job`` lands in the same write.
When the period cap cannot hold the reservation the consent is still recorded: the same
transaction plans the research job as ``paused_budget`` without a reservation or outbox row, and
the response says so in ``research_status``. When another path (a professor's approval) already
started the one research execution of the parent job, the accept links the consent to that
execution instead of planning a second one. A replay with the same ``request_id`` returns the same
result; a second accept with another ``request_id`` returns the same ``execution_id`` and writes
nothing.

Receipts ``offer-{offer_id}.json`` and ``approval-{approval_id}.json`` are written under the
original answer job's prefix through ``lab_store.ReceiptWriter`` only, after the transaction that
references their key and sha256; a replay writes a missing receipt once and tolerates 412.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from botocore.exceptions import ClientError

from byeori import lab_budget, lab_jobs
from byeori.evidence_packet import page_key
from byeori.lab_jobs import TRANSACTION_ATTEMPTS, IdempotencyConflict, NotFound
from byeori.lab_policy import (
    APPROVAL_TTL_SECONDS,
    OFFER_TEMPLATES,
    OFFER_TTL_SECONDS,
    POLICY_REVISION,
    RESEARCH_PROFILE,
    passes_cutoff,
)
from byeori.lab_store import (
    ConditionFailed,
    Operation,
    Put,
    ReceiptWriter,
    StoreError,
    TablePort,
    Update,
    digest,
    keys,
    new_id,
    new_item,
    now_iso,
    receipt_key,
)

__all__ = [
    "ACCEPT", "DECISIONS", "DECLINE", "Expired", "IdempotencyConflict", "KIND_OFFER_RESPONSE", "LINKED_NOTE",
    "MAX_TARGETS", "NotFound", "OFFER_KINDS", "OFFER_STATUSES", "RevisionConflict", "STALE_MESSAGE", "compose_message",
    "is_expired", "issue", "offer_for_job", "offer_view", "respond", "suppressing_offer",
]

NEW_SYNTHESIS, SUPPLEMENT_EXISTING = "new_synthesis", "supplement_existing"
OFFER_KINDS = frozenset({NEW_SYNTHESIS, SUPPLEMENT_EXISTING})

OFFERED, ACCEPTED, DECLINED, EXPIRED, STALE = "offered", "accepted", "declined", "expired", "stale"
OFFER_STATUSES = frozenset({OFFERED, ACCEPTED, DECLINED, EXPIRED, STALE})
# An offer in one of these statuses keeps the same proposal from being offered again in its session.
# An accepted offer suppresses too: the research it started is the answer to the same proposal,
# and a second execution under another parent job would spend the budget twice.
SUPPRESSING_STATUSES = frozenset({OFFERED, ACCEPTED, DECLINED})

ACCEPT, DECLINE = "accept", "decline"
DECISIONS = frozenset({ACCEPT, DECLINE})
_STATUS_OF_DECISION = {ACCEPT: ACCEPTED, DECLINE: DECLINED}

KIND_OFFER_RESPONSE = "offer_response"     # idempotency kind: IDEMP#{member_id}#offer_response#{request_id}
KIND_STUDENT_CONSENT = "student_consent"
MAX_TARGETS = 20
MAX_HASH_CHARS = 128
SESSION_OFFER_PAGES = 5                    # suppressing_offer reads at most this many 100-row pointer pages
STALE_MESSAGE = "offer was issued under an earlier policy; ask again"
LINKED_NOTE = "linked to an execution started by another path"

# The offer hash covers exactly these attributes (plan Task 9).
HASH_FIELDS = ("offer_id", "job_id", "kind", "targets", "message", "policy_revision", "expires_at")
VIEW_FIELDS = ("offer_id", "revision", "hash", "kind", "message", "targets", "expires_at", "status")
# Store-managed attributes never enter a receipt body; neither do attributes a later decision changes,
# so a replay after a crash reproduces the bytes the record's receipt_sha256 names.
_STORE_FIELDS = frozenset({"pk", "sk", "revision", "updated_at", "receipt_key", "receipt_sha256"})
_OFFER_DECISION_FIELDS = frozenset({"status", "decision", "decision_at", "decided_revision", "execution_id",
                                    "approval_id", "expired_at", "stale_at", "research_status"})
_APPROVAL_STATE_FIELDS = frozenset({"status"})
ROLES = frozenset({"student", "admin"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RevisionConflict(StoreError):
    """The echoed revision or hash does not name the current proposal, the offer was already decided, or it is stale."""

    code = "revision_conflict"


class Expired(StoreError):
    """The offer's validity has passed; it is recorded as ``expired`` and cannot be decided."""

    code = "expired"


# ---------------------------------------------------------------------------------------------
# Message and views
# ---------------------------------------------------------------------------------------------

def compose_message(kind: str, targets: list[str]) -> str:
    """The verbatim design sentence for ``kind``, the named targets when any, then the consent note.

    A supplement lists the pages it would supplement; a new synthesis lists the pages it would
    create when the triage proposed them. Every message ends with the sentence saying that
    acceptance starts an asynchronous research run under the server research profile.
    """
    if kind not in OFFER_KINDS:
        raise ValueError(f"offer kind must be one of {sorted(OFFER_KINDS)}, not {kind!r}")
    lines = [OFFER_TEMPLATES[kind]]
    if targets:
        label = OFFER_TEMPLATES["existing_targets_label" if kind == SUPPLEMENT_EXISTING else "new_targets_label"]
        lines.append(f"{label} {', '.join(targets)}")
    lines.append(OFFER_TEMPLATES["consent_note"])
    return "\n".join(lines)


def offer_view(offer: Mapping[str, Any]) -> dict[str, Any]:
    """The client-facing projection: what the student sees and must echo back, nothing internal."""
    return {name: offer.get(name) for name in VIEW_FIELDS}


def is_expired(offer: Mapping[str, Any], now: datetime) -> bool:
    """True once ``now`` reaches ``expires_at`` (inclusive) or the offer was recorded as expired."""
    if offer.get("status") == EXPIRED:
        return True
    return datetime.fromisoformat(offer["expires_at"]) <= _moment(now)


def offer_for_job(table: TablePort, job_id: str) -> dict[str, Any] | None:
    """The job's current offer (the most recently issued one), or ``None``."""
    if not isinstance(job_id, str) or not job_id:
        return None
    pointers, _next = table.query(f"JOB#{job_id}", sk_prefix="OFFER#", limit=100)
    offers = [table.get(*keys.offer(pointer["offer_id"])) for pointer in pointers if pointer.get("offer_id")]
    found = [offer for offer in offers if offer is not None]
    if not found:
        return None
    return max(found, key=lambda offer: (offer.get("created_at", ""), offer["offer_id"]))


def suppressing_offer(table: TablePort, session_id: str | None, kind: str, targets: list[str],
                      input_hash: str | None, *, now: datetime | None = None) -> dict[str, Any] | None:
    """The session's earlier offer that makes a new one for this proposal redundant, or ``None``.

    Walks the ``SESSION#{session_id}/OFFER#`` pointers ``issue`` keeps (at most
    ``SESSION_OFFER_PAGES`` pages of 100) and returns the most recent offer still ``offered`` or
    ``declined`` whose kind and sorted targets equal the given ones, or whose verdict
    ``input_hash`` equals ``input_hash`` when one is given. Accepted, expired and stale offers
    never suppress. With ``now``, an ``offered`` offer that has passed its ``expires_at`` does not
    suppress either, so a student who returns after the TTL can be asked again. A missing
    ``session_id`` returns ``None``: the check is per session only.
    """
    if not isinstance(session_id, str) or not session_id:
        return None
    if kind not in OFFER_KINDS:
        raise ValueError(f"offer kind must be one of {sorted(OFFER_KINDS)}, not {kind!r}")
    wanted_targets = sorted(_targets(targets))
    wanted_hash = input_hash if isinstance(input_hash, str) and input_hash else None
    moment = _moment(now) if now is not None else None
    found: dict[str, Any] | None = None
    cursor: str | None = None
    session_pk = keys.session(session_id)[0]
    for _page in range(SESSION_OFFER_PAGES):
        pointers, cursor = table.query(session_pk, sk_prefix="OFFER#", limit=100, start_after=cursor)
        for pointer in pointers:
            if pointer.get("status") not in SUPPRESSING_STATUSES or not pointer.get("offer_id"):
                continue
            same_proposal = pointer.get("kind") == kind and sorted(pointer.get("targets") or []) == wanted_targets
            same_input = wanted_hash is not None and pointer.get("input_hash") == wanted_hash
            if not (same_proposal or same_input):
                continue
            offer = table.get(*keys.offer(pointer["offer_id"]))
            if offer is None or offer.get("status") not in SUPPRESSING_STATUSES:
                continue
            if moment is not None and offer["status"] == OFFERED and is_expired(offer, moment):
                continue
            found = offer  # pointers sort by created_at, so the last match is the most recent
        if cursor is None:
            break
    return found


# ---------------------------------------------------------------------------------------------
# Issuing
# ---------------------------------------------------------------------------------------------

def issue(table: TablePort, receipts: ReceiptWriter, job: Mapping[str, Any], verdict: Mapping[str, Any], kind: str,
          targets: list[str], scope_check: Mapping[str, Any], now: datetime | None = None, *,
          policy_revision: str = POLICY_REVISION) -> dict[str, Any] | None:
    """Issue one offer for an answer job whose verdict passed the cutoff; idempotent per job.

    One transaction puts ``OFFER#{id}/META`` (revision 1), the ``JOB#{job_id}/OFFER#{id}`` pointer
    and, for a job with a session, the ``SESSION#{session_id}/OFFER#{created_at}#{id}`` pointer, and
    sets ``offer_id`` on the job, so the job's revision advances by one: re-read it before
    ``lab_jobs.set_triage_status``. Then ``offer-{id}.json`` is written under the job's receipt
    prefix. A job that already has an offer gets that offer back unchanged (a missing receipt is
    written once). When ``suppressing_offer`` finds an earlier offer of the same session for the
    same proposal or the same verdict input, nothing is written and ``None`` is returned; the
    caller records the suppression instead of an offer. The offer snapshots the current
    ``RESEARCH_PROFILE`` and ``policy_revision``; ``respond`` applies those pinned values. The
    verdict must pass ``lab_policy.passes_cutoff``.
    """
    moment = _moment(now)
    stamp = now_iso(moment)
    job_id = _job_id_of(job)
    if job.get("kind") != lab_jobs.KIND_ANSWER:
        raise ValueError("offers are issued for answer jobs only")
    if not isinstance(verdict, Mapping) or not passes_cutoff(verdict.get("probabilities")):
        raise ValueError("an offer needs a verdict whose review_candidate probability passes the cutoff")
    if not isinstance(scope_check, Mapping):
        raise ValueError("scope_check must be the wiki-scope check record (a JSON object)")
    targets = _targets(targets)
    if kind not in OFFER_KINDS:
        raise ValueError(f"offer kind must be one of {sorted(OFFER_KINDS)}, not {kind!r}")
    if kind == SUPPLEMENT_EXISTING and not targets:
        raise ValueError("a supplement_existing offer names at least one page to supplement")

    existing = offer_for_job(table, job_id)
    if existing is not None:
        _write_once(receipts, existing["receipt_key"], _offer_receipt(existing))
        return existing
    stored = _require_job(table, job_id)
    if stored.get("kind") != lab_jobs.KIND_ANSWER:
        raise ValueError("offers are issued for answer jobs only")
    input_hash = verdict.get("input_hash")
    if suppressing_offer(table, stored.get("session_id"), kind, targets, input_hash, now=moment) is not None:
        return None

    offer_id = new_id()
    message = compose_message(kind, targets)
    expires_at = now_iso(moment + timedelta(seconds=OFFER_TTL_SECONDS))
    proposal = {"offer_id": offer_id, "job_id": job_id, "kind": kind, "targets": targets, "message": message,
                "policy_revision": policy_revision, "expires_at": expires_at}
    last_error: ConditionFailed | None = None
    for _attempt in range(TRANSACTION_ATTEMPTS):
        stored = _require_job(table, job_id)
        if stored.get("kind") != lab_jobs.KIND_ANSWER:
            raise ValueError("offers are issued for answer jobs only")
        if stored.get("offer_id"):
            existing = table.get(*keys.offer(stored["offer_id"]))
            if existing is not None:
                _write_once(receipts, existing["receipt_key"], _offer_receipt(existing))
                return existing
        session_id = stored.get("session_id")
        offer = new_item(
            *keys.offer(offer_id), stamp,
            offer_id=offer_id, job_id=job_id, member_id=stored["member_id"], session_id=session_id,
            kind=kind, message=message, targets=targets, scope_check=dict(scope_check), hash=digest(proposal),
            policy_revision=policy_revision, expires_at=expires_at, status=OFFERED, offered_at=stamp,
            decision=None, decision_at=None, decided_revision=None, execution_id=None, approval_id=None,
            research_status=None, research_profile=dict(RESEARCH_PROFILE), verdict=_verdict_summary(verdict),
            receipt_key=receipt_key(job_id, f"offer-{offer_id}.json"),
        )
        offer["receipt_sha256"] = digest(_offer_receipt(offer))
        pointer = new_item(*keys.job_offer(job_id, offer_id), stamp, offer_id=offer_id, job_id=job_id, kind=kind,
                           member_id=stored["member_id"], status=OFFERED, execution_id=None)
        operations: list[Operation] = [Put(offer), Put(pointer)]
        if session_id:
            operations.append(Put(new_item(*_session_offer_key(session_id, stamp, offer_id), stamp, offer_id=offer_id,
                                           job_id=job_id, member_id=stored["member_id"], kind=kind, targets=targets,
                                           input_hash=input_hash, status=OFFERED, execution_id=None)))
        operations.append(Update(*keys.job(job_id), stored["revision"], {"offer_id": offer_id}))
        try:
            table.transact(operations)
        except ConditionFailed as exc:
            last_error = exc
            continue
        _write_once(receipts, offer["receipt_key"], _offer_receipt(offer))
        return table.get(*keys.offer(offer_id))
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# Responding
# ---------------------------------------------------------------------------------------------

def respond(table: TablePort, receipts: ReceiptWriter, member: Any, body: Mapping[str, Any],
            now: datetime | None = None) -> dict[str, Any]:
    """Record the owner's ``accept`` or ``decline`` of an offer and return its result.

    The result is ``{offer_id, status, execution_id, approval_id, research_status}``;
    ``research_status`` is the research job's status when the consent was recorded (``queued``,
    ``paused_budget`` when the period cap could not hold the reservation, the linked execution's
    status when another path had already started it) and ``None`` for a decline.

    Checks run in this order: body shape (``ValueError``), ownership (``NotFound`` for anyone but
    the owner), the idempotency key ``IDEMP#{member}#offer_response#{request_id}`` (a replay
    returns the recorded result, a different payload raises ``IdempotencyConflict``), then the
    offer state: ``expired`` or past ``expires_at`` raises ``Expired`` (recording the status once),
    ``stale`` raises ``RevisionConflict``, an already decided offer returns its result when the
    same decision and hash arrive again and raises ``RevisionConflict`` otherwise, and a live offer
    requires the exact ``revision`` and ``hash`` or raises ``RevisionConflict``. A live offer whose
    ``policy_revision`` is not the current ``POLICY_REVISION`` is recorded as ``stale`` in one
    update and refused with ``RevisionConflict`` (``STALE_MESSAGE``): the student asks again and
    gets an offer under the current policy. Contention is re-read and re-planned up to
    ``TRANSACTION_ATTEMPTS`` times. A budget refusal never propagates from an accept; the consent
    is recorded with the research job ``paused_budget``.
    """
    member_id, _role = _identity(member)
    moment = _moment(now)
    request = _response_request(body)
    approval_id = new_id()     # fixed for this call so re-plans never leave a second approval
    last_error: ConditionFailed | None = None
    for _attempt in range(TRANSACTION_ATTEMPTS):
        offer = table.get(*keys.offer(request["offer_id"]))
        if offer is None or offer.get("member_id") != member_id:
            raise NotFound(f"no offer {request['offer_id']}")
        idem = table.get(*keys.idempotency(member_id, KIND_OFFER_RESPONSE, request["request_id"]))
        if idem is not None:
            if idem.get("payload_hash") != request["payload_hash"] or idem.get("offer_id") != offer["offer_id"]:
                raise IdempotencyConflict(f"request {request['request_id']} was already used with a different payload")
            return _replayed(table, receipts, offer, idem)
        if offer["status"] == EXPIRED:
            raise Expired(f"offer {offer['offer_id']} expired at {offer['expires_at']}")
        if offer["status"] == STALE:
            raise RevisionConflict(STALE_MESSAGE)
        if offer["status"] in (ACCEPTED, DECLINED):
            if request["hash"] == offer["hash"] and _STATUS_OF_DECISION[request["decision"]] == offer["status"]:
                return _result(offer)
            raise RevisionConflict(f"offer {offer['offer_id']} was already {offer['status']}")
        if is_expired(offer, moment):
            _close_unanswered(table, offer, EXPIRED, "expired_at", now_iso(moment))
            raise Expired(f"offer {offer['offer_id']} expired at {offer['expires_at']}")
        if request["revision"] != offer["revision"] or request["hash"] != offer["hash"]:
            raise RevisionConflict(f"offer {offer['offer_id']} is at revision {offer['revision']} with another hash")
        if offer.get("policy_revision") != POLICY_REVISION:
            _close_unanswered(table, offer, STALE, "stale_at", now_iso(moment))
            raise RevisionConflict(STALE_MESSAGE)
        try:
            if request["decision"] == DECLINE:
                return _decline(table, offer, member_id, request, moment)
            return _accept(table, receipts, offer, member_id, request, approval_id, moment)
        except ConditionFailed as exc:
            last_error = exc
            continue
    raise last_error  # type: ignore[misc]


def _decline(table: TablePort, offer: dict[str, Any], member_id: str, request: dict[str, Any],
             moment: datetime) -> dict[str, Any]:
    stamp = now_iso(moment)
    decision = {"status": DECLINED, "decision": DECLINE, "decision_at": stamp, "decided_revision": offer["revision"]}
    operations: list[Operation] = [
        Update(*keys.offer(offer["offer_id"]), offer["revision"], decision),
        *_pointer_updates(table, offer, {"status": DECLINED}),
        Put(_idempotency_row(member_id, request, offer, stamp, approval_id=None, execution_id=None)),
    ]
    table.transact(operations)
    return _result(table.get(*keys.offer(offer["offer_id"])))


def _accept(table: TablePort, receipts: ReceiptWriter, offer: dict[str, Any], member_id: str, request: dict[str, Any],
            approval_id: str, moment: datetime) -> dict[str, Any]:
    """Record the consent under the offer's pinned profile and start, pause or link the research execution.

    The parent job's ``EXECUTION`` guard is read first: when another path already holds it the
    consent links to that execution and plans nothing. Otherwise the research job is planned with
    its reservation; a ``BudgetExceeded`` from the period scopes re-plans it as ``paused_budget``
    without reservation or outbox row, so the consent is never lost to a cap. Should the guard
    land between the read and the transaction, the transaction fails with ``ConditionFailed`` and
    the caller's retry reads the winner and links.
    """
    stamp = now_iso(moment)
    job = table.get(*keys.job(offer["job_id"]))
    if job is None:
        raise NotFound(f"no job {offer['job_id']} for offer {offer['offer_id']}")
    scope = _research_scope(offer, job)
    profile = offer["research_profile"]     # pinned at issue; never RESEARCH_PROFILE at accept time
    guard = lab_jobs.existing_execution(table, job["job_id"])
    if guard is not None:
        return _accept_linked(table, receipts, offer, job, member_id, request, approval_id, moment, guard, scope)
    budget_refusal: dict[str, Any] | None = None
    try:
        plan = lab_jobs.plan_research_job(table, parent_job=job, member_id=member_id, approval_id=approval_id, scope=scope,
                                          budget_usd_micros=profile["budget_usd_micros"], now=moment,
                                          policy_revision=offer["policy_revision"])
        research_status = lab_jobs.QUEUED
    except lab_budget.BudgetExceeded as exc:
        plan = lab_jobs.plan_research_job(table, parent_job=job, member_id=member_id, approval_id=approval_id, scope=scope,
                                          budget_usd_micros=profile["budget_usd_micros"], now=moment,
                                          policy_revision=offer["policy_revision"], reserve=False,
                                          status=lab_jobs.PAUSED_BUDGET)
        research_status = lab_jobs.PAUSED_BUDGET
        budget_refusal = {"scope": exc.scope, "requested_micros": exc.requested, "available_micros": exc.available,
                          "reason": str(exc), "refused_at": stamp}
    execution_id = plan.job["job_id"]
    approval = _approval(offer, job, member_id, request, approval_id, scope, moment, execution_id=execution_id,
                         research_status=research_status, budget_refusal=budget_refusal)
    operations: list[Operation] = [
        *_accept_operations(table, offer, member_id, request, approval, execution_id, research_status, stamp),
        *plan.operations,
    ]
    table.transact(operations)
    _write_once(receipts, plan.job["request_key"], plan.request)
    _write_once(receipts, approval["receipt_key"], _approval_receipt(approval))
    return _result(table.get(*keys.offer(offer["offer_id"])))


def _accept_linked(table: TablePort, receipts: ReceiptWriter, offer: dict[str, Any], job: dict[str, Any], member_id: str,
                   request: dict[str, Any], approval_id: str, moment: datetime, guard: Mapping[str, Any],
                   scope: dict[str, Any]) -> dict[str, Any]:
    """Record the consent against the execution another path already started; no job, reservation or outbox."""
    stamp = now_iso(moment)
    execution_id = guard["execution_id"]
    execution = table.get(*keys.job(execution_id))
    research_status = execution.get("status") if execution is not None else None
    approval = _approval(offer, job, member_id, request, approval_id, scope, moment, execution_id=execution_id,
                         research_status=research_status, note=LINKED_NOTE, linked_approval_id=guard.get("approval_id"))
    table.transact(_accept_operations(table, offer, member_id, request, approval, execution_id, research_status, stamp))
    _write_once(receipts, approval["receipt_key"], _approval_receipt(approval))
    return _result(table.get(*keys.offer(offer["offer_id"])))


def _accept_operations(table: TablePort, offer: dict[str, Any], member_id: str, request: dict[str, Any],
                       approval: dict[str, Any], execution_id: str, research_status: str | None,
                       stamp: str) -> list[Operation]:
    """The offer decision, its pointers, the approval and the idempotency row every accept commits together."""
    decision = {"status": ACCEPTED, "decision": ACCEPT, "decision_at": stamp, "decided_revision": offer["revision"],
                "execution_id": execution_id, "approval_id": approval["approval_id"], "research_status": research_status}
    return [
        Update(*keys.offer(offer["offer_id"]), offer["revision"], decision),
        *_pointer_updates(table, offer, {"status": ACCEPTED, "execution_id": execution_id}),
        Put(approval),
        Put(_idempotency_row(member_id, request, offer, stamp, approval_id=approval["approval_id"],
                             execution_id=execution_id)),
    ]


def _approval(offer: Mapping[str, Any], job: Mapping[str, Any], member_id: str, request: Mapping[str, Any],
              approval_id: str, scope: dict[str, Any], moment: datetime, *, execution_id: str,
              research_status: str | None, note: str | None = None, budget_refusal: dict[str, Any] | None = None,
              linked_approval_id: str | None = None) -> dict[str, Any]:
    """The ``student_consent`` approval: the offer's pinned profile and policy revision, bound to one execution."""
    stamp = now_iso(moment)
    profile = offer["research_profile"]
    approval = new_item(
        *keys.approval(approval_id), stamp,
        approval_id=approval_id, kind=KIND_STUDENT_CONSENT, offer_id=offer["offer_id"], candidate_id=None,
        job_id=job["job_id"], proposal_revision=offer["revision"], proposal_hash=offer["hash"], approved_by=member_id,
        policy_revision=offer["policy_revision"], scope=scope, budget_usd_micros=profile["budget_usd_micros"],
        model_id=None, max_calls=profile["max_calls"], reread=profile["reread"],
        expires_at=now_iso(moment + timedelta(seconds=APPROVAL_TTL_SECONDS)), execution_id=execution_id, status="active",
        approved_at=stamp, request_id=request["request_id"], research_status=research_status, note=note,
        budget_refusal=budget_refusal, linked_approval_id=linked_approval_id,
        receipt_key=receipt_key(job["job_id"], f"approval-{approval_id}.json"),
    )
    approval["receipt_sha256"] = digest(_approval_receipt(approval))
    return approval


def _replayed(table: TablePort, receipts: ReceiptWriter, offer: dict[str, Any], idem: Mapping[str, Any]) -> dict[str, Any]:
    """The recorded outcome of an earlier request; an accept's approval receipt is written if it is missing."""
    if idem.get("decision") == ACCEPT and idem.get("approval_id"):
        approval = table.get(*keys.approval(idem["approval_id"]))
        if approval is not None:
            _write_once(receipts, approval["receipt_key"], _approval_receipt(approval))
    return _result(offer)


def _close_unanswered(table: TablePort, offer: dict[str, Any], status: str, stamp_field: str, stamp: str) -> None:
    """Record ``expired`` or ``stale`` on the offer and its pointers once; a concurrent recorder is fine."""
    operations: list[Operation] = [Update(*keys.offer(offer["offer_id"]), offer["revision"], {"status": status, stamp_field: stamp}),
                                   *_pointer_updates(table, offer, {"status": status})]
    try:
        table.transact(operations)
    except ConditionFailed:
        pass


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------

def _result(offer: Mapping[str, Any]) -> dict[str, Any]:
    return {"offer_id": offer["offer_id"], "status": offer["status"], "execution_id": offer.get("execution_id"),
            "approval_id": offer.get("approval_id"), "research_status": offer.get("research_status")}


def _research_scope(offer: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    """The scope the student consented to: the question, the shown targets, and the offer text as the note."""
    targets = list(offer.get("targets") or [])
    supplement = offer["kind"] == SUPPLEMENT_EXISTING
    return {"question": job["question"], "targets": targets if supplement else [], "new_pages": [] if supplement else targets,
            "note": offer["message"], "kind": offer["kind"], "offer_id": offer["offer_id"]}


def _idempotency_row(member_id: str, request: Mapping[str, Any], offer: Mapping[str, Any], stamp: str, *,
                     approval_id: str | None, execution_id: str | None) -> dict[str, Any]:
    return new_item(*keys.idempotency(member_id, KIND_OFFER_RESPONSE, request["request_id"]), stamp,
                    payload_hash=request["payload_hash"], kind=KIND_OFFER_RESPONSE, offer_id=offer["offer_id"],
                    job_id=offer["job_id"], decision=request["decision"], approval_id=approval_id, execution_id=execution_id)


def _session_offer_key(session_id: str, created_at: str, offer_id: str) -> tuple[str, str]:
    """``SESSION#{session_id}`` / ``OFFER#{created_at}#{offer_id}``: the session's offers in issue order."""
    return keys.session(session_id)[0], f"OFFER#{created_at}#{offer_id}"


def _pointer_keys(offer: Mapping[str, Any]) -> list[tuple[str, str]]:
    found = [keys.job_offer(offer["job_id"], offer["offer_id"])]
    if offer.get("session_id") and offer.get("created_at"):
        found.append(_session_offer_key(offer["session_id"], offer["created_at"], offer["offer_id"]))
    return found


def _pointer_updates(table: TablePort, offer: Mapping[str, Any], changes: dict[str, Any]) -> list[Operation]:
    """Revision-checked updates of the offer's job pointer and session pointer, whichever exist."""
    operations: list[Operation] = []
    for pk, sk in _pointer_keys(offer):
        pointer = table.get(pk, sk)
        if pointer is not None:
            operations.append(Update(pk, sk, pointer["revision"], changes))
    return operations


def _verdict_summary(verdict: Mapping[str, Any]) -> dict[str, Any]:
    names = ("input_hash", "choice", "probabilities", "confidence", "cutoff", "passed_cutoff", "model", "policy_revision")
    return {name: verdict.get(name) for name in names}


def _offer_receipt(offer: Mapping[str, Any]) -> dict[str, Any]:
    """The proposal snapshot: the offer without store-managed and decision attributes."""
    return {name: value for name, value in offer.items() if name not in _STORE_FIELDS | _OFFER_DECISION_FIELDS}


def _approval_receipt(approval: Mapping[str, Any]) -> dict[str, Any]:
    """The consent record: the approval without store-managed attributes and its mutable status."""
    return {name: value for name, value in approval.items() if name not in _STORE_FIELDS | _APPROVAL_STATE_FIELDS}


def _write_once(receipts: ReceiptWriter, key: str, body: Any) -> None:
    """Write a receipt with IfNoneMatch; an object already there (412) is the earlier, identical write."""
    try:
        receipts.put_json(key, body)
    except ClientError as exc:
        error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") if hasattr(exc, "response") else None
        if error.get("Code") != "PreconditionFailed" and status != 412:
            raise


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

def _response_request(body: Any) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise ValueError("the response body must be a JSON object")
    request_id = _identifier(body.get("request_id"), "request_id")
    offer_id = _identifier(body.get("offer_id"), "offer_id")
    revision = body.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("revision must be the offer's positive integer revision")
    proposal_hash = body.get("hash")
    if not isinstance(proposal_hash, str) or not proposal_hash or len(proposal_hash) > MAX_HASH_CHARS:
        raise ValueError("hash must be the offer's hash string")
    decision = body.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    payload = {"offer_id": offer_id, "revision": revision, "hash": proposal_hash, "decision": decision}
    return {"request_id": request_id, **payload, "payload_hash": digest(payload)}


def _targets(targets: Any) -> list[str]:
    if isinstance(targets, (str, bytes)) or not isinstance(targets, (list, tuple)):
        raise ValueError("targets must be a list of published wiki page keys")
    if len(targets) > MAX_TARGETS:
        raise ValueError(f"an offer names at most {MAX_TARGETS} target pages")
    unique: list[str] = []
    for target in targets:
        key = page_key(target)
        if key not in unique:
            unique.append(key)
    return unique


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


def _job_id_of(job: Any) -> str:
    if not isinstance(job, Mapping):
        raise ValueError("job must be the answer job record")
    return _identifier(job.get("job_id"), "job_id")


def _require_job(table: TablePort, job_id: str) -> dict[str, Any]:
    job = table.get(*keys.job(job_id))
    if job is None:
        raise NotFound(f"no job {job_id}")
    return job


def _moment(now: datetime | None) -> datetime:
    moment = now or datetime.now(UTC)
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    return moment
