"""Professor-side review of question records (docs/LAB-QUESTION-WORKFLOW.md, sections 6 and 7).

Every question keeps its record whether or not Jev scored it above the cutoff, whether the
student accepted, declined or ignored an offer, and whether the budget let it run at all. This
module reads those records for the administrator. ``list_question_records`` walks the
``RECORDS#{day}`` pointers of a bounded window and joins each job with its verdict and offers;
``propose_research_from_records`` turns a selection into a research candidate with a hashed
proposal; ``decide_research_candidate`` records the professor's approval or rejection of exactly
that proposal revision; ``run_aggregation_round`` counts one window and stores a ``ROUND``
record without calling Jev or creating candidates.

The 0.99 cutoff is the bar for automatic offers to students; it does not limit what the
professor may select here. Probability filters compare the raw ``review_candidate`` value the
verdict stored, so ``0.9899`` and ``0.99`` sort exactly as ``lab_policy.passes_cutoff`` does.
Records never carry conversation context (it lives only in ``request.json``) or principal ARNs.
An approval commits the candidate, the ``APPROVAL`` record, the research job planned by
``lab_jobs.plan_research_job`` and the idempotency key in one transaction; a second approval of
the same revision returns the existing execution and writes nothing. A question may have one
research execution whichever path started it (design section 7): when a student's consent
already started one, or commits while the professor's transaction is in flight, the approval
links that execution instead of planning a second run and reserving its budget again. When the
period cap cannot hold the research reservation the approval is still recorded, with the
research job ``paused_budget`` and no reservation or outbox row. The only S3 keys written are
``approval-{approval_id}.json`` under the original answer job and ``request.json`` under a
research job this module created, both through ``ReceiptWriter``.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from botocore.exceptions import ClientError

from byeori import lab_jobs
from byeori.evidence_packet import page_key
from byeori.lab_budget import BudgetExceeded
from byeori.lab_jobs import Forbidden, IdempotencyConflict, InvalidTransition, NotFound
from byeori.lab_offers import RevisionConflict
from byeori.lab_policy import APPROVAL_TTL_SECONDS, POLICY_REVISION, RESEARCH_PROFILE
from byeori.lab_store import (
    ConditionFailed,
    Operation,
    Put,
    ReceiptWriter,
    StoreError,
    TablePort,
    Update,
    canonical,
    digest,
    keys,
    new_id,
    new_item,
    now_iso,
    receipt_key,
)

__all__ = [
    "APPROVAL_TTL_SECONDS", "CANDIDATE_STATUSES", "COUNT_KEYS", "DECISIONS", "Forbidden", "IdempotencyConflict",
    "InvalidTransition", "MAX_CANDIDATE_JOBS", "MAX_LIMIT", "MAX_WINDOW_DAYS", "NotFound", "RevisionConflict",
    "decide_research_candidate", "list_question_records", "list_research_candidates",
    "propose_research_from_records", "review_probability", "run_aggregation_round",
]

MAX_WINDOW_DAYS = 62                # one list or round covers at most two months of day pointers
MAX_LIMIT = 200                     # records or candidates returned by one call
PAGE_SIZE = 100                     # pointer rows per table query
MAX_DAY_PAGES = 100                 # pointer pages read for one day before a cursor is handed back
MAX_POINTERS_PER_CALL = 1000        # day pointers one list call examines before it hands back a cursor
MAX_ROUND_POINTERS = 20_000         # day pointers one aggregation round examines
MAX_CANDIDATE_JOBS = 25             # question jobs one candidate may bundle
MAX_OFFERS_PER_JOB = 20             # offer pointers read per job
MAX_SCOPE_KEYS = 50                 # wiki keys per scope list
SCOPE_NOTE_MAX_CHARS = 4000
QUESTION_SUMMARY_CHARS = 500        # question text copied into a candidate's proposal; the job keeps the full text

KIND_CANDIDATE, KIND_DECISION = "candidate", "decision"
KIND_PROFESSOR_APPROVAL = "professor_approval"
LINK_NOTE = "linked to an existing execution"   # approval note when the question already had a research run
PAUSED_NOTE = "research paused until the period cap can hold its reservation"
UNSCORED_VERDICT_STATUSES = frozenset({"skipped", "unavailable"})   # triage verdict statuses that carry a reason
PROPOSED, APPROVED, REJECTED, DEFERRED, STALE = "proposed", "approved", "rejected", "deferred", "stale"
CANDIDATE_STATUSES = frozenset({PROPOSED, APPROVED, REJECTED, DEFERRED, STALE})
DECIDABLE_STATUSES = frozenset({PROPOSED, DEFERRED})
APPROVE, REJECT = "approve", "reject"
DECISIONS = frozenset({APPROVE, REJECT})
SCOPE_FIELDS = frozenset({"question", "targets", "new_pages", "note"})
COUNT_KEYS = ("valid_verdicts", "passed", "offers", "accepted", "declined", "unanswered", "expired", "unavailable",
              "skipped", "executions", "rejected_budget")

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")



# ---------------------------------------------------------------------------------------------
# Question records
# ---------------------------------------------------------------------------------------------

def list_question_records(table: TablePort, *, from_day: str, to_day: str, status: str | None = None,
                          min_probability: float | None = None, max_probability: float | None = None,
                          member_id: str | None = None, include_unscored: bool = True, cursor: str | None = None,
                          limit: int = 100) -> dict[str, Any]:
    """Question records of the window ``from_day``..``to_day`` (inclusive ISO dates, at most 62 days).

    Day pointers are walked in creation order and joined with the job, its ``TRIAGE`` verdict and
    its offers. ``status`` filters on the job status, ``member_id`` on the owner, and the
    probability bounds on the raw ``review_candidate`` value of verdicts that carry probabilities.
    Unscored rows (no verdict, or probabilities ``null`` because triage was skipped, failed or is
    pending) are included only while ``include_unscored`` is true. The result is
    ``{"records", "next_cursor"}``; the cursor is opaque and resumes after the last record
    returned. A call examines at most ``MAX_POINTERS_PER_CALL`` pointers and hands back a cursor
    when the window is not exhausted, so a filter that matches little never runs unbounded.
    """
    days = _days(from_day, to_day)
    if status is not None and status not in lab_jobs.STATUSES:
        raise ValueError(f"status must be one of {sorted(lab_jobs.STATUSES)}, not {status!r}")
    if member_id is not None:
        member_id = _identifier(member_id, "member_id")
    low = None if min_probability is None else _probability(min_probability, "min_probability")
    high = None if max_probability is None else _probability(max_probability, "max_probability")
    if low is not None and high is not None and low > high:
        raise ValueError("min_probability must not exceed max_probability")
    if not isinstance(include_unscored, bool):
        raise ValueError("include_unscored must be true or false")
    limit = _limit(limit)
    start = _decode_record_cursor(cursor, days)
    rows, resume = _scan_records(table, days, start, MAX_POINTERS_PER_CALL)
    records: list[dict[str, Any]] = []
    next_cursor = _encode({"day": resume[0], "sk": resume[1]}) if resume is not None else None
    for index, (day, pointer) in enumerate(rows):
        if not _is_question(pointer) or (member_id is not None and pointer.get("member_id") not in (None, member_id)):
            continue
        job = table.get(*keys.job(pointer["job_id"]))
        if job is None or (member_id is not None and job.get("member_id") != member_id):
            continue
        if status is not None and job.get("status") != status:
            continue
        verdict = lab_jobs.read_verdict(table, job["job_id"])
        probability = review_probability(verdict)
        if probability is None:
            if not include_unscored:
                continue
        elif (low is not None and probability < low) or (high is not None and probability > high):
            continue
        records.append(_record_view(job, verdict, _offers_of(table, job["job_id"])))
        if len(records) >= limit:
            more = index < len(rows) - 1 or resume is not None
            next_cursor = _encode({"day": day, "sk": pointer["sk"]}) if more else None
            break
    return {"records": records, "next_cursor": next_cursor}


def review_probability(verdict: Mapping[str, Any] | None) -> float | None:
    """The raw ``review_candidate`` probability of a verdict, or ``None`` when it is unscored.

    Booleans, strings, NaN and a missing or ``null`` ``probabilities`` mapping all mean unscored;
    a score that does not exist is never read as zero.
    """
    if not isinstance(verdict, Mapping):
        return None
    probabilities = verdict.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return None
    value = probabilities.get("review_candidate")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _record_view(job: Mapping[str, Any], verdict: Mapping[str, Any] | None,
                 offers: list[dict[str, Any]]) -> dict[str, Any]:
    latest = offers[-1] if offers else None
    verdict = verdict or {}
    view = {
        "job_id": job["job_id"], "kind": job.get("kind"), "member_id": job.get("member_id"),
        "created_at": job.get("created_at"), "completed_at": job.get("completed_at"), "status": job.get("status"),
        "question": _bounded(job.get("question")), "standalone_question": _bounded(job.get("standalone_question")),
        "triage_status": job.get("triage_status"), "verdict_status": verdict.get("status"),
        "choice": verdict.get("choice"), "probabilities": verdict.get("probabilities"),
        "confidence": verdict.get("confidence"), "passed_cutoff": verdict.get("passed_cutoff"),
        "candidate_status": verdict.get("candidate_status"),
        "offer_id": latest["offer_id"] if latest else None, "offer_status": latest["status"] if latest else None,
        "execution_id": latest.get("execution_id") if latest else None,
        "usd_micros": job.get("usd_micros"),
    }
    if job.get("status") == lab_jobs.REJECTED_BUDGET:
        view["reason"], view["rejected_scope"] = job.get("reason"), job.get("rejected_scope")
    elif job.get("status") in {lab_jobs.FAILED, lab_jobs.OUTCOME_UNKNOWN}:
        view["reason"], view["error_code"] = job.get("reason"), job.get("error_code")
    # A close records how the job reservation ended; an attempt left held keeps period micros on hold.
    for name in ("reservation_status", "orphaned_reserved_micros"):
        if job.get(name) is not None:
            view[name] = job[name]
    # An unscored verdict keeps why it was not scored (design section 6), distinct from the job's reason.
    if verdict.get("status") in UNSCORED_VERDICT_STATUSES:
        view["triage_reason"], view["triage_error_code"] = verdict.get("reason"), verdict.get("error_code")
    return view


def _is_question(pointer: Mapping[str, Any]) -> bool:
    """Day pointers of research executions share the day index; only answer jobs are question records."""
    return pointer.get("kind", lab_jobs.KIND_ANSWER) == lab_jobs.KIND_ANSWER


def _offers_of(table: TablePort, job_id: str) -> list[dict[str, Any]]:
    """The job's offers, oldest first: pointer status joined with the offer record when it exists."""
    pointers, _next = table.query(f"JOB#{job_id}", sk_prefix="OFFER#", limit=MAX_OFFERS_PER_JOB)
    offers = []
    for pointer in pointers:
        offer_id = pointer.get("offer_id") or pointer["sk"].removeprefix("OFFER#")
        meta = table.get(*keys.offer(offer_id)) or {}
        source = meta or pointer
        offers.append({"offer_id": offer_id, "status": source.get("status"), "created_at": source.get("created_at"),
                       "kind": meta.get("kind"), "execution_id": meta.get("execution_id"),
                       "expires_at": meta.get("expires_at"), "decision_at": meta.get("decision_at")})
    offers.sort(key=lambda offer: (offer["created_at"] or "", offer["offer_id"]))
    return offers


# ---------------------------------------------------------------------------------------------
# Research candidates
# ---------------------------------------------------------------------------------------------

def propose_research_from_records(table: TablePort, receipts: ReceiptWriter, admin_member: Any,
                                  body: Mapping[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Record the administrator's selection of question jobs as a research candidate (revision 1).

    ``body`` carries ``request_id``, ``job_ids`` (1..25 existing jobs of any member, any status,
    any score) and ``scope`` (``question``, ``targets``, ``new_pages``, ``note``). One transaction
    writes the ``CANDIDATE`` record, its ``CANDIDATES`` pointer and the idempotency key
    ``IDEMP#{admin}#candidate#{request_id}``. Existing offers and executions of those jobs are
    linked so the professor never re-runs what a student already started. The proposal hash is
    the digest of ``{job_ids sorted, scope, policy_revision}``. No research job, approval or
    receipt is created here; ``receipts`` is accepted for interface symmetry with the decision.
    """
    admin_id = _admin(admin_member)
    if not isinstance(body, Mapping):
        raise ValueError("the request body must be a JSON object")
    request_id = _identifier(body.get("request_id"), "request_id")
    job_ids = _job_ids(body.get("job_ids"))
    scope = _scope(body.get("scope"))
    payload_hash = digest({"job_ids": sorted(job_ids), "scope": scope})
    existing = _existing_candidate(table, admin_id, request_id, payload_hash)
    if existing is not None:
        return _candidate_view(existing)
    stamp = now_iso(_moment(now))
    linked_offers: list[str] = []
    linked_executions: list[str] = []
    questions = []
    for job_id in job_ids:
        job = _require_job(table, job_id)
        offers = _offers_of(table, job_id)
        linked_offers.extend(offer["offer_id"] for offer in offers)
        linked_executions.extend(offer["execution_id"] for offer in offers if offer.get("execution_id"))
        verdict = lab_jobs.read_verdict(table, job_id)
        questions.append({"job_id": job_id, "member_id": job.get("member_id"), "status": job.get("status"),
                          "question": (job.get("question") or "")[:QUESTION_SUMMARY_CHARS],
                          "choice": (verdict or {}).get("choice"), "review_candidate": review_probability(verdict),
                          "offer_status": offers[-1]["status"] if offers else None})
    proposal_hash = digest({"job_ids": sorted(job_ids), "scope": scope, "policy_revision": POLICY_REVISION})
    candidate_id = new_id()
    proposal = {"job_ids": list(job_ids), "scope": scope, "questions": questions, "policy_revision": POLICY_REVISION,
                "budget_usd_micros": RESEARCH_PROFILE["budget_usd_micros"], "max_calls": RESEARCH_PROFILE["max_calls"],
                "reread": RESEARCH_PROFILE["reread"], "model_id": None}
    candidate = new_item(
        *keys.candidate(candidate_id), stamp,
        candidate_id=candidate_id, job_ids=list(job_ids), parent_job_id=job_ids[0], scope=scope, proposal=proposal,
        proposal_hash=proposal_hash, proposal_revision=1, status=PROPOSED,
        linked_offer_ids=list(dict.fromkeys(linked_offers)), linked_execution_ids=list(dict.fromkeys(linked_executions)),
        created_by=admin_id, request_id=request_id, policy_revision=POLICY_REVISION,
        budget_usd_micros=RESEARCH_PROFILE["budget_usd_micros"], approval_id=None, execution_id=None, decision=None,
        decided_by=None, decided_at=None,
    )
    pointer = new_item(*keys.candidate_pointer(stamp, candidate_id), stamp, candidate_id=candidate_id, status=PROPOSED,
                       created_by=admin_id)
    idem = new_item(*keys.idempotency(admin_id, KIND_CANDIDATE, request_id), stamp, payload_hash=payload_hash,
                    candidate_id=candidate_id, kind=KIND_CANDIDATE)
    try:
        table.transact([Put(idem), Put(candidate), Put(pointer)])
    except ConditionFailed:
        existing = _existing_candidate(table, admin_id, request_id, payload_hash)
        if existing is None:
            raise
        return _candidate_view(existing)
    return _candidate_view(candidate)


def list_research_candidates(table: TablePort, *, status: str | None = None, cursor: str | None = None,
                             limit: int = 100) -> dict[str, Any]:
    """Research candidates oldest first, optionally one ``status``; ``{"candidates", "next_cursor"}``."""
    if status is not None and status not in CANDIDATE_STATUSES:
        raise ValueError(f"status must be one of {sorted(CANDIDATE_STATUSES)}, not {status!r}")
    limit = _limit(limit)
    after = _decode(cursor, ("sk",))["sk"] if cursor is not None else None
    candidates: list[dict[str, Any]] = []
    scanned = 0
    for _page in range(MAX_DAY_PAGES):
        items, next_sk = table.query("CANDIDATES", limit=PAGE_SIZE, start_after=after)
        for index, pointer in enumerate(items):
            scanned += 1
            if status is not None and pointer.get("status") != status:
                continue
            candidate = table.get(*keys.candidate(pointer["candidate_id"]))
            if candidate is None or (status is not None and candidate.get("status") != status):
                continue
            candidates.append(_candidate_view(candidate))
            if len(candidates) >= limit:
                more = index < len(items) - 1 or next_sk is not None
                return {"candidates": candidates, "next_cursor": _encode({"sk": pointer["sk"]}) if more else None}
        if next_sk is None:
            return {"candidates": candidates, "next_cursor": None}
        after = next_sk
        if scanned >= MAX_POINTERS_PER_CALL:
            break
    return {"candidates": candidates, "next_cursor": _encode({"sk": after}) if after else None}


def decide_research_candidate(table: TablePort, receipts: ReceiptWriter, admin_member: Any, body: Mapping[str, Any],
                              now: datetime | None = None) -> dict[str, Any]:
    """Approve or reject exactly the proposal revision and hash the administrator reviewed.

    A stale ``proposal_revision`` or ``proposal_hash`` raises ``RevisionConflict`` and writes
    nothing. ``reject`` records ``rejected``. ``approve`` commits one transaction: the candidate
    becomes ``approved`` with its ``execution_id`` and ``research_status``, an ``APPROVAL`` of kind
    ``professor_approval`` fixes the scope and the budget, call and reread limits the proposal
    showed (never the current module constants), and the idempotency key
    ``IDEMP#{admin}#decision#{request_id}`` is written. The execution is one of three:

    - When no research run exists for the bundled questions, ``lab_jobs.plan_research_job``
      plans the job with its reservation and outbox for the candidate's first job and that job's
      member (``research_status`` ``queued``).
    - When a bundled question already has a run (its ``JOB#{job}/EXECUTION`` guard or an offer
      names one), or another path commits one while this transaction is in flight, the approval
      links that execution: ``execution_linked`` is true, the approval carries ``LINK_NOTE`` and
      no job, reservation or outbox row is created.
    - When the period cap cannot hold the reservation (``BudgetExceeded``), the research job is
      recorded ``paused_budget`` without a reservation or outbox row; the candidate and approval
      record the refusal (``budget_reason``/``rejected_scope``) so the consent is not lost.

    Afterwards the research ``request.json`` (for a job created here) and
    ``approval-{approval_id}.json`` (under the original answer job) are written once. A second
    approval of the same revision, under any request id, returns the existing execution and
    writes nothing; a decision against an already decided candidate that contradicts it raises
    ``InvalidTransition``. Contention is re-read and re-planned up to
    ``lab_jobs.TRANSACTION_ATTEMPTS`` times with the jittered pause of ``lab_jobs.backoff_seconds``.
    """
    admin_id = _admin(admin_member)
    if not isinstance(body, Mapping):
        raise ValueError("the request body must be a JSON object")
    request_id = _identifier(body.get("request_id"), "request_id")
    candidate_id = _identifier(body.get("candidate_id"), "candidate_id")
    revision = body.get("proposal_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("proposal_revision must be a positive integer")
    proposal_hash = body.get("proposal_hash")
    if not isinstance(proposal_hash, str) or not _HASH.match(proposal_hash):
        raise ValueError("proposal_hash must be the 64-character hash shown with the proposal")
    decision = body.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    payload_hash = digest({"candidate_id": candidate_id, "proposal_revision": revision, "proposal_hash": proposal_hash,
                           "decision": decision})
    idem = table.get(*keys.idempotency(admin_id, KIND_DECISION, request_id))
    if idem is not None:
        if idem.get("payload_hash") != payload_hash:
            raise IdempotencyConflict(f"request {request_id} was already used with a different decision")
        candidate = _require_candidate(table, candidate_id)
        _ensure_approval_receipts(table, receipts, candidate)
        return _candidate_view(candidate)
    moment = _moment(now)
    stamp = now_iso(moment)
    approval_id = new_id()     # fixed for this call so re-plans never leave a second approval
    last_error: ConditionFailed | None = None
    for attempt in range(lab_jobs.TRANSACTION_ATTEMPTS):
        _pause_before(attempt)
        candidate = _require_candidate(table, candidate_id)
        _require_current(candidate, revision, proposal_hash)
        if candidate["status"] == APPROVED:
            if decision != APPROVE:
                raise InvalidTransition(f"candidate {candidate_id} is already approved; it cannot be rejected")
            _ensure_approval_receipts(table, receipts, candidate)
            return _candidate_view(candidate)
        if candidate["status"] == REJECTED:
            if decision != REJECT:
                raise InvalidTransition(f"candidate {candidate_id} is already rejected; propose a new revision")
            return _candidate_view(candidate)
        if candidate["status"] not in DECIDABLE_STATUSES:
            raise InvalidTransition(f"candidate {candidate_id} is {candidate['status']}; propose a new revision")
        idem_item = new_item(*keys.idempotency(admin_id, KIND_DECISION, request_id), stamp, payload_hash=payload_hash,
                             candidate_id=candidate_id, kind=KIND_DECISION, decision=decision)
        changes: dict[str, Any] = {"decision": decision, "decided_by": admin_id, "decided_at": stamp,
                                   "decision_request_id": request_id}
        operations: list[Operation] = []
        plan: lab_jobs.JobPlan | None = None
        approval: dict[str, Any] | None = None
        if decision == REJECT:
            changes["status"] = REJECTED
        else:
            plan, approval, approved = _plan_approval(table, candidate, approval_id, admin_id, request_id,
                                                      revision, proposal_hash, moment, stamp)
            changes.update(approved)
            operations.extend([Put(approval), *(plan.operations if plan is not None else ())])
        operations = [Update(*keys.candidate(candidate_id), candidate["revision"], changes),
                      *_candidate_pointer_update(table, candidate, changes["status"]),
                      Put(idem_item), *operations]
        try:
            table.transact(operations)
        except ConditionFailed as exc:
            # A consent that committed first fails the EXECUTION guard here; the re-plan links it.
            last_error = exc
            continue
        if approval is not None:
            if plan is not None:
                _write_once(receipts, plan.job["request_key"], plan.request)
            _write_once(receipts, approval["receipt_key"], _public(approval))
        return _candidate_view(_require_candidate(table, candidate_id))
    raise last_error  # type: ignore[misc]


def _plan_approval(table: TablePort, candidate: Mapping[str, Any], approval_id: str, admin_id: str, request_id: str,
                   revision: int, proposal_hash: str, moment: datetime,
                   stamp: str) -> tuple[lab_jobs.JobPlan | None, dict[str, Any], dict[str, Any]]:
    """The research plan (or ``None`` for a link), the ``APPROVAL`` item and the candidate's approval changes.

    Nothing is written. The budget, call and reread limits come from the candidate's stored
    proposal, so the approval fixes what the professor reviewed. An execution a bundled question
    already has is linked; otherwise a queued job is planned, or a ``paused_budget`` one when
    the period cap refuses the reservation.
    """
    parent = _require_job(table, candidate["job_ids"][0])
    proposal = candidate.get("proposal") if isinstance(candidate.get("proposal"), Mapping) else {}
    budget = int(proposal.get("budget_usd_micros", candidate["budget_usd_micros"]))
    max_calls = proposal.get("max_calls", RESEARCH_PROFILE["max_calls"])
    reread = proposal.get("reread", RESEARCH_PROFILE["reread"])
    refusal: dict[str, Any] = {}
    plan: lab_jobs.JobPlan | None = None
    linked = _existing_execution_of(table, candidate)
    if linked is not None:
        execution_id, research_status, note = linked, _execution_status(table, linked), LINK_NOTE
    else:
        research = dict(table=table, parent_job=parent, member_id=parent["member_id"], approval_id=approval_id,
                        scope=candidate["scope"], budget_usd_micros=budget, now=moment,
                        policy_revision=candidate["policy_revision"])
        try:
            plan = lab_jobs.plan_research_job(**research)
            research_status, note = lab_jobs.QUEUED, None
        except BudgetExceeded as exc:
            plan = lab_jobs.plan_research_job(**research, reserve=False, status=lab_jobs.PAUSED_BUDGET)
            research_status, note = lab_jobs.PAUSED_BUDGET, PAUSED_NOTE
            refusal = {"rejection_reason": str(exc), "rejected_scope": exc.scope, "requested_micros": exc.requested,
                       "available_micros": exc.available}
        execution_id = plan.job["job_id"]
    approval = new_item(
        *keys.approval(approval_id), stamp,
        approval_id=approval_id, kind=KIND_PROFESSOR_APPROVAL, candidate_id=candidate["candidate_id"],
        proposal_revision=revision, proposal_hash=proposal_hash, approved_by=admin_id,
        policy_revision=candidate["policy_revision"], scope=candidate["scope"],
        budget_usd_micros=budget, model_id=None, max_calls=max_calls, reread=reread,
        expires_at=now_iso(moment + timedelta(seconds=APPROVAL_TTL_SECONDS)), execution_id=execution_id,
        execution_status=research_status, execution_linked=linked is not None, note=note,
        status="active", parent_job_id=parent["job_id"], member_id=parent["member_id"],
        job_ids=list(candidate["job_ids"]), request_id=request_id,
        receipt_key=receipt_key(parent["job_id"], f"approval-{approval_id}.json"), **refusal,
    )
    changes: dict[str, Any] = {
        "status": APPROVED, "approval_id": approval_id, "execution_id": execution_id,
        "research_status": research_status, "execution_linked": linked is not None,
        "linked_execution_ids": list(dict.fromkeys([*candidate.get("linked_execution_ids", []), execution_id])),
    }
    if refusal:
        changes.update(budget_reason=refusal["rejection_reason"], rejected_scope=refusal["rejected_scope"])
    return plan, approval, changes


def _existing_execution_of(table: TablePort, candidate: Mapping[str, Any]) -> str | None:
    """The research execution a bundled question already has, or ``None``.

    The ``JOB#{job}/EXECUTION`` guard ``lab_jobs.plan_research_job`` puts once is authoritative;
    an offer's ``execution_id`` covers records written before the guard existed. The first job
    in ``job_ids`` is the parent of a new plan, so it is checked first.
    """
    for job_id in candidate["job_ids"]:
        guard = lab_jobs.existing_execution(table, job_id)
        if guard is not None and guard.get("execution_id"):
            return str(guard["execution_id"])
    for job_id in candidate["job_ids"]:
        for offer in _offers_of(table, job_id):
            if offer.get("execution_id"):
                return str(offer["execution_id"])
    return None


def _execution_status(table: TablePort, execution_id: str) -> str | None:
    job = table.get(*keys.job(execution_id))
    return job.get("status") if job is not None else None


def _pause_before(attempt: int) -> None:
    """The jittered pause ``lab_jobs`` takes between attempts on the hot period scopes (none before the first)."""
    if attempt > 0:
        lab_jobs._sleep(lab_jobs.backoff_seconds(attempt - 1))


def _require_current(candidate: Mapping[str, Any], revision: int, proposal_hash: str) -> None:
    if candidate.get("proposal_revision") != revision or candidate.get("proposal_hash") != proposal_hash:
        raise RevisionConflict(f"candidate {candidate['candidate_id']} is at proposal revision "
                               f"{candidate.get('proposal_revision')}; review the current proposal")
    if candidate.get("policy_revision") != POLICY_REVISION:
        raise RevisionConflict(f"candidate {candidate['candidate_id']} was proposed under policy "
                               f"{candidate.get('policy_revision')}; propose it again under {POLICY_REVISION}")


def _candidate_pointer_update(table: TablePort, candidate: Mapping[str, Any], status: str) -> list[Operation]:
    pk, sk = keys.candidate_pointer(candidate["created_at"], candidate["candidate_id"])
    pointer = table.get(pk, sk)
    if pointer is None or pointer.get("status") == status:
        return []
    return [Update(pk, sk, pointer["revision"], {"status": status})]


def _ensure_approval_receipts(table: TablePort, receipts: ReceiptWriter, candidate: Mapping[str, Any]) -> None:
    """Write the approval and research request receipts if a crash left them missing; never twice."""
    if candidate.get("status") != APPROVED:
        return
    approval = table.get(*keys.approval(candidate["approval_id"])) if candidate.get("approval_id") else None
    research = table.get(*keys.job(candidate["execution_id"])) if candidate.get("execution_id") else None
    if research is not None and research.get("request_key"):
        _write_once(receipts, research["request_key"], _research_request(research))
    if approval is not None and approval.get("receipt_key"):
        _write_once(receipts, approval["receipt_key"], _public(approval))


def _research_request(job: Mapping[str, Any]) -> dict[str, Any]:
    """The ``request.json`` body ``lab_jobs.plan_research_job`` writes, rebuilt from the job record."""
    return {"job_id": job["job_id"], "kind": job.get("kind"), "member_id": job.get("member_id"),
            "parent_job_id": job.get("parent_job_id"), "session_id": job.get("session_id"), "turn": job.get("turn"),
            "approval_id": job.get("approval_id"), "scope": job.get("scope"), "scope_hash": job.get("request_hash"),
            "budget_usd_micros": job.get("budget_usd_micros"), "question": job.get("question"),
            "policy_revision": job.get("policy_revision"), "created_at": job.get("created_at")}


# ---------------------------------------------------------------------------------------------
# Aggregation rounds
# ---------------------------------------------------------------------------------------------

def run_aggregation_round(table: TablePort, *, cutoff: float, from_day: str, to_day: str, now: datetime) -> dict[str, Any]:
    """Count one window's records and store a ``ROUND`` record; no Jev call, no candidate is created.

    ``valid_verdicts`` are verdicts with probabilities and ``passed`` those at or above ``cutoff``
    on the raw ``review_candidate`` value. Offers are counted by their recorded status; an
    ``offered`` one past its ``expires_at`` at ``now`` counts as ``expired``, otherwise as
    ``unanswered``. ``unavailable`` and ``skipped`` come from the verdict or the job's triage
    status; jobs still ``pending`` are listed in ``unprocessed_job_ids`` for the next round.
    ``executions`` are distinct research jobs linked from the window's offers and candidates;
    ``candidate_ids`` are the candidates proposed within the window. When the window holds more
    pointers than one round examines, ``complete`` is false and ``cursor`` marks where it stopped.
    """
    cutoff = _probability(cutoff, "cutoff")
    days = _days(from_day, to_day)
    moment = _moment(now)
    stamp = now_iso(moment)
    rows, resume = _scan_records(table, days, None, MAX_ROUND_POINTERS)
    counts = dict.fromkeys(COUNT_KEYS, 0)
    unprocessed: list[str] = []
    executions: set[str] = set()
    seen = missing = 0
    for _day, pointer in rows:
        if not _is_question(pointer):
            continue
        job = table.get(*keys.job(pointer["job_id"]))
        if job is None:
            missing += 1
            continue
        seen += 1
        if job.get("status") == lab_jobs.REJECTED_BUDGET:
            counts["rejected_budget"] += 1
        verdict = lab_jobs.read_verdict(table, job["job_id"])
        probability = review_probability(verdict)
        verdict_status = (verdict or {}).get("status")
        triage_status = job.get("triage_status")
        if probability is not None:
            counts["valid_verdicts"] += 1
            if probability >= cutoff:
                counts["passed"] += 1
        elif verdict is None and triage_status == "pending":
            unprocessed.append(job["job_id"])
        elif verdict_status == "unavailable" or (verdict is None and triage_status == "unavailable"):
            counts["unavailable"] += 1
        else:
            counts["skipped"] += 1
        for offer in _offers_of(table, job["job_id"]):
            counts["offers"] += 1
            offer_status = offer.get("status")
            if offer_status == "accepted":
                counts["accepted"] += 1
            elif offer_status == "declined":
                counts["declined"] += 1
            elif offer_status == "expired" or (offer_status == "offered" and _expired(offer.get("expires_at"), moment)):
                counts["expired"] += 1
            elif offer_status == "offered":
                counts["unanswered"] += 1
            if offer.get("execution_id"):
                executions.add(offer["execution_id"])
    candidate_ids = _candidates_in_window(table, days[0], days[-1])
    for candidate_id in candidate_ids:
        candidate = table.get(*keys.candidate(candidate_id))
        if candidate is not None and candidate.get("execution_id"):
            executions.add(candidate["execution_id"])
    counts["executions"] = len(executions)
    round_id = new_id()
    record = new_item(
        *keys.round(round_id), stamp,
        round_id=round_id, cutoff=cutoff, **{"from": days[0], "to": days[-1]}, counts=counts, candidate_ids=candidate_ids,
        unprocessed_job_ids=unprocessed, policy_revision=POLICY_REVISION, jev_calls=0, executed_at=stamp,
        jobs_seen=seen, missing_jobs=missing, pointers_seen=len(rows), complete=resume is None,
        cursor=_encode({"day": resume[0], "sk": resume[1]}) if resume is not None else None,
    )
    table.put(record)
    return record


def _expired(expires_at: Any, moment: datetime) -> bool:
    if not isinstance(expires_at, str):
        return False
    try:
        deadline = datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return deadline <= moment


def _candidates_in_window(table: TablePort, from_day: str, to_day: str) -> list[str]:
    ids: list[str] = []
    after: str | None = from_day   # pointer keys "{created_at}#{id}" sort after the bare start date
    for _page in range(MAX_DAY_PAGES):
        items, next_sk = table.query("CANDIDATES", limit=PAGE_SIZE, start_after=after)
        for item in items:
            if item["sk"][:10] > to_day:
                return ids
            ids.append(item["candidate_id"])
        if next_sk is None:
            break
        after = next_sk
    return ids


# ---------------------------------------------------------------------------------------------
# Day pointer scan and cursors
# ---------------------------------------------------------------------------------------------

def _scan_records(table: TablePort, days: list[str], start: tuple[str, str] | None,
                  budget: int) -> tuple[list[tuple[str, dict[str, Any]]], tuple[str, str] | None]:
    """Day pointers of ``days`` in order, at most ``budget``; the resume point is ``None`` when exhausted."""
    rows: list[tuple[str, dict[str, Any]]] = []
    start_day, after = start if start is not None else (days[0], None)
    for day in days:
        if day < start_day:
            continue
        cursor = after if day == start_day else None
        for _page in range(MAX_DAY_PAGES):
            if len(rows) >= budget:
                return rows, (rows[-1][0], rows[-1][1]["sk"])
            items, next_sk = table.query(f"RECORDS#{day}", limit=min(PAGE_SIZE, budget - len(rows)), start_after=cursor)
            rows.extend((day, item) for item in items)
            if next_sk is None:
                break
            cursor = next_sk
        else:
            return rows, (day, cursor)  # type: ignore[arg-type]
    return rows, None


def _days(from_day: Any, to_day: Any) -> list[str]:
    start, end = _date(from_day, "from_day"), _date(to_day, "to_day")
    if end < start:
        raise ValueError("to_day must not precede from_day")
    span = (end - start).days + 1
    if span > MAX_WINDOW_DAYS:
        raise ValueError(f"a window covers at most {MAX_WINDOW_DAYS} days")
    return [(start + timedelta(days=offset)).isoformat() for offset in range(span)]


def _date(value: Any, name: str) -> date:
    if not isinstance(value, str) or not _DAY.match(value):
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc


def _encode(payload: Mapping[str, str]) -> str:
    return base64.urlsafe_b64encode(canonical(payload)).decode("ascii").rstrip("=")


def _decode(cursor: Any, fields: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(cursor, str) or not cursor or len(cursor) > 1024:
        raise ValueError("invalid cursor")
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid cursor") from exc
    if (not isinstance(payload, dict) or set(payload) != set(fields)
            or not all(isinstance(payload[name], str) and payload[name] for name in fields)):
        raise ValueError("invalid cursor")
    return payload


def _decode_record_cursor(cursor: Any, days: list[str]) -> tuple[str, str] | None:
    if cursor is None:
        return None
    payload = _decode(cursor, ("day", "sk"))
    if payload["day"] not in days:
        raise ValueError("cursor lies outside the requested window")
    return payload["day"], payload["sk"]


# ---------------------------------------------------------------------------------------------
# Validation and shared helpers
# ---------------------------------------------------------------------------------------------

def _admin(member: Any) -> str:
    """The verified administrator's member id; students and malformed members are refused."""
    if isinstance(member, Mapping):
        member_id, role = member.get("member_id"), member.get("role")
    else:
        member_id, role = getattr(member, "member_id", None), getattr(member, "role", None)
    if not isinstance(member_id, str) or not _IDENTIFIER.match(member_id) or role not in lab_jobs.ROLES:
        raise ValueError("member must be the verified registry member with member_id and role")
    if role != "admin":
        raise Forbidden("this action is limited to administrators")
    return member_id


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ValueError(f"{name} must be a simple identifier of at most 128 characters")
    return value


def _limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    return limit


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1")
    return value


def _job_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > MAX_CANDIDATE_JOBS:
        raise ValueError(f"job_ids must list between 1 and {MAX_CANDIDATE_JOBS} question jobs")
    return list(dict.fromkeys(_identifier(job_id, "job_ids") for job_id in value))


def _scope(scope: Any) -> dict[str, Any]:
    """The research scope with every field present, so omitted keys never change its hash."""
    if not isinstance(scope, Mapping):
        raise ValueError("scope must be a JSON object with question, targets, new_pages and note")
    unknown = set(scope) - SCOPE_FIELDS
    if unknown:
        raise ValueError(f"scope has unknown fields {sorted(unknown)}")
    question = scope.get("question")
    if question is not None and (not isinstance(question, str) or not question.strip()
                                 or len(question) > lab_jobs.QUESTION_MAX_CHARS):
        raise ValueError(f"scope.question must be a non-empty string of at most {lab_jobs.QUESTION_MAX_CHARS} characters")
    note = scope.get("note")
    if note is not None and (not isinstance(note, str) or len(note) > SCOPE_NOTE_MAX_CHARS):
        raise ValueError(f"scope.note must be a string of at most {SCOPE_NOTE_MAX_CHARS} characters")
    return {"question": question, "targets": _page_keys(scope.get("targets"), "targets"),
            "new_pages": _page_keys(scope.get("new_pages"), "new_pages"), "note": note}


def _page_keys(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_SCOPE_KEYS:
        raise ValueError(f"scope.{name} must be a list of at most {MAX_SCOPE_KEYS} wiki keys")
    try:
        return list(dict.fromkeys(page_key(key) for key in value))
    except ValueError as exc:
        raise ValueError(f"scope.{name} must name published Markdown keys under wiki/") from exc


def _existing_candidate(table: TablePort, admin_id: str, request_id: str, payload_hash: str) -> dict[str, Any] | None:
    idem = table.get(*keys.idempotency(admin_id, KIND_CANDIDATE, request_id))
    if idem is None:
        return None
    if idem.get("payload_hash") != payload_hash:
        raise IdempotencyConflict(f"request {request_id} was already used with a different proposal")
    candidate = table.get(*keys.candidate(idem["candidate_id"]))
    if candidate is None:
        raise NotFound(f"idempotency key for {request_id} points at missing candidate {idem['candidate_id']}")
    return candidate


def _require_job(table: TablePort, job_id: str) -> dict[str, Any]:
    job = table.get(*keys.job(job_id)) if isinstance(job_id, str) and job_id else None
    if job is None:
        raise NotFound(f"no job {job_id}")
    return job


def _require_candidate(table: TablePort, candidate_id: str) -> dict[str, Any]:
    candidate = table.get(*keys.candidate(candidate_id))
    if candidate is None:
        raise NotFound(f"no research candidate {candidate_id}")
    return candidate


def _candidate_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return _public(candidate)


def _public(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in record.items() if name not in {"pk", "sk"}}


def _bounded(text: Any) -> str | None:
    return None if text is None else str(text)[:lab_jobs.QUESTION_MAX_CHARS]


def _moment(now: datetime | None) -> datetime:
    moment = now or datetime.now(UTC)
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    return moment


def _write_once(receipts: ReceiptWriter, key: str, value: Any) -> None:
    """Write a receipt unless an earlier attempt already left it in place (S3 412)."""
    try:
        receipts.put_json(key, value)
    except ClientError as exc:
        response = getattr(exc, "response", {}) or {}
        if response.get("Error", {}).get("Code") != "PreconditionFailed" and \
                response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
            raise
