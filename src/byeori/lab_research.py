"""Approved research worker for the student question workflow (docs/LAB-QUESTION-WORKFLOW.md section 7, P4).

A research job exists only because a student accepted a synthesis offer or a professor approved a
candidate (``lab_offers``, ``lab_review``); both paths bind the job to one ``APPROVAL`` record.
``run_research`` takes the job ``lab_jobs.claim`` moved to ``running`` and, before spending anything,
re-reads that approval: it must be ``active``, unexpired, issued under the current policy revision
and bound to this execution and its parent question. Anything else closes the job ``failed`` with
``approval_invalid`` and never calls the model.

The research itself is the campaign engine, ``question_agent.run_answer``: search, read, reread
originals, edit and create pages, then answer. Two things are injected. ``ScopedPublisher`` wraps
``wiki_connections.publish_page`` so every edit, creation, link refresh and the optional question
page passes one scope check: existing pages may be edited only when the approval lists them as
``targets`` (or this execution created them), new pages are limited to the approval's ``new_pages``
plus an allowance of ``NEW_PAGE_ALLOWANCE`` under ``wiki/concepts/`` or ``wiki/overviews/`` for a
``new_synthesis`` consent, and nothing under ``papers/`` or ``index/`` is ever a publication target.
A refusal reaches the model as a tool error naming the follow-up route. Every attempted publication
is recorded in an in-memory ledger (operation id, target, base version, body hash, outcome) that the
receipt carries. The final answer is stored in ``runs/lab-questions/{job_id}/research.json``;
``wiki/questions/`` is written only when the approval's scope says ``publish_question``.

Money: one attempt reservation for the job's whole research budget is taken inside the job
reservation before the engine starts. The engine reports the usage of every call the service
answered; that is settled once. A call that raised after it was sent (a timeout, an unclassified
error) leaves the bill unknown, so the attempt and the job close ``outcome_unknown`` with the hold
kept, as design section 8 requires; refusals the service makes before processing (throttling,
validation, access, credentials) bill nothing. The model is never called twice for one job: a
redelivered job whose ``research.json`` already exists is closed from that record.
"""
from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ParamValidationError,
)

from byeori import evidence_packet, lab_budget, lab_jobs, question_agent, wiki_connections
from byeori.costs import price_table
from byeori.evidence_packet import page_key
from byeori.lab_budget import BudgetExceeded
from byeori.lab_jobs import TRANSACTION_ATTEMPTS
from byeori.lab_policy import POLICY_REVISION, RESEARCH_PROFILE
from byeori.lab_store import ConditionFailed, ReceiptWriter, TablePort, Update, keys, now_iso, receipt_key

__all__ = [
    "APPROVAL_INVALID", "BUDGET_EXCEEDED", "MAX_ENGINE_BUDGET_USD", "NEW_PAGE_ALLOWANCE", "NEW_PAGE_FOLDERS",
    "NEW_SYNTHESIS", "OUT_OF_SCOPE", "RECEIPT_NAME", "ScopedPublisher", "converse_once", "run_research",
    "search_adapter",
]

RECEIPT_NAME = "research.json"
NEW_SYNTHESIS = "new_synthesis"
NEW_PAGE_ALLOWANCE = 2
NEW_PAGE_FOLDERS = ("wiki/concepts/", "wiki/overviews/")
OUT_OF_SCOPE = "outside the approved research scope; describe this as a follow-up in your answer instead"
APPROVAL_INVALID = "approval_invalid"
BUDGET_EXCEEDED = "budget_exceeded"
MODEL_UNPRICED = "model_unpriced"
NO_MODEL_RESPONSE = "no_model_response"
MAX_ENGINE_BUDGET_USD = 20.0      # question_agent.run_answer's own per-run ceiling; the reservation is the real cap
MAX_REASON_CHARS = 500
REREAD_MODES = frozenset({"auto", "never", "always"})
MISSING_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
# Codes Bedrock returns before it processes a request: nothing was billed.
DEFINITE_FAILURE_CODES = frozenset({"ValidationException", "AccessDeniedException", "ThrottlingException",
                                    "TooManyRequestsException"})
ENGINE_STATUS_TO_JOB = {"answer_ready": lab_jobs.COMPLETED}


# ---------------------------------------------------------------------------------------------
# Adapters around the engine
# ---------------------------------------------------------------------------------------------

def converse_once(client: Any, request: Mapping[str, Any]) -> tuple[Any, int, int]:
    """One ``client.converse(**request)`` with no retry, in the ``(response, waited, attempts)`` shape the engine unpacks."""
    return client.converse(**request), 0, 1


def search_adapter(index: Any) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """``evidence_packet.search`` over the shared index in the ``{results, index_etag}`` shape the engine expects."""
    connection, etag = _index_pair(index)

    def search(event: Mapping[str, Any]) -> dict[str, Any]:
        limit = int(event.get("limit") or 12)
        hits = evidence_packet.search(connection, str(event.get("query") or ""), max(1, min(limit, 30)),
                                      event.get("doc_type"))
        # The engine derives a page key from doc_type/doc_id and falls back to ``path`` for paper pages.
        results = [{**hit, "path": "data/" + hit["key"]} for hit in hits]
        return {"query": event.get("query"), "index_etag": etag, "results": results}

    return search


def _index_pair(index: Any) -> tuple[Any, str | None]:
    if isinstance(index, tuple):
        connection, etag = index
        return connection, None if etag is None else str(etag)
    return index, None


class _ModelCalls:
    """Counts what the engine sent and what came back, so an unknown bill is never settled as zero."""

    def __init__(self, converse: Callable[[Any, Mapping[str, Any]], tuple[Any, Any, Any]]):
        self.converse = converse
        self.sent = 0
        self.succeeded = 0
        self.errors: list[BaseException] = []
        self.usage: dict[str, int] = {}

    def __call__(self, client: Any, request: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        self.sent += 1
        try:
            response, waited, attempts = self.converse(client, request)
        except Exception as exc:
            self.errors.append(exc)
            raise
        self.succeeded += 1
        for name, value in ((response.get("usage") if isinstance(response, Mapping) else None) or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                self.usage[name] = self.usage.get(name, 0) + value
        return response, waited, attempts

    @property
    def unknown_bill(self) -> bool:
        """True when a call raised after it may have reached the service."""
        return any(not _definite_failure(exc) for exc in self.errors)

    def summary(self) -> dict[str, Any]:
        return {"sent": self.sent, "succeeded": self.succeeded,
                "errors": [{"code": _error_code(exc), "definite": _definite_failure(exc)} for exc in self.errors]}


# ---------------------------------------------------------------------------------------------
# Scoped publisher
# ---------------------------------------------------------------------------------------------

class ScopedPublisher:
    """``wiki_connections.publish_page`` limited to an approval's scope, with an operation ledger.

    ``targets`` are the existing pages the approval allows the model to edit; ``new_pages`` are
    explicitly approved new keys; ``new_page_allowance`` more pages may be created under
    ``NEW_PAGE_FOLDERS`` (two for a ``new_synthesis`` consent, none otherwise). A page this
    execution created may be edited again; ``question_key`` is the one ``wiki/questions/`` page the
    engine may write when the approval allows it. Keys outside ``wiki/`` are refused before any
    S3 call, so ``papers/`` and ``index/`` are never touched. Every refusal is kept in ``refusals``
    and every attempted publication in ``operations``.
    """

    def __init__(self, *, targets: list[str], new_pages: list[str] = (), new_page_allowance: int = 0,
                 question_key: str | None = None, execution_id: str = "", now: datetime | None = None,
                 clock: Callable[[], datetime] | None = None):
        self.targets = frozenset(targets)
        self.approved_new_pages = frozenset(new_pages)
        self.allowance = max(0, int(new_page_allowance))
        self.question_key = question_key
        self.execution_id = execution_id
        self.stamp = now_iso(now)
        self.clock = clock              # when given, each refusal and operation is stamped as it happens
        self.operations: list[dict[str, Any]] = []
        self.refusals: list[dict[str, Any]] = []
        self.created: list[str] = []
        self.expected: dict[str, str] = {}   # key -> ETag of this execution's latest successful write

    @property
    def published(self) -> list[dict[str, Any]]:
        return [operation for operation in self.operations if operation["status"] == "published"]

    @property
    def published_keys(self) -> list[str]:
        return list(dict.fromkeys(operation["key"] for operation in self.published))

    def _stamp(self) -> str:
        return now_iso(self.clock()) if self.clock is not None else self.stamp

    def check(self, key: Any, *, create_only: bool) -> str:
        """The key when the approval allows this publication; ``ValueError`` (``OUT_OF_SCOPE``) otherwise."""
        if not isinstance(key, str) or not key.startswith("wiki/") or ".." in key:
            raise ValueError(OUT_OF_SCOPE)
        if key == self.question_key:
            return key
        if create_only:
            if key in self.approved_new_pages:
                return key
            if key.startswith(NEW_PAGE_FOLDERS) and len(self.created) < self.allowance:
                return key
            raise ValueError(OUT_OF_SCOPE)
        if key in self.targets or key in self.created:
            return key
        raise ValueError(OUT_OF_SCOPE)

    def __call__(self, s3: Any, bucket: str, key: Any, text: Any, *, check_remaining: Callable[[], None] | None = None,
                 **options: Any) -> dict[str, Any]:
        create_only = bool(options.get("create_only"))
        kind = "create" if create_only else "edit"
        try:
            key = self.check(key, create_only=create_only)
        except ValueError as exc:
            self.refusals.append({"key": str(key)[:200], "kind": kind, "reason": str(exc), "at": self._stamp()})
            raise
        operation: dict[str, Any] = {
            "operation_id": uuid.uuid4().hex, "execution_id": self.execution_id, "key": key, "kind": kind,
            "expected_etag": options.get("expected_etag"),
            "sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(), "status": "pending", "at": self._stamp(),
        }
        self.operations.append(operation)
        try:
            result = wiki_connections.publish_page(s3, bucket, key, text, check_remaining=check_remaining, **options)
        except Exception as exc:
            operation.update(status="failed", error=str(exc)[:MAX_REASON_CHARS])
            raise
        operation.update(status="published", sha256=result["sha256"], etag=result["etag"], replaced=result["replaced"],
                         chars=result["chars"], connections=result["connections"], errors=result["errors"])
        if create_only and key not in self.created:
            self.created.append(key)
        self.expected[key] = result["etag"]
        return result


# ---------------------------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------------------------

def run_research(job: Mapping[str, Any], *, table: TablePort, receipts: ReceiptWriter, s3: Any, bucket: str, index: Any,
                 model_client: Any, converse: Callable[..., tuple[Any, Any, Any]], model_id: str,
                 reasoning: str | None, now: datetime, remaining_ms: Callable[[], int] | None = None,
                 clock: Callable[[], datetime] | None = None) -> dict[str, Any]:
    """Run one claimed research job to a terminal state and return ``{status, job, receipt_key, ...}``.

    ``now`` is the moment the approval is judged against and the stamp of every write when no
    ``clock`` is given. With ``clock`` (the handler passes its UTC clock) publications, budget
    settlement and the closing ``completed_at`` are stamped as they happen, so an eight-minute run
    no longer reports the claim time as its completion time.

    ``job`` is the ``running`` research job ``lab_jobs.claim`` returned. ``index`` is the
    ``(connection, etag)`` pair of the shared index, ``converse`` the ``(client, request) ->
    (response, waited, attempts)`` callable (``converse_once`` in production). The returned
    ``status`` is the job's terminal status: ``completed`` when the engine reported
    ``answer_ready``, ``partial`` for any other engine outcome, ``failed`` when the approval was
    invalid, the budget could not be reserved or no model response arrived, and
    ``outcome_unknown`` when a call's bill is unknown. ``recovered`` is true when the job was
    closed from a receipt an earlier attempt saved, without a model call.
    """
    current = _running_research_job(job)
    moment = _moment(now)
    run = _Run(job=current, table=table, receipts=receipts, model_id=str(model_id), reasoning=reasoning, moment=moment,
               key=receipt_key(current["job_id"], RECEIPT_NAME), clock=clock)
    recovered = _recover(run)
    if recovered is not None:
        return recovered

    approval, problem = _load_approval(table, current, moment)
    if approval is None:
        closed = lab_jobs.fail(table, run.job_id, current["revision"], reason=problem, error_code=APPROVAL_INVALID,
                               now=moment, **run.proof)
        return _outcome(run, closed, reason=problem, error_code=APPROVAL_INVALID)
    try:
        scope = _scope_of(approval, current)
    except ValueError as exc:
        reason = f"approval scope is unusable: {exc}"[:MAX_REASON_CHARS]
        closed = lab_jobs.fail(table, run.job_id, current["revision"], reason=reason, error_code=APPROVAL_INVALID,
                               now=moment, **run.proof)
        return _outcome(run, closed, reason=reason, error_code=APPROVAL_INVALID)
    run.approval, run.scope = approval, scope
    if price_table(run.model_id) is None:
        # Without a price the usage could never be settled; refuse before anything is reserved or sent.
        reason = f"model {run.model_id!r} has no configured token price"
        closed = lab_jobs.fail(table, run.job_id, run.job["revision"], reason=reason, error_code=MODEL_UNPRICED,
                               now=moment, **run.proof)
        return _outcome(run, closed, reason=reason, error_code=MODEL_UNPRICED)

    budget_micros = int(current["budget_usd_micros"])
    attempt_id = f"attempt-{int(current.get('attempt', 0))}-research"
    try:
        run.reservation = _reserve(run, attempt_id, budget_micros)
    except BudgetExceeded as exc:
        reason = str(exc)[:MAX_REASON_CHARS]
        closed = lab_jobs.fail(table, run.job_id, run.job["revision"], reason=reason, error_code=BUDGET_EXCEEDED,
                               now=moment, **run.proof)
        return _outcome(run, closed, reason=reason, error_code=BUDGET_EXCEEDED)

    run.publisher = ScopedPublisher(targets=scope["targets"], new_pages=scope["new_pages"],
                                    new_page_allowance=scope["new_page_allowance"], question_key=scope["question_key"],
                                    execution_id=run.job_id, now=moment, clock=clock)
    run.calls = _ModelCalls(converse)
    event = {"title": scope["question"], "budget_usd": min(budget_micros / 1e6, MAX_ENGINE_BUDGET_USD),
             "reread": scope["reread"], "author": f"lab:{current['member_id']}",
             "tags": ["lab-research", approval.get("kind") or "approval"]}
    try:
        result = question_agent.run_answer(
            event, s3=s3, bucket=bucket, model_client=model_client, model_id=run.model_id, reasoning=reasoning,
            converse=run.calls, search=search_adapter(index), remaining_ms=remaining_ms, publisher=run.publisher,
            publish_question=scope["publish_question"], archive=run.archived.append,
        )
    except Exception as exc:  # noqa: BLE001 - the engine swallows model errors; anything escaping is closed here
        return _close_after_escape(run, exc)
    return _close_from_result(run, result)


class _Run:
    """What one worker pass knows; filled in as the steps succeed."""

    def __init__(self, *, job: dict[str, Any], table: TablePort, receipts: ReceiptWriter, model_id: str,
                 reasoning: str | None, moment: datetime, key: str, clock: Callable[[], datetime] | None = None):
        self.job, self.table, self.receipts = job, table, receipts
        self.model_id, self.reasoning, self._moment, self.key = model_id, reasoning, moment, key
        self.clock = clock
        self.approval: dict[str, Any] | None = None
        self.scope: dict[str, Any] | None = None
        self.reservation: dict[str, Any] | None = None
        self.publisher: ScopedPublisher | None = None
        self.calls: _ModelCalls | None = None
        self.archived: list[dict[str, Any]] = []

    @property
    def job_id(self) -> str:
        return self.job["job_id"]

    @property
    def moment(self) -> datetime:
        """The stamp for the next write: the clock's reading when one was given, else the fixed ``now``."""
        return _moment(self.clock()) if self.clock is not None else self._moment

    @property
    def proof(self) -> dict[str, Any]:
        """Lease ownership for ``lab_jobs`` closes: the claim's attempt and outbox, when the job carries them."""
        if self.job.get("claimed_outbox_id"):
            return {"attempt": int(self.job.get("attempt", 0)), "claimed_outbox_id": self.job["claimed_outbox_id"]}
        return {}

    def refresh(self) -> dict[str, Any]:
        self.job = _running_research_job(self.table.get(*keys.job(self.job_id)))
        return self.job


# ---------------------------------------------------------------------------------------------
# Approval and scope
# ---------------------------------------------------------------------------------------------

def _load_approval(table: TablePort, job: Mapping[str, Any], moment: datetime) -> tuple[dict[str, Any] | None, str | None]:
    """The job's approval when it still authorises this execution, else ``(None, reason)``."""
    approval_id = job.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id:
        return None, "research job names no approval"
    approval = table.get(*keys.approval(approval_id))
    if approval is None:
        return None, f"approval {approval_id} not found"
    if approval.get("status") != "active":
        return None, f"approval {approval_id} is {approval.get('status')!r}, not active"
    expires_at = approval.get("expires_at")
    try:
        expired = not isinstance(expires_at, str) or datetime.fromisoformat(expires_at) <= moment
    except ValueError:
        expired = True
    if expired:
        return None, f"approval {approval_id} expired at {expires_at}"
    if approval.get("policy_revision") != POLICY_REVISION:
        return None, (f"approval {approval_id} was issued under policy {approval.get('policy_revision')!r}; "
                      f"current policy is {POLICY_REVISION!r}")
    if approval.get("execution_id") != job["job_id"]:
        return None, f"approval {approval_id} names execution {approval.get('execution_id')!r}, not this job"
    parent = approval.get("parent_job_id") or approval.get("job_id")
    if parent != job.get("parent_job_id"):
        return None, f"approval {approval_id} belongs to question {parent!r}, not this job's parent"
    if not isinstance(approval.get("scope"), Mapping):
        return None, f"approval {approval_id} carries no scope"
    return approval, None


def _scope_of(approval: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    """The research scope the publisher and the engine run under, taken from the approval only."""
    scope = approval["scope"]
    question = scope.get("question") or job.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("the scope names no research question")
    targets = _page_keys(scope.get("targets"), "targets")
    new_pages = _page_keys(scope.get("new_pages"), "new_pages")
    kind = scope.get("kind")
    allowance = NEW_PAGE_ALLOWANCE if kind == NEW_SYNTHESIS else 0
    publish_question = scope.get("publish_question", False) is True
    reread = approval.get("reread") or RESEARCH_PROFILE["reread"]
    if reread not in REREAD_MODES:
        reread = RESEARCH_PROFILE["reread"]
    question_key = question_agent.question_key_for(question.strip())[1] if publish_question else None
    return {"question": question.strip(), "targets": targets, "new_pages": new_pages, "new_page_allowance": allowance,
            "kind": kind, "publish_question": publish_question, "question_key": question_key, "reread": reread,
            "note": scope.get("note"), "offer_id": scope.get("offer_id")}


def _page_keys(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"scope.{name} must be a list of wiki keys")
    try:
        return list(dict.fromkeys(page_key(key) for key in value))
    except ValueError as exc:
        raise ValueError(f"scope.{name} must name published Markdown keys under wiki/") from exc


# ---------------------------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------------------------

def _reserve(run: _Run, attempt_id: str, micros: int) -> dict[str, Any]:
    """One attempt reservation for the whole research budget, named on the job in the same transaction."""
    plan = lab_budget.plan_attempt_reservation(run.table, run.job_id, attempt_id, micros, now=run.moment)
    run.table.transact([*plan.operations,
                        Update(*keys.job(run.job_id), run.job["revision"],
                               {"attempt_reservation_id": plan.record["reservation_id"], "attempt_id": attempt_id})])
    run.refresh()
    return plan.record


def _definite_failure(exc: BaseException) -> bool:
    if isinstance(exc, (ParamValidationError, NoCredentialsError, EndpointConnectionError, ConnectTimeoutError)):
        return True
    return isinstance(exc, ClientError) and _client_code(exc) in DEFINITE_FAILURE_CODES


def _client_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        return str(response.get("Error", {}).get("Code", ""))
    return ""


def _error_code(exc: BaseException) -> str:
    return _client_code(exc) or type(exc).__name__


# ---------------------------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------------------------

def _close_from_result(run: _Run, result: Mapping[str, Any]) -> dict[str, Any]:
    """Settle or hold the attempt, write the receipt, record the pages on the job and close it."""
    calls, publisher, reservation = run.calls, run.publisher, run.reservation
    usage = dict(result.get("usage") or {})
    actual = _micros(run.model_id, usage)
    published = publisher.published_keys
    receipt: dict[str, Any] = _receipt_body(run, research=dict(result), usage=usage, usd_micros=actual)
    if calls.unknown_bill:
        # A call raised after it may have been processed; its usage never arrived. The hold stays.
        reason = _unknown_reason(calls)
        receipt.update(status=lab_jobs.OUTCOME_UNKNOWN, billing="unknown", reason=reason)
        _write_receipt(run, receipt)
        _record_on_job(run, receipt, published)
        lab_budget.mark_unknown(run.table, reservation["reservation_id"], reason=reason, now=run.moment)
        closed = lab_jobs.mark_unknown(run.table, run.job_id, run.job["revision"], reason=reason, now=run.moment,
                                       **run.proof)
        _update_approval(run, closed["status"], published)
        return _outcome(run, closed, research=result, reason=reason)
    if calls.succeeded == 0 and not published and not result.get("answer"):
        # Nothing came back from the model and nothing was written: a definite failure, nothing billed.
        error_code = _error_code(calls.errors[-1]) if calls.errors else NO_MODEL_RESPONSE
        reason = "; ".join(str(p) for p in result.get("problems") or [])[:MAX_REASON_CHARS] or "no model response"
        receipt.update(status=lab_jobs.FAILED, billing="released", reason=reason, error_code=error_code)
        _write_receipt(run, receipt)
        _record_on_job(run, receipt, published)
        lab_budget.release(run.table, reservation["reservation_id"], now=run.moment)
        closed = lab_jobs.fail(run.table, run.job_id, run.job["revision"], reason=reason, error_code=error_code,
                               usage=usage, usd_micros=0, now=run.moment, **run.proof)
        _update_approval(run, closed["status"], published)
        return _outcome(run, closed, research=result, reason=reason, error_code=error_code)
    status = ENGINE_STATUS_TO_JOB.get(str(result.get("status")), lab_jobs.PARTIAL)
    receipt.update(status=status, billing="settled")
    _write_receipt(run, receipt)
    _record_on_job(run, receipt, published)
    lab_budget.settle(run.table, reservation["reservation_id"], actual, now=run.moment)
    closed = lab_jobs.complete(run.table, run.job_id, run.job["revision"], receipt_key=run.key, evidence_key=None,
                               usage=usage, usd_micros=actual, status=status, now=run.moment, **run.proof)
    _update_approval(run, closed["status"], published)
    return _outcome(run, closed, research=result)


def _close_after_escape(run: _Run, exc: BaseException) -> dict[str, Any]:
    """An exception escaped the engine: bill what the service answered, hold what it may have, and close."""
    calls, publisher, reservation = run.calls, run.publisher, run.reservation
    reason = f"{type(exc).__name__}: {exc}"[:MAX_REASON_CHARS]
    usage = dict(calls.usage)
    actual = _micros(run.model_id, usage)
    published = publisher.published_keys
    research = run.archived[-1] if run.archived else None
    receipt = _receipt_body(run, research=research, usage=usage, usd_micros=actual)
    receipt.update(reason=reason, error_code=type(exc).__name__)
    if calls.unknown_bill:
        held = _unknown_reason(calls)
        receipt.update(status=lab_jobs.OUTCOME_UNKNOWN, billing="unknown", reason=f"{reason}; {held}"[:MAX_REASON_CHARS])
        _write_receipt(run, receipt)
        _record_on_job(run, receipt, published)
        lab_budget.mark_unknown(run.table, reservation["reservation_id"], reason=held, now=run.moment)
        closed = lab_jobs.mark_unknown(run.table, run.job_id, run.job["revision"], reason=receipt["reason"],
                                       now=run.moment, **run.proof)
        _update_approval(run, closed["status"], published)
        return _outcome(run, closed, research=research, reason=receipt["reason"])
    receipt.update(status=lab_jobs.FAILED, billing="settled" if calls.succeeded else "released")
    _write_receipt(run, receipt)
    _record_on_job(run, receipt, published)
    if calls.succeeded:
        lab_budget.settle(run.table, reservation["reservation_id"], actual, now=run.moment)
    else:
        lab_budget.release(run.table, reservation["reservation_id"], now=run.moment)
    closed = lab_jobs.fail(run.table, run.job_id, run.job["revision"], reason=reason, error_code=type(exc).__name__,
                           usage=usage if calls.succeeded else None, usd_micros=actual, now=run.moment, **run.proof)
    _update_approval(run, closed["status"], published)
    return _outcome(run, closed, research=research, reason=reason, error_code=type(exc).__name__)


def _micros(model_id: str, usage: Mapping[str, Any]) -> int:
    """The bill of ``usage``; nothing answered is nothing billed, whatever the model's price table."""
    if not any(int(value or 0) for value in usage.values()):
        return 0
    return lab_budget.micros_for_usage(model_id, dict(usage))


def _unknown_reason(calls: _ModelCalls) -> str:
    codes = ", ".join(_error_code(exc) for exc in calls.errors if not _definite_failure(exc))
    return f"a model call raised after it was sent ({codes}); its bill is unknown"[:MAX_REASON_CHARS]


def _receipt_body(run: _Run, *, research: Mapping[str, Any] | None, usage: Mapping[str, Any], usd_micros: int) -> dict[str, Any]:
    job, approval, scope = run.job, run.approval or {}, run.scope or {}
    publisher, calls = run.publisher, run.calls
    published = publisher.published_keys if publisher else []
    return {
        "job_id": run.job_id, "kind": lab_jobs.KIND_RESEARCH, "member_id": job.get("member_id"),
        "parent_job_id": job.get("parent_job_id"), "session_id": job.get("session_id"),
        "approval_id": job.get("approval_id"), "approval_kind": approval.get("kind"),
        "approved_by": approval.get("approved_by"), "policy_revision": approval.get("policy_revision"),
        "scope": {"question": scope.get("question"), "targets": scope.get("targets"), "new_pages": scope.get("new_pages"),
                  "new_page_allowance": scope.get("new_page_allowance"), "kind": scope.get("kind"),
                  "publish_question": scope.get("publish_question"), "note": scope.get("note"),
                  "offer_id": scope.get("offer_id")},
        "attempt": int(job.get("attempt", 0)), "attempt_id": job.get("attempt_id"),
        "attempt_reservation_id": job.get("attempt_reservation_id"), "budget_usd_micros": job.get("budget_usd_micros"),
        "model_id": run.model_id, "reasoning": run.reasoning, "model_calls": calls.summary() if calls else None,
        "usage": dict(usage), "usd_micros": usd_micros,
        "research": research, "research_status": research.get("status") if research else None,
        "publications": list(publisher.operations) if publisher else [],
        "out_of_scope": list(publisher.refusals) if publisher else [],
        "published_keys": published, "index_pending": bool(published),
        "completed_at": now_iso(run.moment),
    }


def _write_receipt(run: _Run, receipt: Mapping[str, Any]) -> None:
    """Write ``research.json`` once; an object already there is the identical earlier write of this attempt."""
    try:
        run.receipts.put_json(run.key, receipt)
    except ClientError as exc:
        error = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") if hasattr(exc, "response") else None
        if error.get("Code") != "PreconditionFailed" and status != 412:
            raise


def _record_on_job(run: _Run, receipt: Mapping[str, Any], published: list[str]) -> None:
    """Name the receipt, the pages written and the index state on the still-running job."""
    changes = {"receipt_key": run.key, "pages_written": list(published), "index_pending": bool(published),
               "research_status": receipt.get("research_status"), "out_of_scope_count": len(receipt.get("out_of_scope") or []),
               "publication_count": len(receipt.get("publications") or [])}
    run.table.update(*keys.job(run.job_id), run.job["revision"], changes)
    run.refresh()


def _update_approval(run: _Run, status: str, published: list[str]) -> None:
    """Record the execution outcome on the approval; a concurrent change is re-read and re-applied."""
    if run.approval is None:
        return
    changes = {"execution_status": status, "published_keys": list(published), "index_pending": bool(published),
               "execution_receipt_key": run.key, "executed_at": now_iso(run.moment)}
    approval = run.approval
    for _attempt in range(TRANSACTION_ATTEMPTS):
        try:
            run.approval = run.table.update(*keys.approval(approval["approval_id"]), approval["revision"], changes)
            return
        except ConditionFailed:
            approval = run.table.get(*keys.approval(approval["approval_id"]))
            if approval is None:
                return


def _outcome(run: _Run, closed: Mapping[str, Any], *, research: Mapping[str, Any] | None = None,
             reason: str | None = None, error_code: str | None = None, recovered: bool = False) -> dict[str, Any]:
    publisher, calls = run.publisher, run.calls
    return {
        "status": closed["status"], "job": dict(closed), "job_id": run.job_id, "receipt_key": closed.get("receipt_key"),
        "research_status": research.get("status") if research else None,
        "answer": research.get("answer") if research else None,
        "published_keys": publisher.published_keys if publisher else [],
        "out_of_scope": list(publisher.refusals) if publisher else [],
        "index_pending": bool(publisher.published_keys) if publisher else False,
        "usage": closed.get("usage"), "usd_micros": closed.get("usd_micros"),
        "model_calls": calls.sent if calls else 0, "reason": reason, "error_code": error_code, "recovered": recovered,
    }


# ---------------------------------------------------------------------------------------------
# Recovery from a saved receipt (design section 8: never call the model again)
# ---------------------------------------------------------------------------------------------

def _recover(run: _Run) -> dict[str, Any] | None:
    """Close a re-delivered job from the ``research.json`` an earlier attempt saved, or return ``None``."""
    try:
        record = run.receipts.get_json(run.key)
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code", "")) in MISSING_CODES:
            return None
        raise
    if not isinstance(record, Mapping):
        return None
    status = record.get("status")
    usage = dict(record.get("usage") or {})
    micros = int(record.get("usd_micros") or 0)
    reservation_id = record.get("attempt_reservation_id")
    reservation = run.table.get(*keys.reservation(reservation_id)) if reservation_id else None
    if reservation is not None and reservation.get("status") == lab_budget.HELD:
        if record.get("billing") == "unknown":
            lab_budget.mark_unknown(run.table, reservation_id, reason=str(record.get("reason") or "recovered"), now=run.moment)
        elif record.get("billing") == "released":
            lab_budget.release(run.table, reservation_id, now=run.moment)
        else:
            lab_budget.settle(run.table, reservation_id, micros, now=run.moment)
    published = [str(key) for key in record.get("published_keys") or []]
    _record_on_job(run, record, published)
    reason = str(record.get("reason") or "recovered from a saved receipt")[:MAX_REASON_CHARS]
    if status in {lab_jobs.COMPLETED, lab_jobs.PARTIAL}:
        closed = lab_jobs.complete(run.table, run.job_id, run.job["revision"], receipt_key=run.key, evidence_key=None,
                                   usage=usage, usd_micros=micros, status=status, now=run.moment, **run.proof)
    elif status == lab_jobs.OUTCOME_UNKNOWN or record.get("billing") == "unknown":
        closed = lab_jobs.mark_unknown(run.table, run.job_id, run.job["revision"], reason=reason, now=run.moment,
                                       **run.proof)
    else:
        closed = lab_jobs.fail(run.table, run.job_id, run.job["revision"], reason=reason,
                               error_code=str(record.get("error_code") or "recovered"), usage=usage or None,
                               usd_micros=micros, now=run.moment, **run.proof)
    approval = run.table.get(*keys.approval(run.job["approval_id"])) if run.job.get("approval_id") else None
    if approval is not None:
        run.approval = approval
        _update_approval(run, closed["status"], published)
    research = record.get("research") if isinstance(record.get("research"), Mapping) else None
    outcome = _outcome(run, closed, research=research, reason=reason if closed["status"] != lab_jobs.COMPLETED else None,
                       error_code=record.get("error_code"), recovered=True)
    outcome.update(published_keys=published, index_pending=bool(published), out_of_scope=list(record.get("out_of_scope") or []))
    return outcome


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

def _running_research_job(job: Any) -> dict[str, Any]:
    if not isinstance(job, Mapping) or not isinstance(job.get("job_id"), str) or not job.get("job_id"):
        raise ValueError("job must be the claimed research job record")
    if job.get("kind") != lab_jobs.KIND_RESEARCH:
        raise ValueError(f"job {job['job_id']} is a {job.get('kind')!r} job, not research")
    if job.get("status") != lab_jobs.RUNNING:
        raise lab_jobs.InvalidTransition(f"job {job['job_id']} is {job.get('status')!r}, not running")
    if not isinstance(job.get("revision"), int):
        raise ValueError("job must carry its revision")
    return dict(job)


def _moment(now: datetime | None) -> datetime:
    moment = now or datetime.now(UTC)
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime")
    return moment
