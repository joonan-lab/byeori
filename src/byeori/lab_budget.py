"""Integer micro-USD budget ledger for lab question jobs (docs/LAB-QUESTION-WORKFLOW.md, P1).

Money is never a float here: one USD per million tokens is one micro-USD per token, so a usage
report converts with an exact ceiling per token class. A job reserves its cap against the lab
and member period scopes once; each paid model call then reserves an attempt inside that job
reservation, so the period totals are never charged twice. Settle, release and unknown are
conditional state transitions on explicit records, never TTL. Caps are optional: ``None`` means
no operational limit, but every reservation and settlement is still counted.

An answer job now runs with ``cap_micros`` of ``None`` (``lab_policy.ANSWER_JOB_CAP_MICROS``), so
it reserves nothing up front and no attempt is ever refused for money. That changes what a cap an
administrator sets on a period scope can promise: with nothing reserved ahead of the work, the
cap refuses new jobs once the period's recorded spend has reached it, instead of guaranteeing the
line is never crossed. A job already running is never stopped mid-way by a cap.

Every public function is a single transaction through ``lab_store.TablePort``. The ``plan_*``
variants return the operations without writing so ``lab_jobs`` can join them to its own.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from typing import Any

from byeori.costs import price_table
from byeori.lab_store import (
    Check,
    ConditionFailed,
    Operation,
    Put,
    StoreError,
    TablePort,
    Update,
    keys,
    new_id,
    new_item,
    now_iso,
    period_for,
)

__all__ = [
    "BudgetExceeded", "BudgetPlan", "InvalidTransition", "LedgerInconsistent", "NotFound", "TOKEN_CLASSES",
    "job_balance", "mark_unknown", "micros_for_usage", "period_for", "plan_attempt_reservation",
    "plan_job_reservation", "plan_mark_unknown", "plan_release", "plan_settle", "release", "reserve_attempt",
    "reserve_job", "scopes", "set_cap", "settle",
]

# Converse usage field -> price table column (costs.PRICES_USD_PER_MILLION).
TOKEN_CLASSES: tuple[tuple[str, str], ...] = (
    ("inputTokens", "input"),
    ("outputTokens", "output"),
    ("cacheReadInputTokens", "cache_read"),
    ("cacheWriteInputTokens", "cache_write"),
)

HELD, SETTLED, RELEASED, UNKNOWN = "held", "settled", "released", "unknown"
_TRANSITIONS = {SETTLED: {HELD, UNKNOWN}, RELEASED: {HELD, UNKNOWN}, UNKNOWN: {HELD}}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+$")
_PERIOD = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_SCOPE = re.compile(r"^(job:[A-Za-z0-9._-]+|lab:\d{4}-\d{2}|member:[A-Za-z0-9._-]+:\d{4}-\d{2})$")


class BudgetExceeded(ConditionFailed):
    """A reservation would push ``scope`` past its cap; nothing was written."""

    code = "budget_exceeded"

    def __init__(self, scope: str, *, requested: int, available: int):
        super().__init__(f"{scope} cannot reserve {requested} micro-USD; {available} available")
        self.scope, self.requested, self.available = scope, requested, available


class InvalidTransition(StoreError):
    """The reservation is not in a state that allows the requested transition."""

    code = "invalid_transition"


class NotFound(StoreError):
    """No reservation or job scope with that identifier."""

    code = "not_found"


class LedgerInconsistent(StoreError):
    """A scope counter would go negative; the ledger was changed outside this module."""

    code = "ledger_inconsistent"


@dataclass(frozen=True)
class BudgetPlan:
    """The record a transition produces and the operations that commit it. Nothing is written yet."""

    record: dict[str, Any]
    operations: tuple[Operation, ...]


class scopes:
    @staticmethod
    def job(job_id: str) -> str:
        return f"job:{job_id}"

    @staticmethod
    def lab(period: str) -> str:
        return f"lab:{period}"

    @staticmethod
    def member(member_id: str, period: str) -> str:
        return f"member:{member_id}:{period}"


# ---------------------------------------------------------------------------------------------
# Usage -> micro-USD
# ---------------------------------------------------------------------------------------------

def micros_for_usage(model_id: str, usage: dict[str, Any]) -> int:
    """Exact ceiling of ``tokens x USD-per-million`` per token class, summed as an integer."""
    prices = price_table(model_id)
    if prices is None:
        raise ValueError(f"No price table for model {model_id!r}")
    total = 0
    for field_name, column in TOKEN_CLASSES:
        tokens = usage.get(field_name)
        tokens = 0 if tokens is None else _require_amount(tokens, field_name)
        total += math.ceil(Fraction(tokens) * Fraction(str(prices[column])))
    return total


# ---------------------------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------------------------

def plan_job_reservation(table: TablePort, job_id: str, member_id: str, period: str, cap_micros: int | None, *,
                         now: datetime | None = None) -> BudgetPlan:
    """Reserve a job's cap (or, on resume, its remaining balance) against the period scopes.

    The first reservation creates ``BUDGET#job:{job_id}`` with the cap. A later call for the same
    job is a resume in a new period: the cap stays fixed, the previous job reservation must be
    closed, and only ``cap - reserved - settled`` is reserved against the new period.
    """
    _require_identifier(job_id, "job_id")
    _require_identifier(member_id, "member_id")
    if not _PERIOD.match(period or ""):
        raise ValueError(f"period must look like YYYY-MM, got {period!r}")
    stamp = now_iso(now)
    job_scope = scopes.job(job_id)
    job_record = table.get(*keys.budget(job_scope))
    reservation_id = new_id()
    operations: list[Operation] = []
    if job_record is None:
        cap = None if cap_micros is None else _require_amount(cap_micros, "cap_micros", positive=True)
        micros = cap or 0
        operations.append(Put(new_item(*keys.budget(job_scope), stamp, scope=job_scope, job_id=job_id,
                                       member_id=member_id, reserved_micros=0, settled_micros=0, cap_micros=cap,
                                       active_reservation_id=reservation_id)))
    else:
        cap = job_record["cap_micros"]
        if cap_micros is not None and cap_micros != cap:
            raise ValueError(f"{job_scope} cap is fixed at {cap} micro-USD from its first reservation")
        previous_id = job_record.get("active_reservation_id")
        previous = table.get(*keys.reservation(previous_id)) if previous_id else None
        if previous is not None and previous["status"] in {HELD, UNKNOWN}:
            raise InvalidTransition(f"{job_scope} still has reservation {previous_id} in status {previous['status']}")
        if cap is None:
            micros = 0
        else:
            micros = cap - job_record["reserved_micros"] - job_record["settled_micros"]
            if micros <= 0:
                raise BudgetExceeded(job_scope, requested=cap, available=max(micros, 0))
        operations.append(Update(*keys.budget(job_scope), job_record["revision"],
                                 {"active_reservation_id": reservation_id}))
    period_scopes = [scopes.lab(period), scopes.member(member_id, period)]
    for scope in period_scopes:
        operations.append(_bump(table, scope, micros, stamp))
    record = new_item(*keys.reservation(reservation_id), stamp, reservation_id=reservation_id, kind="job",
                      job_id=job_id, member_id=member_id, attempt_id=None, parent_reservation_id=None,
                      period=period, scopes=period_scopes, job_scope=job_scope, micros=micros, status=HELD,
                      settled_micros=0)
    operations.append(Put(record))
    return BudgetPlan(record, tuple(operations))


def reserve_job(table: TablePort, job_id: str, member_id: str, period: str, cap_micros: int | None, *,
                now: datetime | None = None) -> dict[str, Any]:
    plan = plan_job_reservation(table, job_id, member_id, period, cap_micros, now=now)
    table.transact(list(plan.operations))
    return plan.record


def plan_attempt_reservation(table: TablePort, job_id: str, attempt_id: str, micros: int, *,
                             now: datetime | None = None) -> BudgetPlan:
    """Reserve one paid call inside the job's held reservation; period scopes are not touched."""
    _require_identifier(job_id, "job_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("attempt_id must be a non-empty string")
    amount = _require_amount(micros, "micros", positive=True)
    stamp = now_iso(now)
    job_scope = scopes.job(job_id)
    job_record = table.get(*keys.budget(job_scope))
    if job_record is None:
        raise NotFound(f"no budget scope for {job_scope}")
    parent_id = job_record.get("active_reservation_id")
    parent = table.get(*keys.reservation(parent_id)) if parent_id else None
    if parent is None:
        raise NotFound(f"{job_scope} has no active job reservation")
    if parent["status"] != HELD:
        raise InvalidTransition(f"job reservation {parent_id} is {parent['status']}, not {HELD}")
    cap = job_record["cap_micros"]
    if cap is not None:
        free = cap - job_record["reserved_micros"] - job_record["settled_micros"]
        if amount > free:
            raise BudgetExceeded(job_scope, requested=amount, available=max(free, 0))
    reservation_id = new_id()
    record = new_item(*keys.reservation(reservation_id), stamp, reservation_id=reservation_id, kind="attempt",
                      job_id=job_id, member_id=parent["member_id"], attempt_id=attempt_id,
                      parent_reservation_id=parent_id, period=parent["period"], scopes=[job_scope],
                      job_scope=job_scope, micros=amount, status=HELD, settled_micros=0)
    operations = (
        Check(*keys.reservation(parent_id), parent["revision"]),
        Update(*keys.budget(job_scope), job_record["revision"],
               {"reserved_micros": job_record["reserved_micros"] + amount}),
        Put(record),
    )
    return BudgetPlan(record, operations)


def reserve_attempt(table: TablePort, job_id: str, attempt_id: str, micros: int, *,
                    now: datetime | None = None) -> dict[str, Any]:
    plan = plan_attempt_reservation(table, job_id, attempt_id, micros, now=now)
    table.transact(list(plan.operations))
    return plan.record


# ---------------------------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------------------------

def plan_settle(table: TablePort, reservation_id: str, actual_micros: int, *,
                now: datetime | None = None) -> BudgetPlan:
    """Move the reservation's micros out of ``reserved`` and the billed amount into ``settled``."""
    actual = _require_amount(actual_micros, "actual_micros")
    return _plan_transition(table, reservation_id, SETTLED, actual=actual, now=now)


def settle(table: TablePort, reservation_id: str, actual_micros: int, *, now: datetime | None = None) -> dict[str, Any]:
    return _commit(table, plan_settle(table, reservation_id, actual_micros, now=now))


def plan_release(table: TablePort, reservation_id: str, *, now: datetime | None = None) -> BudgetPlan:
    """Return every reserved micro to its scopes; nothing was billed."""
    return _plan_transition(table, reservation_id, RELEASED, actual=0, now=now)


def release(table: TablePort, reservation_id: str, *, now: datetime | None = None) -> dict[str, Any]:
    return _commit(table, plan_release(table, reservation_id, now=now))


def plan_mark_unknown(table: TablePort, reservation_id: str, *, reason: str | None = None,
                      now: datetime | None = None) -> BudgetPlan:
    """Flag a held reservation whose bill is unknown; its micros stay reserved."""
    return _plan_transition(table, reservation_id, UNKNOWN, actual=0, reason=reason, now=now)


def mark_unknown(table: TablePort, reservation_id: str, *, reason: str | None = None,
                 now: datetime | None = None) -> dict[str, Any]:
    return _commit(table, plan_mark_unknown(table, reservation_id, reason=reason, now=now))


def _plan_transition(table: TablePort, reservation_id: str, target: str, *, actual: int, reason: str | None = None,
                     now: datetime | None) -> BudgetPlan:
    record = table.get(*keys.reservation(reservation_id))
    if record is None:
        raise NotFound(f"no reservation {reservation_id}")
    status = record["status"]
    if status == target:
        return BudgetPlan(record, ())
    if status not in _TRANSITIONS[target]:
        raise InvalidTransition(f"reservation {reservation_id} is {status}; cannot become {target}")
    stamp = now_iso(now)
    changes: dict[str, Any] = {"status": target}
    operations: list[Operation] = []
    if target in {SETTLED, RELEASED}:
        for scope in record["scopes"]:
            scope_record = table.get(*keys.budget(scope))
            if scope_record is None:
                raise LedgerInconsistent(f"{scope} is missing while reservation {reservation_id} holds it")
            reserved = scope_record["reserved_micros"] - record["micros"]
            if reserved < 0:
                raise LedgerInconsistent(f"{scope} reserved_micros would fall to {reserved}")
            scope_changes = {"reserved_micros": reserved}
            if actual:
                scope_changes["settled_micros"] = scope_record["settled_micros"] + actual
            operations.append(Update(*keys.budget(scope), scope_record["revision"], scope_changes))
        changes["settled_micros"] = actual
        if target == SETTLED:
            changes["settled_at"] = stamp
        else:
            changes["released_micros"] = record["micros"]
            changes["released_at"] = stamp
    else:
        changes["unknown_at"] = stamp
        if reason:
            changes["reason"] = reason
    operations.append(Update(*keys.reservation(reservation_id), record["revision"], changes))
    return BudgetPlan({**record, **changes}, tuple(operations))


def _commit(table: TablePort, plan: BudgetPlan) -> dict[str, Any]:
    if not plan.operations:
        return plan.record
    table.transact(list(plan.operations))
    return table.get(*keys.reservation(plan.record["reservation_id"]))


# ---------------------------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------------------------

def set_cap(table: TablePort, scope: str, cap_micros: int | None, *, now: datetime | None = None) -> dict[str, Any]:
    """Create or change a scope's cap. ``None`` lifts it; counters are never touched."""
    if not _SCOPE.match(scope or ""):
        raise ValueError(f"unknown budget scope {scope!r}")
    cap = None if cap_micros is None else _require_amount(cap_micros, "cap_micros")
    record = table.get(*keys.budget(scope))
    if record is None:
        return table.put(new_item(*keys.budget(scope), now_iso(now), scope=scope, reserved_micros=0,
                                  settled_micros=0, cap_micros=cap))
    return table.update(*keys.budget(scope), record["revision"], {"cap_micros": cap})


def job_balance(table: TablePort, job_id: str) -> dict[str, Any]:
    record = table.get(*keys.budget(scopes.job(job_id)))
    if record is None:
        raise NotFound(f"no budget scope for {scopes.job(job_id)}")
    cap, reserved, settled = record["cap_micros"], record["reserved_micros"], record["settled_micros"]
    return {"job_id": job_id, "cap_micros": cap, "reserved_micros": reserved, "settled_micros": settled,
            "free_micros": None if cap is None else cap - reserved - settled,
            "active_reservation_id": record.get("active_reservation_id")}


def _bump(table: TablePort, scope: str, micros: int, stamp: str) -> Operation:
    record = table.get(*keys.budget(scope))
    if record is None:
        return Put(new_item(*keys.budget(scope), stamp, scope=scope, reserved_micros=micros, settled_micros=0,
                            cap_micros=None))
    cap = record.get("cap_micros")
    committed = record["reserved_micros"] + record["settled_micros"]
    if cap is not None and (committed >= cap or committed + micros > cap):
        raise BudgetExceeded(scope, requested=micros, available=max(cap - committed, 0))
    return Update(*keys.budget(scope), record["revision"], {"reserved_micros": record["reserved_micros"] + micros})


def _require_amount(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (positive and value == 0):
        bound = "a positive integer" if positive else "a non-negative integer"
        raise ValueError(f"{name} must be {bound} number of micro-USD, got {value!r}")
    return value


def _require_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ValueError(f"{name} must be a simple identifier, got {value!r}")
