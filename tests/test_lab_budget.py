"""Integer micro-USD budgeting: usage conversion, job/attempt reservations and their transitions."""
from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from byeori import lab_budget
from byeori.lab_budget import (
    BudgetExceeded,
    BudgetPlan,
    InvalidTransition,
    LedgerInconsistent,
    NotFound,
    job_balance,
    mark_unknown,
    micros_for_usage,
    period_for,
    plan_attempt_reservation,
    plan_job_reservation,
    release,
    reserve_attempt,
    reserve_job,
    scopes,
    set_cap,
    settle,
)
from byeori.lab_store import ConditionFailed, Put, keys, new_item
from lab_fakes import MemoryTable, fixed_clock

OPUS = "global.anthropic.claude-opus-5"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
PERIOD = "2026-09"
JOB_CAP = 500_000


def usage(**tokens):
    base = {"inputTokens": 0, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
    return {**base, **tokens}


def scope(table, name):
    return table.get(*keys.budget(name))


# ---------------------------------------------------------------------------------------------
# Usage -> micro-USD
# ---------------------------------------------------------------------------------------------

def test_micros_for_usage_matches_the_plan_example():
    assert micros_for_usage(OPUS, usage(inputTokens=1000, outputTokens=100)) == 1000 * 5 + 100 * 25


def test_micros_for_usage_rounds_each_token_class_up_separately():
    # opus cache read is 0.50 USD/M: 3 tokens = 1.5 micro-USD -> 2; cache write 6.25: 1 token -> 7
    assert micros_for_usage(OPUS, usage(cacheReadInputTokens=3)) == 2
    assert micros_for_usage(OPUS, usage(cacheWriteInputTokens=1)) == 7
    assert micros_for_usage(OPUS, usage(cacheReadInputTokens=3, cacheWriteInputTokens=1)) == 9
    # haiku cache read is 0.10 USD/M: 30 tokens are exactly 3 micro-USD, not the float artefact 4
    assert micros_for_usage(HAIKU, usage(cacheReadInputTokens=30)) == 3
    assert micros_for_usage(HAIKU, usage(cacheReadInputTokens=31)) == 4
    assert micros_for_usage(OPUS, usage()) == 0


def test_micros_for_usage_tolerates_missing_and_extra_converse_fields():
    assert micros_for_usage(OPUS, {"inputTokens": 10, "outputTokens": 1, "totalTokens": 11}) == 50 + 25
    assert micros_for_usage(OPUS, {"inputTokens": 10, "cacheReadInputTokens": None}) == 50
    assert isinstance(micros_for_usage(OPUS, usage(inputTokens=7)), int)


def test_micros_for_usage_rejects_unknown_models_and_bad_token_counts():
    with pytest.raises(ValueError):
        micros_for_usage("amazon.nova-pro-v1:0", usage(inputTokens=1))
    for bad in ({"inputTokens": -1}, {"inputTokens": 1.5}, {"inputTokens": "10"}, {"inputTokens": True}):
        with pytest.raises(ValueError):
            micros_for_usage(OPUS, bad)


# ---------------------------------------------------------------------------------------------
# Job reservations
# ---------------------------------------------------------------------------------------------

def test_scope_names_and_period():
    assert scopes.job("j1") == "job:j1"
    assert scopes.lab("2026-09") == "lab:2026-09"
    assert scopes.member("m1", "2026-09") == "member:m1:2026-09"
    assert period_for(datetime(2026, 9, 30, 23, 0, tzinfo=UTC)) == "2026-09"
    assert lab_budget.period_for is period_for


def test_reserve_job_creates_the_job_scope_and_bumps_both_period_scopes_in_one_transaction():
    table = MemoryTable()
    now = fixed_clock()
    reservation = reserve_job(table, "j1", "m1", PERIOD, JOB_CAP, now=now())

    assert len(table.transactions) == 1
    assert re.fullmatch(r"[0-9a-f]{32}", reservation["reservation_id"])
    assert reservation["status"] == "held" and reservation["kind"] == "job"
    assert reservation["job_id"] == "j1" and reservation["attempt_id"] is None
    assert reservation["micros"] == JOB_CAP and reservation["settled_micros"] == 0
    assert reservation["scopes"] == ["lab:2026-09", "member:m1:2026-09"]
    assert reservation["job_scope"] == "job:j1" and reservation["period"] == PERIOD
    assert reservation["revision"] == 1 and reservation["created_at"] == "2026-09-21T09:00:00.000000+00:00"
    assert table.get(*keys.reservation(reservation["reservation_id"])) == reservation

    job = scope(table, "job:j1")
    assert job["cap_micros"] == JOB_CAP and job["reserved_micros"] == 0 and job["settled_micros"] == 0
    assert job["active_reservation_id"] == reservation["reservation_id"] and job["member_id"] == "m1"
    for name in ("lab:2026-09", "member:m1:2026-09"):
        record = scope(table, name)
        assert record["scope"] == name
        assert record["reserved_micros"] == JOB_CAP and record["settled_micros"] == 0
        assert record["cap_micros"] is None


def test_reserve_job_refuses_when_the_lab_cap_would_be_exceeded_and_changes_nothing():
    table = MemoryTable()
    set_cap(table, "lab:2026-09", 600_000)
    reserve_job(table, "j1", "m1", PERIOD, JOB_CAP)
    before = table.rows()
    with pytest.raises(BudgetExceeded) as failure:
        reserve_job(table, "j2", "m2", PERIOD, JOB_CAP)
    assert isinstance(failure.value, ConditionFailed)
    assert failure.value.scope == "lab:2026-09" and failure.value.requested == JOB_CAP
    assert failure.value.available == 100_000 and failure.value.code == "budget_exceeded"
    assert table.rows() == before
    assert scope(table, "job:j2") is None and scope(table, "member:m2:2026-09") is None
    assert scope(table, "lab:2026-09")["reserved_micros"] == JOB_CAP
    assert [row["job_id"] for row in table.rows("RESERVATION#")] == ["j1"]
    # exactly at the cap is allowed: 500k + 100k == 600k
    reserve_job(table, "j3", "m3", PERIOD, 100_000)
    assert scope(table, "lab:2026-09")["reserved_micros"] == 600_000


def test_reserve_job_enforces_a_member_cap_and_counts_settled_usage_against_it():
    table = MemoryTable()
    set_cap(table, "member:m1:2026-09", 300_000)
    first = reserve_job(table, "j1", "m1", PERIOD, 200_000)
    settle(table, first["reservation_id"], 150_000)
    assert scope(table, "member:m1:2026-09")["settled_micros"] == 150_000
    with pytest.raises(BudgetExceeded) as failure:
        reserve_job(table, "j2", "m1", PERIOD, 200_000)
    assert failure.value.scope == "member:m1:2026-09" and failure.value.available == 150_000
    reserve_job(table, "j2", "m1", PERIOD, 150_000)
    reserve_job(table, "j3", "m2", PERIOD, 400_000)  # another member is not limited by m1's cap
    assert scope(table, "lab:2026-09")["reserved_micros"] == 550_000


def test_no_cap_means_unlimited_but_every_reservation_is_still_counted():
    table = MemoryTable()
    for i in range(4):
        reserve_job(table, f"j{i}", "m1", PERIOD, 5_000_000)
    lab = scope(table, "lab:2026-09")
    assert lab["cap_micros"] is None and lab["reserved_micros"] == 20_000_000
    member = scope(table, "member:m1:2026-09")
    assert member["cap_micros"] is None and member["reserved_micros"] == 20_000_000


def test_set_cap_creates_or_updates_a_scope_and_rejects_bad_values():
    table = MemoryTable()
    created = set_cap(table, "lab:2026-09", 1_000_000)
    assert created["cap_micros"] == 1_000_000 and created["reserved_micros"] == 0 and created["revision"] == 1
    lifted = set_cap(table, "lab:2026-09", None)
    assert lifted["cap_micros"] is None and lifted["revision"] == 2
    reserve_job(table, "j1", "m1", PERIOD, 250_000)
    lowered = set_cap(table, "lab:2026-09", 100_000)
    assert lowered["reserved_micros"] == 250_000  # counters are never touched by a cap change
    with pytest.raises(BudgetExceeded):
        reserve_job(table, "j2", "m1", PERIOD, 1)
    for bad in (-1, True, 1.5, "100"):
        with pytest.raises(ValueError):
            set_cap(table, "lab:2026-10", bad)
    with pytest.raises(ValueError):
        set_cap(table, "budget:2026-10", 1)


@pytest.mark.parametrize("cap", [0, -5, 1.5, True, "500000"])
def test_a_job_cap_that_is_given_must_be_a_positive_integer(cap):
    table = MemoryTable()
    with pytest.raises(ValueError):
        reserve_job(table, "j1", "m1", PERIOD, cap)
    assert table.rows() == []


def test_a_job_without_a_cap_reserves_nothing_and_refuses_no_attempt():
    """``None`` is how an answer job runs: the ledger counts, nothing refuses (lab_policy)."""
    table = MemoryTable()
    record = reserve_job(table, "j1", "m1", PERIOD, None)
    assert record["micros"] == 0
    scope = table.get(*keys.budget(scopes.job("j1")))
    assert scope["cap_micros"] is None and scope["reserved_micros"] == 0 and scope["settled_micros"] == 0
    assert job_balance(table, "j1")["free_micros"] is None
    for i in range(4):
        attempt = reserve_attempt(table, "j1", f"a{i}", 900_000)
        settle(table, attempt["reservation_id"], 800_000)
    scope = table.get(*keys.budget(scopes.job("j1")))
    assert (scope["reserved_micros"], scope["settled_micros"]) == (0, 3_200_000)


def test_reserve_job_rejects_malformed_identifiers_and_periods():
    table = MemoryTable()
    with pytest.raises(ValueError):
        reserve_job(table, "", "m1", PERIOD, JOB_CAP)
    with pytest.raises(ValueError):
        reserve_job(table, "j1", "m#1", PERIOD, JOB_CAP)
    with pytest.raises(ValueError):
        reserve_job(table, "j1", "m1", "2026-9", JOB_CAP)
    assert table.rows() == []


# ---------------------------------------------------------------------------------------------
# Attempt reservations inside a job reservation
# ---------------------------------------------------------------------------------------------

def test_reserve_attempt_reduces_the_job_free_micros_and_never_touches_period_scopes():
    table = MemoryTable()
    job = reserve_job(table, "j1", "m1", PERIOD, 100_000)
    attempt = reserve_attempt(table, "j1", "attempt-1", 60_000)
    assert attempt["kind"] == "attempt" and attempt["status"] == "held"
    assert attempt["attempt_id"] == "attempt-1" and attempt["job_id"] == "j1"
    assert attempt["parent_reservation_id"] == job["reservation_id"]
    assert attempt["scopes"] == ["job:j1"] and attempt["micros"] == 60_000 and attempt["period"] == PERIOD
    assert re.fullmatch(r"[0-9a-f]{32}", attempt["reservation_id"]) and attempt["reservation_id"] != job["reservation_id"]
    assert job_balance(table, "j1") == {"job_id": "j1", "cap_micros": 100_000, "reserved_micros": 60_000,
                                        "settled_micros": 0, "free_micros": 40_000,
                                        "active_reservation_id": job["reservation_id"]}
    assert scope(table, "lab:2026-09")["reserved_micros"] == 100_000
    assert scope(table, "member:m1:2026-09")["reserved_micros"] == 100_000
    assert len(table.transactions) == 2


def test_reserve_attempt_refuses_when_the_job_budget_is_exhausted():
    table = MemoryTable()
    reserve_job(table, "j1", "m1", PERIOD, 100_000)
    reserve_attempt(table, "j1", "attempt-1", 60_000)
    before = table.rows()
    with pytest.raises(BudgetExceeded) as failure:
        reserve_attempt(table, "j1", "attempt-2", 50_000)
    assert failure.value.scope == "job:j1" and failure.value.available == 40_000 and failure.value.requested == 50_000
    assert table.rows() == before
    reserve_attempt(table, "j1", "attempt-2", 40_000)
    assert job_balance(table, "j1")["free_micros"] == 0
    with pytest.raises(BudgetExceeded):
        reserve_attempt(table, "j1", "attempt-3", 1)


def test_reserve_attempt_requires_a_held_job_reservation():
    table = MemoryTable()
    with pytest.raises(NotFound):
        reserve_attempt(table, "missing", "attempt-1", 1)
    job = reserve_job(table, "j1", "m1", PERIOD, 100_000)
    release(table, job["reservation_id"])
    with pytest.raises(InvalidTransition):
        reserve_attempt(table, "j1", "attempt-1", 1)
    job2 = reserve_job(table, "j2", "m1", PERIOD, 100_000)
    mark_unknown(table, job2["reservation_id"])
    with pytest.raises(InvalidTransition):
        reserve_attempt(table, "j2", "attempt-1", 1)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            reserve_attempt(table, "j1", "attempt-1", bad)


# ---------------------------------------------------------------------------------------------
# Transitions: settle, release, unknown
# ---------------------------------------------------------------------------------------------

def test_settle_moves_reserved_to_settled_exactly_once():
    table = MemoryTable()
    reserve_job(table, "j1", "m1", PERIOD, 100_000)
    attempt = reserve_attempt(table, "j1", "attempt-1", 60_000)
    transactions = len(table.transactions)
    settled = settle(table, attempt["reservation_id"], 45_000)
    assert settled["status"] == "settled" and settled["settled_micros"] == 45_000 and settled["micros"] == 60_000
    assert settled["revision"] == 2 and settled["settled_at"]
    job = scope(table, "job:j1")
    assert job["reserved_micros"] == 0 and job["settled_micros"] == 45_000
    assert job_balance(table, "j1")["free_micros"] == 55_000
    assert len(table.transactions) == transactions + 1

    again = settle(table, attempt["reservation_id"], 99_999)
    assert again == settled
    assert len(table.transactions) == transactions + 1
    assert scope(table, "job:j1")["settled_micros"] == 45_000


def test_settle_accepts_an_overrun_and_zero_usage():
    table = MemoryTable()
    reserve_job(table, "j1", "m1", PERIOD, 100_000)
    attempt = reserve_attempt(table, "j1", "attempt-1", 10_000)
    settle(table, attempt["reservation_id"], 12_000)  # the model billed more than the estimate
    assert job_balance(table, "j1") == {"job_id": "j1", "cap_micros": 100_000, "reserved_micros": 0,
                                        "settled_micros": 12_000, "free_micros": 88_000,
                                        "active_reservation_id": job_balance(table, "j1")["active_reservation_id"]}
    second = reserve_attempt(table, "j1", "attempt-2", 10_000)
    zero = settle(table, second["reservation_id"], 0)
    assert zero["status"] == "settled" and zero["settled_micros"] == 0
    assert scope(table, "job:j1")["settled_micros"] == 12_000
    for bad in (-1, 1.5, True, None):
        third = reserve_attempt(table, "j1", f"attempt-{bad}", 1)
        with pytest.raises(ValueError):
            settle(table, third["reservation_id"], bad)


def test_settling_the_job_reservation_moves_the_period_scopes():
    table = MemoryTable()
    job = reserve_job(table, "j1", "m1", PERIOD, 100_000)
    attempt = reserve_attempt(table, "j1", "attempt-1", 60_000)
    settle(table, attempt["reservation_id"], 45_000)
    settled = settle(table, job["reservation_id"], job_balance(table, "j1")["settled_micros"])
    assert settled["status"] == "settled" and settled["settled_micros"] == 45_000
    for name in ("lab:2026-09", "member:m1:2026-09"):
        record = scope(table, name)
        assert record["reserved_micros"] == 0 and record["settled_micros"] == 45_000
    # the job scope keeps its own ledger; settling the job reservation does not add to it again
    assert scope(table, "job:j1")["settled_micros"] == 45_000


def test_release_returns_the_unused_micros_to_every_scope():
    table = MemoryTable()
    job = reserve_job(table, "j1", "m1", PERIOD, 100_000)
    attempt = reserve_attempt(table, "j1", "attempt-1", 60_000)
    released = release(table, attempt["reservation_id"])
    assert released["status"] == "released" and released["released_micros"] == 60_000
    assert released["settled_micros"] == 0 and released["released_at"]
    assert job_balance(table, "j1")["reserved_micros"] == 0 and job_balance(table, "j1")["free_micros"] == 100_000
    assert release(table, attempt["reservation_id"]) == released
    released_job = release(table, job["reservation_id"])
    assert released_job["released_micros"] == 100_000
    for name in ("lab:2026-09", "member:m1:2026-09"):
        record = scope(table, name)
        assert record["reserved_micros"] == 0 and record["settled_micros"] == 0


def test_mark_unknown_keeps_the_micros_reserved_until_the_outcome_is_known():
    table = MemoryTable()
    reserve_job(table, "j1", "m1", PERIOD, 100_000)
    attempt = reserve_attempt(table, "j1", "attempt-1", 60_000)
    unknown = mark_unknown(table, attempt["reservation_id"], reason="timeout after send")
    assert unknown["status"] == "unknown" and unknown["reason"] == "timeout after send" and unknown["unknown_at"]
    assert unknown["micros"] == 60_000 and unknown["settled_micros"] == 0
    assert job_balance(table, "j1")["reserved_micros"] == 60_000
    assert job_balance(table, "j1")["free_micros"] == 40_000
    transactions = len(table.transactions)
    assert mark_unknown(table, attempt["reservation_id"]) == unknown
    assert len(table.transactions) == transactions
    # once the bill is known the reservation can still be settled, exactly once
    settled = settle(table, attempt["reservation_id"], 50_000)
    assert settled["status"] == "settled" and scope(table, "job:j1")["settled_micros"] == 50_000
    assert scope(table, "job:j1")["reserved_micros"] == 0


def test_terminal_reservations_reject_other_transitions():
    table = MemoryTable()
    reserve_job(table, "j1", "m1", PERIOD, 100_000)
    settled = reserve_attempt(table, "j1", "a1", 10_000)
    settle(table, settled["reservation_id"], 5_000)
    released = reserve_attempt(table, "j1", "a2", 10_000)
    release(table, released["reservation_id"])
    with pytest.raises(InvalidTransition):
        release(table, settled["reservation_id"])
    with pytest.raises(InvalidTransition):
        mark_unknown(table, settled["reservation_id"])
    with pytest.raises(InvalidTransition):
        settle(table, released["reservation_id"], 1)
    with pytest.raises(InvalidTransition):
        mark_unknown(table, released["reservation_id"])
    for action in (lambda: settle(table, "missing", 1), lambda: release(table, "missing"),
                   lambda: mark_unknown(table, "missing"), lambda: job_balance(table, "missing")):
        with pytest.raises(NotFound):
            action()
    assert scope(table, "job:j1")["settled_micros"] == 5_000 and scope(table, "job:j1")["reserved_micros"] == 0


def test_settle_refuses_to_drive_a_scope_negative():
    table = MemoryTable()
    reserve_job(table, "j1", "m1", PERIOD, 100_000)
    attempt = reserve_attempt(table, "j1", "a1", 60_000)
    corrupted = scope(table, "job:j1")
    table.update(*keys.budget("job:j1"), corrupted["revision"], {"reserved_micros": 10_000})
    with pytest.raises(LedgerInconsistent):
        settle(table, attempt["reservation_id"], 1)
    assert table.get(*keys.reservation(attempt["reservation_id"]))["status"] == "held"


# ---------------------------------------------------------------------------------------------
# Resume in a new period
# ---------------------------------------------------------------------------------------------

def test_resumed_job_in_a_new_period_reserves_the_remaining_balance_without_double_counting_the_cap():
    table = MemoryTable()
    first = reserve_job(table, "j1", "m1", "2026-09", JOB_CAP)
    attempt = reserve_attempt(table, "j1", "a1", 200_000)
    settle(table, attempt["reservation_id"], 150_000)
    settle(table, first["reservation_id"], job_balance(table, "j1")["settled_micros"])
    assert scope(table, "lab:2026-09")["reserved_micros"] == 0 and scope(table, "lab:2026-09")["settled_micros"] == 150_000
    assert first["scopes"] == ["lab:2026-09", "member:m1:2026-09"]

    second = reserve_job(table, "j1", "m1", "2026-10", JOB_CAP)
    assert second["reservation_id"] != first["reservation_id"]
    assert second["micros"] == 350_000 and second["period"] == "2026-10"
    assert second["scopes"] == ["lab:2026-10", "member:m1:2026-10"]
    job = scope(table, "job:j1")
    assert job["cap_micros"] == JOB_CAP and job["active_reservation_id"] == second["reservation_id"]
    assert len(table.rows("BUDGET#job:")) == 1
    assert scope(table, "lab:2026-10")["reserved_micros"] == 350_000
    assert scope(table, "member:m1:2026-10")["reserved_micros"] == 350_000
    assert scope(table, "lab:2026-09")["reserved_micros"] == 0  # the old period is untouched
    assert job_balance(table, "j1")["free_micros"] == 350_000

    with pytest.raises(BudgetExceeded):
        reserve_attempt(table, "j1", "a2", 400_000)
    resumed = reserve_attempt(table, "j1", "a2", 300_000)
    assert resumed["parent_reservation_id"] == second["reservation_id"] and resumed["period"] == "2026-10"


def test_resume_checks_the_new_period_cap_against_the_remaining_balance_only():
    table = MemoryTable()
    first = reserve_job(table, "j1", "m1", "2026-09", JOB_CAP)
    attempt = reserve_attempt(table, "j1", "a1", 400_000)
    settle(table, attempt["reservation_id"], 400_000)
    settle(table, first["reservation_id"], 400_000)
    set_cap(table, "lab:2026-10", 120_000)
    second = reserve_job(table, "j1", "m1", "2026-10", JOB_CAP)  # 100k remaining fits under 120k
    assert second["micros"] == 100_000
    with pytest.raises(BudgetExceeded):
        reserve_job(table, "j2", "m1", "2026-10", 30_000)


def test_resume_refuses_while_the_previous_job_reservation_is_still_open():
    table = MemoryTable()
    first = reserve_job(table, "j1", "m1", "2026-09", JOB_CAP)
    with pytest.raises(InvalidTransition):
        reserve_job(table, "j1", "m1", "2026-10", JOB_CAP)
    mark_unknown(table, first["reservation_id"])
    with pytest.raises(InvalidTransition):
        reserve_job(table, "j1", "m1", "2026-10", JOB_CAP)
    assert scope(table, "lab:2026-10") is None
    settle(table, first["reservation_id"], 0)
    with pytest.raises(ValueError):
        reserve_job(table, "j1", "m1", "2026-10", JOB_CAP + 1)  # the job cap is fixed at first reservation
    resumed = reserve_job(table, "j1", "m1", "2026-10", None)  # None reuses the stored cap on resume
    assert resumed["micros"] == JOB_CAP


def test_resume_with_nothing_left_is_a_budget_refusal():
    table = MemoryTable()
    first = reserve_job(table, "j1", "m1", "2026-09", 10_000)
    attempt = reserve_attempt(table, "j1", "a1", 10_000)
    settle(table, attempt["reservation_id"], 10_000)
    settle(table, first["reservation_id"], 10_000)
    with pytest.raises(BudgetExceeded) as failure:
        reserve_job(table, "j1", "m1", "2026-10", 10_000)
    assert failure.value.scope == "job:j1" and failure.value.available == 0
    assert scope(table, "lab:2026-10") is None


# ---------------------------------------------------------------------------------------------
# Composition with other transactions and concurrency guards
# ---------------------------------------------------------------------------------------------

def test_plans_return_operations_without_writing_so_jobs_can_join_them_to_one_transaction():
    table = MemoryTable()
    table.put(new_item("JOB#j1", "META", "2026-09-21T09:00:00+00:00", job_id="j1", status="received"))
    plan = plan_job_reservation(table, "j1", "m1", PERIOD, JOB_CAP)
    assert isinstance(plan, BudgetPlan)
    assert table.rows("BUDGET#") == [] and table.rows("RESERVATION#") == []
    assert plan.record["status"] == "held" and plan.record["micros"] == JOB_CAP
    assert all(isinstance(op, Put) for op in plan.operations) and len(plan.operations) == 4
    table.transact([*plan.operations, Put(new_item("OUTBOX#x1", "META", "2026-09-21T09:00:00+00:00", job_id="j1"))])
    assert table.get(*keys.reservation(plan.record["reservation_id"])) == plan.record
    assert scope(table, "lab:2026-09")["reserved_micros"] == JOB_CAP
    assert table.get("OUTBOX#x1", "META")["job_id"] == "j1"

    attempt_plan = plan_attempt_reservation(table, "j1", "a1", 1_000)
    assert table.rows("RESERVATION#") == [plan.record]
    assert [type(op).__name__ for op in attempt_plan.operations] == ["Check", "Update", "Put"]
    table.transact(list(attempt_plan.operations))
    assert job_balance(table, "j1")["reserved_micros"] == 1_000


def test_a_stale_plan_fails_the_revision_guard_instead_of_overcommitting():
    table = MemoryTable()
    set_cap(table, "lab:2026-09", 600_000)
    stale = plan_job_reservation(table, "j1", "m1", PERIOD, JOB_CAP)
    reserve_job(table, "j2", "m2", PERIOD, JOB_CAP)  # another gateway invocation wins the race
    with pytest.raises(ConditionFailed) as failure:
        table.transact(list(stale.operations))
    assert not isinstance(failure.value, BudgetExceeded)
    assert scope(table, "lab:2026-09")["reserved_micros"] == JOB_CAP
    assert table.get(*keys.reservation(stale.record["reservation_id"])) is None
    assert scope(table, "job:j1") is None


def test_every_stored_amount_is_an_integer_or_null():
    table = MemoryTable()
    job = reserve_job(table, "j1", "m1", PERIOD, JOB_CAP)
    attempt = reserve_attempt(table, "j1", "a1", 12_345)
    settle(table, attempt["reservation_id"], micros_for_usage(OPUS, usage(inputTokens=1234, outputTokens=321,
                                                                            cacheReadInputTokens=7)))
    other = reserve_attempt(table, "j1", "a2", 1)
    mark_unknown(table, other["reservation_id"])
    settle(table, job["reservation_id"], job_balance(table, "j1")["settled_micros"])
    for row in table.rows():
        for name, value in row.items():
            if name.endswith("_micros") or name == "micros":
                assert value is None or (isinstance(value, int) and not isinstance(value, bool)), (row["pk"], name, value)
                assert value is None or value >= 0, (row["pk"], name, value)
