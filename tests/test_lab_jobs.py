"""Idempotent intake, one-transaction queueing, leases, closes and outbox bookkeeping of byeori.lab_jobs."""
from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime, timedelta

import pytest

from byeori import lab_budget, lab_jobs
from byeori.lab_jobs import (
    Forbidden,
    IdempotencyConflict,
    InvalidTransition,
    JobPlan,
    Member,
    NotFound,
    backoff_seconds,
    claim,
    complete,
    create_research_job,
    existing_execution,
    expired_leases,
    fail,
    intake,
    mark_sent,
    mark_unknown,
    normalise_request,
    pending_outbox,
    plan_research_job,
    queue,
    read_job,
    read_verdict,
    set_triage_status,
    sweep_expired_leases,
)
from byeori.lab_policy import ANSWER_JOB_CAP_MICROS, LEASE_SECONDS, POLICY_REVISION, RESEARCH_PROFILE
from byeori.lab_store import (
    ConditionFailed,
    Delete,
    Put,
    ReceiptWriter,
    StoreError,
    TransactionConflict,
    digest,
    keys,
    new_item,
)
from lab_fakes import MemoryS3, MemoryTable, fixed_clock, member

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
PERIOD = "2026-09"
# A cap these tests hand to ``queue`` so the reservation machinery has an amount to move around.
# The shipped answer job passes ``ANSWER_JOB_CAP_MICROS``, which is ``None``: no ceiling refuses a
# student's answer. The test below named after the default is the one that checks that.
JOB_CAP = 1_500_000
QUESTION = "CHD8 결손은 대두증과 연관되지 않는가? 아시아 코호트(n = 120)에서 확인된 결과만 답해 주세요."
HEX = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record every backoff pause instead of sleeping; tests assert on the recorded seconds."""
    recorded: list[float] = []
    monkeypatch.setattr(lab_jobs, "_sleep", recorded.append)
    return recorded


def assert_backoff_shape(recorded):
    """Each pause stays under its exponential ceiling (full jitter over BASE * 2**i, capped at MAX)."""
    for i, seconds in enumerate(recorded):
        ceiling = min(lab_jobs.BACKOFF_MAX_SECONDS, lab_jobs.BACKOFF_BASE_SECONDS * 2 ** i)
        assert 0 <= seconds <= ceiling, (i, seconds, ceiling)


class World:
    """One control table, one bucket and three registered members shared by a test."""

    def __init__(self):
        self.table = MemoryTable()
        self.s3 = MemoryS3()
        self.receipts = ReceiptWriter(self.s3, "bucket")
        self.student = Member("m1", "student")
        self.other = Member("m2", "student")
        self.admin = Member("prof", "admin")
        member(self.table, "m1")
        member(self.table, "m2")
        member(self.table, "prof", role="admin")
        self.baseline = len(self.table.transactions)

    def transactions(self):
        return self.table.transactions[self.baseline:]

    def scope(self, name):
        return self.table.get(*keys.budget(name))

    def rows(self, prefix):
        return self.table.rows(prefix)

    def queued_job(self, request_id="req-1", **overrides):
        job = intake(self.table, self.receipts, self.student, body(request_id=request_id, **overrides), NOW)
        return queue(self.table, job["job_id"], period=PERIOD, cap=JOB_CAP, now=NOW)

    def running_job(self, **overrides):
        job = self.queued_job(**overrides)
        return claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, NOW)


def body(**overrides):
    return {"action": "ask_byeori", "request_id": "req-1", "question": QUESTION, **overrides}


def op_keys(operations):
    return [(type(op).__name__, op.item["pk"] if isinstance(op, Put) else op.pk) for op in operations]


# ---------------------------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------------------------

def test_intake_writes_idempotency_job_pointers_session_and_request_receipt_in_that_order():
    w = World()
    context = [{"role": "user", "text": "앞선 질문은 CHD8 변이였습니다."}, {"role": "assistant", "text": "네."}]
    job = intake(w.table, w.receipts, w.student, body(context=context), NOW)

    assert HEX.match(job["job_id"]) and HEX.match(job["session_id"])
    assert job["status"] == "received" and job["kind"] == "answer" and job["turn"] == 1
    assert job["member_id"] == "m1" and job["request_id"] == "req-1" and job["parent_job_id"] is None
    assert job["question"] == QUESTION and job["standalone_question"] is None  # a follow-up with context
    assert job["context_hash"] == digest(context)
    assert job["question_hash"] == digest({"question": QUESTION, "context_hash": digest(context)})
    assert job["request_hash"] == digest(normalise_request(body(context=context)))
    assert job["policy_revision"] == POLICY_REVISION and job["attempt"] == 0 and job["lease_until"] is None
    assert job["triage_status"] == "pending" and job["usd_micros"] is None and job["private_material"] is False
    assert job["request_key"] == f"runs/lab-questions/{job['job_id']}/request.json"
    assert job["created_at"] == "2026-09-21T09:00:00.000000+00:00" and job["revision"] == 1
    assert w.table.get(*keys.job(job["job_id"])) == job

    assert len(w.transactions()) == 1
    assert op_keys(w.transactions()[0]) == [
        ("Put", "IDEMP#m1#answer#req-1"), ("Put", f"JOB#{job['job_id']}"), ("Put", "MEMBER#m1"),
        ("Put", "RECORDS#2026-09-21"), ("Put", f"SESSION#{job['session_id']}"),
    ]
    idem = w.table.get("IDEMP#m1#answer#req-1", "KEY")
    assert idem["job_id"] == job["job_id"] and idem["payload_hash"] == job["request_hash"]
    pointer = w.table.get("MEMBER#m1", f"JOB#{job['created_at']}#{job['job_id']}")
    assert pointer["status"] == "received" and pointer["job_id"] == job["job_id"]
    day = w.table.get("RECORDS#2026-09-21", f"{job['created_at']}#{job['job_id']}")
    assert day["member_id"] == "m1" and day["status"] == "received"
    session = w.table.get(*keys.session(job["session_id"]))
    assert session["member_id"] == "m1" and session["last_turn"] == 1 and session["last_job_id"] == job["job_id"]

    assert w.s3.writes == [(job["request_key"], {"IfNoneMatch": "*"})]
    receipt = w.s3.json(job["request_key"])
    assert receipt["question"] == QUESTION and receipt["context"] == context
    assert receipt["member_id"] == "m1" and receipt["job_id"] == job["job_id"] and receipt["request_id"] == "req-1"
    assert receipt["received_at"] == job["created_at"] and receipt["policy_revision"] == POLICY_REVISION


def test_intake_without_context_keeps_the_question_as_its_own_standalone_form():
    w = World()
    job = intake(w.table, w.receipts, w.student, body(), NOW)
    assert job["standalone_question"] == QUESTION and job["context_hash"] == digest([])
    assert w.s3.json(job["request_key"])["context"] == []


def test_same_request_id_and_payload_returns_the_existing_job_without_writing():
    w = World()
    first = intake(w.table, w.receipts, w.student, body(), NOW)
    writes = list(w.s3.writes)
    again = intake(w.table, w.receipts, w.student, body(), NOW + timedelta(minutes=5))
    assert again == first
    assert len(w.transactions()) == 1 and w.s3.writes == writes
    # identity and transport fields never take part in the payload comparison
    decorated = body(author="someone else", member_id="m2", role="admin", action="ask_byeori")
    assert intake(w.table, w.receipts, w.student, decorated, NOW) == first
    assert len(w.transactions()) == 1 and w.s3.writes == writes


def test_same_request_id_with_a_different_payload_raises_idempotency_conflict_without_writing():
    w = World()
    first = intake(w.table, w.receipts, w.student, body(), NOW)
    before = w.rows("")
    with pytest.raises(IdempotencyConflict) as failure:
        intake(w.table, w.receipts, w.student, body(question=QUESTION + " 그리고 유럽 코호트는?"), NOW)
    assert isinstance(failure.value, StoreError) and failure.value.code == "idempotency_conflict"
    with pytest.raises(IdempotencyConflict):
        intake(w.table, w.receipts, w.student, body(context=[{"role": "user", "text": "x"}]), NOW)
    assert w.rows("") == before and len(w.s3.writes) == 1
    # another member may use the same request_id: the key includes the verified member
    other = intake(w.table, w.receipts, w.other, body(), NOW)
    assert other["member_id"] == "m2" and other["job_id"] != first["job_id"]
    assert len(w.rows("JOB#")) == 2


def test_identity_comes_from_the_verified_member_never_from_the_body():
    w = World()
    forged = body(member_id="prof", author="prof", role="admin")
    job = intake(w.table, w.receipts, w.student, forged, NOW)
    assert job["member_id"] == "m1"
    assert w.table.get("IDEMP#m1#answer#req-1", "KEY") is not None
    assert w.table.get("IDEMP#prof#answer#req-1", "KEY") is None
    assert w.rows("MEMBER#prof") == [w.table.get("MEMBER#prof", "PROFILE")]
    receipt = w.s3.json(job["request_key"])
    assert receipt["member_id"] == "m1" and "author" not in receipt and "role" not in receipt
    profile = w.table.get("MEMBER#m1", "PROFILE")  # a registry profile works as the member too
    assert intake(w.table, w.receipts, profile, body(request_id="req-2"), NOW)["member_id"] == "m1"


def test_own_session_advances_the_turn_and_other_or_unknown_sessions_are_refused():
    w = World()
    first = intake(w.table, w.receipts, w.student, body(request_id="req-1"), NOW)
    second = intake(w.table, w.receipts, w.student, body(request_id="req-2", session_id=first["session_id"],
                                                          parent_job_id=first["job_id"]), NOW + timedelta(minutes=1))
    assert second["session_id"] == first["session_id"] and second["turn"] == 2
    assert second["parent_job_id"] == first["job_id"] and second["standalone_question"] is None
    session = w.table.get(*keys.session(first["session_id"]))
    assert session["last_turn"] == 2 and session["last_job_id"] == second["job_id"] and session["revision"] == 2
    assert op_keys(w.transactions()[-1])[-1] == ("Update", f"SESSION#{first['session_id']}")

    before = w.rows("")
    # another member's session and an unknown one are indistinguishable: NotFound both ways, never Forbidden
    with pytest.raises(NotFound) as stolen:
        intake(w.table, w.receipts, w.other, body(request_id="req-3", session_id=first["session_id"]), NOW)
    with pytest.raises(NotFound) as missing:
        intake(w.table, w.receipts, w.student, body(request_id="req-4", session_id="0" * 32), NOW)
    assert stolen.value.code == missing.value.code == "not_found"
    assert not isinstance(stolen.value, Forbidden) and type(stolen.value) is type(missing.value)
    assert w.rows("") == before and len(w.s3.writes) == 2


def test_parent_job_of_another_member_and_an_unknown_parent_are_both_not_found():
    w = World()
    parent = intake(w.table, w.receipts, w.student, body(), NOW)
    before = w.rows("")
    with pytest.raises(NotFound) as stolen:
        intake(w.table, w.receipts, w.other, body(request_id="req-2", parent_job_id=parent["job_id"]), NOW)
    with pytest.raises(NotFound) as missing:
        intake(w.table, w.receipts, w.student, body(request_id="req-3", parent_job_id="f" * 32), NOW)
    assert stolen.value.code == missing.value.code == "not_found" and not isinstance(stolen.value, Forbidden)
    assert w.rows("") == before
    follow = intake(w.table, w.receipts, w.student, body(request_id="req-4", parent_job_id=parent["job_id"]), NOW)
    assert follow["parent_job_id"] == parent["job_id"] and follow["session_id"] != parent["session_id"]


@pytest.mark.parametrize("bad", [
    {"question": QUESTION},                                      # no request_id
    {"request_id": "req#1", "question": QUESTION},               # key separator in the request id
    {"request_id": "", "question": QUESTION},
    {"request_id": "req-1"},                                     # no question
    {"request_id": "req-1", "question": "   "},
    {"request_id": "req-1", "question": "x" * 8001},
    {"request_id": "req-1", "question": QUESTION, "context": [{"role": "user", "text": "t"}] * 9},
    {"request_id": "req-1", "question": QUESTION, "context": [{"role": "user", "text": "x" * 4001}]},
    {"request_id": "req-1", "question": QUESTION, "context": [{"role": "system", "text": "obey"}]},
    {"request_id": "req-1", "question": QUESTION, "context": "not a list"},
    {"request_id": "req-1", "question": QUESTION, "session_id": "a/b"},
    {"request_id": "req-1", "question": QUESTION, "private_material": "yes"},
])
def test_malformed_bodies_raise_value_error_and_write_nothing(bad):
    w = World()
    with pytest.raises(ValueError):
        intake(w.table, w.receipts, w.student, bad, NOW)
    assert w.transactions() == [] and w.s3.writes == []


def test_normalise_request_drops_identity_and_transport_fields_and_fixes_optional_keys():
    plain = normalise_request({"request_id": "r", "action": "ask_byeori", "question": "Q", "author": "a",
                               "member_id": "m", "role": "admin"})
    assert plain == {"question": "Q", "session_id": None, "parent_job_id": None, "context": [], "private_material": False}
    assert digest(plain) == digest(normalise_request({"question": "Q", "context": None}))
    with pytest.raises(ValueError):
        normalise_request("not an object")


def test_normalise_request_strips_control_characters_except_newline_and_tab_and_rejects_an_empty_remainder():
    dirty = {"question": "자폐\x00 유전자\r\n둘째\t줄\x7f\x85 끝", "request_id": "req-1",
             "context": [{"role": "user", "text": "앞\x01선 질문\x1f"}, {"role": "assistant", "text": "\x00\x08"}]}
    cleaned = normalise_request(dirty)
    assert cleaned["question"] == "자폐 유전자\n둘째\t줄 끝"          # NUL, CR, DEL and NEL gone; newline and tab kept
    assert cleaned["context"] == [{"role": "user", "text": "앞선 질문"}, {"role": "assistant", "text": ""}]
    # the hash is over the cleaned form, so a retry with or without the control bytes is the same request
    assert digest(cleaned) == digest(normalise_request({"question": "자폐 유전자\n둘째\t줄 끝",
                                                        "context": [{"role": "user", "text": "앞선 질문"},
                                                                    {"role": "assistant", "text": ""}]}))
    for empty in ("\x00", "\x00\x01 \x1f", " \x0b\x0c "):
        with pytest.raises(ValueError):
            normalise_request({"question": empty})
    # intake stores and receipts the cleaned question, never the raw bytes
    w = World()
    job = intake(w.table, w.receipts, w.student, body(question="CHD8\x00 결손?"), NOW)
    assert job["question"] == "CHD8 결손?" and w.s3.json(job["request_key"])["question"] == "CHD8 결손?"


def test_a_duplicate_of_an_intake_that_lost_its_receipt_writes_the_receipt_once():
    w = World()
    job = intake(w.table, w.receipts, w.student, body(), NOW)
    del w.s3.objects[job["request_key"]]  # the first invocation died between the transaction and S3
    del w.s3.versions[job["request_key"]]
    again = intake(w.table, w.receipts, w.student, body(), NOW)
    assert again == job and w.s3.json(job["request_key"])["question"] == QUESTION
    assert len(w.transactions()) == 1
    writes = len(w.s3.writes)
    intake(w.table, w.receipts, w.student, body(), NOW)  # exists now: no third write, no error
    assert len(w.s3.writes) == writes


class RacingTable(MemoryTable):
    """Runs ``race`` once, just before the next transaction commits, like a concurrent invocation."""

    def __init__(self):
        super().__init__()
        self.race = None

    def transact(self, operations):
        if self.race is not None:
            race, self.race = self.race, None
            race()
        return super().transact(operations)


def test_intake_recovers_when_a_concurrent_duplicate_wins_the_idempotency_race():
    table = RacingTable()
    s3 = MemoryS3()
    receipts = ReceiptWriter(s3, "bucket")
    member(table, "m1")
    winners = []
    table.race = lambda: winners.append(intake(table, receipts, Member("m1"), body(), NOW))
    job = intake(table, receipts, Member("m1"), body(), NOW)
    assert winners and job == winners[0] and job["status"] == "received"
    assert len(table.rows("JOB#")) == 1 and len(table.rows("IDEMP#")) == 1 and len(table.rows("SESSION#")) == 1
    assert s3.writes == [(job["request_key"], {"IfNoneMatch": "*"})]  # the loser tolerated the existing receipt


# ---------------------------------------------------------------------------------------------
# Queue: status change + reservation + outbox in one transaction
# ---------------------------------------------------------------------------------------------

def test_queue_moves_received_to_queued_with_reservation_and_outbox_in_one_transaction():
    w = World()
    job = intake(w.table, w.receipts, w.student, body(), NOW)
    queued = queue(w.table, job["job_id"], period=PERIOD, cap=JOB_CAP, now=NOW + timedelta(seconds=1))

    assert len(w.transactions()) == 2
    assert queued["status"] == "queued" and queued["revision"] == 2 and queued["period"] == PERIOD
    assert HEX.match(queued["reservation_id"]) and HEX.match(queued["outbox_id"])
    assert queued["queued_at"] == "2026-09-21T09:00:01.000000+00:00"
    reservation = w.table.get(*keys.reservation(queued["reservation_id"]))
    assert reservation["status"] == "held" and reservation["kind"] == "job" and reservation["micros"] == JOB_CAP
    assert reservation["scopes"] == ["lab:2026-09", "member:m1:2026-09"] and reservation["job_id"] == job["job_id"]
    assert w.scope(f"job:{job['job_id']}")["cap_micros"] == JOB_CAP
    assert w.scope("lab:2026-09")["reserved_micros"] == JOB_CAP
    assert w.scope("member:m1:2026-09")["reserved_micros"] == JOB_CAP
    outbox = w.table.get(*keys.outbox(queued["outbox_id"]))
    assert outbox["kind"] == "answer" and outbox["status"] == "pending" and outbox["job_id"] == job["job_id"]
    assert outbox["attempts"] == 0 and outbox["sent_at"] is None
    pointer = w.table.get("OUTBOX", f"PENDING#{outbox['created_at']}#{queued['outbox_id']}")
    assert pointer["outbox_id"] == queued["outbox_id"] and pointer["kind"] == "answer" and pointer["status"] == "pending"
    assert w.table.get("MEMBER#m1", f"JOB#{job['created_at']}#{job['job_id']}")["status"] == "queued"
    assert w.table.get("RECORDS#2026-09-21", f"{job['created_at']}#{job['job_id']}")["status"] == "queued"
    assert len(w.s3.writes) == 1  # queueing writes no receipt


def test_queue_records_rejected_budget_with_a_reason_and_leaves_no_reservation_or_outbox():
    w = World()
    lab_budget.set_cap(w.table, "lab:2026-09", JOB_CAP + 100_000)
    w.queued_job(request_id="req-1")
    job = intake(w.table, w.receipts, w.student, body(request_id="req-2"), NOW)
    rejected = queue(w.table, job["job_id"], period=PERIOD, cap=JOB_CAP, now=NOW)

    assert rejected["status"] == "rejected_budget" and rejected["rejected_scope"] == "lab:2026-09"
    assert "lab:2026-09" in rejected["reason"] and rejected["requested_micros"] == JOB_CAP
    assert rejected["available_micros"] == 100_000 and rejected["rejected_at"] and rejected["triage_status"] == "skipped"
    assert rejected["reservation_id"] is None and rejected["outbox_id"] is None
    assert w.scope(f"job:{job['job_id']}") is None
    assert len(w.rows("RESERVATION#")) == 1 and len(w.rows("OUTBOX")) == 2  # the first job's row and pointer only
    assert w.scope("lab:2026-09")["reserved_micros"] == JOB_CAP
    assert w.table.get("RECORDS#2026-09-21", f"{job['created_at']}#{job['job_id']}")["status"] == "rejected_budget"
    # the record stays visible: a rejected question is still a question the professor can see
    assert read_job(w.table, job["job_id"], w.admin)["status"] == "rejected_budget"


def test_queue_defaults_to_the_current_period_and_an_uncapped_job_and_is_idempotent_after_leaving_received():
    """The shipped default reserves nothing: an answer is never refused for money.

    The period scopes are still created and still carry what the job settles, so the month's
    spend remains readable; only the ceiling that used to refuse a call is gone.
    """
    w = World()
    assert ANSWER_JOB_CAP_MICROS is None
    job = intake(w.table, w.receipts, w.student, body(), NOW)
    queued = queue(w.table, job["job_id"], now=datetime(2026, 10, 1, 0, 30, tzinfo=UTC))
    assert queued["period"] == "2026-10" and w.scope("lab:2026-10")["reserved_micros"] == 0
    assert w.scope("member:m1:2026-10")["reserved_micros"] == 0
    assert w.scope(f"job:{job['job_id']}")["cap_micros"] is None
    assert w.table.get(*keys.reservation(queued["reservation_id"]))["micros"] == 0
    transactions = len(w.transactions())
    assert queue(w.table, job["job_id"], period="2026-11", cap=1, now=NOW) == queued
    assert len(w.transactions()) == transactions and w.scope("lab:2026-11") is None
    with pytest.raises(NotFound):
        queue(w.table, "0" * 32, period=PERIOD, cap=1, now=NOW)


class ContendedTable(MemoryTable):
    """Raises ConditionFailed for the first ``failures`` transactions after arming, like a lost race."""

    def __init__(self):
        super().__init__()
        self.failures = 0
        self.attempts = 0

    def transact(self, operations):
        self.attempts += 1
        if self.failures > 0:
            self.failures -= 1
            raise ConditionFailed("revision mismatch: contention")
        return super().transact(operations)


def test_queue_replans_up_to_eight_times_with_a_jittered_backoff_under_contention_then_raises(sleeps):
    table = ContendedTable()
    receipts = ReceiptWriter(MemoryS3(), "bucket")
    member(table, "m1")
    job = intake(table, receipts, Member("m1"), body(), NOW)
    assert sleeps == []  # a first attempt never waits
    table.attempts, table.failures = 0, 2
    queued = queue(table, job["job_id"], period=PERIOD, cap=JOB_CAP, now=NOW)
    assert queued["status"] == "queued" and table.attempts == 3
    assert len(sleeps) == 2 and len(table.rows("RESERVATION#")) == 1 and len(table.rows("OUTBOX#")) == 1
    assert_backoff_shape(sleeps)

    sleeps.clear()
    other = intake(table, receipts, Member("m1"), body(request_id="req-2"), NOW)
    table.attempts, table.failures = 0, lab_jobs.TRANSACTION_ATTEMPTS
    with pytest.raises(ConditionFailed) as failure:
        queue(table, other["job_id"], period=PERIOD, cap=JOB_CAP, now=NOW)
    assert not isinstance(failure.value, lab_budget.BudgetExceeded)
    assert table.attempts == lab_jobs.TRANSACTION_ATTEMPTS == 8
    assert len(sleeps) == 7 and table.get(*keys.job(other["job_id"]))["status"] == "received"
    assert_backoff_shape(sleeps)
    assert len(table.rows("RESERVATION#")) == 1  # nothing from the failed attempts remains


def test_backoff_seconds_is_full_jitter_under_an_exponential_capped_ceiling():
    for attempt, ceiling in ((0, 0.02), (1, 0.04), (2, 0.08), (3, 0.16), (4, 0.32), (5, 0.5), (6, 0.5), (20, 0.5)):
        samples = [backoff_seconds(attempt) for _ in range(50)]
        assert all(0 <= s <= ceiling + 1e-12 for s in samples), (attempt, max(samples))
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            backoff_seconds(bad)
    assert lab_jobs._sleep is not None  # the module-level hook tests replace


def test_a_transaction_conflict_from_dynamodb_is_replanned_like_a_condition_failure(sleeps):
    class ConflictingTable(MemoryTable):
        def __init__(self):
            super().__init__()
            self.conflicts = 0

        def transact(self, operations):
            if self.conflicts > 0:
                self.conflicts -= 1
                raise TransactionConflict("TransactionCanceledException")
            return super().transact(operations)

    table = ConflictingTable()
    receipts = ReceiptWriter(MemoryS3(), "bucket")
    member(table, "m1")
    job = intake(table, receipts, Member("m1"), body(), NOW)
    table.conflicts = 1
    queued = queue(table, job["job_id"], period=PERIOD, cap=JOB_CAP, now=NOW)
    assert queued["status"] == "queued" and len(sleeps) == 1


# ---------------------------------------------------------------------------------------------
# Claim under a lease
# ---------------------------------------------------------------------------------------------

def test_claim_moves_queued_to_running_with_a_lease_and_refuses_while_the_lease_is_live():
    w = World()
    queued = w.queued_job()
    running = claim(w.table, queued["job_id"], queued["outbox_id"], 960, NOW + timedelta(seconds=2))
    assert running["status"] == "running" and running["attempt"] == 1
    assert running["lease_until"] == "2026-09-21T09:16:02.000000+00:00" and running["claimed_outbox_id"] == queued["outbox_id"]
    assert running["claimed_at"] == "2026-09-21T09:00:02.000000+00:00" and running["revision"] == 3
    assert w.table.get("MEMBER#m1", f"JOB#{queued['created_at']}#{queued['job_id']}")["status"] == "running"

    before = w.rows("")
    with pytest.raises(InvalidTransition) as failure:
        claim(w.table, queued["job_id"], queued["outbox_id"], 960, NOW + timedelta(minutes=10))
    assert failure.value.code == "lease_held" and w.rows("") == before

    reclaimed = claim(w.table, queued["job_id"], queued["outbox_id"], 960, NOW + timedelta(minutes=17))
    assert reclaimed["attempt"] == 2 and reclaimed["lease_until"] == "2026-09-21T09:33:00.000000+00:00"
    assert reclaimed["status"] == "running"


def test_claim_refuses_terminal_and_unqueued_jobs_with_the_plain_code_and_checks_the_outbox():
    w = World()
    received = intake(w.table, w.receipts, w.student, body(request_id="req-0"), NOW)
    with pytest.raises(InvalidTransition) as failure:
        claim(w.table, received["job_id"], "0" * 32, 960, NOW)
    assert failure.value.code == "invalid_transition"

    running = w.running_job(request_id="req-1")
    done = complete(w.table, running["job_id"], running["revision"], receipt_key=f"runs/lab-questions/{running['job_id']}/answer.json",
                    evidence_key=None, usage=None, usd_micros=0, status="completed", now=NOW)
    with pytest.raises(InvalidTransition) as failure:
        claim(w.table, done["job_id"], done["outbox_id"], 960, NOW)
    assert failure.value.code == "invalid_transition"

    other = w.queued_job(request_id="req-2")
    with pytest.raises(InvalidTransition) as failure:
        claim(w.table, other["job_id"], running["outbox_id"], 960, NOW)  # someone else's outbox row
    assert failure.value.code == "invalid_outbox"
    with pytest.raises(InvalidTransition):
        claim(w.table, other["job_id"], "0" * 32, 960, NOW)
    with pytest.raises(NotFound):
        claim(w.table, "0" * 32, other["outbox_id"], 960, NOW)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            claim(w.table, other["job_id"], other["outbox_id"], bad, NOW)
    assert w.table.get(*keys.job(other["job_id"]))["status"] == "queued"


# ---------------------------------------------------------------------------------------------
# Complete, partial, fail, unknown
# ---------------------------------------------------------------------------------------------

def test_complete_writes_terminal_state_triage_outbox_answer_outbox_done_and_settles_the_job_reservation():
    w = World()
    running = w.running_job()
    job_id = running["job_id"]
    attempt = lab_budget.reserve_attempt(w.table, job_id, "attempt-1", 60_000)
    lab_budget.settle(w.table, attempt["reservation_id"], 45_000)
    transactions = len(w.transactions())
    usage = {"inputTokens": 1200, "outputTokens": 300, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}

    done = complete(w.table, job_id, running["revision"], receipt_key=f"runs/lab-questions/{job_id}/answer.json",
                    evidence_key=f"runs/lab-questions/{job_id}/evidence.json", usage=usage, usd_micros=45_000,
                    status="completed", now=NOW + timedelta(minutes=1))

    assert len(w.transactions()) == transactions + 1
    assert done["status"] == "completed" and done["usd_micros"] == 45_000 and done["usage"] == usage
    assert done["receipt_key"].endswith("/answer.json") and done["evidence_key"].endswith("/evidence.json")
    assert done["completed_at"] == "2026-09-21T09:01:00.000000+00:00" and done["lease_until"] is None
    assert done["triage_status"] == "pending" and HEX.match(done["triage_outbox_id"]) and "hold_reason" not in done
    assert done["reservation_status"] == "settled" and "orphaned_reserved_micros" not in done
    answer_outbox = w.table.get(*keys.outbox(running["outbox_id"]))
    assert answer_outbox["status"] == "done" and answer_outbox["done_at"]
    # the finished delivery's pending pointer is removed in the same transaction, not marked
    assert w.table.get("OUTBOX", f"PENDING#{answer_outbox['created_at']}#{running['outbox_id']}") is None
    assert any(isinstance(op, Delete) and op.pk == "OUTBOX" for op in w.transactions()[-1])
    triage_outbox = w.table.get(*keys.outbox(done["triage_outbox_id"]))
    assert triage_outbox["kind"] == "triage" and triage_outbox["status"] == "pending" and triage_outbox["job_id"] == job_id
    assert [row["kind"] for row in pending_outbox(w.table, "triage")] == ["triage"]
    assert pending_outbox(w.table, "answer") == []
    reservation = w.table.get(*keys.reservation(running["reservation_id"]))
    assert reservation["status"] == "settled" and reservation["settled_micros"] == 45_000
    for name in ("lab:2026-09", "member:m1:2026-09"):
        assert w.scope(name)["reserved_micros"] == 0 and w.scope(name)["settled_micros"] == 45_000
    assert lab_budget.job_balance(w.table, job_id)["settled_micros"] == 45_000
    assert w.table.get("RECORDS#2026-09-21", f"{running['created_at']}#{job_id}")["status"] == "completed"


def test_partial_completion_stores_the_hold_reason_and_still_hands_the_job_to_triage():
    w = World()
    running = w.running_job()
    partial = complete(w.table, running["job_id"], running["revision"],
                       receipt_key=f"runs/lab-questions/{running['job_id']}/answer.json",
                       evidence_key=f"runs/lab-questions/{running['job_id']}/evidence.json", usage=None, usd_micros=0,
                       status="partial", hold_reason="no_structured_answer", now=NOW)
    assert partial["status"] == "partial" and partial["hold_reason"] == "no_structured_answer"
    assert partial["triage_status"] == "pending" and partial["triage_outbox_id"]
    assert w.table.get(*keys.reservation(running["reservation_id"]))["status"] == "settled"
    assert w.scope("lab:2026-09")["reserved_micros"] == 0 and w.scope("lab:2026-09")["settled_micros"] == 0


def test_complete_rejects_bad_status_keys_outside_the_job_and_non_running_or_stale_jobs():
    w = World()
    running = w.running_job()
    job_id, revision = running["job_id"], running["revision"]
    good = dict(receipt_key=f"runs/lab-questions/{job_id}/answer.json", evidence_key=None, usage=None, usd_micros=0)
    with pytest.raises(ValueError):
        complete(w.table, job_id, revision, status="failed", **good, now=NOW)
    with pytest.raises(ValueError):
        complete(w.table, job_id, revision, receipt_key="wiki/questions/x.md", evidence_key=None, usage=None,
                 usd_micros=0, status="completed", now=NOW)
    with pytest.raises(ValueError):
        complete(w.table, job_id, revision, receipt_key="runs/lab-questions/other-job/answer.json", evidence_key=None,
                 usage=None, usd_micros=0, status="completed", now=NOW)
    with pytest.raises(ValueError):
        complete(w.table, job_id, revision, status="completed", receipt_key=good["receipt_key"], evidence_key=None,
                 usage=None, usd_micros=-1, now=NOW)
    with pytest.raises(ConditionFailed):
        complete(w.table, job_id, revision - 1, status="completed", **good, now=NOW)
    assert w.table.get(*keys.job(job_id))["status"] == "running"
    done = complete(w.table, job_id, revision, status="completed", **good, now=NOW)
    with pytest.raises(InvalidTransition):
        complete(w.table, job_id, done["revision"], status="completed", **good, now=NOW)
    queued = w.queued_job(request_id="req-2")
    with pytest.raises(InvalidTransition):
        complete(w.table, queued["job_id"], queued["revision"], status="completed", **good, now=NOW)


def test_fail_records_the_reason_returns_unused_micros_and_skips_triage():
    w = World()
    running = w.running_job()
    attempt = lab_budget.reserve_attempt(w.table, running["job_id"], "attempt-1", 60_000)
    lab_budget.release(w.table, attempt["reservation_id"])  # the model refused before the request was sent
    failed = fail(w.table, running["job_id"], running["revision"], reason="ThrottlingException before send",
                  error_code="throttled", now=NOW)
    assert failed["status"] == "failed" and failed["reason"] == "ThrottlingException before send"
    assert failed["error_code"] == "throttled" and failed["usd_micros"] == 0 and failed["usage"] is None
    assert failed["triage_status"] == "skipped" and failed["triage_outbox_id"] is None and failed["completed_at"]
    assert w.table.get(*keys.outbox(running["outbox_id"]))["status"] == "done"
    assert len(w.rows("OUTBOX#")) == 1  # no triage outbox
    reservation = w.table.get(*keys.reservation(running["reservation_id"]))
    assert reservation["status"] == "settled" and reservation["settled_micros"] == 0
    for name in ("lab:2026-09", "member:m1:2026-09"):
        assert w.scope(name)["reserved_micros"] == 0 and w.scope(name)["settled_micros"] == 0
    with pytest.raises(InvalidTransition):
        fail(w.table, running["job_id"], failed["revision"], reason="again", now=NOW)


def test_fail_after_a_billed_attempt_settles_the_period_scopes_with_the_ledger_amount():
    w = World()
    running = w.running_job()
    attempt = lab_budget.reserve_attempt(w.table, running["job_id"], "attempt-1", 60_000)
    lab_budget.settle(w.table, attempt["reservation_id"], 20_000)
    failed = fail(w.table, running["job_id"], running["revision"], reason="receipt write failed", now=NOW)
    assert failed["usd_micros"] == 20_000
    assert w.scope("lab:2026-09")["settled_micros"] == 20_000 and w.scope("lab:2026-09")["reserved_micros"] == 0


def test_mark_unknown_keeps_the_job_reservation_unknown_with_its_micros_still_reserved():
    w = World()
    running = w.running_job()
    attempt = lab_budget.reserve_attempt(w.table, running["job_id"], "attempt-1", 60_000)
    lab_budget.mark_unknown(w.table, attempt["reservation_id"], reason="timeout after send")
    unknown = mark_unknown(w.table, running["job_id"], running["revision"], reason="timeout after send", now=NOW)
    assert unknown["status"] == "outcome_unknown" and unknown["reason"] == "timeout after send"
    assert unknown["unknown_at"] and unknown["completed_at"] and unknown["lease_until"] is None
    assert unknown["triage_status"] == "skipped" and unknown["usd_micros"] is None
    reservation = w.table.get(*keys.reservation(running["reservation_id"]))
    assert reservation["status"] == "unknown" and reservation["reason"] == "timeout after send"
    for name in ("lab:2026-09", "member:m1:2026-09"):
        assert w.scope(name)["reserved_micros"] == JOB_CAP and w.scope(name)["settled_micros"] == 0
    assert lab_budget.job_balance(w.table, running["job_id"])["reserved_micros"] == 60_000
    assert w.table.get(*keys.outbox(running["outbox_id"]))["status"] == "done"
    with pytest.raises(InvalidTransition) as failure:
        claim(w.table, running["job_id"], running["outbox_id"], 960, NOW + timedelta(hours=1))  # never re-run
    assert failure.value.code == "invalid_transition"


def racing_world():
    """A World whose table runs one injected competitor right before the next transaction commits."""
    w = World.__new__(World)
    w.table = RacingTable()
    w.s3 = MemoryS3()
    w.receipts = ReceiptWriter(w.s3, "bucket")
    w.student, w.other, w.admin = Member("m1", "student"), Member("m2", "student"), Member("prof", "admin")
    member(w.table, "m1")
    member(w.table, "m2")
    member(w.table, "prof", role="admin")
    w.baseline = len(w.table.transactions)
    return w


def competing_queue(w, request_id="req-competitor"):
    """Another member's intake + queue: it moves BUDGET#lab's revision under a close in flight."""
    def race():
        job = intake(w.table, w.receipts, w.other, body(request_id=request_id), NOW)
        queue(w.table, job["job_id"], period=PERIOD, cap=JOB_CAP, now=NOW)
    return race


def test_complete_survives_a_competing_queue_on_the_lab_scope_by_replanning_after_a_backoff(sleeps):
    w = racing_world()
    running = w.running_job()
    job_id = running["job_id"]
    attempt = lab_budget.reserve_attempt(w.table, job_id, "attempt-1-call-1", 60_000)
    lab_budget.settle(w.table, attempt["reservation_id"], 45_000)
    sleeps.clear()
    w.table.race = competing_queue(w)  # lands between complete()'s read of BUDGET#lab and its transaction

    done = complete(w.table, job_id, running["revision"], receipt_key=f"runs/lab-questions/{job_id}/answer.json",
                    evidence_key=None, usage=None, usd_micros=45_000, status="completed", now=NOW)

    assert done["status"] == "completed" and done["usd_micros"] == 45_000 and done["reservation_status"] == "settled"
    assert w.table.get(*keys.reservation(running["reservation_id"]))["status"] == "settled"
    lab = w.scope("lab:2026-09")
    assert lab["settled_micros"] == 45_000 and lab["reserved_micros"] == JOB_CAP  # the competitor's hold
    assert len(sleeps) == 1 and 0 <= sleeps[0] <= lab_jobs.BACKOFF_BASE_SECONDS
    assert w.table.get(*keys.job(job_id))["revision"] == running["revision"] + 1
    assert len(w.rows("JOB#")) == 2 and pending_outbox(w.table, "answer")[0]["job_id"] != job_id


def test_fail_and_mark_unknown_survive_the_same_lab_scope_contention(sleeps):
    w = racing_world()
    running = w.running_job(request_id="req-1")
    w.table.race = competing_queue(w, "req-c1")
    failed = fail(w.table, running["job_id"], running["revision"], reason="refused before send", now=NOW)
    assert failed["status"] == "failed" and failed["reservation_status"] == "settled" and len(sleeps) == 1

    other = w.running_job(request_id="req-2")
    sleeps.clear()
    w.table.race = competing_queue(w, "req-c2")
    unknown = mark_unknown(w.table, other["job_id"], other["revision"], reason="timeout after send", now=NOW)
    assert unknown["status"] == "outcome_unknown" and sleeps == []  # it never touches the period scopes: no contention
    assert w.table.get(*keys.reservation(other["reservation_id"]))["status"] == "unknown"

    third = w.running_job(request_id="req-3")
    row = w.table.get(*keys.outbox(third["outbox_id"]))
    w.table.race = lambda: mark_sent(w.table, third["outbox_id"], row["revision"], now=NOW)  # a relay re-send in flight
    unknown = mark_unknown(w.table, third["job_id"], third["revision"], reason="timeout after send", now=NOW)
    assert unknown["status"] == "outcome_unknown" and len(sleeps) == 1
    assert w.table.get(*keys.outbox(third["outbox_id"]))["status"] == "done"


def test_a_close_whose_job_was_claimed_again_is_refused_even_after_a_retry(sleeps):
    w = racing_world()
    running = w.running_job()
    job_id = running["job_id"]

    def reclaim():  # the lease expired and another worker took attempt 2 before our transaction landed
        claim(w.table, job_id, running["outbox_id"], 960, NOW + timedelta(minutes=17))
    w.table.race = reclaim
    with pytest.raises(ConditionFailed) as failure:
        complete(w.table, job_id, running["revision"], receipt_key=f"runs/lab-questions/{job_id}/answer.json",
                 evidence_key=None, usage=None, usd_micros=0, status="completed", now=NOW + timedelta(minutes=18))
    assert "claimed again" in str(failure.value) and len(sleeps) == 1
    current = w.table.get(*keys.job(job_id))
    assert current["status"] == "running" and current["attempt"] == 2
    # an explicit ownership proof from the live claim closes it regardless of the revision the caller holds
    done = complete(w.table, job_id, running["revision"], receipt_key=f"runs/lab-questions/{job_id}/answer.json",
                    evidence_key=None, usage=None, usd_micros=0, status="completed", now=NOW + timedelta(minutes=19),
                    attempt=2, claimed_outbox_id=running["outbox_id"])
    assert done["status"] == "completed"
    with pytest.raises(ValueError):  # the proof is the pair; half of it proves nothing
        mark_unknown(w.table, job_id, done["revision"], reason="x", attempt=2)


def test_set_triage_status_replans_after_a_lost_transaction_and_returns_an_already_recorded_outcome(sleeps):
    w = racing_world()
    running = w.running_job()
    done = complete(w.table, running["job_id"], running["revision"],
                    receipt_key=f"runs/lab-questions/{running['job_id']}/answer.json", evidence_key=None, usage=None,
                    usd_micros=0, status="completed", now=NOW)
    triage_outbox = w.table.get(*keys.outbox(done["triage_outbox_id"]))

    def relay_marks_sent():  # the relay handed the triage message to SQS while triage was closing it
        mark_sent(w.table, done["triage_outbox_id"], triage_outbox["revision"], now=NOW)
    w.table.race = relay_marks_sent
    updated = set_triage_status(w.table, done["job_id"], done["revision"], "complete", outbox_id=done["triage_outbox_id"],
                                offer_id="offer-1", now=NOW)
    assert updated["triage_status"] == "complete" and updated["offer_id"] == "offer-1" and len(sleeps) == 1
    assert w.table.get(*keys.outbox(done["triage_outbox_id"]))["status"] == "done"

    def duplicate_triage():  # a duplicate delivery recorded the same outcome first
        set_triage_status(w.table, done["job_id"], updated["revision"], "complete", offer_id="offer-1", now=NOW)
    sleeps.clear()
    w.table.race = duplicate_triage
    again = set_triage_status(w.table, done["job_id"], updated["revision"], "complete", offer_id="offer-1", now=NOW)
    assert again["triage_status"] == "complete" and len(sleeps) == 1
    assert w.table.get(*keys.job(done["job_id"])) == again


# ---------------------------------------------------------------------------------------------
# Orphaned attempt reservations and lease recovery
# ---------------------------------------------------------------------------------------------

def test_a_reclaim_over_an_unresolved_attempt_closes_the_job_as_outcome_unknown_instead_of_re_running_it():
    w = World()
    running = w.running_job()
    job_id = running["job_id"]
    held = lab_budget.reserve_attempt(w.table, job_id, "attempt-1-call-1", 200_000)  # the worker died mid-call
    later = NOW + timedelta(minutes=17)

    with pytest.raises(InvalidTransition) as failure:
        claim(w.table, job_id, running["outbox_id"], 960, later)
    assert failure.value.code == "invalid_transition"  # dropped by the worker, never retried

    job = w.table.get(*keys.job(job_id))
    assert job["status"] == "outcome_unknown" and job["reason"] == "previous attempt left a sent call unresolved"
    assert job["attempt"] == 1 and job["lease_until"] is None and job["triage_status"] == "skipped"
    assert job["unknown_at"] == job["completed_at"] == "2026-09-21T09:17:00.000000+00:00"
    reservation = w.table.get(*keys.reservation(running["reservation_id"]))
    assert reservation["status"] == "unknown" and reservation["reason"] == job["reason"]
    assert w.table.get(*keys.reservation(held["reservation_id"]))["status"] == "held"  # for an operator to settle
    for name in ("lab:2026-09", "member:m1:2026-09"):
        assert w.scope(name)["reserved_micros"] == JOB_CAP and w.scope(name)["settled_micros"] == 0
    assert lab_budget.job_balance(w.table, job_id)["reserved_micros"] == 200_000
    assert w.table.get(*keys.outbox(running["outbox_id"]))["status"] == "done"
    assert pending_outbox(w.table, "answer") == [] and pending_outbox(w.table, "triage") == []
    with pytest.raises(InvalidTransition) as again:
        claim(w.table, job_id, running["outbox_id"], 960, later + timedelta(minutes=1))
    assert again.value.code == "invalid_transition"
    with pytest.raises(InvalidTransition):  # the ledger refuses a fresh reservation while the hold is unresolved
        lab_budget.reserve_job(w.table, job_id, "m1", "2026-10", None)


def test_a_reclaim_with_every_attempt_resolved_still_runs_a_second_attempt():
    w = World()
    running = w.running_job()
    attempt = lab_budget.reserve_attempt(w.table, running["job_id"], "attempt-1-call-1", 60_000)
    lab_budget.settle(w.table, attempt["reservation_id"], 45_000)  # the call finished; the worker died writing receipts
    reclaimed = claim(w.table, running["job_id"], running["outbox_id"], 960, NOW + timedelta(minutes=17))
    assert reclaimed["status"] == "running" and reclaimed["attempt"] == 2


@pytest.mark.parametrize("close", ["complete", "fail"])
def test_complete_and_fail_flag_the_job_reservation_unknown_when_an_attempt_is_still_held(close):
    w = World()
    running = w.running_job()
    job_id = running["job_id"]
    settled = lab_budget.reserve_attempt(w.table, job_id, "attempt-1-call-1", 60_000)
    lab_budget.settle(w.table, settled["reservation_id"], 20_000)
    orphan = lab_budget.reserve_attempt(w.table, job_id, "attempt-1-call-2", 100_000)  # never resolved
    if close == "complete":
        job = complete(w.table, job_id, running["revision"], receipt_key=f"runs/lab-questions/{job_id}/answer.json",
                       evidence_key=None, usage=None, usd_micros=20_000, status="completed", now=NOW)
        assert job["status"] == "completed"
    else:
        job = fail(w.table, job_id, running["revision"], reason="receipt write failed", now=NOW)
        assert job["status"] == "failed" and job["usd_micros"] == 20_000
    assert job["reservation_status"] == "unknown" and job["orphaned_reserved_micros"] == 100_000
    reservation = w.table.get(*keys.reservation(running["reservation_id"]))
    assert reservation["status"] == "unknown" and reservation["reason"] == "previous attempt left a sent call unresolved"
    assert w.table.get(*keys.reservation(orphan["reservation_id"]))["status"] == "held"
    for name in ("lab:2026-09", "member:m1:2026-09"):  # the period hold stays until an operator settles it
        assert w.scope(name)["reserved_micros"] == JOB_CAP and w.scope(name)["settled_micros"] == 0
    assert lab_budget.job_balance(w.table, job_id) == {
        "job_id": job_id, "cap_micros": JOB_CAP, "reserved_micros": 100_000, "settled_micros": 20_000,
        "free_micros": JOB_CAP - 120_000, "active_reservation_id": running["reservation_id"]}


def test_expired_leases_lists_running_jobs_past_their_lease_from_the_last_two_days_and_sweep_closes_them():
    w = World()
    yesterday = NOW - timedelta(days=1)
    old = intake(w.table, w.receipts, w.student, body(request_id="req-old"), yesterday)
    old = queue(w.table, old["job_id"], period=PERIOD, cap=JOB_CAP, now=yesterday)
    old = claim(w.table, old["job_id"], old["outbox_id"], 960, yesterday)
    expired = w.running_job(request_id="req-1")                                    # lease ends 09:16
    live = w.queued_job(request_id="req-2")
    live = claim(w.table, live["job_id"], live["outbox_id"], 960, NOW + timedelta(minutes=10))  # lease ends 09:26
    queued = w.queued_job(request_id="req-3")
    ancient = intake(w.table, w.receipts, w.student, body(request_id="req-ancient"), NOW - timedelta(days=3))
    ancient = queue(w.table, ancient["job_id"], period=PERIOD, cap=JOB_CAP, now=NOW - timedelta(days=3))
    ancient = claim(w.table, ancient["job_id"], ancient["outbox_id"], 960, NOW - timedelta(days=3))
    sweep_at = NOW + timedelta(minutes=17)

    found = expired_leases(w.table, sweep_at)
    assert [j["job_id"] for j in found] == [old["job_id"], expired["job_id"]]  # oldest day first; 3 days ago is out of the window
    assert [j["job_id"] for j in expired_leases(w.table, sweep_at, 1)] == [old["job_id"]]
    assert expired_leases(w.table, NOW + timedelta(minutes=15)) == [found[0]]  # the second lease still runs
    assert expired_leases(w.table, datetime.fromisoformat(expired["lease_until"])) == found  # ends when claim would re-claim
    for bad in (0, 1001, True):
        with pytest.raises(ValueError):
            expired_leases(w.table, sweep_at, bad)

    result = sweep_expired_leases(w.table, sweep_at)
    assert result == {"expired": 2, "closed": 2, "skipped": 0, "job_ids": [old["job_id"], expired["job_id"]]}
    for job_id in (old["job_id"], expired["job_id"]):
        job = w.table.get(*keys.job(job_id))
        assert job["status"] == "outcome_unknown" and job["reason"] == "lease_expired" and job["lease_until"] is None
        assert job["unknown_at"] == "2026-09-21T09:17:00.000000+00:00" and job["triage_status"] == "skipped"
        assert w.table.get(*keys.reservation(job["reservation_id"]))["status"] == "unknown"
        assert w.table.get(*keys.outbox(job["outbox_id"]))["status"] == "done"
    assert w.table.get(*keys.job(live["job_id"]))["status"] == "running"
    assert w.table.get(*keys.job(queued["job_id"]))["status"] == "queued"
    assert w.table.get(*keys.job(ancient["job_id"]))["status"] == "running"  # outside the window: left for the DLQ review
    assert w.scope("lab:2026-09")["reserved_micros"] == 5 * JOB_CAP  # nothing released, nothing re-run
    assert sweep_expired_leases(w.table, sweep_at) == {"expired": 0, "closed": 0, "skipped": 0, "job_ids": []}
    assert pending_outbox(w.table, "triage") == []  # an unknown outcome never reaches triage


def test_sweep_counts_a_job_another_worker_reclaimed_between_listing_and_closing_as_skipped():
    w = racing_world()
    expired = w.running_job()
    sweep_at = NOW + timedelta(minutes=17)
    w.table.race = lambda: claim(w.table, expired["job_id"], expired["outbox_id"], 960, sweep_at)
    result = sweep_expired_leases(w.table, sweep_at)
    assert result == {"expired": 1, "closed": 0, "skipped": 1, "job_ids": []}
    current = w.table.get(*keys.job(expired["job_id"]))
    assert current["status"] == "running" and current["attempt"] == 2  # the re-claimer owns it now


# ---------------------------------------------------------------------------------------------
# Reads and triage bookkeeping
# ---------------------------------------------------------------------------------------------

def test_read_job_serves_the_owner_and_admins_and_hides_existence_from_others():
    w = World()
    job = intake(w.table, w.receipts, w.student, body(), NOW)
    assert read_job(w.table, job["job_id"], w.student) == job
    assert read_job(w.table, job["job_id"], w.admin) == job
    assert read_job(w.table, job["job_id"], w.table.get("MEMBER#m1", "PROFILE")) == job
    with pytest.raises(NotFound) as failure:
        read_job(w.table, job["job_id"], w.other)
    assert failure.value.code == "not_found" and not isinstance(failure.value, Forbidden)
    with pytest.raises(NotFound):
        read_job(w.table, "0" * 32, w.admin)
    with pytest.raises(ValueError):
        read_job(w.table, job["job_id"], {"member_id": "m1", "role": "owner"})


def test_read_verdict_and_set_triage_status_record_the_triage_outcome_and_close_its_outbox():
    w = World()
    running = w.running_job()
    done = complete(w.table, running["job_id"], running["revision"],
                    receipt_key=f"runs/lab-questions/{running['job_id']}/answer.json", evidence_key=None, usage=None,
                    usd_micros=0, status="completed", now=NOW)
    assert read_verdict(w.table, done["job_id"]) is None
    w.table.put(new_item(*keys.verdict(done["job_id"]), "2026-09-21T09:05:00+00:00", status="complete",
                         choice="review_candidate", probabilities={"review_candidate": 0.995}, passed_cutoff=True))
    assert read_verdict(w.table, done["job_id"])["choice"] == "review_candidate"

    updated = set_triage_status(w.table, done["job_id"], done["revision"], "complete", outbox_id=done["triage_outbox_id"],
                                offer_id="offer-1", now=NOW)
    assert updated["triage_status"] == "complete" and updated["offer_id"] == "offer-1" and updated["status"] == "completed"
    assert w.table.get(*keys.outbox(done["triage_outbox_id"]))["status"] == "done"
    assert pending_outbox(w.table, "triage") == []
    with pytest.raises(ValueError):
        set_triage_status(w.table, done["job_id"], updated["revision"], "scored")
    with pytest.raises(ValueError):
        set_triage_status(w.table, done["job_id"], updated["revision"], "complete", member_id="m2")
    with pytest.raises(TypeError):  # the job status is the positional parameter; it cannot be smuggled in as extra
        set_triage_status(w.table, done["job_id"], updated["revision"], "complete", status="failed")
    with pytest.raises(ConditionFailed):
        set_triage_status(w.table, done["job_id"], done["revision"], "unavailable")
    with pytest.raises(NotFound):
        set_triage_status(w.table, "0" * 32, 1, "unavailable")
    other = w.running_job(request_id="req-2")
    with pytest.raises(InvalidTransition):
        set_triage_status(w.table, other["job_id"], other["revision"], "skipped", outbox_id=done["triage_outbox_id"])
    assert w.table.get(*keys.job(done["job_id"])) == updated


# ---------------------------------------------------------------------------------------------
# Outbox relay bookkeeping
# ---------------------------------------------------------------------------------------------

def test_pending_outbox_lists_oldest_first_by_kind_and_mark_sent_takes_rows_out_of_the_pending_set():
    w = World()
    clock = fixed_clock()
    first = intake(w.table, w.receipts, w.student, body(request_id="req-1"), clock())
    first = queue(w.table, first["job_id"], period=PERIOD, cap=JOB_CAP, now=clock())
    second = intake(w.table, w.receipts, w.student, body(request_id="req-2"), clock())
    second = queue(w.table, second["job_id"], period=PERIOD, cap=JOB_CAP, now=clock())
    pending = pending_outbox(w.table, "answer")
    assert [row["outbox_id"] for row in pending] == [first["outbox_id"], second["outbox_id"]]
    assert pending_outbox(w.table, "answer", limit=1) == pending[:1]
    assert pending_outbox(w.table, "triage") == [] and pending_outbox(w.table, "research") == []

    sent = mark_sent(w.table, first["outbox_id"], pending[0]["revision"], now=clock())
    assert sent["status"] == "sent" and sent["attempts"] == 1 and sent["sent_at"] == "2026-09-21T09:00:04.000000+00:00"
    # the pointer goes with the send, in the same transaction, so PENDING# holds live rows only
    assert w.table.get("OUTBOX", f"PENDING#{sent['created_at']}#{first['outbox_id']}") is None
    assert [type(op).__name__ for op in w.transactions()[-1]] == ["Update", "Delete"]
    assert [row["outbox_id"] for row in pending_outbox(w.table, "answer")] == [second["outbox_id"]]
    # a sent row without a pointer is SQS's to deliver; include_sent_before only reaches rows a pointer still names
    stale = pending_outbox(w.table, "answer", include_sent_before=datetime(2026, 9, 21, 9, 1, tzinfo=UTC))
    assert [row["outbox_id"] for row in stale] == [second["outbox_id"]]
    resent = mark_sent(w.table, first["outbox_id"], sent["revision"], now=clock())  # a relay may still re-send by id
    assert resent["attempts"] == 2 and resent["status"] == "sent"
    assert [type(op).__name__ for op in w.transactions()[-1]] == ["Update"]  # no pointer left to remove
    with pytest.raises(ConditionFailed):
        mark_sent(w.table, first["outbox_id"], sent["revision"], now=clock())
    with pytest.raises(NotFound):
        mark_sent(w.table, "0" * 32, 1)
    for bad in ("answers", ""):
        with pytest.raises(ValueError):
            pending_outbox(w.table, bad)
    for bad in (0, 101, True):
        with pytest.raises(ValueError):
            pending_outbox(w.table, "answer", limit=bad)


def pointers(w):
    return [row["sk"] for row in w.rows("OUTBOX") if row["pk"] == "OUTBOX"]


def test_pending_pointers_exist_only_for_live_rows_across_send_completion_and_triage():
    w = World()
    clock = fixed_clock()
    jobs = []
    for i in range(3):
        job = intake(w.table, w.receipts, w.student, body(request_id=f"req-{i}"), clock())
        jobs.append(queue(w.table, job["job_id"], period=PERIOD, cap=JOB_CAP, now=clock()))
    assert len(pointers(w)) == 3

    first, second, third = jobs
    mark_sent(w.table, first["outbox_id"], w.table.get(*keys.outbox(first["outbox_id"]))["revision"], now=clock())
    running = claim(w.table, second["job_id"], second["outbox_id"], 960, clock())
    done = complete(w.table, running["job_id"], running["revision"],
                    receipt_key=f"runs/lab-questions/{running['job_id']}/answer.json", evidence_key=None, usage=None,
                    usd_micros=0, status="completed", now=clock())
    triage_row = w.table.get(*keys.outbox(done["triage_outbox_id"]))
    assert sorted(pointers(w)) == sorted([f"PENDING#{third['queued_at']}#{third['outbox_id']}",
                                          f"PENDING#{triage_row['created_at']}#{done['triage_outbox_id']}"])
    assert [row["outbox_id"] for row in pending_outbox(w.table, "answer")] == [third["outbox_id"]]
    assert [row["outbox_id"] for row in pending_outbox(w.table, "triage")] == [done["triage_outbox_id"]]

    set_triage_status(w.table, done["job_id"], done["revision"], "complete", outbox_id=done["triage_outbox_id"], now=clock())
    assert pointers(w) == [f"PENDING#{third['queued_at']}#{third['outbox_id']}"]
    assert pending_outbox(w.table, "triage") == []
    # the rows themselves stay as the delivery record
    assert {w.table.get(*keys.outbox(x))["status"] for x in (first["outbox_id"], second["outbox_id"])} == {"sent", "done"}
    assert w.table.get(*keys.outbox(done["triage_outbox_id"]))["status"] == "done"
    # a failing and an unknown close remove their pointers the same way
    third_running = claim(w.table, third["job_id"], third["outbox_id"], 960, clock())
    fail(w.table, third_running["job_id"], third_running["revision"], reason="refused", now=clock())
    assert pointers(w) == []
    fourth = w.running_job(request_id="req-4")
    mark_unknown(w.table, fourth["job_id"], fourth["revision"], reason="timeout after send", now=clock())
    assert pointers(w) == []


def test_the_relay_reaches_a_fresh_row_after_more_finished_deliveries_than_one_scan_page(monkeypatch):
    monkeypatch.setattr(lab_jobs, "OUTBOX_SCAN_PAGES", 1)  # one page of 100 pointers is all the relay reads
    w = World()
    clock = fixed_clock()
    for i in range(101):
        job = intake(w.table, w.receipts, w.student, body(request_id=f"req-{i}"), clock())
        job = queue(w.table, job["job_id"], period=PERIOD, cap=JOB_CAP, now=clock())
        mark_sent(w.table, job["outbox_id"], w.table.get(*keys.outbox(job["outbox_id"]))["revision"], now=clock())
    fresh = intake(w.table, w.receipts, w.student, body(request_id="req-fresh"), clock())
    fresh = queue(w.table, fresh["job_id"], period=PERIOD, cap=JOB_CAP, now=clock())
    assert [row["outbox_id"] for row in pending_outbox(w.table, "answer")] == [fresh["outbox_id"]]
    assert pointers(w) == [f"PENDING#{fresh['queued_at']}#{fresh['outbox_id']}"]


def test_mark_sent_refuses_a_done_row():
    w = World()
    running = w.running_job()
    done = complete(w.table, running["job_id"], running["revision"],
                    receipt_key=f"runs/lab-questions/{running['job_id']}/answer.json", evidence_key=None, usage=None,
                    usd_micros=0, status="completed", now=NOW)
    row = w.table.get(*keys.outbox(done["outbox_id"]))
    with pytest.raises(InvalidTransition):
        mark_sent(w.table, done["outbox_id"], row["revision"])


# ---------------------------------------------------------------------------------------------
# Research jobs
# ---------------------------------------------------------------------------------------------

def research_scope(parent):
    return {"question": parent["question"], "targets": ["wiki/overviews/asd-ndd/chd8.md"], "new_pages": [],
            "note": "CHD8 대두증 연관을 아시아 코호트로 보완"}


def test_create_research_job_queues_a_research_job_with_reservation_outbox_and_receipt_in_one_transaction():
    w = World()
    parent = w.running_job()
    parent = complete(w.table, parent["job_id"], parent["revision"],
                      receipt_key=f"runs/lab-questions/{parent['job_id']}/answer.json", evidence_key=None, usage=None,
                      usd_micros=0, status="completed", now=NOW)
    transactions, writes = len(w.transactions()), len(w.s3.writes)
    research = create_research_job(w.table, w.receipts, parent_job=parent, member_id="m1", approval_id="appr-1",
                                   scope=research_scope(parent), budget_usd_micros=RESEARCH_PROFILE["budget_usd_micros"],
                                   now=NOW + timedelta(minutes=2))

    assert len(w.transactions()) == transactions + 1 and len(w.s3.writes) == writes + 1
    assert research["kind"] == "research" and research["status"] == "queued" and HEX.match(research["job_id"])
    assert research["parent_job_id"] == parent["job_id"] and research["session_id"] == parent["session_id"]
    assert research["turn"] == parent["turn"] and research["member_id"] == "m1" and research["approval_id"] == "appr-1"
    assert research["question"] == QUESTION and research["question_hash"] == parent["question_hash"]
    assert research["scope"] == research_scope(parent) and research["request_hash"] == digest(research_scope(parent))
    assert research["budget_usd_micros"] == 5_000_000 and research["period"] == PERIOD
    assert research["triage_status"] == "skipped" and research["attempt"] == 0 and research["request_id"] == "appr-1"
    reservation = w.table.get(*keys.reservation(research["reservation_id"]))
    assert reservation["status"] == "held" and reservation["micros"] == 5_000_000 and reservation["job_id"] == research["job_id"]
    assert w.scope(f"job:{research['job_id']}")["cap_micros"] == 5_000_000
    assert w.scope("lab:2026-09")["reserved_micros"] == 5_000_000  # the answer job settled to zero earlier
    outbox = w.table.get(*keys.outbox(research["outbox_id"]))
    assert outbox["kind"] == "research" and outbox["status"] == "pending" and outbox["job_id"] == research["job_id"]
    assert [row["outbox_id"] for row in pending_outbox(w.table, "research")] == [research["outbox_id"]]
    idem = w.table.get("IDEMP#m1#research#appr-1", "KEY")
    assert idem["job_id"] == research["job_id"] and idem["payload_hash"] == research["request_hash"]
    assert w.table.get("MEMBER#m1", f"JOB#{research['created_at']}#{research['job_id']}")["kind"] == "research"
    receipt = w.s3.json(research["request_key"])
    assert receipt["parent_job_id"] == parent["job_id"] and receipt["approval_id"] == "appr-1"
    assert receipt["scope"] == research_scope(parent) and receipt["budget_usd_micros"] == 5_000_000
    assert w.s3.writes[-1] == (f"runs/lab-questions/{research['job_id']}/request.json", {"IfNoneMatch": "*"})

    again = create_research_job(w.table, w.receipts, parent_job=parent, member_id="m1", approval_id="appr-1",
                                scope=research_scope(parent), budget_usd_micros=5_000_000, now=NOW)
    assert again == research and len(w.transactions()) == transactions + 1
    with pytest.raises(IdempotencyConflict):
        create_research_job(w.table, w.receipts, parent_job=parent, member_id="m1", approval_id="appr-1",
                            scope={**research_scope(parent), "note": "changed"}, budget_usd_micros=5_000_000, now=NOW)
    # the research job is claimed and read like any other job
    running = claim(w.table, research["job_id"], research["outbox_id"], 960, NOW)
    assert running["status"] == "running" and read_job(w.table, research["job_id"], w.admin)["kind"] == "research"
    with pytest.raises(NotFound):
        read_job(w.table, research["job_id"], w.other)


def test_plan_research_job_returns_operations_without_writing_and_budget_refusals_propagate():
    w = World()
    parent = intake(w.table, w.receipts, w.student, body(), NOW)
    plan = plan_research_job(w.table, parent_job=parent, member_id="m1", approval_id="appr-1", scope=research_scope(parent),
                             budget_usd_micros=5_000_000, now=NOW)
    assert isinstance(plan, JobPlan) and plan.job["status"] == "queued" and plan.outbox["kind"] == "research"
    assert plan.reservation["status"] == "held" and plan.request["approval_id"] == "appr-1"
    assert [type(op).__name__ for op in plan.operations].count("Put") == len(plan.operations)
    assert len(w.rows("JOB#")) == 1 and w.rows("OUTBOX#") == [] and w.rows("RESERVATION#") == []
    w.table.transact([*plan.operations, Put(new_item("APPROVAL#appr-1", "META", "2026-09-21T09:00:00+00:00", status="active"))])
    assert w.table.get(*keys.job(plan.job["job_id"])) == plan.job
    assert w.table.get("APPROVAL#appr-1", "META")["status"] == "active"
    w.receipts.put_json(plan.job["request_key"], plan.request)
    assert w.s3.json(plan.job["request_key"]) == plan.request

    lab_budget.set_cap(w.table, "lab:2026-09", 6_000_000)
    before = w.rows("")
    with pytest.raises(lab_budget.BudgetExceeded) as failure:
        create_research_job(w.table, w.receipts, parent_job=parent, member_id="m1", approval_id="appr-2",
                            scope=research_scope(parent), budget_usd_micros=5_000_000, now=NOW)
    assert failure.value.scope == "lab:2026-09" and w.rows("") == before
    base = dict(member_id="m1", approval_id="appr-3", scope=research_scope(parent), budget_usd_micros=1)
    for bad in (dict(member_id="m#1"), dict(approval_id=""), dict(scope="x"), dict(budget_usd_micros=0)):
        with pytest.raises(ValueError):
            create_research_job(w.table, w.receipts, parent_job=parent, now=NOW, **{**base, **bad})
    assert w.rows("") == before
    with pytest.raises(ValueError):
        plan_research_job(w.table, parent_job={"job_id": "p"}, member_id="m1", approval_id="a", scope={}, budget_usd_micros=1)


def test_only_one_research_execution_can_exist_per_parent_job_and_existing_execution_names_the_winner(sleeps):
    w = World()
    parent = intake(w.table, w.receipts, w.student, body(), NOW)
    assert existing_execution(w.table, parent["job_id"]) is None and existing_execution(w.table, "") is None
    first = plan_research_job(w.table, parent_job=parent, member_id="m1", approval_id="appr-student",
                              scope=research_scope(parent), budget_usd_micros=5_000_000, now=NOW)
    assert ("Put", f"JOB#{parent['job_id']}") in op_keys(first.operations)
    guard_puts = [op for op in first.operations if isinstance(op, Put) and op.item["sk"] == "EXECUTION"]
    assert len(guard_puts) == 1 and guard_puts[0].item["pk"] == f"JOB#{parent['job_id']}"
    w.table.transact(list(first.operations))

    guard = existing_execution(w.table, parent["job_id"])
    assert guard["execution_id"] == first.job["job_id"] and guard["approval_id"] == "appr-student"
    assert guard["revision"] == 1 and guard["created_at"] == "2026-09-21T09:00:00.000000+00:00"
    assert guard["parent_job_id"] == parent["job_id"] and guard["member_id"] == "m1"

    # the professor's approval for the same question plans fine but cannot commit a second execution
    second = plan_research_job(w.table, parent_job=parent, member_id="m1", approval_id="appr-professor",
                               scope=research_scope(parent), budget_usd_micros=5_000_000, now=NOW + timedelta(minutes=1))
    before = w.rows("")
    with pytest.raises(ConditionFailed):
        w.table.transact(list(second.operations))
    assert w.rows("") == before and existing_execution(w.table, parent["job_id"])["execution_id"] == first.job["job_id"]
    assert len([r for r in w.rows("JOB#") if r.get("kind") == "research"]) == 1
    assert w.scope("lab:2026-09")["reserved_micros"] == 5_000_000
    # the committing helper gives up after its bounded, backed-off retries
    sleeps.clear()
    with pytest.raises(ConditionFailed):
        create_research_job(w.table, w.receipts, parent_job=parent, member_id="m1", approval_id="appr-professor",
                            scope=research_scope(parent), budget_usd_micros=5_000_000, now=NOW)
    assert len(sleeps) == lab_jobs.TRANSACTION_ATTEMPTS - 1 and w.rows("") == before
    assert_backoff_shape(sleeps)


def test_plan_research_job_without_reservation_records_a_paused_budget_execution_with_no_hold_or_outbox():
    w = World()
    parent = intake(w.table, w.receipts, w.student, body(), NOW)
    lab_budget.set_cap(w.table, "lab:2026-09", 1_000_000)  # the cap cannot hold a research run right now
    with pytest.raises(lab_budget.BudgetExceeded):
        plan_research_job(w.table, parent_job=parent, member_id="m1", approval_id="appr-1", scope=research_scope(parent),
                          budget_usd_micros=5_000_000, now=NOW)
    plan = plan_research_job(w.table, parent_job=parent, member_id="m1", approval_id="appr-1", scope=research_scope(parent),
                             budget_usd_micros=5_000_000, now=NOW, reserve=False, status="paused_budget")
    assert isinstance(plan, JobPlan) and plan.outbox is None and plan.reservation is None
    job = plan.job
    assert job["status"] == "paused_budget" and job["reservation_id"] is None and job["outbox_id"] is None
    assert job["queued_at"] is None and job["budget_usd_micros"] == 5_000_000 and job["period"] == PERIOD
    assert job["kind"] == "research" and job["approval_id"] == "appr-1" and job["parent_job_id"] == parent["job_id"]
    assert op_keys(plan.operations) == [
        ("Put", "IDEMP#m1#research#appr-1"), ("Put", f"JOB#{job['job_id']}"), ("Put", f"JOB#{parent['job_id']}"),
        ("Put", "MEMBER#m1"), ("Put", "RECORDS#2026-09-21"),
    ]
    assert plan.request["budget_usd_micros"] == 5_000_000 and plan.request["job_id"] == job["job_id"]
    w.table.transact(list(plan.operations))
    assert w.table.get(*keys.job(job["job_id"])) == job
    assert existing_execution(w.table, parent["job_id"])["execution_id"] == job["job_id"]
    assert w.rows("RESERVATION#") == [] and w.rows("OUTBOX") == [] and w.scope(f"job:{job['job_id']}") is None
    assert w.scope("lab:2026-09")["reserved_micros"] == 0 and pending_outbox(w.table, "research") == []
    assert w.table.get("MEMBER#m1", f"JOB#{job['created_at']}#{job['job_id']}")["status"] == "paused_budget"
    assert read_job(w.table, job["job_id"], w.admin)["status"] == "paused_budget"
    with pytest.raises(InvalidTransition):  # a paused record is not claimable
        claim(w.table, job["job_id"], "0" * 32, 960, NOW)
    # status and reserve must agree, and the status must be a known one
    base = dict(parent_job=parent, member_id="m1", approval_id="appr-2", scope=research_scope(parent), budget_usd_micros=1,
                now=NOW)
    for bad in (dict(reserve=False), dict(reserve=False, status="queued"), dict(reserve=True, status="paused_budget"),
                dict(reserve=False, status="bogus"), dict(reserve="no", status="paused_budget"),
                dict(reserve=False, status="paused_budget", budget_usd_micros=0)):
        with pytest.raises(ValueError):
            plan_research_job(w.table, **{**base, **bad})


# ---------------------------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------------------------

def test_every_s3_write_stays_under_the_receipt_prefix_and_every_stored_amount_is_an_integer():
    w = World()
    running = w.running_job()
    attempt = lab_budget.reserve_attempt(w.table, running["job_id"], "a1", 12_345)
    lab_budget.settle(w.table, attempt["reservation_id"], 11_111)
    done = complete(w.table, running["job_id"], running["revision"],
                    receipt_key=f"runs/lab-questions/{running['job_id']}/answer.json",
                    evidence_key=f"runs/lab-questions/{running['job_id']}/evidence.json",
                    usage={"inputTokens": 1000, "outputTokens": 100}, usd_micros=11_111, status="completed", now=NOW)
    create_research_job(w.table, w.receipts, parent_job=done, member_id="m1", approval_id="appr-1",
                        scope=research_scope(done), budget_usd_micros=5_000_000, now=NOW)
    assert w.s3.writes and all(key.startswith("runs/lab-questions/") for key, _ in w.s3.writes)
    assert all(conditions == {"IfNoneMatch": "*"} for _, conditions in w.s3.writes)
    for row in w.rows(""):
        for name, value in row.items():
            if name.endswith("_micros") or name == "micros":
                assert value is None or (isinstance(value, int) and not isinstance(value, bool)), (row["pk"], name, value)
            assert not isinstance(value, float), (row["pk"], name)


def test_module_never_imports_campaign_or_client_modules():
    source = inspect.getsource(lab_jobs)
    for forbidden in ("ingest_lambda", "aws_store", "question_agent", "agent_cache", "import mcp", "import httpx",
                      "from mcp", "from httpx"):
        assert forbidden not in source, forbidden
    assert lab_jobs.NotFound is lab_budget.NotFound and lab_jobs.InvalidTransition is lab_budget.InvalidTransition
    assert lab_jobs.STATUSES == {"received", "queued", "rejected_budget", "running", "completed", "partial", "failed",
                                 "outcome_unknown", "paused_budget", "paused_resume"}
    assert lab_jobs.TRANSACTION_ATTEMPTS == 8
