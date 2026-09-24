"""Professor record queries, research candidates, decisions and aggregation rounds of byeori.lab_review."""
from __future__ import annotations

import base64
import inspect
import json
import re
from datetime import UTC, datetime, timedelta

import pytest

from byeori import lab_budget, lab_jobs, lab_offers, lab_policy, lab_review
from byeori.lab_jobs import Forbidden, IdempotencyConflict, InvalidTransition, Member, NotFound, claim, complete, intake, queue
from byeori.lab_policy import LEASE_SECONDS, OFFER_TTL_SECONDS, POLICY_REVISION, RESEARCH_PROFILE
from byeori.lab_review import (
    RevisionConflict,
    decide_research_candidate,
    list_question_records,
    list_research_candidates,
    propose_research_from_records,
    review_probability,
    run_aggregation_round,
)
from byeori.lab_store import Put, ReceiptWriter, digest, keys, new_id, new_item, now_iso
from lab_fakes import MemoryS3, MemoryTable, member

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
DAY1, DAY2, DAY3 = NOW - timedelta(days=2), NOW - timedelta(days=1), NOW
QUESTION = "CHD8 결손은 대두증과 연관되지 않는가? 아시아 코호트(n = 120)에서 확인된 결과만 답해 주세요."
HEX = re.compile(r"^[0-9a-f]{32}$")
SCOPE = {"question": "CHD8 결손과 대두증의 연관을 아시아 코호트로 보완", "targets": ["wiki/overviews/asd-ndd/chd8.md"],
         "new_pages": [], "note": "학생 세 명의 질문이 같은 공백을 가리킨다."}
LINK_NOTE = "linked to an existing execution"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Record the jittered pauses between transaction attempts instead of sleeping through them."""
    pauses: list[float] = []
    monkeypatch.setattr(lab_jobs, "_sleep", pauses.append)
    return pauses


class World:
    """One control table, one bucket, two students and one administrator."""

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
        self.counter = 0

    def transactions(self):
        return self.table.transactions

    def rows(self, prefix):
        return self.table.rows(prefix)

    def answered(self, who=None, *, when=NOW, question=QUESTION, status="completed"):
        """A question job that went received -> queued -> running -> completed at ``when`` (plus a distinct second)."""
        self.counter += 1
        when = when + timedelta(seconds=self.counter)
        who = who or self.student
        job = intake(self.table, self.receipts, who, {"request_id": f"req-{self.counter}", "question": question}, when)
        job = queue(self.table, job["job_id"], period="2026-09", now=when)
        job = claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, when)
        return complete(self.table, job["job_id"], job["revision"],
                        receipt_key=f"runs/lab-questions/{job['job_id']}/answer.json", evidence_key=None, usage=None,
                        usd_micros=0, status=status, now=when)

    def orphaned(self, who=None, *, when=NOW):
        """A completed job whose worker left an attempt reservation held, so its job reservation is flagged unknown."""
        self.counter += 1
        when = when + timedelta(seconds=self.counter)
        who = who or self.student
        job = intake(self.table, self.receipts, who, {"request_id": f"req-{self.counter}", "question": QUESTION}, when)
        job = queue(self.table, job["job_id"], period="2026-09", now=when)
        job = claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, when)
        lab_budget.reserve_attempt(self.table, job["job_id"], "attempt-1-call-1", 200_000, now=when)
        return complete(self.table, job["job_id"], job["revision"],
                        receipt_key=f"runs/lab-questions/{job['job_id']}/answer.json", evidence_key=None, usage=None,
                        usd_micros=0, status="completed", now=when)

    def received(self, who=None, *, when=NOW):
        self.counter += 1
        when = when + timedelta(seconds=self.counter)
        return intake(self.table, self.receipts, who or self.student, {"request_id": f"req-{self.counter}", "question": QUESTION}, when)

    def consented(self, job, *, request_id="acc-1"):
        """The student's own path: a passing verdict, a new_synthesis offer and its explicit acceptance."""
        stored = self.table.get(*keys.job(job["job_id"]))
        offer = lab_offers.issue(self.table, self.receipts, stored, lab_jobs.read_verdict(self.table, job["job_id"]),
                                 "new_synthesis", [], {}, NOW)
        body = {"request_id": request_id, "offer_id": offer["offer_id"], "revision": offer["revision"],
                "hash": offer["hash"], "decision": "accept"}
        return lab_offers.respond(self.table, self.receipts, Member(job["member_id"]), body, NOW)

    def research_jobs(self):
        return [row for row in self.rows("JOB#") if row["sk"] == "META" and row.get("kind") == "research"]

    def lab_reserved(self):
        return self.table.get(*keys.budget("lab:2026-09"))["reserved_micros"]

    def rejected(self, who=None, *, when=NOW):
        """A question the lab cap refused: status rejected_budget, never triaged.

        An answer job reserves nothing now, so a cap refuses once the period's recorded spend has
        reached it. A cap of zero is how an administrator stops the period outright.
        """
        job = self.received(who, when=when)
        lab_budget.set_cap(self.table, "lab:2026-09", 0)
        try:
            return queue(self.table, job["job_id"], period="2026-09", now=when)
        finally:
            lab_budget.set_cap(self.table, "lab:2026-09", None)

    def verdict(self, job, probabilities, *, choice="answer_only", status="complete", reason=None, candidate_status=None,
                confidence=0.8):
        """Store the triage record the way lab_triage does and mirror the job's triage_status."""
        passed = lab_policy.passes_cutoff(probabilities) if probabilities is not None else False
        self.table.put(new_item(*keys.verdict(job["job_id"]), now_iso(NOW), status=status, choice=choice,
                                probabilities=probabilities, confidence=confidence if probabilities is not None else None,
                                cutoff=lab_policy.REVIEW_CANDIDATE_CUTOFF, passed_cutoff=passed, input_hash="a" * 64,
                                model=lab_policy.JEV_MODEL, policy_revision=POLICY_REVISION, usage=None, usd_micros=None,
                                error_code=None if status != "unavailable" else "http_429", reason=reason,
                                reused_from_job_id=None, candidate_status=candidate_status))
        current = self.table.get(*keys.job(job["job_id"]))
        triage = "complete" if status in {"complete", "reused"} else status
        return lab_jobs.set_triage_status(self.table, job["job_id"], current["revision"], triage,
                                          outbox_id=current.get("triage_outbox_id"), now=NOW)

    def offer(self, job, status, *, execution_id=None, expires_at=None):
        offer_id = new_id()
        stamp = now_iso(NOW)
        self.table.put(new_item(*keys.offer(offer_id), stamp, offer_id=offer_id, job_id=job["job_id"],
                                member_id=job["member_id"], session_id=job["session_id"], kind="new_synthesis",
                                message="현재 검색한 위키에서는 이 질문을 종합한 문서를 찾지 못했습니다.", targets=[],
                                scope_check={}, hash="b" * 64, policy_revision=POLICY_REVISION,
                                expires_at=expires_at or now_iso(NOW + timedelta(seconds=OFFER_TTL_SECONDS)),
                                status=status, decision_at=None, execution_id=execution_id))
        self.table.put(new_item(*keys.job_offer(job["job_id"], offer_id), stamp, offer_id=offer_id, status=status))
        return offer_id


def probabilities(review):
    rest = round(1 - review, 6)
    return {"answer_only": round(rest * 0.6, 6), "needs_lookup": round(rest * 0.4, 6), "review_candidate": review}


def decode_cursor(cursor):
    return json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))


def op_keys(operations):
    return sorted((type(op).__name__, op.item["pk"] if isinstance(op, Put) else op.pk) for op in operations)


# ---------------------------------------------------------------------------------------------
# list_question_records
# ---------------------------------------------------------------------------------------------

def test_records_walk_the_window_day_by_day_and_join_job_verdict_and_offer():
    w = World()
    early = w.answered(when=DAY1)
    scored = w.verdict(early, probabilities(0.995), choice="review_candidate", candidate_status="review_candidate")
    offer_id = w.offer(early, "accepted", execution_id="e" * 32)
    unscored = w.answered(w.other, when=DAY2)
    w.verdict(unscored, None, choice=None, status="unavailable", reason="http_429")
    pending = w.answered(when=DAY3)
    before = w.answered(when=NOW - timedelta(days=5))   # outside the window

    result = list_question_records(w.table, from_day="2026-09-19", to_day="2026-09-21")
    ids = [r["job_id"] for r in result["records"]]
    assert ids == [early["job_id"], unscored["job_id"], pending["job_id"]] and before["job_id"] not in ids
    assert result["next_cursor"] is None

    first = result["records"][0]
    assert first["member_id"] == "m1" and first["status"] == "completed" and first["question"] == QUESTION
    assert first["standalone_question"] == QUESTION and first["created_at"] == early["created_at"]
    assert first["triage_status"] == "complete" and first["verdict_status"] == "complete"
    assert first["choice"] == "review_candidate" and first["probabilities"] == probabilities(0.995)
    assert first["probabilities"]["review_candidate"] == 0.995 and isinstance(first["probabilities"]["review_candidate"], float)
    assert first["confidence"] == 0.8 and first["passed_cutoff"] is True and first["candidate_status"] == "review_candidate"
    assert first["offer_id"] == offer_id and first["offer_status"] == "accepted" and first["execution_id"] == "e" * 32
    assert scored["triage_status"] == "complete"

    second = result["records"][1]
    assert second["member_id"] == "m2" and second["probabilities"] is None and second["choice"] is None
    assert second["triage_status"] == "unavailable" and second["verdict_status"] == "unavailable"
    assert second["passed_cutoff"] is False and second["offer_id"] is None and second["execution_id"] is None

    third = result["records"][2]
    assert third["triage_status"] == "pending" and third["verdict_status"] is None and third["probabilities"] is None


def test_records_never_expose_context_principals_or_table_keys_and_bound_the_question():
    w = World()
    context = [{"role": "user", "text": "앞선 질문은 CHD8 변이였습니다."}]
    job = intake(w.table, w.receipts, w.student, {"request_id": "ctx-1", "question": QUESTION, "context": context}, NOW)
    result = list_question_records(w.table, from_day="2026-09-21", to_day="2026-09-21")
    record = result["records"][0]
    assert record["job_id"] == job["job_id"] and record["status"] == "received" and record["standalone_question"] is None
    text = json.dumps(result, ensure_ascii=False)
    assert "앞선 질문" not in text and "arn:aws" not in text and "principal" not in text
    assert not {"pk", "sk", "context", "context_hash", "request_hash", "session_id"} & set(record)
    assert len(record["question"]) <= lab_jobs.QUESTION_MAX_CHARS


def test_probability_filters_compare_the_raw_review_candidate_value_and_keep_unscored_rows_only_on_request():
    w = World()
    below = w.answered(when=DAY1)
    w.verdict(below, probabilities(0.9899), choice="review_candidate")
    at_cutoff = w.answered(when=DAY1)
    w.verdict(at_cutoff, probabilities(0.99), choice="review_candidate")
    low = w.answered(when=DAY2)
    w.verdict(low, probabilities(0.05))
    skipped = w.answered(when=DAY2)
    w.verdict(skipped, None, choice=None, status="skipped", reason="links_only")
    untriaged = w.answered(when=DAY3)

    def ids(**filters):
        return [r["job_id"] for r in list_question_records(w.table, from_day="2026-09-19", to_day="2026-09-21", **filters)["records"]]

    assert ids(min_probability=0.99) == [at_cutoff["job_id"], skipped["job_id"], untriaged["job_id"]]
    assert ids(min_probability=0.99, include_unscored=False) == [at_cutoff["job_id"]]
    assert ids(max_probability=0.9899, include_unscored=False) == [below["job_id"], low["job_id"]]
    assert ids(min_probability=0.05, max_probability=0.9899, include_unscored=False) == [below["job_id"], low["job_id"]]
    assert ids(include_unscored=False) == [below["job_id"], at_cutoff["job_id"], low["job_id"]]
    assert ids() == [below["job_id"], at_cutoff["job_id"], low["job_id"], skipped["job_id"], untriaged["job_id"]]
    assert review_probability(w.table.get(*keys.verdict(below["job_id"]))) == 0.9899
    assert review_probability(w.table.get(*keys.verdict(skipped["job_id"]))) is None
    assert review_probability({"probabilities": {"review_candidate": True}}) is None
    assert review_probability({"probabilities": {"review_candidate": "0.99"}}) is None
    assert review_probability(None) is None


def test_status_and_member_filters_and_a_budget_rejection_surfaces_its_reason():
    w = World()
    done = w.answered(when=DAY1)
    partial = w.answered(w.other, when=DAY2, status="partial")
    rejected = w.rejected(w.other, when=DAY3)
    assert rejected["status"] == "rejected_budget"

    window = dict(from_day="2026-09-19", to_day="2026-09-21")
    assert [r["job_id"] for r in list_question_records(w.table, status="partial", **window)["records"]] == [partial["job_id"]]
    mine = list_question_records(w.table, member_id="m2", **window)["records"]
    assert [r["job_id"] for r in mine] == [partial["job_id"], rejected["job_id"]]
    assert [r["job_id"] for r in list_question_records(w.table, member_id="m1", **window)["records"]] == [done["job_id"]]

    record = list_question_records(w.table, status="rejected_budget", **window)["records"][0]
    assert record["job_id"] == rejected["job_id"] and record["triage_status"] == "skipped"
    assert record["rejected_scope"] == "lab:2026-09" and "cannot reserve" in record["reason"]
    assert record["probabilities"] is None and record["offer_status"] is None
    assert "reason" not in list_question_records(w.table, status="completed", **window)["records"][0]


def test_record_view_surfaces_triage_reasons_and_the_reservation_state():
    w = World()
    skipped = w.answered(when=DAY1)
    w.verdict(skipped, None, choice=None, status="skipped", reason="evidence_links_only", candidate_status="needs_lookup")
    unavailable = w.answered(w.other, when=DAY2)
    w.verdict(unavailable, None, choice=None, status="unavailable", reason="jev_error:http_429")
    scored = w.answered(when=DAY3)
    w.verdict(scored, probabilities(0.3))
    orphaned = w.orphaned(when=DAY3)
    assert orphaned["reservation_status"] == "unknown" and orphaned["orphaned_reserved_micros"] == 200_000

    records = list_question_records(w.table, from_day="2026-09-19", to_day="2026-09-21")["records"]
    view = {record["job_id"]: record for record in records}

    assert view[skipped["job_id"]]["triage_reason"] == "evidence_links_only"
    assert view[skipped["job_id"]]["triage_error_code"] is None
    assert view[unavailable["job_id"]]["triage_reason"] == "jev_error:http_429"
    assert view[unavailable["job_id"]]["triage_error_code"] == "http_429"
    assert not {"triage_reason", "triage_error_code"} & set(view[scored["job_id"]])
    assert view[skipped["job_id"]]["reservation_status"] == "settled"
    assert "orphaned_reserved_micros" not in view[skipped["job_id"]]
    assert view[orphaned["job_id"]]["reservation_status"] == "unknown"
    assert view[orphaned["job_id"]]["orphaned_reserved_micros"] == 200_000
    assert view[orphaned["job_id"]]["status"] == "completed"
    received = w.received()
    fresh = list_question_records(w.table, from_day="2026-09-21", to_day="2026-09-21", status="received")["records"][0]
    assert fresh["job_id"] == received["job_id"] and "reservation_status" not in fresh


def test_cursor_pages_across_days_and_is_opaque_base64url_json():
    w = World()
    jobs = [w.answered(when=DAY1), w.answered(when=DAY1), w.answered(w.other, when=DAY2), w.answered(when=DAY3), w.answered(when=DAY3)]
    window = dict(from_day="2026-09-19", to_day="2026-09-21")

    page1 = list_question_records(w.table, limit=2, **window)
    assert [r["job_id"] for r in page1["records"]] == [jobs[0]["job_id"], jobs[1]["job_id"]]
    cursor = page1["next_cursor"]
    assert isinstance(cursor, str) and "=" not in cursor and re.fullmatch(r"[A-Za-z0-9_-]+", cursor)
    assert decode_cursor(cursor) == {"day": "2026-09-19", "sk": f"{jobs[1]['created_at']}#{jobs[1]['job_id']}"}

    page2 = list_question_records(w.table, limit=2, cursor=cursor, **window)
    assert [r["job_id"] for r in page2["records"]] == [jobs[2]["job_id"], jobs[3]["job_id"]]
    assert decode_cursor(page2["next_cursor"])["day"] == "2026-09-21"
    page3 = list_question_records(w.table, limit=2, cursor=page2["next_cursor"], **window)
    assert [r["job_id"] for r in page3["records"]] == [jobs[4]["job_id"]] and page3["next_cursor"] is None

    exact = list_question_records(w.table, limit=5, **window)
    assert len(exact["records"]) == 5 and exact["next_cursor"] is None

    with pytest.raises(ValueError):
        list_question_records(w.table, cursor="not-a-cursor!", **window)
    with pytest.raises(ValueError):
        list_question_records(w.table, cursor=base64.urlsafe_b64encode(b'{"day": "2026-09-19"}').decode(), **window)
    outside = base64.urlsafe_b64encode(json.dumps({"day": "2026-09-01", "sk": "x"}).encode()).decode().rstrip("=")
    with pytest.raises(ValueError):
        list_question_records(w.table, cursor=outside, **window)


def test_scan_bound_hands_back_a_cursor_before_the_window_is_exhausted(monkeypatch):
    w = World()
    jobs = [w.answered(when=DAY1) for _ in range(3)] + [w.answered(when=DAY3)]
    monkeypatch.setattr(lab_review, "MAX_POINTERS_PER_CALL", 2)
    window = dict(from_day="2026-09-19", to_day="2026-09-21")
    first = list_question_records(w.table, status="partial", **window)
    assert first["records"] == [] and decode_cursor(first["next_cursor"])["sk"].endswith(jobs[1]["job_id"])
    second = list_question_records(w.table, cursor=first["next_cursor"], **window)
    assert [r["job_id"] for r in second["records"]] == [jobs[2]["job_id"], jobs[3]["job_id"]]
    # DynamoDB hands back a LastEvaluatedKey whenever a page is exactly full, so the scan may
    # end with a cursor whose next page is empty; clients loop until the cursor is None.
    if second["next_cursor"] is not None:
        third = list_question_records(w.table, cursor=second["next_cursor"], **window)
        assert third["records"] == [] and third["next_cursor"] is None


def test_record_query_rejects_malformed_windows_filters_and_limits():
    w = World()
    good = dict(from_day="2026-09-19", to_day="2026-09-21")
    for bad in (dict(from_day="2026-09-22", to_day="2026-09-21"), dict(from_day="2026-07-01", to_day="2026-09-21"),
                dict(from_day="20260919", to_day="2026-09-21"), dict(from_day="2026-13-01", to_day="2026-13-02"),
                dict(from_day=None, to_day="2026-09-21")):
        with pytest.raises(ValueError):
            list_question_records(w.table, **bad)
    assert len(lab_review._days("2026-07-22", "2026-09-21")) == 62
    for bad in (dict(status="done"), dict(member_id="m#1"), dict(min_probability=1.5), dict(max_probability=-0.1),
                dict(min_probability=0.9, max_probability=0.1), dict(min_probability=True), dict(min_probability="0.99"),
                dict(include_unscored="yes"), dict(limit=0), dict(limit=201), dict(limit=True), dict(cursor=12)):
        with pytest.raises(ValueError):
            list_question_records(w.table, **good, **bad)
    assert list_question_records(w.table, **good) == {"records": [], "next_cursor": None}


# ---------------------------------------------------------------------------------------------
# propose_research_from_records
# ---------------------------------------------------------------------------------------------

def test_propose_creates_candidate_pointer_and_idempotency_key_in_one_transaction_and_no_research_job():
    w = World()
    below = w.answered(when=DAY1)
    w.verdict(below, probabilities(0.42), choice="answer_only")
    declined = w.answered(w.other, when=DAY2)
    w.verdict(declined, probabilities(0.995), choice="review_candidate", candidate_status="review_candidate")
    declined_offer = w.offer(declined, "declined")
    accepted = w.answered(w.other, when=DAY2)
    w.verdict(accepted, probabilities(0.999), choice="review_candidate", candidate_status="review_candidate")
    accepted_offer = w.offer(accepted, "accepted", execution_id="e" * 32)
    before_jobs, before_tx, before_reservations = len(w.rows("JOB#")), len(w.transactions()), w.rows("RESERVATION#")
    body = {"request_id": "prop-1", "job_ids": [below["job_id"], declined["job_id"], accepted["job_id"], below["job_id"]],
            "scope": SCOPE}

    candidate = propose_research_from_records(w.table, w.receipts, w.admin, body, NOW)

    assert HEX.match(candidate["candidate_id"]) and candidate["status"] == "proposed" and candidate["revision"] == 1
    assert candidate["proposal_revision"] == 1 and candidate["created_by"] == "prof"
    assert candidate["job_ids"] == [below["job_id"], declined["job_id"], accepted["job_id"]]
    assert candidate["parent_job_id"] == below["job_id"]
    assert candidate["scope"] == SCOPE and candidate["policy_revision"] == POLICY_REVISION
    assert candidate["proposal_hash"] == digest({"job_ids": sorted(candidate["job_ids"]), "scope": SCOPE,
                                                 "policy_revision": POLICY_REVISION})
    assert candidate["linked_offer_ids"] == [declined_offer, accepted_offer]
    assert candidate["linked_execution_ids"] == ["e" * 32]
    assert candidate["budget_usd_micros"] == RESEARCH_PROFILE["budget_usd_micros"] == 5_000_000
    assert candidate["proposal"]["budget_usd_micros"] == 5_000_000 and candidate["proposal"]["max_calls"] == 40
    assert candidate["proposal"]["model_id"] is None
    summaries = candidate["proposal"]["questions"]
    assert [q["job_id"] for q in summaries] == candidate["job_ids"]
    assert summaries[0]["review_candidate"] == 0.42 and summaries[0]["offer_status"] is None
    assert summaries[1]["offer_status"] == "declined" and summaries[2]["offer_status"] == "accepted"
    assert all(len(q["question"]) <= lab_review.QUESTION_SUMMARY_CHARS for q in summaries)
    assert candidate["approval_id"] is None and candidate["execution_id"] is None and candidate["decision"] is None
    assert "pk" not in candidate and "sk" not in candidate

    assert len(w.transactions()) == before_tx + 1
    assert op_keys(w.transactions()[-1]) == sorted([("Put", f"CANDIDATE#{candidate['candidate_id']}"), ("Put", "CANDIDATES"),
                                                    ("Put", "IDEMP#prof#candidate#prop-1")])
    stored = w.table.get(*keys.candidate(candidate["candidate_id"]))
    assert {k: v for k, v in stored.items() if k not in {"pk", "sk"}} == candidate
    pointer = w.table.get("CANDIDATES", f"{candidate['created_at']}#{candidate['candidate_id']}")
    assert pointer["status"] == "proposed" and pointer["candidate_id"] == candidate["candidate_id"]
    idem = w.table.get("IDEMP#prof#candidate#prop-1", "KEY")
    assert idem["candidate_id"] == candidate["candidate_id"] and idem["kind"] == "candidate"
    assert len(w.rows("JOB#")) == before_jobs and w.rows("APPROVAL#") == [] and w.rows("RESERVATION#") == before_reservations
    assert all(row["kind"] != "research" for row in w.rows("OUTBOX#"))
    assert all(key.startswith("runs/lab-questions/") for key, _ in w.s3.writes)


def test_propose_is_idempotent_per_request_id_and_refuses_a_different_payload_or_unknown_job():
    w = World()
    job = w.answered(when=DAY1)
    body = {"request_id": "prop-1", "job_ids": [job["job_id"]], "scope": {"question": "Q", "targets": None}}
    first = propose_research_from_records(w.table, w.receipts, w.admin, body, NOW)
    assert first["scope"] == {"question": "Q", "targets": [], "new_pages": [], "note": None}
    count = len(w.transactions())
    again = propose_research_from_records(w.table, w.receipts, w.admin, {**body, "scope": {"question": "Q"}}, NOW + timedelta(hours=1))
    assert again == first and len(w.transactions()) == count
    with pytest.raises(IdempotencyConflict) as failure:
        propose_research_from_records(w.table, w.receipts, w.admin, {**body, "scope": {"question": "Other"}}, NOW)
    assert failure.value.code == "idempotency_conflict" and len(w.transactions()) == count
    with pytest.raises(NotFound):
        propose_research_from_records(w.table, w.receipts, w.admin, {"request_id": "prop-2", "job_ids": ["0" * 32], "scope": {}}, NOW)
    other_admin = {"member_id": "prof2", "role": "admin"}
    second = propose_research_from_records(w.table, w.receipts, other_admin, {**body, "request_id": "prop-9"}, NOW)
    assert second["candidate_id"] != first["candidate_id"] and second["created_by"] == "prof2"
    assert len(w.rows("CANDIDATE#")) == 2


def test_propose_validates_scope_job_ids_and_refuses_students():
    w = World()
    job = w.answered(when=DAY1)
    base = {"request_id": "prop-1", "job_ids": [job["job_id"]], "scope": SCOPE}
    with pytest.raises(Forbidden) as failure:
        propose_research_from_records(w.table, w.receipts, w.student, base, NOW)
    assert failure.value.code == "forbidden"
    with pytest.raises(Forbidden):
        propose_research_from_records(w.table, w.receipts, w.table.get("MEMBER#m2", "PROFILE"), base, NOW)
    with pytest.raises(ValueError):
        propose_research_from_records(w.table, w.receipts, {"member_id": "x", "role": "owner"}, base, NOW)
    for bad in (dict(request_id=None), dict(request_id="a/b"), dict(job_ids=[]), dict(job_ids="abc"),
                dict(job_ids=[job["job_id"]] + [f"j{i}" for i in range(25)]), dict(job_ids=[123]),
                dict(scope=None), dict(scope={"question": ""}), dict(scope={"budget_usd": 3}),
                dict(scope={"targets": ["papers/x.pdf"]}), dict(scope={"targets": ["wiki/drafts/x.md"]}),
                dict(scope={"new_pages": "wiki/overviews/x.md"}), dict(scope={"note": "n" * 4001}),
                dict(scope={"targets": [f"wiki/overviews/p{i}.md" for i in range(51)]})):
        with pytest.raises(ValueError):
            propose_research_from_records(w.table, w.receipts, w.admin, {**base, **bad}, NOW)
    assert w.rows("CANDIDATE#") == []
    assert propose_research_from_records(w.table, w.receipts, w.table.get("MEMBER#prof", "PROFILE"), base, NOW)["status"] == "proposed"


# ---------------------------------------------------------------------------------------------
# list_research_candidates
# ---------------------------------------------------------------------------------------------

def test_list_candidates_filters_by_status_and_pages_with_a_cursor():
    w = World()
    job = w.answered(when=DAY1)
    made = [propose_research_from_records(w.table, w.receipts, w.admin,
                                          {"request_id": f"p-{i}", "job_ids": [job["job_id"]], "scope": {"note": str(i)}},
                                          NOW + timedelta(minutes=i)) for i in range(3)]
    decide_research_candidate(w.table, w.receipts, w.admin, {"request_id": "d-1", "candidate_id": made[1]["candidate_id"],
                                                             "proposal_revision": 1, "proposal_hash": made[1]["proposal_hash"],
                                                             "decision": "reject"}, NOW)
    page = list_research_candidates(w.table, limit=2)
    assert [c["candidate_id"] for c in page["candidates"]] == [made[0]["candidate_id"], made[1]["candidate_id"]]
    assert page["candidates"][1]["status"] == "rejected" and decode_cursor(page["next_cursor"]) == {"sk": f"{made[1]['created_at']}#{made[1]['candidate_id']}"}
    rest = list_research_candidates(w.table, limit=2, cursor=page["next_cursor"])
    assert [c["candidate_id"] for c in rest["candidates"]] == [made[2]["candidate_id"]] and rest["next_cursor"] is None
    proposed = list_research_candidates(w.table, status="proposed")
    assert [c["candidate_id"] for c in proposed["candidates"]] == [made[0]["candidate_id"], made[2]["candidate_id"]]
    assert list_research_candidates(w.table, status="approved") == {"candidates": [], "next_cursor": None}
    assert all("pk" not in c for c in proposed["candidates"])
    for bad in (dict(status="open"), dict(limit=0), dict(cursor="???")):
        with pytest.raises(ValueError):
            list_research_candidates(w.table, **bad)


# ---------------------------------------------------------------------------------------------
# decide_research_candidate
# ---------------------------------------------------------------------------------------------

def proposed(w, *jobs, request_id="prop-1", scope=SCOPE):
    return propose_research_from_records(w.table, w.receipts, w.admin,
                                         {"request_id": request_id, "job_ids": [j["job_id"] for j in jobs], "scope": scope}, NOW)


def decision(candidate, decision="approve", *, request_id="dec-1", revision=None, proposal_hash=None):
    return {"request_id": request_id, "candidate_id": candidate["candidate_id"],
            "proposal_revision": candidate["proposal_revision"] if revision is None else revision,
            "proposal_hash": candidate["proposal_hash"] if proposal_hash is None else proposal_hash, "decision": decision}


def test_approve_commits_candidate_approval_research_job_reservation_outbox_and_idempotency_in_one_transaction():
    w = World()
    first = w.answered(w.other, when=DAY1)
    w.verdict(first, probabilities(0.3))
    second = w.answered(when=DAY2)
    candidate = proposed(w, first, second)
    tx_before, writes_before = len(w.transactions()), list(w.s3.writes)

    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)

    assert approved["status"] == "approved" and approved["decision"] == "approve" and approved["decided_by"] == "prof"
    assert approved["decided_at"] == now_iso(NOW) and approved["revision"] == 2 and approved["proposal_revision"] == 1
    execution_id, approval_id = approved["execution_id"], approved["approval_id"]
    assert HEX.match(execution_id) and HEX.match(approval_id) and approved["linked_execution_ids"] == [execution_id]

    research = w.table.get(*keys.job(execution_id))
    assert len(w.transactions()) == tx_before + 1
    ops = w.transactions()[-1]
    assert op_keys(ops) == sorted([
        ("Update", f"CANDIDATE#{candidate['candidate_id']}"), ("Update", "CANDIDATES"),
        ("Put", "IDEMP#prof#decision#dec-1"), ("Put", f"APPROVAL#{approval_id}"),
        ("Put", f"IDEMP#m2#research#{approval_id}"), ("Put", f"JOB#{execution_id}"), ("Put", "MEMBER#m2"),
        ("Put", f"JOB#{first['job_id']}"),  # the one-execution-per-parent guard item
        ("Put", "RECORDS#2026-09-21"), ("Put", f"BUDGET#job:{execution_id}"), ("Update", "BUDGET#lab:2026-09"),
        ("Update", "BUDGET#member:m2:2026-09"), ("Put", f"RESERVATION#{research['reservation_id']}"),
        ("Put", f"OUTBOX#{research['outbox_id']}"), ("Put", "OUTBOX"),
    ])
    assert w.table.get("CANDIDATES", f"{candidate['created_at']}#{candidate['candidate_id']}")["status"] == "approved"

    approval = w.table.get(*keys.approval(approval_id))
    assert approval["kind"] == "professor_approval" and approval["candidate_id"] == candidate["candidate_id"]
    assert approval["proposal_revision"] == 1 and approval["proposal_hash"] == candidate["proposal_hash"]
    assert approval["approved_by"] == "prof" and approval["policy_revision"] == POLICY_REVISION
    assert approval["scope"] == SCOPE and approval["budget_usd_micros"] == RESEARCH_PROFILE["budget_usd_micros"]
    assert approval["model_id"] is None and approval["max_calls"] == RESEARCH_PROFILE["max_calls"]
    assert approval["expires_at"] == now_iso(NOW + timedelta(seconds=OFFER_TTL_SECONDS))
    assert approval["execution_id"] == execution_id and approval["status"] == "active" and approval["revision"] == 1
    assert approval["parent_job_id"] == first["job_id"] and approval["member_id"] == "m2"
    assert "offer_id" not in approval

    research = w.table.get(*keys.job(execution_id))
    assert research["kind"] == "research" and research["status"] == "queued" and research["member_id"] == "m2"
    assert research["parent_job_id"] == first["job_id"] and research["approval_id"] == approval_id
    assert research["scope"] == SCOPE and research["question"] == SCOPE["question"]
    assert research["budget_usd_micros"] == 5_000_000 and research["triage_status"] == "skipped"
    reservation = w.table.get(*keys.reservation(research["reservation_id"]))
    assert reservation["status"] == "held" and reservation["micros"] == 5_000_000 and reservation["member_id"] == "m2"
    outbox = w.table.get(*keys.outbox(research["outbox_id"]))
    assert outbox["kind"] == "research" and outbox["status"] == "pending" and outbox["job_id"] == execution_id
    assert w.table.get("IDEMP#prof#decision#dec-1", "KEY")["candidate_id"] == candidate["candidate_id"]

    new_writes = w.s3.writes[len(writes_before):]
    assert new_writes == [(f"runs/lab-questions/{execution_id}/request.json", {"IfNoneMatch": "*"}),
                          (f"runs/lab-questions/{first['job_id']}/approval-{approval_id}.json", {"IfNoneMatch": "*"})]
    receipt = w.s3.json(new_writes[1][0])
    assert receipt["approval_id"] == approval_id and receipt["kind"] == "professor_approval"
    assert receipt["execution_id"] == execution_id and receipt["approved_by"] == "prof" and receipt["scope"] == SCOPE
    assert receipt["job_ids"] == [first["job_id"], second["job_id"]] and "pk" not in receipt
    request = w.s3.json(new_writes[0][0])
    assert request["job_id"] == execution_id and request["approval_id"] == approval_id and request["member_id"] == "m2"


def test_second_approval_of_the_same_revision_returns_the_existing_execution_and_writes_nothing():
    w = World()
    job = w.answered(when=DAY1)
    candidate = proposed(w, job)
    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)
    tx, writes, rows = len(w.transactions()), list(w.s3.writes), w.rows("")

    again = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, request_id="dec-2"), NOW + timedelta(hours=1))
    assert again["execution_id"] == approved["execution_id"] and again == approved
    same = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW + timedelta(hours=2))
    assert same == approved
    other_admin = {"member_id": "prof2", "role": "admin"}
    third = decide_research_candidate(w.table, w.receipts, other_admin, decision(candidate, request_id="dec-3"), NOW)
    assert third["execution_id"] == approved["execution_id"]
    assert len(w.transactions()) == tx and w.s3.writes == writes and w.rows("") == rows
    assert len([r for r in w.rows("JOB#") if r.get("kind") == "research"]) == 1

    with pytest.raises(InvalidTransition):
        decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, "reject", request_id="dec-4"), NOW)
    with pytest.raises(IdempotencyConflict):
        decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, "reject"), NOW)
    assert len(w.transactions()) == tx and w.rows("") == rows


def test_missing_receipts_of_an_approval_are_written_once_on_a_repeated_approval():
    w = World()
    job = w.answered(when=DAY1)
    candidate = proposed(w, job)
    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)
    approval = w.table.get(*keys.approval(approved["approval_id"]))
    research = w.table.get(*keys.job(approved["execution_id"]))
    for key in (approval["receipt_key"], research["request_key"]):   # simulate a crash between transaction and receipts
        del w.s3.objects[key]
    w.s3.writes.clear()
    decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, request_id="dec-2"), NOW)
    assert [key for key, _ in w.s3.writes] == [research["request_key"], approval["receipt_key"]]
    assert w.s3.json(research["request_key"])["scope_hash"] == research["request_hash"]
    decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, request_id="dec-3"), NOW)
    assert len(w.s3.writes) == 2


def test_stale_revision_or_hash_raises_revision_conflict_and_writes_nothing():
    w = World()
    job = w.answered(when=DAY1)
    candidate = proposed(w, job)
    tx, rows = len(w.transactions()), w.rows("")
    with pytest.raises(RevisionConflict) as failure:
        decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, proposal_hash="f" * 64), NOW)
    assert failure.value.code == "revision_conflict" and isinstance(failure.value, lab_review.StoreError)
    with pytest.raises(RevisionConflict):
        decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, revision=2), NOW)
    with pytest.raises(RevisionConflict):
        decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, "reject", proposal_hash="f" * 64), NOW)
    stale_policy = w.table.get(*keys.candidate(candidate["candidate_id"]))
    w.table.update(*keys.candidate(candidate["candidate_id"]), stale_policy["revision"], {"policy_revision": "2026-01-01-v0"})
    with pytest.raises(RevisionConflict):
        decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)
    assert len(w.transactions()) == tx + 1 and w.rows("APPROVAL#") == [] and w.rows("IDEMP#prof#decision") == []
    assert w.table.get(*keys.candidate(candidate["candidate_id"]))["status"] == "proposed"


def test_reject_records_the_decision_without_approval_job_or_reservation():
    w = World()
    job = w.answered(when=DAY1)
    candidate = proposed(w, job)
    jobs_before, tx, reservations = len(w.rows("JOB#")), len(w.transactions()), w.rows("RESERVATION#")
    rejected = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, "reject"), NOW)
    assert rejected["status"] == "rejected" and rejected["decision"] == "reject" and rejected["decided_by"] == "prof"
    assert rejected["execution_id"] is None and rejected["approval_id"] is None and rejected["decided_at"] == now_iso(NOW)
    assert len(w.transactions()) == tx + 1
    assert op_keys(w.transactions()[-1]) == sorted([("Update", f"CANDIDATE#{candidate['candidate_id']}"), ("Update", "CANDIDATES"),
                                                    ("Put", "IDEMP#prof#decision#dec-1")])
    assert len(w.rows("JOB#")) == jobs_before and w.rows("APPROVAL#") == [] and w.rows("RESERVATION#") == reservations
    assert w.table.get("CANDIDATES", f"{candidate['created_at']}#{candidate['candidate_id']}")["status"] == "rejected"
    assert decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, "reject", request_id="dec-2"), NOW) == rejected
    with pytest.raises(InvalidTransition):
        decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, request_id="dec-3"), NOW)
    assert len(w.transactions()) == tx + 1 and not w.s3.writes[len(w.s3.writes):]


def test_approval_over_the_lab_cap_records_the_decision_and_a_paused_research_job_without_a_reservation():
    w = World()
    job = w.answered(when=DAY1)
    candidate = proposed(w, job)
    lab_budget.set_cap(w.table, "lab:2026-09", 4_000_000)
    tx, writes, reservations = len(w.transactions()), len(w.s3.writes), w.rows("RESERVATION#")

    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)

    assert approved["status"] == "approved" and approved["decision"] == "approve" and approved["decided_by"] == "prof"
    assert approved["research_status"] == "paused_budget" and approved["execution_linked"] is False
    assert approved["rejected_scope"] == "lab:2026-09" and "cannot reserve 5000000" in approved["budget_reason"]
    execution_id, approval_id = approved["execution_id"], approved["approval_id"]
    assert HEX.match(execution_id) and HEX.match(approval_id) and approved["linked_execution_ids"] == [execution_id]

    research = w.table.get(*keys.job(execution_id))
    assert research["kind"] == "research" and research["status"] == "paused_budget" and research["member_id"] == "m1"
    assert research["reservation_id"] is None and research["outbox_id"] is None and research["queued_at"] is None
    assert research["budget_usd_micros"] == 5_000_000 and research["scope"] == SCOPE and research["approval_id"] == approval_id
    assert w.rows("RESERVATION#") == reservations and w.lab_reserved() == 0
    assert all(row["kind"] != "research" for row in w.rows("OUTBOX#"))
    assert lab_jobs.existing_execution(w.table, job["job_id"])["execution_id"] == execution_id

    approval = w.table.get(*keys.approval(approval_id))
    assert approval["kind"] == "professor_approval" and approval["execution_id"] == execution_id and approval["status"] == "active"
    assert approval["execution_status"] == "paused_budget" and approval["execution_linked"] is False
    assert approval["rejected_scope"] == "lab:2026-09" and approval["rejection_reason"] == approved["budget_reason"]
    assert approval["requested_micros"] == 5_000_000 and approval["available_micros"] == 4_000_000
    assert approval["budget_usd_micros"] == 5_000_000 and approval["max_calls"] == 40 and approval["reread"] == "auto"

    assert len(w.transactions()) == tx + 1
    assert op_keys(w.transactions()[-1]) == sorted([
        ("Update", f"CANDIDATE#{candidate['candidate_id']}"), ("Update", "CANDIDATES"),
        ("Put", "IDEMP#prof#decision#dec-1"), ("Put", f"APPROVAL#{approval_id}"),
        ("Put", f"IDEMP#m1#research#{approval_id}"), ("Put", f"JOB#{execution_id}"), ("Put", "MEMBER#m1"),
        ("Put", f"JOB#{job['job_id']}"), ("Put", "RECORDS#2026-09-21"),
    ])
    assert [key for key, _ in w.s3.writes[writes:]] == [research["request_key"], approval["receipt_key"]]
    assert w.s3.json(research["request_key"])["job_id"] == execution_id
    assert w.s3.json(approval["receipt_key"])["execution_status"] == "paused_budget"

    again = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, request_id="dec-2"), NOW)
    assert again == approved and len(w.transactions()) == tx + 1 and len(w.s3.writes) == writes + 2
    assert w.table.get("CANDIDATES", f"{candidate['created_at']}#{candidate['candidate_id']}")["status"] == "approved"


def test_approval_fixes_the_budget_and_profile_the_proposal_showed_not_the_current_constants(monkeypatch):
    w = World()
    job = w.answered(when=DAY1)
    candidate = proposed(w, job)
    assert candidate["proposal"]["max_calls"] == 40 and candidate["proposal"]["reread"] == "auto"
    monkeypatch.setattr(lab_review, "RESEARCH_PROFILE", {"budget_usd_micros": 50_000_000, "reread": "always", "max_calls": 400})

    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)

    approval = w.table.get(*keys.approval(approved["approval_id"]))
    assert approval["budget_usd_micros"] == 5_000_000 and approval["max_calls"] == 40 and approval["reread"] == "auto"
    research = w.table.get(*keys.job(approved["execution_id"]))
    assert research["budget_usd_micros"] == 5_000_000 and research["status"] == "queued"
    assert w.table.get(*keys.reservation(research["reservation_id"]))["micros"] == 5_000_000
    assert w.lab_reserved() == 5_000_000
    assert approved["research_status"] == "queued" and approved["execution_linked"] is False
    assert approval["execution_status"] == "queued" and approval["execution_linked"] is False and approval["note"] is None


def test_approval_links_the_execution_a_student_consent_already_started_instead_of_a_second_run():
    w = World()
    job = w.answered(when=DAY1)
    w.verdict(job, probabilities(0.995), choice="review_candidate", candidate_status="review_candidate")
    consent = w.consented(job)
    assert consent["status"] == "accepted" and w.lab_reserved() == 5_000_000
    candidate = proposed(w, job)
    assert candidate["linked_execution_ids"] == [consent["execution_id"]]
    tx, writes, reservations = len(w.transactions()), len(w.s3.writes), w.rows("RESERVATION#")

    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)

    assert approved["status"] == "approved" and approved["execution_id"] == consent["execution_id"]
    assert approved["execution_linked"] is True and approved["research_status"] == "queued"
    assert approved["linked_execution_ids"] == [consent["execution_id"]]
    assert len(w.research_jobs()) == 1 and w.research_jobs()[0]["job_id"] == consent["execution_id"]
    assert w.lab_reserved() == 5_000_000 and w.rows("RESERVATION#") == reservations
    assert w.table.get(*keys.budget("member:m1:2026-09"))["reserved_micros"] == 5_000_000
    assert [row["kind"] for row in w.rows("OUTBOX#")].count("research") == 1

    approval = w.table.get(*keys.approval(approved["approval_id"]))
    assert approval["kind"] == "professor_approval" and approval["execution_id"] == consent["execution_id"]
    assert approval["execution_linked"] is True and approval["note"] == LINK_NOTE and approval["execution_status"] == "queued"
    assert approval["budget_usd_micros"] == 5_000_000 and approval["status"] == "active"
    assert len(w.transactions()) == tx + 1
    assert op_keys(w.transactions()[-1]) == sorted([
        ("Update", f"CANDIDATE#{candidate['candidate_id']}"), ("Update", "CANDIDATES"),
        ("Put", "IDEMP#prof#decision#dec-1"), ("Put", f"APPROVAL#{approved['approval_id']}"),
    ])
    assert w.s3.writes[writes:] == [(approval["receipt_key"], {"IfNoneMatch": "*"})]
    assert w.s3.json(approval["receipt_key"])["note"] == LINK_NOTE

    again = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate, request_id="dec-2"), NOW)
    assert again == approved and len(w.transactions()) == tx + 1 and len(w.research_jobs()) == 1


def test_approval_links_an_execution_named_only_by_the_parent_job_guard():
    w = World()
    job = w.answered(when=DAY1)
    research = lab_jobs.create_research_job(w.table, w.receipts, parent_job=job, member_id="m1", approval_id=new_id(),
                                            scope={"question": QUESTION, "targets": [], "new_pages": [], "note": None},
                                            budget_usd_micros=5_000_000, now=NOW)
    candidate = proposed(w, job)
    assert candidate["linked_execution_ids"] == []   # no offer names it; only the JOB#{parent}/EXECUTION guard does
    tx = len(w.transactions())

    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)

    assert approved["execution_id"] == research["job_id"] and approved["execution_linked"] is True
    assert approved["linked_execution_ids"] == [research["job_id"]] and approved["research_status"] == "queued"
    assert len(w.research_jobs()) == 1 and w.lab_reserved() == 5_000_000 and len(w.transactions()) == tx + 1
    assert w.table.get(*keys.approval(approved["approval_id"]))["note"] == LINK_NOTE


def test_a_consent_that_lands_during_the_approval_transaction_is_linked_on_the_replan(no_backoff):
    w = World()
    job = w.answered(when=DAY1)
    candidate = proposed(w, job)
    original = w.table.transact
    state: dict = {"armed": False, "consent": None}

    def racing(operations):
        # The student's acceptance commits between the professor's plan and its transaction, once.
        if not state["armed"] and any(isinstance(op, Put) and op.item["sk"] == "EXECUTION" for op in operations):
            state["armed"] = True
            state["consent"] = lab_jobs.create_research_job(
                w.table, w.receipts, parent_job=job, member_id="m1", approval_id="student-consent",
                scope={"question": QUESTION, "targets": [], "new_pages": [], "note": None}, budget_usd_micros=5_000_000, now=NOW)
        return original(operations)

    w.table.transact = racing
    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)

    assert state["consent"] is not None and approved["execution_id"] == state["consent"]["job_id"]
    assert approved["execution_linked"] is True and approved["status"] == "approved"
    assert len(w.research_jobs()) == 1 and w.lab_reserved() == 5_000_000
    assert len(no_backoff) == 1 and 0 <= no_backoff[0] <= lab_jobs.BACKOFF_MAX_SECONDS
    assert w.table.get(*keys.approval(approved["approval_id"]))["note"] == LINK_NOTE
    assert len([r for r in w.rows("APPROVAL#")]) == 1


def test_decision_validates_its_body_and_refuses_students():
    w = World()
    job = w.answered(when=DAY1)
    candidate = proposed(w, job)
    good = decision(candidate)
    with pytest.raises(Forbidden):
        decide_research_candidate(w.table, w.receipts, w.student, good, NOW)
    with pytest.raises(NotFound):
        decide_research_candidate(w.table, w.receipts, w.admin, {**good, "candidate_id": "0" * 32}, NOW)
    for bad in (dict(request_id=None), dict(proposal_revision="1"), dict(proposal_revision=0), dict(proposal_revision=True),
                dict(proposal_hash="abc"), dict(proposal_hash=None), dict(decision="approved"), dict(decision=None)):
        with pytest.raises(ValueError):
            decide_research_candidate(w.table, w.receipts, w.admin, {**good, **bad}, NOW)
    with pytest.raises(ValueError):
        decide_research_candidate(w.table, w.receipts, w.admin, good, datetime(2026, 9, 21, 9, 0))
    assert w.table.get(*keys.candidate(candidate["candidate_id"]))["status"] == "proposed"
    with pytest.raises(ValueError):
        run_aggregation_round(w.table, cutoff=0.99, from_day="2026-09-19", to_day="2026-09-21", now=datetime(2026, 9, 21))


# ---------------------------------------------------------------------------------------------
# run_aggregation_round
# ---------------------------------------------------------------------------------------------

def test_aggregation_round_counts_the_window_lists_unprocessed_jobs_and_stores_a_round_without_jev_calls():
    w = World()
    passed_accepted = w.answered(when=DAY1)
    w.verdict(passed_accepted, probabilities(0.995), choice="review_candidate", candidate_status="review_candidate")
    w.offer(passed_accepted, "accepted", execution_id="e1" * 16)
    passed_declined = w.answered(w.other, when=DAY1)
    w.verdict(passed_declined, probabilities(0.99), choice="review_candidate", candidate_status="review_candidate")
    w.offer(passed_declined, "declined")
    passed_unanswered = w.answered(when=DAY2)
    w.verdict(passed_unanswered, probabilities(0.999), choice="review_candidate", candidate_status="review_candidate")
    w.offer(passed_unanswered, "offered")
    passed_lapsed = w.answered(when=DAY2)
    w.verdict(passed_lapsed, probabilities(0.9999), choice="review_candidate", candidate_status="review_candidate")
    w.offer(passed_lapsed, "offered", expires_at=now_iso(NOW - timedelta(seconds=1)))
    below = w.answered(when=DAY2)
    w.verdict(below, probabilities(0.9899), choice="review_candidate", candidate_status="unconfirmed_candidate")
    unavailable = w.answered(w.other, when=DAY3)
    w.verdict(unavailable, None, choice=None, status="unavailable", reason="http_429")
    skipped = w.answered(when=DAY3)
    w.verdict(skipped, None, choice=None, status="skipped", reason="links_only")
    pending = w.answered(when=DAY3)
    rejected = w.rejected(when=DAY3)
    outside = w.answered(when=NOW - timedelta(days=10))
    w.verdict(outside, probabilities(0.999), choice="review_candidate")
    candidate = proposed(w, below, passed_declined)
    approved = decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)
    earlier = propose_research_from_records(w.table, w.receipts, w.admin, {"request_id": "old", "job_ids": [outside["job_id"]], "scope": {}},
                                            NOW - timedelta(days=10))
    tx, candidates = len(w.transactions()), len(w.rows("CANDIDATE#"))

    record = run_aggregation_round(w.table, cutoff=0.99, from_day="2026-09-19", to_day="2026-09-21", now=NOW)

    assert record["counts"] == {"valid_verdicts": 5, "passed": 4, "offers": 4, "accepted": 1, "declined": 1, "unanswered": 1,
                                "expired": 1, "unavailable": 1, "skipped": 2, "executions": 2, "rejected_budget": 1}
    assert record["unprocessed_job_ids"] == [pending["job_id"]]
    # the approved research job shares today's day index but is an execution, not a question record
    listed = list_question_records(w.table, from_day="2026-09-19", to_day="2026-09-21")["records"]
    assert approved["execution_id"] not in {r["job_id"] for r in listed} and all(r["kind"] == "answer" for r in listed)
    assert len(listed) == 9
    assert record["candidate_ids"] == [candidate["candidate_id"]] and earlier["candidate_id"] not in record["candidate_ids"]
    assert record["cutoff"] == 0.99 and record["from"] == "2026-09-19" and record["to"] == "2026-09-21"
    assert record["policy_revision"] == POLICY_REVISION and record["jev_calls"] == 0 and record["complete"] is True
    assert record["cursor"] is None and record["jobs_seen"] == 9 and record["missing_jobs"] == 0
    assert record["executed_at"] == now_iso(NOW) and HEX.match(record["round_id"]) and record["revision"] == 1
    assert w.table.get(*keys.round(record["round_id"])) == record
    assert len(w.transactions()) == tx + 1 and op_keys(w.transactions()[-1]) == [("Put", f"ROUND#{record['round_id']}")]
    assert len(w.rows("CANDIDATE#")) == candidates and approved["execution_id"] in {"e1" * 16, approved["execution_id"]}
    assert "jev" not in inspect.signature(run_aggregation_round).parameters

    stricter = run_aggregation_round(w.table, cutoff=0.9999, from_day="2026-09-19", to_day="2026-09-21", now=NOW)
    assert stricter["counts"]["passed"] == 1 and stricter["counts"]["valid_verdicts"] == 5
    looser = run_aggregation_round(w.table, cutoff=0.5, from_day="2026-09-20", to_day="2026-09-21", now=NOW)
    assert looser["counts"]["passed"] == 3 and looser["counts"]["valid_verdicts"] == 3 and looser["counts"]["offers"] == 2
    for bad in (dict(cutoff=1.5), dict(cutoff="0.99"), dict(cutoff=True), dict(from_day="2026-09-22")):
        with pytest.raises(ValueError):
            run_aggregation_round(w.table, **{**dict(cutoff=0.99, from_day="2026-09-19", to_day="2026-09-21", now=NOW), **bad})


def test_aggregation_round_reports_an_incomplete_scan_with_its_cursor(monkeypatch):
    w = World()
    jobs = [w.answered(when=DAY1) for _ in range(3)]
    monkeypatch.setattr(lab_review, "MAX_ROUND_POINTERS", 2)
    record = run_aggregation_round(w.table, cutoff=0.99, from_day="2026-09-19", to_day="2026-09-21", now=NOW)
    assert record["complete"] is False and record["jobs_seen"] == 2
    assert decode_cursor(record["cursor"]) == {"day": "2026-09-19", "sk": f"{jobs[1]['created_at']}#{jobs[1]['job_id']}"}
    assert record["unprocessed_job_ids"] == [jobs[0]["job_id"], jobs[1]["job_id"]]


# ---------------------------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------------------------

def test_module_never_imports_campaign_client_or_parallel_modules_and_reuses_lab_jobs_errors():
    source = inspect.getsource(lab_review)
    for forbidden in ("ingest_lambda", "aws_store", "question_agent", "agent_cache", "import mcp", "from mcp",
                      "import httpx", "from httpx", "lab_triage", "lab_answer", "lab_gateway", "boto3"):
        assert forbidden not in source, forbidden
    assert lab_review.Forbidden is lab_jobs.Forbidden and lab_review.NotFound is lab_jobs.NotFound
    assert lab_review.IdempotencyConflict is lab_jobs.IdempotencyConflict
    assert lab_review.InvalidTransition is lab_jobs.InvalidTransition
    from byeori import lab_offers
    assert lab_review.RevisionConflict is lab_offers.RevisionConflict  # one owner for the conflict code
    assert issubclass(RevisionConflict, lab_review.StoreError) and RevisionConflict.code == "revision_conflict"
    assert set(lab_review.COUNT_KEYS) == {"valid_verdicts", "passed", "offers", "accepted", "declined", "unanswered", "expired",
                                          "unavailable", "skipped", "executions", "rejected_budget"}
    assert lab_review.MAX_WINDOW_DAYS == 62 and lab_review.MAX_LIMIT == 200


def test_every_s3_write_stays_under_the_receipt_prefix_and_no_stored_amount_is_a_float():
    w = World()
    job = w.answered(when=DAY1)
    w.verdict(job, probabilities(0.995), choice="review_candidate")
    candidate = proposed(w, job)
    decide_research_candidate(w.table, w.receipts, w.admin, decision(candidate), NOW)
    run_aggregation_round(w.table, cutoff=0.99, from_day="2026-09-19", to_day="2026-09-21", now=NOW)
    assert w.s3.writes and all(key.startswith("runs/lab-questions/") for key, _ in w.s3.writes)
    assert all(conditions == {"IfNoneMatch": "*"} for _, conditions in w.s3.writes)
    for row in w.rows(""):
        for name, value in row.items():
            if name.endswith("_micros") or name == "micros":
                assert value is None or (isinstance(value, int) and not isinstance(value, bool)), (row["pk"], name, value)
