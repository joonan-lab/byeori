"""Approved research worker: scope-checked wiki writes through the campaign engine (P4).

Every scenario runs ``lab_research.run_research`` over a claimed research job with in-memory
fakes: the control table, a bucket holding a small wiki with its BM25 index, and a scripted
Bedrock client. The tests pin what the design demands of the worker: no model call without a
valid approval, one attempt reservation settled from the returned usage, every write confined to
the approval's targets plus the new-page allowance, the answer kept in ``research.json`` unless the
approval publishes it, and money held (never released) when a call's bill is unknown.
"""
from __future__ import annotations

import copy
import itertools
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from botocore.exceptions import ClientError

from byeori import lab_budget, lab_jobs, lab_research
from byeori.lab_jobs import Member, claim, complete, create_research_job, intake, queue
from byeori.lab_policy import ANSWER_JOB_CAP_MICROS, APPROVAL_TTL_SECONDS, LEASE_SECONDS, POLICY_REVISION
from byeori.lab_research import OUT_OF_SCOPE, ScopedPublisher, converse_once, run_research, search_adapter
from byeori.lab_store import ReceiptWriter, keys, new_item, now_iso, receipt_key
from byeori.question_agent import question_key_for
from lab_fakes import MemoryTable, index_connection, member, source_note, wiki_with_index

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
MODEL = "global.anthropic.claude-opus-5"
INDEX_KEY = "index/wiki-index-v2.sqlite3"
QUESTION = "Was regional inheritance stable in the cohort?"
NOTE = "wiki/sources/paper-one.md"
OVERVIEW = "wiki/overviews/existing.md"
OTHER = "wiki/overviews/other.md"
CONCEPT = "wiki/concepts/regional-stability.md"
SECOND = "wiki/concepts/second-insight.md"
THIRD = "wiki/overviews/third-synthesis.md"
OVERVIEW_TEXT = "# Existing synthesis\n\nEarlier claim.\n\nUnrelated conclusion stays exactly as written.\n"
OTHER_TEXT = "# Another synthesis\n\nAn unrelated claim.\n"
NEW_CLAIM = "Regional inheritance was stable in 120 families [[sources/paper-one]]."
NEW_PAGE = "Regional stability is a distinct unit. [[sources/paper-one]] [[overviews/existing]]\n"
ANSWER = "Regional inheritance was stable in the cohort (n = 120, p = 0.01). [[sources/paper-one]]"
PAGES = {NOTE: source_note(), OVERVIEW: OVERVIEW_TEXT, OTHER: OTHER_TEXT}
BUDGET_MICROS = 5_000_000
ALLOWED_WRITE_PREFIXES = ("wiki/concepts/", "wiki/overviews/", "wiki/sources/", "wiki/index.md", "wiki/indexes/",
                          "runs/agents/", "runs/lab-questions/")
CALL_IDS = itertools.count()


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(lab_jobs, "_sleep", lambda seconds: None)


def turn(*calls, answer=None, usage=None):
    content = [{"text": answer}] if answer is not None else []
    for name, args in calls:
        content.append({"toolUse": {"toolUseId": f"call-{next(CALL_IDS)}", "name": name, "input": args}})
    return {"output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use" if calls else "end_turn",
            "usage": usage or {"inputTokens": 1200, "outputTokens": 300, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}}


def throttled():
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Converse")


class FakeBedrock:
    """``converse(**request)`` returns scripted responses or raises; every request is recorded."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def converse(self, **request):
        self.requests.append(copy.deepcopy(request))
        if not self.responses:
            raise AssertionError("FakeBedrock received more calls than scripted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def scope(**overrides):
    base = {"question": QUESTION, "targets": [OVERVIEW], "new_pages": [], "note": "offer text", "kind": "new_synthesis",
            "offer_id": "offer-1"}
    return {**base, **overrides}


class World:
    def __init__(self, pages=PAGES):
        self.table = MemoryTable()
        self.s3 = wiki_with_index(pages, INDEX_KEY)
        self.receipts = ReceiptWriter(self.s3, "bucket")
        member(self.table, "m1")
        self.index = (index_connection(self.s3.objects[INDEX_KEY]), self.s3.etag(INDEX_KEY))
        self.bedrock = FakeBedrock()
        self.parent = self._finished_parent()

    def _finished_parent(self) -> dict:
        job = intake(self.table, self.receipts, Member("m1"), {"request_id": "req-1", "question": QUESTION}, NOW)
        job = queue(self.table, job["job_id"], period="2026-09", cap=ANSWER_JOB_CAP_MICROS, now=NOW)
        job = claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, NOW)
        receipt = self.receipts.put_json(receipt_key(job["job_id"], "answer.json"), {"answer": "short answer"})
        usage = {"inputTokens": 100, "outputTokens": 20}
        return complete(self.table, job["job_id"], job["revision"], receipt_key=receipt["key"], evidence_key=None,
                        usage=usage, usd_micros=1000, status="completed", now=NOW)

    def research_job(self, research_scope=None, *, approval_id="appr-1", approval=True, **approval_overrides) -> dict:
        """A claimed (running) research job with its approval record."""
        research_scope = scope() if research_scope is None else research_scope
        job = create_research_job(self.table, self.receipts, parent_job=self.parent, member_id="m1",
                                  approval_id=approval_id, scope=research_scope, budget_usd_micros=BUDGET_MICROS, now=NOW)
        if approval:
            self.table.put(self.approval(job, research_scope, **approval_overrides))
        return claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, NOW)

    def approval(self, job, research_scope, *, status="active", expires_in=APPROVAL_TTL_SECONDS,
                 policy_revision=POLICY_REVISION, execution_id=None, parent_job_id=None, kind="student_consent",
                 reread="auto") -> dict:
        stamp = now_iso(NOW)
        return new_item(
            *keys.approval(job["approval_id"]), stamp,
            approval_id=job["approval_id"], kind=kind, offer_id=research_scope.get("offer_id"), candidate_id=None,
            job_id=parent_job_id or self.parent["job_id"], proposal_revision=1, proposal_hash="hash", approved_by="m1",
            policy_revision=policy_revision, scope=research_scope, budget_usd_micros=BUDGET_MICROS, model_id=None,
            max_calls=40, reread=reread, expires_at=now_iso(NOW + timedelta(seconds=expires_in)),
            execution_id=execution_id or job["job_id"], status=status, approved_at=stamp, request_id="resp-1",
            research_status="queued", note=None, budget_refusal=None, linked_approval_id=None,
            receipt_key=receipt_key(self.parent["job_id"], f"approval-{job['approval_id']}.json"),
        )

    def run(self, job, *, now=NOW, reasoning="xhigh", remaining_ms=None):
        return run_research(job, table=self.table, receipts=self.receipts, s3=self.s3, bucket="bucket", index=self.index,
                            model_client=self.bedrock, converse=converse_once, model_id=MODEL, reasoning=reasoning,
                            now=now, remaining_ms=remaining_ms)

    # reads ------------------------------------------------------------------------------------
    def job(self, job_id):
        return self.table.get(*keys.job(job_id))

    def reservation(self, reservation_id):
        return self.table.get(*keys.reservation(reservation_id))

    def attempt_reservations(self):
        return [row for row in self.table.rows("RESERVATION#") if row.get("kind") == "attempt"]

    def receipt(self, job_id):
        return self.receipts.get_json(receipt_key(job_id, "research.json"))

    def approval_record(self, approval_id="appr-1"):
        return self.table.get(*keys.approval(approval_id))

    def written_keys(self):
        return [key for key, _ in self.s3.writes]


def tool_results(request):
    return [block["toolResult"] for block in request["messages"][-1]["content"] if "toolResult" in block]


def assert_writes_confined(world: World):
    keys_written = world.written_keys()
    assert keys_written, "the scenario wrote nothing"
    for key in keys_written:
        assert key.startswith(ALLOWED_WRITE_PREFIXES), key
        assert not key.startswith(("papers/", "index/", "wiki/questions/")) or key.startswith("wiki/questions/"), key
    assert not any(key.startswith(("papers/", "index/")) for key in keys_written)


def successful_turns():
    return [
        turn(("search_wiki", {"query": "regional inheritance cohort"})),
        turn(("read_page", {"key": OVERVIEW})),
        turn(("edit_page", {"key": OVERVIEW, "old_text": "Earlier claim.", "new_text": NEW_CLAIM}),
             ("write_page", {"key": CONCEPT, "markdown": NEW_PAGE})),
        turn(answer=ANSWER),
    ]


# ---------------------------------------------------------------------------------------------
# Approval checks: no model call without a live, current, bound approval
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["missing", "revoked", "expired", "stale_policy", "other_execution", "other_parent"])
def test_invalid_approval_fails_the_job_without_a_model_call_or_reservation(variant):
    w = World()
    options = {
        "missing": {"approval": False},
        "revoked": {"status": "revoked"},
        "expired": {"expires_in": 0},
        "stale_policy": {"policy_revision": "1999-01-01-v0"},
        "other_execution": {"execution_id": "another-execution"},
        "other_parent": {"parent_job_id": "another-question"},
    }[variant]
    job = w.research_job(**options)
    reservations_before = len(w.table.rows("RESERVATION#"))

    result = w.run(job)

    assert result["status"] == "failed" and result["error_code"] == "approval_invalid"
    assert result["model_calls"] == 0 and w.bedrock.requests == []
    stored = w.job(job["job_id"])
    assert stored["status"] == "failed" and stored["error_code"] == "approval_invalid" and stored["usd_micros"] == 0
    assert "appr-1" in stored["reason"] or variant == "missing"
    assert len(w.table.rows("RESERVATION#")) == reservations_before      # no attempt reservation was taken
    assert w.reservation(job["reservation_id"])["status"] == "settled"    # the job hold went back to the period scopes
    assert lab_budget.job_balance(w.table, job["job_id"])["reserved_micros"] == 0
    assert not any(key.startswith("wiki/") for key in w.written_keys())
    assert not any(key.endswith("/research.json") for key in w.written_keys())


def test_expiry_is_judged_against_the_worker_clock():
    w = World()
    job = w.research_job()
    late = NOW + timedelta(seconds=APPROVAL_TTL_SECONDS + 1)

    result = w.run(job, now=late)

    assert result["status"] == "failed" and result["error_code"] == "approval_invalid"
    assert "expired" in result["reason"] and w.bedrock.requests == []


# ---------------------------------------------------------------------------------------------
# A scoped run
# ---------------------------------------------------------------------------------------------

def test_scoped_run_edits_the_target_creates_an_allowed_page_and_completes_with_one_settled_attempt():
    w = World()
    w.bedrock.responses = successful_turns()
    job = w.research_job()

    result = w.run(job)

    assert result["status"] == "completed", result
    assert result["research_status"] == "answer_ready" and result["answer"] == ANSWER
    assert result["published_keys"] == [OVERVIEW, CONCEPT] and result["index_pending"] is True
    assert result["out_of_scope"] == [] and result["recovered"] is False and result["model_calls"] == 4
    assert len(w.bedrock.requests) == 4
    assert all(request["modelId"] == MODEL for request in w.bedrock.requests)
    assert w.bedrock.requests[0]["additionalModelRequestFields"]["output_config"]["effort"] == "xhigh"

    # The wiki changed only where the approval allowed.
    assert NEW_CLAIM in w.s3.text(OVERVIEW) and "Unrelated conclusion stays exactly as written." in w.s3.text(OVERVIEW)
    assert w.s3.text(CONCEPT).startswith(NEW_PAGE)
    assert "[[concepts/regional-stability|" in w.s3.text(NOTE)         # reciprocal link into the cited note
    assert "[[concepts/regional-stability|" in w.s3.text("wiki/indexes/concepts.md")
    assert w.s3.text(OTHER) == OTHER_TEXT
    assert not any(key.startswith("wiki/questions/") for key in w.s3.objects)
    assert_writes_confined(w)

    # The receipt carries the engine result, the ledger and the scope.
    receipt = w.receipt(job["job_id"])
    assert receipt["status"] == "completed" and receipt["billing"] == "settled"
    assert receipt["research"]["status"] == "answer_ready" and receipt["research"]["answer"] == ANSWER
    assert receipt["research"]["question_key"] is None
    assert [op["key"] for op in receipt["publications"]] == [OVERVIEW, CONCEPT]
    for operation in receipt["publications"]:
        assert set(operation) >= {"operation_id", "key", "sha256", "etag", "replaced", "connections", "errors", "at"}
        assert operation["status"] == "published" and operation["execution_id"] == job["job_id"]
    assert receipt["publications"][0]["replaced"] is True and receipt["publications"][1]["replaced"] is False
    assert receipt["out_of_scope"] == [] and receipt["index_pending"] is True
    assert receipt["published_keys"] == [OVERVIEW, CONCEPT]
    assert receipt["scope"]["targets"] == [OVERVIEW] and receipt["scope"]["new_page_allowance"] == 2
    assert receipt["approval_id"] == "appr-1" and receipt["member_id"] == "m1" and receipt["model_id"] == MODEL
    assert receipt["model_calls"] == {"sent": 4, "succeeded": 4, "errors": []}
    assert receipt["research"]["trace_key"].startswith("runs/agents/") and receipt["research"]["trace_key"] in w.s3.objects

    # Money: one attempt reservation for the whole budget, settled from the returned usage.
    attempts = w.attempt_reservations()
    assert len(attempts) == 1
    attempt = attempts[0]
    expected = lab_budget.micros_for_usage(MODEL, receipt["usage"])
    assert receipt["usage"] == {"inputTokens": 4800, "outputTokens": 1200, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
    assert attempt["micros"] == BUDGET_MICROS and attempt["status"] == "settled" and attempt["settled_micros"] == expected
    assert attempt["attempt_id"] == "attempt-1-research"
    balance = lab_budget.job_balance(w.table, job["job_id"])
    assert balance["reserved_micros"] == 0 and balance["settled_micros"] == expected
    assert w.reservation(job["reservation_id"])["status"] == "settled"

    # The job and the approval record the execution.
    stored = w.job(job["job_id"])
    assert stored["status"] == "completed" and stored["usd_micros"] == expected and stored["usage"] == receipt["usage"]
    assert stored["receipt_key"] == f"runs/lab-questions/{job['job_id']}/research.json"
    assert stored["pages_written"] == [OVERVIEW, CONCEPT] and stored["index_pending"] is True
    assert stored["attempt_reservation_id"] == attempt["reservation_id"] and stored["research_status"] == "answer_ready"
    assert stored["triage_status"] == "skipped" and stored["lease_until"] is None
    approval = w.approval_record()
    assert approval["execution_status"] == "completed" and approval["published_keys"] == [OVERVIEW, CONCEPT]
    assert approval["index_pending"] is True and approval["execution_receipt_key"] == stored["receipt_key"]
    assert approval["status"] == "active"


def test_edit_outside_the_targets_is_a_tool_error_and_the_run_continues():
    w = World()
    w.bedrock.responses = [
        turn(("read_page", {"key": OTHER}), ("read_page", {"key": OVERVIEW})),
        turn(("edit_page", {"key": OTHER, "old_text": "An unrelated claim.", "new_text": "A changed claim."})),
        turn(("edit_page", {"key": OVERVIEW, "old_text": "Earlier claim.", "new_text": NEW_CLAIM})),
        turn(answer=ANSWER),
    ]
    job = w.research_job()

    result = w.run(job)

    refused = tool_results(w.bedrock.requests[2])
    assert [item["status"] for item in refused] == ["error"]
    assert refused[0]["content"][0]["json"]["error"] == OUT_OF_SCOPE
    assert tool_results(w.bedrock.requests[3])[0]["status"] == "success"
    assert result["status"] == "partial"                     # the engine counts the refused write among page_errors
    assert result["answer"] == ANSWER and result["published_keys"] == [OVERVIEW]
    assert [(item["key"], item["kind"]) for item in result["out_of_scope"]] == [(OTHER, "edit")]
    assert w.s3.text(OTHER) == OTHER_TEXT and NEW_CLAIM in w.s3.text(OVERVIEW)
    receipt = w.receipt(job["job_id"])
    assert receipt["status"] == "partial" and receipt["out_of_scope"][0]["reason"] == OUT_OF_SCOPE
    assert [error["key"] for error in receipt["research"]["page_errors"]] == [OTHER]
    assert w.job(job["job_id"])["status"] == "partial" and w.job(job["job_id"])["out_of_scope_count"] == 1
    assert w.approval_record()["execution_status"] == "partial"
    assert w.attempt_reservations()[0]["status"] == "settled"
    assert_writes_confined(w)


def test_new_synthesis_allows_two_new_pages_and_refuses_the_third():
    w = World()
    w.bedrock.responses = [
        turn(("write_page", {"key": CONCEPT, "markdown": NEW_PAGE})),
        turn(("write_page", {"key": SECOND, "markdown": "A second insight. [[sources/paper-one]]\n"})),
        turn(("write_page", {"key": THIRD, "markdown": "A third synthesis. [[sources/paper-one]]\n"})),
        turn(answer=ANSWER),
    ]
    job = w.research_job()

    result = w.run(job)

    assert CONCEPT in w.s3.objects and SECOND in w.s3.objects and THIRD not in w.s3.objects
    assert tool_results(w.bedrock.requests[3])[0]["content"][0]["json"]["error"] == OUT_OF_SCOPE
    assert result["published_keys"] == [CONCEPT, SECOND]
    assert [(item["key"], item["kind"]) for item in result["out_of_scope"]] == [(THIRD, "create")]
    assert_writes_confined(w)


def test_without_new_synthesis_only_explicitly_approved_new_pages_may_be_created():
    w = World()
    w.bedrock.responses = [
        turn(("write_page", {"key": CONCEPT, "markdown": NEW_PAGE})),
        turn(("write_page", {"key": SECOND, "markdown": "A second insight. [[sources/paper-one]]\n"})),
        turn(answer=ANSWER),
    ]
    professor_scope = {"question": QUESTION, "targets": [OVERVIEW], "new_pages": [CONCEPT], "note": None}
    job = w.research_job(professor_scope, kind="professor_approval")

    result = w.run(job)

    assert CONCEPT in w.s3.objects and SECOND not in w.s3.objects
    assert result["published_keys"] == [CONCEPT]
    assert [item["key"] for item in result["out_of_scope"]] == [SECOND]
    assert w.receipt(job["job_id"])["scope"]["new_page_allowance"] == 0


def test_a_page_created_by_this_execution_may_be_edited_again():
    w = World()
    w.bedrock.responses = [
        turn(("write_page", {"key": CONCEPT, "markdown": NEW_PAGE})),
        turn(("read_page", {"key": CONCEPT})),
        turn(("edit_page", {"key": CONCEPT, "old_text": "distinct unit", "new_text": "distinct measurement unit"})),
        turn(answer=ANSWER),
    ]
    job = w.research_job()

    result = w.run(job)

    assert result["status"] == "completed" and "distinct measurement unit" in w.s3.text(CONCEPT)
    receipt = w.receipt(job["job_id"])
    assert [(op["key"], op["kind"]) for op in receipt["publications"]] == [(CONCEPT, "create"), (CONCEPT, "edit")]
    assert receipt["publications"][1]["expected_etag"] == receipt["publications"][0]["etag"]


# ---------------------------------------------------------------------------------------------
# The question page
# ---------------------------------------------------------------------------------------------

def test_question_page_is_not_published_by_default():
    w = World()
    w.bedrock.responses = [turn(answer=ANSWER)]
    job = w.research_job()

    result = w.run(job)

    assert result["status"] == "completed" and result["answer"] == ANSWER
    assert not any(key.startswith("wiki/questions/") for key in w.s3.objects)
    assert not any(key.startswith("wiki/questions/") for key in w.s3.reads)
    receipt = w.receipt(job["job_id"])
    assert receipt["research"]["answer"] == ANSWER and receipt["research"]["question_key"] is None
    assert receipt["published_keys"] == [] and receipt["index_pending"] is False
    assert w.job(job["job_id"])["index_pending"] is False
    assert [key for key in w.written_keys() if key.startswith("wiki/")] == []


def test_question_page_is_published_only_when_the_approval_allows_it():
    w = World()
    w.bedrock.responses = [turn(answer=ANSWER)]
    job = w.research_job(scope(publish_question=True))

    result = w.run(job)

    _slug, question_key = question_key_for(QUESTION)
    assert result["status"] == "completed" and result["published_keys"] == [question_key]
    assert ANSWER in w.s3.text(question_key) and "author: \"lab:m1\"" in w.s3.text(question_key)
    receipt = w.receipt(job["job_id"])
    assert receipt["research"]["question_key"] == question_key and receipt["index_pending"] is True
    assert receipt["publications"][0]["key"] == question_key and receipt["publications"][0]["kind"] == "create"


# ---------------------------------------------------------------------------------------------
# Model failures and money
# ---------------------------------------------------------------------------------------------

def test_a_refusal_before_processing_fails_the_job_and_releases_the_attempt():
    w = World()
    w.bedrock.responses = [throttled(), throttled()]          # the research turn and the text-only final call
    job = w.research_job()

    result = w.run(job)

    assert result["status"] == "failed" and result["error_code"] == "ThrottlingException"
    assert len(w.bedrock.requests) == 2 and result["model_calls"] == 2
    attempt, = w.attempt_reservations()
    assert attempt["status"] == "released" and attempt["released_micros"] == BUDGET_MICROS
    balance = lab_budget.job_balance(w.table, job["job_id"])
    assert balance["reserved_micros"] == 0 and balance["settled_micros"] == 0
    stored = w.job(job["job_id"])
    assert stored["status"] == "failed" and stored["usd_micros"] == 0 and stored["error_code"] == "ThrottlingException"
    assert "Model call failed" in stored["reason"]
    assert w.reservation(job["reservation_id"])["status"] == "settled"
    receipt = w.receipt(job["job_id"])
    assert receipt["status"] == "failed" and receipt["billing"] == "released"
    assert receipt["model_calls"]["errors"] == [{"code": "ThrottlingException", "definite": True}] * 2
    assert w.approval_record()["execution_status"] == "failed"
    assert not any(key.startswith("wiki/") for key in w.written_keys())


def test_an_error_after_sending_keeps_the_money_held_as_outcome_unknown():
    w = World()
    w.bedrock.responses = [
        turn(("write_page", {"key": CONCEPT, "markdown": NEW_PAGE})),
        RuntimeError("read timed out after the request was sent"),
        RuntimeError("read timed out again"),
    ]
    job = w.research_job()

    result = w.run(job)

    assert result["status"] == "outcome_unknown" and "bill is unknown" in result["reason"]
    assert len(w.bedrock.requests) == 3
    attempt, = w.attempt_reservations()
    assert attempt["status"] == "unknown"
    assert lab_budget.job_balance(w.table, job["job_id"])["reserved_micros"] == BUDGET_MICROS
    assert w.reservation(job["reservation_id"])["status"] == "unknown"      # period scopes keep the hold
    stored = w.job(job["job_id"])
    assert stored["status"] == "outcome_unknown" and stored["pages_written"] == [CONCEPT]
    assert CONCEPT in w.s3.objects                                          # the saved page stays
    receipt = w.receipt(job["job_id"])
    assert receipt["status"] == "outcome_unknown" and receipt["billing"] == "unknown"
    assert receipt["usage"]["inputTokens"] == 1200                          # the one answered call is on record
    assert [error["definite"] for error in receipt["model_calls"]["errors"]] == [False, False]
    assert w.approval_record()["execution_status"] == "outcome_unknown"


def test_an_exception_escaping_the_engine_before_any_call_fails_and_releases():
    w = World()
    job = w.research_job(scope(publish_question=True))
    real_get = w.s3.get_object

    def denied(*, Bucket, Key, **options):                                   # the question-page version pre-read fails hard
        if Key.startswith("wiki/questions/"):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        return real_get(Bucket=Bucket, Key=Key, **options)

    w.s3.get_object = denied
    result = w.run(job)

    assert result["status"] == "failed" and result["error_code"] == "ClientError"
    assert "AccessDenied" in result["reason"] and w.bedrock.requests == [] and result["model_calls"] == 0
    attempt, = w.attempt_reservations()
    assert attempt["status"] == "released"
    stored = w.job(job["job_id"])
    assert stored["status"] == "failed" and stored["usd_micros"] == 0 and stored["error_code"] == "ClientError"
    assert w.receipt(job["job_id"])["billing"] == "released"
    assert w.approval_record()["execution_status"] == "failed"


def test_an_exception_escaping_the_engine_after_an_answered_call_settles_that_call_and_fails():
    w = World()
    usage = {"inputTokens": 700, "outputTokens": 90, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
    w.bedrock.responses = [{"stopReason": "end_turn", "usage": usage}]          # no "output": the engine raises KeyError
    job = w.research_job()

    result = w.run(job)

    assert result["status"] == "failed" and result["error_code"] == "KeyError"
    assert len(w.bedrock.requests) == 1
    attempt, = w.attempt_reservations()
    expected = lab_budget.micros_for_usage(MODEL, usage)
    assert attempt["status"] == "settled" and attempt["settled_micros"] == expected
    stored = w.job(job["job_id"])
    assert stored["status"] == "failed" and stored["usd_micros"] == expected and stored["usage"] == usage
    receipt = w.receipt(job["job_id"])
    assert receipt["billing"] == "settled" and receipt["usage"] == usage and receipt["research"] is None


def test_a_model_without_a_price_table_is_refused_before_any_reservation():
    w = World()
    job = w.research_job()

    result = run_research(job, table=w.table, receipts=w.receipts, s3=w.s3, bucket="bucket", index=w.index,
                          model_client=w.bedrock, converse=converse_once, model_id="unknown-model", reasoning=None,
                          now=NOW)

    assert result["status"] == "failed" and result["error_code"] == "model_unpriced"
    assert w.bedrock.requests == [] and w.attempt_reservations() == []
    assert w.job(job["job_id"])["status"] == "failed" and "unknown-model" in w.job(job["job_id"])["reason"]


def test_budget_that_cannot_be_reserved_fails_before_any_call():
    w = World()
    job = w.research_job()
    earlier = lab_budget.reserve_attempt(w.table, job["job_id"], "attempt-0-research", BUDGET_MICROS, now=NOW)
    lab_budget.settle(w.table, earlier["reservation_id"], BUDGET_MICROS, now=NOW)   # the cap is spent

    result = w.run(job)

    assert result["status"] == "failed" and result["error_code"] == "budget_exceeded"
    assert w.bedrock.requests == [] and len(w.attempt_reservations()) == 1
    assert w.job(job["job_id"])["status"] == "failed"


# ---------------------------------------------------------------------------------------------
# Recovery: a saved receipt closes a redelivered job without a model call
# ---------------------------------------------------------------------------------------------

def test_a_saved_receipt_closes_a_redelivered_job_without_calling_the_model():
    w = World()
    job = w.research_job()
    reservation = lab_budget.reserve_attempt(w.table, job["job_id"], "attempt-1-research", BUDGET_MICROS, now=NOW)
    usage = {"inputTokens": 3000, "outputTokens": 900, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
    micros = lab_budget.micros_for_usage(MODEL, usage)
    w.receipts.put_json(receipt_key(job["job_id"], "research.json"), {
        "status": "completed", "billing": "settled", "usage": usage, "usd_micros": micros,
        "attempt_reservation_id": reservation["reservation_id"], "published_keys": [OVERVIEW],
        "research": {"status": "answer_ready", "answer": ANSWER}, "out_of_scope": [], "publications": [],
        "research_status": "answer_ready",
    })

    result = w.run(job)

    assert result["status"] == "completed" and result["recovered"] is True and result["answer"] == ANSWER
    assert w.bedrock.requests == [] and result["model_calls"] == 0
    assert w.reservation(reservation["reservation_id"])["status"] == "settled"
    assert w.reservation(reservation["reservation_id"])["settled_micros"] == micros
    stored = w.job(job["job_id"])
    assert stored["status"] == "completed" and stored["usd_micros"] == micros and stored["pages_written"] == [OVERVIEW]
    assert w.approval_record()["execution_status"] == "completed"


# ---------------------------------------------------------------------------------------------
# Units: the publisher, the search adapter and the one-shot converse
# ---------------------------------------------------------------------------------------------

def test_scoped_publisher_refuses_everything_outside_wiki_and_outside_the_scope():
    publisher = ScopedPublisher(targets=[OVERVIEW], new_pages=[CONCEPT], new_page_allowance=0,
                                question_key="wiki/questions/q-1.md", execution_id="exec", now=NOW)
    for key in ("papers/paper-one/clean.md", "index/wiki-index-v2.sqlite3", "runs/agents/x.json", "wiki/../papers/x.md", None):
        with pytest.raises(ValueError, match="outside the approved research scope"):
            publisher.check(key, create_only=False)
        with pytest.raises(ValueError, match="outside the approved research scope"):
            publisher.check(key, create_only=True)
    assert publisher.check(OVERVIEW, create_only=False) == OVERVIEW
    assert publisher.check(CONCEPT, create_only=True) == CONCEPT
    assert publisher.check("wiki/questions/q-1.md", create_only=True) == "wiki/questions/q-1.md"
    assert publisher.check("wiki/questions/q-1.md", create_only=False) == "wiki/questions/q-1.md"
    with pytest.raises(ValueError):
        publisher.check(OTHER, create_only=False)
    with pytest.raises(ValueError):
        publisher.check(SECOND, create_only=True)            # no allowance and not approved
    with pytest.raises(ValueError):
        publisher.check(CONCEPT, create_only=False)           # approved for creation, not yet created

    s3 = wiki_with_index(PAGES, INDEX_KEY)
    with pytest.raises(ValueError):
        publisher(s3, "bucket", "papers/paper-one/clean.md", "text", create_only=True)
    assert s3.writes == [] and publisher.operations == []
    assert [item["key"] for item in publisher.refusals] == ["papers/paper-one/clean.md"]


def test_scoped_publisher_ledger_records_pending_then_published_or_failed():
    s3 = wiki_with_index(PAGES, INDEX_KEY)
    publisher = ScopedPublisher(targets=[OVERVIEW], new_page_allowance=2, execution_id="exec", now=NOW)
    saved = publisher(s3, "bucket", CONCEPT, NEW_PAGE, create_only=True)
    assert saved["key"] == CONCEPT and publisher.created == [CONCEPT] and publisher.expected[CONCEPT] == saved["etag"]
    with pytest.raises(Exception):
        publisher(s3, "bucket", OVERVIEW, "changed", expected_etag='"stale"')
    statuses = [(op["key"], op["status"]) for op in publisher.operations]
    assert statuses == [(CONCEPT, "published"), (OVERVIEW, "failed")]
    assert "error" in publisher.operations[1] and publisher.published_keys == [CONCEPT]
    assert publisher.operations[0]["sha256"] == saved["sha256"] and publisher.operations[0]["at"] == now_iso(NOW)


def test_search_adapter_returns_the_engine_shape_over_the_shared_index():
    s3 = wiki_with_index(PAGES, INDEX_KEY)
    index = (index_connection(s3.objects[INDEX_KEY]), s3.etag(INDEX_KEY))
    search = search_adapter(index)

    result = search({"query": "regional inheritance", "limit": 5, "doc_type": None})

    assert result["index_etag"] == s3.etag(INDEX_KEY) and result["results"]
    hit = result["results"][0]
    assert {"doc_type", "doc_id", "title", "section", "score", "path", "key"} <= set(hit)
    assert hit["doc_type"] == "note" and hit["doc_id"] == "paper-one" and hit["key"] == NOTE
    assert hit["path"] == "data/" + NOTE
    only_notes = search({"query": "regional inheritance", "limit": 5, "doc_type": "note"})
    assert {h["doc_type"] for h in only_notes["results"]} == {"note"}
    with pytest.raises(ValueError):
        search({"query": "   "})
    connection_only = search_adapter(index[0])({"query": "inheritance"})
    assert connection_only["index_etag"] is None and connection_only["results"]


def test_converse_once_calls_the_client_once_and_returns_the_engine_tuple():
    client = FakeBedrock([turn(answer="ok"), throttled()])
    request = {"modelId": MODEL, "messages": []}

    response, waited, attempts = converse_once(client, request)

    assert response["output"]["message"]["content"][0]["text"] == "ok" and (waited, attempts) == (0, 1)
    with pytest.raises(ClientError):
        converse_once(client, request)
    assert client.requests == [request, request]


def test_run_research_refuses_jobs_that_are_not_running_research_jobs():
    w = World()
    job = w.research_job()
    with pytest.raises(lab_jobs.InvalidTransition):
        w.run({**job, "status": "queued"})
    with pytest.raises(ValueError):
        w.run({**job, "kind": "answer"})
    with pytest.raises(ValueError):
        w.run({**w.parent})
    assert w.bedrock.requests == []


# ---------------------------------------------------------------------------------------------
# Clock: with a live clock the stamps follow the run instead of repeating the claim time
# ---------------------------------------------------------------------------------------------

def test_completion_and_publication_stamps_follow_the_worker_clock_when_one_is_given():
    w = World()
    w.bedrock.responses = successful_turns()
    job = w.research_job()
    ticks = itertools.count(1)
    clock = lambda: NOW + timedelta(seconds=next(ticks))    # noqa: E731 - one second per reading

    result = run_research(job, table=w.table, receipts=w.receipts, s3=w.s3, bucket="bucket", index=w.index,
                          model_client=w.bedrock, converse=converse_once, model_id=MODEL, reasoning="xhigh",
                          now=NOW, clock=clock)

    assert result["status"] == "completed", result
    receipt = w.receipt(job["job_id"])
    claimed = now_iso(NOW)
    assert receipt["completed_at"] > claimed
    stamps = [operation["at"] for operation in receipt["publications"]]
    assert len(stamps) == 2 and claimed < stamps[0] < stamps[1] <= receipt["completed_at"]
    # Each write reads the clock when it happens: receipt, then job close, then the approval update.
    stored = w.job(job["job_id"])
    assert stored["claimed_at"] == claimed and receipt["completed_at"] <= stored["completed_at"]
    assert stored["completed_at"] <= w.approval_record()["executed_at"]


def test_without_a_clock_every_stamp_is_the_fixed_now():
    w = World()
    w.bedrock.responses = successful_turns()
    job = w.research_job()

    result = w.run(job)

    assert result["status"] == "completed"
    receipt = w.receipt(job["job_id"])
    assert receipt["completed_at"] == now_iso(NOW)
    assert {operation["at"] for operation in receipt["publications"]} == {now_iso(NOW)}
