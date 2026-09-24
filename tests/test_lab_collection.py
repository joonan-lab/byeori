"""byeori.lab_collection: the gap a failed answer leaves, who is told, and who decides."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from byeori import lab_collection as collection
from byeori.lab_store import keys
from lab_fakes import MemoryTable

NOW = datetime(2026, 9, 22, 9, 0, tzinfo=UTC)
CITED = [{"key": "wiki/sources/paper-one.md", "section": "Results"}]
ANSWER = {
    "job_id": "j1", "member_id": "m1", "session_id": "s1",
    "question": "한국인 코호트의 희귀 구조변이 부담은 얼마인가요?",
    "standalone_question": "한국인 코호트의 희귀 구조변이 부담은 얼마인가요?",
    "english_query": "rare structural variant burden Korean autism cohort",
    "status": "partial", "evidence_state": "insufficient", "citations": [],
    "policy_revision": "2026-09-21-v1",
}


def table_with_gap(answer=None) -> MemoryTable:
    table = MemoryTable()
    collection.record_gap(table, answer or ANSWER, now=NOW)
    return table


# ---------------------------------------------------------------------------------------------
# Which answers count as the wiki falling short
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("answer, expected", [
    ({"status": "partial", "evidence_state": "insufficient", "citations": []}, collection.NO_EVIDENCE),
    ({"status": "completed", "evidence_state": "insufficient", "citations": CITED}, collection.NO_EVIDENCE),
    ({"status": "completed", "evidence_state": "partial", "citations": []}, collection.THIN_EVIDENCE),
    ({"status": "partial", "evidence_state": "sufficient", "citations": []}, collection.THIN_EVIDENCE),
    # An answer standing on the wiki is not a gap, whatever else it says.
    ({"status": "completed", "evidence_state": "sufficient", "citations": CITED}, None),
    ({"status": "partial", "evidence_state": "partial", "citations": CITED}, None),
    # A job that failed or whose outcome is unknown has no answer to judge.
    ({"status": "failed", "evidence_state": None, "citations": []}, None),
    ({"status": "outcome_unknown", "evidence_state": None, "citations": []}, None),
    (None, None),
])
def test_only_an_answer_the_wiki_could_not_stand_behind_is_a_gap(answer, expected):
    assert collection.gap_reason(answer) == expected


def test_a_sufficient_answer_records_nothing():
    table = MemoryTable()
    assert collection.record_gap(table, {**ANSWER, "evidence_state": "sufficient",
                                         "citations": CITED}, now=NOW) is None
    assert table.rows("COLLECT#") == []


# ---------------------------------------------------------------------------------------------
# The query the search will use
# ---------------------------------------------------------------------------------------------

def test_the_recorded_query_is_the_english_one_the_worker_already_wrote():
    """A Korean query returns Korean papers; the lab's 65 journals are English (measured 2026-09-22)."""
    table = table_with_gap()
    gap = collection.gap_for_job(table, "j1")
    assert gap["query"] == "rare structural variant burden Korean autism cohort"
    assert gap["question"] == ANSWER["question"], "the member's own words are kept as the question"


def test_without_an_english_query_the_standalone_question_is_used():
    table = table_with_gap({**ANSWER, "english_query": None})
    assert collection.gap_for_job(table, "j1")["query"] == ANSWER["standalone_question"]


def test_a_query_longer_than_the_bound_is_cut():
    table = table_with_gap({**ANSWER, "english_query": "x" * 500})
    assert len(collection.gap_for_job(table, "j1")["query"]) == collection.MAX_QUERY_CHARS


# ---------------------------------------------------------------------------------------------
# Both sides are told (user, 2026-09-22)
# ---------------------------------------------------------------------------------------------

def test_the_gap_is_recorded_before_anybody_asks_for_anything():
    """The professor sees a subject the wiki keeps failing on without a member noticing."""
    table = table_with_gap()
    gap = collection.gap_for_job(table, "j1")
    assert gap["status"] == collection.UNREQUESTED
    assert [g["job_id"] for g in collection.list_gaps(table)["gaps"]] == ["j1"]


def test_the_member_is_offered_it_and_accepting_records_a_request_and_nothing_else():
    table = table_with_gap()
    offer = collection.offer_view(collection.gap_for_job(table, "j1"))

    assert offer["decided"] is False and sorted(offer["decisions"]) == ["accept", "decline"]
    assert "승인 뒤에 실행됩니다" in offer["note"], "the member must be told nothing starts yet"

    record = collection.respond(table, "j1", "m1", "accept", now=NOW)

    assert record["status"] == collection.REQUESTED and record["decided_by"] == "m1"
    # Nothing but this row moved: no job, no reservation, no outbox, no candidate.
    assert table.rows("JOB#") == [] and table.rows("RESERVATION#") == []
    assert table.rows("OUTBOX") == [] and table.rows("CANDIDATE#") == []


def test_a_declined_offer_stays_in_the_professors_queue():
    """Declining is the member's answer, not a reason to forget the subject."""
    table = table_with_gap()

    collection.respond(table, "j1", "m1", "decline", now=NOW)

    assert collection.gap_for_job(table, "j1")["status"] == collection.DECLINED
    assert [g["job_id"] for g in collection.list_gaps(table)["gaps"]] == ["j1"]
    assert collection.approve(table, "j1", "admin.a", now=NOW)["status"] == collection.APPROVED


def test_another_members_gap_is_not_found_rather_than_forbidden():
    table = table_with_gap()
    with pytest.raises(collection.NotFound):
        collection.respond(table, "j1", "m2", "accept", now=NOW)
    assert collection.gap_for_job(table, "j1")["status"] == collection.UNREQUESTED


# ---------------------------------------------------------------------------------------------
# The professor decides, and the row carries the whole life of the gap
# ---------------------------------------------------------------------------------------------

def test_the_professor_may_approve_a_question_no_member_asked_about():
    table = table_with_gap()
    record = collection.approve(table, "j1", "admin.a", query="CHD8 structural variants", now=NOW)
    assert record["status"] == collection.APPROVED
    assert record["query"] == "CHD8 structural variants", "the professor may rewrite the query"
    assert record["approved_by"] == "admin.a"


def test_a_rejected_question_can_be_approved_later_but_a_collected_one_is_done():
    table = table_with_gap()
    collection.reject(table, "j1", "admin.a", reason="이미 다른 질문으로 다룸", now=NOW)
    assert collection.gap_for_job(table, "j1")["status"] == collection.REJECTED

    assert collection.approve(table, "j1", "admin.a", now=NOW)["status"] == collection.APPROVED
    assert collection.mark_collected(table, "j1", candidates=7, now=NOW)["candidates_saved"] == 7
    with pytest.raises(collection.InvalidTransition):
        collection.respond(table, "j1", "m1", "accept", now=NOW)


def test_collecting_requires_an_approval_first():
    table = table_with_gap()
    with pytest.raises(collection.InvalidTransition):
        collection.mark_collected(table, "j1", candidates=3, now=NOW)


def test_a_decision_on_a_question_with_no_gap_is_not_found():
    table = MemoryTable()
    for call in (lambda: collection.respond(table, "nope", "m1", "accept", now=NOW),
                 lambda: collection.approve(table, "nope", "admin.a", now=NOW),
                 lambda: collection.reject(table, "nope", "admin.a", now=NOW)):
        with pytest.raises(collection.NotFound):
            call()


# ---------------------------------------------------------------------------------------------
# Re-running a job never resets a decision
# ---------------------------------------------------------------------------------------------

def test_recording_the_same_gap_again_keeps_the_decision_already_made():
    """A redelivered job answers the same question; it must not un-approve it."""
    table = table_with_gap()
    collection.respond(table, "j1", "m1", "accept", now=NOW)
    collection.approve(table, "j1", "admin.a", now=NOW)

    again = collection.record_gap(table, ANSWER, now=NOW)

    assert again["status"] == collection.APPROVED
    # One record and its one pointer; the second write created neither.
    assert len([row for row in table.rows("COLLECT#") if row["sk"] == "META"]) == 1
    assert len([row for row in table.rows("COLLECT#") if row["pk"] == "COLLECT#ALL"]) == 1


def test_the_pointer_follows_the_status_so_the_queue_can_be_filtered():
    table = table_with_gap()
    collection.record_gap(table, {**ANSWER, "job_id": "j2"}, now=NOW)
    collection.approve(table, "j1", "admin.a", now=NOW)

    approved = collection.list_gaps(table, status=collection.APPROVED)
    waiting = collection.list_gaps(table, status=collection.UNREQUESTED)

    assert [g["job_id"] for g in approved["gaps"]] == ["j1"]
    assert [g["job_id"] for g in waiting["gaps"]] == ["j2"]
    pointer = table.get(*keys.collection_gap_pointer(table.get(*keys.collection_gap("j1"))["created_at"], "j1"))
    assert pointer["status"] == collection.APPROVED


def test_an_unknown_status_filter_is_refused():
    with pytest.raises(ValueError):
        collection.list_gaps(MemoryTable(), status="whatever")
    with pytest.raises(ValueError):
        collection.respond(table_with_gap(), "j1", "m1", "maybe", now=NOW)
