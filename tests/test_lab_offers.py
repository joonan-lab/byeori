"""Offer issuance, owner-only responses and the one-transaction consent record of byeori.lab_offers."""
from __future__ import annotations

import ast
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from byeori import lab_budget, lab_jobs, lab_offers
from byeori.lab_jobs import (
    Forbidden,
    IdempotencyConflict,
    Member,
    NotFound,
    claim,
    complete,
    intake,
    pending_outbox,
    queue,
)
from byeori.lab_offers import (
    LINKED_NOTE,
    STALE_MESSAGE,
    Expired,
    RevisionConflict,
    is_expired,
    issue,
    offer_for_job,
    offer_view,
    respond,
    suppressing_offer,
)
from byeori.lab_policy import (
    ANSWER_JOB_CAP_MICROS,
    APPROVAL_TTL_SECONDS,
    JEV_MODEL,
    LEASE_SECONDS,
    OFFER_TEMPLATES,
    OFFER_TTL_SECONDS,
    POLICY_REVISION,
    RESEARCH_PROFILE,
    REVIEW_CANDIDATE_CUTOFF,
)
from byeori.lab_store import ConditionFailed, Put, ReceiptWriter, StoreError, Update, digest, keys, new_item, now_iso
from lab_fakes import MemoryS3, MemoryTable, member

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
LATER = NOW + timedelta(hours=1)
PERIOD = "2026-09"
QUESTION = "CHD8 결손은 대두증과 연관되지 않는가? 아시아 코호트(n = 120)에서 확인된 결과만 답해 주세요."
TARGET = "wiki/overviews/asd-ndd/chd8.md"
SECOND_TARGET = "wiki/concepts/macrocephaly.md"
HEX = re.compile(r"^[0-9a-f]{32}$")
SCOPE_CHECK = {"index_etag": '"abc"', "queries": ["CHD8 macrocephaly"], "pages_checked": [TARGET], "best_score": 24.5}
INPUT_HASH = "a" * 64
OTHER_INPUT_HASH = "b" * 64
DRIFTED_PROFILE = {"budget_usd_micros": 50_000_000, "reread": "always", "max_calls": 400}
DRIFTED_POLICY = "2026-10-01-v2"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """lab_jobs retries pause between attempts; record the pauses instead of sleeping."""
    pauses: list[float] = []
    monkeypatch.setattr(lab_jobs, "_sleep", pauses.append)
    return pauses


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

    def rows(self, prefix):
        return self.table.rows(prefix)

    def research_jobs(self):
        return [row for row in self.rows("JOB#") if row["sk"] == "META" and row["kind"] == "research"]

    def reserved(self, scope="lab:2026-09"):
        record = self.table.get(*keys.budget(scope))
        return None if record is None else record["reserved_micros"]

    def completed_job(self, request_id="req-1", who=None, session_id=None):
        who = who or self.student
        body = {"action": "ask_byeori", "request_id": request_id, "question": QUESTION}
        if session_id is not None:
            body["session_id"] = session_id
        job = intake(self.table, self.receipts, who, body, NOW)
        job = queue(self.table, job["job_id"], period=PERIOD, cap=ANSWER_JOB_CAP_MICROS, now=NOW)
        job = claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, NOW)
        return complete(self.table, job["job_id"], job["revision"],
                        receipt_key=f"runs/lab-questions/{job['job_id']}/answer.json",
                        evidence_key=f"runs/lab-questions/{job['job_id']}/evidence.json", usage=None,
                        usd_micros=0, status="completed", now=NOW)

    def verdict(self, job, review_candidate=0.995, input_hash=INPUT_HASH):
        probabilities = {"answer_only": round(1 - review_candidate - 0.002, 4), "needs_lookup": 0.002,
                         "review_candidate": review_candidate}
        record = new_item(*keys.verdict(job["job_id"]), now_iso(NOW), status="complete", choice="review_candidate",
                          probabilities=probabilities, confidence=0.91, cutoff=REVIEW_CANDIDATE_CUTOFF,
                          passed_cutoff=review_candidate >= REVIEW_CANDIDATE_CUTOFF, input_hash=input_hash,
                          model=JEV_MODEL, policy_revision=POLICY_REVISION, usage={"input_tokens": 3000, "output_tokens": 40},
                          usd_micros=126, error_code=None, reason=None, reused_from_job_id=None,
                          candidate_status="review_candidate")
        self.table.put(record)
        return record

    def offered(self, kind="supplement_existing", targets=(TARGET,), now=NOW, input_hash=INPUT_HASH, **overrides):
        job = overrides.pop("job", None) or self.completed_job(**overrides)
        verdict = self.verdict(job, input_hash=input_hash)
        offer = issue(self.table, self.receipts, job, verdict, kind, list(targets), SCOPE_CHECK, now)
        return job, offer

    def professor_execution(self, job, approval_id="prof-approval-1", now=NOW):
        """A research execution the professor's approval path started for ``job``."""
        parent = self.table.get(*keys.job(job["job_id"]))
        scope = {"question": QUESTION, "targets": [TARGET], "new_pages": [], "note": "professor", "kind": "supplement_existing"}
        return lab_jobs.create_research_job(self.table, self.receipts, parent_job=parent, member_id=parent["member_id"],
                                            approval_id=approval_id, scope=scope, budget_usd_micros=5_000_000, now=now)


def response(offer, decision="accept", request_id="resp-1", **overrides):
    return {"request_id": request_id, "offer_id": offer["offer_id"], "revision": offer["revision"], "hash": offer["hash"],
            "decision": decision, **overrides}


def op_keys(operations):
    return [(type(op).__name__, (op.item["pk"], op.item["sk"]) if isinstance(op, Put) else (op.pk, op.sk))
            for op in operations]


def expected_hash(offer):
    return digest({"offer_id": offer["offer_id"], "job_id": offer["job_id"], "kind": offer["kind"],
                   "targets": offer["targets"], "message": offer["message"], "policy_revision": offer["policy_revision"],
                   "expires_at": offer["expires_at"]})


def session_pointer(table, offer):
    return table.get(f"SESSION#{offer['session_id']}", f"OFFER#{offer['created_at']}#{offer['offer_id']}")


def session_pointer_key(offer):
    return (f"SESSION#{offer['session_id']}", f"OFFER#{offer['created_at']}#{offer['offer_id']}")


# ---------------------------------------------------------------------------------------------
# Issuing
# ---------------------------------------------------------------------------------------------

def test_issue_writes_offer_pointers_and_job_link_in_one_transaction_with_hash_expiry_and_receipt():
    w = World()
    job = w.completed_job()
    verdict = w.verdict(job)
    transactions, writes = len(w.transactions()), len(w.s3.writes)
    offer = issue(w.table, w.receipts, job, verdict, "supplement_existing", [TARGET, SECOND_TARGET], SCOPE_CHECK, NOW)

    assert HEX.match(offer["offer_id"]) and offer["revision"] == 1 and offer["status"] == "offered"
    assert offer["job_id"] == job["job_id"] and offer["member_id"] == "m1" and offer["session_id"] == job["session_id"]
    assert offer["kind"] == "supplement_existing" and offer["targets"] == [TARGET, SECOND_TARGET]
    assert offer["scope_check"] == SCOPE_CHECK and offer["policy_revision"] == POLICY_REVISION
    assert offer["expires_at"] == now_iso(NOW + timedelta(seconds=OFFER_TTL_SECONDS)) == "2026-09-28T09:00:00.000000+00:00"
    assert offer["hash"] == expected_hash(offer) and len(offer["hash"]) == 64
    assert offer["decision_at"] is None and offer["execution_id"] is None and offer["approval_id"] is None
    assert offer["research_status"] is None
    assert offer["created_at"] == now_iso(NOW) and offer["offered_at"] == offer["created_at"]
    assert offer["verdict"]["probabilities"]["review_candidate"] == 0.995 and offer["verdict"]["input_hash"] == INPUT_HASH
    assert offer["research_profile"] == RESEARCH_PROFILE
    assert w.table.get(*keys.offer(offer["offer_id"])) == offer

    assert len(w.transactions()) == transactions + 1
    assert op_keys(w.transactions()[-1]) == [
        ("Put", (f"OFFER#{offer['offer_id']}", "META")),
        ("Put", (f"JOB#{job['job_id']}", f"OFFER#{offer['offer_id']}")),
        ("Put", session_pointer_key(offer)),
        ("Update", (f"JOB#{job['job_id']}", "META")),
    ]
    pointer = w.table.get(*keys.job_offer(job["job_id"], offer["offer_id"]))
    assert pointer["offer_id"] == offer["offer_id"] and pointer["status"] == "offered" and pointer["kind"] == "supplement_existing"
    session = session_pointer(w.table, offer)
    assert session["offer_id"] == offer["offer_id"] and session["job_id"] == job["job_id"] and session["status"] == "offered"
    assert session["kind"] == "supplement_existing" and session["targets"] == [TARGET, SECOND_TARGET]
    assert session["input_hash"] == INPUT_HASH and session["member_id"] == "m1" and session["revision"] == 1
    linked = w.table.get(*keys.job(job["job_id"]))
    assert linked["offer_id"] == offer["offer_id"] and linked["revision"] == job["revision"] + 1
    assert linked["triage_status"] == job["triage_status"] and linked["status"] == "completed"
    assert w.table.get(*keys.session(job["session_id"]))["revision"] == 1  # the session META row is untouched

    key = f"runs/lab-questions/{job['job_id']}/offer-{offer['offer_id']}.json"
    assert offer["receipt_key"] == key
    assert w.s3.writes[writes:] == [(key, {"IfNoneMatch": "*"})]
    receipt = w.s3.json(key)
    assert receipt["hash"] == offer["hash"] and receipt["message"] == offer["message"] and receipt["targets"] == offer["targets"]
    assert receipt["scope_check"] == SCOPE_CHECK and receipt["verdict"] == offer["verdict"]
    assert receipt["offered_at"] == offer["offered_at"] and receipt["expires_at"] == offer["expires_at"]
    assert receipt["research_profile"] == RESEARCH_PROFILE and receipt["policy_revision"] == POLICY_REVISION
    assert "pk" not in receipt and "sk" not in receipt and "receipt_sha256" not in receipt and "revision" not in receipt
    # the receipt is the proposal snapshot, so a later decision never changes what its sha256 names
    assert not {"status", "decision", "decision_at", "execution_id", "approval_id", "research_status", "stale_at"} & set(receipt)
    assert digest(receipt) == offer["receipt_sha256"]


def test_supplement_message_is_the_verbatim_sentence_then_the_target_links_then_the_consent_note():
    w = World()
    _job, offer = w.offered("supplement_existing", [TARGET, SECOND_TARGET])
    message = offer["message"]
    sentence, note, label = OFFER_TEMPLATES["supplement_existing"], OFFER_TEMPLATES["consent_note"], OFFER_TEMPLATES["existing_targets_label"]
    assert message.startswith(sentence) and message.endswith(note)
    assert "관련 합성 위키가 있지만 이 부분을 보완할 수 있습니다. 보완할까요?" in message
    assert "동의하면 서버의 연구 프로필로 비동기 연구 실행이 시작됩니다." in message
    assert message.index(sentence) < message.index(label) < message.index(TARGET) < message.index(SECOND_TARGET) < message.index(note)
    assert OFFER_TEMPLATES["new_synthesis"] not in message and OFFER_TEMPLATES["new_targets_label"] not in message
    assert message == f"{sentence}\n{label} {TARGET}, {SECOND_TARGET}\n{note}"


def test_new_synthesis_message_uses_its_verbatim_sentence_and_names_new_pages_only_when_given():
    w = World()
    _job, bare = w.offered("new_synthesis", [])
    sentence, note = OFFER_TEMPLATES["new_synthesis"], OFFER_TEMPLATES["consent_note"]
    assert bare["message"] == f"{sentence}\n{note}" and bare["targets"] == []
    assert "현재 검색한 위키에서는 이 질문을 종합한 문서를 찾지 못했습니다. 관련 논문을 바탕으로 합성 위키를 만들까요?" in bare["message"]
    assert OFFER_TEMPLATES["supplement_existing"] not in bare["message"]

    _job, named = w.offered("new_synthesis", ["wiki/concepts/chd8-macrocephaly.md"], request_id="req-2", input_hash=OTHER_INPUT_HASH)
    label = OFFER_TEMPLATES["new_targets_label"]
    assert named["message"] == f"{sentence}\n{label} wiki/concepts/chd8-macrocephaly.md\n{note}"
    assert OFFER_TEMPLATES["existing_targets_label"] not in named["message"]
    assert named["hash"] != bare["hash"]


def test_issue_returns_the_existing_offer_for_a_job_and_rewrites_only_a_missing_receipt():
    w = World()
    job, offer = w.offered()
    transactions, writes = len(w.transactions()), len(w.s3.writes)
    verdict = w.table.get(*keys.verdict(job["job_id"]))
    again = issue(w.table, w.receipts, w.table.get(*keys.job(job["job_id"])), verdict, "new_synthesis", [], {}, LATER)
    assert again == offer and len(w.transactions()) == transactions and len(w.s3.writes) == writes
    assert offer_for_job(w.table, job["job_id"]) == offer

    del w.s3.objects[offer["receipt_key"]]  # a crash between the transaction and the receipt put
    replayed = issue(w.table, w.receipts, w.table.get(*keys.job(job["job_id"])), verdict, "supplement_existing", [TARGET], SCOPE_CHECK, LATER)
    assert replayed == offer and len(w.transactions()) == transactions
    assert w.s3.writes[writes:] == [(offer["receipt_key"], {"IfNoneMatch": "*"})]
    assert digest(w.s3.json(offer["receipt_key"])) == offer["receipt_sha256"]


def test_issue_validates_kind_targets_verdict_and_job():
    w = World()
    job = w.completed_job()
    verdict = w.verdict(job)
    before = w.rows("")
    with pytest.raises(ValueError):
        issue(w.table, w.receipts, job, verdict, "correction", [TARGET], SCOPE_CHECK, NOW)
    with pytest.raises(ValueError):  # a supplement names what it supplements
        issue(w.table, w.receipts, job, verdict, "supplement_existing", [], SCOPE_CHECK, NOW)
    for bad in (["wiki/drafts/x.md"], ["papers/x.pdf"], [""], ["wiki/../x.md"], "wiki/overviews/x.md", [TARGET, 3]):
        with pytest.raises(ValueError):
            issue(w.table, w.receipts, job, verdict, "supplement_existing", bad, SCOPE_CHECK, NOW)
    with pytest.raises(ValueError):
        issue(w.table, w.receipts, job, verdict, "supplement_existing", [TARGET], ["not", "a", "dict"], NOW)
    below = {**verdict, "probabilities": {"answer_only": 0.0101, "needs_lookup": 0.0, "review_candidate": 0.9899}}
    with pytest.raises(ValueError):
        issue(w.table, w.receipts, job, below, "supplement_existing", [TARGET], SCOPE_CHECK, NOW)
    with pytest.raises(ValueError):
        issue(w.table, w.receipts, job, None, "supplement_existing", [TARGET], SCOPE_CHECK, NOW)
    with pytest.raises(ValueError):
        issue(w.table, w.receipts, {**job, "kind": "research"}, verdict, "supplement_existing", [TARGET], SCOPE_CHECK, NOW)
    with pytest.raises(ValueError):
        issue(w.table, w.receipts, job, verdict, "supplement_existing", [TARGET], SCOPE_CHECK, datetime(2026, 9, 21, 9, 0))
    with pytest.raises(NotFound):
        issue(w.table, w.receipts, {**job, "job_id": "0" * 32}, verdict, "supplement_existing", [TARGET], SCOPE_CHECK, NOW)
    assert w.rows("") == before and len(w.s3.writes) == 1  # only intake's request.json; nothing from the refusals


def test_issue_deduplicates_targets_and_rereads_the_job_revision_under_contention():
    w = World()
    job = w.completed_job()
    verdict = w.verdict(job)
    stale = {**job, "revision": job["revision"] - 1}  # the caller holds an older copy of the job
    offer = issue(w.table, w.receipts, stale, verdict, "supplement_existing", [TARGET, TARGET, SECOND_TARGET], SCOPE_CHECK, NOW)
    assert offer["targets"] == [TARGET, SECOND_TARGET]
    assert w.table.get(*keys.job(job["job_id"]))["offer_id"] == offer["offer_id"]


def test_offer_for_job_and_offer_view():
    w = World()
    assert offer_for_job(w.table, "0" * 32) is None and offer_for_job(w.table, "") is None
    job, offer = w.offered()
    assert offer_for_job(w.table, job["job_id"]) == offer
    view = offer_view(offer)
    assert view == {"offer_id": offer["offer_id"], "revision": 1, "hash": offer["hash"], "kind": "supplement_existing",
                    "message": offer["message"], "targets": [TARGET], "expires_at": offer["expires_at"], "status": "offered"}
    assert "member_id" not in view and "scope_check" not in view and "verdict" not in view
    assert "research_profile" not in view and "research_status" not in view
    assert is_expired(offer, NOW) is False and is_expired(offer, NOW + timedelta(days=7)) is True
    assert is_expired(offer, NOW + timedelta(days=7) - timedelta(microseconds=1)) is False


# ---------------------------------------------------------------------------------------------
# Session re-offer suppression
# ---------------------------------------------------------------------------------------------

def test_the_same_proposal_is_not_offered_again_in_a_session_while_the_first_offer_is_open_or_declined():
    w = World()
    first_job, first = w.offered("supplement_existing", [TARGET, SECOND_TARGET])
    session_id = first_job["session_id"]
    second_job = w.completed_job(request_id="req-2", session_id=session_id)
    assert second_job["session_id"] == session_id and second_job["turn"] == 2
    verdict = w.verdict(second_job, input_hash=OTHER_INPUT_HASH)  # another verdict input; the proposal is what repeats
    rows, transactions, writes = w.rows(""), len(w.transactions()), len(w.s3.writes)

    # the same kind and targets (in another order) are suppressed while the first offer is open
    assert issue(w.table, w.receipts, second_job, verdict, "supplement_existing", [SECOND_TARGET, TARGET], SCOPE_CHECK, LATER) is None
    assert w.rows("") == rows and len(w.transactions()) == transactions and len(w.s3.writes) == writes
    assert offer_for_job(w.table, second_job["job_id"]) is None and w.table.get(*keys.job(second_job["job_id"]))["offer_id"] is None
    suppressor = suppressing_offer(w.table, session_id, "supplement_existing", [SECOND_TARGET, TARGET], OTHER_INPUT_HASH)
    assert suppressor == first and suppressor["status"] == "offered"

    # ... and still after the student declined it
    respond(w.table, w.receipts, w.student, response(first, decision="decline", request_id="dec-1"), LATER)
    assert session_pointer(w.table, first)["status"] == "declined"
    rows, transactions = w.rows(""), len(w.transactions())
    assert issue(w.table, w.receipts, second_job, verdict, "supplement_existing", [TARGET, SECOND_TARGET], SCOPE_CHECK, LATER) is None
    assert w.rows("") == rows and len(w.transactions()) == transactions
    assert suppressing_offer(w.table, session_id, "supplement_existing", [TARGET, SECOND_TARGET], None)["status"] == "declined"

    # a different proposal from a different verdict input is a new offer
    other = issue(w.table, w.receipts, second_job, verdict, "supplement_existing", [TARGET], SCOPE_CHECK, LATER)
    assert other is not None and other["offer_id"] != first["offer_id"] and other["job_id"] == second_job["job_id"]
    assert session_pointer(w.table, other)["input_hash"] == OTHER_INPUT_HASH
    assert suppressing_offer(w.table, session_id, "supplement_existing", [TARGET], None) == other
    # another session of the same student is asked independently
    third_job = w.completed_job(request_id="req-3")
    assert third_job["session_id"] != session_id
    fresh = issue(w.table, w.receipts, third_job, w.verdict(third_job, input_hash=OTHER_INPUT_HASH), "supplement_existing",
                  [TARGET, SECOND_TARGET], SCOPE_CHECK, LATER)
    assert fresh is not None and fresh["offer_id"] not in {first["offer_id"], other["offer_id"]}


def test_the_same_verdict_input_is_not_offered_again_in_a_session_even_with_other_targets():
    w = World()
    first_job, first = w.offered("supplement_existing", [TARGET])
    second_job = w.completed_job(request_id="req-2", session_id=first_job["session_id"])
    same_input = w.verdict(second_job, input_hash=INPUT_HASH)
    transactions = len(w.transactions())

    assert issue(w.table, w.receipts, second_job, same_input, "new_synthesis", ["wiki/concepts/chd8-macrocephaly.md"], SCOPE_CHECK, LATER) is None
    assert issue(w.table, w.receipts, second_job, same_input, "new_synthesis", [], SCOPE_CHECK, LATER) is None
    assert len(w.transactions()) == transactions
    assert suppressing_offer(w.table, first_job["session_id"], "new_synthesis", [], INPUT_HASH) == first
    assert suppressing_offer(w.table, first_job["session_id"], "new_synthesis", [], OTHER_INPUT_HASH) is None
    assert suppressing_offer(w.table, first_job["session_id"], "new_synthesis", [], None) is None
    # a verdict without an input hash falls back to the proposal comparison only
    no_hash = {**same_input, "input_hash": None}
    made = issue(w.table, w.receipts, second_job, no_hash, "new_synthesis", [], SCOPE_CHECK, LATER)
    assert made is not None and session_pointer(w.table, made)["input_hash"] is None


def test_accepted_offers_suppress_while_expired_and_stale_offers_do_not(monkeypatch):
    w = World()
    # accepted: the research it started answers the same proposal; the same session is not offered it again
    accepted_job, accepted = w.offered("supplement_existing", [TARGET])
    respond(w.table, w.receipts, w.student, response(accepted, request_id="acc-1"), LATER)
    assert session_pointer(w.table, accepted)["status"] == "accepted"
    second = w.completed_job(request_id="req-2", session_id=accepted_job["session_id"])
    assert suppressing_offer(w.table, accepted_job["session_id"], "supplement_existing", [TARGET], INPUT_HASH)["offer_id"] == accepted["offer_id"]
    assert issue(w.table, w.receipts, second, w.verdict(second), "supplement_existing", [TARGET], SCOPE_CHECK, LATER) is None
    assert len(w.research_jobs()) == 1

    # expired but not yet recorded: with `now` the open offer past its TTL does not suppress; recorded expired never does
    expired_job, expired = w.offered("new_synthesis", [], request_id="req-3", who=w.other)
    at_expiry = NOW + timedelta(seconds=OFFER_TTL_SECONDS)
    assert suppressing_offer(w.table, expired_job["session_id"], "new_synthesis", [], INPUT_HASH) == expired
    assert suppressing_offer(w.table, expired_job["session_id"], "new_synthesis", [], INPUT_HASH, now=at_expiry) is None
    assert suppressing_offer(w.table, expired_job["session_id"], "new_synthesis", [], INPUT_HASH, now=at_expiry - timedelta(seconds=1)) == expired
    with pytest.raises(Expired):
        respond(w.table, w.receipts, w.other, response(expired, request_id="exp-1"), at_expiry)
    assert session_pointer(w.table, expired)["status"] == "expired"
    assert suppressing_offer(w.table, expired_job["session_id"], "new_synthesis", [], INPUT_HASH) is None
    fourth = w.completed_job(request_id="req-4", who=w.other, session_id=expired_job["session_id"])
    assert issue(w.table, w.receipts, fourth, w.verdict(fourth), "new_synthesis", [], SCOPE_CHECK, at_expiry) is not None

    # stale: an offer refused for its policy revision is re-offered under the current policy
    stale_job, stale = w.offered("supplement_existing", [SECOND_TARGET], request_id="req-5", input_hash=OTHER_INPUT_HASH)
    monkeypatch.setattr(lab_offers, "POLICY_REVISION", DRIFTED_POLICY)
    with pytest.raises(RevisionConflict):
        respond(w.table, w.receipts, w.student, response(stale, request_id="st-1"), LATER)
    assert session_pointer(w.table, stale)["status"] == "stale"
    assert suppressing_offer(w.table, stale_job["session_id"], "supplement_existing", [SECOND_TARGET], OTHER_INPUT_HASH) is None
    sixth = w.completed_job(request_id="req-6", session_id=stale_job["session_id"])
    reissued = issue(w.table, w.receipts, sixth, w.verdict(sixth, input_hash=OTHER_INPUT_HASH), "supplement_existing",
                     [SECOND_TARGET], SCOPE_CHECK, LATER, policy_revision=DRIFTED_POLICY)
    assert reissued is not None and reissued["policy_revision"] == DRIFTED_POLICY


def test_a_job_without_a_session_gets_an_offer_without_a_session_pointer_and_skips_the_check():
    w = World()
    job = w.completed_job()
    w.table.items[keys.job(job["job_id"])]["session_id"] = None  # a record written before sessions were kept
    verdict = w.verdict(job)
    offer = issue(w.table, w.receipts, {**job, "session_id": None}, verdict, "supplement_existing", [TARGET], SCOPE_CHECK, NOW)
    assert offer is not None and offer["session_id"] is None
    assert op_keys(w.transactions()[-1]) == [
        ("Put", (f"OFFER#{offer['offer_id']}", "META")),
        ("Put", (f"JOB#{job['job_id']}", f"OFFER#{offer['offer_id']}")),
        ("Update", (f"JOB#{job['job_id']}", "META")),
    ]
    assert [row for row in w.rows("SESSION#") if row["sk"].startswith("OFFER#")] == []
    assert suppressing_offer(w.table, None, "supplement_existing", [TARGET], INPUT_HASH) is None
    assert suppressing_offer(w.table, "", "supplement_existing", [TARGET], INPUT_HASH) is None
    # decisions on such an offer update the job pointer only
    result = respond(w.table, w.receipts, w.student, response(offer, decision="decline", request_id="dec-1"), LATER)
    assert result["status"] == "declined"
    assert w.table.get(*keys.job_offer(job["job_id"], offer["offer_id"]))["status"] == "declined"


def test_suppressing_offer_validates_its_inputs_and_walks_pointer_pages():
    w = World()
    job, offer = w.offered()
    with pytest.raises(ValueError):
        suppressing_offer(w.table, job["session_id"], "correction", [TARGET], INPUT_HASH)
    with pytest.raises(ValueError):
        suppressing_offer(w.table, job["session_id"], "supplement_existing", "wiki/overviews/x.md", INPUT_HASH)
    with pytest.raises(ValueError):
        suppressing_offer(w.table, job["session_id"], "supplement_existing", [TARGET], INPUT_HASH, now=datetime(2026, 9, 21))
    # many earlier pointers of other proposals in the session: the match on the last page is still found
    for n in range(230):
        stamp = now_iso(NOW - timedelta(days=1) + timedelta(seconds=n))
        w.table.put(new_item(f"SESSION#{job['session_id']}", f"OFFER#{stamp}#{n:032x}", stamp, offer_id=f"{n:032x}",
                             job_id="x", kind="new_synthesis", targets=[f"wiki/concepts/other-{n}.md"], input_hash=f"{n:064x}",
                             status="declined"))
    assert suppressing_offer(w.table, job["session_id"], "supplement_existing", [TARGET], None) == offer
    assert suppressing_offer(w.table, job["session_id"], "new_synthesis", ["wiki/concepts/other-7.md"], None) is None  # no OFFER# record
    assert suppressing_offer(w.table, job["session_id"], "new_synthesis", ["wiki/concepts/none.md"], None) is None


# ---------------------------------------------------------------------------------------------
# Responding: ownership, revision, hash, expiry
# ---------------------------------------------------------------------------------------------

def test_non_owners_get_not_found_never_forbidden_and_nothing_is_written():
    w = World()
    _job, offer = w.offered()
    before, transactions, writes = w.rows(""), len(w.transactions()), len(w.s3.writes)
    for who in (w.other, w.admin, w.table.get("MEMBER#m2", "PROFILE")):
        with pytest.raises(NotFound) as failure:
            respond(w.table, w.receipts, who, response(offer), LATER)
        assert failure.value.code == "not_found" and isinstance(failure.value, StoreError)
        assert not isinstance(failure.value, Forbidden)
    with pytest.raises(NotFound):
        respond(w.table, w.receipts, w.student, response(offer, offer_id="0" * 32), LATER)
    assert w.rows("") == before and len(w.transactions()) == transactions and len(w.s3.writes) == writes


def test_wrong_revision_or_hash_is_a_revision_conflict_without_writing():
    w = World()
    _job, offer = w.offered()
    before = w.rows("")
    for body in (response(offer, revision=2), response(offer, revision=1_000), response(offer, hash="f" * 64),
                 response(offer, decision="decline", hash="f" * 64), response(offer, decision="decline", revision=2)):
        with pytest.raises(RevisionConflict) as failure:
            respond(w.table, w.receipts, w.student, body, LATER)
        assert failure.value.code == "revision_conflict" and isinstance(failure.value, StoreError)
    assert w.rows("") == before and len(w.s3.writes) == 2  # request.json and the offer receipt only


def test_expired_offer_is_marked_expired_once_and_refuses_both_decisions():
    w = World()
    job, offer = w.offered()
    at_expiry = NOW + timedelta(seconds=OFFER_TTL_SECONDS)
    transactions = len(w.transactions())
    with pytest.raises(Expired) as failure:
        respond(w.table, w.receipts, w.student, response(offer), at_expiry)
    assert failure.value.code == "expired" and isinstance(failure.value, StoreError)
    expired = w.table.get(*keys.offer(offer["offer_id"]))
    assert expired["status"] == "expired" and expired["expired_at"] == now_iso(at_expiry) and expired["revision"] == 2
    assert expired["decision_at"] is None and expired["execution_id"] is None
    assert w.table.get(*keys.job_offer(job["job_id"], offer["offer_id"]))["status"] == "expired"
    assert session_pointer(w.table, offer)["status"] == "expired"
    assert len(w.transactions()) == transactions + 1
    assert op_keys(w.transactions()[-1]) == [("Update", (f"OFFER#{offer['offer_id']}", "META")),
                                             ("Update", (f"JOB#{job['job_id']}", f"OFFER#{offer['offer_id']}")),
                                             ("Update", session_pointer_key(offer))]
    for decision in ("accept", "decline"):
        with pytest.raises(Expired):
            respond(w.table, w.receipts, w.student, response(offer, decision=decision, request_id=f"r-{decision}"),
                    at_expiry + timedelta(days=1))
    assert len(w.transactions()) == transactions + 1 and w.rows("APPROVAL#") == [] and w.rows("IDEMP#m1#offer") == []
    assert offer_view(expired)["status"] == "expired"
    # one microsecond before expiry the offer is still live
    live_job, live = w.offered(request_id="req-2")
    result = respond(w.table, w.receipts, w.student, response(live, decision="decline", request_id="r-live"),
                     NOW + timedelta(seconds=OFFER_TTL_SECONDS) - timedelta(microseconds=1))
    assert result["status"] == "declined" and w.table.get(*keys.job_offer(live_job["job_id"], live["offer_id"]))["status"] == "declined"


def test_malformed_response_bodies_raise_value_error():
    w = World()
    _job, offer = w.offered()
    transactions = len(w.transactions())
    for body in (response(offer, revision="1"), response(offer, revision=True), response(offer, revision=0),
                 response(offer, revision=-1), response(offer, decision="yes"),
                 response(offer, request_id="bad id"), response(offer, request_id=None), {**response(offer), "hash": 5},
                 {k: v for k, v in response(offer).items() if k != "hash"}, response(offer, offer_id="../x"), "not a body"):
        with pytest.raises(ValueError):
            respond(w.table, w.receipts, w.student, body, LATER)
    with pytest.raises(ValueError):
        respond(w.table, w.receipts, {"member_id": "m1", "role": "owner"}, response(offer), LATER)
    with pytest.raises(ValueError):
        respond(w.table, w.receipts, w.student, response(offer), datetime(2026, 9, 21, 10, 0))
    assert len(w.transactions()) == transactions and w.rows("IDEMP#m1#offer") == []


# ---------------------------------------------------------------------------------------------
# Pinned profile and policy revision
# ---------------------------------------------------------------------------------------------

def test_accept_applies_the_profile_pinned_on_the_offer_not_the_module_constant(monkeypatch):
    w = World()
    job, offer = w.offered()
    assert offer["research_profile"] == {"budget_usd_micros": 5_000_000, "reread": "auto", "max_calls": 40}
    monkeypatch.setattr(lab_offers, "RESEARCH_PROFILE", DRIFTED_PROFILE)  # the server profile moved after the offer was shown
    assert lab_offers.RESEARCH_PROFILE["budget_usd_micros"] == 50_000_000

    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)

    assert result["status"] == "accepted" and result["research_status"] == "queued"
    approval = w.table.get(*keys.approval(result["approval_id"]))
    assert approval["budget_usd_micros"] == offer["research_profile"]["budget_usd_micros"] == 5_000_000
    assert approval["max_calls"] == offer["research_profile"]["max_calls"] == 40
    assert approval["reread"] == offer["research_profile"]["reread"] == "auto"
    assert approval["policy_revision"] == offer["policy_revision"] == POLICY_REVISION
    research = w.table.get(*keys.job(result["execution_id"]))
    assert research["budget_usd_micros"] == 5_000_000 and research["policy_revision"] == offer["policy_revision"]
    reservation = w.table.get(*keys.reservation(research["reservation_id"]))
    assert reservation["micros"] == 5_000_000 and w.table.get(*keys.budget(f"job:{research['job_id']}"))["cap_micros"] == 5_000_000
    assert w.reserved("lab:2026-09") == 5_000_000 and w.reserved("member:m1:2026-09") == 5_000_000
    receipt = w.s3.json(approval["receipt_key"])
    assert receipt["budget_usd_micros"] == 5_000_000 and receipt["max_calls"] == 40 and receipt["reread"] == "auto"
    # a new offer issued now snapshots the moved profile; the earlier one keeps its own
    later_job, later = w.offered(request_id="req-2", input_hash=OTHER_INPUT_HASH)
    assert later["research_profile"] == DRIFTED_PROFILE and w.table.get(*keys.offer(offer["offer_id"]))["research_profile"] == offer["research_profile"]


def test_an_offer_issued_under_an_earlier_policy_is_marked_stale_and_refused(monkeypatch):
    w = World()
    job, offer = w.offered()
    monkeypatch.setattr(lab_offers, "RESEARCH_PROFILE", DRIFTED_PROFILE)
    monkeypatch.setattr(lab_offers, "POLICY_REVISION", DRIFTED_POLICY)
    transactions, writes = len(w.transactions()), len(w.s3.writes)

    with pytest.raises(RevisionConflict) as failure:
        respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    assert str(failure.value) == STALE_MESSAGE == "offer was issued under an earlier policy; ask again"
    assert failure.value.code == "revision_conflict"

    stale = w.table.get(*keys.offer(offer["offer_id"]))
    assert stale["status"] == "stale" and stale["stale_at"] == now_iso(LATER) and stale["revision"] == 2
    assert stale["decision"] is None and stale["execution_id"] is None and stale["approval_id"] is None
    assert stale["policy_revision"] == POLICY_REVISION and stale["research_profile"] == offer["research_profile"]
    assert w.table.get(*keys.job_offer(job["job_id"], offer["offer_id"]))["status"] == "stale"
    assert session_pointer(w.table, offer)["status"] == "stale"
    assert len(w.transactions()) == transactions + 1  # the one stale update; nothing else
    assert op_keys(w.transactions()[-1]) == [("Update", (f"OFFER#{offer['offer_id']}", "META")),
                                             ("Update", (f"JOB#{job['job_id']}", f"OFFER#{offer['offer_id']}")),
                                             ("Update", session_pointer_key(offer))]
    assert w.rows("APPROVAL#") == [] and w.rows("IDEMP#m1#offer") == [] and w.research_jobs() == []
    assert w.rows("RESERVATION#") == [r for r in w.rows("RESERVATION#") if r["job_id"] == job["job_id"]]
    assert w.reserved("lab:2026-09") == 0 and len(w.s3.writes) == writes
    assert offer_view(stale)["status"] == "stale"

    # a stale offer stays refused for either decision, with nothing more written
    for decision, request_id in (("accept", "acc-2"), ("decline", "dec-1")):
        with pytest.raises(RevisionConflict) as again:
            respond(w.table, w.receipts, w.student, response(offer, decision=decision, request_id=request_id), LATER)
        assert str(again.value) == STALE_MESSAGE
    with pytest.raises(RevisionConflict):
        respond(w.table, w.receipts, w.student, response(offer, revision=2, request_id="acc-3"), LATER)
    assert len(w.transactions()) == transactions + 1 and w.rows("IDEMP#m1#offer") == []
    # the echoed revision and hash are judged before the policy: a wrong hash on a live offer issued
    # under the earlier policy is a plain conflict that writes nothing and leaves the offer open
    _other_job, other = w.offered(request_id="req-2", input_hash=OTHER_INPUT_HASH, targets=[SECOND_TARGET])
    assert other["policy_revision"] == POLICY_REVISION != lab_offers.POLICY_REVISION
    transactions = len(w.transactions())
    with pytest.raises(RevisionConflict):
        respond(w.table, w.receipts, w.student, response(other, hash="f" * 64, request_id="acc-4"), LATER)
    assert len(w.transactions()) == transactions and w.table.get(*keys.offer(other["offer_id"]))["status"] == "offered"


def test_an_offer_explicitly_issued_under_another_policy_revision_is_stale_without_monkeypatching():
    w = World()
    job = w.completed_job()
    verdict = w.verdict(job)
    old = issue(w.table, w.receipts, job, verdict, "supplement_existing", [TARGET], SCOPE_CHECK, NOW, policy_revision="2026-01-01-v0")
    assert old["policy_revision"] == "2026-01-01-v0" and old["hash"] == expected_hash(old)
    with pytest.raises(RevisionConflict) as failure:
        respond(w.table, w.receipts, w.student, response(old, request_id="acc-1"), LATER)
    assert str(failure.value) == STALE_MESSAGE
    assert w.table.get(*keys.offer(old["offer_id"]))["status"] == "stale" and w.rows("APPROVAL#") == []


def test_a_decision_recorded_before_the_policy_moved_is_still_replayed(monkeypatch):
    w = World()
    _job, offer = w.offered()
    first = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    monkeypatch.setattr(lab_offers, "POLICY_REVISION", DRIFTED_POLICY)
    transactions = len(w.transactions())
    assert respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER) == first
    assert respond(w.table, w.receipts, w.student, response(offer, request_id="acc-2"), LATER) == first
    assert len(w.transactions()) == transactions and w.table.get(*keys.offer(offer["offer_id"]))["status"] == "accepted"


# ---------------------------------------------------------------------------------------------
# Decline
# ---------------------------------------------------------------------------------------------

def test_decline_records_the_decision_and_nothing_else():
    w = World()
    job, offer = w.offered()
    transactions, writes = len(w.transactions()), len(w.s3.writes)
    reservations, outboxes = w.rows("RESERVATION#"), w.rows("OUTBOX")
    result = respond(w.table, w.receipts, w.student, response(offer, decision="decline", request_id="dec-1"), LATER)

    assert result == {"offer_id": offer["offer_id"], "status": "declined", "execution_id": None, "approval_id": None,
                      "research_status": None}
    declined = w.table.get(*keys.offer(offer["offer_id"]))
    assert declined["status"] == "declined" and declined["decision"] == "decline" and declined["decision_at"] == now_iso(LATER)
    assert declined["execution_id"] is None and declined["approval_id"] is None and declined["revision"] == 2
    assert declined["decided_revision"] == 1 and declined["hash"] == offer["hash"]
    assert w.table.get(*keys.job_offer(job["job_id"], offer["offer_id"]))["status"] == "declined"
    assert session_pointer(w.table, offer)["status"] == "declined"
    assert len(w.transactions()) == transactions + 1
    assert op_keys(w.transactions()[-1]) == [
        ("Update", (f"OFFER#{offer['offer_id']}", "META")),
        ("Update", (f"JOB#{job['job_id']}", f"OFFER#{offer['offer_id']}")),
        ("Update", session_pointer_key(offer)),
        ("Put", ("IDEMP#m1#offer_response#dec-1", "KEY")),
    ]
    idem = w.table.get("IDEMP#m1#offer_response#dec-1", "KEY")
    assert idem["offer_id"] == offer["offer_id"] and idem["decision"] == "decline" and idem["execution_id"] is None
    assert idem["payload_hash"] == digest({"offer_id": offer["offer_id"], "revision": 1, "hash": offer["hash"], "decision": "decline"})
    assert w.rows("APPROVAL#") == [] and w.rows("RESERVATION#") == reservations and w.rows("OUTBOX") == outboxes
    assert [row["kind"] for row in w.rows("JOB#") if row["sk"] == "META"] == ["answer"]
    assert pending_outbox(w.table, "research") == [] and len(w.s3.writes) == writes
    assert w.table.get(*keys.job(job["job_id"]))["offer_id"] == offer["offer_id"]  # the question and its record stay

    # replays: same request_id, or another decline, return the same result without writing
    assert respond(w.table, w.receipts, w.student, response(offer, decision="decline", request_id="dec-1"), LATER) == result
    assert respond(w.table, w.receipts, w.student, response(offer, decision="decline", request_id="dec-2"), LATER) == result
    assert len(w.transactions()) == transactions + 1 and w.rows("IDEMP#m1#offer_response#dec-2") == []
    # a decision is final: accepting a declined offer conflicts, and reusing the request with another decision conflicts
    with pytest.raises(RevisionConflict):
        respond(w.table, w.receipts, w.student, response(offer, decision="accept", request_id="dec-3"), LATER)
    with pytest.raises(IdempotencyConflict) as failure:
        respond(w.table, w.receipts, w.student, response(offer, decision="accept", request_id="dec-1"), LATER)
    assert failure.value.code == "idempotency_conflict"
    assert len(w.transactions()) == transactions + 1 and w.rows("APPROVAL#") == []


# ---------------------------------------------------------------------------------------------
# Accept
# ---------------------------------------------------------------------------------------------

def test_accept_records_consent_and_queues_the_research_job_in_one_transaction():
    w = World()
    job, offer = w.offered("supplement_existing", [TARGET, SECOND_TARGET])
    transactions, writes = len(w.transactions()), len(w.s3.writes)
    assert w.reserved("lab:2026-09") == 0  # the answer job settled to zero
    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)

    accepted = w.table.get(*keys.offer(offer["offer_id"]))
    approval_id, execution_id = accepted["approval_id"], accepted["execution_id"]
    assert HEX.match(approval_id) and HEX.match(execution_id) and execution_id != job["job_id"]
    assert result == {"offer_id": offer["offer_id"], "status": "accepted", "execution_id": execution_id,
                      "approval_id": approval_id, "research_status": "queued"}
    assert accepted["status"] == "accepted" and accepted["decision"] == "accept" and accepted["decision_at"] == now_iso(LATER)
    assert accepted["decided_revision"] == 1 and accepted["revision"] == 2 and accepted["hash"] == offer["hash"]
    assert accepted["research_status"] == "queued"
    pointer = w.table.get(*keys.job_offer(job["job_id"], offer["offer_id"]))
    assert pointer["status"] == "accepted" and pointer["execution_id"] == execution_id
    session = session_pointer(w.table, offer)
    assert session["status"] == "accepted" and session["execution_id"] == execution_id

    approval = w.table.get(*keys.approval(approval_id))
    scope = {"question": QUESTION, "targets": [TARGET, SECOND_TARGET], "new_pages": [], "note": offer["message"],
             "kind": "supplement_existing", "offer_id": offer["offer_id"]}
    assert approval["approval_id"] == approval_id and approval["kind"] == "student_consent" and approval["revision"] == 1
    assert approval["offer_id"] == offer["offer_id"] and approval["candidate_id"] is None and approval["job_id"] == job["job_id"]
    assert approval["proposal_revision"] == 1 and approval["proposal_hash"] == offer["hash"]
    assert approval["approved_by"] == "m1" and approval["policy_revision"] == offer["policy_revision"] == POLICY_REVISION
    assert approval["scope"] == scope and approval["budget_usd_micros"] == offer["research_profile"]["budget_usd_micros"] == 5_000_000
    assert approval["model_id"] is None and approval["max_calls"] == offer["research_profile"]["max_calls"] == 40
    assert approval["reread"] == offer["research_profile"]["reread"]
    assert approval["expires_at"] == now_iso(LATER + timedelta(seconds=APPROVAL_TTL_SECONDS))
    assert approval["execution_id"] == execution_id and approval["status"] == "active" and approval["approved_at"] == now_iso(LATER)
    assert approval["request_id"] == "acc-1" and approval["research_status"] == "queued"
    assert approval["note"] is None and approval["budget_refusal"] is None and approval["linked_approval_id"] is None

    research = w.table.get(*keys.job(execution_id))
    assert research["kind"] == "research" and research["status"] == "queued" and research["parent_job_id"] == job["job_id"]
    assert research["approval_id"] == approval_id and research["member_id"] == "m1" and research["scope"] == scope
    assert research["budget_usd_micros"] == 5_000_000 and research["question"] == QUESTION
    assert research["session_id"] == job["session_id"] and research["question_hash"] == job["question_hash"]
    assert research["request_hash"] == digest(scope) and research["policy_revision"] == POLICY_REVISION
    reservation = w.table.get(*keys.reservation(research["reservation_id"]))
    assert reservation["status"] == "held" and reservation["micros"] == 5_000_000 and reservation["job_id"] == execution_id
    assert w.table.get(*keys.budget(f"job:{execution_id}"))["cap_micros"] == 5_000_000
    assert w.reserved("lab:2026-09") == 5_000_000 and w.reserved("member:m1:2026-09") == 5_000_000
    outbox = w.table.get(*keys.outbox(research["outbox_id"]))
    assert outbox["kind"] == "research" and outbox["status"] == "pending" and outbox["job_id"] == execution_id
    assert [row["outbox_id"] for row in pending_outbox(w.table, "research")] == [research["outbox_id"]]
    guard = lab_jobs.existing_execution(w.table, job["job_id"])
    assert guard["execution_id"] == execution_id and guard["approval_id"] == approval_id
    idem = w.table.get("IDEMP#m1#offer_response#acc-1", "KEY")
    assert idem["offer_id"] == offer["offer_id"] and idem["decision"] == "accept"
    assert idem["approval_id"] == approval_id and idem["execution_id"] == execution_id
    assert idem["payload_hash"] == digest({"offer_id": offer["offer_id"], "revision": 1, "hash": offer["hash"], "decision": "accept"})
    assert w.table.get(f"IDEMP#m1#research#{approval_id}", "KEY")["job_id"] == execution_id
    assert w.table.get(*keys.job(job["job_id"]))["revision"] == job["revision"] + 1  # untouched since the offer linked it

    assert len(w.transactions()) == transactions + 1
    touched = op_keys(w.transactions()[-1])
    assert len(touched) == len(set(touched))  # each item once
    assert touched[:5] == [
        ("Update", (f"OFFER#{offer['offer_id']}", "META")),
        ("Update", (f"JOB#{job['job_id']}", f"OFFER#{offer['offer_id']}")),
        ("Update", session_pointer_key(offer)),
        ("Put", (f"APPROVAL#{approval_id}", "META")),
        ("Put", ("IDEMP#m1#offer_response#acc-1", "KEY")),
    ]
    assert ("Put", (f"IDEMP#m1#research#{approval_id}", "KEY")) in touched
    assert ("Put", (f"JOB#{execution_id}", "META")) in touched
    assert ("Put", (f"JOB#{job['job_id']}", "EXECUTION")) in touched
    assert ("Put", (f"RESERVATION#{research['reservation_id']}", "META")) in touched
    assert ("Put", (f"OUTBOX#{research['outbox_id']}", "META")) in touched
    assert ("Update", ("BUDGET#lab:2026-09", "META")) in touched and ("Update", ("BUDGET#member:m1:2026-09", "META")) in touched
    assert ("Put", (f"BUDGET#job:{execution_id}", "META")) in touched

    approval_key = f"runs/lab-questions/{job['job_id']}/approval-{approval_id}.json"
    assert approval["receipt_key"] == approval_key
    assert w.s3.writes[writes:] == [(research["request_key"], {"IfNoneMatch": "*"}), (approval_key, {"IfNoneMatch": "*"})]
    receipt = w.s3.json(approval_key)
    assert receipt["approval_id"] == approval_id and receipt["kind"] == "student_consent" and receipt["approved_by"] == "m1"
    assert receipt["proposal_hash"] == offer["hash"] and receipt["scope"] == scope and receipt["execution_id"] == execution_id
    assert receipt["budget_usd_micros"] == 5_000_000 and receipt["expires_at"] == approval["expires_at"]
    assert receipt["approved_at"] == now_iso(LATER) and receipt["policy_revision"] == POLICY_REVISION
    assert receipt["research_status"] == "queued"
    assert "pk" not in receipt and "receipt_sha256" not in receipt and "status" not in receipt
    assert digest(receipt) == approval["receipt_sha256"]
    request = w.s3.json(research["request_key"])
    assert request["approval_id"] == approval_id and request["parent_job_id"] == job["job_id"] and request["scope"] == scope
    assert all(key.startswith("runs/lab-questions/") for key, _ in w.s3.writes)


def test_new_synthesis_acceptance_puts_the_targets_under_new_pages():
    w = World()
    job, offer = w.offered("new_synthesis", ["wiki/concepts/chd8-macrocephaly.md"])
    respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    approval = w.table.get(*keys.approval(w.table.get(*keys.offer(offer["offer_id"]))["approval_id"]))
    assert approval["scope"]["targets"] == [] and approval["scope"]["new_pages"] == ["wiki/concepts/chd8-macrocephaly.md"]
    assert approval["scope"]["kind"] == "new_synthesis" and approval["scope"]["question"] == job["question"]


def test_second_accept_returns_the_same_execution_without_writing():
    w = World()
    job, offer = w.offered()
    first = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    transactions, writes, rows = len(w.transactions()), len(w.s3.writes), w.rows("")

    different = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-2"), LATER + timedelta(hours=1))
    assert different == first and different["execution_id"] == first["execution_id"]
    assert len(w.transactions()) == transactions and len(w.s3.writes) == writes and w.rows("") == rows
    assert w.table.get("IDEMP#m1#offer_response#acc-2", "KEY") is None

    same = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER + timedelta(hours=2))
    assert same == first and len(w.transactions()) == transactions and len(w.s3.writes) == writes and w.rows("") == rows

    # the accepted offer is reported with its execution; the store revision moved on but the hash still identifies it
    accepted = w.table.get(*keys.offer(offer["offer_id"]))
    assert accepted["revision"] == 2 and offer_view(accepted)["status"] == "accepted"
    with pytest.raises(RevisionConflict):  # a decline after acceptance does not undo the consent
        respond(w.table, w.receipts, w.student, response(offer, decision="decline", request_id="acc-3"), LATER)
    with pytest.raises(RevisionConflict):
        respond(w.table, w.receipts, w.student, response(offer, request_id="acc-4", hash="0" * 64), LATER)
    with pytest.raises(IdempotencyConflict):
        respond(w.table, w.receipts, w.student, response(offer, decision="decline", request_id="acc-1"), LATER)
    assert len(w.transactions()) == transactions and len(w.research_jobs()) == 1
    assert len(pending_outbox(w.table, "research")) == 1 and len(w.rows("APPROVAL#")) == 1
    assert job["job_id"] != first["execution_id"]


def test_replaying_the_same_request_rewrites_a_missing_approval_receipt_once():
    w = World()
    _job, offer = w.offered()
    first = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    approval = w.table.get(*keys.approval(first["approval_id"]))
    writes, transactions = len(w.s3.writes), len(w.transactions())
    del w.s3.objects[approval["receipt_key"]]  # a crash between the transaction and the receipt put

    assert respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER) == first
    assert w.s3.writes[writes:] == [(approval["receipt_key"], {"IfNoneMatch": "*"})]
    assert digest(w.s3.json(approval["receipt_key"])) == approval["receipt_sha256"]
    assert respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER) == first  # 412 tolerated
    assert len(w.s3.writes) == writes + 1 and len(w.transactions()) == transactions


# ---------------------------------------------------------------------------------------------
# Accept under a budget refusal: the consent is kept, the research job pauses
# ---------------------------------------------------------------------------------------------

def test_budget_refusal_on_accept_records_the_consent_with_the_research_job_paused():
    w = World()
    job, offer = w.offered()
    lab_budget.set_cap(w.table, "lab:2026-09", 1_000_000, now=NOW)
    reservations, transactions, writes = w.rows("RESERVATION#"), len(w.transactions()), len(w.s3.writes)

    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)

    accepted = w.table.get(*keys.offer(offer["offer_id"]))
    approval_id, execution_id = accepted["approval_id"], accepted["execution_id"]
    assert HEX.match(approval_id) and HEX.match(execution_id)
    assert result == {"offer_id": offer["offer_id"], "status": "accepted", "execution_id": execution_id,
                      "approval_id": approval_id, "research_status": "paused_budget"}
    assert accepted["status"] == "accepted" and accepted["decision"] == "accept" and accepted["research_status"] == "paused_budget"
    assert w.table.get(*keys.job_offer(job["job_id"], offer["offer_id"]))["execution_id"] == execution_id
    assert session_pointer(w.table, offer)["status"] == "accepted"

    approval = w.table.get(*keys.approval(approval_id))
    assert approval["kind"] == "student_consent" and approval["status"] == "active" and approval["execution_id"] == execution_id
    assert approval["budget_usd_micros"] == 5_000_000 and approval["policy_revision"] == offer["policy_revision"]
    assert approval["research_status"] == "paused_budget" and approval["note"] is None
    assert approval["budget_refusal"] == {"scope": "lab:2026-09", "requested_micros": 5_000_000, "available_micros": 1_000_000,
                                          "reason": "lab:2026-09 cannot reserve 5000000 micro-USD; 1000000 available",
                                          "refused_at": now_iso(LATER)}

    research = w.table.get(*keys.job(execution_id))
    assert research["kind"] == "research" and research["status"] == "paused_budget" and research["parent_job_id"] == job["job_id"]
    assert research["approval_id"] == approval_id and research["budget_usd_micros"] == 5_000_000
    assert research["reservation_id"] is None and research["outbox_id"] is None and research["queued_at"] is None
    assert research["period"] == "2026-09" and research["policy_revision"] == offer["policy_revision"]
    assert lab_jobs.existing_execution(w.table, job["job_id"])["execution_id"] == execution_id
    assert w.rows("RESERVATION#") == reservations and w.table.get(*keys.budget(f"job:{execution_id}")) is None
    assert w.reserved("lab:2026-09") == 0 and w.reserved("member:m1:2026-09") == 0
    assert pending_outbox(w.table, "research") == [] and w.rows("OUTBOX") == [r for r in w.rows("OUTBOX") if r.get("kind") != "research"]
    idem = w.table.get("IDEMP#m1#offer_response#acc-1", "KEY")
    assert idem["approval_id"] == approval_id and idem["execution_id"] == execution_id
    assert w.table.get(f"IDEMP#m1#research#{approval_id}", "KEY")["job_id"] == execution_id

    assert len(w.transactions()) == transactions + 1
    touched = op_keys(w.transactions()[-1])
    assert touched[:5] == [
        ("Update", (f"OFFER#{offer['offer_id']}", "META")),
        ("Update", (f"JOB#{job['job_id']}", f"OFFER#{offer['offer_id']}")),
        ("Update", session_pointer_key(offer)),
        ("Put", (f"APPROVAL#{approval_id}", "META")),
        ("Put", ("IDEMP#m1#offer_response#acc-1", "KEY")),
    ]
    assert ("Put", (f"JOB#{execution_id}", "META")) in touched and ("Put", (f"JOB#{job['job_id']}", "EXECUTION")) in touched
    assert not any(pk.startswith(("BUDGET#", "RESERVATION#", "OUTBOX")) for _kind, (pk, _sk) in touched)
    assert w.s3.writes[writes:] == [(research["request_key"], {"IfNoneMatch": "*"}), (approval["receipt_key"], {"IfNoneMatch": "*"})]
    receipt = w.s3.json(approval["receipt_key"])
    assert receipt["research_status"] == "paused_budget" and receipt["budget_refusal"]["scope"] == "lab:2026-09"
    assert digest(receipt) == approval["receipt_sha256"]

    # replays return the recorded consent; lifting the cap does not start a second execution
    assert respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER) == result
    lab_budget.set_cap(w.table, "lab:2026-09", None, now=NOW)
    assert respond(w.table, w.receipts, w.student, response(offer, request_id="acc-2"), LATER) == result
    assert len(w.research_jobs()) == 1 and len(w.rows("APPROVAL#")) == 1 and w.reserved("lab:2026-09") == 0


def test_a_member_cap_refusal_pauses_the_research_job_as_well():
    w = World()
    job, offer = w.offered()
    lab_budget.set_cap(w.table, "member:m1:2026-09", 4_999_999, now=NOW)
    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    assert result["status"] == "accepted" and result["research_status"] == "paused_budget"
    approval = w.table.get(*keys.approval(result["approval_id"]))
    assert approval["budget_refusal"]["scope"] == "member:m1:2026-09" and approval["budget_refusal"]["available_micros"] == 4_999_999
    assert w.table.get(*keys.job(result["execution_id"]))["status"] == "paused_budget"
    assert w.reserved("lab:2026-09") == 0 and w.reserved("member:m1:2026-09") == 0


def test_accept_rereads_and_replans_when_a_concurrent_writer_moves_a_budget_scope():
    w = World()
    _job, offer = w.offered()
    real_transact = w.table.transact
    state = {"interfered": False}

    def interfering_transact(operations):
        if not state["interfered"] and any(isinstance(op, Update) and op.pk == "BUDGET#lab:2026-09" for op in operations):
            state["interfered"] = True
            lab_budget.set_cap(w.table, "lab:2026-09", None, now=NOW)  # another writer bumps the scope's revision
        return real_transact(operations)

    w.table.transact = interfering_transact
    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    assert result["status"] == "accepted" and result["research_status"] == "queued" and state["interfered"]
    research = w.table.get(*keys.job(result["execution_id"]))
    assert research["status"] == "queued" and len(w.rows("APPROVAL#")) == 1
    assert len(w.research_jobs()) == 1
    assert w.reserved("lab:2026-09") == 5_000_000


def test_a_concurrent_accept_that_lands_first_makes_the_second_request_return_its_execution():
    w = World()
    _job, offer = w.offered()
    real_transact = w.table.transact
    state = {"raced": False}

    def racing_transact(operations):
        if not state["raced"] and any(isinstance(op, Put) and op.item["pk"].startswith("APPROVAL#") for op in operations):
            state["raced"] = True
            w.table.transact = real_transact
            respond(w.table, w.receipts, w.student, response(offer, request_id="other-device"), LATER)
            w.table.transact = racing_transact
        return real_transact(operations)

    w.table.transact = racing_transact
    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    first = w.table.get(*keys.offer(offer["offer_id"]))
    assert state["raced"] and result["execution_id"] == first["execution_id"] and result["status"] == "accepted"
    assert len(w.rows("APPROVAL#")) == 1 and len(w.research_jobs()) == 1
    assert w.table.get("IDEMP#m1#offer_response#acc-1", "KEY") is None  # the loser wrote nothing


def test_accepting_an_offer_whose_job_vanished_is_not_found():
    w = World()
    job, offer = w.offered()
    del w.table.items[keys.job(job["job_id"])]
    with pytest.raises(NotFound):
        respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    assert w.table.get(*keys.offer(offer["offer_id"]))["status"] == "offered"


# ---------------------------------------------------------------------------------------------
# One execution per parent job: a consent links to an execution another path started
# ---------------------------------------------------------------------------------------------

def test_accept_links_the_execution_a_professor_already_started_instead_of_planning_another():
    w = World()
    job, offer = w.offered()
    professor = w.professor_execution(job, approval_id="prof-approval-1")
    assert professor["status"] == "queued" and professor["parent_job_id"] == job["job_id"] and w.reserved("lab:2026-09") == 5_000_000
    assert lab_jobs.existing_execution(w.table, job["job_id"])["execution_id"] == professor["job_id"]
    rows_before = {k: v for k, v in w.table.items.items()}
    reservations, outboxes = w.rows("RESERVATION#"), w.rows("OUTBOX")
    transactions, writes = len(w.transactions()), len(w.s3.writes)

    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)

    accepted = w.table.get(*keys.offer(offer["offer_id"]))
    approval_id = accepted["approval_id"]
    assert result == {"offer_id": offer["offer_id"], "status": "accepted", "execution_id": professor["job_id"],
                      "approval_id": approval_id, "research_status": "queued"}
    assert accepted["status"] == "accepted" and accepted["execution_id"] == professor["job_id"]
    assert w.table.get(*keys.job_offer(job["job_id"], offer["offer_id"]))["execution_id"] == professor["job_id"]
    assert session_pointer(w.table, offer)["execution_id"] == professor["job_id"]
    approval = w.table.get(*keys.approval(approval_id))
    assert approval["kind"] == "student_consent" and approval["execution_id"] == professor["job_id"] and approval["status"] == "active"
    assert approval["note"] == LINKED_NOTE == "linked to an execution started by another path"
    assert approval["linked_approval_id"] == "prof-approval-1" and approval["research_status"] == "queued"
    assert approval["budget_usd_micros"] == 5_000_000 and approval["policy_revision"] == offer["policy_revision"]
    assert approval["budget_refusal"] is None and approval["offer_id"] == offer["offer_id"] and approval["job_id"] == job["job_id"]
    idem = w.table.get("IDEMP#m1#offer_response#acc-1", "KEY")
    assert idem["approval_id"] == approval_id and idem["execution_id"] == professor["job_id"]

    # exactly one research execution, no second reservation, outbox row or research idempotency key
    assert [row["job_id"] for row in w.research_jobs()] == [professor["job_id"]]
    assert w.table.get(*keys.job(professor["job_id"])) == rows_before[keys.job(professor["job_id"])]
    assert w.rows("RESERVATION#") == reservations and w.rows("OUTBOX") == outboxes
    assert w.reserved("lab:2026-09") == 5_000_000 and w.reserved("member:m1:2026-09") == 5_000_000
    assert [row["job_id"] for row in pending_outbox(w.table, "research")] == [professor["job_id"]]
    assert w.table.get(f"IDEMP#m1#research#{approval_id}", "KEY") is None
    assert lab_jobs.existing_execution(w.table, job["job_id"])["approval_id"] == "prof-approval-1"
    assert len(w.transactions()) == transactions + 1
    assert op_keys(w.transactions()[-1]) == [
        ("Update", (f"OFFER#{offer['offer_id']}", "META")),
        ("Update", (f"JOB#{job['job_id']}", f"OFFER#{offer['offer_id']}")),
        ("Update", session_pointer_key(offer)),
        ("Put", (f"APPROVAL#{approval_id}", "META")),
        ("Put", ("IDEMP#m1#offer_response#acc-1", "KEY")),
    ]
    assert w.s3.writes[writes:] == [(approval["receipt_key"], {"IfNoneMatch": "*"})]
    receipt = w.s3.json(approval["receipt_key"])
    assert receipt["note"] == LINKED_NOTE and receipt["execution_id"] == professor["job_id"] and digest(receipt) == approval["receipt_sha256"]
    # replays and a second accept return the linked execution without writing
    assert respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER) == result
    assert respond(w.table, w.receipts, w.student, response(offer, request_id="acc-2"), LATER) == result
    assert len(w.transactions()) == transactions + 1 and len(w.rows("APPROVAL#")) == 1


def test_a_professor_execution_that_lands_during_the_accept_is_linked_on_the_retry():
    w = World()
    job, offer = w.offered()
    real_transact = w.table.transact
    state: dict = {"raced": False, "execution_id": None, "failed": 0}

    def racing_transact(operations):
        if not state["raced"] and any(isinstance(op, Put) and op.item["sk"] == lab_jobs.EXECUTION_SK for op in operations):
            state["raced"] = True
            w.table.transact = real_transact
            state["execution_id"] = w.professor_execution(job, approval_id="prof-approval-2")["job_id"]
            w.table.transact = racing_transact
        try:
            return real_transact(operations)
        except ConditionFailed:
            state["failed"] += 1
            raise

    w.table.transact = racing_transact
    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    w.table.transact = real_transact

    assert state["raced"] and state["failed"] == 1  # the student's first transaction lost to the guard
    assert result["status"] == "accepted" and result["execution_id"] == state["execution_id"] and result["research_status"] == "queued"
    approval = w.table.get(*keys.approval(result["approval_id"]))
    assert approval["execution_id"] == state["execution_id"] and approval["note"] == LINKED_NOTE
    assert approval["linked_approval_id"] == "prof-approval-2"
    assert [row["job_id"] for row in w.research_jobs()] == [state["execution_id"]]
    assert w.reserved("lab:2026-09") == 5_000_000 and len(w.rows("APPROVAL#")) == 1
    assert len([row for row in w.rows("IDEMP#m1#research")]) == 1  # the professor's only; the losing plan wrote nothing


def test_a_paused_execution_started_by_another_path_is_linked_with_its_status():
    w = World()
    job, offer = w.offered()
    parent = w.table.get(*keys.job(job["job_id"]))
    plan = lab_jobs.plan_research_job(w.table, parent_job=parent, member_id="m1", approval_id="prof-paused", scope={"question": QUESTION},
                                      budget_usd_micros=5_000_000, now=NOW, reserve=False, status="paused_budget")
    w.table.transact(list(plan.operations))
    result = respond(w.table, w.receipts, w.student, response(offer, request_id="acc-1"), LATER)
    assert result["execution_id"] == plan.job["job_id"] and result["research_status"] == "paused_budget"
    assert w.table.get(*keys.approval(result["approval_id"]))["research_status"] == "paused_budget"
    assert w.reserved("lab:2026-09") == 0 and len(w.research_jobs()) == 1


# ---------------------------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------------------------

def test_module_imports_stay_inside_the_lab_boundary():
    source = Path(lab_offers.__file__).read_text(encoding="utf-8")
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    forbidden = {"byeori.ingest_lambda", "byeori.aws_store", "byeori.question_agent",
                 "byeori.agent_cache", "mcp", "httpx", "boto3"}
    assert not (names & forbidden), names & forbidden
    assert not any(name.startswith(("mcp.", "httpx.", "boto3.")) for name in names)
    assert "put_object" not in source and "Delete(" not in source and "delete_" not in source
    assert lab_offers.RevisionConflict.code == "revision_conflict" and lab_offers.Expired.code == "expired"
    assert lab_offers.NotFound is lab_jobs.NotFound and lab_offers.IdempotencyConflict is lab_jobs.IdempotencyConflict
    assert issubclass(RevisionConflict, StoreError) and issubclass(Expired, StoreError)
    assert not issubclass(RevisionConflict, ConditionFailed)
    assert lab_offers.OFFER_STATUSES == {"offered", "accepted", "declined", "expired", "stale"}
    assert lab_offers.SUPPRESSING_STATUSES == {"offered", "accepted", "declined"}
