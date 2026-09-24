"""Asynchronous Jev triage of byeori.lab_triage: server-side skips, reuse, one call, 0.99 offers."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from byeori import jev_client, lab_jobs, lab_offers, lab_triage
from byeori.evidence_packet import build_packet
from byeori.jev_client import JevError, payload_state
from byeori.lab_jobs import InvalidTransition, Member, NotFound, claim, complete, intake, queue
from byeori.lab_offers import offer_for_job
from byeori.lab_policy import (
    ANSWER_JOB_CAP_MICROS,
    JEV_MODEL,
    LEASE_SECONDS,
    OFFER_TEMPLATES,
    POLICY_REVISION,
    REVIEW_CANDIDATE_CUTOFF,
    SCOPE_MATCH_SCORE,
)
from byeori.lab_store import ConditionFailed, Put, ReceiptWriter, canonical, digest, keys, receipt_key
from byeori.lab_triage import TriagePolicy, triage_job
from lab_fakes import FakeJev, MemoryTable, index_connection, jev_response, member, source_note, wiki_with_index

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
INDEX_KEY = "index/wiki-index-v2.sqlite3"
SECRET = "sk-live-9f8e7d6c5b4a-SECRET-VALUE"
QUESTION = "Was regional inheritance stable in the cohort?"
KOREAN_QUESTION = "코호트에서 지역 유전 안정성 결과는 무엇인가요? 표본 크기도 함께 알려 주세요."
KOREAN_NOTE = ("---\ntitle: 지역 유전 노트\ncategory: asd-ndd\nyear: 2021\n---\n\n# 지역 유전 노트\n\n한 줄 요약.\n\n"
               "## 결과\n\n코호트에서 지역 유전 안정성 결과는 유지되었다 (n = 120, p = 0.01).\n\n## 한계\n\n표본이 작고 단일 기관이다.\n")
OVERVIEW_KEY = "wiki/overviews/asd-ndd/regional-inheritance.md"
CONCEPT_KEY = "wiki/concepts/macrocephaly.md"
OVERVIEW = ("---\ntitle: Regional inheritance overview\n---\n\n# Regional inheritance overview\n\n"
            "## Synthesis\n\nAcross cohorts, regional inheritance was stable in most families; one study of 120 families "
            "reported p = 0.01.\n\n## Open questions\n\nWhether stability holds in larger cohorts is untested.\n")
CONCEPT = "---\ntitle: Macrocephaly\n---\n\n# Macrocephaly\n\n## Definition\n\nHead circumference above the 97th percentile.\n"
PAGES = {
    "wiki/sources/paper-one.md": source_note(),
    "wiki/sources/paper-two.md": source_note("Paper two", stem="paper-two",
                                             results="Inheritance patterns differed by region in 300 families.",
                                             limitations="Ancestry was self-reported."),
    "wiki/sources/korean-note.md": KOREAN_NOTE,
    OVERVIEW_KEY: OVERVIEW,
    CONCEPT_KEY: CONCEPT,
}
ANSWER_TEXT = "Regional inheritance was stable in the cohort (n = 120, p = 0.01); the cohort was small and single-site."
LIMITATIONS = ["Single-site cohort of 120 families."]
NO_HINT = {"kind": "none", "target_keys": [], "note": ""}
NEW_HINT = {"kind": "new_synthesis", "target_keys": [], "note": "No synthesis compares regional stability across cohorts."}
SUPPLEMENT_HINT = {"kind": "supplement_existing", "target_keys": [OVERVIEW_KEY],
                   "note": "The overview lacks the 300-family regional difference."}
USAGE = {"inputTokens": 1200, "outputTokens": 300, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
PASS = {"answer_only": 0.005, "needs_lookup": 0.005, "review_candidate": 0.99}
NEAR = {"answer_only": 0.0051, "needs_lookup": 0.005, "review_candidate": 0.9899}
HIGH = {"answer_only": 0.002, "needs_lookup": 0.003, "review_candidate": 0.995}

# Session-level offer suppression lands in lab_offers alongside this module; until then the
# re-offer test has nothing to observe and is skipped rather than pinned to the old behaviour.
needs_offer_suppression = pytest.mark.skipif(not hasattr(lab_offers, "suppressing_offer"),
                                             reason="lab_offers.suppressing_offer has not landed")


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Record the jittered pauses between transaction attempts instead of sleeping through them."""
    pauses: list[float] = []
    monkeypatch.setattr(lab_jobs, "_sleep", pauses.append)
    return pauses


class World:
    """A control table, a wiki bucket with its index, two students and finished answer jobs."""

    def __init__(self, pages: dict | None = None):
        self.table = MemoryTable()
        self.s3 = wiki_with_index(pages or PAGES)
        self.receipts = ReceiptWriter(self.s3, "bucket")
        member(self.table, "m1")
        member(self.table, "m2")
        self.index = (index_connection(self.s3.objects[INDEX_KEY]), self.s3.etag(INDEX_KEY))
        self.baseline_writes = len(self.s3.writes)
        self.baseline_transactions = len(self.table.transactions)
        self.marked = False

    def mark(self):
        """Take the write and transaction baseline once, after the fixture jobs were set up."""
        if not self.marked:
            self.baseline_writes = len(self.s3.writes)
            self.baseline_transactions = len(self.table.transactions)
            self.marked = True

    def finished_job(self, *, question=QUESTION, answer=ANSWER_TEXT, hint=None, limitations=None, packet=None,
                     hold_reason=None, status="completed", private_material=False, who="m1", request_id="req-1",
                     context=None, evidence_state=None, answer_evidence_state="sufficient", session_id=None):
        body = {"request_id": request_id, "question": question, "private_material": private_material}
        if context is not None:
            body["context"] = context
        if session_id is not None:
            body["session_id"] = session_id
        job = intake(self.table, self.receipts, Member(who), body, NOW)
        job = queue(self.table, job["job_id"], period="2026-09", cap=ANSWER_JOB_CAP_MICROS, now=NOW)
        job = claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, NOW)
        job_id = job["job_id"]
        if packet is None:
            packet = build_packet(question, index=self.index, s3=self.s3, bucket="bucket")
            if evidence_state is not None:
                packet = {**packet, "evidence_state": evidence_state}
        evidence = self.receipts.put_json(receipt_key(job_id, "evidence.json"), packet)
        record = {
            "job_id": job_id, "kind": "answer", "member_id": who, "session_id": job["session_id"], "turn": job["turn"],
            "parent_job_id": None, "question": question, "standalone_question": job.get("standalone_question"),
            "context": context or [], "question_hash": job["question_hash"], "context_hash": job["context_hash"],
            "request_id": request_id, "policy_revision": POLICY_REVISION, "attempt": 1, "model_id": "claude-opus-5",
            "reasoning": None, "thinking": "off", "status": status, "hold_reason": hold_reason, "answer": answer,
            "citations": [{"key": "wiki/sources/paper-one.md", "section": "Results", "verified": True}],
            "limitations": LIMITATIONS if limitations is None else limitations, "evidence_state": answer_evidence_state,
            "unresolved_items": [], "maintenance_hint": hint or NO_HINT, "packet_evidence_state": packet["evidence_state"],
            "index_etag": packet.get("index_etag"), "queries": [q["query"] for q in packet["queries"]],
            "english_query": None, "lookup": None,
            "documents": [{"key": d["key"], "etag": d["etag"], "version_id": d["version_id"], "sha256": d["sha256"]}
                          for d in packet["documents"]],
            "evidence_key": evidence["key"], "evidence_sha256": evidence["sha256"], "calls": [], "holds": [],
            "usage": USAGE, "usd_micros": 1200, "completed_at": "2026-09-21T09:00:00.000000+00:00",
        }
        receipt = self.receipts.put_json(receipt_key(job_id, "answer.json"), record)
        return complete(self.table, job_id, job["revision"], receipt_key=receipt["key"], evidence_key=evidence["key"],
                        usage=USAGE, usd_micros=1200, status=status, hold_reason=hold_reason, now=NOW)

    def triage(self, job, jev, **overrides):
        self.mark()
        options = dict(table=self.table, receipts=self.receipts, s3=self.s3, bucket="bucket", index=self.index,
                       jev_post=jev, secret_reader=lambda: SECRET, now=NOW)
        options.update(overrides)
        return triage_job(job, **options)

    def job(self, job_id):
        return self.table.get(*keys.job(job_id))

    def outbox(self, job):
        return self.table.get(*keys.outbox(job["triage_outbox_id"]))

    def receipt_writes(self):
        return self.s3.writes[self.baseline_writes:]

    def transactions(self):
        return self.table.transactions[self.baseline_transactions:]

    def packet(self, job):
        return self.s3.json(f"runs/lab-questions/{job['job_id']}/evidence.json")

    def expected_hash(self, job, answer=ANSWER_TEXT):
        return digest({"question": job["question"], "context_hash": job["context_hash"],
                       "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                       "evidence_sha256": digest(self.packet(job)), "policy_revision": POLICY_REVISION,
                       "model": JEV_MODEL})


def jev(*responses):
    return FakeJev(list(responses))


def assert_secret_absent(w: World):
    for item in w.table.items.values():
        assert SECRET not in canonical(item).decode("utf-8")
    for key, body in w.s3.objects.items():
        assert SECRET not in body.decode("utf-8", "ignore"), key


def assert_only_receipt_writes(w: World, job, *names):
    prefix = f"runs/lab-questions/{job['job_id']}/"
    assert [k for k, _ in w.receipt_writes()] == [prefix + name for name in names]
    assert all(conditions == {"IfNoneMatch": "*"} for _, conditions in w.receipt_writes())
    assert all(k.startswith("runs/lab-questions/") for k, _ in w.s3.writes)


def transaction_with(w: World, pk, sk):
    for operations in w.transactions():
        if any(isinstance(op, Put) and (op.item["pk"], op.item["sk"]) == (pk, sk) for op in operations):
            return operations
    raise AssertionError(f"no transaction put {pk}/{sk}")


# ---------------------------------------------------------------------------------------------
# Server-side skips: no HTTP call
# ---------------------------------------------------------------------------------------------

def test_private_material_is_skipped_without_a_call_and_the_job_closes_as_skipped():
    w = World()
    job = w.finished_job(private_material=True, hint=NEW_HINT)
    fake = jev()

    verdict = w.triage(job, fake)

    assert fake.calls == []
    assert verdict["status"] == "skipped" and verdict["reason"] == "private_material"
    assert verdict["probabilities"] is None and verdict["choice"] is None and verdict["confidence"] is None
    assert verdict["passed_cutoff"] is False and verdict["candidate_status"] == "needs_lookup"
    assert verdict["jev_called"] is False and verdict["usd_micros"] == 0 and verdict["usage"] is None
    assert verdict["input_hash"] == w.expected_hash(job)
    assert verdict == w.table.get(*keys.verdict(job["job_id"]))
    stored = w.job(job["job_id"])
    assert stored["triage_status"] == "skipped" and stored["offer_id"] is None and stored["status"] == "completed"
    assert w.outbox(stored)["status"] == "done"
    assert w.table.get(*keys.verdict_by_input(verdict["input_hash"])) is None
    assert offer_for_job(w.table, job["job_id"]) is None
    assert_only_receipt_writes(w, job, "triage.json")
    assert_secret_absent(w)


@pytest.mark.parametrize("state", ["insufficient", "links_only", "truncated_decisive"])
def test_server_detected_needs_lookup_states_skip_jev_with_null_probabilities(state):
    w = World()
    job = w.finished_job(hint=NEW_HINT, evidence_state=state)
    fake = jev()

    verdict = w.triage(job, fake)

    assert fake.calls == []
    assert verdict["status"] == "skipped" and verdict["reason"] == f"evidence_{state}"
    assert verdict["evidence_state"] == state
    assert verdict["probabilities"] is None and verdict["candidate_status"] == "needs_lookup"
    assert verdict["cutoff"] == REVIEW_CANDIDATE_CUTOFF and verdict["model"] == JEV_MODEL
    assert verdict["policy_revision"] == POLICY_REVISION
    assert w.job(job["job_id"])["triage_status"] == "skipped"
    assert offer_for_job(w.table, job["job_id"]) is None


def test_a_real_insufficient_packet_from_a_question_with_no_hits_is_skipped():
    w = World()
    job = w.finished_job(question="zebrafish fin regeneration telomerase", hint=NEW_HINT)
    assert w.packet(job)["evidence_state"] == "insufficient"
    fake = jev()

    verdict = w.triage(job, fake)

    assert fake.calls == [] and verdict["status"] == "skipped" and verdict["reason"] == "evidence_insufficient"


def test_an_answer_that_declares_its_evidence_insufficient_is_skipped_although_the_packet_was_sufficient():
    w = World()
    job = w.finished_job(hint=NEW_HINT, answer_evidence_state="insufficient")
    assert w.packet(job)["evidence_state"] == "sufficient"
    assert w.s3.json(f"runs/lab-questions/{job['job_id']}/answer.json")["hold_reason"] is None
    fake = jev(jev_response("review_candidate", probabilities=PASS))

    verdict = w.triage(job, fake)

    assert fake.calls == [] and verdict["jev_called"] is False
    assert verdict["status"] == "skipped" and verdict["reason"] == "answer_insufficient"
    assert verdict["candidate_status"] == "needs_lookup" and verdict["probabilities"] is None
    assert verdict["evidence_state"] == "sufficient"   # the packet's own state stays on record
    assert verdict["scope_check"] is None and verdict["offer"] is None
    assert offer_for_job(w.table, job["job_id"]) is None
    assert w.job(job["job_id"])["triage_status"] == "skipped" and w.job(job["job_id"])["offer_id"] is None
    assert w.table.get(*keys.verdict_by_input(verdict["input_hash"])) is None
    assert_only_receipt_writes(w, job, "triage.json")


def test_empty_and_held_answers_are_skipped_as_answer_only():
    w = World()
    empty = w.finished_job(answer="", hold_reason="empty_answer", status="partial", request_id="req-e")
    held = w.finished_job(answer="Partial text.", hold_reason="time_budget", status="partial", request_id="req-h")
    fake = jev()

    first = w.triage(empty, fake)
    second = w.triage(held, fake)

    assert fake.calls == []
    assert first["status"] == "skipped" and first["reason"] == "empty_answer" and first["candidate_status"] == "answer_only"
    assert second["status"] == "skipped" and second["reason"] == "held_answer" and second["candidate_status"] == "answer_only"
    assert first["probabilities"] is None and second["probabilities"] is None
    assert w.job(empty["job_id"])["triage_status"] == "skipped" and w.job(held["job_id"])["triage_status"] == "skipped"


def test_payload_that_cannot_fit_is_needs_lookup_with_input_too_large():
    w = World()
    job = w.finished_job(answer="가" * 20_000, hint=NEW_HINT)
    fake = jev()

    verdict = w.triage(job, fake)

    assert fake.calls == []
    assert verdict["status"] == "skipped" and verdict["reason"] == "input_too_large"
    assert verdict["candidate_status"] == "needs_lookup" and verdict["probabilities"] is None
    assert w.job(job["job_id"])["triage_status"] == "skipped"


# ---------------------------------------------------------------------------------------------
# A valid verdict
# ---------------------------------------------------------------------------------------------

def test_valid_response_stores_raw_probabilities_confidence_cutoff_and_the_reuse_pointer():
    w = World()
    job = w.finished_job(hint=NO_HINT)
    fake = jev(jev_response("answer_only", probabilities={"answer_only": 0.9, "needs_lookup": 0.05, "review_candidate": 0.05},
                            confidence=0.81, input_tokens=3000, output_tokens=40))

    verdict = w.triage(job, fake)

    assert len(fake.calls) == 1 and fake.calls[0][1] == SECRET
    assert verdict["status"] == "complete" and verdict["choice"] == "answer_only"
    assert verdict["probabilities"] == {"answer_only": 0.9, "needs_lookup": 0.05, "review_candidate": 0.05}
    assert all(type(p) is float for p in verdict["probabilities"].values())
    assert verdict["confidence"] == 0.81 and verdict["cutoff"] == 0.99 and verdict["passed_cutoff"] is False
    assert verdict["candidate_status"] == "answer_only" and verdict["jev_called"] is True
    assert verdict["model"] == JEV_MODEL and verdict["policy_revision"] == POLICY_REVISION
    assert verdict["usage"] == {"input_tokens": 3000, "output_tokens": 40} and verdict["usd_micros"] == 126
    assert verdict["error_code"] is None and verdict["reason"] is None and verdict["reused_from_job_id"] is None
    assert verdict["input_hash"] == w.expected_hash(job) and len(verdict["input_hash"]) == 64
    assert verdict["scope_check"] is None and verdict["offer"] is None
    assert verdict["revision"] == 1 and verdict["created_at"] == verdict["updated_at"]

    meta = w.table.get(*keys.verdict_by_input(verdict["input_hash"]))
    assert meta["job_id"] == job["job_id"] and meta["probabilities"] == verdict["probabilities"]
    assert meta["passed_cutoff"] is False and meta["model"] == JEV_MODEL
    operations = transaction_with(w, f"JOB#{job['job_id']}", "TRIAGE")
    assert {(op.item["pk"], op.item["sk"]) for op in operations if isinstance(op, Put)} == {
        (f"JOB#{job['job_id']}", "TRIAGE"), (f"VERDICT#{verdict['input_hash']}", "META")}

    stored = w.job(job["job_id"])
    assert stored["triage_status"] == "complete" and stored["offer_id"] is None
    assert w.outbox(stored)["status"] == "done"
    assert offer_for_job(w.table, job["job_id"]) is None
    assert_only_receipt_writes(w, job, "triage.json")
    receipt = w.s3.json(f"runs/lab-questions/{job['job_id']}/triage.json")
    assert receipt["probabilities"] == verdict["probabilities"] and receipt["input_hash"] == verdict["input_hash"]
    assert not {"pk", "sk", "revision", "updated_at", "receipt_key", "receipt_sha256"} & set(receipt)
    assert digest(receipt) == verdict["receipt_sha256"] and verdict["receipt_key"] == f"runs/lab-questions/{job['job_id']}/triage.json"
    assert_secret_absent(w)


def test_payload_carries_question_context_answer_excerpts_passages_limitations_and_hint_only():
    w = World()
    context = [{"role": "user", "text": "Earlier turn."}, {"role": "assistant", "text": "Earlier reply."}]
    job = w.finished_job(hint=NEW_HINT, context=context, limitations=["Only two cohorts were read."])
    fake = jev(jev_response("answer_only"))

    w.triage(job, fake)

    payload, secret = fake.calls[0]
    assert secret == SECRET and SECRET.encode() not in payload
    request = json.loads(payload)
    assert request["model"] == JEV_MODEL and set(request["questions"]) == {"route"}
    state = payload_state(payload)
    assert state["question"] == QUESTION and state["answer"] == ANSWER_TEXT
    assert state["context"] == context and state["limitations"] == ["Only two cohorts were read."]
    assert state["hints"] == NEW_HINT and state["dropped"] == []
    packet = w.packet(job)
    documents = {d["key"]: d for d in packet["documents"]}
    assert {item["key"] for item in state["evidence_excerpts"]} == {k for k, d in documents.items() if d["doc_type"] == "note"}
    assert {item["key"] for item in state["existing_passages"]} == {OVERVIEW_KEY}
    assert all(item["doc_type"] == "overview" and "kind" not in item for item in state["existing_passages"])
    assert all(item["kind"] in {"results", "methods", "limitations"} for item in state["evidence_excerpts"])
    section_texts = {s["text"] for d in packet["documents"] for s in d["sections"]}
    assert all(item["text"] in section_texts for item in state["evidence_excerpts"] + state["existing_passages"])
    for forbidden in (job["job_id"], job["member_id"], job["session_id"], "m1", "AIDA"):
        assert forbidden not in json.dumps(state, ensure_ascii=False)


def test_korean_question_travels_verbatim():
    w = World()
    job = w.finished_job(question=KOREAN_QUESTION, answer="코호트에서 지역 유전 안정성은 유지되었다 (n = 120).", hint=NO_HINT)
    assert w.packet(job)["evidence_state"] == "sufficient"
    fake = jev(jev_response("answer_only"))

    verdict = w.triage(job, fake)

    assert payload_state(fake.calls[0][0])["question"] == KOREAN_QUESTION
    assert verdict["status"] == "complete"


# ---------------------------------------------------------------------------------------------
# Cutoff, candidate status and offers
# ---------------------------------------------------------------------------------------------

def test_0_9899_does_not_pass_the_cutoff_and_stays_unconfirmed_without_an_offer():
    w = World()
    job = w.finished_job(hint=NEW_HINT)
    fake = jev(jev_response("review_candidate", probabilities=NEAR, confidence=0.99))

    verdict = w.triage(job, fake)

    assert verdict["status"] == "complete" and verdict["choice"] == "review_candidate"
    assert verdict["probabilities"]["review_candidate"] == 0.9899 and verdict["passed_cutoff"] is False
    assert verdict["confidence"] == 0.99  # recorded, never compared with the cutoff
    assert verdict["candidate_status"] == "unconfirmed_candidate"
    assert verdict["scope_check"] is None and verdict["offer"] is None
    assert offer_for_job(w.table, job["job_id"]) is None
    stored = w.job(job["job_id"])
    assert stored["triage_status"] == "complete" and stored["offer_id"] is None
    assert_only_receipt_writes(w, job, "triage.json")


def test_candidate_status_follows_the_choice_for_answer_only_and_needs_lookup():
    w = World()
    first = w.finished_job(hint=NEW_HINT, request_id="req-a")
    second = w.finished_job(hint=NEW_HINT, request_id="req-b", who="m2", answer="The evidence read does not settle it.")
    lookup = {"answer_only": 0.1, "needs_lookup": 0.85, "review_candidate": 0.05}

    a = w.triage(first, jev(jev_response("answer_only")))
    b = w.triage(second, jev(jev_response("needs_lookup", probabilities=lookup)))

    assert a["candidate_status"] == "answer_only" and b["candidate_status"] == "needs_lookup"
    assert offer_for_job(w.table, first["job_id"]) is None and offer_for_job(w.table, second["job_id"]) is None
    assert lab_triage.candidate_status("review_candidate", True, NEW_HINT) == "review_candidate"
    assert lab_triage.candidate_status("review_candidate", True, {"kind": "correction", "target_keys": [], "note": "  "}) == "unconfirmed_candidate"
    assert lab_triage.candidate_status("review_candidate", False, NEW_HINT) == "unconfirmed_candidate"
    assert lab_triage.candidate_status("review_candidate", True, NO_HINT) == "unconfirmed_candidate"


def test_0_99_with_a_concrete_hint_and_no_scope_hit_issues_a_new_synthesis_offer():
    w = World()
    job = w.finished_job(hint=NEW_HINT)
    fake = jev(jev_response("review_candidate", probabilities=PASS, confidence=0.93))

    verdict = w.triage(job, fake)

    assert verdict["passed_cutoff"] is True and verdict["candidate_status"] == "review_candidate"
    assert verdict["probabilities"]["review_candidate"] == 0.99
    check = verdict["scope_check"]
    assert check["index_etag"] == w.s3.etag(INDEX_KEY)
    assert [q["doc_type"] for q in check["queries"]] == ["concept", "overview", "question"]
    assert all(q["query"] == QUESTION for q in check["queries"])
    assert check["pages_checked"] == [OVERVIEW_KEY]  # the overview is found but scores below the match threshold
    assert check["queries"][1]["hits"][0]["key"] == OVERVIEW_KEY and check["queries"][1]["hits"][0]["score"] < SCOPE_MATCH_SCORE
    assert check["matched"] == [] and check["targets"] == [] and check["rejected_targets"] == []
    assert check["kind"] == "new_synthesis" and check["scope_match_score"] == SCOPE_MATCH_SCORE
    assert verdict["offer"] == {"kind": "new_synthesis", "targets": []}

    offer = offer_for_job(w.table, job["job_id"])
    assert offer["kind"] == "new_synthesis" and offer["targets"] == [] and offer["status"] == "offered"
    assert offer["message"].startswith(OFFER_TEMPLATES["new_synthesis"])
    assert offer["message"].endswith(OFFER_TEMPLATES["consent_note"])
    assert offer["scope_check"] == check and offer["verdict"]["input_hash"] == verdict["input_hash"]
    assert offer["verdict"]["probabilities"] == verdict["probabilities"] and offer["policy_revision"] == POLICY_REVISION
    stored = w.job(job["job_id"])
    assert stored["offer_id"] == offer["offer_id"] and stored["triage_status"] == "complete"
    assert stored["status"] == "completed" and w.outbox(stored)["status"] == "done"
    assert_only_receipt_writes(w, job, "triage.json", f"offer-{offer['offer_id']}.json")
    assert_secret_absent(w)


def test_0_99_with_an_existing_target_issues_a_supplement_offer_and_drops_unusable_targets():
    w = World()
    hint = {"kind": "supplement_existing",
            "target_keys": [OVERVIEW_KEY, "wiki/drafts/secret.md", "wiki/concepts/missing.md", "papers/x/paper.pdf", OVERVIEW_KEY],
            "note": "The overview lacks the 300-family regional difference."}
    job = w.finished_job(hint=hint)
    fake = jev(jev_response("review_candidate", probabilities=HIGH, confidence=0.97))

    verdict = w.triage(job, fake)

    check = verdict["scope_check"]
    assert check["targets"] == [OVERVIEW_KEY] and check["kind"] == "supplement_existing"
    assert check["rejected_targets"] == [
        {"key": "wiki/drafts/secret.md", "reason": "invalid_key"},
        {"key": "wiki/concepts/missing.md", "reason": "not_indexed"},
        {"key": "papers/x/paper.pdf", "reason": "invalid_key"},
    ]
    assert check["hint_kind"] == "supplement_existing"
    offer = offer_for_job(w.table, job["job_id"])
    assert offer["kind"] == "supplement_existing" and offer["targets"] == [OVERVIEW_KEY]
    assert offer["message"].splitlines() == [OFFER_TEMPLATES["supplement_existing"],
                                              f"{OFFER_TEMPLATES['existing_targets_label']} {OVERVIEW_KEY}",
                                              OFFER_TEMPLATES["consent_note"]]
    assert w.job(job["job_id"])["offer_id"] == offer["offer_id"]


def test_a_scope_hit_at_or_above_the_match_score_becomes_a_supplement_target():
    w = World()
    job = w.finished_job(hint=NEW_HINT)
    fake = jev(jev_response("review_candidate", probabilities=PASS))

    verdict = w.triage(job, fake, policy=TriagePolicy(scope_match_score=1.0))

    check = verdict["scope_check"]
    assert check["matched"] == [{"key": OVERVIEW_KEY, "doc_type": "overview", "score": check["queries"][1]["hits"][0]["score"]}]
    assert check["targets"] == [OVERVIEW_KEY] and check["kind"] == "supplement_existing" and check["scope_match_score"] == 1.0
    offer = offer_for_job(w.table, job["job_id"])
    assert offer["kind"] == "supplement_existing" and offer["targets"] == [OVERVIEW_KEY]


def test_0_99_with_hint_none_is_an_unconfirmed_candidate_without_an_offer():
    w = World()
    job = w.finished_job(hint=NO_HINT)
    fake = jev(jev_response("review_candidate", probabilities=HIGH, confidence=0.95))

    verdict = w.triage(job, fake)

    assert verdict["passed_cutoff"] is True and verdict["candidate_status"] == "unconfirmed_candidate"
    assert verdict["scope_check"] is None and verdict["offer"] is None
    assert offer_for_job(w.table, job["job_id"]) is None
    assert w.table.rows("OFFER#") == []
    stored = w.job(job["job_id"])
    assert stored["offer_id"] is None and stored["triage_status"] == "complete"
    assert_only_receipt_writes(w, job, "triage.json")


# ---------------------------------------------------------------------------------------------
# Failure: unavailable, never retried
# ---------------------------------------------------------------------------------------------

def test_http_429_records_unavailable_once_and_leaves_the_answer_untouched():
    w = World()
    job = w.finished_job(hint=NEW_HINT)
    answer_key = f"runs/lab-questions/{job['job_id']}/answer.json"
    answer_bytes = w.s3.objects[answer_key]
    fake = jev(JevError("http_429", http_status=429), jev_response("review_candidate", probabilities=PASS))

    verdict = w.triage(job, fake)

    assert len(fake.calls) == 1
    assert verdict["status"] == "unavailable" and verdict["error_code"] == "http_429"
    assert verdict["reason"] == "jev_error:http_429" and verdict["probabilities"] is None
    assert verdict["candidate_status"] == "needs_lookup" and verdict["jev_called"] is True
    assert verdict["usage"] is None and verdict["usd_micros"] == 0
    stored = w.job(job["job_id"])
    assert stored["triage_status"] == "unavailable" and stored["status"] == "completed"
    assert stored["receipt_key"] == answer_key and w.s3.objects[answer_key] == answer_bytes
    assert w.outbox(stored)["status"] == "done" and stored["offer_id"] is None
    assert w.table.get(*keys.verdict_by_input(verdict["input_hash"])) is None
    assert_only_receipt_writes(w, job, "triage.json")

    again = w.triage(job, fake)  # a redelivery: no second call, same verdict
    assert again == verdict and len(fake.calls) == 1 and len(fake.responses) == 1
    assert_only_receipt_writes(w, job, "triage.json")
    assert_secret_absent(w)


def test_secret_unavailable_and_invalid_response_are_unavailable_with_their_codes():
    w = World()
    first = w.finished_job(hint=NEW_HINT, request_id="req-s")
    second = w.finished_job(hint=NEW_HINT, request_id="req-i", who="m2")
    never = jev()

    def failing_reader():
        raise JevError("secret_unavailable")

    a = w.triage(first, never, secret_reader=failing_reader)
    invalid = jev(b'{"model": "jev-1.13.0", "answers": {}}')
    b = w.triage(second, invalid)

    assert never.calls == [] and a["status"] == "unavailable" and a["error_code"] == "secret_unavailable"
    assert len(invalid.calls) == 1 and b["status"] == "unavailable" and b["error_code"] == "invalid_response"
    assert w.job(first["job_id"])["triage_status"] == "unavailable"
    assert w.job(second["job_id"])["triage_status"] == "unavailable"


def test_an_unexpected_transport_exception_propagates_without_recording_anything():
    w = World()
    job = w.finished_job(hint=NEW_HINT)
    fake = jev(RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        w.triage(job, fake)

    assert lab_jobs.read_verdict(w.table, job["job_id"]) is None
    assert w.job(job["job_id"])["triage_status"] == "pending"
    assert w.receipt_writes() == []


# ---------------------------------------------------------------------------------------------
# Reuse and idempotency
# ---------------------------------------------------------------------------------------------

def test_identical_input_on_another_job_reuses_the_verdict_without_a_call_and_still_offers():
    w = World()
    first = w.finished_job(hint=NEW_HINT, request_id="req-1", who="m1")
    second = w.finished_job(hint=NEW_HINT, request_id="req-2", who="m2")
    fake = jev(jev_response("review_candidate", probabilities=PASS, confidence=0.9))

    original = w.triage(first, fake)
    reused = w.triage(second, fake)

    assert len(fake.calls) == 1 and fake.responses == []
    assert original["input_hash"] == reused["input_hash"] == w.expected_hash(second)
    assert reused["status"] == "reused" and reused["reused_from_job_id"] == first["job_id"]
    assert reused["probabilities"] == original["probabilities"] and reused["confidence"] == 0.9
    assert reused["choice"] == "review_candidate" and reused["passed_cutoff"] is True
    assert reused["candidate_status"] == "review_candidate" and reused["jev_called"] is False
    assert reused["usage"] is None and reused["usd_micros"] == 0 and reused["error_code"] is None
    assert reused["job_id"] == second["job_id"] and reused["member_id"] == "m2"
    assert len(w.table.rows("VERDICT#")) == 1 and w.table.get(*keys.verdict_by_input(reused["input_hash"]))["job_id"] == first["job_id"]
    offers = {job["job_id"]: offer_for_job(w.table, job["job_id"]) for job in (first, second)}
    assert offers[first["job_id"]]["offer_id"] != offers[second["job_id"]]["offer_id"]
    assert all(offer["kind"] == "new_synthesis" for offer in offers.values())
    assert w.job(second["job_id"])["triage_status"] == "complete"
    assert w.job(second["job_id"])["offer_id"] == offers[second["job_id"]]["offer_id"]
    assert w.s3.json(f"runs/lab-questions/{second['job_id']}/triage.json")["status"] == "reused"


@needs_offer_suppression
def test_a_declined_offer_is_not_reissued_for_the_same_question_in_the_same_session():
    w = World()
    first = w.finished_job(hint=NEW_HINT, request_id="req-1")
    fake = jev(jev_response("review_candidate", probabilities=PASS, confidence=0.9))
    original = w.triage(first, fake)
    offer = offer_for_job(w.table, first["job_id"])
    assert original["candidate_status"] == "review_candidate" and offer["status"] == "offered"
    declined = lab_offers.respond(w.table, w.receipts, Member("m1"),
                                  {"request_id": "dec-1", "offer_id": offer["offer_id"], "revision": offer["revision"],
                                   "hash": offer["hash"], "decision": "decline"}, NOW)
    assert declined["status"] == "declined"
    second = w.finished_job(hint=NEW_HINT, request_id="req-2", session_id=first["session_id"])
    assert second["session_id"] == first["session_id"] and second["turn"] == 2
    writes = len(w.receipt_writes())

    verdict = w.triage(second, fake)

    assert len(fake.calls) == 1 and fake.responses == []   # the verdict was reused, Jev was not asked again
    assert verdict["status"] == "reused" and verdict["reused_from_job_id"] == first["job_id"]
    assert verdict["candidate_status"] == "review_candidate"   # the professor still sees the candidate
    assert verdict["offer"] is None and verdict["offer_suppressed"] == {"duplicate_of": offer["offer_id"]}
    stored = w.table.get(*keys.verdict(second["job_id"]))
    assert stored["offer"] is None and stored["offer_suppressed"] == {"duplicate_of": offer["offer_id"]}
    assert stored["scope_check"] is not None and stored["scope_check"]["kind"] == "new_synthesis"
    assert offer_for_job(w.table, second["job_id"]) is None and len(w.table.rows("OFFER#")) == 1
    job = w.job(second["job_id"])
    assert job["triage_status"] == "complete" and job["offer_id"] is None and w.outbox(job)["status"] == "done"
    assert [k for k, _ in w.receipt_writes()[writes:]] == [f"runs/lab-questions/{second['job_id']}/triage.json"]
    assert w.table.get(*keys.offer(offer["offer_id"]))["status"] == "declined"

    again = w.triage(second, jev())   # a redelivery finds the suppression recorded and issues nothing
    assert again == verdict and len(w.table.rows("OFFER#")) == 1


def test_a_different_answer_text_changes_the_hash_and_calls_jev_again():
    w = World()
    first = w.finished_job(request_id="req-1", who="m1")
    second = w.finished_job(request_id="req-2", who="m2", answer=ANSWER_TEXT + " Ancestry was self-reported.")
    fake = jev(jev_response("answer_only"), jev_response("answer_only"))

    a = w.triage(first, fake)
    b = w.triage(second, fake)

    assert len(fake.calls) == 2 and a["input_hash"] != b["input_hash"]
    assert a["status"] == b["status"] == "complete" and len(w.table.rows("VERDICT#")) == 2


def test_redelivery_returns_the_stored_verdict_and_writes_nothing_new():
    w = World()
    job = w.finished_job(hint=NEW_HINT)
    fake = jev(jev_response("review_candidate", probabilities=PASS))

    verdict = w.triage(job, fake)
    transactions, writes = len(w.transactions()), len(w.receipt_writes())
    again = w.triage(job, jev())

    assert again == verdict and len(fake.calls) == 1
    assert len(w.transactions()) == transactions and len(w.receipt_writes()) == writes
    assert len(w.table.rows("OFFER#")) == 1


def test_a_receipt_left_by_a_crashed_attempt_is_recovered_without_calling_jev_again():
    w = World()
    job = w.finished_job(hint=NO_HINT)
    verdict = w.triage(job, jev(jev_response("answer_only")))
    # Simulate the crash window: the receipt exists, the verdict transaction never landed.
    del w.table.items[keys.verdict(job["job_id"])]
    del w.table.items[keys.verdict_by_input(verdict["input_hash"])]
    w.table.items[keys.job(job["job_id"])]["triage_status"] = "pending"
    writes = len(w.receipt_writes())

    recovered = w.triage(job, jev())

    assert recovered == verdict and len(w.receipt_writes()) == writes
    assert w.table.get(*keys.verdict_by_input(verdict["input_hash"]))["job_id"] == job["job_id"]
    assert w.job(job["job_id"])["triage_status"] == "complete"


def test_triage_status_is_rewritten_after_contention_on_the_job_revision():
    w = World()
    job = w.finished_job(hint=NO_HINT)
    original_transact = w.table.transact
    state = {"bumped": False}

    def contended(operations):
        # Another writer moves the job between the re-read and the triage_status update, once.
        if not state["bumped"] and any(not isinstance(op, Put) and op.sk == "META" and op.pk.startswith("JOB#") for op in operations):
            state["bumped"] = True
            current = w.table.items[keys.job(job["job_id"])]
            current["revision"] += 1
            raise ConditionFailed("revision mismatch")
        return original_transact(operations)

    w.table.transact = contended
    verdict = w.triage(job, jev(jev_response("answer_only")))

    assert verdict["status"] == "complete" and state["bumped"] is True
    assert w.job(job["job_id"])["triage_status"] == "complete"


# ---------------------------------------------------------------------------------------------
# Inputs and boundaries
# ---------------------------------------------------------------------------------------------

def test_jobs_that_are_not_finished_answer_jobs_are_refused():
    w = World()
    running = intake(w.table, w.receipts, Member("m1"), {"request_id": "req-r", "question": QUESTION}, NOW)
    running = queue(w.table, running["job_id"], period="2026-09", cap=ANSWER_JOB_CAP_MICROS, now=NOW)
    running = claim(w.table, running["job_id"], running["outbox_id"], LEASE_SECONDS, NOW)
    fake = jev()

    with pytest.raises(InvalidTransition):
        w.triage(running, fake)
    with pytest.raises(NotFound):
        w.triage({"job_id": "missing"}, fake)
    with pytest.raises(ValueError):
        w.triage({"job_id": "../x"}, fake)
    with pytest.raises(ValueError):
        w.triage("job", fake)
    research = copy.deepcopy(running)
    research.update(pk="JOB#research-1", job_id="research-1", kind="research", status="completed")
    w.table.put({**research, "revision": 1})
    with pytest.raises(ValueError):
        w.triage(research, fake)
    with pytest.raises(ValueError):
        w.triage(running, fake, now=datetime(2026, 9, 21, 9, 0))
    with pytest.raises(ValueError):
        w.triage(running, fake, policy={"scope_match_score": 1})
    assert fake.calls == [] and w.receipt_writes() == []


def test_triage_policy_validates_its_fields():
    assert TriagePolicy().policy_revision == POLICY_REVISION and TriagePolicy().scope_match_score == SCOPE_MATCH_SCORE
    with pytest.raises(ValueError):
        TriagePolicy(policy_revision="")
    with pytest.raises(ValueError):
        TriagePolicy(scope_match_score=True)
    with pytest.raises(ValueError):
        TriagePolicy(scope_search_limit=0)


def test_input_hash_covers_exactly_the_six_fields():
    assert lab_triage.input_hash("q", "c", "a", "e", "p", "m") == digest(
        {"question": "q", "context_hash": "c", "answer_sha256": "a", "evidence_sha256": "e", "policy_revision": "p", "model": "m"})
    assert lab_triage.input_hash("q", "c", "a", "e", "p", "m") != lab_triage.input_hash("q", "c", "a", "e", "p2", "m")


def test_scope_check_tolerates_a_bare_connection_and_a_hostile_question():
    w = World()
    check = lab_triage.scope_check(w.index[0], "regional inheritance\x00 stable", NEW_HINT)
    assert check["index_etag"] is None and "\x00" not in check["question"]
    assert [q["doc_type"] for q in check["queries"]] == ["concept", "overview", "question"]
    assert check["pages_checked"] == [OVERVIEW_KEY] and check["kind"] == "new_synthesis"
    with pytest.raises(TypeError):
        lab_triage.scope_check("not-an-index", QUESTION, NEW_HINT)


def test_module_imports_stay_inside_the_lab_boundary_and_never_write_the_wiki():
    source = Path(lab_triage.__file__).read_text(encoding="utf-8")
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    forbidden = {"byeori.ingest_lambda", "byeori.aws_store", "byeori.question_agent",
                 "byeori.agent_cache", "mcp", "httpx", "boto3", "logging", "urllib.request"}
    assert not (names & forbidden), names & forbidden
    assert not any(name.startswith(("mcp.", "httpx.", "boto3.", "logging.")) for name in names)
    assert "put_object" not in source and "Delete(" not in source and "delete_" not in source
    assert "print(" not in source
    assert lab_triage.VERDICT_STATUSES == {"complete", "unavailable", "skipped", "reused"}
    assert lab_triage.CANDIDATE_STATUSES == {"answer_only", "needs_lookup", "review_candidate", "unconfirmed_candidate"}
    assert lab_triage.SERVER_NEEDS_LOOKUP_STATES == {"insufficient", "links_only", "truncated_decisive"}
    assert jev_client.JEV_MODEL == JEV_MODEL
