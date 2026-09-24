"""End-to-end flows over the real lab modules and ``lab_fakes`` (docs/LAB-QUESTION-WORKFLOW.md, sections 6-8).

Two paths a claimed question can take, exercised module by module exactly as the handlers call
them, with nothing mocked inside the package. In the first, the model answers with a concrete
``supplement_existing`` hint whose target is an indexed overview, Jev passes the 0.99 cutoff, the
triage issues the supplement offer and the student's accept through the gateway commits exactly
one research execution behind the ``JOB#{parent}/EXECUTION`` guard. In the second, the wiki's only
page is a link-list hub, so the packet is ``links_only``: the server records ``needs_lookup``
without calling Jev and nothing is offered, whatever the model's hint said. ``lab_lambda`` has its
own tests; the Lambda wiring plays no part here.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from byeori import lab_answer, lab_gateway, lab_jobs, lab_offers, lab_triage
from byeori.lab_gateway import GatewayDeps
from byeori.lab_jobs import Member
from byeori.lab_policy import LEASE_SECONDS, OFFER_TEMPLATES, POLICY_REVISION, RESEARCH_PROFILE
from byeori.lab_store import ReceiptWriter, keys
from lab_fakes import (
    FakeConverse,
    FakeJev,
    FakeSqs,
    MemoryTable,
    iam_event,
    index_connection,
    jev_response,
    member,
    source_note,
    tool_use,
    wiki_with_index,
)

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
INDEX_KEY = "index/wiki-index-v2.sqlite3"
ACCOUNT = "123456789012"
ANSWER_URL = "https://sqs.ap-northeast-2.amazonaws.com/123456789012/byeori-lab-answer"
SECRET = "sk-live-flow-9f8e7d6c5b4a-SECRET-VALUE"
MODEL = "global.anthropic.claude-opus-5"
QUESTION = "Was regional inheritance stable in the cohort?"
RECEIPTS = "runs/lab-questions/"

OVERVIEW_KEY = "wiki/overviews/asd-ndd/regional-inheritance.md"
OVERVIEW = ("---\ntitle: Regional inheritance overview\n---\n\n# Regional inheritance overview\n\n"
            "## Synthesis\n\nAcross cohorts, regional inheritance was stable in most families; one study of 120 families "
            "reported p = 0.01.\n\n## Open questions\n\nWhether stability holds in larger cohorts is untested.\n")
PAGES = {
    "wiki/sources/paper-one.md": source_note(),
    "wiki/sources/paper-two.md": source_note("Paper two", stem="paper-two",
                                             results="Inheritance patterns differed by region in 300 families.",
                                             limitations="Ancestry was self-reported."),
    OVERVIEW_KEY: OVERVIEW,
}
SUPPLEMENT_HINT = {"kind": "supplement_existing", "target_keys": [OVERVIEW_KEY],
                   "note": "The overview lacks the 300-family regional difference."}
SUBMIT = {
    "answer": "Regional inheritance was stable in the cohort (n = 120, p = 0.01); a 300-family cohort differed by region.",
    "citations": [{"key": "wiki/sources/paper-one.md", "section": "Results"},
                  {"key": "wiki/sources/paper-two.md", "section": "Results"}],
    "limitations": ["Single-site cohort of 120 families."],
    "evidence_state": "sufficient",
    "unresolved_items": [],
    "maintenance_hint": SUPPLEMENT_HINT,
}
HIGH = {"answer_only": 0.002, "needs_lookup": 0.003, "review_candidate": 0.995}

# A hub page: nothing but a link list under the heading evidence_packet treats as links.
HUB_KEY = "wiki/concepts/regional-inheritance-hub.md"
HUB = ("---\ntitle: Regional inheritance hub\n---\n\n# Regional inheritance hub\n\n### Linked pages\n\n"
       "- [[sources/paper-one]]\n- [[sources/paper-two]]\n- [[overviews/asd-ndd/regional-inheritance]]\n")
HUB_SUBMIT = {
    "answer": "The wiki lists pages about regional inheritance but the packet holds no findings to quote.",
    "citations": [{"key": HUB_KEY, "section": "Linked pages"}],
    "limitations": ["Only a link list was retrieved."],
    "evidence_state": "partial",
    "unresolved_items": ["Read the linked source notes."],
    # Concrete on its face: the server's own evidence check must still keep it from Jev and from an offer.
    "maintenance_hint": {"kind": "new_synthesis", "target_keys": [], "note": "No synthesis compares the cohorts."},
}


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Record the jittered pauses ``lab_jobs`` takes between transaction attempts instead of sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(lab_jobs, "_sleep", recorded.append)
    return recorded


class World:
    """One member, a control table, a wiki bucket with its index, and the fakes the modules are handed."""

    def __init__(self, pages: dict[str, str]):
        self.table = MemoryTable()
        self.s3 = wiki_with_index(pages, INDEX_KEY)
        self.receipts = ReceiptWriter(self.s3, "bucket")
        self.profile = member(self.table, "m1")
        self.index = (index_connection(self.s3.objects[INDEX_KEY]), self.s3.etag(INDEX_KEY))
        self.sqs = FakeSqs()
        self.fixture_objects = set(self.s3.objects)
        self.gateway = GatewayDeps(
            table=self.table, receipts=self.receipts, s3=self.s3, bucket="bucket", account_id=ACCOUNT,
            answer_queue_url=ANSWER_URL,
            queue_sender=lambda url, body: self.sqs.send_message(QueueUrl=url, MessageBody=json.dumps(body)),
            index=self.index, now=lambda: NOW, policy_revision=POLICY_REVISION,
        )

    # the gateway's and the worker's first steps ---------------------------------------------
    def claimed_job(self, question: str = QUESTION, request_id: str = "req-1") -> dict:
        job = lab_jobs.intake(self.table, self.receipts, Member("m1"), {"request_id": request_id, "question": question}, NOW)
        assert job["status"] == "received"
        job = lab_jobs.queue(self.table, job["job_id"], now=NOW)
        assert job["status"] == "queued" and job["outbox_id"]
        job = lab_jobs.claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, NOW)
        assert job["status"] == "running" and job["attempt"] == 1
        return job

    def answer(self, job: dict, model: FakeConverse) -> dict:
        return lab_answer.answer_job(job, table=self.table, receipts=self.receipts, s3=self.s3, bucket="bucket",
                                     index=self.index, model=model, model_id=MODEL, reasoning="medium", now=NOW,
                                     remaining_ms=850_000)

    def triage(self, job_id: str, jev: FakeJev) -> dict:
        return lab_triage.triage_job({"job_id": job_id}, table=self.table, receipts=self.receipts, s3=self.s3,
                                     bucket="bucket", index=self.index, jev_post=jev, secret_reader=lambda: SECRET, now=NOW)

    def call(self, action: str, body: dict) -> tuple[int, dict]:
        event = iam_event(action, body, user_id=self.profile["principal_id"], user_arn=self.profile["principal_arn"])
        response = lab_gateway.handle(event, deps=self.gateway)
        payload = json.loads(response["body"])
        assert payload["ok"] is (response["statusCode"] == 200)
        return response["statusCode"], payload

    # records ----------------------------------------------------------------------------------
    def job(self, job_id: str) -> dict:
        return self.table.get(*keys.job(job_id))

    def research_jobs(self) -> list[dict]:
        return [row for row in self.table.rows("JOB#") if row["sk"] == "META" and row.get("kind") == "research"]

    def assert_only_receipts_written(self) -> None:
        """Every S3 write is a conditional receipt put; the wiki pages, the index and the table hold no secret."""
        assert self.s3.writes, "the flow wrote nothing"
        assert all(key.startswith(RECEIPTS) for key, _ in self.s3.writes)
        assert all(conditions == {"IfNoneMatch": "*"} for _, conditions in self.s3.writes)
        assert all(key.startswith(RECEIPTS) for key in set(self.s3.objects) - self.fixture_objects)
        assert all(len(self.s3.versions[key]) == 1 for key in self.fixture_objects)   # no page or index rewritten
        assert all(SECRET not in json.dumps(item, default=str) for item in self.table.items.values())
        assert all(SECRET not in body.decode("utf-8", "ignore") for body in self.s3.objects.values())


# ---------------------------------------------------------------------------------------------
# (a) supplement offer: question -> answer -> triage -> offer -> accept -> one research execution
# ---------------------------------------------------------------------------------------------

def test_supplement_offer_flow_ends_in_exactly_one_guarded_research_execution():
    w = World(PAGES)
    model = FakeConverse([tool_use("submit_answer", SUBMIT)])
    jev = FakeJev([jev_response("review_candidate", probabilities=HIGH, confidence=0.97)])
    job = w.claimed_job()
    job_id = job["job_id"]

    # Answer: one model call over a sufficient packet, closed with the triage outbox row.
    result = w.answer(job, model)
    assert result["status"] == "completed" and result["recovered"] is False
    assert result["calls"] == 1 and model.sent == 1 and model.requests[0]["modelId"] == MODEL
    answer = result["answer"]
    assert answer["packet_evidence_state"] == "sufficient" and answer["evidence_state"] == "sufficient"
    assert answer["maintenance_hint"] == SUPPLEMENT_HINT and answer["hold_reason"] is None
    assert {c["key"] for c in answer["citations"] if c["verified"]} == {"wiki/sources/paper-one.md", "wiki/sources/paper-two.md"}
    stored = w.job(job_id)
    assert stored["status"] == "completed" and stored["triage_status"] == "pending" and stored["triage_outbox_id"]
    assert stored["receipt_key"] == f"{RECEIPTS}{job_id}/answer.json"
    assert [row["job_id"] for row in lab_jobs.pending_outbox(w.table, "triage")] == [job_id]

    # Triage: Jev once with the secret, the cutoff passed, the hint's target confirmed in the index.
    verdict = w.triage(job_id, jev)
    assert len(jev.calls) == 1 and jev.calls[0][1] == SECRET
    assert verdict["status"] == "complete" and verdict["jev_called"] is True
    assert verdict["choice"] == "review_candidate" and verdict["probabilities"] == HIGH
    assert verdict["passed_cutoff"] is True and verdict["candidate_status"] == "review_candidate"
    assert verdict["scope_check"]["targets"] == [OVERVIEW_KEY] and verdict["scope_check"]["rejected_targets"] == []
    assert verdict["scope_check"]["index_etag"] == w.s3.etag(INDEX_KEY)
    assert verdict["offer"] == {"kind": "supplement_existing", "targets": [OVERVIEW_KEY]}

    # Offer: the supplement proposal the student sees, linked from the job.
    offer = lab_offers.offer_for_job(w.table, job_id)
    assert offer is not None
    assert offer["kind"] == "supplement_existing" and offer["targets"] == [OVERVIEW_KEY] and offer["status"] == "offered"
    assert offer["member_id"] == "m1" and offer["job_id"] == job_id and offer["policy_revision"] == POLICY_REVISION
    assert offer["message"].splitlines() == [OFFER_TEMPLATES["supplement_existing"],
                                             f"{OFFER_TEMPLATES['existing_targets_label']} {OVERVIEW_KEY}",
                                             OFFER_TEMPLATES["consent_note"]]
    stored = w.job(job_id)
    assert stored["triage_status"] == "complete" and stored["offer_id"] == offer["offer_id"]
    assert w.table.get(*keys.outbox(stored["triage_outbox_id"]))["status"] == "done"
    assert lab_jobs.pending_outbox(w.table, "triage") == []
    status, view = w.call("get_byeori_answer", {"job_id": job_id})
    assert status == 200 and view["triage_status"] == "complete" and view["answer"] == SUBMIT["answer"]
    assert view["synthesis_offer"] == lab_offers.offer_view(offer) and "research" not in view
    assert lab_jobs.existing_execution(w.table, job_id) is None

    # Accept through the gateway with the member's IAM identity: one transaction, one execution.
    status, payload = w.call("respond_to_synthesis_offer", {
        "request_id": "resp-1", "offer_id": offer["offer_id"], "revision": offer["revision"], "hash": offer["hash"],
        "decision": "accept",
    })
    assert status == 200 and payload["action"] == "respond_to_synthesis_offer"
    assert payload["offer_id"] == offer["offer_id"] and payload["status"] == "accepted"
    execution_id, approval_id = payload["execution_id"], payload["approval_id"]
    assert execution_id and approval_id

    guard = lab_jobs.existing_execution(w.table, job_id)
    assert guard is not None
    assert guard["execution_id"] == execution_id and guard["approval_id"] == approval_id
    assert guard["parent_job_id"] == job_id and guard["member_id"] == "m1"
    assert (guard["pk"], guard["sk"]) == (f"JOB#{job_id}", "EXECUTION")
    research_jobs = w.research_jobs()
    assert [job["job_id"] for job in research_jobs] == [execution_id]
    research = research_jobs[0]
    assert research["parent_job_id"] == job_id and research["status"] == "queued" and research["approval_id"] == approval_id
    assert research["scope"]["targets"] == [OVERVIEW_KEY] and research["scope"]["new_pages"] == []
    assert research["budget_usd_micros"] == RESEARCH_PROFILE["budget_usd_micros"] and research["reservation_id"]
    assert [row["job_id"] for row in lab_jobs.pending_outbox(w.table, "research")] == [execution_id]
    approval = w.table.get(*keys.approval(approval_id))
    assert approval["kind"] == "student_consent" and approval["approved_by"] == "m1" and approval["execution_id"] == execution_id
    assert approval["proposal_hash"] == offer["hash"] and approval["proposal_revision"] == offer["revision"]
    assert w.table.get(*keys.offer(offer["offer_id"]))["status"] == "accepted"
    assert w.sqs.messages == []  # research messages leave through the relay, never the gateway

    # The student's view links the execution; a second consent adds nothing.
    status, view = w.call("get_byeori_answer", {"job_id": job_id})
    assert view["synthesis_offer"]["status"] == "accepted"
    assert view["research"] == {"execution_id": execution_id, "status": "queued"}
    status, again = w.call("respond_to_synthesis_offer", {
        "request_id": "resp-2", "offer_id": offer["offer_id"], "revision": offer["revision"], "hash": offer["hash"],
        "decision": "accept",
    })
    assert status == 200 and again["execution_id"] == execution_id
    assert [job["job_id"] for job in w.research_jobs()] == [execution_id]
    assert lab_jobs.existing_execution(w.table, job_id) == guard

    # Every S3 write of the whole flow is a receipt under the answer job or its execution.
    w.assert_only_receipts_written()
    written = sorted(key for key, _ in w.s3.writes)
    assert written == sorted([
        f"{RECEIPTS}{job_id}/request.json", f"{RECEIPTS}{job_id}/evidence.json", f"{RECEIPTS}{job_id}/answer.json",
        f"{RECEIPTS}{job_id}/triage.json", f"{RECEIPTS}{job_id}/offer-{offer['offer_id']}.json",
        f"{RECEIPTS}{job_id}/approval-{approval_id}.json", f"{RECEIPTS}{execution_id}/request.json",
    ])


# ---------------------------------------------------------------------------------------------
# (b) links-only wiki: the packet is links_only, triage skips without Jev, nothing is offered
# ---------------------------------------------------------------------------------------------

def test_link_list_hub_yields_links_only_and_triage_skips_without_jev_or_an_offer():
    w = World({HUB_KEY: HUB})
    model = FakeConverse([tool_use("submit_answer", HUB_SUBMIT)])
    jev = FakeJev([])
    job = w.claimed_job()
    job_id = job["job_id"]

    result = w.answer(job, model)

    packet = w.s3.json(result["evidence_key"])
    assert packet["evidence_state"] == "links_only"
    assert [document["key"] for document in packet["documents"]] == [HUB_KEY]
    assert [(section["name"], section["kind"]) for document in packet["documents"] for section in document["sections"]] \
        == [("Linked pages", "links")]
    assert result["status"] == "completed" and result["calls"] == 1   # a links-only packet still gets its one call
    assert result["answer"]["packet_evidence_state"] == "links_only" and result["answer"]["hold_reason"] is None
    assert result["answer"]["maintenance_hint"] == HUB_SUBMIT["maintenance_hint"]
    assert w.job(job_id)["status"] == "completed" and w.job(job_id)["triage_status"] == "pending"

    verdict = w.triage(job_id, jev)

    assert jev.calls == [] and verdict["jev_called"] is False and verdict["payload_bytes"] is None
    assert verdict["status"] == "skipped" and verdict["reason"] == "evidence_links_only"
    assert verdict["evidence_state"] == "links_only" and verdict["candidate_status"] == "needs_lookup"
    assert verdict["probabilities"] is None and verdict["choice"] is None and verdict["passed_cutoff"] is False
    assert verdict["scope_check"] is None and verdict["offer"] is None and verdict["usd_micros"] == 0
    assert lab_offers.offer_for_job(w.table, job_id) is None
    assert w.table.rows("OFFER#") == [] and w.table.rows("APPROVAL#") == [] and w.research_jobs() == []
    assert w.table.get(*keys.verdict_by_input(verdict["input_hash"])) is None   # a skip is never reused
    stored = w.job(job_id)
    assert stored["triage_status"] == "skipped" and stored["offer_id"] is None
    assert w.table.get(*keys.outbox(stored["triage_outbox_id"]))["status"] == "done"
    assert lab_jobs.existing_execution(w.table, job_id) is None
    status, view = w.call("get_byeori_answer", {"job_id": job_id})
    assert status == 200 and view["triage_status"] == "skipped" and "synthesis_offer" not in view
    assert view["evidence_state"] == "partial" and view["answer"] == HUB_SUBMIT["answer"]

    # A redelivery of the triage message finds the verdict and still makes no call.
    assert w.triage(job_id, jev)["status"] == "skipped" and jev.calls == []
    w.assert_only_receipts_written()
    assert sorted(key for key, _ in w.s3.writes) == sorted([
        f"{RECEIPTS}{job_id}/request.json", f"{RECEIPTS}{job_id}/evidence.json", f"{RECEIPTS}{job_id}/answer.json",
        f"{RECEIPTS}{job_id}/triage.json",
    ])
