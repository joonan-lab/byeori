"""Composition root of byeori.lab_lambda: env config, client wiring, dispatch and SQS batch failures."""
from __future__ import annotations

import inspect
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta

import boto3
import pytest
from botocore.exceptions import ClientError

from byeori import lab_lambda
from byeori.evidence_packet import build_packet
from byeori.jev_client import JevError
from byeori.lab_jobs import Member, claim, complete, create_research_job, intake, mark_sent, pending_outbox, queue
from byeori.lab_lambda import Config, Deps, bedrock_config, bedrock_model, build_deps, handler
from byeori.lab_policy import ANSWER_JOB_CAP_MICROS, LEASE_SECONDS, POLICY_REVISION
from byeori.lab_store import ReceiptWriter, keys, receipt_key
from lab_fakes import (
    FakeContext,
    FakeJev,
    FakeSqs,
    FakeSsm,
    MemoryTable,
    iam_event,
    jev_response,
    member,
    source_note,
    sqs_event,
    text_only,
    tool_use,
    wiki_with_index,
)

NOW = datetime(2026, 9, 21, 9, 0, tzinfo=UTC)
ACCOUNT = "123456789012"
REGION = "ap-northeast-2"
ARN = f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:byeori-lab-answer"
FOREIGN_ARN = "arn:aws:lambda:eu-west-1:210987654321:function:byeori-lab-gateway"
QUEUES = "https://sqs.ap-northeast-2.amazonaws.com/123456789012/"
ANSWER_URL, TRIAGE_URL, RESEARCH_URL = QUEUES + "byeori-lab-answer", QUEUES + "byeori-lab-triage", QUEUES + "byeori-lab-research"
INDEX_KEY = "index/wiki-index-v2.sqlite3"
# Every S3 key a lab handler may create: immutable receipts, and the answered questions kept as
# Markdown outside the search index. A scientific page, an original and the index stay closed.
WRITABLE_PREFIXES = ("runs/lab-questions/", "wiki/lab-questions/")
SECRET = "sk-live-9f8e7d6c5b4a-SECRET-VALUE"
JEV_PARAMETER = "/byeori/jev/api-key"
MODEL = "global.anthropic.claude-opus-5"
QUESTION = "Was regional inheritance stable in the cohort?"
ANSWER_TEXT = "Regional inheritance was stable in the cohort (n = 120, p = 0.01); the cohort was small and single-site."
SUBMIT = {
    "answer": ANSWER_TEXT,
    "citations": [{"key": "wiki/sources/paper-one.md", "section": "Results"}],
    "limitations": ["Single-site cohort of 120 families."],
    "evidence_state": "sufficient",
    "unresolved_items": [],
    "maintenance_hint": {"kind": "none", "target_keys": [], "note": ""},
}
USAGE = {"inputTokens": 1200, "outputTokens": 300, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0}
PAGES = {
    "wiki/sources/paper-one.md": source_note(),
    "wiki/sources/paper-two.md": source_note("Paper two", stem="paper-two",
                                             results="Inheritance patterns differed by region in 300 families.",
                                             limitations="Ancestry was self-reported."),
}
ENV = {
    "LAB_TABLE": "byeori-lab-control", "LAB_BUCKET": "bucket", "LAB_INDEX_KEY": INDEX_KEY,
    "LAB_ANSWER_MODEL_ID": MODEL, "LAB_ANSWER_REASONING": "medium", "LAB_JEV_PARAMETER": JEV_PARAMETER,
    "LAB_ANSWER_QUEUE_URL": ANSWER_URL, "LAB_TRIAGE_QUEUE_URL": TRIAGE_URL, "LAB_RESEARCH_QUEUE_URL": RESEARCH_URL,
    "LAB_POLICY_REVISION": POLICY_REVISION,
}
# Nothing a log record may carry: the question, the secret, a principal ARN or a session token.
FORBIDDEN_LOG_TEXT = (QUESTION, ANSWER_TEXT, SECRET, "arn:aws:iam", "X-Amz-Security-Token", "AIDA")


def env_for(kind: str, **overrides) -> dict[str, str]:
    return {**ENV, "LAB_HANDLER": kind, **overrides}


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Record the jittered pauses ``lab_jobs`` takes between transaction attempts instead of sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(lab_lambda.lab_jobs, "_sleep", recorded.append)
    return recorded


class Clock:
    """A ``now()`` callable the test moves explicitly."""

    def __init__(self, start: datetime = NOW):
        self.moment = start

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, seconds: float) -> None:
        self.moment += timedelta(seconds=seconds)


class FakeBedrock:
    """A ``bedrock-runtime`` client whose ``converse`` returns scripted responses or raises."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def converse(self, **request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("FakeBedrock received more calls than scripted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class UntouchableDynamo:
    """A DynamoDB client no handler may call; any method access fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"the control table was touched through {name}")


class Recorder:
    """Stands in for ``boto3.client``: records ``(service, kwargs)`` and hands out the fakes it was given."""

    def __init__(self, **fakes):
        self.fakes = fakes
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, service, **kwargs):
        self.calls.append((service, kwargs))
        return self.fakes[service]

    def services(self) -> list[str]:
        return [service for service, _ in self.calls]


class World:
    """One control table, a wiki bucket with its index, a member and fakes for every client."""

    def __init__(self, tmp_path, *, sqs: FakeSqs | None = None, jev: FakeJev | None = None,
                 bedrock: FakeBedrock | None = None, ssm: FakeSsm | None = None):
        self.tmp_path = tmp_path
        self.table = MemoryTable()
        self.s3 = wiki_with_index(PAGES, INDEX_KEY)
        self.receipts = ReceiptWriter(self.s3, "bucket")
        self.sqs = sqs or FakeSqs()
        self.ssm = ssm or FakeSsm(SECRET, JEV_PARAMETER)
        self.bedrock = bedrock or FakeBedrock()
        self.jev = jev or FakeJev([])
        self.clock = Clock()
        self.context = FakeContext(function_arn=ARN)
        self.profile = member(self.table, "m1")
        self.last_deps: Deps | None = None

    def deps(self, kind: str, **overrides) -> Deps:
        config = Config.from_env(env_for(kind, **overrides))
        self.last_deps = build_deps(config, self.context, table=self.table, receipts=self.receipts, s3=self.s3,
                                    sqs=self.sqs, ssm=self.ssm, bedrock=self.bedrock, jev_post=self.jev,
                                    now=self.clock, index_cache_dir=self.tmp_path)
        return self.last_deps

    def run(self, kind: str, event, **overrides):
        return handler(event, self.context, deps=self.deps(kind, **overrides))

    # jobs -----------------------------------------------------------------------------------
    def queued(self, request_id: str = "req-1", question: str = QUESTION) -> dict:
        self.clock.advance(1)                         # distinct created_at stamps keep outbox order deterministic
        job = intake(self.table, self.receipts, Member("m1"), {"request_id": request_id, "question": question},
                     self.clock.moment)
        return queue(self.table, job["job_id"], period="2026-09", cap=ANSWER_JOB_CAP_MICROS, now=self.clock.moment)

    def running(self, request_id: str = "req-1") -> dict:
        job = self.queued(request_id)
        return claim(self.table, job["job_id"], job["outbox_id"], LEASE_SECONDS, self.clock.moment)

    def finished(self, request_id: str = "req-1") -> dict:
        """What the answer worker leaves behind: evidence.json, answer.json and the completion."""
        job = self.running(request_id)
        job_id = job["job_id"]
        index = (sqlite3.connect(":memory:"), None)
        index[0].deserialize(self.s3.objects[INDEX_KEY])
        packet = build_packet(QUESTION, index=index, s3=self.s3, bucket="bucket")
        index[0].close()
        evidence = self.receipts.put_json(receipt_key(job_id, "evidence.json"), packet)
        record = {
            "job_id": job_id, "kind": "answer", "member_id": "m1", "session_id": job["session_id"], "turn": job["turn"],
            "parent_job_id": None, "question": QUESTION, "standalone_question": QUESTION, "context": [],
            "question_hash": job["question_hash"], "context_hash": job["context_hash"], "request_id": request_id,
            "policy_revision": POLICY_REVISION, "attempt": 1, "model_id": MODEL, "reasoning": None, "thinking": "off",
            "status": "completed", "hold_reason": None, "answer": ANSWER_TEXT,
            "citations": [{"key": "wiki/sources/paper-one.md", "section": "Results", "verified": True}],
            "limitations": SUBMIT["limitations"], "evidence_state": "sufficient", "unresolved_items": [],
            "maintenance_hint": SUBMIT["maintenance_hint"], "packet_evidence_state": packet["evidence_state"],
            "index_etag": packet.get("index_etag"), "queries": [q["query"] for q in packet["queries"]],
            "english_query": None, "lookup": None,
            "documents": [{"key": d["key"], "etag": d["etag"], "version_id": d["version_id"], "sha256": d["sha256"]}
                          for d in packet["documents"]],
            "evidence_key": evidence["key"], "evidence_sha256": evidence["sha256"], "calls": [], "holds": [],
            "usage": USAGE, "usd_micros": 1200, "completed_at": "2026-09-21T09:00:00.000000+00:00",
        }
        receipt = self.receipts.put_json(receipt_key(job_id, "answer.json"), record)
        return complete(self.table, job_id, job["revision"], receipt_key=receipt["key"], evidence_key=evidence["key"],
                        usage=USAGE, usd_micros=1200, status="completed", now=self.clock.moment)

    def research(self, parent: dict, approval_id: str = "appr-1") -> dict:
        return create_research_job(self.table, self.receipts, parent_job=parent, member_id="m1", approval_id=approval_id,
                                   scope={"question": QUESTION, "targets": [], "new_pages": [], "note": None},
                                   budget_usd_micros=5_000_000, now=self.clock.moment)

    def job(self, job_id: str) -> dict:
        return self.table.get(*keys.job(job_id))

    def outbox(self, outbox_id: str) -> dict:
        return self.table.get(*keys.outbox(outbox_id))

    def event(self, action: str, body: dict | None = None, **overrides) -> dict:
        return iam_event(action, body, user_id=self.profile["principal_id"], user_arn=self.profile["principal_arn"],
                         **overrides)


def message(job: dict, kind: str = "answer", outbox_id: str | None = None) -> dict:
    return {"outbox_id": outbox_id or job["outbox_id"], "job_id": job["job_id"], "kind": kind}


def failures(result: dict) -> list[str]:
    assert set(result) == {"batchItemFailures"}
    assert all(set(item) == {"itemIdentifier"} for item in result["batchItemFailures"])
    return [item["itemIdentifier"] for item in result["batchItemFailures"]]


def envelope(response: dict) -> tuple[int, dict]:
    assert set(response) == {"statusCode", "headers", "body"}
    payload = json.loads(response["body"])
    assert "ok" in payload
    return response["statusCode"], payload


def assert_clean_logs(caplog) -> None:
    text = caplog.text
    for forbidden in FORBIDDEN_LOG_TEXT:
        assert forbidden not in text, forbidden


# ---------------------------------------------------------------------------------------------
# Configuration and invocation identity
# ---------------------------------------------------------------------------------------------

def test_config_reads_every_variable_and_applies_the_defaults():
    config = Config.from_env(env_for("answer"))

    assert config.table_name == "byeori-lab-control" and config.bucket == "bucket" and config.handler == "answer"
    assert config.index_key == INDEX_KEY and config.answer_model_id == MODEL and config.answer_reasoning == "medium"
    assert config.jev_parameter == JEV_PARAMETER
    assert (config.answer_queue_url, config.triage_queue_url, config.research_queue_url) == (ANSWER_URL, TRIAGE_URL, RESEARCH_URL)
    assert config.policy_revision == POLICY_REVISION and config.research_consumer_enabled is False

    sparse = Config.from_env({"LAB_TABLE": "t", "LAB_BUCKET": "b", "LAB_HANDLER": "gateway",
                              "LAB_ANSWER_REASONING": "", "LAB_RESEARCH_CONSUMER_ENABLED": "true"})
    assert sparse.index_key == "index/wiki-index-v2.sqlite3" and sparse.answer_reasoning is None
    assert sparse.policy_revision == POLICY_REVISION and sparse.research_consumer_enabled is True
    assert Config.from_env(env_for("outbox", LAB_RESEARCH_CONSUMER_ENABLED="TRUE ")).research_consumer_enabled is True
    assert Config.from_env(env_for("outbox", LAB_RESEARCH_CONSUMER_ENABLED="yes")).research_consumer_enabled is False


@pytest.mark.parametrize("missing", ["LAB_TABLE", "LAB_BUCKET"])
def test_config_names_the_missing_table_or_bucket_variable(missing):
    environment = env_for("gateway")
    del environment[missing]
    with pytest.raises(RuntimeError, match=missing):
        Config.from_env(environment)
    environment[missing] = "   "
    with pytest.raises(RuntimeError, match=missing):
        Config.from_env(environment)


@pytest.mark.parametrize("value", ["", "publisher", "Gateway", None])
def test_config_refuses_an_unknown_handler_kind(value):
    environment = env_for("gateway")
    if value is None:
        del environment["LAB_HANDLER"]
    else:
        environment["LAB_HANDLER"] = value
    with pytest.raises(RuntimeError, match="LAB_HANDLER"):
        Config.from_env(environment)


def test_handler_reads_the_environment_when_invoked_not_when_imported(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LAB_HANDLER", raising=False)
    monkeypatch.setattr(boto3, "client", Recorder())  # must never be reached

    with pytest.raises(RuntimeError, match="LAB_TABLE"):
        handler({}, FakeContext(function_arn=ARN))

    for name, value in env_for("research").items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("LAB_BUCKET")
    with pytest.raises(RuntimeError, match="LAB_BUCKET"):
        handler({}, FakeContext(function_arn=ARN))


def test_account_and_region_come_from_the_context_arn_only(monkeypatch):
    monkeypatch.setenv("AWS_ACCOUNT_ID", "999999999999")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    config = Config.from_env(env_for("gateway"))

    deps = build_deps(config, FakeContext(function_arn=FOREIGN_ARN), table=MemoryTable())
    assert (deps.account_id, deps.region) == ("210987654321", "eu-west-1")

    with pytest.raises(RuntimeError, match="invoked_function_arn"):
        build_deps(config, None)
    with pytest.raises(RuntimeError, match="invoked_function_arn"):
        build_deps(config, FakeContext(function_arn="not-an-arn"))
    with pytest.raises(RuntimeError, match="invoked_function_arn"):
        build_deps(config, FakeContext(function_arn="arn:aws:lambda:ap-northeast-2:not-digits:function:x"))


def test_gateway_compares_the_caller_account_with_the_arn_account(tmp_path):
    w = World(tmp_path)
    w.context = FakeContext(function_arn=FOREIGN_ARN)

    status, payload = envelope(w.run("gateway", w.event("search_wiki", {"query": "inheritance"})))

    assert (status, payload["error"]) == (403, "wrong_account")
    assert len(w.table.transactions) == 2            # only the registry puts of the fixture


# ---------------------------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------------------------

def test_default_clients_come_from_boto3_lazily_and_bedrock_makes_one_attempt(monkeypatch):
    recorder = Recorder(dynamodb=UntouchableDynamo(), s3=object(), sqs=object(), ssm=object(), **{"bedrock-runtime": object()})
    monkeypatch.setattr(boto3, "client", recorder)
    deps = build_deps(Config.from_env(env_for("answer")), FakeContext(function_arn=ARN))

    assert recorder.calls == []                      # nothing is created before it is used
    bedrock = deps.bedrock
    assert bedrock is deps.bedrock is recorder.fakes["bedrock-runtime"]
    assert recorder.services() == ["bedrock-runtime"]
    service, kwargs = recorder.calls[0]
    config = kwargs["config"]
    assert kwargs["region_name"] == REGION
    assert config.retries == {"total_max_attempts": 1}
    assert isinstance(config.read_timeout, int) and 0 < config.read_timeout < 900
    assert isinstance(config.connect_timeout, int) and config.connect_timeout > 0

    assert deps.table.table_name == "byeori-lab-control" and deps.table.client is recorder.fakes["dynamodb"]
    assert deps.receipts.bucket == "bucket" and deps.receipts.s3 is deps.s3 is recorder.fakes["s3"]
    assert deps.sqs is recorder.fakes["sqs"] and deps.ssm is recorder.fakes["ssm"]
    assert sorted(recorder.services()) == ["bedrock-runtime", "dynamodb", "s3", "sqs", "ssm"]
    assert all(call_kwargs["region_name"] == REGION for _, call_kwargs in recorder.calls)

    standalone = bedrock_config()
    assert standalone.retries == {"total_max_attempts": 1} and standalone.read_timeout == config.read_timeout


def test_bedrock_model_calls_converse_once_per_request_and_lets_exceptions_through():
    response = tool_use("submit_answer", SUBMIT)
    refused = ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "Converse")
    client = FakeBedrock([response, refused])
    model = bedrock_model(client)
    request = {"modelId": MODEL, "messages": [{"role": "user", "content": [{"text": "{}"}]}]}

    assert model(request) is response
    with pytest.raises(ClientError):
        model(request)
    assert client.requests == [request, request]


# ---------------------------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------------------------

def test_gateway_returns_the_envelope_queues_the_job_and_sends_the_answer_message(tmp_path, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)

    status, payload = envelope(w.run("gateway", w.event("ask_byeori", {"request_id": "req-1", "question": QUESTION})))

    assert status == 200 and payload["ok"] is True and payload["action"] == "ask_byeori"
    assert payload["status"] == "queued" and payload["delivery"] == "sent"
    job = w.job(payload["job_id"])
    assert job["status"] == "queued" and job["policy_revision"] == POLICY_REVISION
    assert w.sqs.messages == [(ANSWER_URL, {"outbox_id": job["outbox_id"], "job_id": job["job_id"], "kind": "answer"})]
    assert w.outbox(job["outbox_id"])["status"] == "sent"

    status, payload = envelope(w.run("gateway", w.event("get_byeori_answer", {"job_id": job["job_id"]})))
    assert status == 200 and payload["status"] == "queued"
    assert_clean_logs(caplog)
    assert "status=200" in caplog.text


def test_gateway_read_opens_the_index_from_s3_once_and_the_handler_closes_it(tmp_path):
    w = World(tmp_path)
    deps = w.deps("gateway")
    downloads_before = w.s3.reads.count(INDEX_KEY)

    response = handler(w.event("search_wiki", {"query": "regional inheritance"}), w.context, deps=deps)
    status, payload = envelope(response)

    assert status == 200 and payload["results"] and payload["index_etag"] == w.s3.etag(INDEX_KEY)
    assert w.s3.reads.count(INDEX_KEY) == downloads_before + 1
    assert (tmp_path / "wiki-index-v2.sqlite3").exists()
    assert deps.index is None                        # closed at the end of the invocation

    # The connection the invocation used is closed, not merely forgotten.
    deps = w.deps("gateway")
    connection, etag = deps.open_index()
    assert etag == w.s3.etag(INDEX_KEY)
    handler(w.event("wiki_backlinks", {"key": "wiki/sources/paper-one.md"}), w.context, deps=deps)
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT count(*) FROM docs")
    assert w.s3.reads.count(INDEX_KEY) == downloads_before + 1   # the cached file was reused


def test_gateway_through_boto3_fakes_refuses_get_before_touching_the_table(monkeypatch):
    recorder = Recorder(dynamodb=UntouchableDynamo(), s3=object(), sqs=object())
    monkeypatch.setattr(boto3, "client", recorder)
    for name, value in env_for("gateway").items():
        monkeypatch.setenv(name, value)
    event = iam_event("search_wiki", {"query": "x"}, user_id="AIDAM1", user_arn=f"arn:aws:iam::{ACCOUNT}:user/m1",
                      method="GET")

    status, payload = envelope(handler(event, FakeContext(function_arn=ARN)))

    assert (status, payload["error"]) == (405, "method_not_allowed")
    assert set(recorder.services()) <= {"dynamodb", "s3"}


# ---------------------------------------------------------------------------------------------
# Answer worker
# ---------------------------------------------------------------------------------------------

def test_answer_batch_claims_answers_and_completes_the_job(tmp_path, caplog):
    w = World(tmp_path, bedrock=FakeBedrock([tool_use("submit_answer", SUBMIT)]))
    caplog.set_level(logging.DEBUG)
    job = w.queued()

    result = w.run("answer", sqs_event(message(job)))

    assert failures(result) == []
    stored = w.job(job["job_id"])
    assert stored["status"] == "completed" and stored["attempt"] == 1 and stored["triage_status"] == "pending"
    assert stored["receipt_key"] == f"runs/lab-questions/{job['job_id']}/answer.json"
    answer = w.receipts.get_json(stored["receipt_key"])
    assert answer["answer"] == ANSWER_TEXT and answer["model_id"] == MODEL and answer["reasoning"] == "medium"
    request = w.bedrock.requests[0]
    assert request["modelId"] == MODEL and len(w.bedrock.requests) == 1
    assert all(key.startswith(WRITABLE_PREFIXES) for key, _ in w.s3.writes)
    # The answer is kept as Markdown outside the index, and what it cited links back to it.
    page = f"wiki/lab-questions/{stored['period']}/{job['job_id']}.md"
    assert page in w.s3.objects
    body = w.s3.objects[page].decode("utf-8")
    assert "indexed: false" in body and ANSWER_TEXT in body
    hubs = [key for key in w.s3.objects if key.startswith("wiki/lab-questions/by-page/")]
    assert hubs, "a cited page must carry a hub the answer can be reached from"
    assert all(f"[[lab-questions/{stored['period']}/{job['job_id']}" in w.s3.objects[key].decode("utf-8")
               for key in hubs)
    assert w.last_deps.index is None
    assert job["job_id"] in caplog.text and "status=completed" in caplog.text
    assert_clean_logs(caplog)


def test_answer_retries_a_live_lease_and_drops_terminal_or_unknown_jobs(tmp_path, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    leased = w.running("req-1")                       # another worker holds the lease until NOW + 960 s
    finished = w.finished("req-2")                    # duplicate delivery of a completed job
    unknown = {"outbox_id": "0" * 32, "job_id": "does-not-exist", "kind": "answer"}
    other_outbox = message(w.queued("req-3"), outbox_id=finished["outbox_id"])   # outbox row of another job

    result = w.run("answer", sqs_event(message(leased), message(finished), unknown, other_outbox))

    assert failures(result) == ["m1"]
    assert w.bedrock.requests == []
    assert w.job(leased["job_id"])["status"] == "running" and w.job(leased["job_id"])["attempt"] == 1
    assert w.job(finished["job_id"])["status"] == "completed"
    assert w.job(other_outbox["job_id"])["status"] == "queued"
    assert "lease_held" in caplog.text and "invalid_outbox" in caplog.text
    assert_clean_logs(caplog)

    w.clock.advance(LEASE_SECONDS + 1)                # the lease expired: the redelivery is claimed
    w.bedrock.responses.append(tool_use("submit_answer", SUBMIT))
    result = w.run("answer", sqs_event(message(leased)))
    assert failures(result) == []
    assert w.job(leased["job_id"])["status"] == "completed" and w.job(leased["job_id"])["attempt"] == 2


@pytest.mark.parametrize("body", [
    "not json", json.dumps([1, 2]), json.dumps({"job_id": "j", "kind": "answer"}),
    json.dumps({"job_id": "j", "outbox_id": "o", "kind": "triage"}),
    json.dumps({"job_id": 5, "outbox_id": "o", "kind": "answer"}),
])
def test_answer_reports_malformed_or_foreign_kind_messages_as_failures(tmp_path, body, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    event = {"Records": [{"messageId": "bad-1", "body": body, "eventSource": "aws:sqs"}]}

    assert failures(w.run("answer", event)) == ["bad-1"]
    assert w.bedrock.requests == [] and w.table.transactions[2:] == []
    assert "invalid_message" in caplog.text and body not in caplog.text
    assert_clean_logs(caplog)


def test_answer_reports_an_unexpected_exception_and_continues_the_batch(tmp_path, monkeypatch, caplog):
    w = World(tmp_path, bedrock=FakeBedrock([tool_use("submit_answer", SUBMIT)]))
    caplog.set_level(logging.DEBUG)
    first, second = w.queued("req-1"), w.queued("req-2")
    calls = []
    real = lab_lambda.lab_answer.answer_job

    def flaky(job, **options):
        calls.append(job["job_id"])
        if job["job_id"] == first["job_id"]:
            raise RuntimeError(f"index unreadable while answering {QUESTION}")
        return real(job, **options)

    monkeypatch.setattr(lab_lambda.lab_answer, "answer_job", flaky)

    result = w.run("answer", sqs_event(message(first), message(second)))

    assert failures(result) == ["m1"]
    assert calls == [first["job_id"], second["job_id"]]
    assert w.job(first["job_id"])["status"] == "running"      # left to its lease; SQS redelivers
    assert w.job(second["job_id"])["status"] == "completed"
    assert "RuntimeError" in caplog.text
    assert_clean_logs(caplog)


def test_answer_claims_before_opening_the_index_and_dropped_deliveries_never_download_it(tmp_path, monkeypatch):
    w = World(tmp_path, bedrock=FakeBedrock([tool_use("submit_answer", SUBMIT)]))
    fresh = w.queued("req-1")
    leased = w.running("req-2")                       # another worker holds the lease
    finished = w.finished("req-3")                    # duplicate delivery of a completed job
    statuses: list[str] = []
    real_open = lab_lambda.evidence_packet.open_index

    def observed_open(s3, bucket, key, cache_dir):
        statuses.append(w.job(fresh["job_id"])["status"])   # what the fresh job is when the index is opened
        return real_open(s3, bucket, key, cache_dir)

    monkeypatch.setattr(lab_lambda.evidence_packet, "open_index", observed_open)
    downloads_before = w.s3.reads.count(INDEX_KEY)

    # Only refused claims in the batch: the index is never opened, let alone downloaded.
    result = w.run("answer", sqs_event(message(leased), message(finished),
                                       {"outbox_id": "0" * 32, "job_id": "does-not-exist", "kind": "answer"}))
    assert failures(result) == ["m1"]
    assert statuses == [] and w.s3.reads.count(INDEX_KEY) == downloads_before
    assert w.bedrock.requests == []

    # A successful claim comes first; the download follows it once and the job is answered.
    result = w.run("answer", sqs_event(message(fresh)))
    assert failures(result) == []
    assert statuses == ["running"]
    assert w.s3.reads.count(INDEX_KEY) == downloads_before + 1
    assert w.job(fresh["job_id"])["status"] == "completed" and len(w.bedrock.requests) == 1


def test_answer_reports_an_index_failure_after_the_claim_and_leaves_the_job_to_its_lease(tmp_path, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    job = w.queued()
    raw_index = w.s3.objects.pop(INDEX_KEY)          # the shared index cannot be read this invocation

    result = w.run("answer", sqs_event(message(job)))

    assert failures(result) == ["m1"]
    stored = w.job(job["job_id"])
    assert stored["status"] == "running" and stored["attempt"] == 1 and stored["claimed_outbox_id"] == job["outbox_id"]
    assert w.bedrock.requests == [] and w.last_deps.index is None
    assert "action=open_index status=failed" in caplog.text and job["job_id"] in caplog.text
    assert_clean_logs(caplog)
    assert all(key.startswith(WRITABLE_PREFIXES) for key, _ in w.s3.writes)

    # The lease ends, the redelivery is claimed again and answered with the index back in place.
    w.s3._store(INDEX_KEY, raw_index)
    w.clock.advance(LEASE_SECONDS + 1)
    w.bedrock.responses.append(tool_use("submit_answer", SUBMIT))
    assert failures(w.run("answer", sqs_event(message(job)))) == []
    assert w.job(job["job_id"])["status"] == "completed" and w.job(job["job_id"])["attempt"] == 2


def test_answer_handler_refuses_to_claim_without_a_model_id(tmp_path):
    w = World(tmp_path)
    job = w.queued()

    with pytest.raises(RuntimeError, match="LAB_ANSWER_MODEL_ID"):
        w.run("answer", sqs_event(message(job)), LAB_ANSWER_MODEL_ID="")
    assert w.job(job["job_id"])["status"] == "queued"


def test_answer_passes_the_remaining_time_of_the_context(tmp_path):
    w = World(tmp_path)
    w.context = FakeContext(remaining_ms=10_000, function_arn=ARN)   # below lab_answer.MIN_CALL_MS
    job = w.queued()

    assert failures(w.run("answer", sqs_event(message(job)))) == []
    stored = w.job(job["job_id"])
    assert stored["status"] == "partial" and stored["hold_reason"] == "time_budget"
    assert w.bedrock.requests == []


# ---------------------------------------------------------------------------------------------
# Triage worker
# ---------------------------------------------------------------------------------------------

def test_triage_judges_the_finished_job_with_the_ssm_secret_and_never_logs_it(tmp_path, caplog):
    w = World(tmp_path, jev=FakeJev([jev_response("answer_only")]))
    caplog.set_level(logging.DEBUG)
    job = w.finished()
    triage_message = message(job, "triage", outbox_id=job["triage_outbox_id"])

    result = w.run("triage", sqs_event(triage_message))

    assert failures(result) == []
    assert len(w.jev.calls) == 1 and w.jev.calls[0][1] == SECRET
    assert w.ssm.calls == [{"Name": JEV_PARAMETER, "WithDecryption": True}]
    verdict = w.table.get(*keys.verdict(job["job_id"]))
    assert verdict["status"] == "complete" and verdict["choice"] == "answer_only"
    stored = w.job(job["job_id"])
    assert stored["triage_status"] == "complete" and w.outbox(job["triage_outbox_id"])["status"] == "done"
    assert w.last_deps.index is None
    assert job["job_id"] in caplog.text and "status=complete" in caplog.text
    assert_clean_logs(caplog)

    # A redelivery finds the verdict and makes no second call.
    assert failures(w.run("triage", sqs_event(triage_message))) == []
    assert len(w.jev.calls) == 1


def test_triage_drops_running_or_unknown_jobs_records_jev_errors_and_reports_the_rest(tmp_path, caplog):
    w = World(tmp_path, jev=FakeJev([JevError("http_429", http_status=429), RuntimeError(f"socket closed: {SECRET}")]))
    caplog.set_level(logging.DEBUG)
    running = w.running("req-1")
    unavailable = w.finished("req-2")
    broken = w.finished("req-3")
    unknown = {"outbox_id": "0" * 32, "job_id": "does-not-exist", "kind": "triage"}

    result = w.run("triage", sqs_event(message(running, "triage"), unknown,
                                       message(unavailable, "triage", outbox_id=unavailable["triage_outbox_id"]),
                                       message(broken, "triage", outbox_id=broken["triage_outbox_id"])))

    assert failures(result) == ["m4"]
    assert w.job(running["job_id"])["status"] == "running" and w.table.get(*keys.verdict(running["job_id"])) is None
    verdict = w.table.get(*keys.verdict(unavailable["job_id"]))
    assert verdict["status"] == "unavailable" and verdict["error_code"] == "http_429"
    assert w.table.get(*keys.verdict(broken["job_id"])) is None
    assert w.job(broken["job_id"])["triage_status"] == "pending"
    assert "http_429" in caplog.text and "RuntimeError" in caplog.text
    assert_clean_logs(caplog)


def test_triage_handler_requires_the_jev_parameter_name(tmp_path):
    w = World(tmp_path)
    job = w.finished()

    with pytest.raises(RuntimeError, match="LAB_JEV_PARAMETER"):
        w.run("triage", sqs_event(message(job, "triage", outbox_id=job["triage_outbox_id"])), LAB_JEV_PARAMETER="")
    assert w.jev.calls == [] and w.table.get(*keys.verdict(job["job_id"])) is None


def test_triage_reports_a_malformed_message_as_a_failure(tmp_path):
    w = World(tmp_path)
    job = w.finished()
    event = sqs_event(message(job, "answer", outbox_id=job["triage_outbox_id"]))   # wrong kind for this queue

    assert failures(w.run("triage", event)) == ["m1"]
    assert w.jev.calls == [] and w.table.get(*keys.verdict(job["job_id"])) is None


# ---------------------------------------------------------------------------------------------
# Outbox relay
# ---------------------------------------------------------------------------------------------

def test_outbox_relays_pending_rows_marks_sent_after_a_successful_send_and_skips_research(tmp_path):
    w = World(tmp_path, sqs=FakeSqs(fail_urls={TRIAGE_URL}))
    first, second = w.queued("req-1"), w.queued("req-2")
    finished = w.finished("req-3")                    # leaves a pending triage row
    research = w.research(finished)                   # leaves a pending research row

    result = w.run("outbox", {"Records": [{"messageId": "ignored", "body": "{\"kind\": \"research\"}"}]})

    assert result["status"] == "ok" and set(result["kinds"]) == {"answer", "triage", "research"}
    assert result["kinds"]["answer"] == {"status": "relayed", "pending": 2, "sent": 2, "failed": 0, "unmarked": 0}
    assert result["kinds"]["triage"] == {"status": "relayed", "pending": 1, "sent": 0, "failed": 1, "unmarked": 0}
    assert result["kinds"]["research"] == {"status": "skipped", "reason": "research consumer not activated"}
    assert w.sqs.messages == [
        (ANSWER_URL, {"outbox_id": first["outbox_id"], "job_id": first["job_id"], "kind": "answer"}),
        (ANSWER_URL, {"outbox_id": second["outbox_id"], "job_id": second["job_id"], "kind": "answer"}),
    ]
    assert w.outbox(first["outbox_id"])["status"] == "sent" and w.outbox(first["outbox_id"])["attempts"] == 1
    assert w.outbox(second["outbox_id"])["status"] == "sent"
    assert w.outbox(finished["triage_outbox_id"])["status"] == "pending"      # the failed send left it for the next run
    assert w.outbox(research["outbox_id"])["status"] == "pending"
    assert all(key.startswith(WRITABLE_PREFIXES) for key, _ in w.s3.writes)

    # A second run finds nothing pending for answers and still nothing sendable for triage.
    again = w.run("outbox", "anything")
    assert again["kinds"]["answer"] == {"status": "relayed", "pending": 0, "sent": 0, "failed": 0, "unmarked": 0}
    assert again["kinds"]["triage"]["failed"] == 1 and w.sqs.messages[2:] == []


def test_outbox_relays_research_only_when_enabled_and_configured(tmp_path):
    w = World(tmp_path)
    research = w.research(w.finished("req-1"))

    unconfigured = w.run("outbox", None, LAB_RESEARCH_CONSUMER_ENABLED="true", LAB_RESEARCH_QUEUE_URL="")
    assert unconfigured["kinds"]["research"]["status"] == "skipped"
    assert w.outbox(research["outbox_id"])["status"] == "pending"
    assert all(url != RESEARCH_URL for url, _ in w.sqs.messages)

    enabled = w.run("outbox", None, LAB_RESEARCH_CONSUMER_ENABLED="true")
    assert enabled["kinds"]["research"] == {"status": "relayed", "pending": 1, "sent": 1, "failed": 0, "unmarked": 0}
    relayed = {"outbox_id": research["outbox_id"], "job_id": research["job_id"], "kind": "research"}
    assert (RESEARCH_URL, relayed) in w.sqs.messages
    assert w.outbox(research["outbox_id"])["status"] == "sent"


def test_outbox_counts_a_sent_message_whose_row_changed_underneath_as_unmarked(tmp_path, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    job = w.queued("req-1")

    class RacingSqs(FakeSqs):
        def send_message(self, *, QueueUrl, MessageBody, **_):
            response = super().send_message(QueueUrl=QueueUrl, MessageBody=MessageBody)
            lab_lambda.lab_jobs.mark_sent(w.table, job["outbox_id"], w.outbox(job["outbox_id"])["revision"], now=NOW)
            return response

    w.sqs = RacingSqs()
    result = w.run("outbox", {})

    assert result["kinds"]["answer"] == {"status": "relayed", "pending": 1, "sent": 1, "failed": 0, "unmarked": 1}
    assert w.outbox(job["outbox_id"])["status"] == "sent" and w.outbox(job["outbox_id"])["attempts"] == 1
    assert_clean_logs(caplog)


def test_outbox_sweeps_expired_leases_once_per_invocation_and_reports_the_counts(tmp_path, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    expired = w.running("req-1")                      # its worker died; the lease ends at claim + 960 s
    mark_sent(w.table, expired["outbox_id"], w.outbox(expired["outbox_id"])["revision"], now=w.clock.moment)
    w.clock.advance(LEASE_SECONDS + 1)
    live = w.running("req-2")                         # claimed just now, lease live
    mark_sent(w.table, live["outbox_id"], w.outbox(live["outbox_id"])["revision"], now=w.clock.moment)
    queued = w.queued("req-3")                        # pending row for the relay
    sweep_at = w.clock.moment

    result = w.run("outbox", None)

    assert result["status"] == "ok"
    assert result["kinds"]["answer"] == {"status": "relayed", "pending": 1, "sent": 1, "failed": 0, "unmarked": 0}
    assert result["lease_sweep"] == {"status": "swept", "expired": 1, "closed": 1, "skipped": 0,
                                     "job_ids": [expired["job_id"]]}
    closed = w.job(expired["job_id"])
    assert closed["status"] == "outcome_unknown" and closed["reason"] == "lease_expired"
    assert closed["lease_until"] is None and closed["triage_status"] == "skipped"
    assert closed["unknown_at"] == closed["completed_at"] == sweep_at.isoformat(timespec="microseconds")
    assert w.table.get(*keys.reservation(closed["reservation_id"]))["status"] == "unknown"   # money stays held
    assert w.outbox(expired["outbox_id"])["status"] == "done"
    assert w.job(live["job_id"])["status"] == "running" and w.job(queued["job_id"])["status"] == "queued"
    assert pending_outbox(w.table, "triage") == []   # an unknown outcome never reaches triage
    assert [job_id for _url, body in w.sqs.messages for job_id in [body["job_id"]]] == [queued["job_id"]]
    assert all(key.startswith(WRITABLE_PREFIXES) for key, _ in w.s3.writes)
    assert "action=lease_sweep" in caplog.text and expired["job_id"] in caplog.text
    assert_clean_logs(caplog)

    # The next invocation finds nothing expired; the swept job is not touched again.
    again = w.run("outbox", "anything")
    assert again["lease_sweep"] == {"status": "swept", "expired": 0, "closed": 0, "skipped": 0, "job_ids": []}
    assert w.job(expired["job_id"])["revision"] == closed["revision"]


def test_outbox_reports_a_failed_sweep_without_hiding_the_relay_counts(tmp_path, monkeypatch, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    job = w.queued("req-1")
    calls: list[tuple] = []

    def throttled(table, now=None, limit=100):
        calls.append((table, now, limit))
        raise ClientError({"Error": {"Code": "ProvisionedThroughputExceededException",
                                     "Message": f"slow down while sweeping {QUESTION}"}}, "Query")

    monkeypatch.setattr(lab_lambda.lab_jobs, "sweep_expired_leases", throttled)

    result = w.run("outbox", {})

    assert result["status"] == "ok"
    assert result["kinds"]["answer"] == {"status": "relayed", "pending": 1, "sent": 1, "failed": 0, "unmarked": 0}
    assert result["lease_sweep"] == {"status": "failed", "error": "ProvisionedThroughputExceededException"}
    assert calls == [(w.table, w.clock.moment, lab_lambda.LEASE_SWEEP_LIMIT)]
    assert w.outbox(job["outbox_id"])["status"] == "sent"
    assert "action=lease_sweep status=failed" in caplog.text
    assert_clean_logs(caplog)


# ---------------------------------------------------------------------------------------------
# Research consumer and batch shapes
# ---------------------------------------------------------------------------------------------

def test_research_handler_is_disabled_until_the_flag_is_true_and_creates_no_client(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(boto3, "client", recorder)
    for flag in ("", "false", "yes", "TRUE_"):
        for name, value in env_for("research", LAB_RESEARCH_CONSUMER_ENABLED=flag).items():
            monkeypatch.setenv(name, value)

        result = handler(sqs_event({"outbox_id": "o", "job_id": "j", "kind": "research"}), FakeContext(function_arn=ARN))

        assert result == {"status": "disabled", "reason": "research consumer not activated"}, flag
        assert recorder.calls == []


def test_research_config_defaults_to_the_answer_model_and_high_reasoning():
    config = Config.from_env(env_for("research"))
    assert config.research_model_id == MODEL and config.research_reasoning == "high"
    assert config.research_consumer_enabled is False

    explicit = Config.from_env(env_for("research", LAB_RESEARCH_MODEL_ID="global.anthropic.claude-sonnet-5",
                                       LAB_RESEARCH_REASONING="high", LAB_RESEARCH_CONSUMER_ENABLED="true"))
    assert explicit.research_model_id == "global.anthropic.claude-sonnet-5" and explicit.research_reasoning == "high"
    assert explicit.research_consumer_enabled is True

    blank = Config.from_env(env_for("research", LAB_RESEARCH_MODEL_ID="  ", LAB_RESEARCH_REASONING=" "))
    assert blank.research_model_id == MODEL and blank.research_reasoning == "high"
    assert lab_lambda.DEFAULT_RESEARCH_REASONING == "high"


def test_research_handler_requires_a_model_id_when_enabled(tmp_path):
    w = World(tmp_path)
    research = w.research(w.finished("req-1"))

    with pytest.raises(RuntimeError, match="LAB_RESEARCH_MODEL_ID"):
        w.run("research", sqs_event(message(research, "research")), LAB_RESEARCH_CONSUMER_ENABLED="true",
              LAB_ANSWER_MODEL_ID="", LAB_RESEARCH_MODEL_ID="")
    assert w.job(research["job_id"])["status"] == "queued"


def research_approval(w: World, research: dict, parent: dict, scope: dict) -> dict:
    """The APPROVAL record a student consent or professor approval leaves for the research job."""
    from byeori.lab_store import new_item, now_iso
    stamp = now_iso(w.clock.moment)
    approval = new_item(
        *keys.approval(research["approval_id"]), stamp,
        approval_id=research["approval_id"], kind="student_consent", offer_id="offer-1", candidate_id=None,
        job_id=parent["job_id"], proposal_revision=1, proposal_hash="hash", approved_by="m1",
        policy_revision=POLICY_REVISION, scope=scope, budget_usd_micros=5_000_000, model_id=None, max_calls=40,
        reread="auto", expires_at=now_iso(w.clock.moment + timedelta(days=7)), execution_id=research["job_id"],
        status="active", approved_at=stamp, request_id="resp-1", research_status="queued", note=None,
        budget_refusal=None, linked_approval_id=None,
        receipt_key=receipt_key(parent["job_id"], f"approval-{research['approval_id']}.json"),
    )
    w.table.put(approval)
    return approval


def test_research_handler_claims_runs_the_engine_under_scope_and_completes_the_job(tmp_path, caplog):
    answer = "Regional inheritance was stable (n = 120, p = 0.01). [[sources/paper-one]]"
    new_page = "Regional stability is a distinct unit. [[sources/paper-one]]\n"
    w = World(tmp_path, bedrock=FakeBedrock([
        tool_use("search_wiki", {"query": "regional inheritance cohort"}, tool_use_id="c1"),
        tool_use("write_page", {"key": "wiki/concepts/regional-stability.md", "markdown": new_page}, tool_use_id="c2"),
        text_only(answer),
    ]))
    caplog.set_level(logging.DEBUG)
    parent = w.finished("req-1")
    scope = {"question": QUESTION, "targets": [], "new_pages": [], "note": "offer", "kind": "new_synthesis", "offer_id": "offer-1"}
    research = create_research_job(w.table, w.receipts, parent_job=parent, member_id="m1", approval_id="appr-1",
                                   scope=scope, budget_usd_micros=5_000_000, now=w.clock.moment)
    research_approval(w, research, parent, scope)

    result = w.run("research", sqs_event(message(research, "research")), LAB_RESEARCH_CONSUMER_ENABLED="true",
                   LAB_RESEARCH_REASONING="high")

    assert failures(result) == []
    stored = w.job(research["job_id"])
    assert stored["status"] == "completed" and stored["attempt"] == 1
    assert stored["receipt_key"] == f"runs/lab-questions/{research['job_id']}/research.json"
    assert stored["pages_written"] == ["wiki/concepts/regional-stability.md"] and stored["index_pending"] is True
    receipt = w.receipts.get_json(stored["receipt_key"])
    assert receipt["research"]["answer"] == answer and receipt["model_id"] == MODEL and receipt["reasoning"] == "high"
    assert receipt["research"]["question_key"] is None
    assert len(w.bedrock.requests) == 3
    assert w.bedrock.requests[0]["additionalModelRequestFields"]["output_config"]["effort"] == "high"
    assert "wiki/concepts/regional-stability.md" in w.s3.objects
    assert not any(key.startswith("wiki/questions/") for key in w.s3.objects)
    written = [key for key, _ in w.s3.writes]
    assert all(key.startswith(("wiki/concepts/", "wiki/sources/", "wiki/index.md", "wiki/indexes/", "runs/agents/",
                               "runs/lab-questions/")) for key in written), written
    assert w.table.get(*keys.approval("appr-1"))["execution_status"] == "completed"
    assert w.last_deps.index is None
    assert research["job_id"] in caplog.text and "status=completed" in caplog.text
    assert_clean_logs(caplog)

    # A redelivery finds the job terminal and is dropped without a claim or a call.
    assert failures(w.run("research", sqs_event(message(research, "research")), LAB_RESEARCH_CONSUMER_ENABLED="true")) == []
    assert len(w.bedrock.requests) == 3


def test_research_handler_fails_a_job_without_a_valid_approval_and_never_calls_the_model(tmp_path, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    research = w.research(w.finished("req-1"))           # no APPROVAL record exists for appr-1

    result = w.run("research", sqs_event(message(research, "research")), LAB_RESEARCH_CONSUMER_ENABLED="true")

    assert failures(result) == []
    stored = w.job(research["job_id"])
    assert stored["status"] == "failed" and stored["error_code"] == "approval_invalid"
    assert w.bedrock.requests == []
    assert all(key.startswith(WRITABLE_PREFIXES) for key, _ in w.s3.writes)
    assert "error=approval_invalid" in caplog.text and "status=failed" in caplog.text
    assert_clean_logs(caplog)


def test_research_handler_retries_a_live_lease_and_drops_or_reports_the_rest(tmp_path, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    leased = w.research(w.finished("req-1"))
    claim(w.table, leased["job_id"], leased["outbox_id"], LEASE_SECONDS, w.clock.moment)   # another worker holds it
    unknown = {"outbox_id": "0" * 32, "job_id": "does-not-exist", "kind": "research"}
    wrong_kind = message(leased, "answer")

    result = w.run("research", sqs_event(message(leased, "research"), unknown, wrong_kind),
                   LAB_RESEARCH_CONSUMER_ENABLED="true")

    assert failures(result) == ["m1", "m3"]
    assert w.bedrock.requests == [] and w.job(leased["job_id"])["status"] == "running"
    assert "lease_held" in caplog.text and "invalid_message" in caplog.text
    assert_clean_logs(caplog)


def test_research_handler_reports_an_index_failure_after_the_claim(tmp_path, caplog):
    w = World(tmp_path)
    caplog.set_level(logging.DEBUG)
    research = w.research(w.finished("req-1"))
    w.s3.objects.pop(INDEX_KEY)

    result = w.run("research", sqs_event(message(research, "research")), LAB_RESEARCH_CONSUMER_ENABLED="true")

    assert failures(result) == ["m1"]
    stored = w.job(research["job_id"])
    assert stored["status"] == "running" and stored["attempt"] == 1
    assert w.bedrock.requests == [] and "action=open_index status=failed" in caplog.text
    assert_clean_logs(caplog)


@pytest.mark.parametrize("kind", ["answer", "triage"])
@pytest.mark.parametrize("event", [{}, {"Records": []}, {"Records": "x"}, None, "text", {"Records": [{"body": "{}"}]}])
def test_sqs_handlers_return_the_partial_batch_shape_for_events_without_usable_records(tmp_path, kind, event):
    w = World(tmp_path)

    assert w.run(kind, event) == {"batchItemFailures": []}
    assert w.bedrock.requests == [] and w.jev.calls == []


def test_every_handler_kind_dispatches_and_closes_the_index(tmp_path):
    w = World(tmp_path)
    for kind in lab_lambda.HANDLERS:
        deps = w.deps(kind)
        deps.open_index()
        handler({} if kind != "gateway" else w.event("search_wiki", {"query": "inheritance"}), w.context, deps=deps)
        assert deps.index is None, kind


# ---------------------------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------------------------

def test_module_boundaries_no_campaign_imports_no_wiki_writes_no_env_endpoint():
    source = inspect.getsource(lab_lambda)
    for forbidden in ("ingest_lambda", "aws_store", "question_agent", "agent_cache", "import mcp", "from mcp",
                      "import httpx", "from httpx", "put_object", "endpoint=", "delete_", "boto3.resource"):
        assert forbidden not in source, forbidden
    # Identity never comes from the environment and the Jev endpoint is jev_client's constant.
    assert "AWS_ACCOUNT_ID" not in source and "AWS_REGION" not in source
    assert "invoked_function_arn" in source
    assert lab_lambda.HANDLERS == ("gateway", "answer", "triage", "outbox", "research")
    assert lab_lambda.RESEARCH_DISABLED == {"status": "disabled", "reason": "research consumer not activated"}
    assert lab_lambda.LEASE_SWEEP_LIMIT == 100


def test_policy_revision_mismatch_between_environment_and_code_is_refused():
    from byeori.lab_lambda import Config
    from byeori.lab_policy import POLICY_REVISION
    env = {"LAB_TABLE": "t", "LAB_BUCKET": "b", "LAB_HANDLER": "gateway", "LAB_POLICY_REVISION": "1999-01-01-v0"}
    with pytest.raises(RuntimeError) as failure:
        Config.from_env(env)
    assert "LAB_POLICY_REVISION" in str(failure.value) and POLICY_REVISION in str(failure.value)
    assert Config.from_env({**env, "LAB_POLICY_REVISION": POLICY_REVISION}).policy_revision == POLICY_REVISION
    assert Config.from_env({**env, "LAB_POLICY_REVISION": ""}).policy_revision == POLICY_REVISION
