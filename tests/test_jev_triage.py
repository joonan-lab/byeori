from __future__ import annotations

import copy
import hashlib
import io
import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from byeori import jev_eval, jev_triage as triage


SECRET = "triage-test-secret-never-expose"
PRIVATE_REASON = "BLIND-REFERENCE-REASON-NOT-FOR-MODEL"
RESPONSE = {"model": "jev-1.13.0",
            "answers": {"route": {"type": "choice", "choice": "answer_only", "confidence": 0.8,
                                   "probabilities": {"answer_only": 0.9, "review_candidate": 0.1}}},
            "usage": {"input_tokens": 400, "output_tokens": 70}}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


class FakeS3:
    def __init__(self):
        self.objects, self.writes, self.reads = {}, [], []
        self.lock = threading.Lock()
        self.fail_result_write = False

    def get_object(self, *, Bucket, Key):
        self.reads.append(Key)
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": SECRET}}, "GetObject")
        raw = self.objects[Key]
        return {"Body": io.BytesIO(raw), "ContentLength": len(raw), "VersionId": "fixture-version",
                "ETag": '"' + hashlib.md5(raw).hexdigest() + '"'}

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=1):
        return {"Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)][:MaxKeys]}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **kwargs):
        assert Bucket == "byeori-test-results"
        assert Key.startswith(triage.PILOT_PREFIX) and IfNoneMatch == "*"
        with self.lock:
            if Key in self.objects:
                raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": SECRET}}, "PutObject")
            if self.fail_result_write and "/results/" in Key:
                raise RuntimeError(SECRET)
            self.objects[Key] = Body
            self.writes.append(Key)
        return {}


@pytest.fixture
def harness(monkeypatch):
    s3 = FakeS3()
    h = SimpleNamespace(s3=s3, posts=[], parameters=[], clients=[], response=encoded(RESPONSE),
                        post_error=None, ssm_error=None, parameter_type="SecureString", remaining_ms=60_000)
    monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_Lambda_python3.12")
    monkeypatch.setenv("JEV_RESULTS_BUCKET", "byeori-test-results")
    monkeypatch.setenv("JEV_API_KEY_PARAMETER", "/byeori/jev/api-key")
    h.context = SimpleNamespace(invoked_function_arn="arn:aws:lambda:us-east-1:123456789012:function:jev-test",
                                get_remaining_time_in_millis=lambda: h.remaining_ms)

    def get_parameter(**kwargs):
        h.parameters.append(kwargs)
        if h.ssm_error:
            raise h.ssm_error
        return {"Parameter": {"Type": h.parameter_type, "Value": SECRET}}

    def client(name, **kwargs):
        h.clients.append(name)
        if name == "s3":
            return s3
        assert name == "ssm"
        return SimpleNamespace(get_parameter=get_parameter)

    def post(payload, secret):
        assert secret == SECRET
        case_claims = [key for key in s3.objects if "/claims/" in key]
        assert case_claims, "A durable claim must precede every model call"
        h.posts.append(payload)
        if h.post_error:
            raise h.post_error
        return h.response

    monkeypatch.setattr(triage.boto3, "client", client)
    h.call = lambda event: triage.handle(event, h.context, post=post, validate=jev_eval._validated)
    cases = []
    for number in range(1, 13):
        case = {"case_id": f"C{number:02}", "source_id": f"Q{number}", "origin": "synthetic-test",
                "question": f"What evidence establishes test observation {number}?", "retrieval_ms": 1.0,
                "evidence": [{"key": f"wiki/sources/test-{number}.md", "version_id": "fixture-version",
                              "title": "A synthetic evidence fixture", "section": "Results", "start": 0,
                              "text": "The reported observation is limited to the measured condition.",
                              "has_more": False, "section_missing": False}]}
        payload = triage._payload(case)
        case.update(request_bytes=len(payload), request_sha256=hashlib.sha256(payload).hexdigest())
        cases.append(case)
    h.snapshot = {"at": "2026-09-21T00:00:00Z", "cases": cases,
                  "budget": {"max_calls": 12, "currency_cap_usd": None}}
    h.save_snapshot = lambda: s3.objects.update({triage.PILOT_PREFIX + "snapshot.json": encoded(h.snapshot)})
    h.save_snapshot()
    h.labels = [{"case_id": f"C{number:02}", "label": "answer_only", "reason": PRIVATE_REASON}
                for number in range(1, 13)]
    return h


def lock_reference(h):
    result = h.call({"action": "triage_reference", "labels": h.labels})
    assert result["status"] == "reference_locked"


def run(h, case_id="C01"):
    return h.call({"action": "triage_run", "case_id": case_id})


def assert_no_secret(h, result, capsys):
    captured = capsys.readouterr()
    material = json.dumps(result) + captured.out + captured.err
    material += "".join(raw.decode() for raw in h.s3.objects.values())
    assert SECRET not in material and "Authorization" not in material


@pytest.mark.parametrize("event", [None, [], {}, {"action": "unknown"}, {"action": []},
                                   {"action": "triage_run"}, {"action": "triage_prepare", "case_id": "C01"},
                                   *({"action": "triage_run", "case_id": "C01", key: SECRET} for key in
                                     ("state", "url", "model", "bucket", "secret", "budget_usd", "prefix"))])
def test_event_scope_rejects_arbitrary_inputs_without_aws_or_model(harness, event, capsys):
    result = harness.call(event)
    assert result["status"] == "rejected" and result["error"] == "invalid_event"
    assert not harness.clients and not harness.posts
    assert_no_secret(harness, result, capsys)


def test_runtime_guard_precedes_aws_access(harness, monkeypatch):
    monkeypatch.delenv("AWS_EXECUTION_ENV")
    assert run(harness)["error"] == "aws_runtime_required"
    assert not harness.clients and not harness.posts


def test_secret_parameter_location_is_fixed(harness, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY_PARAMETER", "/another/key")
    assert run(harness)["error"] == "missing_configuration"
    assert not harness.clients and not harness.posts


def test_preparation_reuses_frozen_snapshot_without_refreshing_sources(harness):
    result = harness.call({"action": "triage_prepare"})
    assert result["status"] == "prepared" and result["reused"] is True
    assert result["case_ids"] == [f"C{number:02}" for number in range(1, 13)]
    assert harness.s3.reads == [triage.PILOT_PREFIX + "snapshot.json"]
    assert not harness.s3.writes and not harness.posts


def test_sampling_is_reproducible_and_independent_of_campaign_outcome():
    manifest = [{"id": f"q{number}", "question": f"Synthetic campaign query {number}",
                 "origin": "synthetic-test", "status": "answer_ready"} for number in range(20)]
    selected = triage._selected(manifest)
    changed = [{**row, "status": "answer_partial", "pages_written": ["ignored"]} for row in reversed(manifest)]
    assert triage._selected(changed) == selected
    assert len(selected) == 12 and len({row["question"] for row in selected}) == 12


@pytest.mark.parametrize("case_id", ["C00", "C13", "c01", "../C01", SECRET])
def test_only_twelve_frozen_cases_can_run(harness, case_id, capsys):
    assert run(harness, case_id)["error"] == "unknown_case"
    assert not harness.posts and not harness.parameters
    assert_no_secret(harness, {}, capsys)


def test_model_call_requires_reference_and_enough_time(harness):
    assert run(harness)["status"] == "failed"
    assert not harness.posts and not harness.parameters and not harness.s3.writes
    lock_reference(harness)
    harness.remaining_ms = 34_999
    assert run(harness)["error"] == "insufficient_execution_time"
    assert not harness.posts and not harness.parameters


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(label="answer_ready"),
    lambda rows: rows[0].update(label="answer_partial"),
    lambda rows: rows[0].update(label=SECRET),
    lambda rows: rows[0].update(reason=""),
    lambda rows: rows[0].update(reason="x" * 2001),
    lambda rows: rows[0].update(extra=SECRET),
])
def test_reference_must_be_complete_unique_and_explicit(harness, change, capsys):
    change(harness.labels)
    result = harness.call({"action": "triage_reference", "labels": harness.labels})
    assert result["status"] == "failed"
    assert triage.PILOT_PREFIX + "reference.json" not in harness.s3.objects
    assert not harness.posts
    assert_no_secret(harness, result, capsys)


def test_reference_is_immutable_and_never_enters_model_payload(harness, capsys):
    lock_reference(harness)
    original = harness.s3.objects[triage.PILOT_PREFIX + "reference.json"]
    result = run(harness)
    assert result["status"] == "ok" and len(harness.posts) == 1
    payload = harness.posts[0]
    assert PRIVATE_REASON.encode() not in payload
    state = json.loads(payload)["state"]
    assert set(state) == {"question", "evidence", "retrieval_scope"}
    assert not {"label", "reason", "reference", "status", "pages_written"} & set(state)
    assert len(payload) <= 30_000
    assert harness.parameters == [{"Name": "/byeori/jev/api-key", "WithDecryption": True}]
    harness.labels[0]["label"] = "review_candidate"
    assert harness.call({"action": "triage_reference", "labels": harness.labels})["status"] == "failed"
    assert harness.s3.objects[triage.PILOT_PREFIX + "reference.json"] == original
    assert_no_secret(harness, result, capsys)


def test_repeat_success_reuses_receipt_without_ssm_or_second_post(harness):
    lock_reference(harness)
    first, second = run(harness), run(harness)
    assert first["status"] == second["status"] == "ok" and second["reused"] is True
    assert len(harness.posts) == len(harness.parameters) == 1


def test_twelve_cases_claim_at_most_twelve_calls_even_when_replayed(harness):
    lock_reference(harness)
    for _ in range(2):
        for number in range(1, 13):
            assert run(harness, f"C{number:02}")["status"] == "ok"
    assert len(harness.posts) == 12
    assert sum("/claims/" in key for key in harness.s3.writes) == 12
    assert run(harness, "C13")["error"] == "unknown_case"


def test_unresolved_claim_does_not_trigger_another_request(harness):
    lock_reference(harness)
    harness.s3.objects[triage.PILOT_PREFIX + "claims/C01.json"] = encoded({"request_sha256": "prior-attempt"})
    result = run(harness)
    assert result["status"] == "unresolved_claim" and result["request_attempted"] == "unknown"
    assert not harness.posts and not harness.parameters
    summary = harness.call({"action": "triage_summary"})
    assert sum(row["claimed_without_result"] for row in summary["rows"]) == 1
    assert summary["unknown_usage_cases"] == 1


def test_result_storage_failure_leaves_claim_and_blocks_retry(harness, capsys):
    lock_reference(harness)
    harness.s3.fail_result_write = True
    first = run(harness)
    second = run(harness)
    assert first["status"] == "failed" and second["status"] == "unresolved_claim"
    assert len(harness.posts) == 1
    assert_no_secret(harness, first, capsys)


def test_parallel_duplicate_calls_have_at_most_one_post(harness):
    lock_reference(harness)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: run(harness), range(12)))
    assert len(harness.posts) == 1
    assert sum("/claims/" in key for key in harness.s3.writes) == 1


@pytest.mark.parametrize("oversized", [False, True])
def test_changed_or_oversized_payload_never_reaches_model(harness, oversized):
    case = harness.snapshot["cases"][0]
    case["question"] = "가" * 12_000 if oversized else "Changed after payload was frozen"
    if oversized:
        payload = triage._payload(case)
        assert len(payload) > 30_000
        case["request_sha256"] = hashlib.sha256(payload).hexdigest()
    harness.save_snapshot()
    lock_reference(harness)
    assert run(harness)["status"] == "failed"
    assert not harness.posts and not harness.parameters


def test_reference_is_bound_to_exact_snapshot(harness):
    lock_reference(harness)
    harness.snapshot["cases"][0]["question"] = "A different question"
    harness.save_snapshot()
    assert run(harness)["status"] == "failed" and not harness.posts


@pytest.mark.parametrize("code", [302, 401, 429, 500])
def test_http_error_is_redacted_terminal_and_not_retried(harness, code, capsys):
    class Unreadable(io.BytesIO):
        def read(self, *args):
            pytest.fail("Provider error body must not be read")

    lock_reference(harness)
    harness.post_error = urllib.error.HTTPError(SECRET, code, SECRET, {}, Unreadable(SECRET.encode()))
    result = run(harness)
    assert result["error"] == "upstream_http_error" and result["http_status"] == code
    assert result["usage"] is None and result["estimated_usd"] is None
    assert run(harness)["reused"] is True and len(harness.posts) == 1
    assert_no_secret(harness, result, capsys)


def test_transport_failure_and_invalid_response_never_echo_external_text(harness, capsys):
    lock_reference(harness)
    harness.post_error = RuntimeError("Authorization " + SECRET)
    first = run(harness)
    assert first["error"] == "upstream_request_failed"
    harness.post_error = None
    harness.response = ("Authorization " + SECRET).encode()
    second = run(harness, "C02")
    assert second["error"] == "invalid_response" and len(harness.posts) == 2
    assert_no_secret(harness, [first, second], capsys)


def test_ssm_error_cannot_leak_or_trigger_post(harness, capsys):
    lock_reference(harness)
    harness.ssm_error = RuntimeError("Authorization " + SECRET)
    result = run(harness)
    assert result["error"] == "secret_unavailable" and not harness.posts
    assert_no_secret(harness, result, capsys)


def test_excess_input_usage_is_retained_without_claiming_success(harness):
    lock_reference(harness)
    response = copy.deepcopy(RESPONSE)
    response["usage"]["input_tokens"] = 32_769
    harness.response = encoded(response)
    result = run(harness)
    assert result["status"] == "failed" and result["error"] == "input_usage_exceeds_reservation"
    assert result["usage"]["input_tokens"] == 32_769 and result["estimated_usd"] > 0
    assert run(harness)["reused"] is True and len(harness.posts) == 1


def test_summary_separates_missing_evidence_from_reference_agreement(harness):
    harness.labels[0]["label"] = "insufficient_retrieval"
    lock_reference(harness)
    run(harness)
    run(harness, "C02")
    summary = harness.call({"action": "triage_summary"})
    assert summary["successful"] == 2 and summary["reference_scored"] == 1
    assert summary["agreement"] == 1 and summary["confusion"] == {"answer_only->answer_only": 1}
    assert summary["budget"]["currency_cap_usd"] is None
    assert summary["reported_estimated_usd"] > 0
    assert len(harness.posts) == 2
