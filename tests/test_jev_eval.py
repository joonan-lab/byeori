from __future__ import annotations

import copy
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from byeori import jev_eval as jev


SECRET = "test-secret-never-record-this"
RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {"route": {"type": "choice", "choice": "answer_only", "confidence": 0.78,
                           "probabilities": {"answer_only": 0.85, "review_candidate": 0.15}}},
    "usage": {"input_tokens": 392, "output_tokens": 65},
}


class Response(io.BytesIO):
    status = 200


@pytest.fixture
def harness(monkeypatch):
    state = SimpleNamespace(posts=[], writes=[], parameters=[], clients=[], body=json.dumps(RESPONSE).encode(),
                            http_error=None, ssm_error=None, s3_error=None, parameter_type="SecureString",
                            secret=SECRET)
    monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_Lambda_python3.12")
    monkeypatch.setenv("JEV_API_KEY_PARAMETER", "/byeori/jev/key")
    monkeypatch.setenv("JEV_RESULTS_BUCKET", "byeori-test-results")

    def get_parameter(**kwargs):
        state.parameters.append(kwargs)
        if state.ssm_error:
            raise state.ssm_error
        return {"Parameter": {"Type": state.parameter_type, "Value": state.secret}}

    def put_object(**kwargs):
        state.writes.append(kwargs)
        if state.s3_error:
            raise state.s3_error
        return {}

    def client(name, **kwargs):
        state.clients.append((name, kwargs))
        return SimpleNamespace(get_parameter=get_parameter, put_object=put_object)

    def open_request(request, timeout):
        state.posts.append((request, timeout))
        if state.http_error:
            raise state.http_error
        return Response(state.body)

    monkeypatch.setattr(jev.boto3, "client", client)
    monkeypatch.setattr(jev.urllib.request.OpenerDirector, "open", lambda self, request, timeout: open_request(request, timeout))
    state.context = SimpleNamespace(invoked_function_arn="arn:aws:lambda:us-east-1:123456789012:function:jev-smoke")
    return state


def call(harness, event=None):
    return jev.handler({"action": "smoke"} if event is None else event, harness.context)


def assert_redacted(result, harness, capsys):
    output = capsys.readouterr()
    material = json.dumps(result) + "".join(write["Body"].decode() for write in harness.writes)
    material += output.out + output.err
    assert SECRET not in material
    assert "Authorization" not in material


def test_smoke_posts_once_and_stores_only_sanitized_metrics(harness, capsys):
    response = copy.deepcopy(RESPONSE)
    response["debug"] = {"Authorization": SECRET}
    harness.body = json.dumps(response).encode()
    result = call(harness)
    assert result["status"] == "ok"
    assert result["usage"] == RESPONSE["usage"]
    assert result["estimated_usd"] == pytest.approx(392 * 0.042 / 1_000_000)
    assert result["choice"] == "answer_only"
    assert result["confidence"] == 0.78
    assert result["probabilities"] == RESPONSE["answers"]["route"]["probabilities"]
    assert len(harness.posts) == 1
    request, timeout = harness.posts[0]
    assert request.full_url == "https://api.typesafe.ai/v1/systemone"
    assert request.method == "POST" and timeout == 20
    assert request.get_header("Authorization") == "Bearer " + SECRET
    assert json.loads(request.data)["model"] == jev.MODEL
    assert json.loads(request.data)["state"] == "Existing wiki fully answers the request, no missing evidence."
    assert harness.parameters == [{"Name": "/byeori/jev/key", "WithDecryption": True}]
    assert all(config["config"].retries["total_max_attempts"] == 1 for _, config in harness.clients)
    write = harness.writes[0]
    assert write["Bucket"] == "byeori-test-results" and write["IfNoneMatch"] == "*"
    assert write["Key"].startswith("runs/jev-evaluations/")
    stored = json.loads(write["Body"])
    assert "state" not in stored and "answers" not in stored and "debug" not in stored
    assert stored["budget"]["request_bytes"] == len(request.data) <= 1_024
    assert stored["budget"]["estimated_cost_ceiling_usd"] < stored["budget"]["budget_usd"] <= 0.0002
    assert_redacted(result, harness, capsys)


def test_receipts_never_upsert_an_existing_key(harness):
    first, second = call(harness), call(harness)
    assert first["receipt_key"] != second["receipt_key"]
    assert len(harness.posts) == 2
    assert all(write["IfNoneMatch"] == "*" for write in harness.writes)


def test_dry_run_has_no_aws_or_http_calls(harness, monkeypatch):
    monkeypatch.delenv("AWS_EXECUTION_ENV")
    result = call(harness, {"action": "smoke", "dry_run": True})
    assert result["status"] == "dry_run" and not result["request_attempted"]
    assert not harness.clients and not harness.posts


def test_api_latency_is_separate_from_ssm_and_total_evaluation(harness, monkeypatch):
    moments = iter([10.0, 11.0, 11.125, 12.0])
    monkeypatch.setattr(jev.time, "monotonic", lambda: next(moments))
    result = call(harness)
    assert result["api_elapsed_ms"] == 125.0
    assert result["elapsed_ms"] == 2_000.0


@pytest.mark.parametrize("event", [None, [], {}, {"action": "evaluate"}, {"action": "smoke", "dry_run": "true"},
                                   *({"action": "smoke", key: SECRET} for key in
                                     ("state", "url", "secret", "model", "bucket", "budget_usd", "prefix"))])
def test_event_cannot_override_scope_or_configuration(harness, event, capsys):
    result = jev.handler(event, harness.context)
    assert result == {"status": "rejected", "error": "invalid_event"}
    assert not harness.clients and not harness.posts
    assert_redacted(result, harness, capsys)


def test_live_calls_require_aws_runtime(harness, monkeypatch):
    monkeypatch.delenv("AWS_EXECUTION_ENV")
    assert call(harness)["error"] == "aws_runtime_required"
    assert not harness.clients and not harness.posts


@pytest.mark.parametrize("value, flag", [
    (SECRET, None),
    ('"' + SECRET + '"', "has_quote_or_backtick"),
    ("`" + SECRET + "`", "has_quote_or_backtick"),
    (json.dumps({"key": SECRET}), "looks_like_json_container"),
    ("JEV_API_KEY=" + SECRET, "looks_like_assignment"),
    ("Bearer " + SECRET, "has_bearer_prefix"),
    (SECRET + "\n", "has_whitespace"),
    (SECRET + "\u200b", "has_non_ascii_or_control"),
])
def test_key_format_inspection_never_calls_api_or_returns_secret(harness, value, flag, capsys):
    harness.secret = value
    result = call(harness, {"action": "check_key_format"})
    assert result["status"] == "format_checked"
    assert not result["request_attempted"] and not harness.posts
    assert result["usage"] is None and result["api_elapsed_ms"] is None
    assert result["key_format"]["nonempty"]
    assert all(type(value) is bool for value in result["key_format"].values())
    if flag:
        assert result["key_format"][flag]
    else:
        assert all(not value for key, value in result["key_format"].items()
                   if key not in ("nonempty", "length_within_limit"))
    assert_redacted(result, harness, capsys)


def test_key_format_inspection_requires_aws_runtime(harness, monkeypatch):
    monkeypatch.delenv("AWS_EXECUTION_ENV")
    assert call(harness, {"action": "check_key_format"})["error"] == "aws_runtime_required"
    assert not harness.clients and not harness.posts


def test_budget_guard_precedes_secret_retrieval(harness, monkeypatch):
    monkeypatch.setattr(jev, "BUDGET_USD", 0.00001)
    assert call(harness)["error"] == "budget_guard"
    assert not harness.clients and not harness.posts


@pytest.mark.parametrize("parameter_type", ["String", "StringList"])
def test_secret_must_be_a_secure_string(harness, parameter_type, capsys):
    harness.parameter_type = parameter_type
    result = call(harness)
    assert result["error"] == "secret_unavailable" and not harness.posts
    assert_redacted(result, harness, capsys)


def test_secret_lookup_error_is_redacted(harness, capsys):
    harness.ssm_error = RuntimeError("Authorization " + SECRET)
    result = call(harness)
    assert result["error"] == "secret_unavailable" and not harness.posts
    assert_redacted(result, harness, capsys)


@pytest.mark.parametrize("code", [301, 302, 307, 401, 429, 500])
def test_http_errors_record_only_status_and_never_read_body_or_retry(harness, code, capsys):
    class Unreadable(io.BytesIO):
        def read(self, *args):
            pytest.fail("External error body must never be read")

    harness.http_error = urllib.error.HTTPError(SECRET, code, SECRET, {"Authorization": SECRET}, Unreadable(SECRET.encode()))
    result = call(harness)
    assert result["error"] == "upstream_http_error" and result["http_status"] == code
    assert result["usage"] is None and result["estimated_usd"] is None
    assert len(harness.posts) == 1
    assert_redacted(result, harness, capsys)


def test_redirect_handler_refuses_credentials_forwarding():
    request = jev.urllib.request.Request(jev.ENDPOINT, data=b"{}", headers={"Authorization": SECRET})
    assert jev._NoRedirect().redirect_request(request, None, 302, "", {}, "https://attacker.invalid") is None


def test_network_error_has_no_retry_or_exception_text(harness, capsys):
    harness.http_error = urllib.error.URLError("Authorization " + SECRET)
    result = call(harness)
    assert result["error"] == "upstream_request_failed" and len(harness.posts) == 1
    assert_redacted(result, harness, capsys)


def test_response_bytes_are_bounded(harness, capsys):
    harness.body = b"x" * (jev.MAX_RESPONSE_BYTES + 1) + SECRET.encode()
    result = call(harness)
    assert result["error"] == "upstream_request_failed" and len(harness.posts) == 1
    assert_redacted(result, harness, capsys)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(model=SECRET),
    lambda r: r.update(usage={"input_tokens": SECRET, "output_tokens": 1}),
    lambda r: r.update(usage={"input_tokens": True, "output_tokens": 1}),
    lambda r: r.update(usage={"input_tokens": -1, "output_tokens": 1}),
    lambda r: r["answers"]["route"].update(choice=SECRET),
    lambda r: r["answers"]["route"].update(choice="review_candidate"),
    lambda r: r["answers"]["route"].update(type=SECRET),
    lambda r: r["answers"]["route"].update(confidence=float("nan")),
    lambda r: r["answers"]["route"].update(probabilities={"answer_only": 0.4, "review_candidate": 0.1}),
    lambda r: r["answers"]["route"].update(probabilities={"answer_only": 1.1, "review_candidate": -0.1}),
    lambda r: r["answers"]["route"].update(probabilities={"answer_only": 1, SECRET: 0}),
])
def test_response_schema_rejects_untrusted_fields(harness, mutation, capsys):
    response = copy.deepcopy(RESPONSE)
    mutation(response)
    harness.body = json.dumps(response).encode()
    result = call(harness)
    assert result["error"] == "invalid_response"
    assert result["usage"] is None and "choice" not in result
    assert len(harness.posts) == 1
    assert_redacted(result, harness, capsys)


def test_rounded_probabilities_are_accepted(harness):
    response = copy.deepcopy(RESPONSE)
    response["answers"]["route"]["probabilities"] = {"answer_only": 0.84, "review_candidate": 0.15}
    harness.body = json.dumps(response).encode()
    assert call(harness)["status"] == "ok"


def test_excess_usage_is_reported_without_claiming_budget_enforced_after_call(harness):
    response = copy.deepcopy(RESPONSE)
    response["usage"]["input_tokens"] = jev.INPUT_TOKEN_CEILING + 1
    harness.body = json.dumps(response).encode()
    result = call(harness)
    assert result["status"] == "failed" and result["error"] == "usage_exceeds_reservation"
    assert result["usage"]["input_tokens"] == 4_097 and result["estimated_usd"] > 0
    assert len(harness.posts) == 1


def test_storage_error_is_redacted_and_does_not_repeat_model_call(harness, capsys):
    harness.s3_error = RuntimeError("Authorization " + SECRET)
    result = call(harness)
    assert result["error"] == "receipt_store_failed" and not result["receipt_saved"]
    assert result["usage"] == RESPONSE["usage"] and len(harness.posts) == 1
    assert_redacted(result, harness, capsys)


# ---------------------------------------------------------------------------------------------
# Reranking probe: does one request carry a judgement per candidate?
# ---------------------------------------------------------------------------------------------

PROBE_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {name: {"type": "choice", "choice": choice, "confidence": 0.8,
                       "probabilities": {"useful": p, "not_useful": round(1 - p, 2)}}
                for name, choice, p in (("c0", "useful", 0.92), ("c1", "not_useful", 0.11),
                                        ("c2", "not_useful", 0.2))},
    "usage": {"input_tokens": 640, "output_tokens": 48},
}


def test_probe_asks_one_question_per_candidate_in_a_single_request(harness, capsys):
    harness.body = json.dumps(PROBE_RESPONSE).encode()
    result = call(harness, {"action": "probe_multi"})

    assert len(harness.posts) == 1
    sent = json.loads(harness.posts[0][0].data)
    assert sorted(sent["questions"]) == ["c0", "c1", "c2"]
    assert all(q["type"] == "choice" and sorted(q["criteria"]) == ["not_useful", "useful"]
               for q in sent["questions"].values())
    assert result["status"] == "ok" and result["multi_question_supported"] is True
    assert result["answered"] == ["c0", "c1", "c2"]
    assert result["answers_per_candidate"]["c0"]["useful_probability"] == 0.92
    assert result["answers_per_candidate"]["c1"]["choice"] == "not_useful"
    assert_redacted(result, harness, capsys)


def test_probe_records_a_provider_that_answers_only_the_first_question(harness, capsys):
    """The outcome that would make per-candidate reranking too expensive to build."""
    single = {**PROBE_RESPONSE, "answers": {"c0": PROBE_RESPONSE["answers"]["c0"]}}
    harness.body = json.dumps(single).encode()
    result = call(harness, {"action": "probe_multi"})

    assert result["status"] == "ok" and result["multi_question_supported"] is False
    assert result["asked"] == ["c0", "c1", "c2"] and result["answered"] == ["c0"]
    assert_redacted(result, harness, capsys)


def test_probe_keeps_its_own_budget_and_never_loosens_the_smoke_guard(harness):
    assert jev.PROBE_MAX_REQUEST_BYTES > jev.MAX_REQUEST_BYTES
    assert jev.PROBE_BUDGET_USD > jev.BUDGET_USD
    assert len(jev._payload()) <= jev.MAX_REQUEST_BYTES          # the smoke path is untouched
    assert len(jev._probe_payload()) <= jev.PROBE_MAX_REQUEST_BYTES
    assert jev._budget(jev._payload())["budget_usd"] == jev.BUDGET_USD


def test_probe_dry_run_makes_no_request(harness):
    result = call(harness, {"action": "probe_multi", "dry_run": True})
    assert result["status"] == "dry_run" and harness.posts == []


def test_probe_sends_no_wiki_content(harness):
    """The probe is synthetic: nothing read from the corpus may leave AWS through it."""
    harness.body = json.dumps(PROBE_RESPONSE).encode()
    call(harness, {"action": "probe_multi"})
    sent = harness.posts[0][0].data.decode()
    assert "wiki/" not in sent and "s3://" not in sent and "runs/" not in sent


# ---------------------------------------------------------------------------------------------
# Reranking: same candidates, Jev's order against BM25's
# ---------------------------------------------------------------------------------------------

CANDIDATES = [
    {"name": "c00", "key": "wiki/sources/a.md", "doc_type": "note", "title": "A", "section": "7. Glossary",
     "bm25_rank": 1, "bm25_score": 140.0, "text": "definitions of STR and GWAS"},
    {"name": "c01", "key": "wiki/sources/b.md", "doc_type": "note", "title": "B", "section": "4. Key Results",
     "bm25_rank": 2, "bm25_score": 120.0, "text": "imputation r2 fell below 0.5 under 1% frequency"},
    {"name": "c02", "key": "wiki/concepts/c.md", "doc_type": "concept", "title": "C", "section": "1",
     "bm25_rank": 3, "bm25_score": 100.0, "text": "arrays miss repeat variation structurally"},
]


def rerank_response(probabilities, tokens=(900, 120)):
    return json.dumps({
        "model": "jev-1.13.0",
        "answers": {name: {"type": "choice", "choice": "useful" if p >= 0.5 else "not_useful",
                           "confidence": 0.9, "probabilities": {"useful": p, "not_useful": round(1 - p, 2)}}
                    for name, p in probabilities.items()},
        "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
    }).encode()


def test_rerank_reorders_candidates_by_usefulness_and_keeps_the_bm25_order(harness, capsys, monkeypatch):
    monkeypatch.setattr(jev, "_rerank_candidates", lambda question, limit, bucket: list(CANDIDATES))
    harness.body = rerank_response({"c00": 0.02, "c01": 0.97, "c02": 0.61})

    result = call(harness, {"action": "rerank", "question": "does STR imputation miss rare alleles?"})

    assert result["status"] == "ok" and result["answered"] == 3
    assert result["bm25_order"] == ["wiki/sources/a.md", "wiki/sources/b.md", "wiki/concepts/c.md"]
    # The glossary section led BM25 and is last once Jev has judged it.
    assert result["jev_order"] == ["wiki/sources/b.md", "wiki/concepts/c.md", "wiki/sources/a.md"]
    sent = json.loads(harness.posts[0][0].data)
    assert sorted(sent["questions"]) == ["c00", "c01", "c02"]
    assert "does STR imputation miss rare alleles?" in sent["state"]
    assert_redacted(result, harness, capsys)


def test_rerank_keeps_a_candidate_jev_did_not_answer_but_ranks_it_last(harness, monkeypatch):
    monkeypatch.setattr(jev, "_rerank_candidates", lambda question, limit, bucket: list(CANDIDATES))
    harness.body = rerank_response({"c00": 0.4, "c02": 0.9})     # c01 missing from the response

    result = call(harness, {"action": "rerank", "question": "q"})

    assert result["answered"] == 2 and result["candidates"] == 3
    assert result["jev_order"][-1] == "wiki/sources/b.md"
    assert next(c for c in result["scored"] if c["name"] == "c01")["useful_probability"] is None


def test_rerank_trims_candidates_until_the_request_fits_jevs_input_cap(harness, monkeypatch):
    fat = [{**CANDIDATES[0], "name": f"c{i:02d}", "key": f"wiki/sources/{i}.md",
            "text": "x" * jev.RERANK_SNIPPET_CHARS, "bm25_rank": i + 1} for i in range(jev.RERANK_CANDIDATES)]
    monkeypatch.setattr(jev, "_rerank_candidates", lambda question, limit, bucket: list(fat))
    harness.body = rerank_response({f"c{i:02d}": 0.5 for i in range(jev.RERANK_CANDIDATES)})

    result = call(harness, {"action": "rerank", "question": "q"})

    assert result["status"] == "ok"
    assert len(harness.posts[0][0].data) <= jev.RERANK_MAX_REQUEST_BYTES
    assert result["candidates"] < jev.RERANK_CANDIDATES      # the weakest BM25 candidates were dropped


@pytest.mark.parametrize("event", [
    {"action": "rerank"},                                   # no question
    {"action": "rerank", "question": "   "},
    {"action": "rerank", "question": "q", "limit": 0},
    {"action": "rerank", "question": "q", "limit": jev.RERANK_CANDIDATES + 1},
    {"action": "rerank", "question": "q", "bucket": "other"},
    {"action": "rerank", "question": "x" * (jev.RERANK_QUESTION_CHARS + 1)},
])
def test_rerank_rejects_a_malformed_event_before_any_call(harness, event):
    assert call(harness, event) == {"status": "rejected", "error": "invalid_event"}
    assert harness.posts == [] and harness.parameters == []


def test_rerank_reports_a_search_failure_without_calling_the_provider(harness, monkeypatch):
    def boom(question, limit, bucket):
        raise RuntimeError("index unavailable")
    monkeypatch.setattr(jev, "_rerank_candidates", boom)
    assert call(harness, {"action": "rerank", "question": "q"}) == {"status": "failed",
                                                                    "error": "candidates_unavailable"}
    assert harness.posts == []


def test_rerank_criteria_do_not_penalise_the_sections_the_packet_carries_on_purpose(harness, monkeypatch):
    """Revision 1 rejected a passage that 'recites procedure', which is a methods section."""
    assert "recites procedure" not in json.dumps(jev.RERANK_CRITERIA)
    assert "methods" in jev.RERANK_INSTRUCTIONS
    # What the ranking must still reject is a passage that matches any question in the field.
    assert "glossary" in jev.RERANK_CRITERIA["not_useful"]
    monkeypatch.setattr(jev, "_rerank_candidates", lambda question, limit, bucket: list(CANDIDATES))
    harness.body = rerank_response({"c00": 0.1, "c01": 0.9, "c02": 0.8})
    result = call(harness, {"action": "rerank", "question": "q"})
    assert result["criteria_revision"] == jev.RERANK_CRITERIA_REVISION
