"""Verify one-time question dispatch and honest AWS-only campaign progress."""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from botocore.exceptions import ClientError

from byeori.question_campaign import plan_batch, progress, run_one
from test_wiki_connections import S3


RUN = "research-1437"
PREFIX = f"runs/questions/{RUN}"
MANIFEST = f"{PREFIX}/manifest.json"


class CampaignS3(S3):
    def __init__(self, rows):
        super().__init__({MANIFEST: json.dumps(rows, ensure_ascii=False)})
        self.lock = threading.Lock()

    def put_object(self, **kwargs):
        with self.lock:
            return super().put_object(**kwargs)

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        cloud = self

        class Paginator:
            def paginate(self, *, Bucket, Prefix):
                keys = sorted(key for key in cloud.objects if key.startswith(Prefix))
                for start in range(0, len(keys), 3):
                    yield {"Contents": [{"Key": key} for key in keys[start:start + 3]]}

        return Paginator()


def rows(count):
    return [{"id": f"q{number}", "question": f"What does experiment {number} establish?",
             "origin": f"questions/original-{number}.md"} for number in range(count)]


def event(item):
    return {"run_id": RUN, "manifest_key": MANIFEST, **item}


def answer(number=0, **changes):
    return {"status": "answer_ready", "answer": "A substantive answer. " * 500,
            "question_key": f"wiki/questions/answer-{number}.md", "estimated_usd": 0.25,
            "seconds": 30.5, "trace_key": f"runs/agents/2026-09-21/{number:032x}.json",
            "pages_written": [{"key": f"wiki/concepts/insight-{number}.md", "replaced": False}],
            **changes}


def test_plan_has_bounded_windows_and_no_scientific_page_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = rows(49)
    cloud = CampaignS3(source)
    first = plan_batch({"run_id": RUN}, s3=cloud, bucket="bucket")
    second = plan_batch({"run_id": RUN, "offset": first["next_offset"]}, s3=cloud, bucket="bucket")
    last = plan_batch({"run_id": RUN, "offset": second["next_offset"]}, s3=cloud, bucket="bucket")

    assert first["concurrency"] == 6 and first["batch_size"] == 24
    assert first["offset"] == 0 and first["next_offset"] == 24 and not first["done"]
    assert second["next_offset"] == 48 and not second["done"]
    assert last["total"] == 49 and not last["done"] and last["next_offset"] == 49
    exhausted = plan_batch({"run_id": RUN, "offset": 49}, s3=cloud, bucket="bucket")
    assert exhausted["done"] and exhausted["items"] == []
    assert first["items"] + second["items"] + last["items"] == [event(item) for item in source]
    assert cloud.writes == [] and list(tmp_path.iterdir()) == []


def test_last_nonempty_batch_is_not_done_before_the_state_machine_runs_its_items():
    cloud = CampaignS3(rows(26))
    last = plan_batch({"run_id": RUN, "offset": 24}, s3=cloud, bucket="bucket")
    assert len(last["items"]) == 2 and last["next_offset"] == 26 and last["done"] is False
    exhausted = plan_batch({"run_id": RUN, "offset": 26}, s3=cloud, bucket="bucket")
    assert exhausted["items"] == [] and exhausted["done"] is True


def test_run_once_keeps_complete_model_result_in_s3_and_returns_tiny_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = rows(1)
    cloud = CampaignS3(source)
    result = answer(future_metadata={"preserve": [1, 2, 3]})
    original = copy.deepcopy(result)
    calls = []

    def callback(request):
        calls.append(request)
        return result

    first = run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
    repeated = run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
    receipt = json.loads(cloud.objects[f"{PREFIX}/results/q0.json"])
    assert calls == [{"action": "answer_question", "title": source[0]["question"]}]
    assert receipt["result"] == original == result
    assert receipt["question"] == source[0]["question"] and receipt["terminal"] is True
    assert first["status"] == "answer_ready" and first["pages_count"] == 1
    assert not first["reused"] and repeated == {**first, "reused": True}
    assert len(json.dumps(first).encode()) < 1024 and "answer" not in first
    assert all(conditions == {"IfNoneMatch": "*"} for _, conditions in cloud.writes)
    planned = plan_batch({"run_id": RUN}, s3=cloud, bucket="bucket")
    assert planned["items"] == [] and planned["skipped"] == [repeated] and planned["claimed"] == []
    assert list(tmp_path.iterdir()) == []


def test_parallel_duplicate_dispatch_calls_model_once():
    source = rows(1)
    cloud = CampaignS3(source)
    started, release = threading.Event(), threading.Event()
    calls = []

    def callback(request):
        calls.append(request)
        started.set()
        assert release.wait(5)
        return answer()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run_one, event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
        assert started.wait(5)
        try:
            duplicate = pool.submit(run_one, event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback).result(5)
            assert duplicate["status"] == "claimed" and duplicate["question_key"] is None
        finally:
            release.set()
        assert first.result(5)["status"] == "answer_ready"
    assert len(calls) == 1


def test_abrupt_interruption_is_never_automatically_replayed():
    class Interrupted(BaseException):
        pass

    source = rows(1)
    cloud = CampaignS3(source)
    calls = []

    def callback(request):
        calls.append(request)
        raise Interrupted()

    with pytest.raises(Interrupted):
        run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
    repeated = run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
    planned = plan_batch({"run_id": RUN}, s3=cloud, bucket="bucket")
    assert len(calls) == 1 and repeated["status"] == "claimed"
    assert planned["items"] == planned["skipped"] == []
    assert planned["claimed"] == [{"id": "q0", "status": "claimed", "claim_key": f"{PREFIX}/claims/q0.json"}]
    summary = progress({"run_id": RUN}, s3=cloud, bucket="bucket")
    assert summary["completed"] == 0 and summary["claimed"] == 1 and summary["answer_saved"] == 0


@pytest.mark.parametrize("failure", [RuntimeError("Model failed"), ClientError(
    {"Error": {"Code": "ThrottlingException", "Message": "Model capacity"}}, "Converse")])
def test_model_exception_has_one_terminal_failure_receipt_and_no_retry(failure):
    source = rows(1)
    cloud = CampaignS3(source)
    calls = []

    def callback(request):
        calls.append(request)
        raise failure

    result = run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
    assert result["status"] == "answer_failed" and result["question_key"] is None
    stored = json.loads(cloud.objects[f"{PREFIX}/results/q0.json"])
    assert stored["terminal"] and stored["result"]["error"] == str(failure)
    assert plan_batch({"run_id": RUN}, s3=cloud, bucket="bucket")["items"] == []
    assert run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)["reused"]
    assert len(calls) == 1


@pytest.mark.parametrize("operation,code", [("PutObject", "AccessDenied"), ("GetObject", "NoSuchKey"),
                                            ("Converse", "AccessDeniedException")])
def test_aws_storage_and_permission_errors_are_not_disguised_as_model_failures(operation, code):
    source = rows(1)
    cloud = CampaignS3(source)
    failure = ClientError({"Error": {"Code": code}}, operation)

    def callback(request):
        raise failure

    with pytest.raises(ClientError) as caught:
        run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
    assert caught.value is failure
    assert f"{PREFIX}/results/q0.json" not in cloud.objects
    assert progress({"run_id": RUN}, s3=cloud, bucket="bucket")["claimed"] == 1


def test_result_storage_failure_leaves_claim_and_does_not_repeat_model_call():
    source = rows(1)
    cloud = CampaignS3(source)
    calls = []

    def callback(request):
        calls.append(request)
        cloud.denied.add(f"{PREFIX}/results/q0.json")
        return answer()

    with pytest.raises(ClientError, match="AccessDenied"):
        run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
    cloud.denied.clear()
    result = run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=callback)
    assert result["status"] == "claimed" and len(calls) == 1


def test_progress_counts_real_saved_answers_and_page_operations_without_bodies():
    source = rows(10)
    cloud = CampaignS3(source)
    results = [answer(number) for number in range(7)]
    results[1]["status"] = "answer_partial"
    results[2]["answer"] = ""
    results[3]["question_key"] = None
    results[4]["estimated_usd"] = None
    results[5]["pages_written"] = [{"key": "wiki/concepts/insight-0.md", "replaced": True}]
    results[6]["status"] = "answer_failed"
    results[6]["answer"] = ""
    for item, result in zip(source, results):
        run_one(event(item), s3=cloud, bucket="bucket", answer_callback=lambda request, result=result: result)
    cloud.objects[f"{PREFIX}/claims/q7.json"] = json.dumps(event(source[7])).encode()

    summary = progress({"run_id": RUN}, s3=cloud, bucket="bucket")
    assert summary["total"] == 10 and summary["completed"] == 7 and summary["claimed"] == 1
    assert summary["not_started"] == 2 and summary["answer_saved"] == 4
    assert summary["statuses"] == {"answer_ready": 5, "answer_partial": 1, "answer_failed": 1}
    assert summary["total_estimated_usd"] == 1.5 and summary["cost_unavailable"] == 1
    assert summary["total_seconds"] == 213.5
    assert summary["pages_created"] == 6 and summary["pages_updated"] == 1 and summary["distinct_pages"] == 6
    assert len(summary["recent"]) == 5 and summary["recent"][0]["id"] == "q6"
    assert "A substantive answer" not in json.dumps(summary)


def test_existing_completed_window_still_advances_to_unstarted_questions():
    source = rows(25)
    cloud = CampaignS3(source)
    for number, item in enumerate(source[:24]):
        run_one(event(item), s3=cloud, bucket="bucket", answer_callback=lambda request, number=number: answer(number))
    planned = plan_batch({"run_id": RUN}, s3=cloud, bucket="bucket")
    assert planned["items"] == [] and len(planned["skipped"]) == 24
    assert planned["next_offset"] == 24 and planned["done"] is False
    assert plan_batch({"run_id": RUN, "offset": 24}, s3=cloud, bucket="bucket")["items"] == [event(source[24])]


def test_changed_question_identity_is_not_silently_skipped():
    source = rows(1)
    cloud = CampaignS3(source)
    run_one(event(source[0]), s3=cloud, bucket="bucket", answer_callback=lambda request: answer())
    source[0]["question"] = "A different question under the same id?"
    cloud.objects[MANIFEST] = json.dumps(source).encode()
    with pytest.raises(ValueError, match="identity changed"):
        plan_batch({"run_id": RUN}, s3=cloud, bucket="bucket")


@pytest.mark.parametrize("override", [{"run_id": "../outside"}, {"manifest_key": "runs/other/manifest.json"},
                                      {"offset": -1}, {"batch_size": 0}, {"concurrency": 0}])
def test_invalid_campaign_location_or_window_is_rejected(override):
    cloud = CampaignS3(rows(1))
    with pytest.raises(ValueError):
        plan_batch({"run_id": RUN, **override}, s3=cloud, bucket="bucket")
    assert cloud.writes == []


def test_duplicate_ids_and_unreadable_manifest_fail_before_dispatch():
    cloud = CampaignS3(rows(1) * 2)
    with pytest.raises(ValueError, match="unique"):
        plan_batch({"run_id": RUN}, s3=cloud, bucket="bucket")
    cloud.denied.add(MANIFEST)
    with pytest.raises(ClientError, match="AccessDenied"):
        progress({"run_id": RUN}, s3=cloud, bucket="bucket")
    assert cloud.writes == []
