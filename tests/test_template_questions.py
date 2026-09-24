from __future__ import annotations

from io import BytesIO
import json
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest

from byeori.question_campaign import plan_batch
from test_template_synthesis import TEMPLATE, _definition


def _assert_reachable_scope(machine):
    states = machine["States"]
    pending = [machine["StartAt"]]
    reached = set()
    while pending:
        name = pending.pop()
        assert name in states, f"State target {name!r} is outside its state-machine scope"
        if name in reached:
            continue
        reached.add(name)
        state = states[name]
        assert not (state.get("End") and "Next" in state), name
        for field in ("Next", "Default"):
            if field in state:
                pending.append(state[field])
        pending.extend(choice["Next"] for choice in state.get("Choices", []))
        pending.extend(catch["Next"] for catch in state.get("Catch", []))
        if "ItemProcessor" in state:
            _assert_reachable_scope(state["ItemProcessor"])
    assert reached == set(states), f"Unreachable states: {set(states) - reached}"


def test_question_workflow_has_reachable_states_and_stops_when_manifest_is_exhausted():
    machine = _definition("QuestionStateMachine")
    _assert_reachable_scope(machine)
    states = machine["States"]
    assert machine["StartAt"] == "InitialIndex"
    assert states["InitialIndex"]["Parameters"]["Payload"] == {"action": "build_index"}
    assert states["InitialIndex"]["ResultPath"] is None
    assert states["InitialIndex"]["Next"] == "LoadBatch"
    assert states["HaveQuestions"]["Choices"] == [
        {"Variable": "$.done", "BooleanEquals": True, "Next": "Finished"}]
    assert states["HaveQuestions"]["Default"] == "AskQuestions"
    assert states["Finished"]["Type"] == "Task" and states["Finished"]["End"] is True
    assert states["Finished"]["Parameters"]["Payload"]["action"] == "question_campaign_progress"


@pytest.mark.parametrize("total,offset,ask", [(1, 0, True), (24, 0, True), (26, 24, True),
                                            (1437, 1416, True), (1437, 1437, False)])
def test_actual_batch_plan_does_not_skip_the_final_questions(total, offset, ask):
    key = "runs/questions/test-run/manifest.json"
    manifest = [{"id": f"q{index}", "question": f"Question {index}", "origin": f"wiki/questions/q{index}.md"}
                for index in range(total)]
    def get_object(**request):
        if request["Key"] == key:
            return {"Body": BytesIO(json.dumps(manifest).encode())}
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    batch = plan_batch({"run_id": "test-run", "manifest_key": key, "offset": offset},
                       s3=SimpleNamespace(get_object=get_object), bucket="b")
    choice = _definition("QuestionStateMachine")["States"]["HaveQuestions"]
    branch = choice["Choices"][0]
    value = batch[branch["Variable"].removeprefix("$.")]
    next_state = branch["Next"] if value == branch["BooleanEquals"] else choice["Default"]
    assert (next_state == "AskQuestions") is ask
    assert bool(batch["items"]) is ask


def test_question_batches_finish_before_index_refresh_and_advance_without_losing_identity():
    states = _definition("QuestionStateMachine")["States"]
    batch = states["AskQuestions"]
    assert batch["MaxConcurrency"] == 6
    assert batch["ItemProcessor"]["ProcessorConfig"]["Mode"] == "INLINE"
    assert batch["ItemsPath"] == "$.items"
    assert batch["ResultPath"] is None
    assert batch["Next"] == "RefreshIndex"
    refresh = states["RefreshIndex"]
    assert refresh["Parameters"]["Payload"] == {"action": "build_index"}
    assert refresh["ResultPath"] is None and refresh["Next"] == "Advance"
    assert states["Advance"]["Parameters"] == {
        "run_id.$": "$.run_id", "manifest_key.$": "$.manifest_key", "offset.$": "$.next_offset"}
    assert states["Advance"]["Next"] == "LoadBatch"


def test_only_pre_execution_throttling_retries_a_scientific_question():
    states = _definition("QuestionStateMachine")["States"]
    batch = states["AskQuestions"]
    assert "Retry" not in batch, "Retrying the Map can replay questions that already wrote pages"
    loop = batch["ItemProcessor"]["States"]
    task = loop["AskOne"]
    assert len(task["Retry"]) == 1
    retry = task["Retry"][0]
    assert retry["ErrorEquals"] == ["Lambda.TooManyRequestsException"]
    assert 0 < retry["MaxAttempts"] <= 3
    assert retry["IntervalSeconds"] >= 1
    assert task["Catch"] == [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.invocation_error", "Next": "KeepFailure"}]
    assert loop["KeepFailure"]["Type"] == "Pass" and loop["KeepFailure"]["End"] is True


def test_workflow_discards_lambda_envelopes_and_carries_only_batch_checkpoint_between_batches():
    states = _definition("QuestionStateMachine")["States"]
    load = states["LoadBatch"]
    assert load["OutputPath"] == "$.Payload"
    assert load["Parameters"]["Payload"] == {
        "action": "question_campaign_plan", "run_id.$": "$.run_id",
        "manifest_key.$": "$.manifest_key", "offset.$": "$.offset"}
    task = states["AskQuestions"]["ItemProcessor"]["States"]["AskOne"]
    assert task["ResultPath"] is None, "Answer body and trace belong in S3, not the Map result array"
    assert set(task["Parameters"]["Payload"]) == {
        "action", "run_id.$", "manifest_key.$", "id.$", "question.$", "origin.$"}
    assert set(states["Advance"]["Parameters"]) == {"run_id.$", "manifest_key.$", "offset.$"}


def test_question_workflow_uses_existing_ingest_function_and_existing_lambda_invoke_role():
    text = TEMPLATE.read_text()
    resource = text.split("  QuestionStateMachine:\n", 1)[1].split("  AuditBucket:\n", 1)[0]
    role = text.split("  NotesStateMachineRole:\n", 1)[1].split("  NotesStateMachine:\n", 1)[0]
    assert "RoleArn: !GetAtt NotesStateMachineRole.Arn" in resource
    assert resource.count('"FunctionName": "${IngestFunction.Arn}"') == 5
    assert "Action: lambda:InvokeFunction" in role
    assert 'Resource: !Sub "${IngestFunction.Arn}*"' in role
    assert "Principal: {Service: states.amazonaws.com}" in role
    assert "  QuestionStateMachineArn:" in text
