from __future__ import annotations

import json
import re
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / "infra" / "template.yaml"


def _definition(resource: str) -> dict:
    lines = TEMPLATE.read_text().splitlines()
    start = lines.index(f"  {resource}:")
    begin = next(i for i in range(start, len(lines)) if lines[i].strip() == "DefinitionString: !Sub |") + 1
    block = []
    for line in lines[begin:]:
        if line.strip() and not line.startswith("        "):
            break
        block.append(line[8:])
    text = re.sub(r"\$\{(?:Synthesis|Notes)Concurrency\}", "4", "\n".join(block))
    text = re.sub(r"\$\{[^}]+\}", "X", text)
    return json.loads(text)


def _reachable(states: dict) -> set[str]:
    names = set()
    for name, state in states.items():
        names.add(name)
        if "ItemProcessor" in state:
            names |= _reachable(state["ItemProcessor"]["States"])
    return names


def _targets(states: dict) -> list[tuple[str, str]]:
    out = []
    for name, state in states.items():
        for key in ("Next", "Default"):
            if key in state:
                out.append((name, state[key]))
        for choice in state.get("Choices", []):
            out.append((name, choice["Next"]))
        for catch in state.get("Catch", []):
            out.append((name, catch["Next"]))
        if "ItemProcessor" in state:
            out += _targets(state["ItemProcessor"]["States"])
    return out


def test_synthesis_state_machine_is_well_formed() -> None:
    machine = _definition("SynthesisStateMachine")
    states = machine["States"]
    assert machine["StartAt"] == "ResolveScope"
    for name in ("ResolveScope", "PlanSubtopics", "ConceptsWanted", "PlanConcepts", "ConceptPlanDone", "PlanOnly", "Planned", "WriteConcepts",
                 "PlanSubtopicPages", "WriteSubtopics", "CategoryPages", "BuildIndex", "PlanFailures", "RetryFailures",
                 "RetryCategoryPages", "RebuildIndex"):
        assert name in states, name
    names = _reachable(states)
    for source, target in _targets(states):
        assert target in names, f"{source} -> {target}"
    loop = states["WriteConcepts"]["ItemProcessor"]["States"]
    assert loop["Done"]["Choices"][0] == {"Variable": "$.result.status", "StringEquals": "partial", "Next": "Page"}
    assert states["WriteConcepts"]["ToleratedFailurePercentage"] == 100
    assert states["PlanOnly"]["Choices"][0]["Variable"] == "$.plan_only"
    assert states["ConceptPlanDone"]["Choices"][0] == {"Variable": "$.concept_plan.status", "StringEquals": "partial", "Next": "PlanConcepts"}
    assert states["RetryFailures"]["Next"] == "RetryCategoryPages" and states["RebuildIndex"]["End"] is True


def test_synthesis_function_is_packaged_from_src() -> None:
    text = TEMPLATE.read_text()
    assert "Handler: byeori.synthesis_lambda.handler" in text
    assert "Code: ../src" in text
    for output in ("SynthesisFunctionName", "SynthesisStateMachineArn"):
        assert f"  {output}:" in text
    assert "SynthesisConcurrency:" in text and "SynthesisMaxPages:" in text


def test_category_planning_resumes_without_losing_category_or_resetting_checkpoint():
    states = _definition("SynthesisStateMachine")["States"]
    group = states["PlanSubtopics"]
    assert group["ItemSelector"]["resume"] is None
    loop = group["ItemProcessor"]["States"]
    task = loop["PlanCategory"]
    assert task["Parameters"]["Payload"]["resume.$"] == "$.resume"
    assert task["Parameters"]["Payload"]["run_id.$"] == "$$.Execution.Id"
    assert task["ResultPath"] == "$.category_plan"
    assert task["ResultSelector"]["checkpoint.$"] == "$.Payload.checkpoint"
    assert task["Next"] == "CategoryPlanDone" and "End" not in task
    assert loop["CategoryPlanDone"]["Choices"] == [
        {"Variable": "$.category_plan.status", "StringEquals": "partial", "Next": "ResumeCategoryPlan"}]
    assert loop["ResumeCategoryPlan"]["Parameters"] == {
        "category.$": "$.category", "replan": False, "resume.$": "$.category_plan.checkpoint"}
    assert loop["ResumeCategoryPlan"]["Next"] == "PlanCategory"


def test_category_planning_failure_blocks_concepts_and_page_generation():
    states = _definition("SynthesisStateMachine")["States"]
    assert states["PlanSubtopics"]["Next"] == "CheckCategoryPlans"
    check = states["CheckCategoryPlans"]
    assert check["Parameters"]["Payload"] == {
        "action": "category_plan_status", "categories.$": "$.scope_info.categories", "run_id.$": "$$.Execution.Id"}
    assert check["ResultPath"] == "$.category_plans"
    gate = states[check["Next"]]
    assert gate["Choices"] == [
        {"Variable": "$.category_plans.ready", "BooleanEquals": True, "Next": "ConceptsWanted"}]
    failure = states[gate["Default"]]
    assert failure["Type"] == "Fail" and failure["Error"] == "CategoryPlanningIncomplete"
    assert "Next" not in failure


def test_note_retry_rebuilds_index_after_recovered_notes():
    states = _definition("NotesStateMachine")["States"]
    assert states["RetryFailures"]["Next"] == "RebuildAfterRetry"
    assert states["RebuildAfterRetry"]["Parameters"]["Payload"] == {"action": "build_index"}
    assert states["RebuildAfterRetry"]["End"] is True


def test_openalex_loop_keeps_identity_and_obeys_server_wait():
    machine = _definition("OpenAlexMatchStateMachine")
    states = machine["States"]
    assert states["Initialize"]["Parameters"]["run_id.$"] == "$$.Execution.Name"
    task = states["MatchBatch"]
    assert task["Parameters"]["Payload"]["action"] == "openalex_match_batch"
    assert task["Parameters"]["Payload"]["run_id.$"] == "$.run_id"
    assert task["ResultPath"] == "$.progress"
    retry = task["Retry"][0]
    assert {"States.TaskFailed", "States.Timeout"} <= set(retry["ErrorEquals"])
    assert 0 < retry["MaxAttempts"] <= 3
    assert states["WaitForBudgetWindow"]["SecondsPath"] == "$.progress.result.wait_seconds"
    assert states["WaitForBudgetWindow"]["Next"] == "MatchBatch"
    assert states["Done"]["Choices"][0]["Variable"] == "$.progress.result.done"


def test_permanent_version_protection_is_declared_and_retained():
    text = TEMPLATE.read_text()
    policy = TEMPLATE.with_name("storage-protection.yaml").read_text()
    assert "DataBucketPolicy:" not in text
    assert "ExistingBucketName" in policy
    assert "DataBucket.Arn" not in policy
    assert "DeletionPolicy: Retain" in policy and "UpdateReplacePolicy: Retain" in policy
    for action in ("s3:DeleteObjectVersion", "s3:DeleteBucket", "s3:PutBucketVersioning"):
        assert action in policy
    assert policy.count('Principal: "*"') == 2


def test_extraction_role_can_distinguish_missing_report_without_delete_permission():
    text = TEMPLATE.read_text().split("  ExtractionTaskRole:", 1)[1].split("  ExtractionTaskDefinition:", 1)[0]
    assert "Action: s3:ListBucket\n                Resource: !GetAtt DataBucket.Arn" in text
    assert "s3:Delete" not in text


def test_the_index_is_rebuilt_on_a_schedule_at_a_quiet_hour():
    """Nothing rebuilt the index automatically, so a new page stayed unsearchable until someone ran
    the command. A rebuild also changes the object's ETag, which drops the /tmp copy in every warm
    Lambda, so the hour it runs decides who pays the cold fetch."""
    template = (Path(__file__).parents[1] / "infra" / "template.yaml").read_text()
    assert "IndexRebuildRule:" in template and "AWS::Events::Rule" in template
    assert "'{\"action\": \"build_index\"}'" in template, "the rule takes no arguments, as from a phone"
    assert "IndexRebuildPermission:" in template and "Principal: events.amazonaws.com" in template, \
        "EventBridge cannot invoke the function without the permission"
    assert 'Default: "cron(0 19 * * ? *)"' in template, "19:00 UTC is 04:00 KST, outside the lab's day"
    assert "HasIndexRebuildSchedule: !Not [!Equals [!Ref IndexRebuildSchedule, \"\"]]" in template, \
        "an empty schedule turns the rule off rather than forcing a fake cron"
    assert template.count("\nConditions:\n") == 1, "one Conditions block, or CloudFormation keeps the last"
