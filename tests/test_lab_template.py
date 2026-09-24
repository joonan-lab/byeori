"""Shape and permission checks for the separate student stack (infra/lab-template.json).

The template is plain JSON so these tests need only the stdlib plus the answer-worker lease
constant from ``byeori.lab_policy``, which the answer queue's visibility timeout must
cover. The deploy script is inspected as text and syntax-checked with ``bash -n``; it is never
executed here. Its parameter block alone is run in a throwaway shell, which only builds an array
of strings and reaches no AWS API, because whether a setting is sent or carried over cannot be
read off the text.
"""
from __future__ import annotations

import json
import os
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

from byeori.lab_policy import LEASE_SECONDS


ROOT = Path(__file__).parents[1]
TEMPLATE_PATH = ROOT / "infra/lab-template.json"
SCRIPT_PATH = ROOT / "scripts/deploy_lab.sh"
TEMPLATE = json.loads(TEMPLATE_PATH.read_text())
PARAMETERS = TEMPLATE["Parameters"]
RESOURCES = TEMPLATE["Resources"]
BUCKET_ARN = "arn:${AWS::Partition}:s3:::${DataBucket}/"
JEV_PARAMETER_ARN = (
    "arn:${AWS::Partition}:ssm:${AWS::Region}:${AWS::AccountId}:parameter${JevApiKeyParameter}"
)
FUNCTIONS = {
    "GatewayFunction": "gateway",
    "AnswerFunction": "answer",
    "TriageFunction": "triage",
    "OutboxFunction": "outbox",
    "ResearchFunction": "research",
}
ROLES = ["GatewayRole", "AnswerRole", "TriageRole", "OutboxRole", "ResearchRole"]
# Reserved concurrency keeps the student stack out of the campaign's Lambda pool: workers stay
# at 5 or below (research at 2, matching its event source's MaximumConcurrency) and the gateway
# takes 10 so one student cannot drive unbounded concurrent 60 s invocations against the account.
GATEWAY_RESERVED_CONCURRENCY = 40   # 25 members' simultaneous first calls throttled at 10 on 2026-09-21
WORKER_RESERVED_CONCURRENCY_CAP = 5
RESEARCH_RESERVED_CONCURRENCY = 2
# The worker claims its job some seconds after SQS delivers the message (index download first),
# so the lease ends later than a visibility window equal to LEASE_SECONDS would; the margin keeps
# the first redelivery from landing on a still-live lease and being spent as lease_held.
ANSWER_PRE_CLAIM_MARGIN_SECONDS = 120
QUEUE_TIMEOUT_MARGIN_SECONDS = 60
WIKI_PREFIXES = {BUCKET_ARN + "wiki/*", BUCKET_ARN + "papers/*", BUCKET_ARN + "index/*"}
LISTABLE_PREFIXES = ["runs/lab-questions/*", "wiki/*"]   # ListBucket condition shared by the four S3-reading roles
# The approved research worker (P4) publishes through the campaign's publisher inside an approval's
# scope: these are the only wiki objects it may put, plus the engine's traces and its receipts.
RESEARCH_PUT_PREFIXES = {
    BUCKET_ARN + "wiki/concepts/*", BUCKET_ARN + "wiki/overviews/*", BUCKET_ARN + "wiki/questions/*",
    BUCKET_ARN + "wiki/sources/*", BUCKET_ARN + "wiki/index.md", BUCKET_ARN + "wiki/indexes/*",
    BUCKET_ARN + "runs/agents/*", BUCKET_ARN + "runs/lab-questions/*",
}
RESEARCH_MODULE_ROLE = "ResearchRole"
# The answer worker keeps each answered question as Markdown here (user, 2026-09-22). The prefix is
# skipped by the index builder, so a question never competes with a source note for a result, and
# it is the only wiki object this role may put: everything else under wiki/ stays read-only to it.
ANSWER_PAGE_PREFIX_ARN = BUCKET_ARN + "wiki/lab-questions/*"
ANSWER_PUT_PREFIXES = {BUCKET_ARN + "runs/lab-questions/*", ANSWER_PAGE_PREFIX_ARN}
ANSWER_MODULE_ROLE = "AnswerRole"
# Console writes by administrators reach only the member registry, principal pointers and budget
# caps; every other partition of the control table is written by the functions alone.
ADMIN_REGISTRY_CONDITION = {
    "ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["MEMBER#*", "PRINCIPAL#*", "BUDGET#*"]},
}
ENV_KEYS = {
    "LAB_TABLE", "LAB_BUCKET", "LAB_INDEX_KEY", "LAB_ANSWER_MODEL_ID", "LAB_ANSWER_REASONING",
    "LAB_JEV_PARAMETER", "LAB_ANSWER_QUEUE_URL", "LAB_TRIAGE_QUEUE_URL", "LAB_RESEARCH_QUEUE_URL",
    "LAB_POLICY_REVISION", "LAB_HANDLER", "LAB_RESEARCH_CONSUMER_ENABLED", "LAB_RESEARCH_MODEL_ID",
    "LAB_RESEARCH_REASONING",
}


def values(value):
    return value if isinstance(value, list) else [value]


def flatten(value):
    """Render an IAM Resource entry as a comparable string."""
    if isinstance(value, dict):
        if "Fn::Sub" in value and isinstance(value["Fn::Sub"], str):
            return value["Fn::Sub"]
        if "Fn::GetAtt" in value:
            return "GetAtt:" + ".".join(value["Fn::GetAtt"])
        if "Ref" in value:
            return "Ref:" + value["Ref"]
    return value


def resources(statement):
    return [flatten(value) for value in values(statement["Resource"])]


def statements(role_name):
    role = RESOURCES[role_name]["Properties"]
    return [statement for policy in role["Policies"]
            for statement in policy["PolicyDocument"]["Statement"]]


def matching(role_name, effect, action):
    return [statement for statement in statements(role_name) if statement["Effect"] == effect
            and any(fnmatchcase(action.lower(), item.lower())
                    for item in values(statement["Action"]))]


def allowed_actions(role_name):
    return {action for statement in statements(role_name) if statement["Effect"] == "Allow"
            for action in values(statement["Action"])}


def walk(value, path=()):
    """Yield (path, key, value) for every mapping entry in the template."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield path, key, item
            yield from walk(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, path + (index,))


def function_properties(name):
    return RESOURCES[name]["Properties"]


# --- parameters -------------------------------------------------------------------------------


def test_parameters_are_the_planned_set_with_fixed_defaults():
    assert set(PARAMETERS) == {
        "DataBucket", "ArtifactBucket", "ArtifactKey", "JevApiKeyParameter",
        "ParameterKmsKeyArn", "AnswerModelId", "AnswerReasoning", "PolicyRevision",
        "ResearchConsumerEnabled", "ResearchModelId", "ResearchReasoning", "OutboxScheduleEnabled",
    }
    for name in ("DataBucket", "ArtifactBucket", "ArtifactKey"):
        assert "Default" not in PARAMETERS[name], name
    assert PARAMETERS["JevApiKeyParameter"]["Default"] == "/byeori/jev/api-key"
    assert PARAMETERS["JevApiKeyParameter"]["AllowedValues"] == ["/byeori/jev/api-key"]
    assert PARAMETERS["AnswerModelId"]["Default"] == "global.anthropic.claude-opus-5"
    assert PARAMETERS["AnswerReasoning"]["Default"] == "high"
    assert PARAMETERS["PolicyRevision"]["Default"] == "2026-09-21-v1"
    # The research consumer ships inert: the deploy flips the flag deliberately, never by default.
    assert PARAMETERS["ResearchConsumerEnabled"]["Default"] == "false"
    assert PARAMETERS["ResearchConsumerEnabled"]["AllowedValues"] == ["true", "false"]
    assert PARAMETERS["ResearchModelId"]["Default"] == "global.anthropic.claude-opus-5"
    assert PARAMETERS["ResearchReasoning"]["Default"] == "high"
    assert all(parameter["Type"] == "String" for parameter in PARAMETERS.values())


def test_research_enabled_condition_compares_the_flag_with_true():
    assert TEMPLATE["Conditions"] == {
        "ResearchEnabled": {"Fn::Equals": [{"Ref": "ResearchConsumerEnabled"}, "true"]},
        "OutboxScheduleOn": {"Fn::Equals": [{"Ref": "OutboxScheduleEnabled"}, "true"]},
    }
    conditional = {name for name, resource in RESOURCES.items() if "Condition" in resource}
    assert conditional == {"ResearchEventSource"}
    assert RESOURCES["ResearchEventSource"]["Condition"] == "ResearchEnabled"


# --- resource inventory -----------------------------------------------------------------------


def test_resources_are_exactly_the_planned_inventory():
    assert set(RESOURCES) == {
        "ControlTable",
        "AnswerQueue", "AnswerDeadLetterQueue", "TriageQueue", "TriageDeadLetterQueue",
        "ResearchQueue", "ResearchDeadLetterQueue",
        "GatewayLogGroup", "AnswerLogGroup", "TriageLogGroup", "OutboxLogGroup", "ResearchLogGroup",
        "GatewayRole", "AnswerRole", "TriageRole", "OutboxRole", "ResearchRole",
        "GatewayFunction", "GatewayUrl", "GatewayUrlPermission",
        "AnswerFunction", "AnswerEventSource", "TriageFunction", "TriageEventSource",
        "OutboxFunction", "OutboxRule", "OutboxPermission", "ResearchFunction", "ResearchEventSource",
        "StudentAccessPolicy", "AdminAccessPolicy",
    }


def test_control_table_is_a_single_on_demand_table_with_recovery_and_retention():
    table = RESOURCES["ControlTable"]
    assert table["Type"] == "AWS::DynamoDB::Table"
    assert table["DeletionPolicy"] == "Retain"
    assert table["UpdateReplacePolicy"] == "Retain"
    properties = table["Properties"]
    assert properties["BillingMode"] == "PAY_PER_REQUEST"
    assert properties["KeySchema"] == [
        {"AttributeName": "pk", "KeyType": "HASH"},
        {"AttributeName": "sk", "KeyType": "RANGE"},
    ]
    assert {item["AttributeName"]: item["AttributeType"] for item in properties["AttributeDefinitions"]} == {
        "pk": "S", "sk": "S",
    }
    assert properties["PointInTimeRecoverySpecification"] == {"PointInTimeRecoveryEnabled": True}
    assert properties["DeletionProtectionEnabled"] is True
    assert "GlobalSecondaryIndexes" not in properties and "LocalSecondaryIndexes" not in properties
    assert "TimeToLiveSpecification" not in properties


@pytest.mark.parametrize("queue, dead_letter, receives", [
    # An answer worker that dies at its 900 s timeout meets a still-live lease on the first
    # redelivery (lease_held), so three receives left a single real recovery attempt; five
    # leave several before the message reaches the dead-letter queue.
    ("AnswerQueue", "AnswerDeadLetterQueue", 5),
    ("TriageQueue", "TriageDeadLetterQueue", 3),
    ("ResearchQueue", "ResearchDeadLetterQueue", 3),
])
def test_each_work_queue_redrives_to_its_own_dead_letter_queue(queue, dead_letter, receives):
    assert RESOURCES[queue]["Type"] == "AWS::SQS::Queue"
    assert RESOURCES[dead_letter]["Type"] == "AWS::SQS::Queue"
    redrive = RESOURCES[queue]["Properties"]["RedrivePolicy"]
    assert redrive["deadLetterTargetArn"] == {"Fn::GetAtt": [dead_letter, "Arn"]}
    assert redrive["maxReceiveCount"] == receives
    assert "RedrivePolicy" not in RESOURCES[dead_letter]["Properties"]


def test_answer_queue_outlives_the_answer_function_timeout():
    visibility = RESOURCES["AnswerQueue"]["Properties"]["VisibilityTimeout"]
    assert function_properties("AnswerFunction")["Timeout"] <= 900 < LEASE_SECONDS
    assert visibility >= LEASE_SECONDS + ANSWER_PRE_CLAIM_MARGIN_SECONDS
    assert visibility >= 1080
    for queue, function in (("TriageQueue", "TriageFunction"), ("ResearchQueue", "ResearchFunction")):
        timeout = function_properties(function)["Timeout"]
        assert RESOURCES[queue]["Properties"]["VisibilityTimeout"] >= timeout + QUEUE_TIMEOUT_MARGIN_SECONDS
    # The research worker claims under the same lease as the answer worker and runs to the 900 s limit.
    assert function_properties("ResearchFunction")["Timeout"] == 900
    assert RESOURCES["ResearchQueue"]["Properties"]["VisibilityTimeout"] >= LEASE_SECONDS + ANSWER_PRE_CLAIM_MARGIN_SECONDS
    assert RESOURCES["ResearchQueue"]["Properties"]["VisibilityTimeout"] >= 1080


# --- functions --------------------------------------------------------------------------------


@pytest.mark.parametrize("name, handler_kind", sorted(FUNCTIONS.items()))
def test_every_function_uses_the_artifact_zip_and_the_lab_handler(name, handler_kind):
    assert RESOURCES[name]["Type"] == "AWS::Lambda::Function"
    properties = function_properties(name)
    assert properties["Code"] == {"S3Bucket": {"Ref": "ArtifactBucket"}, "S3Key": {"Ref": "ArtifactKey"}}
    assert properties["Runtime"] == "python3.12"
    assert properties["Architectures"] == ["arm64"]
    assert properties["Handler"] == "byeori.lab_lambda.handler"
    assert properties["Timeout"] <= 900
    cap = GATEWAY_RESERVED_CONCURRENCY if name == "GatewayFunction" else WORKER_RESERVED_CONCURRENCY_CAP
    assert properties.get("ReservedConcurrentExecutions", 0) <= cap
    role = name.replace("Function", "Role")
    assert properties["Role"] == {"Fn::GetAtt": [role, "Arn"]}
    log_group = name.replace("Function", "LogGroup")
    assert RESOURCES[name]["DependsOn"] == log_group
    assert RESOURCES[log_group]["Type"] == "AWS::Logs::LogGroup"
    assert RESOURCES[log_group]["Properties"]["LogGroupName"] == {
        "Fn::Sub": "/aws/lambda/${AWS::StackName}-" + handler_kind,
    }
    assert properties["FunctionName"] == {"Fn::Sub": "${AWS::StackName}-" + handler_kind}
    variables = properties["Environment"]["Variables"]
    assert set(variables) == ENV_KEYS
    assert variables["LAB_HANDLER"] == handler_kind
    assert variables["LAB_TABLE"] == {"Ref": "ControlTable"}
    assert variables["LAB_BUCKET"] == {"Ref": "DataBucket"}
    assert variables["LAB_INDEX_KEY"] == "index/wiki-index-v2.sqlite3"
    assert variables["LAB_ANSWER_MODEL_ID"] == {"Ref": "AnswerModelId"}
    assert variables["LAB_ANSWER_REASONING"] == {"Ref": "AnswerReasoning"}
    assert variables["LAB_JEV_PARAMETER"] == {"Ref": "JevApiKeyParameter"}
    assert variables["LAB_POLICY_REVISION"] == {"Ref": "PolicyRevision"}
    assert variables["LAB_ANSWER_QUEUE_URL"] == {"Ref": "AnswerQueue"}
    assert variables["LAB_TRIAGE_QUEUE_URL"] == {"Ref": "TriageQueue"}
    assert variables["LAB_RESEARCH_QUEUE_URL"] == {"Ref": "ResearchQueue"}
    # One parameter switches the relay and the consumer together; the code path stays inert at "false".
    assert variables["LAB_RESEARCH_CONSUMER_ENABLED"] == {"Ref": "ResearchConsumerEnabled"}
    assert variables["LAB_RESEARCH_MODEL_ID"] == {"Ref": "ResearchModelId"}
    assert variables["LAB_RESEARCH_REASONING"] == {"Ref": "ResearchReasoning"}


def test_gateway_reserves_bounded_concurrency_so_one_student_cannot_flood_the_account_pool():
    assert function_properties("GatewayFunction")["ReservedConcurrentExecutions"] == GATEWAY_RESERVED_CONCURRENCY
    reserved = {name: function_properties(name).get("ReservedConcurrentExecutions") for name in FUNCTIONS}
    assert all(isinstance(value, int) for value in reserved.values()), reserved
    assert reserved["ResearchFunction"] == RESEARCH_RESERVED_CONCURRENCY
    assert max(value for name, value in reserved.items() if name != "GatewayFunction") <= WORKER_RESERVED_CONCURRENCY_CAP


def test_gateway_url_requires_iam_auth_and_is_shared_only_with_this_account():
    url = RESOURCES["GatewayUrl"]
    assert url["Type"] == "AWS::Lambda::Url"
    assert url["Properties"]["AuthType"] == "AWS_IAM"
    assert url["Properties"]["TargetFunctionArn"] == {"Fn::GetAtt": ["GatewayFunction", "Arn"]}
    permission = RESOURCES["GatewayUrlPermission"]
    assert permission["Type"] == "AWS::Lambda::Permission"
    assert permission["Properties"]["Action"] == "lambda:InvokeFunctionUrl"
    assert permission["Properties"]["FunctionName"] == {"Ref": "GatewayFunction"}
    assert permission["Properties"]["FunctionUrlAuthType"] == "AWS_IAM"
    assert permission["Properties"]["Principal"] == {"Ref": "AWS::AccountId"}
    urls = [name for name, resource in RESOURCES.items() if resource["Type"] == "AWS::Lambda::Url"]
    assert urls == ["GatewayUrl"]
    public = [name for name, resource in RESOURCES.items()
              if resource["Type"] == "AWS::Lambda::Permission"
              and resource["Properties"].get("Principal") == "*"]
    assert public == []


def test_each_worker_consumes_its_own_queue_and_the_research_mapping_is_conditional():
    mappings = {name: resource["Properties"] for name, resource in RESOURCES.items()
                if resource["Type"] == "AWS::Lambda::EventSourceMapping"}
    assert set(mappings) == {"AnswerEventSource", "TriageEventSource", "ResearchEventSource"}
    assert mappings["AnswerEventSource"]["FunctionName"] == {"Ref": "AnswerFunction"}
    assert mappings["AnswerEventSource"]["EventSourceArn"] == {"Fn::GetAtt": ["AnswerQueue", "Arn"]}
    assert mappings["TriageEventSource"]["FunctionName"] == {"Ref": "TriageFunction"}
    assert mappings["TriageEventSource"]["EventSourceArn"] == {"Fn::GetAtt": ["TriageQueue", "Arn"]}
    assert mappings["ResearchEventSource"]["FunctionName"] == {"Ref": "ResearchFunction"}
    assert mappings["ResearchEventSource"]["EventSourceArn"] == {"Fn::GetAtt": ["ResearchQueue", "Arn"]}
    for mapping in mappings.values():
        assert mapping["BatchSize"] == 1
        assert mapping["Enabled"] is True
        assert mapping["FunctionResponseTypes"] == ["ReportBatchItemFailures"]
    assert mappings["ResearchEventSource"]["ScalingConfig"] == {"MaximumConcurrency": RESEARCH_RESERVED_CONCURRENCY}
    # The research function is wired to nothing but its (conditional) event source and its own log group.
    research_targets = [path for path, key, value in walk(RESOURCES)
                        if key == "Ref" and value == "ResearchFunction"]
    assert research_targets == [("ResearchEventSource", "Properties", "FunctionName")], research_targets
    research_attributes = [path for path, key, value in walk(RESOURCES)
                           if key == "Fn::GetAtt" and value[0] == "ResearchFunction"]
    assert research_attributes == [], research_attributes
    assert RESOURCES["ResearchEventSource"]["Condition"] == "ResearchEnabled"
    assert "Condition" not in RESOURCES["AnswerEventSource"] and "Condition" not in RESOURCES["TriageEventSource"]
    assert function_properties("ResearchFunction")["ReservedConcurrentExecutions"] == RESEARCH_RESERVED_CONCURRENCY


def test_outbox_schedule_follows_the_parameter_and_defaults_to_enabled():
    # Disabled while the service was being built; the user opened it on 2026-09-22, so the default is
    # now true and LAB_OUTBOX_SCHEDULE_ENABLED=false parks the relay again without a template edit.
    parameter = PARAMETERS["OutboxScheduleEnabled"]
    assert parameter["Type"] == "String" and parameter["Default"] == "true"
    assert sorted(parameter["AllowedValues"]) == ["false", "true"]
    assert TEMPLATE["Conditions"]["OutboxScheduleOn"] == {"Fn::Equals": [{"Ref": "OutboxScheduleEnabled"}, "true"]}
    rule = RESOURCES["OutboxRule"]
    assert rule["Type"] == "AWS::Events::Rule"
    assert rule["Properties"]["State"] == {"Fn::If": ["OutboxScheduleOn", "ENABLED", "DISABLED"]}
    assert "add_optional LAB_OUTBOX_SCHEDULE_ENABLED OutboxScheduleEnabled" in SCRIPT_PATH.read_text()
    assert "ScheduleExpression" in rule["Properties"]
    assert rule["Properties"]["Targets"] == [
        {"Arn": {"Fn::GetAtt": ["OutboxFunction", "Arn"]}, "Id": "outbox"},
    ]
    permission = RESOURCES["OutboxPermission"]["Properties"]
    assert permission["Action"] == "lambda:InvokeFunction"
    assert permission["FunctionName"] == {"Ref": "OutboxFunction"}
    assert permission["Principal"] == "events.amazonaws.com"
    assert permission["SourceArn"] == {"Fn::GetAtt": ["OutboxRule", "Arn"]}


# --- roles ------------------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_every_role_denies_wiki_writes_and_lambda_invocation(role):
    assert RESOURCES[role]["Type"] == "AWS::IAM::Role"
    trust = RESOURCES[role]["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert trust == [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                      "Action": "sts:AssumeRole"}]
    for action in ("s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:PutObjectAcl",
                   "s3:PutObjectVersionAcl", "s3:PutObjectTagging", "s3:PutObjectVersionTagging", "s3:RestoreObject",
                   "s3:AbortMultipartUpload", "s3:PutObjectRetention", "s3:PutObjectLegalHold"):
        denials = matching(role, "Deny", action)
        denied = {resource for statement in denials for resource in resources(statement)}
        # The research worker alone may put wiki pages (P4); papers/ and index/ stay denied for it
        # too, and every wiki deletion, ACL, tagging, restore and retention change stays denied.
        puts_a_wiki_page = role in {RESEARCH_MODULE_ROLE, ANSWER_MODULE_ROLE} and action == "s3:PutObject"
        expected = WIKI_PREFIXES - ({BUCKET_ARN + "wiki/*"} if puts_a_wiki_page else set())
        assert expected <= denied, (role, action, denied)
        if puts_a_wiki_page:
            # The Allow names the exact folders; a blanket wiki/* deny would override it.
            assert BUCKET_ARN + "wiki/*" not in denied
        assert all("Condition" not in statement for statement in denials)
    for action in ("lambda:InvokeFunction", "lambda:InvokeFunctionUrl", "lambda:InvokeAsync"):
        invoke_denials = matching(role, "Deny", action)
        assert any(resources(statement) == ["*"] and "Condition" not in statement
                   for statement in invoke_denials), (role, action)
    assert not any(action.lower().startswith("lambda:") for action in allowed_actions(role)), role


@pytest.mark.parametrize("role", ROLES)
def test_s3_writes_are_receipts_only_and_never_deletions(role):
    puts = matching(role, "Allow", "s3:PutObject")
    put_resources = {resource for statement in puts for resource in resources(statement)}
    if role == RESEARCH_MODULE_ROLE:
        assert put_resources == RESEARCH_PUT_PREFIXES
    elif role == ANSWER_MODULE_ROLE:
        assert put_resources == ANSWER_PUT_PREFIXES
    else:
        assert put_resources <= {BUCKET_ARN + "runs/lab-questions/*"}
    assert not matching(role, "Allow", "s3:DeleteObject")
    assert not matching(role, "Allow", "s3:PutObjectAcl")
    for statement in statements(role):
        if statement["Effect"] != "Allow":
            continue
        for action in values(statement["Action"]):
            assert action != "s3:*" and not action.endswith(":*"), (role, action)
            assert action != "*", role
        if any(action.lower().startswith("s3:") for action in values(statement["Action"])):
            for resource in resources(statement):
                assert resource == BUCKET_ARN.rstrip("/") or resource.startswith(BUCKET_ARN), (role, resource)
                # papers/ and index/ are read-only for every role; wiki/ is read-only except for the
                # research worker's PutObject grant, which names folders rather than wiki/*.
                if resource.startswith((BUCKET_ARN + "papers/", BUCKET_ARN + "index/")):
                    assert all(action in {"s3:GetObject", "s3:GetObjectVersion"}
                               for action in values(statement["Action"])), (role, resource)
                elif resource.startswith(BUCKET_ARN + "wiki/"):
                    assert all(action in {"s3:GetObject", "s3:GetObjectVersion"} for action in values(statement["Action"])) \
                        or (role == RESEARCH_MODULE_ROLE and values(statement["Action"]) == ["s3:PutObject"]
                            and resource != BUCKET_ARN + "wiki/*") \
                        or (role == ANSWER_MODULE_ROLE and values(statement["Action"]) == ["s3:PutObject"]
                            and resource == ANSWER_PAGE_PREFIX_ARN), (role, resource)


@pytest.mark.parametrize("role", ROLES)
def test_reads_of_shared_wiki_and_index_are_read_only(role):
    readable = {BUCKET_ARN + "wiki/*", BUCKET_ARN + "index/wiki-index-v2.sqlite3", BUCKET_ARN + "runs/lab-questions/*"}
    listable = list(LISTABLE_PREFIXES)
    if role in {RESEARCH_MODULE_ROLE, ANSWER_MODULE_ROLE}:
        # Both read the paper itself when the notes cannot settle a point. For the answer worker
        # this is the user's rule of 2026-09-20 -- when the wiki has nothing, read the original --
        # reaching the student path on 2026-09-22, along with the figure and table text stored
        # beside each extraction. It is a read: every write under papers/ stays denied.
        readable.add(BUCKET_ARN + "papers/*")
    if role == "GatewayRole":
        # read_source serves the stored extraction behind a note, never the PDF (2026-09-22).
        readable |= {BUCKET_ARN + "papers/*/clean.md", BUCKET_ARN + "sources/*.md"}
        listable += ["papers/*", "sources/*"]
    for action in ("s3:GetObject", "s3:GetObjectVersion"):
        grants = matching(role, "Allow", action)
        allowed = {resource for statement in grants for resource in resources(statement)}
        assert allowed <= readable, (role, allowed)
    for statement in matching(role, "Allow", "s3:ListBucket"):
        assert resources(statement) == [BUCKET_ARN.rstrip("/")]
        assert statement["Condition"] == {"StringLike": {"s3:prefix": listable}}


@pytest.mark.parametrize("role", ["GatewayRole", "TriageRole", "OutboxRole"])
@pytest.mark.parametrize("action", ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"])
def test_model_invocation_is_denied_outside_the_model_workers(role, action):
    assert not matching(role, "Allow", action)
    assert any(resources(statement) == ["*"] and "Condition" not in statement
               for statement in matching(role, "Deny", action)), (role, action)


@pytest.mark.parametrize("role", ["AnswerRole", "ResearchRole"])
def test_model_workers_may_invoke_anthropic_models_and_inference_profiles_only(role):
    bedrock_allows = [statement for statement in statements(role)
                      if statement["Effect"] == "Allow"
                      and any(action.startswith("bedrock:") for action in values(statement["Action"]))]
    assert {action for statement in bedrock_allows for action in values(statement["Action"])} == {
        "bedrock:InvokeModel",
    }
    assert {resource for statement in bedrock_allows for resource in resources(statement)} == {
        "arn:${AWS::Partition}:bedrock:*::foundation-model/anthropic.*",
        "arn:${AWS::Partition}:bedrock:*:${AWS::AccountId}:inference-profile/*",
    }
    assert not matching(role, "Allow", "bedrock:InvokeModelWithResponseStream")


def test_research_worker_denies_streaming_invocation_explicitly():
    assert any(resources(statement) == ["*"] and "Condition" not in statement
               for statement in matching("ResearchRole", "Deny", "bedrock:InvokeModelWithResponseStream"))
    assert not matching("ResearchRole", "Deny", "bedrock:InvokeModel")


def test_only_triage_reads_the_jev_secret():
    grants = matching("TriageRole", "Allow", "ssm:GetParameter")
    assert [resources(statement) for statement in grants] == [[JEV_PARAMETER_ARN]]
    decrypt, = matching("TriageRole", "Allow", "kms:Decrypt")
    assert decrypt["Resource"] == {"Ref": "ParameterKmsKeyArn"}
    assert decrypt["Condition"] == {"StringEquals": {
        "kms:ViaService": {"Fn::Sub": "ssm.${AWS::Region}.${AWS::URLSuffix}"},
        "kms:EncryptionContext:PARAMETER_ARN": {"Fn::Sub": JEV_PARAMETER_ARN},
    }}
    for role in ROLES:
        if role == "TriageRole":
            continue
        assert not any(action.lower().startswith(("ssm:", "kms:")) for action in allowed_actions(role)), role
        assert matching(role, "Deny", "ssm:GetParameter"), role
    assert not matching("TriageRole", "Allow", "ssm:GetParameters")
    assert not matching("TriageRole", "Allow", "ssm:GetParametersByPath")


def test_queue_permissions_follow_the_data_flow():
    def sqs(role, action):
        return {resource for statement in matching(role, "Allow", action) for resource in resources(statement)}

    consume = {"sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"}
    for action in consume:
        assert sqs("AnswerRole", action) == {"GetAtt:AnswerQueue.Arn"}, action
        assert sqs("TriageRole", action) == {"GetAtt:TriageQueue.Arn"}, action
        assert sqs("ResearchRole", action) == {"GetAtt:ResearchQueue.Arn"}, action
        assert sqs("GatewayRole", action) == set()
        assert sqs("OutboxRole", action) == set()
    assert sqs("GatewayRole", "sqs:SendMessage") == {"GetAtt:AnswerQueue.Arn"}
    assert sqs("AnswerRole", "sqs:SendMessage") == {"GetAtt:TriageQueue.Arn"}
    assert sqs("OutboxRole", "sqs:SendMessage") == {
        "GetAtt:AnswerQueue.Arn", "GetAtt:TriageQueue.Arn", "GetAtt:ResearchQueue.Arn",
    }
    assert sqs("TriageRole", "sqs:SendMessage") == set()
    assert sqs("ResearchRole", "sqs:SendMessage") == set()


def test_control_table_access_uses_item_level_actions_only():
    # Every close (complete, fail, mark_unknown) and the relay's mark_sent delete the finished
    # outbox row's PENDING# pointer inside the same transaction, so DeleteItem is part of the item
    # set; it stays confined to the control table, and no role may scan, batch-write or
    # administer the table.
    item_actions = {"dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem",
                    "dynamodb:Query", "dynamodb:ConditionCheckItem"}
    # The research worker closes its job like the answer worker does (claim, attempt reservation,
    # receipt pointer, completion with the outbox row's pending pointer deleted), so it takes the
    # same item set.
    for role in ("GatewayRole", "AnswerRole", "TriageRole", "ResearchRole"):
        grants = [statement for statement in statements(role) if statement["Effect"] == "Allow"
                  and any(action.startswith("dynamodb:") for action in values(statement["Action"]))]
        assert grants, role
        assert {action for statement in grants for action in values(statement["Action"])} == item_actions
        assert {resource for statement in grants for resource in resources(statement)} == {
            "GetAtt:ControlTable.Arn",
        }
    # The relay marks rows sent and sweeps expired leases (closing jobs, pointers, outbox rows and
    # flagging reservations): reads, updates, pointer deletes and queries, never a new item.
    outbox = {action for statement in statements("OutboxRole") if statement["Effect"] == "Allow"
              for action in values(statement["Action"]) if action.startswith("dynamodb:")}
    assert outbox == {"dynamodb:GetItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query"}
    for role in ROLES:
        deletes = matching(role, "Allow", "dynamodb:DeleteItem")
        assert {resource for statement in deletes for resource in resources(statement)} <= {"GetAtt:ControlTable.Arn"}
        assert all("Condition" not in statement for statement in deletes), role
        for action in ("dynamodb:Scan", "dynamodb:BatchWriteItem", "dynamodb:BatchGetItem", "dynamodb:DeleteTable",
                       "dynamodb:UpdateTable", "dynamodb:PartiQLDelete", "dynamodb:PartiQLUpdate"):
            assert not matching(role, "Allow", action), (role, action)


def test_research_role_grants_exactly_what_the_scoped_worker_needs():
    assert allowed_actions("ResearchRole") == {
        "bedrock:InvokeModel",
        "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:Query",
        "dynamodb:ConditionCheckItem",
        "s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:ListBucket",
        "sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility",
        "logs:CreateLogStream", "logs:PutLogEvents",
    }
    reads = {resource for statement in matching("ResearchRole", "Allow", "s3:GetObject") for resource in resources(statement)}
    assert reads == {BUCKET_ARN + "wiki/*", BUCKET_ARN + "papers/*", BUCKET_ARN + "index/wiki-index-v2.sqlite3",
                     BUCKET_ARN + "runs/lab-questions/*"}
    # Without ListBucket S3 answers a GET of a missing key with 403 instead of 404: that broke the
    # research worker's recovery check (receipts) and turned a dangling wikilink into a connection
    # error (wiki) on 2026-09-21. Only those two prefixes are listable; papers/ and index/ are not.
    listing = matching("ResearchRole", "Allow", "s3:ListBucket")
    assert len(listing) == 1 and listing[0]["Condition"] == {"StringLike": {"s3:prefix": LISTABLE_PREFIXES}}
    assert not matching("ResearchRole", "Allow", "sqs:SendMessage")
    # Deny statements: papers/ and index/ writes in full, every wiki change other than PutObject,
    # Lambda invocation, streaming model calls and parameter reads.
    denies = {statement["Sid"]: statement for statement in statements("ResearchRole") if statement["Effect"] == "Deny"}
    assert set(denies) == {"DenyPaperAndIndexWrites", "DenyWikiDeletionsAndMetadataChanges", "DenyLambdaInvocation",
                           "DenyStreamingModelInvocation", "DenyParameterReads"}
    assert set(resources(denies["DenyPaperAndIndexWrites"])) == {BUCKET_ARN + "papers/*", BUCKET_ARN + "index/*"}
    assert "s3:PutObject" in values(denies["DenyPaperAndIndexWrites"]["Action"])
    assert resources(denies["DenyWikiDeletionsAndMetadataChanges"]) == [BUCKET_ARN + "wiki/*"]
    wiki_denied = set(values(denies["DenyWikiDeletionsAndMetadataChanges"]["Action"]))
    assert "s3:PutObject" not in wiki_denied
    assert {"s3:DeleteObject", "s3:DeleteObjectVersion", "s3:PutObjectAcl", "s3:PutObjectTagging",
            "s3:RestoreObject", "s3:PutObjectRetention", "s3:PutObjectLegalHold"} <= wiki_denied


@pytest.mark.parametrize("role", ROLES)
def test_logs_are_limited_to_the_functions_own_log_group(role):
    log_group = role.replace("Role", "LogGroup")
    for action in ("logs:CreateLogStream", "logs:PutLogEvents"):
        grants = matching(role, "Allow", action)
        assert {resource for statement in grants for resource in resources(statement)} == {
            "GetAtt:" + log_group + ".Arn",
        }, (role, action)
    assert not matching(role, "Allow", "logs:CreateLogGroup")


def test_template_has_no_managed_policy_attachments_or_negated_statements():
    forbidden = [(path, key) for path, key, _ in walk(TEMPLATE)
                 if key in {"ManagedPolicyArns", "NotAction", "NotResource", "NotPrincipal"}]
    assert forbidden == []
    for role in ROLES:
        assert "RoleName" not in RESOURCES[role]["Properties"]
        assert "PermissionsBoundary" not in RESOURCES[role]["Properties"]


# --- caller policies --------------------------------------------------------------------------


def test_student_policy_grants_url_invocation_and_nothing_else():
    policy = RESOURCES["StudentAccessPolicy"]
    assert policy["Type"] == "AWS::IAM::ManagedPolicy"
    assert "ManagedPolicyName" not in policy["Properties"]
    document = policy["Properties"]["PolicyDocument"]
    assert document["Version"] == "2012-10-17"
    gateway_arn = {"Fn::GetAtt": ["GatewayFunction", "Arn"]}
    assert document["Statement"] == [
        {
            "Sid": "InvokeGatewayUrl",
            "Effect": "Allow",
            "Action": "lambda:InvokeFunctionUrl",
            "Resource": gateway_arn,
            "Condition": {"StringEquals": {"lambda:FunctionUrlAuthType": "AWS_IAM"}},
        },
        {
            "Sid": "InvokeGatewayViaUrlOnly",
            "Effect": "Allow",
            "Action": "lambda:InvokeFunction",
            "Resource": gateway_arn,
            "Condition": {"Bool": {"lambda:InvokedViaFunctionUrl": "true"}},
        },
    ]
    for key in ("Users", "Groups", "Roles"):
        assert key not in policy["Properties"], key


def test_admin_policy_adds_registry_maintenance_and_relay_without_wiki_or_model_access():
    policy = RESOURCES["AdminAccessPolicy"]
    assert policy["Type"] == "AWS::IAM::ManagedPolicy"
    document = policy["Properties"]["PolicyDocument"]
    admin_statements = document["Statement"]
    student_statements = RESOURCES["StudentAccessPolicy"]["Properties"]["PolicyDocument"]["Statement"]
    assert admin_statements[:2] == student_statements
    actions = {action for statement in admin_statements for action in values(statement["Action"])}
    assert not any(action.startswith(("bedrock:", "ssm:", "kms:", "states:", "iam:", "sts:"))
                   for action in actions), actions
    assert not any(action.endswith("*") or action == "*" for action in actions), actions
    assert "s3:PutObject" not in actions and "s3:DeleteObject" not in actions
    for statement in admin_statements:
        assert statement["Effect"] == "Allow"
        for action in values(statement["Action"]):
            if action == "lambda:InvokeFunction" and "Condition" not in statement:
                assert resources(statement) == ["GetAtt:OutboxFunction.Arn"]
            if action.startswith("dynamodb:"):
                assert resources(statement) == ["GetAtt:ControlTable.Arn"]
                assert statement["Condition"] == ADMIN_REGISTRY_CONDITION, statement["Sid"]
            if action in {"s3:GetObject", "s3:GetObjectVersion"}:
                assert resources(statement) == [BUCKET_ARN + "runs/lab-questions/*"]
    assert "dynamodb:DeleteItem" not in actions and "dynamodb:Scan" not in actions
    for key in ("Users", "Groups", "Roles"):
        assert key not in policy["Properties"], key


def test_admin_console_writes_are_limited_to_the_registry_and_budget_partitions():
    """The professor maintains members, principals and budget caps by hand; jobs, offers, verdicts,
    approvals, reservations and outbox rows are written by the functions only."""
    admin_statements = RESOURCES["AdminAccessPolicy"]["Properties"]["PolicyDocument"]["Statement"]
    registry, = [statement for statement in admin_statements if statement["Sid"] == "MaintainMemberRegistry"]
    assert set(values(registry["Action"])) == {"dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem",
                                               "dynamodb:Query"}
    assert resources(registry) == ["GetAtt:ControlTable.Arn"]
    assert registry["Condition"] == ADMIN_REGISTRY_CONDITION
    prefixes = registry["Condition"]["ForAllValues:StringLike"]["dynamodb:LeadingKeys"]
    assert prefixes == ["MEMBER#*", "PRINCIPAL#*", "BUDGET#*"]
    # Partitions the functions own never match the allowed prefixes.
    for partition in ("JOB#abc", "OFFER#abc", "APPROVAL#abc", "RESERVATION#abc", "OUTBOX#abc", "OUTBOX",
                      "IDEMP#m1#answer#req", "SESSION#abc", "VERDICT#abc", "RECORDS#2026-09-21", "CANDIDATE#abc",
                      "CANDIDATES", "ROUND#abc"):
        assert not any(fnmatchcase(partition, prefix) for prefix in prefixes), partition
    for partition in ("MEMBER#m1", "PRINCIPAL#AIDAM1", "BUDGET#lab:2026-09", "BUDGET#member:m1:2026-09"):
        assert any(fnmatchcase(partition, prefix) for prefix in prefixes), partition
    dynamodb_statements = [statement for statement in admin_statements
                           if any(action.startswith("dynamodb:") for action in values(statement["Action"]))]
    assert dynamodb_statements == [registry]


def test_outputs_expose_the_url_table_and_caller_policies():
    outputs = TEMPLATE["Outputs"]
    assert outputs["GatewayUrl"]["Value"] == {"Fn::GetAtt": ["GatewayUrl", "FunctionUrl"]}
    assert outputs["ControlTableName"]["Value"] == {"Ref": "ControlTable"}
    assert outputs["StudentAccessPolicyArn"]["Value"] == {"Ref": "StudentAccessPolicy"}
    assert outputs["AdminAccessPolicyArn"]["Value"] == {"Ref": "AdminAccessPolicy"}


# --- deploy script (inspected, never executed) ------------------------------------------------


def test_deploy_script_is_executable_and_parses():
    assert SCRIPT_PATH.exists()
    assert os.access(SCRIPT_PATH, os.X_OK)
    text = SCRIPT_PATH.read_text()
    assert text.startswith("#!/bin/bash\n")
    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)


def test_deploy_script_is_guarded_against_the_running_campaign_and_defaults_to_review():
    text = SCRIPT_PATH.read_text()
    header = "\n".join(line for line in text.splitlines()[:20] if line.startswith("#"))
    assert "not executed on 2026-09-21" in header
    assert "after the question campaign is finished" in header
    assert "set -euo pipefail" in text
    assert '${AWS_KIRO_WIKI_BUCKET:?' in text
    assert '${LAB_JEV_KMS_KEY_ARN:?' in text
    assert 'LAB_STACK:-byeori-lab' in text
    assert "--allow-during-campaign" in text
    assert "aws stepfunctions list-executions" in text
    assert "--status-filter RUNNING" in text
    assert "QUESTION_STATE_MACHINE_ARN" in text
    assert "--no-execute-changeset" in text
    assert "--execute" in text
    assert "execute-change-set" in text
    assert "describe-change-set" in text
    assert "validate-template" in text
    assert "cfn/lab/" in text
    assert "sha256" in text
    assert "infra/lab-template.json" in text
    assert "ArtifactKey=" in text and "DataBucket=" in text and "ParameterKmsKeyArn=" in text
    assert "--capabilities CAPABILITY_IAM" in text
    # The campaign guard must come before any upload or deploy call.
    guard = text.index("list-executions")
    assert guard < text.index("aws s3 cp")
    assert guard < text.index("cloudformation deploy")
    # Nothing in the script deletes or synchronises S3 content.
    assert "s3 rm" not in text and "s3 sync" not in text and "delete-stack" not in text


def _deploy_parameters(**environment: str) -> tuple[list[str], str]:
    """Run the script's parameter block alone and return the array it built plus its output."""
    block = SCRIPT_PATH.read_text().split("parameters=(", 1)[1]
    block = "parameters=(" + block.split("\nstatus=", 1)[0]
    script = (
        'set -euo pipefail\n'
        'AWS_KIRO_WIKI_BUCKET=bucket\nLAB_JEV_KMS_KEY_ARN=arn:kms\nartifact_key=cfn/lab/abc.zip\n'
        + block
        + '\nprintf "%s\\n" "${parameters[@]}"\n'
    )
    completed = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True,
                               env={"PATH": os.environ["PATH"], **environment})
    lines = completed.stdout.splitlines()
    carried = "\n".join(line for line in lines if line.startswith("Keeping"))
    return [line for line in lines if "=" in line and not line.startswith("Keeping")], carried


OPTIONAL_PARAMETERS = {
    "LAB_ANSWER_MODEL_ID": "AnswerModelId",
    "LAB_ANSWER_REASONING": "AnswerReasoning",
    "LAB_POLICY_REVISION": "PolicyRevision",
    "LAB_RESEARCH_CONSUMER_ENABLED": "ResearchConsumerEnabled",
    "LAB_RESEARCH_MODEL_ID": "ResearchModelId",
    "LAB_RESEARCH_REASONING": "ResearchReasoning",
    "LAB_OUTBOX_SCHEDULE_ENABLED": "OutboxScheduleEnabled",
}


def test_every_optional_setting_goes_through_add_optional_and_never_carries_a_shell_default():
    """A shell default here would reset a deployed setting on a deploy that only ships code.

    The stack had ResearchConsumerEnabled=true while the script sent
    ``${LAB_RESEARCH_CONSUMER_ENABLED:-false}``, so a plain re-deploy would have detached the
    approved-research queue silently. Every optional setting must reach the array only through
    ``add_optional``, which omits it when its variable is unset.
    """
    text = SCRIPT_PATH.read_text()
    for variable, parameter in OPTIONAL_PARAMETERS.items():
        assert f"add_optional {variable} {parameter}" in text, parameter
        assert f"{parameter}=${{{variable}:-" not in text, parameter
    # Required values have no previous-value fallback: the build determines them every time.
    for required in ("DataBucket", "ArtifactBucket", "ArtifactKey", "ParameterKmsKeyArn"):
        assert f'"{required}=$' in text, required


def test_an_unset_setting_is_left_out_so_cloudformation_keeps_the_deployed_value():
    sent, carried = _deploy_parameters()
    assert sent == ["DataBucket=bucket", "ArtifactBucket=bucket", "ArtifactKey=cfn/lab/abc.zip",
                    "ParameterKmsKeyArn=arn:kms"]
    for parameter in OPTIONAL_PARAMETERS.values():
        assert parameter in carried, parameter


def test_a_named_setting_is_sent_and_the_rest_are_still_carried_over():
    sent, carried = _deploy_parameters(LAB_RESEARCH_CONSUMER_ENABLED="true", LAB_ANSWER_REASONING="xhigh")
    assert "ResearchConsumerEnabled=true" in sent
    assert "AnswerReasoning=xhigh" in sent
    assert "ResearchConsumerEnabled" not in carried and "AnswerReasoning" not in carried
    assert "OutboxScheduleEnabled" in carried and "PolicyRevision" in carried


def test_the_answer_worker_may_put_only_the_unindexed_question_prefix_under_wiki():
    """One wiki prefix, and it is the one the index skips.

    Answered questions are kept as Markdown so a reader can follow them (user, 2026-09-22), but a
    scientific page must stay out of reach of a path 18 students can trigger. The Allow names
    ``wiki/lab-questions/*`` alone, every destructive and metadata action on ``wiki/*`` stays
    denied, and ``papers/`` and ``index/`` stay closed to every write.
    """
    puts = matching(ANSWER_MODULE_ROLE, "Allow", "s3:PutObject")
    assert {resource for statement in puts for resource in resources(statement)} == ANSWER_PUT_PREFIXES
    for folder in ("wiki/sources/*", "wiki/overviews/*", "wiki/concepts/*", "wiki/questions/*", "wiki/*"):
        assert BUCKET_ARN + folder not in {resource for statement in puts for resource in resources(statement)}
    for action in ("s3:DeleteObject", "s3:DeleteObjectVersion", "s3:PutObjectAcl", "s3:PutObjectTagging"):
        denied = {resource for statement in matching(ANSWER_MODULE_ROLE, "Deny", action)
                  for resource in resources(statement)}
        assert WIKI_PREFIXES <= denied, action
    denied_puts = {resource for statement in matching(ANSWER_MODULE_ROLE, "Deny", "s3:PutObject")
                   for resource in resources(statement)}
    assert denied_puts == {BUCKET_ARN + "papers/*", BUCKET_ARN + "index/*"}


def test_the_answer_worker_may_read_a_paper_but_never_write_one():
    """Reading the original is the rule; papers/ stays closed to every write (user, 2026-09-20)."""
    readable = {resource for statement in matching(ANSWER_MODULE_ROLE, "Allow", "s3:GetObject")
                for resource in resources(statement)}
    assert BUCKET_ARN + "papers/*" in readable
    for action in ("s3:PutObject", "s3:DeleteObject", "s3:PutObjectTagging"):
        denied = {resource for statement in matching(ANSWER_MODULE_ROLE, "Deny", action)
                  for resource in resources(statement)}
        assert BUCKET_ARN + "papers/*" in denied, action
    puts = {resource for statement in matching(ANSWER_MODULE_ROLE, "Allow", "s3:PutObject")
            for resource in resources(statement)}
    assert BUCKET_ARN + "papers/*" not in puts and puts == ANSWER_PUT_PREFIXES
