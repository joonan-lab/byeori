from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path

import pytest


TEMPLATE = json.loads((Path(__file__).parents[1] / "infra/jev-eval-template.json").read_text())
RESOURCES = TEMPLATE["Resources"]
ROLE = RESOURCES["WorkerRole"]["Properties"]
STATEMENTS = [statement for policy in ROLE["Policies"]
              for statement in policy["PolicyDocument"]["Statement"]]
BUCKET_ARN = "arn:${AWS::Partition}:s3:::${ResultsBucket}/"


def values(value):
    return value if isinstance(value, list) else [value]


def resources(statement):
    return [value["Fn::Sub"] if isinstance(value, dict) and "Fn::Sub" in value else value
            for value in values(statement["Resource"])]


def matching(effect, action):
    return [statement for statement in STATEMENTS if statement["Effect"] == effect
            and any(fnmatchcase(action.lower(), item.lower())
                    for item in values(statement["Action"]))]


def test_isolated_function_has_bounded_resources_and_no_triggers():
    assert set(RESOURCES) == {"LogGroup", "WorkerRole", "Worker"}
    function = RESOURCES["Worker"]["Properties"]
    assert RESOURCES["Worker"]["Type"] == "AWS::Lambda::Function"
    assert function["FunctionName"] == {"Ref": "AWS::StackName"}
    assert function["MemorySize"] == 512 and function["Timeout"] == 120
    assert function["ReservedConcurrentExecutions"] == 1
    assert function["Runtime"] == "python3.12" and function["Handler"] == "index.handler"
    assert function["Architectures"] == ["arm64"]
    assert "Events" not in function and "ManagedPolicyArns" not in ROLE
    assert all("NotAction" not in item and "NotResource" not in item for item in STATEMENTS)
    assert "Default" not in TEMPLATE["Parameters"]["ResultsBucket"]
    assert function["Environment"]["Variables"]["JEV_RESULTS_BUCKET"] == {"Ref": "ResultsBucket"}


@pytest.mark.parametrize("action", ["s3:GetObject", "s3:GetObjectVersion"])
def test_reads_cover_only_approved_manifest_index_wiki_and_experiment(action):
    grants = matching("Allow", action)
    assert grants and all("Condition" not in statement for statement in grants)
    assert {resource for statement in grants for resource in resources(statement)} == {
        BUCKET_ARN + "runs/questions/original-1437-20260921/manifest.json",
        BUCKET_ARN + "index/wiki-index-v2.sqlite3",
        BUCKET_ARN + "wiki/*",
        BUCKET_ARN + "runs/jev-evaluations/triage-20260921-v1/*",
    }
    assert {item for statement in grants for item in values(statement["Action"])} == {
        "s3:GetObject", "s3:GetObjectVersion",
    }


def test_listing_is_limited_to_the_exact_experiment_prefix_in_the_results_bucket():
    statement, = matching("Allow", "s3:ListBucket")
    assert resources(statement) == [BUCKET_ARN.rstrip("/")]
    assert statement["Condition"] == {"StringLike": {
        "s3:prefix": "runs/jev-evaluations/triage-20260921-v1/*",
    }}


def test_s3_writes_are_receipt_only_and_wiki_publication_is_explicitly_denied():
    assert {resource for statement in matching("Allow", "s3:PutObject")
            for resource in resources(statement)} == {BUCKET_ARN + "runs/jev-evaluations/*"}
    denials = matching("Deny", "s3:PutObject")
    assert {resource for item in denials for resource in resources(item)} == {BUCKET_ARN + "wiki/*"}
    assert all("Condition" not in item for item in denials)
    allowed_s3_actions = {action for item in STATEMENTS if item["Effect"] == "Allow"
                          for action in values(item["Action"]) if action.lower().startswith("s3:")}
    assert allowed_s3_actions == {"s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:ListBucket"}


@pytest.mark.parametrize("action", [
    "bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "lambda:InvokeFunction",
])
def test_other_model_and_lambda_invocations_are_denied(action):
    assert not matching("Allow", action)
    assert any(resources(item) == ["*"] and "Condition" not in item
               for item in matching("Deny", action))


def test_role_allows_only_required_actions_without_other_secret_or_compute_access():
    assert {action for item in STATEMENTS if item["Effect"] == "Allow"
            for action in values(item["Action"])} == {
        "ssm:GetParameter", "kms:Decrypt", "s3:GetObject", "s3:GetObjectVersion",
        "s3:PutObject", "s3:ListBucket", "logs:CreateLogStream", "logs:PutLogEvents",
    }


def test_parameter_and_kms_permissions_remain_exact_and_context_bound():
    parameter_arn = "arn:${AWS::Partition}:ssm:${AWS::Region}:${AWS::AccountId}:parameter${JevApiKeyParameter}"
    assert TEMPLATE["Parameters"]["JevApiKeyParameter"]["AllowedValues"] == ["/byeori/jev/api-key"]
    assert [resources(item) for item in matching("Allow", "ssm:GetParameter")] == [[parameter_arn]]
    decrypt, = matching("Allow", "kms:Decrypt")
    assert decrypt["Resource"] == {"Ref": "ParameterKmsKeyArn"}
    assert decrypt["Condition"] == {"StringEquals": {
        "kms:ViaService": {"Fn::Sub": "ssm.${AWS::Region}.${AWS::URLSuffix}"},
        "kms:EncryptionContext:PARAMETER_ARN": {"Fn::Sub": parameter_arn},
    }}
