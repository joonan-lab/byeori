"""The installer subcommands another lab runs (spec 2026-09-24, 'What an installer does')."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from byeori import cli, installer
from byeori.cli import build_parser


def test_env_file_round_trip_keeps_other_keys(tmp_path):
    path = tmp_path / ".env"
    path.write_text("export KEEP=1\nexport AWS_REGION=old\n")
    installer.write_env(path, {"AWS_REGION": "eu-west-1", "KIRO_WIKI_STACK": "byeori"})
    assert installer.read_env(path) == {"KEEP": "1", "AWS_REGION": "eu-west-1", "KIRO_WIKI_STACK": "byeori"}
    assert path.read_text().startswith("export KEEP=1\n")


def test_ask_settings_asks_for_the_vpc_only_when_not_creating_one():
    answers = {"AWS_PROFILE": "byeori", "AWS_REGION": "us-east-1", "KIRO_WIKI_STACK": "byeori",
               "KIRO_WIKI_OPENALEX_PARAMETER": "/byeori/openalex-api-key", "KIRO_WIKI_CONTACT_EMAIL": "lab@example.org",
               "KIRO_WIKI_CREATE_VPC": "true"}
    asked: list[str] = []
    def ask(prompt, default):
        asked.append(prompt); return answers.get(prompt, default)
    values = installer.ask_settings(ask, {})
    assert values == {**answers, "BYEORI_CONTACT_EMAIL": "lab@example.org"}
    assert "KIRO_WIKI_VPC_ID" not in asked
    answers["KIRO_WIKI_CREATE_VPC"] = "false"; answers["KIRO_WIKI_VPC_ID"] = "vpc-1"; answers["KIRO_WIKI_SUBNET_IDS"] = "subnet-1,subnet-2"
    values = installer.ask_settings(ask, {})
    assert values["KIRO_WIKI_VPC_ID"] == "vpc-1" and values["KIRO_WIKI_SUBNET_IDS"] == "subnet-1,subnet-2"


def test_ask_settings_offers_the_existing_value_as_default():
    seen = {}
    def ask(prompt, default):
        seen[prompt] = default; return default
    installer.ask_settings(ask, {"AWS_REGION": "ap-southeast-1", "KIRO_WIKI_CREATE_VPC": "true"})
    assert seen["AWS_REGION"] == "ap-southeast-1"
    assert seen["KIRO_WIKI_OPENALEX_PARAMETER"] == "/byeori/openalex-api-key"


OUTPUTS = {"BucketName": "b-1", "TableName": "t-1", "IngestFunctionName": "f-1", "ExtractionClusterName": "c-1",
           "ExtractionExecutionRoleArn": "arn:aws:iam::111122223333:role/x-Exec", "ExtractionTaskRoleArn": "arn:aws:iam::111122223333:role/x-Task"}


def test_env_from_outputs():
    assert installer.env_from_outputs(OUTPUTS) == {"AWS_KIRO_WIKI_BUCKET": "b-1", "AWS_KIRO_WIKI_TABLE": "t-1",
                                                   "AWS_KIRO_WIKI_INGEST_FUNCTION": "f-1"}


def test_client_policy_is_built_from_outputs_not_typed():
    policy = installer.build_client_policy(OUTPUTS, "us-east-1", "111122223333", "byeori")
    by_sid = {s["Sid"]: s for s in policy["Statement"]}
    assert by_sid["RunTheWikiPipeline"]["Resource"] == "arn:aws:lambda:us-east-1:111122223333:function:f-1"
    assert by_sid["ReadAndWriteTheWiki"]["Resource"] == "arn:aws:s3:::b-1/*"
    assert by_sid["ListTheWikiBucket"]["Resource"] == "arn:aws:s3:::b-1"
    assert by_sid["CatalogAndState"]["Resource"] == ["arn:aws:dynamodb:us-east-1:111122223333:table/t-1",
                                                     "arn:aws:dynamodb:us-east-1:111122223333:table/t-1/index/*"]
    assert by_sid["ReadStackOutputs"]["Resource"] == "arn:aws:cloudformation:us-east-1:111122223333:stack/byeori/*"
    assert by_sid["RunExtractionTasks"]["Condition"]["ArnEquals"]["ecs:cluster"] == "arn:aws:ecs:us-east-1:111122223333:cluster/c-1"
    assert by_sid["LetEcsUseItsOwnRoles"]["Resource"] == [OUTPUTS["ExtractionExecutionRoleArn"], OUTPUTS["ExtractionTaskRoleArn"]]
    assert by_sid["ReadTheExtractionLogs"]["Resource"] == "arn:aws:logs:us-east-1:111122223333:log-group:*"
    json.dumps(policy)


def test_bedrock_check_names_the_refused_model_and_never_raises():
    client = MagicMock()
    def converse(modelId, **kwargs):
        if modelId == "denied":
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "Converse")
        return {"output": {}}
    client.converse.side_effect = converse
    assert installer.check_bedrock_models(client, ["ok-model", "denied"]) == {"ok-model": "ok", "denied": "AccessDeniedException"}
    call = client.converse.call_args_list[0].kwargs
    assert call["inferenceConfig"] == {"maxTokens": 1}


def test_aws_errors_from_installer_commands_are_reported_as_plain_cli_errors(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / ".env"
    installer.write_env(env_file, {"AWS_REGION": "us-east-1", "KIRO_WIKI_STACK": "x"})

    class Client:
        def describe_stacks(self, StackName):
            raise ClientError({"Error": {"Code": "ValidationError", "Message": "Stack with id x does not exist"}},
                              "DescribeStacks")

    class Session:
        def __init__(self, *a, **kw): pass
        region_name = "us-east-1"
        def client(self, name):
            return Client()

    monkeypatch.setattr(cli, "boto3", MagicMock(Session=Session))
    monkeypatch.setattr(sys, "argv", ["byeori", "grant-client", "--env-file", str(env_file)])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "does not exist" in err


def test_parser_has_the_installer_commands():
    choices = build_parser()._subparsers._group_actions[0].choices
    assert {"init", "deploy", "build-workers", "deploy-lab", "grant-client", "doctor"} <= set(choices)
    init = choices["init"]
    assert any("--non-interactive" in a.option_strings for a in init._actions)


def test_ensure_deploy_bucket_creates_with_a_location_constraint():
    calls = []

    class S3:
        def create_bucket(self, **kwargs):
            calls.append(kwargs)

    class Session:
        def client(self, name):
            assert name == "s3"
            return S3()

    name = installer.ensure_deploy_bucket(Session(), "111122223333", "eu-west-1")
    assert name == "byeori-deploy-111122223333-eu-west-1"
    assert calls == [{"Bucket": name, "CreateBucketConfiguration": {"LocationConstraint": "eu-west-1"}}]


def test_ensure_deploy_bucket_omits_the_location_constraint_for_us_east_1():
    calls = []

    class S3:
        def create_bucket(self, **kwargs):
            calls.append(kwargs)

    class Session:
        def client(self, name):
            return S3()

    name = installer.ensure_deploy_bucket(Session(), "111122223333", "us-east-1")
    assert name == "byeori-deploy-111122223333-us-east-1"
    assert calls == [{"Bucket": name}]


def test_ensure_deploy_bucket_tolerates_bucket_already_owned_by_you():
    class S3:
        def create_bucket(self, **kwargs):
            raise ClientError({"Error": {"Code": "BucketAlreadyOwnedByYou"}}, "CreateBucket")

    class Session:
        def client(self, name):
            return S3()

    name = installer.ensure_deploy_bucket(Session(), "111122223333", "us-east-1")
    assert name == "byeori-deploy-111122223333-us-east-1"


def test_ensure_deploy_bucket_propagates_other_client_errors():
    class S3:
        def create_bucket(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "CreateBucket")

    class Session:
        def client(self, name):
            return S3()

    with pytest.raises(ClientError):
        installer.ensure_deploy_bucket(Session(), "111122223333", "us-east-1")


def test_command_deploy_passes_the_bootstrap_bucket_when_the_env_file_lacks_one(tmp_path, monkeypatch):
    from byeori.config import Settings

    env_file = tmp_path / ".env"
    installer.write_env(env_file, {"AWS_PROFILE": "byeori", "AWS_REGION": "us-east-1", "KIRO_WIKI_STACK": "byeori"})

    captured: dict = {}

    def fake_run_script(name, env, *args):
        captured["script"] = name
        captured["env"] = env
        return 0

    monkeypatch.setattr(installer, "run_script", fake_run_script)

    class Cloudformation:
        def describe_stacks(self, StackName):
            return {"Stacks": [{"Outputs": [{"OutputKey": "BucketName", "OutputValue": "byeori-data-111122223333"}]}]}

    class Sts:
        def get_caller_identity(self):
            return {"Account": "111122223333"}

    class S3:
        def create_bucket(self, **kwargs):
            captured["create_bucket_kwargs"] = kwargs

    class FakeSession:
        def __init__(self, *a, **kw):
            pass
        region_name = "us-east-1"

        def client(self, name):
            return {"cloudformation": Cloudformation(), "sts": Sts(), "s3": S3()}[name]

    monkeypatch.setattr(cli, "boto3", MagicMock(Session=FakeSession))

    args = build_parser().parse_args(["deploy", "--env-file", str(env_file)])
    code = cli.command_deploy(Settings.from_env(), args)

    assert code == 0
    assert captured["script"] == "deploy.sh"
    assert captured["env"]["AWS_KIRO_WIKI_BUCKET"] == "byeori-deploy-111122223333-us-east-1"
    assert installer.read_env(env_file)["AWS_KIRO_WIKI_BUCKET"] == "byeori-data-111122223333"


def test_command_deploy_does_not_bootstrap_a_bucket_when_the_env_file_already_has_one(tmp_path, monkeypatch):
    from byeori.config import Settings

    env_file = tmp_path / ".env"
    installer.write_env(env_file, {"AWS_PROFILE": "byeori", "AWS_REGION": "us-east-1", "KIRO_WIKI_STACK": "byeori",
                                   "AWS_KIRO_WIKI_BUCKET": "byeori-data-111122223333"})

    captured: dict = {}

    def fake_run_script(name, env, *args):
        captured["env"] = env
        return 0

    monkeypatch.setattr(installer, "run_script", fake_run_script)

    class Cloudformation:
        def describe_stacks(self, StackName):
            return {"Stacks": [{"Outputs": [{"OutputKey": "BucketName", "OutputValue": "byeori-data-111122223333"}]}]}

    class FakeSession:
        def __init__(self, *a, **kw):
            pass
        region_name = "us-east-1"

        def client(self, name):
            assert name == "cloudformation"
            return Cloudformation()

    monkeypatch.setattr(cli, "boto3", MagicMock(Session=FakeSession))

    args = build_parser().parse_args(["deploy", "--env-file", str(env_file)])
    code = cli.command_deploy(Settings.from_env(), args)

    assert code == 0
    assert captured["env"]["AWS_KIRO_WIKI_BUCKET"] == "byeori-data-111122223333"
