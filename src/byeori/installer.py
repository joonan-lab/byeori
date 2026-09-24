"""What another lab runs to install Byeori in its own account: settings, deploy, workers, policy, checks.

Thin on purpose: `deploy`, `build-workers` and `deploy-lab` run the same shell scripts this
repository deploys with, so the release does not fork the deployment logic.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ".byeori.env"
QUESTIONS: tuple[tuple[str, str, str], ...] = (
    ("AWS_PROFILE", "AWS CLI profile to use", "byeori"),
    ("AWS_REGION", "Region where Bedrock offers your models (for example us-east-1)", "us-east-1"),
    ("KIRO_WIKI_STACK", "Name of the CloudFormation stack to create", "byeori"),
    ("KIRO_WIKI_OPENALEX_PARAMETER", "Parameter Store name holding your OpenAlex API key", "/byeori/openalex-api-key"),
    ("KIRO_WIKI_CONTACT_EMAIL", "Contact e-mail sent to OpenAlex, Crossref and NCBI as the polite-pool address (may be empty)", ""),
    ("KIRO_WIKI_CREATE_VPC", "Let the stack create its own network? (true/false)", "true"),
)
VPC_QUESTIONS: tuple[tuple[str, str, str], ...] = (
    ("KIRO_WIKI_VPC_ID", "Existing VPC id whose public subnets run extraction", ""),
    ("KIRO_WIKI_SUBNET_IDS", "Comma-separated public subnet ids in that VPC", ""),
)
OUTPUT_TO_ENV = {"BucketName": "AWS_KIRO_WIKI_BUCKET", "TableName": "AWS_KIRO_WIKI_TABLE",
                 "IngestFunctionName": "AWS_KIRO_WIKI_INGEST_FUNCTION"}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def write_env(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.removeprefix("export ").split("=", 1)[0].strip() if "=" in line else None
        if key in values:
            out.append(f"export {key}={values[key]}"); seen.add(key)
        else:
            out.append(line)
    for key, value in values.items():
        if key not in seen:
            out.append(f"export {key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    path.chmod(0o600)


def ask_settings(ask: Callable[[str, str], str], existing: dict[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, _prompt, default in QUESTIONS:
        values[key] = ask(key, existing.get(key, default)).strip()
    if values["KIRO_WIKI_CREATE_VPC"] != "true":
        for key, _prompt, default in VPC_QUESTIONS:
            values[key] = ask(key, existing.get(key, default)).strip()
    values["BYEORI_CONTACT_EMAIL"] = values["KIRO_WIKI_CONTACT_EMAIL"]
    return values


def prompt_on_terminal(key: str, default: str) -> str:
    prompt = next(p for k, p, _d in QUESTIONS + VPC_QUESTIONS if k == key)
    answer = input(f"{prompt} [{default}]: ")
    return answer or default


def outputs_of(cloudformation: Any, stack: str) -> dict[str, str]:
    stacks = cloudformation.describe_stacks(StackName=stack)["Stacks"]
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


def env_from_outputs(outputs: dict[str, str]) -> dict[str, str]:
    return {env: outputs[key] for key, env in OUTPUT_TO_ENV.items() if key in outputs}


def build_client_policy(outputs: dict[str, str], region: str, account: str, stack: str,
                        partition: str = "aws") -> dict[str, Any]:
    bucket, table = outputs["BucketName"], outputs["TableName"]
    return {"Version": "2012-10-17", "Statement": [
        {"Sid": "RunTheWikiPipeline", "Effect": "Allow", "Action": ["lambda:InvokeFunction"],
         "Resource": f"arn:{partition}:lambda:{region}:{account}:function:{outputs['IngestFunctionName']}"},
        {"Sid": "ReadAndWriteTheWiki", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
         "Resource": f"arn:{partition}:s3:::{bucket}/*"},
        {"Sid": "ListTheWikiBucket", "Effect": "Allow", "Action": ["s3:ListBucket"],
         "Resource": f"arn:{partition}:s3:::{bucket}"},
        {"Sid": "CatalogAndState", "Effect": "Allow",
         "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query", "dynamodb:Scan"],
         "Resource": [f"arn:{partition}:dynamodb:{region}:{account}:table/{table}",
                      f"arn:{partition}:dynamodb:{region}:{account}:table/{table}/index/*"]},
        {"Sid": "ReadStackOutputs", "Effect": "Allow", "Action": ["cloudformation:DescribeStacks"],
         "Resource": f"arn:{partition}:cloudformation:{region}:{account}:stack/{stack}/*"},
        {"Sid": "RunExtractionTasks", "Effect": "Allow", "Action": ["ecs:RunTask", "ecs:ListTasks", "ecs:DescribeTasks"],
         "Resource": "*", "Condition": {"ArnEquals": {
             "ecs:cluster": f"arn:{partition}:ecs:{region}:{account}:cluster/{outputs['ExtractionClusterName']}"}}},
        {"Sid": "LetEcsUseItsOwnRoles", "Effect": "Allow", "Action": ["iam:PassRole"],
         "Resource": [outputs["ExtractionExecutionRoleArn"], outputs["ExtractionTaskRoleArn"]],
         "Condition": {"StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}}},
        {"Sid": "ReadTheExtractionLogs", "Effect": "Allow",
         "Action": ["logs:DescribeLogGroups", "logs:DescribeLogStreams", "logs:FilterLogEvents", "logs:GetLogEvents"],
         "Resource": f"arn:{partition}:logs:{region}:{account}:log-group:*"},
    ]}


def check_bedrock_models(client: Any, model_ids: list[str]) -> dict[str, str]:
    """One one-token call per model; the error code says what the account lacks."""
    report: dict[str, str] = {}
    for model_id in model_ids:
        try:
            client.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": "ping"}]}],
                            inferenceConfig={"maxTokens": 1})
            report[model_id] = "ok"
        except ClientError as exc:
            report[model_id] = exc.response.get("Error", {}).get("Code", "ClientError")
        except BotoCoreError as exc:
            report[model_id] = type(exc).__name__
    return report


def model_ids_of(cloudformation: Any, stack: str) -> list[str]:
    stacks = cloudformation.describe_stacks(StackName=stack)["Stacks"]
    return sorted({p["ParameterValue"] for p in stacks[0].get("Parameters", []) if p["ParameterKey"].endswith("ModelId")})


def run_script(name: str, env: dict[str, str], *args: str) -> int:
    merged = {**os.environ, **env}
    return subprocess.run(["bash", str(REPO_ROOT / "scripts" / name), *args], cwd=REPO_ROOT, env=merged).returncode


def ensure_deploy_bucket(session: Any, account: str, region: str) -> str:
    """The small bucket `deploy.sh` packages CloudFormation templates into. A first install has no
    other bucket yet: the stack's own data bucket is one of the stack's *outputs*, so it cannot
    hold the templates that create it. This bucket is created once, stays (a few kilobytes), and is
    reused by every later deploy that already has a data bucket configured (`command_deploy` only
    calls this when the env file has none)."""
    name = f"byeori-deploy-{account}-{region}"
    kwargs: dict[str, Any] = {} if region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": region}}
    try:
        session.client("s3").create_bucket(Bucket=name, **kwargs)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "BucketAlreadyOwnedByYou":
            raise
    return name
