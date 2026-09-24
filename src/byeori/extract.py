"""Run the Fargate extraction worker over uploaded PDFs, in shards, and report progress.

``run_extraction`` publishes ``infra/extract_worker.py`` to S3 ``worker/extract.py``, selects the
stems still waiting (DynamoDB ``ingest_status`` in pdf_uploaded / extract_failed, or all given
stems with force), writes one job manifest per task to S3 ``jobs/<run>/<n>.json``, and starts
that many Fargate tasks (GROBID + worker) from the stack's task definition. ``extraction_status``
lists the tasks and counts DynamoDB statuses. Cost: about $0.10 per task-hour (2 vCPU, 8 GB).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr

from .config import Settings

WORKER_SOURCE = Path(__file__).resolve().parents[2] / "infra" / "extract_worker.py"
WORKER_KEY = "worker/extract.py"
WAITING = ("pdf_uploaded", "extract_failed")


def stack_outputs(settings: Settings) -> dict[str, str]:
    stack = os.environ.get("KIRO_WIKI_STACK")
    if not stack:
        raise RuntimeError("KIRO_WIKI_STACK is not set; source .byeori.env")
    client = boto3.Session(region_name=settings.aws_region).client("cloudformation")
    outputs = client.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}


def waiting_stems(settings: Settings, *, statuses: tuple[str, ...] = WAITING) -> list[str]:
    table = boto3.Session(region_name=settings.aws_region).resource("dynamodb").Table(settings.aws_table)
    condition = Attr("id_kind").eq("stem") & Attr("ingest_status").is_in(list(statuses))
    request: dict[str, Any] = {"FilterExpression": condition, "ProjectionExpression": "work_id"}
    stems: list[str] = []
    while True:
        page = table.scan(**request)
        stems.extend(item["work_id"] for item in page.get("Items", []))
        if not page.get("LastEvaluatedKey"):
            break
        request["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    return sorted(stems)


PREPROCESS_CHOICES = ("ocr", "shrink")


def worker_environment(job_key: str, *, force: bool = False, preprocess: str | None = None) -> list[dict[str, str]]:
    """The worker container's overrides: its job, and how to read PDFs GROBID cannot read as they are."""
    if preprocess and preprocess not in PREPROCESS_CHOICES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_CHOICES}")
    env = [{"name": "JOB_KEY", "value": job_key}]
    if force:
        env.append({"name": "FORCE", "value": "1"})
    if preprocess:
        env.append({"name": "PREPROCESS", "value": preprocess})
    return env


def run_extraction(settings: Settings, *, stems: list[str] | None = None, tasks: int = 4, limit: int = 0,
                   force: bool = False, dry_run: bool = False, preprocess: str | None = None) -> dict[str, Any]:
    if preprocess and preprocess not in PREPROCESS_CHOICES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_CHOICES}")
    if not settings.aws_bucket or not settings.aws_table:
        raise RuntimeError("bucket and table must be configured")
    if not 1 <= tasks <= 20:
        raise ValueError("tasks must be 1 to 20")
    selected = sorted(stems) if stems else waiting_stems(settings)
    if limit:
        selected = selected[:limit]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tasks = min(tasks, max(1, len(selected)))
    shards = [selected[i::tasks] for i in range(tasks)] if selected else []
    result: dict[str, Any] = {"run_id": run_id, "selected": len(selected), "tasks": len(shards),
                              "shard_sizes": [len(s) for s in shards], "dry_run": dry_run, "started": []}
    if dry_run or not selected:
        return result
    session = boto3.Session(region_name=settings.aws_region)
    s3 = session.client("s3")
    s3.upload_file(str(WORKER_SOURCE), settings.aws_bucket, WORKER_KEY)
    outputs = stack_outputs(settings)
    ecs = session.client("ecs")
    for index, shard in enumerate(shards):
        job_key = f"jobs/{run_id}/{index}.json"
        s3.put_object(Bucket=settings.aws_bucket, Key=job_key, Body=json.dumps(shard).encode("utf-8"),
                      ContentType="application/json")
        env = worker_environment(job_key, force=force, preprocess=preprocess)
        response = ecs.run_task(
            cluster=outputs["ExtractionClusterName"], taskDefinition=outputs["ExtractionTaskDefinitionArn"],
            launchType="FARGATE", count=1, startedBy=f"byeori-extract-{run_id}",
            networkConfiguration={"awsvpcConfiguration": {
                "subnets": outputs["ExtractionSubnetIds"].split(","),
                "securityGroups": [outputs["ExtractionSecurityGroupId"]], "assignPublicIp": "ENABLED"}},
            overrides={"containerOverrides": [{"name": "worker", "environment": env}]},
        )
        failures = response.get("failures") or []
        if failures:
            raise RuntimeError(f"run_task failed: {failures}")
        arn = response["tasks"][0]["taskArn"]
        result["started"].append({"job_key": job_key, "task_arn": arn, "papers": len(shard)})
        print(f"task {index}: {len(shard)} papers -> {arn.rsplit('/', 1)[-1]}", file=sys.stderr, flush=True)
    report = settings.state_dir / f"extract-{run_id}.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["report"] = str(report)
    return result


def extraction_status(settings: Settings) -> dict[str, Any]:
    session = boto3.Session(region_name=settings.aws_region)
    outputs = stack_outputs(settings)
    ecs = session.client("ecs")
    cluster = outputs["ExtractionClusterName"]
    tasks: list[dict[str, Any]] = []
    for status in ("RUNNING", "PENDING", "STOPPED"):
        arns = ecs.list_tasks(cluster=cluster, desiredStatus=status).get("taskArns", [])
        if arns:
            for task in ecs.describe_tasks(cluster=cluster, tasks=arns[:100])["tasks"]:
                tasks.append({"task": task["taskArn"].rsplit("/", 1)[-1], "status": task["lastStatus"],
                              "started_by": task.get("startedBy"), "stopped_reason": task.get("stoppedReason"),
                              "started_at": str(task.get("startedAt", "")), "stopped_at": str(task.get("stoppedAt", ""))})
    table = session.resource("dynamodb").Table(settings.aws_table)
    counts: dict[str, int] = {}
    request: dict[str, Any] = {"FilterExpression": Attr("id_kind").eq("stem"), "ProjectionExpression": "ingest_status"}
    while True:
        page = table.scan(**request)
        for item in page.get("Items", []):
            counts[item.get("ingest_status", "none")] = counts.get(item.get("ingest_status", "none"), 0) + 1
        if not page.get("LastEvaluatedKey"):
            break
        request["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    return {"cluster": cluster, "tasks": tasks, "papers_by_status": counts}
