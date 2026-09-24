"""Start the figure-and-table worker for one paper, the moment that paper's text is stored.

A paper's figures used to arrive in a batch, hours or days after the paper did, because the
extraction ran on a laptop over thousands of PDFs at once. There is no batch any more: the image
holds marker and its models, one paper converts in seconds, and this is what notices the paper.

What it listens for is the end of extraction rather than the arrival of the PDF. A PDF alone is
not yet a paper here; the stored text is what says the paper is in. Both ingest routes announce
that in their own way and both are watched:

    papers/{stem}/clean.md    an uploaded original, extracted by the GROBID worker
    sources/{work_id}.md      an OpenAlex-hosted original, extracted on the way in

Nothing here opens a PDF. It works out which paper the object belongs to, checks that the paper
does not already have its crops, and starts one Fargate task for it. The worker itself decides
what a figure is.
"""
from __future__ import annotations

import json
import os
from typing import Any

import boto3

BUCKET = os.environ.get("BUCKET_NAME") or ""
CLUSTER = os.environ.get("ASSET_CLUSTER") or ""
TASK_DEFINITION = os.environ.get("ASSET_TASK_DEFINITION") or ""
SUBNETS = [part for part in os.environ.get("ASSET_SUBNETS", "").split(",") if part]
SECURITY_GROUPS = [part for part in os.environ.get("ASSET_SECURITY_GROUPS", "").split(",") if part]
CONTAINER = os.environ.get("ASSET_CONTAINER") or "worker"

s3 = boto3.client("s3")
ecs = boto3.client("ecs")


def stem_of(key: str) -> str | None:
    """The paper an extraction object belongs to, or ``None`` if the object is not one.

    ``wiki/sources/`` is the evidence notes, which are written about papers rather than being
    them, so only a key that starts at ``sources/`` counts.
    """
    if key.startswith("papers/") and key.endswith("/clean.md"):
        stem = key[len("papers/"): -len("/clean.md")]
    elif key.startswith("sources/") and key.endswith(".md"):
        stem = key[len("sources/"): -len(".md")]
    else:
        return None
    return stem if stem and "/" not in stem else None


def already_done(stem: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=f"papers/{stem}/assets/assets.md")
        return True
    except Exception:  # noqa: BLE001 - absent, or unreadable; either way, let the worker decide
        return False


def start(stem: str) -> str:
    response = ecs.run_task(
        cluster=CLUSTER,
        taskDefinition=TASK_DEFINITION,
        launchType="FARGATE",
        count=1,
        startedBy=f"byeori-assets-{stem}"[:36],
        networkConfiguration={"awsvpcConfiguration": {
            "subnets": SUBNETS, "securityGroups": SECURITY_GROUPS, "assignPublicIp": "ENABLED"}},
        overrides={"containerOverrides": [
            {"name": CONTAINER, "environment": [{"name": "STEMS", "value": stem}]}]},
    )
    failures = response.get("failures") or []
    if failures:
        raise RuntimeError(f"run_task refused {stem}: {failures}")
    return response["tasks"][0]["taskArn"]


def handler(event: dict[str, Any], _context: Any = None) -> dict[str, Any]:
    """One EventBridge ``Object Created`` notification from the data bucket."""
    detail = event.get("detail") or {}
    key = ((detail.get("object") or {}).get("key")) or ""
    stem = stem_of(key)
    if stem is None:
        return {"skipped": "not an extraction", "key": key}
    if already_done(stem):
        return {"skipped": "assets already stored", "stem": stem}
    if not (CLUSTER and TASK_DEFINITION and SUBNETS and SECURITY_GROUPS):
        raise RuntimeError("the asset task is not configured: cluster, task definition, subnets "
                           "and security groups are all required")
    arn = start(stem)
    print(json.dumps({"stem": stem, "key": key, "task": arn.rsplit("/", 1)[-1]}))
    return {"started": stem, "task_arn": arn}
