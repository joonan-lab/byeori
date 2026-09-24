"""Start and watch the AWS-side notes run.

Nothing here does the work. The loop, the planning and the retries all happen in Step Functions,
so these two calls are the whole local side of a thirty-hour job: one to start it, one to ask how
it is going. That is deliberate. A run used to live inside a process on one laptop, which meant it
ended when the laptop changed networks and could not be seen at all from the other machine.
"""

from __future__ import annotations

import os
from typing import Any

from .config import Settings


def _state_machine_arn(settings: Settings, output_key: str = "NotesStateMachineArn") -> str:
    import boto3
    stack = os.environ.get("KIRO_WIKI_STACK")
    if not stack:
        raise ValueError("KIRO_WIKI_STACK is not set; source .byeori.env")
    session = boto3.Session(region_name=settings.aws_region)
    outputs = session.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]["Outputs"]
    for output in outputs:
        if output["OutputKey"] == output_key:
            return output["OutputValue"]
    raise ValueError(f"stack {stack} has no {output_key} output; deploy the template first")


def start_run(settings: Settings, output_key: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start an execution of the state machine named by a stack output, with the given JSON input."""
    import json
    import boto3
    client = boto3.Session(region_name=settings.aws_region).client("stepfunctions")
    kwargs: dict[str, Any] = {"stateMachineArn": _state_machine_arn(settings, output_key)}
    if payload is not None:
        kwargs["input"] = json.dumps(payload)
    started = client.start_execution(**kwargs)
    return {"execution": started["executionArn"], "started_at": started["startDate"].isoformat(), "input": payload}


def run_status(settings: Settings, output_key: str, execution: str | None = None) -> dict[str, Any]:
    """Status of a run, and of the maps inside it - which is where the per-page counts live."""
    import boto3
    client = boto3.Session(region_name=settings.aws_region).client("stepfunctions")
    arn = execution
    if not arn:
        listed = client.list_executions(stateMachineArn=_state_machine_arn(settings, output_key), maxResults=1)
        if not listed["executions"]:
            return {"status": "no run has been started"}
        arn = listed["executions"][0]["executionArn"]
    described = client.describe_execution(executionArn=arn)
    out: dict[str, Any] = {"execution": arn, "status": described["status"],
                           "started_at": described["startDate"].isoformat()}
    if described.get("stopDate"):
        out["stopped_at"] = described["stopDate"].isoformat()
    out["maps"] = []
    for run in client.list_map_runs(executionArn=arn).get("mapRuns", []):
        detail = client.describe_map_run(mapRunArn=run["mapRunArn"])
        counts = detail.get("itemCounts", {})
        done = counts.get("succeeded", 0) + counts.get("failed", 0) + counts.get("timedOut", 0)
        total = counts.get("total", 0)
        out["maps"].append({
            "started_at": run["startDate"].isoformat(), "status": detail.get("status"),
            "total": total, "done": done, "succeeded": counts.get("succeeded", 0),
            "failed": counts.get("failed", 0) + counts.get("timedOut", 0),
            "percent": round(100 * done / total, 1) if total else 0.0,
        })
    return out


def start_notes(settings: Settings) -> dict[str, Any]:
    """Start the notes run. It takes no arguments: what to cover is decided in AWS, from the catalogue."""
    return start_run(settings, "NotesStateMachineArn")


def notes_status(settings: Settings, execution: str | None = None) -> dict[str, Any]:
    return run_status(settings, "NotesStateMachineArn", execution)
