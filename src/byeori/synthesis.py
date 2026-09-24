"""Local side of the synthesis run: start it, watch it, review its manifests, write one page for a pilot.

Nothing here does the work; the state machine and the synthesis Lambda do. These are the calls a
laptop makes to operate them, and every one of them finishes in seconds except ``synthesis_page``,
which waits for one page on purpose.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from . import costs
from .config import Settings
from .extract import stack_outputs
from .runs import run_status, start_run
from .synthesis_support import HGNC_URL
from .synthesis_manifest import scope_categories

STATE_MACHINE = "SynthesisStateMachineArn"
LAMBDA_CONFIG = Config(read_timeout=920, connect_timeout=10, retries={"max_attempts": 0})


def start_synthesis(settings: Settings, *, scope: str, plan_only: bool = False, categories: list[str] | None = None,
                    skip_concepts: bool = False, replan: bool = False) -> dict[str, Any]:
    scope_categories(scope)  # raises on an unknown scope before anything starts
    payload = {"scope": scope, "plan_only": plan_only, "categories": list(categories) if categories else None,
               "skip_concepts": skip_concepts, "replan": replan}
    return start_run(settings, STATE_MACHINE, payload)


def synthesis_status(settings: Settings, execution: str | None = None) -> dict[str, Any]:
    return run_status(settings, STATE_MACHINE, execution)


def _function_name(settings: Settings) -> str:
    name = stack_outputs(settings).get("SynthesisFunctionName")
    if not name:
        raise RuntimeError("the stack has no SynthesisFunctionName output; deploy the template first")
    return name


def invoke_synthesis(settings: Settings, payload: dict[str, Any], function_name: str | None = None) -> dict[str, Any]:
    name = function_name or _function_name(settings)
    client = boto3.Session(region_name=settings.aws_region).client("lambda", config=LAMBDA_CONFIG)
    response = client.invoke(FunctionName=name, InvocationType="RequestResponse", Payload=json.dumps(payload).encode("utf-8"))
    body = json.loads(response["Payload"].read())
    if response.get("FunctionError"):
        message = body.get("errorMessage") if isinstance(body, dict) else body
        error_type = body.get("errorType") if isinstance(body, dict) else None
        detail = f"{error_type}: {message}" if error_type else str(message)
        raise RuntimeError(f"synthesis Lambda failed: {detail}")
    return body


def synthesis_page(settings: Settings, *, kind: str, ident: str, mode: str = "generate", force: bool = False,
                   model_id: str = "global.anthropic.claude-opus-5", max_rounds: int = 30,
                   established: str = "", corrections: str = "", evidence_notes: list[str] | None = None) -> dict[str, Any]:
    """Write one page synchronously (the pilot): call the Lambda until it stops answering 'partial'.

    With mode='update' the page is revised rather than rewritten: ``established`` is what has just
    been shown, ``corrections`` names what the page currently gets wrong, and ``evidence_notes``
    are the stems behind them, which join the page's members so their links validate.
    """
    if kind == "concept":
        payload: dict[str, Any] = {"action": "page", "kind": "concept", "slug": ident, "mode": mode}
    elif kind == "subtopic":
        category, _, slug = ident.partition("/")
        if not slug:
            raise ValueError("a subtopic id is category/slug, such as asd-ndd/de-novo-variants")
        payload = {"action": "page", "kind": "subtopic", "category": category, "slug": slug, "mode": mode}
    elif kind == "category":
        payload = {"action": "page", "kind": "category", "category": ident, "force": force}
    else:
        raise ValueError("kind must be concept, subtopic, or category")
    if mode == "update":
        if kind == "category":
            raise ValueError("a category page is assembled from its subtopic pages; update those instead")
        if not (established or corrections):
            raise ValueError("an update needs established, corrections, or both")
        payload.update(established=established, corrections=corrections, evidence_notes=evidence_notes or [])
    function_name = _function_name(settings)
    started, rounds = time.monotonic(), 0
    total_usd = 0.0
    while True:
        result = invoke_synthesis(settings, payload, function_name=function_name)
        rounds += 1
        model = result.get("model") or model_id
        usage = {"inputTokens": result.get("input_tokens") or 0, "outputTokens": result.get("output_tokens") or 0,
                 "cacheReadInputTokens": result.get("cache_read_tokens") or 0, "cacheWriteInputTokens": result.get("cache_write_tokens") or 0}
        estimated_usd = split_synthesis_usd(model, result)
        total_usd += estimated_usd or 0.0
        costs.record(settings.state_dir, {
            "step": f"synthesis_{kind}", "work_id": ident, "status": result.get("status"), "model_id": model,
            "seconds": result.get("seconds"), "input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"],
            "estimated_usd": estimated_usd,
            "basis": "Anthropic list prices applied to reported token counts; not reconciled against the AWS bill"})
        print(f"round {rounds}: {result.get('status')}, {result.get('calls')} calls, {result.get('seconds')} s, ${total_usd:.4f} so far",
              file=sys.stderr)
        if result.get("status") != "partial":
            return {**result, "rounds": rounds, "wall_seconds": round(time.monotonic() - started, 1)}
        if rounds >= max_rounds:
            return {**result, "rounds": rounds, "status": "partial", "wall_seconds": round(time.monotonic() - started, 1)}


def split_synthesis_usd(model: str, result: dict[str, Any]) -> float | None:
    """Price a synthesis invocation's tokens by the model that spent them.

    The Lambda reports every text-producing call in ``input_tokens``/``output_tokens`` and names the
    fallback model's share (``fallback_*``) and the calls the primary model declined (``filtered_*``,
    paid for but discarded) separately, so each part gets its own model's rates.
    """
    fb_in, fb_out = int(result.get("fallback_input_tokens") or 0), int(result.get("fallback_output_tokens") or 0)
    primary = {"inputTokens": int(result.get("input_tokens") or 0) - fb_in,
               "outputTokens": int(result.get("output_tokens") or 0) - fb_out,
               "cacheReadInputTokens": int(result.get("cache_read_tokens") or 0),
               "cacheWriteInputTokens": int(result.get("cache_write_tokens") or 0)}
    total = costs.estimate_draft_usd(model, primary)
    if total is None:
        return None
    if fb_in or fb_out:
        total += costs.estimate_draft_usd(result.get("fallback_model_id") or model,
                                          {"inputTokens": fb_in, "outputTokens": fb_out}) or 0.0
    declined = {"inputTokens": int(result.get("filtered_input_tokens") or 0),
                "outputTokens": int(result.get("filtered_output_tokens") or 0)}
    if declined["inputTokens"] or declined["outputTokens"]:
        total += costs.estimate_draft_usd(model, declined) or 0.0
    return round(total, 5)


def pull_manifests(settings: Settings, *, scope: str) -> dict[str, Any]:
    """Ask AWS for counts and review summaries; full manifests stay in S3."""
    return invoke_synthesis(settings, {"action": "manifest_summary", "scope": scope})


def read_manifest(settings: Settings, *, kind: str, category: str | None = None, section: str | None = None,
                  offset: int = 0, max_chars: int = 4000) -> dict[str, Any]:
    """Read a bounded excerpt, optionally selecting a concept, subtopic, or overrides field."""
    return invoke_synthesis(settings, {"action": "manifest_read", "kind": kind, "category": category,
                                      "section": section, "offset": offset, "max_chars": max_chars})


def push_manifest(settings: Settings, path: Path | None = None, *, content: str | dict | None = None,
                  kind: str | None = None, category: str | None = None) -> dict[str, Any]:
    """Submit supplied content for AWS validation; existing user input files are read only."""
    if path is not None:
        if content is not None:
            raise ValueError("provide content or an existing input file, not both")
        path = Path(path)
        if path.name not in ("overrides.json", "subtopics.json"):
            raise ValueError("input file must be overrides.json or subtopics.json")
        kind = kind or path.stem
        content = path.read_text(encoding="utf-8")
    if content is None:
        raise ValueError("manifest content is required")
    return invoke_synthesis(settings, {"action": "manifest_submit", "kind": kind, "category": category,
                                      "content": content})


def upload_hgnc(settings: Settings, url: str = HGNC_URL) -> dict[str, Any]:
    """Ask AWS to fetch, validate and store the configured HGNC reference."""
    return invoke_synthesis(settings, {"action": "reference", "url": url})


def failed_synthesis(settings: Settings, *, verbose: bool = False, offset: int = 0) -> dict[str, Any]:
    """Read the bounded AWS report of failed synthesis attempts."""
    return invoke_synthesis(settings, {"action": "failures", "verbose": verbose, "offset": offset})
