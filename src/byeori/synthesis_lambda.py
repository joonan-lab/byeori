"""The synthesis Lambda: plans and writes concept, subtopic and category pages from the notes in S3.

Deployed from this package (``Code: ../src`` in infra/template.yaml, zipped by
``aws cloudformation package``), so the rules it applies are the modules the tests import. Every
action takes and returns plain JSON; the state machine in the template strings them together.
Nothing here is called by the notes state machine.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal

from botocore.exceptions import ClientError

from . import synthesis_pages as pages
from .synthesis_manifest import (catalog_line, chunked, member_digest, note_metadata, parse_frontmatter, scope_categories,
                                 stale_mode, synthesis_input, validate_partition)
from .synthesis_terms import (STOP_TERMS, HgncTable, MentionIndex, apply_title_overrides, co_occurrence,
                              count_candidates, merge_candidates, note_terms, safe_model_merges)
from .wiki_connections import BACKLINK_BLOCK, publish_page

BUCKET = os.environ.get("BUCKET_NAME", "")
TABLE = os.environ.get("TABLE_NAME", "")
# Synthesis has its own model and effort (user, 2026-09-23: Opus 5.5 at xhigh). It used to take
# DraftModelId and IngestReasoning, which is how it went from xhigh to high when notes did on
# 2026-09-22. Opus 5.5's biology classifier declines some biomedical input, so a filtered call is
# made again by the fallback model; a trial never falls back, because it is there to see one model.
CONFIGURED_MODEL_ID = os.environ.get("SYNTHESIS_MODEL_ID") or os.environ.get("DRAFT_MODEL_ID", "global.anthropic.claude-opus-5")
CONFIGURED_REASONING = os.environ.get("SYNTHESIS_REASONING") or os.environ.get("INGEST_REASONING", "default")
FALLBACK_MODEL_ID = os.environ.get("SYNTHESIS_FALLBACK_MODEL_ID", "")
MODEL_ID = CONFIGURED_MODEL_ID
REASONING = CONFIGURED_REASONING
TRIAL: dict | None = None
TRIAL_RUN_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
MAX_PAGES = int(os.environ.get("SYNTHESIS_MAX_PAGES", "1500"))
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
ENTITY_TYPES = ("gene", "method", "cohort", "phenomenon", "other")
HGNC_KEY = "reference/hgnc.tsv"
CANDIDATES_KEY = "runs/synthesis/concepts/candidates.json"
PLAN_ITEM = "synthesis#concept-plan"   # the catalogue item that carries the planning spend; outside the concept# namespace
OVERRIDES_KEY = "runs/synthesis/concepts/overrides.json"
THRESHOLD = 5
NOTES_PER_CALL = 15            # about 200k characters of sections 2-5
PARTIALS_PER_MERGE = 15
MAX_CALLS_PER_INVOCATION = 3   # three Opus calls at xhigh fit inside the 900 s Lambda limit
MAX_INPUT_CHARS = 400_000
PAGE_MAX_TOKENS = 8_000
PLAN_MAX_TOKENS = 32_000       # a partition lists every stem of a category
PARTITION_SPLIT = 250          # a category with more notes is proposed in halves, then merged
PARTITION_LIMITS = {"min_subtopics": 4, "max_subtopics": 12, "min_notes": 5}
def partition_limits(note_count: int) -> dict:
    return {"min_subtopics": 3,
            "max_subtopics": max(8, note_count // PARTITION_MIN_NOTES),
            "min_notes": PARTITION_MIN_NOTES}
CANDIDATE_CHUNK = 100
TYPING_CALLS_PER_INVOCATION = 3
TYPED_KEY = "runs/synthesis/concepts/typed.json"
CONCEPT_IDENTITY_VERSION = 2
TYPING_ATTEMPTS_LIMIT = 2      # a slug that fails this many typing attempts is admitted as "other"
MENTION_POSTING_CAP = 400
TIME_BUDGET_MS = 240_000       # stop starting new calls with less than this much of the invocation left
SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
CATEGORY_RE = re.compile(r"[a-z0-9-]+")
DEADLINE_MS: float | None = None


class _Aws:
    """Lazy clients, so the module imports without credentials; tests replace this object."""

    def __init__(self) -> None:
        self._s3 = self._table = self._bedrock = None

    @property
    def s3(self):
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3")
        return self._s3

    @property
    def table(self):
        if self._table is None:
            import boto3
            self._table = boto3.resource("dynamodb").Table(TABLE)
        return self._table

    @property
    def bedrock(self):
        if self._bedrock is None:
            import boto3
            import botocore.config
            self._bedrock = boto3.client("bedrock-runtime", config=botocore.config.Config(
                read_timeout=880, connect_timeout=10, retries={"max_attempts": 0}))
        return self._bedrock

    def bedrock_with_timeout(self, seconds: int):
        import boto3
        import botocore.config
        return boto3.client("bedrock-runtime", config=botocore.config.Config(
            read_timeout=seconds, connect_timeout=10, retries={"max_attempts": 0}))


aws = _Aws()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return _now().date().isoformat()


def _stamp() -> str:
    return _now().strftime("%Y%m%dT%H%M%SZ")


def _time_left_ms() -> float:
    """Milliseconds until this Lambda invocation's own deadline, set by ``handler`` from the context
    it receives; infinity when there is none (no context, e.g. under test or invoked locally), so a
    time-budget check never trips outside a real invocation."""
    if DEADLINE_MS is None:
        return float("inf")
    return DEADLINE_MS - _now().timestamp() * 1000


def _get_text(key: str) -> str:
    return aws.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8")


def _publication_snapshot(key: str):
    """Capture the published body and its version before any model call can revise it."""
    try:
        response = aws.s3.get_object(Bucket=BUCKET, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    stream = response["Body"]
    try:
        text = stream.read().decode("utf-8")
    finally:
        stream.close()
    if not response.get("ETag"):
        raise RuntimeError(f"S3 returned no ETag for {key}")
    return {"text": text, "etag": response["ETag"]}


def _published_model_body(snapshot, kind):
    """Read live science while leaving code-owned navigation to the page builder."""
    _, body = parse_frontmatter(snapshot["text"])
    body = BACKLINK_BLOCK.sub("", body)
    owned = {"Related concepts", "Notes"} if kind == "concept" else {"Concepts", "Notes"}
    parts = re.split(r"^(## .+)$", body, flags=re.M)
    return (parts[0] + "".join(heading + content for heading, content in zip(parts[1::2], parts[2::2])
                               if heading[3:].strip() not in owned)).strip()


def _get_json(key: str, default=None):
    try:
        return json.loads(_get_text(key))
    except aws.s3.exceptions.NoSuchKey:
        return default


def _put_text(key: str, text: str, content_type: str = "text/markdown; charset=utf-8", *,
              model_id: str | None = None) -> None:
    extra = {"Metadata": {"model_id": model_id}} if model_id else {}
    aws.s3.put_object(Bucket=BUCKET, Key=key, Body=text.encode("utf-8"), ContentType=content_type, **extra)


def _fallback_partials(keys: list[str]) -> int:
    """How many of a page's stored partials the fallback model wrote, read from their metadata."""
    if not FALLBACK_MODEL_ID:
        return 0
    return sum(1 for key in keys
               if (aws.s3.head_object(Bucket=BUCKET, Key=key).get("Metadata") or {}).get("model_id") == FALLBACK_MODEL_ID)


def _put_json(key: str, value) -> None:
    _put_text(key, json.dumps(value, ensure_ascii=False, indent=1), "application/json")


def _exists(key: str) -> bool:
    try:
        aws.s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except aws.s3.exceptions.NoSuchKey:
        return False
    except Exception as exc:
        import botocore.exceptions
        if isinstance(exc, botocore.exceptions.ClientError):
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
        raise


def _read_many(keys: list[str], missing: list | None = None) -> dict[str, str]:
    """Read every key in parallel; a key that no longer exists is dropped from the result and, if
    the caller passed a list, appended there instead of raising."""
    def read(key: str):
        try:
            return key, _get_text(key)
        except aws.s3.exceptions.NoSuchKey:
            return key, None
    with ThreadPoolExecutor(max_workers=32) as pool:
        pairs = list(pool.map(read, keys))
    result: dict[str, str] = {}
    for key, text in pairs:
        if text is None:
            if missing is not None:
                missing.append(key)
        else:
            result[key] = text
    return result


def _scan(condition, projection: str, names: dict[str, str] | None = None) -> list[dict]:
    request = {"FilterExpression": condition, "ProjectionExpression": projection}
    if names:
        request["ExpressionAttributeNames"] = names
    items: list[dict] = []
    while True:
        page = aws.table.scan(**request)
        items.extend(page.get("Items", []))
        if not page.get("LastEvaluatedKey"):
            break
        request["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    return items


def _ready_notes(categories=None) -> list[dict]:
    """Every paper with a validated note: work_id, source_note_key, source_note_sha256, category."""
    from boto3.dynamodb.conditions import Attr
    items = _scan(Attr("id_kind").eq("stem") & Attr("source_note_status").eq("source_ready"),
                  "work_id, source_note_key, source_note_sha256, category")
    if categories is not None:
        items = [i for i in items if i.get("category") in set(categories)]
    return sorted(items, key=lambda i: i["work_id"])


def _scan_kind(kind: str) -> list[dict]:
    from boto3.dynamodb.conditions import Attr
    return _scan(Attr("id_kind").eq(kind), "work_id, id_kind, #st, members, model_text_key, generation, created",
                 names={"#st": "synthesis_status"})


def _item(work_id: str) -> dict:
    return aws.table.get_item(Key={"work_id": work_id}).get("Item") or {}


def _record(work_id: str, id_kind: str, fields: dict, add: dict | None = None) -> None:
    """SET the given attributes (and ADD counters) on the page's catalogue item, creating it if new."""
    if TRIAL is not None:
        return   # a trial is compared, not recorded: the catalogue keeps describing the wiki
    names: dict[str, str] = {}
    values: dict = {":kind": id_kind, ":at": _now().replace(microsecond=0).isoformat()}
    sets = ["id_kind = :kind", "updated_at = :at"]
    for i, (k, v) in enumerate(fields.items()):
        names[f"#s{i}"], values[f":s{i}"] = k, v
        sets.append(f"#s{i} = :s{i}")
    adds = []
    for i, (k, v) in enumerate((add or {}).items()):
        names[f"#a{i}"], values[f":a{i}"] = k, v
        adds.append(f"#a{i} :a{i}")
    expression = "SET " + ", ".join(sets) + (" ADD " + ", ".join(adds) if adds else "")
    request = {"Key": {"work_id": work_id}, "UpdateExpression": expression, "ExpressionAttributeValues": values}
    if names:
        request["ExpressionAttributeNames"] = names
    aws.table.update_item(**request)


class Usage:
    """Tokens of the calls that produced text, with the fallback's share and the declined attempts
    kept apart, because the two models are priced differently and a declined call is still paid for."""

    def __init__(self) -> None:
        self.calls, self.input, self.output, self.seconds = 0, 0, 0, 0.0
        self.cache_read, self.cache_write = 0, 0
        self.fallback = {"calls": 0, "input": 0, "output": 0}
        self.filtered = {"calls": 0, "input": 0, "output": 0}

    def add(self, result: dict) -> None:
        self.calls += 1
        self.input += int(result["usage"].get("inputTokens", 0))
        self.output += int(result["usage"].get("outputTokens", 0))
        self.cache_read += int(result["usage"].get("cacheReadInputTokens", 0))
        self.cache_write += int(result["usage"].get("cacheWriteInputTokens", 0))
        self.seconds += float(result.get("seconds", 0))
        declined = result.get("filtered_attempt")
        if declined:
            self.filtered["calls"] += 1
            self.filtered["input"] += int(declined["usage"].get("inputTokens", 0))
            self.filtered["output"] += int(declined["usage"].get("outputTokens", 0))
            self.seconds += float(declined.get("seconds", 0))
        if FALLBACK_MODEL_ID and result.get("model_id") == FALLBACK_MODEL_ID and result.get("model_id") != MODEL_ID:
            self.fallback["calls"] += 1
            self.fallback["input"] += int(result["usage"].get("inputTokens", 0))
            self.fallback["output"] += int(result["usage"].get("outputTokens", 0))

    def _split(self) -> dict:
        split = {}
        for name, counts in (("fallback", self.fallback), ("filtered", self.filtered)):
            if counts["calls"]:
                split |= {f"{name}_calls": counts["calls"], f"{name}_input_tokens": counts["input"],
                          f"{name}_output_tokens": counts["output"]}
        return split

    def fields(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input, "output_tokens": self.output,
                "cache_read_tokens": self.cache_read, "cache_write_tokens": self.cache_write,
                "seconds": Decimal(str(round(self.seconds, 1))), **self._split()}

    def json(self) -> dict:
        # Names are for the caller that prices the run; `fields` stays numeric, since it is ADDed.
        names = {"fallback_model_id": FALLBACK_MODEL_ID} if self.fallback["calls"] else {}
        return {"calls": self.calls, "input_tokens": self.input, "output_tokens": self.output,
                "cache_read_tokens": self.cache_read, "cache_write_tokens": self.cache_write,
                "seconds": round(self.seconds, 1), **self._split(), **names}


# Adaptive thinking and the answer share one maxTokens budget, so the budget has to cover the
# reasoning before a single character of output is written. A flat 24,000 was not enough at xhigh:
# on 2026-09-20 the asd-models and asd-ndd partitions spent all 30,000 tokens thinking, returned an
# empty string with stopReason max_tokens, and their category plans were discarded. Opus 5 accepts
# up to 128,000, so the headroom now scales with the effort that decides how much thinking happens.
THINKING_HEADROOM = {"low": 8000, "medium": 16000, "high": 32000, "xhigh": 56000, "max": 96000}
LOWER_EFFORT = {"max": "xhigh", "xhigh": "high", "high": "medium", "medium": "low"}


# A throttled Bedrock call is rejected before any inference runs, so retrying it costs nothing and
# bills nothing twice. A read timeout is the opposite: the model may still be generating, and a
# retry would run it, and bill it, again - which is why botocore's own retries stay off. So only
# the pre-inference rejections are retried here, never a timeout and never an ambiguous 5xx.
THROTTLE_CODES = ("ThrottlingException", "TooManyRequestsException",
                  "ServiceUnavailableException", "ModelNotReadyException")
THROTTLE_MAX_WAIT_SECONDS = 150


def _converse_with_backoff(client, request, *, budget=THROTTLE_MAX_WAIT_SECONDS):
    """Call Bedrock, waiting out throttling. Returns (response, seconds spent waiting, attempts)."""
    delay, waited, attempts = 2.0, 0.0, 0
    while True:
        attempts += 1
        try:
            return client.converse(**request), round(waited, 1), attempts
        except botocore.exceptions.ClientError as exc:
            code = (exc.response.get("Error") or {}).get("Code")
            if code not in THROTTLE_CODES or waited + delay > budget:
                raise
            pause = delay + random.uniform(0, 1)
            time.sleep(pause)
            waited += pause
            delay = min(delay * 2, 30.0)


def _generate(system: str, prompt: str, *, max_tokens: int = PAGE_MAX_TOKENS,
              timeout_seconds: int | None = None, effort: str | None = None) -> dict:
    """One Bedrock Converse call. Opus 5 takes adaptive thinking steered by output_config.effort;
    the reasoning blocks carry no text and never reach a page."""
    started = _now()
    effort = effort or REASONING
    thinking = effort in EFFORT_LEVELS
    request = {"modelId": MODEL_ID, "system": [{"text": system}],
               "messages": [{"role": "user", "content": [{"text": prompt}]}],
               "inferenceConfig": {"maxTokens": max_tokens + (THINKING_HEADROOM.get(effort, 24000) if thinking else 0)}}
    if thinking:
        request["additionalModelRequestFields"] = {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    client = aws.bedrock if timeout_seconds is None else aws.bedrock_with_timeout(timeout_seconds)
    response, throttled_seconds, attempts = _converse_with_backoff(client, request)
    model_id, declined = MODEL_ID, None
    if (response.get("stopReason") == "content_filtered" and FALLBACK_MODEL_ID and FALLBACK_MODEL_ID != MODEL_ID
            and (TRIAL is None or TRIAL.get("fallback"))):
        declined = {"model_id": MODEL_ID,
                    "usage": {k: int(v) for k, v in (response.get("usage") or {}).items() if isinstance(v, int)},
                    "seconds": round((_now() - started).total_seconds(), 1)}
        model_id = FALLBACK_MODEL_ID
        response, waited, more = _converse_with_backoff(client, {**request, "modelId": model_id})
        throttled_seconds, attempts = throttled_seconds + waited, attempts + more
    text = "".join(block.get("text", "") for block in response["output"]["message"]["content"]).strip()
    usage = response.get("usage") or {}
    result = {"text": text, "usage": {k: int(v) for k, v in usage.items() if isinstance(v, int)},
              "request_id": response.get("ResponseMetadata", {}).get("RequestId"), "effort": effort if thinking else None,
              "throttled_seconds": throttled_seconds, "attempts": attempts, "model_id": model_id,
              "stop_reason": response.get("stopReason"), "seconds": round((_now() - started).total_seconds(), 1)}
    if declined:
        result["filtered_attempt"] = declined
    return result


def _generate_json(system: str, prompt: str, *, max_tokens: int = PLAN_MAX_TOKENS,
                   timeout_seconds: int | None = None, strict: bool = False) -> dict:
    kwargs = {} if timeout_seconds is None else {"timeout_seconds": timeout_seconds}
    parser = pages.parse_plan_json if strict else pages.parse_json

    def once(effort=None):
        # The first attempt takes the configured effort by omitting the argument entirely.
        extra = {"effort": effort} if effort else {}
        result = _generate(system, prompt, max_tokens=max_tokens, **extra, **kwargs)
        result["data"], result["problem"] = parser(result["text"])
        result["parse_problem"] = result["problem"]
        if result["stop_reason"] not in (None, "end_turn"):
            result["problem"] = result["problem"] or f"stop reason {result['stop_reason']}"
        return result

    result = once()
    # Thinking ate the whole budget and left no JSON. Raising the budget again would only move the
    # failure to the call timeout, so the retry thinks less instead of longer. One step only: a
    # second overrun is a real problem with the request, not a setting to keep walking down.
    if result["data"] is None and result["stop_reason"] == "max_tokens":
        lower = LOWER_EFFORT.get(result.get("effort") or REASONING)
        if lower:
            retry = once(lower)
            retry["retried_from_effort"] = result.get("effort")
            retry["first_attempt_usage"] = result["usage"]
            if retry["data"] is not None:
                return retry
            retry["problem"] = retry["problem"] or result["problem"]
            return retry
    return result


def _categories(event: dict):
    explicit = event.get("categories")
    if explicit:
        for c in explicit:
            if not CATEGORY_RE.fullmatch(str(c)):
                raise ValueError(f"bad category {c!r}")
        return tuple(str(c) for c in explicit)
    return scope_categories(str(event.get("scope") or "all"))


def resolve_scope(event: dict) -> dict:
    categories = _categories(event)
    if categories is None:
        categories = sorted({i.get("category") for i in _ready_notes() if i.get("category")})
    return {"scope": str(event.get("scope") or "all"), "categories": list(categories)}


def manifest_summary(event: dict) -> dict:
    """Aggregate in AWS, returning counts without any complete manifest or member list."""
    scope = resolve_scope(event)
    notes = _ready_notes()
    counts = Counter(i.get("category") for i in notes)
    report = {**scope, "concepts": None, "categories": {}, "corpus_notes": len(notes),
              "scope_notes": sum(counts[c] for c in scope["categories"]),
              "notes_by_category": {c: counts[c] for c in scope["categories"]}}
    candidates = _get_json(CANDIDATES_KEY)
    if candidates is not None:
        concepts = candidates.get("concepts") or []
        report["concepts"] = {"key": CANDIDATES_KEY, "count": len(concepts), "sha256": candidates.get("sha256"),
                              "identity_version": candidates.get("identity_version"),
                              "rejected_model_merges": len(candidates.get("rejected_model_merges") or []),
                              "by_entity_type": dict(Counter(c["entity_type"] for c in concepts)),
                              "by_mode": dict(Counter(c["mode"] for c in concepts)),
                              "largest": [[c["slug"], c["count_total"]] for c in
                                          sorted(concepts, key=lambda c: c["count_total"], reverse=True)[:15]]}
    for category in scope["categories"]:
        key = f"runs/synthesis/{category}/subtopics.json"
        manifest = _get_json(key)
        report["categories"][category] = None if manifest is None else {
            "key": key, "notes": manifest.get("note_count"), "sha256": manifest.get("sha256"),
            "problems": (manifest.get("problems") or [])[:10],
            "subtopics": [[st["slug"], len(st["stems"])] for st in manifest.get("subtopics", [])]}
    overrides = _get_json(OVERRIDES_KEY) or {}
    report["overrides"] = {"key": OVERRIDES_KEY, **{name: len(overrides.get(name) or []) for name in ("exclude", "merge", "title")}}
    report["usage"] = _synthesis_usage()
    return report


def _synthesis_usage() -> dict:
    """Cumulative AWS counters, including planning and retries, without downloading catalogue rows."""
    from boto3.dynamodb.conditions import Attr
    fields = ("calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "unknown_usage_attempts")
    rows = _scan(Attr("id_kind").is_in(["concept", "subtopic", "category", "category_plan"]) | Attr("work_id").eq(PLAN_ITEM),
                 "id_kind, " + ", ".join(fields))
    by_kind: dict[str, dict] = {}
    for row in rows:
        group = by_kind.setdefault(row.get("id_kind", "plan"), {"items": 0, **dict.fromkeys(fields, 0)})
        group["items"] += 1
        for name in fields:
            group[name] += int(row.get(name) or 0)
    return {"scope": "all recorded synthesis attempts", "by_kind": by_kind,
            "totals": {name: sum(group[name] for group in by_kind.values()) for name in ("items", *fields)}}


def _manifest_key(kind: str, category: str | None = None) -> str:
    if kind == "concepts":
        return CANDIDATES_KEY
    if kind == "overrides":
        return OVERRIDES_KEY
    if kind == "subtopics" and CATEGORY_RE.fullmatch(category or ""):
        return f"runs/synthesis/{category}/subtopics.json"
    raise ValueError("kind must be concepts, overrides, or subtopics with a category slug")


def manifest_read(event: dict) -> dict:
    """Read one JSON excerpt in AWS, with an optional concept/subtopic slug or override field."""
    kind = str(event.get("kind") or "")
    key = _manifest_key(kind, event.get("category"))
    offset = int(event.get("offset") or 0)
    max_chars = int(event.get("max_chars", 4000))
    if offset < 0 or not 1 <= max_chars <= 8000:
        raise ValueError("offset must be nonnegative and max_chars between 1 and 8000")
    manifest = _get_json(key)
    if manifest is None:
        return {"key": key, "found": False}
    section = event.get("section")
    if section:
        if kind == "overrides":
            if section not in ("exclude", "merge", "title"):
                raise ValueError("overrides section must be exclude, merge, or title")
            manifest = manifest.get(section, [] if section == "exclude" else {})
        else:
            manifest = next((item for item in manifest.get(kind, []) if item.get("slug") == section), None)
            if manifest is None:
                raise ValueError(f"manifest section not found: {section}")
    text = json.dumps(manifest, ensure_ascii=False, indent=1)
    excerpt = text[offset:offset + max_chars]
    end = offset + len(excerpt)
    return {"key": key, "found": True, "section": section, "offset": offset, "max_chars": max_chars,
            "total_chars": len(text), "next_offset": end if end < len(text) else None, "text": excerpt}


def manifest_submit(event: dict) -> dict:
    """Validate an edit against the current AWS corpus before replacing a run manifest."""
    data = event.get("content")
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError("manifest content must be a JSON object")
    kind = str(event.get("kind") or "")
    if kind == "overrides":
        exclude, merge, titles = data.get("exclude", []), data.get("merge", {}), data.get("title", {})
        if (not isinstance(exclude, list) or not all(isinstance(s, str) and SLUG_RE.fullmatch(s) for s in exclude)
                or not isinstance(merge, dict) or not all(isinstance(s, str) and SLUG_RE.fullmatch(s)
                    and isinstance(t, str) and SLUG_RE.fullmatch(t) for s, t in merge.items())
                or not isinstance(titles, dict) or not all(isinstance(s, str) and SLUG_RE.fullmatch(s)
                    and isinstance(t, str) and t.strip() for s, t in titles.items())):
            raise ValueError("overrides require exclude (slugs), merge (slug to slug), and title (slug to nonempty title)")
        key = OVERRIDES_KEY
        data = {"exclude": exclude, "merge": merge, "title": titles}
    elif kind == "subtopics":
        category = event.get("category") or data.get("category")
        key = _manifest_key(kind, category)
        if data.get("category") and data["category"] != category:
            raise ValueError("content category does not match the requested category")
        items, metas, dropped = _category_notes(category)
        if dropped:
            raise ValueError(f"cannot validate complete membership: {dropped} ready notes lack readable content or hashes")
        if not items:
            raise ValueError(f"category has no ready notes: {category}")
        problems = validate_partition(data, sorted(metas), category=category, **PARTITION_LIMITS)
        if problems:
            raise ValueError("manifest is not valid: " + "; ".join(problems[:10]))
        years = sorted(m["year"] for m in metas.values() if m["year"].isdigit())
        data = {**data, "category": category, "validated": "aws", "note_count": len(metas),
                "year_range": [years[0], years[-1]] if years else ["", ""], "updated": _today(), "problems": []}
        data["sha256"] = hashlib.sha256(json.dumps(data["subtopics"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    else:
        raise ValueError("only overrides or subtopics can be submitted")
    _put_json(key, data)
    return {"key": key, "validated": "aws", "note_count": data.get("note_count"), "sha256": data.get("sha256")}


def _brief(meta: dict) -> dict:
    return {k: meta[k] for k in ("stem", "title", "first_author", "year", "category", "sha256", "source_note_key")}


def plan_concepts(event: dict) -> dict:
    """Count concept candidates over every note, type them once with the model, decide which pages
    to write, and leave the manifest and the work list in S3 for the state machine.

    Typing is resumable: decisions are persisted to ``TYPED_KEY`` after every chunk, so a run that
    exceeds ``TYPING_CALLS_PER_INVOCATION`` chunks or the time budget returns ``status: partial``
    without writing the candidates manifest, and the next invocation picks up where this one
    stopped. A slug whose chunk keeps failing is not stuck forever: each failed attempt is counted,
    and after ``TYPING_ATTEMPTS_LIMIT`` of them the slug is admitted as
    ``{"entity_type": "other", "typing_failed": True}`` so it leaves the retry pool. An invocation
    that ends without having typed anything new never reports ``partial`` either way (that would
    just repeat the same stuck chunks forever): it treats every still-untyped slug as "other" for
    this run only (not persisted) and completes normally.
    """
    started = _now()
    scope = str(event.get("scope") or "all")
    categories = _categories(event)
    items = _ready_notes()
    keys = {i["work_id"]: i.get("source_note_key") or f"wiki/sources/{i['work_id']}.md" for i in items}
    missing: list[str] = []
    texts = _read_many(sorted(set(keys.values())), missing)
    hgnc = HgncTable.from_tsv(_get_text(HGNC_KEY)) if _exists(HGNC_KEY) else None
    if hgnc is not None and not hgnc.names:
        raise ValueError("Refresh the HGNC reference with official names before concept planning")
    overrides = _get_json(OVERRIDES_KEY, {}) or {}
    previous = _get_json(CANDIDATES_KEY, {}) or {}
    if previous.get("identity_version") != CONCEPT_IDENTITY_VERSION:
        previous = {}
    # `remembered` is only an entity_type fallback for a candidate this run does not type: whether a
    # candidate still needs typing is decided from typed.json alone, so a candidate whose typing
    # failed keeps being eligible for retry (up to TYPING_ATTEMPTS_LIMIT attempts, after which it is
    # admitted as "other" with typing_failed set) even after a manifest has been written showing it
    # as "other" - the manifest is not itself a record of a typing decision.
    remembered = {c["slug"]: c for c in previous.get("concepts", [])}
    typed_store = _get_json(TYPED_KEY, {}) or {}
    if typed_store.get("identity_version") != CONCEPT_IDENTITY_VERSION:
        typed_store = {}
    typed: dict[str, dict] = dict(typed_store.get("typed") or {})
    attempts: dict[str, int] = dict(typed_store.get("attempts") or {})
    failed_chunks_total = int(typed_store.get("failed_chunks") or 0)
    stop = set(STOP_TERMS)
    for category in {i.get("category") for i in items if i.get("category")}:
        stop |= {category, category.replace("-", " ")}
    notes, metas, bodies = [], {}, {}
    without_hash = 0
    for item in items:
        stem, key = item["work_id"], keys[item["work_id"]]
        if key not in texts:
            continue
        sha = item.get("source_note_sha256") or ""
        if not sha:
            without_hash += 1
            continue
        text = texts[key]
        meta = note_metadata(stem, text)
        meta["sha256"] = sha
        meta["category"] = meta["category"] or item.get("category") or ""
        meta["source_note_key"] = key
        metas[stem], bodies[stem] = meta, text
        notes.append(note_terms(stem, meta["category"], text, hgnc))
    in_scope = (lambda stem: True) if categories is None else (lambda stem: metas[stem]["category"] in categories)
    raw_candidates = count_candidates(notes, scope=None, threshold=1, stop_terms=stop, merge={}, exclude=[])
    prior_merges, rejected_prior = safe_model_merges(raw_candidates, previous.get("model_merges") or {})
    merge = {**prior_merges, **(overrides.get("merge") or {})}
    candidates = count_candidates(notes, scope=None if categories is None else set(categories), threshold=THRESHOLD,
                                  stop_terms=stop, merge=merge, exclude=list(overrides.get("exclude") or []))
    usage = Usage()
    new_merges = {slug: row["merge_into"] for slug, row in typed.items() if row.get("merge_into")}
    all_slugs = {c.slug for c in candidates} | set(typed)
    fresh = sorted((c for c in candidates if c.slug not in typed), key=lambda c: c.slug)
    chunks = chunked(fresh, CANDIDATE_CHUNK)
    failed_chunks = 0
    attempted = 0
    typed_before = len(typed)
    for chunk in chunks:
        if attempted >= TYPING_CALLS_PER_INVOCATION or _time_left_ms() < TIME_BUDGET_MS:
            break
        prompt = (json.dumps([{"slug": c.slug, "title": c.title, "aliases": c.aliases[:8], "kind": c.kind, "samples": c.samples}
                              for c in chunk], ensure_ascii=False)
                  + "\n\nAll candidate slugs (the only legal merge_into values): " + ", ".join(sorted(all_slugs)))
        result = _generate_json(pages.CANDIDATE_SYSTEM, prompt, max_tokens=PLAN_MAX_TOKENS)
        usage.add(result)
        attempted += 1
        data = result["data"]
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), None)
        rows = data if isinstance(data, list) else None
        chunk_slugs = {c.slug for c in chunk}
        row_by_slug: dict[str, dict] = {}
        if result["problem"] or rows is None or len(rows) != len(chunk):
            failed_chunks += 1
        else:
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("slug"), str) and row["slug"] in chunk_slugs:
                    row_by_slug[row["slug"]] = row
        for c in chunk:
            row = row_by_slug.get(c.slug)
            if row is not None:
                target = row.get("merge_into")
                valid_target = target if (isinstance(target, str) and target in all_slugs and target != c.slug) else None
                if valid_target:
                    new_merges[c.slug] = valid_target
                entity_type = row.get("entity_type") if row.get("entity_type") in ENTITY_TYPES else "other"
                typed[c.slug] = {"entity_type": entity_type, "merge_into": valid_target}
                attempts.pop(c.slug, None)
            else:
                attempts[c.slug] = attempts.get(c.slug, 0) + 1
                if attempts[c.slug] >= TYPING_ATTEMPTS_LIMIT:
                    typed[c.slug] = {"entity_type": "other", "merge_into": None, "typing_failed": True}
                    attempts.pop(c.slug, None)
        _put_json(TYPED_KEY, {"identity_version": CONCEPT_IDENTITY_VERSION, "typed": typed,
                             "attempts": attempts, "failed_chunks": failed_chunks_total + failed_chunks})
    unattempted = [c for chunk in chunks[attempted:] for c in chunk]
    typed_new = len(typed) > typed_before
    if unattempted and typed_new:
        _record(PLAN_ITEM, "plan", {"synthesis_status": "partial", "candidates": len(candidates)}, add=usage.fields())
        return {"status": "partial", "typed": len(typed), "untyped": len(unattempted), "work_manifest": None,
                "count": 0, "scope": scope, **usage.json()}
    if unattempted and not typed_new:
        # Nothing typed this invocation, whether from a call budget/time cutoff before any attempt or
        # every attempted chunk failing: returning partial here would just repeat forever. Show every
        # remaining candidate as "other" for this run without persisting it, so a later run still gets
        # to retry it.
        for c in unattempted:
            typed.setdefault(c.slug, {"entity_type": "other"})
    new_merges, rejected_new = safe_model_merges(candidates, new_merges)
    if new_merges:
        candidates = merge_candidates(candidates, new_merges, in_scope=in_scope)
    candidates = apply_title_overrides(candidates, overrides.get("title") or {})
    index = MentionIndex(candidates)
    for stem, body in bodies.items():
        index.add(stem, body)
    members_by_slug = {c.slug: set(c.glossary_stems) for c in candidates}
    existing = {i["work_id"]: i for i in _scan_kind("concept")}
    concepts, work, skipped, mentions_skipped = [], [], 0, 0
    for c in candidates:
        glossary = set(c.glossary_stems)
        skip_mentions = len(index.postings.get(c.slug, ())) > MENTION_POSTING_CAP
        if skip_mentions:
            c.mention_stems = []
            mentions_skipped += 1
        else:
            c.mention_stems = [s for s in index.confirm(c.slug, bodies) if s not in glossary]
        members = {s: metas[s]["sha256"] for s in c.glossary_stems}
        prev = existing.get(f"concept#{c.slug}") or {}
        mode = stale_mode(dict(prev.get("members") or {}) if prev.get("synthesis_status") == "ready" else None, members)
        entry = {"slug": c.slug, "title": c.title, "aliases": c.aliases, "kind": c.kind,
                 "entity_type": "gene" if c.kind == "gene" else (
                     typed.get(c.slug, {}).get("entity_type") or remembered.get(c.slug, {}).get("entity_type", "other")),
                 "count_total": c.count_total, "count_in_scope": c.count_in_scope,
                 "members": [_brief(metas[s]) for s in c.glossary_stems],
                 "mentions": [_brief(metas[s]) for s in c.mention_stems],
                 "related": [list(p) for p in co_occurrence(members_by_slug, c.slug)], "mode": mode}
        if skip_mentions:
            entry["mentions_skipped"] = True
        if typed.get(c.slug, {}).get("typing_failed"):
            entry["typing_failed"] = True
        concepts.append(entry)
        if mode == "skip":
            skipped += 1
        else:
            work.append({"action": "page", "kind": "concept", "slug": c.slug, "mode": mode})
    manifest = {"scope": scope, "created": _today(), "threshold": THRESHOLD, "notes_read": len(items), "model": MODEL_ID,
                "identity_version": CONCEPT_IDENTITY_VERSION,
                "model_merges": {**prior_merges, **new_merges},
                "rejected_model_merges": rejected_prior + rejected_new, "concepts": concepts}
    manifest["sha256"] = hashlib.sha256(json.dumps(
        [{k: v for k, v in c.items() if k != "mode"} for c in concepts], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    capped = max(0, len(work) - MAX_PAGES)
    work = work[:MAX_PAGES]
    manifest["capped"] = capped
    _put_json(CANDIDATES_KEY, manifest)
    key = f"runs/synthesis/work/concepts-{_stamp()}.json"
    _put_json(key, work)
    _record(PLAN_ITEM, "plan", {"synthesis_status": "planned", "candidates": len(concepts)}, add=usage.fields())
    return {"status": "planned", "scope": scope, "notes_read": len(items), "unreadable_notes": len(missing),
            "members_without_hash": without_hash, "candidates": len(concepts), "count": len(work), "skipped": skipped,
            "capped": capped, "mentions_skipped": mentions_skipped, "typing_failed_chunks": failed_chunks_total + failed_chunks,
            "manifest": CANDIDATES_KEY, "work_manifest": {"bucket": BUCKET, "key": key},
            "plan_seconds": round((_now() - started).total_seconds(), 1), **usage.json()}


# llm-wiki puts the whole question into bm25s, takes about twenty candidates, and reads those. It
# does not read every paper in a topic, and the candidates it gets back cross categories and
# include the overviews and concepts written earlier, so each synthesis builds on the last. The
# exhaustive read this replaces sent every member note in full: 1.4 MB and six calls for a
# 70-note subtopic, against 314 KB and one call locally (measured 2026-09-20).
RETRIEVED_PAGES = 20
RETRIEVAL_EXCERPT_CHARS = 12_000
WIKI_INDEX_KEY = "index/wiki-index-v2.sqlite3"
WIKI_INDEX_LOCAL = "/tmp/wiki-index-v2.sqlite3"
QUERY_STOPWORDS = set(
    "a an the of and or to in for with by on from is are was were be been does do did what which how why "
    "when where who whom whose can could would should may might will shall than then that this these those "
    "it its as at into onto over under between among across about versus vs not no any all some more most "
    "less much many such same other another each per via".split())


def _index_connection():
    """Open the BM25 index, fetching it to /tmp only when the object in S3 changed."""
    head = aws.s3.head_object(Bucket=BUCKET, Key=WIKI_INDEX_KEY)
    etag = head.get("ETag", "")
    marker = WIKI_INDEX_LOCAL + ".etag"
    if not os.path.exists(WIKI_INDEX_LOCAL) or not os.path.exists(marker) or open(marker).read() != etag:
        aws.s3.download_file(BUCKET, WIKI_INDEX_KEY, WIKI_INDEX_LOCAL)
        with open(marker, "w") as fh:
            fh.write(etag)
    return sqlite3.connect(f"file:{WIKI_INDEX_LOCAL}?mode=ro", uri=True), etag


def _retrieve(query: str, *, limit: int = RETRIEVED_PAGES, exclude: set[str] | None = None):
    """Whole-question BM25 over every page, returning distinct documents rather than sections.

    A synthesis page wants twenty different documents, not twenty slices of the same one, so the
    best-scoring section decides each document's rank and the rest are dropped.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", query.lower())
             if len(w) > 2 and w not in QUERY_STOPWORDS]
    if not words:
        return [], ""
    match = " OR ".join('"' + w.replace('"', '""') + '"' for w in dict.fromkeys(words))
    try:
        con, etag = _index_connection()
    except Exception:
        # No index yet, or it could not be fetched. The catalogue alone still names every member,
        # so the page is written from that rather than failing for want of the depth layer.
        return [], ""
    try:
        rows = con.execute(
            "SELECT m.doc_type, m.doc_id, d.title, d.s3_key, bm25(sections, 5.0, 2.0, 1.0) AS score "
            "FROM sections s JOIN section_map m ON m.rowid = s.rowid "
            "JOIN docs d ON d.doc_type = m.doc_type AND d.doc_id = m.doc_id "
            "WHERE sections MATCH ? ORDER BY score LIMIT ?", (match, limit * 8)).fetchall()
    except sqlite3.OperationalError:
        return [], etag
    finally:
        con.close()
    best: dict[str, dict] = {}
    for doc_type, doc_id, title, s3_key, score in rows:
        if doc_id in (exclude or ()) or doc_id in best or not s3_key:
            continue
        best[doc_id] = {"doc_type": doc_type, "doc_id": doc_id, "title": title or doc_id,
                        "s3_key": s3_key, "score": round(-score, 3)}
        if len(best) >= limit:
            break
    return list(best.values()), etag


def _retrieved_inputs(hits):
    """Read each retrieved page once, bounded, as the model sees it."""
    unreadable: list[str] = []
    texts = _read_many([h["s3_key"] for h in hits], unreadable)
    out = []
    for hit in hits:
        text = texts.get(hit["s3_key"])
        if not text:
            continue
        out.append(f"=== Retrieved {hit['doc_type']} {hit['doc_id']} | {hit['title']} ===\n"
                   + text[:RETRIEVAL_EXCERPT_CHARS] + "\n")
    return out


def _batches(inputs: list[str], header: str) -> list[list[str]]:
    """Group inputs into batches kept under MAX_INPUT_CHARS - 20_000 with the header. A single
    input over that budget by itself goes into its own batch, truncated to fit.

    Size is the only limit. A count cap made sense when every input was a full evidence note and
    fifteen of them filled the window; a page now sends one catalogue plus the documents it
    retrieved, which is 21 small inputs that fit one call. Capping the count split them in two and
    sent the page back through the merge path for nothing.
    """
    budget = MAX_INPUT_CHARS - 20_000
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = len(header)
    for item in inputs:
        if len(item) > budget:
            if current:
                batches.append(current)
                current, current_len = [], len(header)
            batches.append([item[:budget]])
            continue
        if current and current_len + len(item) > budget:
            batches.append(current)
            current, current_len = [], len(header)
        current.append(item)
        current_len += len(item)
    if current:
        batches.append(current)
    return batches


def _hierarchical(folder: str, ident: str, header: str, inputs: list[str], system: str, merge_system: str,
                  sections: tuple[str, ...], validate, *, work_id: str, kind: str) -> dict:
    """Write one page from many inputs, at most MAX_CALLS_PER_INVOCATION Bedrock calls per invocation.

    Partials live under runs/synthesis/{folder}/partials/{ident}/n{NOTES_PER_CALL}m{PARTIALS_PER_MERGE}/
    so a later invocation resumes where this one stopped and retuning either constant mid-run cannot
    reuse partials cut at the old size; ``ident`` carries the member digest, so changed membership
    starts fresh without deleting anything. Level 0 writes one partial per batch; every further level
    merges up to PARTIALS_PER_MERGE partials, until one page remains.

    Every completed Bedrock call is recorded onto the catalogue item immediately (so a hard Lambda
    timeout does not lose it), and its usage is then dropped from the accumulator this function
    returns as ``"usage"`` (only the calls a caller still needs to record itself); ``"total"`` carries
    every call made in this invocation, for the JSON the caller reports back.

    A failure below the top level returns ``text: ""``: a partial or a merge that fails validation, or
    the final merged text failing at full strength, has no single coherent page to assemble, so the
    caller must not try.
    """
    usage, total = Usage(), Usage()
    batches = _batches(inputs, header)
    if len(batches) == 1:
        result = _generate(system, header + "\n\n" + "\n".join(batches[0]))
        usage.add(result)
        total.add(result)
        # The page's shape is asked for in the prompt, not enforced here. Only a generation that did
        # not finish has nothing to publish.
        problems = [] if result["stop_reason"] in (None, "end_turn") else [f"stop reason {result['stop_reason']}"]
        return {"status": "failed" if problems else "ready", "text": result["text"], "problems": problems,
                "usage": usage, "total": total, "generation": "single",
                "fallback_calls": int(bool(FALLBACK_MODEL_ID) and result.get("model_id") == FALLBACK_MODEL_ID != MODEL_ID)}
    prefix = f"runs/synthesis/{folder}/partials/{ident}/n{NOTES_PER_CALL}m{PARTIALS_PER_MERGE}"
    if TRIAL is not None:
        prefix = TRIAL["prefix"] + prefix
    calls, level, parts = 0, 0, batches
    written: list[str] = []
    while True:
        keys = [f"{prefix}/L{level}-{i:03d}.md" for i in range(len(parts))]
        final_merge = len(parts) == 1
        for i, key in enumerate(keys):
            if _exists(key):
                continue
            # The time guard applies only after one call: an invocation that starts short of time
            # must still make progress, or the state machine would loop on partial for nothing.
            if calls >= MAX_CALLS_PER_INVOCATION or (calls and _time_left_ms() < TIME_BUDGET_MS):
                return {"status": "partial", "text": "", "problems": [], "usage": usage, "total": total,
                        "generation": "hierarchical"}
            if level == 0:
                prompt = header + "\n\n" + "\n".join(parts[i])
            else:
                prompt = header + "\n\n" + "\n\n".join(f"=== Partial {j + 1} ===\n{p}" for j, p in enumerate(parts[i]))
            result = _generate(system if level == 0 else merge_system, prompt)
            usage.add(result)
            total.add(result)
            calls += 1
            # The merge that produces the final page is checked at full strength, including the
            # caller's own link/content check, before it is written: a merge that only looks fine at
            # the loose in-progress strength must not be persisted, so a retry regenerates only it.
            if final_merge:
                problems = pages.validate_structure(result["text"], sections) + validate(result["text"])
            else:
                problems = pages.validate_structure(result["text"], sections, min_chars=200)
            if result["stop_reason"] not in (None, "end_turn"):
                problems.append(f"stop reason {result['stop_reason']}")
            if problems:
                return {"status": "failed", "text": "", "problems": [f"partial {key}: {p}" for p in problems],
                        "usage": usage, "total": total, "generation": "hierarchical"}
            _put_text(key, result["text"], model_id=result.get("model_id"))
            _record(work_id, kind, {"synthesis_status": "partial"}, add=usage.fields())
            usage = Usage()
        texts = [_get_text(k) for k in keys]
        written.extend(keys)
        if len(texts) == 1:
            # Every call of this generation left a partial, so the partials say which model wrote
            # what even when the page took several invocations.
            return {"status": "ready", "text": texts[0], "problems": [], "usage": usage, "total": total,
                    "generation": "hierarchical", "fallback_calls": _fallback_partials(written)}
        parts, level = chunked(texts, PARTIALS_PER_MERGE), level + 1


def _publish(*, kind: str, work_id: str, ident: str, page_key: str, failed_key: str, page_text: str,
             model_text: str, result: dict, members: dict, note_count: int, created: str | None,
             extra: dict | None = None, expected_etag: str | None = None) -> dict:
    """Write the page and record the outcome. On success, write to ``page_key`` and record everything
    a refresh needs. On failure, write only to ``failed_key`` (and the model text under its own
    ``failed/`` prefix) and record just the failure and what was attempted, so a good page already at
    ``page_key`` from an earlier run is untouched and its catalogue fields are left as they were.
    ``extra`` is merged into the recorded fields either way, for flags a scan needs (e.g. truncated).
    Format is not enforced here (user, 2026-09-20). Nobody reads this wiki end to end - that premise
    is why it is built with agents at all - so a rule that a bullet must carry its link cannot be
    checked by a reader who was never going to read it, and an agent that does read the page sees
    the missing citation in the text without being told. Enforcing it withheld 22 of 24 pages and
    $31.73 of sound generation on that date, and recording the defects instead would only hand the
    next agent a to-do list. What the page should be is stated in the prompt; a page that turns out
    wrong is fixed through update_wiki_page, the way the local wiki has always been fixed.
    ``failed_key`` is kept for a page there is no point publishing at all: no text, or a generation
    that was truncated or stopped rather than finished."""
    problems = list(result["problems"])
    unusable = (not model_text.strip()) or any(p.startswith("stop reason") for p in problems)
    fallback_calls = int(result.get("fallback_calls") or 0)
    if fallback_calls and not unusable:
        page_text = _insert_frontmatter_field(page_text, "ingest_fallback_model_id", FALLBACK_MODEL_ID)
        page_text = _insert_frontmatter_field(page_text, "ingest_fallback_calls", fallback_calls)
        extra = {**(extra or {}), "fallback_model": FALLBACK_MODEL_ID, "fallback_calls_last_generation": fallback_calls}
    sha = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
    usage = result.get("usage")
    display = result.get("total") or usage
    if TRIAL is not None:
        # The page and the model's own text go under the trial's folder, keyed as they would be in
        # the wiki; nothing is published, linked or recorded.
        key = TRIAL["prefix"] + (failed_key if unusable else page_key)
        _put_text(key, page_text)
        _put_text(TRIAL["prefix"] + f"runs/synthesis/model-text/{kind}/{ident}.md", model_text)
        return {"kind": kind, "id": ident, "status": "failed" if unusable else "ready", "key": key, "sha256": sha,
                "problems": problems, "trial_run": TRIAL["run"], "model": MODEL_ID, "reasoning": REASONING,
                "retrieved": result.get("retrieved", []), **(display.json() if display else {})}
    if unusable:
        _put_text(failed_key, page_text)
        _put_text(f"runs/synthesis/model-text/{kind}/failed/{ident}.md", model_text)
        fields = {"synthesis_status": "failed", "failed_key": failed_key, "problems": problems,
                  "attempted_members": members, "model": MODEL_ID, **(extra or {})}
        _record(work_id, kind, fields, add=usage.fields() if usage else None)
        return {"kind": kind, "id": ident, "status": "failed", "key": failed_key, "sha256": sha, "problems": problems,
                "retrieved": result.get("retrieved", []), "index_etag": result.get("index_etag"),
                "model": MODEL_ID, **(display.json() if display else {})}
    text_key = f"runs/synthesis/model-text/{kind}/{ident}.md"
    publication = publish_page(aws.s3, BUCKET, page_key, page_text, expected_etag=expected_etag,
                               create_only=expected_etag is None)
    sha = publication["sha256"]
    _put_text(text_key, model_text)
    fields = {"synthesis_status": "ready", "page_key": page_key, "page_sha256": sha, "model_text_key": text_key,
              "member_digest": member_digest(members), "members": members, "note_count": note_count,
              "generation": result.get("generation", "single"), "problems": [],
              "created": created or _today(), "model": MODEL_ID,
              "connection_errors": publication["errors"], **(extra or {})}
    _record(work_id, kind, fields, add=usage.fields() if usage else None)
    return {"kind": kind, "id": ident, "status": "ready", "key": page_key, "sha256": sha, "problems": [],
            "retrieved": result.get("retrieved", []), "index_etag": result.get("index_etag"),
            "etag": publication["etag"], "connections": publication["connections"],
            "catalogs": publication["catalogs"], "connection_errors": publication["errors"],
            "model": MODEL_ID, **(display.json() if display else {})}


def _update_payload(event, member_map):
    """Read the update an agent supplied and the evidence notes it cites.

    Returns (prompt_tail, added) or raises. The cited notes join the page's member map, because an
    update is how a paper that was not in the original partition gets into the page at all, and
    link validation only allows members.
    """
    established = str(event.get("established") or "").strip()
    corrections = str(event.get("corrections") or "").strip()
    if not established and not corrections:
        raise ValueError("an update needs established, corrections, or both")
    stems = [str(x) for x in (event.get("evidence_notes") or [])][:20]
    bad = [x for x in stems if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,200}", x)]
    if bad:
        raise ValueError(f"evidence_notes must be llm-wiki stems: {bad[0]}")
    added, blocks = {}, []
    for stem in stems:
        item = _item(f"{stem}")
        sha = item.get("source_note_sha256") or ""
        key = item.get("source_note_key") or f"wiki/sources/{stem}.md"
        try:
            text = _get_text(key)
        except Exception:
            continue
        added[stem] = sha
        blocks.append(f"=== Evidence note {stem} ===\n{text[:RETRIEVAL_EXCERPT_CHARS]}\n")
    missing = [x for x in stems if x not in added]
    tail = ""
    if established:
        tail += f"\n\n=== What has just been established ===\n{established[:20_000]}\n"
    if corrections:
        tail += f"\n\n=== Corrections to the page as it stands ===\n{corrections[:20_000]}\n"
    if blocks:
        tail += "\n\n" + "\n".join(blocks)
    return tail, added, missing


def _updated_page(event, *, item, sections, member_map, validate_links_against, snapshot=None, kind=None):
    """Revise the stored body of a page with what an agent has just established."""
    text_key = item.get("model_text_key")
    if not text_key or item.get("synthesis_status") != "ready":
        raise ValueError("this page has no reviewed body to update; generate it first")
    body = _published_model_body(snapshot, kind) if snapshot else _get_text(text_key)
    tail, added, missing = _update_payload(event, member_map)
    result = _generate(pages.UPDATE_SYSTEM, f"=== The page as it stands ===\n{body}{tail}")
    problems = pages.validate_structure(result["text"], sections)
    problems += pages.validate_links(result["text"], allowed_stems=validate_links_against | set(added))
    if result["stop_reason"] not in (None, "end_turn"):
        problems.append(f"stop reason {result['stop_reason']}")
    usage = Usage()
    usage.add(result)
    return {"status": "failed" if problems else "ready", "text": result["text"], "problems": problems,
            "usage": usage, "total": usage, "generation": "update", "added_notes": added,
            "unreadable_notes_requested": missing,
            "previous_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}


def concept_page(event: dict) -> dict:
    slug = str(event.get("slug") or "")
    if not SLUG_RE.fullmatch(slug):
        raise ValueError("slug must be a lowercase slug such as scn2a")
    mode = str(event.get("mode") or "generate")
    if mode not in ("generate", "refresh", "update"):
        raise ValueError("mode must be generate, refresh, or update")
    manifest = _get_json(CANDIDATES_KEY)
    if not manifest:
        raise ValueError("no concept manifest; run plan_concepts first")
    concept = next((c for c in manifest["concepts"] if c["slug"] == slug), None)
    if concept is None:
        raise ValueError(f"{slug} is not in the concept manifest")
    work_id, members = f"concept#{slug}", concept["members"]
    member_map = {m["stem"]: m["sha256"] for m in members}
    digest = member_digest(member_map)
    item = _item(work_id)
    snapshot = _publication_snapshot(f"wiki/concepts/{slug}.md")

    def build(text: str, generation: str) -> str:
        return pages.concept_page(text, concept=concept, members=members, mentions=concept.get("mentions") or [],
                                  related=[tuple(r) for r in concept.get("related") or []], model_id=MODEL_ID,
                                  reasoning=REASONING, generation=generation,
                                  manifest_ref=f"{CANDIDATES_KEY}@{manifest.get('sha256', '')}",
                                  created=item.get("created"), today=_today())

    def publish(text: str, result: dict) -> dict:
        return _publish(kind="concept", work_id=work_id, ident=slug, page_key=f"wiki/concepts/{slug}.md",
                        failed_key=f"wiki/concepts/failed/{slug}.md", page_text=build(text, result["generation"]),
                        model_text=text, result=result, members=member_map, note_count=len(members),
                        created=item.get("created"), expected_etag=snapshot["etag"] if snapshot else None)

    if mode == "update":
        result = _updated_page(event, item=item, sections=pages.CONCEPT_SECTIONS, member_map=member_map,
                               validate_links_against=set(member_map), snapshot=snapshot, kind="concept")
        member_map = {**member_map, **result["added_notes"]}
        return publish(result["text"], result)
    if mode == "refresh" and item.get("model_text_key") and item.get("synthesis_status") == "ready":
        text = _published_model_body(snapshot, "concept") if snapshot else _get_text(item["model_text_key"])
        problems = pages.validate_structure(text, pages.CONCEPT_SECTIONS) + pages.validate_links(text, allowed_stems=set(member_map))
        return publish(text, {"problems": problems, "generation": item.get("generation", "single"), "usage": None})
    keys = {m["stem"]: m.get("source_note_key") or f"wiki/sources/{m['stem']}.md" for m in members}
    unreadable: list[str] = []
    texts = _read_many(list(keys.values()), unreadable)
    members = [m for m in members if keys[m["stem"]] in texts]
    member_map = {m["stem"]: m["sha256"] for m in members}
    digest = member_digest(member_map)
    # Same shape as a subtopic page: every member as a catalogue row for coverage, and the pages
    # the concept's own words retrieve read in full for depth. 57 of the 184 concept candidates
    # hold more than NOTES_PER_CALL members and the largest holds 70, the size that hit the 900s
    # limit under the exhaustive read.
    catalog = [catalog_line(m, texts[keys[m["stem"]]]) for m in members]
    query = f"{concept['title']}. {' '.join(concept.get('aliases') or [])}"
    hits, index_etag = _retrieve(query, exclude={slug})
    retrieved = _retrieved_inputs(hits)
    inputs = [f"=== Every evidence note assigned to this concept ({len(catalog)}) ===\n"
              + "\n".join(catalog) + "\n"] + retrieved
    header = (f"Concept: {concept['title']}\nAliases: {', '.join(concept.get('aliases') or []) or 'none'}\n"
              f"Entity type: {concept.get('entity_type') or 'other'}\nEvidence notes: {len(members)}\n"
              f"Pages read in full: {len(retrieved)}")
    result = _hierarchical("concepts", f"{slug}/{digest[:12]}", header, inputs, pages.CONCEPT_SYSTEM,
                           pages.CONCEPT_MERGE_SYSTEM, pages.CONCEPT_SECTIONS,
                           lambda text: pages.validate_links(text, allowed_stems=set(member_map)),
                           work_id=work_id, kind="concept")
    result["retrieved"] = [{"doc_id": h["doc_id"], "doc_type": h["doc_type"], "score": h["score"]} for h in hits]
    result["index_etag"] = index_etag
    if result["status"] == "partial":
        _record(work_id, "concept", {"synthesis_status": "partial", "members": member_map}, add=result["usage"].fields())
        return {"kind": "concept", "id": slug, "status": "partial", "model": MODEL_ID, "unreadable_notes": len(unreadable),
                **result["total"].json()}
    if result["status"] == "failed" and not result["text"]:
        _record(work_id, "concept", {"synthesis_status": "failed", "problems": result["problems"], "attempted_members": member_map},
               add=result["usage"].fields())
        return {"kind": "concept", "id": slug, "status": "failed", "problems": result["problems"], "model": MODEL_ID,
                "unreadable_notes": len(unreadable), **result["total"].json()}
    published = publish(result["text"], result)
    published["unreadable_notes"] = len(unreadable)
    return published


def _start_trial(event: dict) -> dict:
    """Make this invocation a trial: another model or effort, writing under runs/model-trials/ only."""
    global MODEL_ID, REASONING, TRIAL
    run = str(event.get("trial_run") or "")
    if not TRIAL_RUN_RE.fullmatch(run):
        raise ValueError("trial_run must be a lowercase slug such as synth-opus55-20260923")
    if str(event.get("mode") or "generate") != "generate":
        raise ValueError("a trial only generates; refresh, update and create act on the wiki")
    model = str(event.get("model_id") or CONFIGURED_MODEL_ID)
    if not re.fullmatch(r"[a-z0-9.:-]+", model) or "anthropic" not in model:
        raise ValueError("model_id must be an Anthropic Bedrock model or inference profile id")
    effort = str(event.get("effort") or CONFIGURED_REASONING)
    if effort not in (*EFFORT_LEVELS, "default"):   # "default" sends no thinking, as the Lambda does
        raise ValueError(f"effort must be one of {EFFORT_LEVELS} or default")
    MODEL_ID, REASONING = model, effort
    slug = re.sub(r"[^a-z0-9-]+", "-", model.rsplit("anthropic.", 1)[-1]).strip("-")
    # A trial measures one model unless it asks for the fallback, which is how the production path
    # (declined, then written by the fallback) is watched without writing to the wiki.
    fallback = bool(event.get("fallback"))
    TRIAL = {"run": run, "fallback": fallback,
             "prefix": f"runs/model-trials/{run}/{slug}-{effort}{'-fallback' if fallback else ''}/"}
    return TRIAL


def page(event: dict) -> dict:
    kind = str(event.get("kind") or "")
    if event.get("trial_run") is not None:
        _start_trial(event)
    if kind not in ("concept", "subtopic", "category"):
        raise ValueError("kind must be concept, subtopic, or category")
    concepts = _get_json(CANDIDATES_KEY)
    if concepts is not None and concepts.get("identity_version") != CONCEPT_IDENTITY_VERSION:
        raise ValueError("Replan concepts before generation: the stored concept identities use an obsolete schema")
    if kind == "concept":
        return concept_page(event)
    if kind == "subtopic":
        return subtopic_page(event)
    if kind == "category":
        return category_page(event)
    raise ValueError("kind must be concept, subtopic, or category")


def _category_notes(category: str) -> tuple[list[dict], dict, int]:
    """Every ready note of the category with a readable original and a note sha256, plus how many
    were dropped for lacking either (``unreadable_notes`` + ``members_without_hash``, combined)."""
    items = _ready_notes({category})
    keys = {i["work_id"]: i.get("source_note_key") or f"wiki/sources/{i['work_id']}.md" for i in items}
    unreadable: list[str] = []
    texts = _read_many(sorted(set(keys.values())), unreadable)
    metas = {}
    dropped = 0
    for i in items:
        stem, key = i["work_id"], keys[i["work_id"]]
        if key not in texts or not i.get("source_note_sha256"):
            dropped += 1
            continue
        metas[stem] = {**note_metadata(stem, texts[key]), "sha256": i.get("source_note_sha256", "")}
        metas[stem]["category"] = metas[stem]["category"] or category
    return [i for i in items if i["work_id"] in metas], metas, dropped


def plan_subtopics(event: dict) -> dict:
    from .synthesis_planner import plan
    import sys
    return plan(event, runtime=sys.modules[__name__])


def category_plan_status(event: dict) -> dict:
    from .synthesis_planner import status
    import sys
    return status(event, runtime=sys.modules[__name__])


def plan_diagnostics_read(event: dict) -> dict:
    from .synthesis_planner import diagnostics
    import sys
    return diagnostics(event, runtime=sys.modules[__name__])


def plan_subtopic_pages(event: dict) -> dict:
    """One work item per subtopic whose members changed since its page was written."""
    categories = _categories(event)
    items = _ready_notes(set(categories) if categories else None)
    members_without_hash = sum(1 for i in items if not i.get("source_note_sha256"))
    sha = {i["work_id"]: i["source_note_sha256"] for i in items if i.get("source_note_sha256")}
    cats = sorted(categories or {i.get("category") for i in items if i.get("category")})
    previous = {i["work_id"]: i for i in _scan_kind("subtopic")}
    work, skipped, invalid = [], 0, []
    for category in cats:
        manifest = _get_json(f"runs/synthesis/{category}/subtopics.json")
        if not manifest or manifest.get("problems"):
            invalid.append(category)
            continue
        for st in manifest["subtopics"]:
            members = {s: sha[s] for s in st["stems"] if s in sha}
            if not members:
                continue
            prev = previous.get(f"subtopic#{category}/{st['slug']}") or {}
            mode = stale_mode(dict(prev.get("members") or {}) if prev.get("synthesis_status") == "ready" else None, members)
            if mode == "skip":
                skipped += 1
                continue
            work.append({"action": "page", "kind": "subtopic", "category": category, "slug": st["slug"], "mode": mode})
    capped = max(0, len(work) - MAX_PAGES)
    work = work[:MAX_PAGES]
    key = f"runs/synthesis/work/subtopics-{_stamp()}.json"
    _put_json(key, work)
    return {"status": "planned", "manifest": {"bucket": BUCKET, "key": key}, "count": len(work), "skipped": skipped,
            "capped": capped, "members_without_hash": members_without_hash, "invalid_categories": invalid, "categories": cats}


def subtopic_page(event: dict) -> dict:
    category, slug = str(event.get("category") or ""), str(event.get("slug") or "")
    if not CATEGORY_RE.fullmatch(category) or not SLUG_RE.fullmatch(slug):
        raise ValueError("category and slug must be lowercase slugs")
    mode = str(event.get("mode") or "generate")
    if mode not in ("generate", "refresh", "update"):
        raise ValueError("mode must be generate, refresh, or update")
    manifest = _get_json(f"runs/synthesis/{category}/subtopics.json")
    if not manifest:
        raise ValueError(f"{category} has no subtopic manifest; run plan_subtopics first")
    subtopic = next((st for st in manifest["subtopics"] if st["slug"] == slug), None)
    if subtopic is None:
        raise ValueError(f"{category}/{slug} is not in the subtopic manifest")
    items = {i["work_id"]: i for i in _ready_notes({category})}
    stems = [s for s in subtopic["stems"] if s in items]
    without_hash = sum(1 for s in stems if not items[s].get("source_note_sha256"))
    stems = [s for s in stems if items[s].get("source_note_sha256")]
    keys = {s: items[s].get("source_note_key") or f"wiki/sources/{s}.md" for s in stems}
    unreadable: list[str] = []
    texts = _read_many(list(keys.values()), unreadable)
    stems = [s for s in stems if keys[s] in texts]
    members = [{**note_metadata(s, texts[keys[s]]), "sha256": items[s].get("source_note_sha256", "")} for s in stems]
    for m in members:
        m["category"] = m["category"] or category
    member_map = {m["stem"]: m["sha256"] for m in members}
    digest = member_digest(member_map)
    candidates = _get_json(CANDIDATES_KEY, {}) or {}
    counts = sorted(((c["slug"], len({m["stem"] for m in c["members"]} & set(stems))) for c in candidates.get("concepts", [])),
                    key=lambda p: (-p[1], p[0]))
    concepts = [p for p in counts if p[1] > 0][:10]
    work_id, ident = f"subtopic#{category}/{slug}", f"{category}/{slug}"
    item = _item(work_id)
    snapshot = _publication_snapshot(f"wiki/overviews/{category}/{slug}.md")
    manifest_ref = f"runs/synthesis/{category}/subtopics.json@{manifest.get('sha256', '')}"

    def build(text: str, generation: str) -> str:
        return pages.subtopic_page(text, category=category, subtopic=subtopic, members=members, concepts=concepts,
                                   model_id=MODEL_ID, reasoning=REASONING, generation=generation,
                                   manifest_ref=manifest_ref, created=item.get("created"), today=_today())

    def publish(text: str, result: dict) -> dict:
        return _publish(kind="subtopic", work_id=work_id, ident=ident, page_key=f"wiki/overviews/{category}/{slug}.md",
                        failed_key=f"wiki/overviews/{category}/failed/{slug}.md", page_text=build(text, result["generation"]),
                        model_text=text, result=result, members=member_map, note_count=len(members), created=item.get("created"),
                        expected_etag=snapshot["etag"] if snapshot else None)

    if mode == "update":
        result = _updated_page(event, item=item, sections=pages.SUBTOPIC_SECTIONS, member_map=member_map,
                               validate_links_against=set(member_map), snapshot=snapshot, kind="subtopic")
        member_map = {**member_map, **result["added_notes"]}
        return publish(result["text"], result)
    if mode == "refresh" and item.get("model_text_key") and item.get("synthesis_status") == "ready":
        text = _published_model_body(snapshot, "subtopic") if snapshot else _get_text(item["model_text_key"])
        problems = pages.validate_structure(text, pages.SUBTOPIC_SECTIONS) + pages.validate_links(text, allowed_stems=set(member_map))
        return publish(text, {"problems": problems, "generation": item.get("generation", "single"), "usage": None})
    # Every assigned member appears as a catalogue row, so nothing the plan assigned is invisible,
    # and the pages the subtopic's own words retrieve are read in full. That is llm-wiki's shape:
    # the catalogue supplies coverage, retrieval supplies depth, and one call writes the page.
    catalog = [catalog_line(m, texts[keys[m["stem"]]]) for m in members]
    query = f"{subtopic['title']}. {subtopic.get('scope', '')}"
    hits, index_etag = _retrieve(query, exclude={f"{category}/{slug}"})
    retrieved = _retrieved_inputs(hits)
    inputs = [f"=== Every evidence note assigned to this subtopic ({len(catalog)}) ===\n"
              + "\n".join(catalog) + "\n"] + retrieved
    header = (f"Category: {category}\nSubtopic: {subtopic['title']}\nScope: {subtopic.get('scope', '')}\n"
              f"Evidence notes: {len(members)}\nPages read in full: {len(retrieved)}")
    result = _hierarchical("subtopics", f"{ident}/{digest[:12]}", header, inputs, pages.SUBTOPIC_SYSTEM,
                           pages.SUBTOPIC_MERGE_SYSTEM, pages.SUBTOPIC_SECTIONS,
                           lambda text: pages.validate_links(text, allowed_stems=set(member_map)),
                           work_id=work_id, kind="subtopic")
    result["retrieved"] = [{"doc_id": h["doc_id"], "doc_type": h["doc_type"], "score": h["score"]} for h in hits]
    result["index_etag"] = index_etag
    if result["status"] == "partial":
        _record(work_id, "subtopic", {"synthesis_status": "partial", "members": member_map}, add=result["usage"].fields())
        return {"kind": "subtopic", "id": ident, "status": "partial", "model": MODEL_ID,
                "unreadable_notes": len(unreadable), "members_without_hash": without_hash, **result["total"].json()}
    if result["status"] == "failed" and not result["text"]:
        _record(work_id, "subtopic", {"synthesis_status": "failed", "problems": result["problems"], "attempted_members": member_map},
               add=result["usage"].fields())
        return {"kind": "subtopic", "id": ident, "status": "failed", "problems": result["problems"], "model": MODEL_ID,
                "unreadable_notes": len(unreadable), "members_without_hash": without_hash, **result["total"].json()}
    published = publish(result["text"], result)
    published["unreadable_notes"] = len(unreadable)
    published["members_without_hash"] = without_hash
    return published


def _model_sections(body: str, drop: tuple[str, ...] = ()) -> str:
    keep = [(h, c) for h, c in pages.sections(body) if h in ("Scope", "Findings", "Comparison", "Open questions") and h not in drop]
    return "\n\n".join(f"## {h}\n{c}" for h, c in keep)


def _insert_frontmatter_field(text: str, key: str, value) -> str:
    """Insert one more ``key: json`` line into an already-assembled page's frontmatter block, right
    before the closing fence."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end < 0:
        return text
    return text[:end] + "\n" + f"{key}: {json.dumps(value, ensure_ascii=False)}" + text[end:]


def category_page(event: dict) -> dict:
    """The category's landscape, written from whichever of its subtopic pages exist.

    A subtopic is left off the page - and named under ``omitted_subtopics`` in the frontmatter -
    when none of its notes are ready any more (retracted, recategorised), or when its own page has
    not been written yet; one missing or omitted subtopic no longer blocks the whole category page.
    It fails only when no subtopic page exists at all.
    """
    category = str(event.get("category") or "")
    if not CATEGORY_RE.fullmatch(category):
        raise ValueError("category must be a lowercase slug such as asd-ndd")
    manifest = _get_json(f"runs/synthesis/{category}/subtopics.json")
    if not manifest or manifest.get("problems"):
        raise ValueError(f"{category} has no valid subtopic manifest")
    ready = {i["work_id"] for i in _ready_notes({category})}
    named = [st for st in manifest["subtopics"] if st.get("stems")]
    live = [st for st in named if set(st["stems"]) & ready]
    page_keys_all = {st["slug"]: f"wiki/overviews/{category}/{st['slug']}.md" for st in live}
    subtopics = [st for st in live if _exists(page_keys_all[st["slug"]])]
    page_keys = {st["slug"]: page_keys_all[st["slug"]] for st in subtopics}
    omitted = [st["slug"] for st in named if st not in subtopics]
    work_id = f"category#{category}"
    item = _item(work_id)
    if not subtopics:
        problems = ["no subtopic pages exist yet"]
        _record(work_id, "category", {"synthesis_status": "failed", "problems": problems})
        return {"kind": "category", "id": category, "status": "failed", "problems": problems, "model": MODEL_ID}
    texts = _read_many(list(page_keys.values()))
    sub_pages, members = [], {}
    for st in subtopics:
        text = texts[page_keys[st["slug"]]]
        fields, body = parse_frontmatter(text)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Reciprocal navigation can change after this category is published; it
        # does not change the science that would justify another model call.
        members[st["slug"]] = hashlib.sha256(BACKLINK_BLOCK.sub("", text).rstrip().encode("utf-8")).hexdigest()
        scope = next((c for h, c in pages.sections(body) if h == "Scope"), st.get("scope", ""))
        sub_pages.append({"slug": st["slug"], "title": st["title"], "scope": scope, "sha256": sha, "body": body,
                          "note_count": int(fields.get("note_count") or len(st["stems"]))})
    if item.get("synthesis_status") == "ready" and dict(item.get("members") or {}) == members and not event.get("force"):
        return {"kind": "category", "id": category, "status": "skipped", "model": MODEL_ID}
    snapshot = _publication_snapshot(f"wiki/overviews/{category}/index.md")

    def prompt_for(drop: tuple[str, ...]) -> str:
        return f"Category: {category}\nSubtopic pages ({len(sub_pages)}):\n\n" + "\n\n".join(
            f"=== [[overviews/{category}/{p['slug']}]] | {p['title']} ===\n{_model_sections(p['body'], drop)}" for p in sub_pages)

    prompt = prompt_for(())
    truncated = len(prompt) > MAX_INPUT_CHARS
    if truncated:
        prompt = prompt_for(("Comparison",))[:MAX_INPUT_CHARS]
    generation = "truncated" if truncated else "single"
    usage = Usage()
    result = _generate(pages.CATEGORY_SYSTEM, prompt)
    usage.add(result)
    all_stems = {s for st in subtopics for s in st["stems"]}
    problems = pages.validate_structure(result["text"], pages.CATEGORY_SECTIONS) + pages.validate_links(
        result["text"], allowed_stems=all_stems, allowed_pages={f"{category}/{slug}" for slug in page_keys}, linked_sections=set())
    if result["stop_reason"] not in (None, "end_turn"):
        problems.append(f"stop reason {result['stop_reason']}")
    candidates = _get_json(CANDIDATES_KEY, {}) or {}
    counts = sorted(((c["slug"], len({m["stem"] for m in c["members"]} & all_stems)) for c in candidates.get("concepts", [])),
                    key=lambda p: (-p[1], p[0]))
    year_range = manifest.get("year_range") or ["", ""]
    note_count = sum(p["note_count"] for p in sub_pages)
    coverage = {"note_count": note_count, "year_min": year_range[0], "year_max": year_range[1],
                "generated_at": _now().replace(microsecond=0).isoformat()}
    page_text = pages.category_page(result["text"], category=category, subtopics=sub_pages,
                                    key_concepts=[p for p in counts if p[1] > 0][:20], coverage=coverage, model_id=MODEL_ID,
                                    reasoning=REASONING, generation=generation, manifest_ref=f"runs/synthesis/{category}/subtopics.json",
                                    created=item.get("created"), today=_today())
    if omitted:
        page_text = _insert_frontmatter_field(page_text, "omitted_subtopics", omitted)
    return _publish(kind="category", work_id=work_id, ident=category, page_key=f"wiki/overviews/{category}/index.md",
                    failed_key=f"wiki/overviews/{category}/failed/index.md", page_text=page_text, model_text=result["text"],
                    result={"problems": problems, "generation": generation, "usage": usage}, members=members,
                    note_count=coverage["note_count"], created=item.get("created"),
                    expected_etag=snapshot["etag"] if snapshot else None,
                    extra={"truncated": True} if truncated else None)


def plan_failures(event: dict) -> dict:
    """Pages a run tried and did not finish, as a work list for the retry map. Category pages are not
    listed here: the state machine rewrites every category page in scope after the retry map runs and
    then rebuilds the index, so a stale category page is repaired by that pass instead."""
    categories = _categories(event)
    allowed = None if categories is None else set(categories)
    work = []
    for kind in ("concept", "subtopic"):
        for item in sorted(_scan_kind(kind), key=lambda i: i["work_id"]):
            if item.get("synthesis_status") not in ("failed", "partial"):
                continue
            ident = item["work_id"].split("#", 1)[1]
            if kind == "concept":
                work.append({"action": "page", "kind": "concept", "slug": ident, "mode": "generate"})
            else:
                category, slug = ident.split("/", 1)
                if allowed is None or category in allowed:
                    work.append({"action": "page", "kind": "subtopic", "category": category, "slug": slug, "mode": "generate"})
    capped = max(0, len(work) - MAX_PAGES)
    work = work[:MAX_PAGES]
    key = f"runs/synthesis/work/failures-{_stamp()}.json"
    _put_json(key, work)
    return {"status": "planned", "manifest": {"bucket": BUCKET, "key": key}, "count": len(work), "capped": capped}


def reference(event):
    from .synthesis_support import reference as intake
    return intake(event, s3=aws.s3, bucket=BUCKET)


def failures(event):
    from .synthesis_support import failures as report
    return report(event, table=aws.table)


ACTIONS = {"resolve_scope": resolve_scope, "plan_concepts": plan_concepts, "plan_subtopics": plan_subtopics,
           "plan_subtopic_pages": plan_subtopic_pages, "page": page, "plan_failures": plan_failures,
           "manifest_summary": manifest_summary, "manifest_read": manifest_read, "manifest_submit": manifest_submit,
           "reference": reference, "failures": failures, "category_plan_status": category_plan_status,
           "plan_diagnostics_read": plan_diagnostics_read}


def handler(event, context):
    global DEADLINE_MS, MODEL_ID, REASONING, TRIAL
    DEADLINE_MS = None
    # A warm container keeps module state; a trial must not leak into the next invocation.
    MODEL_ID, REASONING, TRIAL = CONFIGURED_MODEL_ID, CONFIGURED_REASONING, None
    if context is not None:
        DEADLINE_MS = _now().timestamp() * 1000 + context.get_remaining_time_in_millis()
    action = str(event.get("action") or "")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
    return ACTIONS[action](event)
