"""Checkpointed category planning. Documents and attempt receipts never leave AWS."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from decimal import Decimal

from . import synthesis_pages as pages
from .synthesis_manifest import validate_partition

VERSION = 1
MAX_ATTEMPTS = 2
OUTPUT_TOKENS = 6000
CALL_TIMEOUT_SECONDS = 600
SAVE_MARGIN_SECONDS = 60
MIN_CALL_SECONDS = CALL_TIMEOUT_SECONDS
TOKEN = re.compile(r"[a-f0-9]{64}")
COUNTERS = ("calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "unknown_usage_attempts")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _corpus(items):
    return _hash(sorted((x["work_id"], x.get("source_note_sha256") or "", x.get("source_note_key") or "") for x in items))


def _config(r):
    return _hash({"version": VERSION, "model": r.MODEL_ID, "reasoning": r.REASONING,
                  "chunk": r.PARTITION_SPLIT, "limits": r.PARTITION_LIMITS, "output": OUTPUT_TOKENS,
                  "prompts": [pages.PARTITION_SYSTEM, pages.PARTITION_MERGE_SYSTEM, pages.ASSIGN_SYSTEM]})


def _prefix(category, token):
    return f"runs/synthesis/{category}/plans/{token}"


def _load(r, category, token):
    if not isinstance(token, str) or not TOKEN.fullmatch(token):
        raise ValueError("resume/checkpoint must be a 64-character planning identifier")
    state = r._get_json(f"{_prefix(category, token)}/checkpoint.json")
    if not state or state.get("category") != category or state.get("checkpoint") != token:
        raise ValueError("checkpoint not found for this category")
    return state


def _live(r, key):
    try:
        result = r.aws.s3.get_object(Bucket=r.BUCKET, Key=key)
    except r.aws.s3.exceptions.NoSuchKey:
        return None, None
    return json.loads(result["Body"].read().decode()), result.get("ETag")


def _save(r, state):
    r._put_json(f"{_prefix(state['category'], state['checkpoint'])}/checkpoint.json", state)
    # SET cumulative counters on an immutable plan identity: replay cannot double ADD usage.
    fields = {**state["usage"], "seconds": Decimal(str(state["usage"].get("seconds", 0))),
              "category": state["category"], "plan_status": state["status"]}
    r._record(f"category-plan#{state['category']}#{state['checkpoint']}", "category_plan", fields)
    r._record(f"category#{state['category']}", "category", {
        "plan_status": state["status"], "plan_checkpoint": state["checkpoint"],
        "plan_run_id": state.get("run_id"),
        "plan_input_digest": state["input_digest"], "plan_config_digest": state["config_digest"],
        "plan_problems": state.get("problems", [])[:10]})


def _response(state, current=None):
    current = current or {}
    usage = current.get("usage") or {}
    return {"category": state["category"], "status": state["status"], "checkpoint": state["checkpoint"],
            "notes": len(state["stems"]), "dropped_notes": state["dropped_notes"],
            "subtopics": len((state.get("partition") or {}).get("subtopics", [])),
            "assigned": len(state.get("new", [])), "problems": state.get("problems", [])[:10],
            "calls": int(bool(current)), "input_tokens": int(usage.get("inputTokens", 0)),
            "output_tokens": int(usage.get("outputTokens", 0)), "seconds": current.get("seconds", 0),
            "plan_calls": state["usage"]["calls"], "unknown_usage_attempts": state["usage"]["unknown_usage_attempts"]}


def _fail(r, state, problems, status="failed"):
    state.update(status=status, problems=problems)
    r._put_json(f"runs/synthesis/{state['category']}/subtopics.failed.json", {
        "category": state["category"], "checkpoint": state["checkpoint"], "problems": problems,
        "note_count": len(state["stems"]), "subtopics": [], "model": r.MODEL_ID})
    _save(r, state)


def _chunks(stems, limit):
    n = max(1, (len(stems) + limit - 1) // limit)
    return [stems[i::n] for i in range(n)]


def _initial(r, event, category):
    # Capture the catalogue identity before reading bodies; later calls need only this small scan.
    input_digest = _corpus(r._ready_notes({category}))
    config_digest = _config(r)
    # Execution identity is stable even if sources change before a transport retry. The old
    # receipt must be accounted for before its immutable input fingerprint is invalidated.
    token = _hash([category, event["run_id"]]) if event.get("run_id") else _hash([category, uuid.uuid4().hex])
    previous = r._get_json(f"{_prefix(category, token)}/checkpoint.json")
    if previous:
        return previous
    items, metas, dropped = r._category_notes(category)
    stems = sorted(metas)
    key = f"runs/synthesis/{category}/subtopics.json"
    live, live_etag = _live(r, key)
    state = {"version": VERSION, "category": category, "checkpoint": token,
             "input_digest": input_digest, "config_digest": config_digest,
             "metas": metas, "stems": stems, "dropped_notes": dropped,
             "base": live, "live_hash": _hash(live), "live_etag": live_etag, "run_id": event.get("run_id"),
             "status": "partial", "phase": "propose",
             "chunks": _chunks(stems, r.PARTITION_SPLIT), "next": 0, "proposals": [],
             "attempts": {}, "pending": None, "problems": [], "deferrals": 0,
             "usage": {**dict.fromkeys(COUNTERS, 0), "seconds": 0.0}}
    if not stems:
        state.update(status="empty", phase="done", partition={"subtopics": []})
    elif live and not event.get("replan"):
        known = {s for st in live.get("subtopics", []) for s in st.get("stems", [])}
        new = [s for s in stems if s not in known]
        if len(new) / (live.get("note_count") or len(stems)) < .2:
            state.update(phase="assign", new=new, chunks=_chunks(new, r.PARTITION_SPLIT), assignments={})
            state["partition"] = {"subtopics": [{**st, "stems": [s for s in st["stems"] if s in metas]}
                                                 for st in live["subtopics"]]}
            if not new:
                state["phase"] = "final"
    _save(r, state)
    return state


def _valid_text_rows(rows, field):
    if not isinstance(rows, list):
        return [f"{field} must be a list"]
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                            for k in ("slug", "title", "scope")):
            return ["Each subtopic requires nonempty string slug, title and scope"]
    return []


def _groups(state):
    return {f"g{i:04d}": st for i, st in enumerate(st for p in state["proposals"] for st in p["subtopics"])}


def _request(r, state):
    phase = state["phase"]
    other = f"{state['category']}-other"
    if phase == "merge":
        groups = _groups(state)
        prompt = json.dumps({"other": other, "groups": [{"id": i, **{k: st[k] for k in ("title", "scope")},
                                                        "papers": len(st["stems"])} for i, st in groups.items()]}, ensure_ascii=False)
        system = pages.PARTITION_MERGE_SYSTEM.format(**r.PARTITION_LIMITS)
        mapping = groups
    else:
        subset = state["chunks"][state["next"]]
        mapping = {f"p{i:04d}": stem for i, stem in enumerate(subset)}
        lines = [f"{ident} | {state['metas'][s]['title']} | {state['metas'][s]['summary']}"[:400]
                 for ident, s in mapping.items()]
        prompt = f"Category: {state['category']}\nPapers ({len(subset)}):\n" + "\n".join(lines)
        if phase == "assign":
            system = pages.ASSIGN_SYSTEM.format(other=other)
            prompt = "Subtopics:\n" + json.dumps([{k: st[k] for k in ("slug", "title", "scope")}
                                                   for st in state["partition"]["subtopics"]]) + "\n" + prompt
        else:
            system = pages.PARTITION_SYSTEM.format(other=other, **r.PARTITION_LIMITS)
    if state.get("problems"):
        prompt += "\nThe previous attempt failed. Return a complete corrected JSON value.\n" + "\n".join(state["problems"][:12])
    return system, prompt, mapping


def _decode(r, state, result, mapping):
    if result.get("problem"):
        return None, [result["problem"]]
    data = result.get("data")
    if not isinstance(data, dict):
        return None, ["The response must be an object"]
    if state["phase"] == "assign":
        assignments = data.get("assignments")
        legal = {st["slug"] for st in state["partition"]["subtopics"]} | {f"{state['category']}-other"}
        if not isinstance(assignments, dict) or set(assignments) != set(mapping):
            return None, ["Assignments must contain every supplied paper ID exactly once, and no other IDs"]
        if any(not isinstance(slug, str) or slug not in legal for slug in assignments.values()):
            return None, ["Assignments contain an unknown or non-string target slug"]
        return {mapping[i]: slug for i, slug in assignments.items()}, []
    rows = data.get("subtopics")
    problems = _valid_text_rows(rows, "subtopics")
    if problems:
        return None, problems
    if state["phase"] == "merge":
        targets = {st["slug"]: {**st, "stems": []} for st in rows}
        merges = data.get("merge")
        if len(targets) != len(rows):
            return None, ["Duplicate target subtopic definitions"]
        if not isinstance(merges, dict) or set(merges) != set(mapping):
            return None, ["Map every supplied group ID exactly once, and no other group IDs"]
        if any(not isinstance(slug, str) or slug not in targets for slug in merges.values()):
            return None, ["Merge target is not a declared subtopic slug"]
        for ident, slug in merges.items():
            targets[slug]["stems"].extend(mapping[ident]["stems"])
        partition = {"subtopics": [st for st in targets.values() if st["stems"]]}
        expected = state["stems"]
    else:
        ids = [i for st in rows for i in (st.get("ids") if isinstance(st.get("ids"), list) else [])]
        if any(not isinstance(st.get("ids"), list) for st in rows) or any(not isinstance(i, str) for i in ids):
            return None, ["Every subtopic must contain a list of paper IDs"]
        if len(ids) != len(mapping) or len(set(ids)) != len(ids) or set(ids) != set(mapping):
            return None, ["Use every supplied paper ID exactly once; no duplicates, omissions or unknown IDs"]
        # Validate with short IDs so retry messages never reintroduce long filenames.
        short = {"subtopics": [{**st, "stems": st["ids"]} for st in rows]}
        problems = validate_partition(short, list(mapping), category=state["category"], **r.PARTITION_LIMITS)
        if problems:
            return None, problems
        partition = {"subtopics": [{"slug": st["slug"], "title": st["title"], "scope": st["scope"],
                                    "stems": [mapping[i] for i in st["ids"]]} for st in rows]}
        expected = list(mapping.values())
    return partition, validate_partition(partition, expected, category=state["category"], **r.PARTITION_LIMITS)


def _consume(r, state, receipt):
    unit = state["pending"]["unit"]
    result = receipt.get("result") or {}
    u = state["usage"]
    u["calls"] += 1
    if receipt.get("usage_known"):
        for field, api in [("input_tokens", "inputTokens"), ("output_tokens", "outputTokens"),
                           ("cache_read_tokens", "cacheReadInputTokens"), ("cache_write_tokens", "cacheWriteInputTokens")]:
            u[field] += int((result.get("usage") or {}).get(api, 0))
    else:
        u["unknown_usage_attempts"] += 1
    u["seconds"] = round(u["seconds"] + float(result.get("seconds") or 0), 1)
    state["pending"] = None
    data, problems = receipt.get("decoded"), receipt.get("problems") or []
    if problems:
        state["problems"] = problems
        # Bedrock's content filter charges for the input in full before it stops the call, and the
        # same prompt trips the same classifier every time. Retrying only spends the money again,
        # so a filter stop ends this unit at once (user, 2026-09-20).
        filtered = result.get("stop_reason") == "content_filtered"
        if filtered or state["attempts"][unit] >= MAX_ATTEMPTS:
            _fail(r, state, problems, status="filtered" if filtered else "failed")
        else:
            _save(r, state)
        return
    state["problems"] = []
    if state["phase"] == "merge":
        state.update(partition=data, phase="final")
    elif state["phase"] == "assign":
        state["assignments"].update(data)
        state["next"] += 1
        if state["next"] == len(state["chunks"]):
            by_slug = {st["slug"]: st for st in state["partition"]["subtopics"]}
            for stem, slug in state["assignments"].items():
                by_slug.setdefault(slug, {"slug": slug, "title": f"Other {state['category']} papers",
                                           "scope": "Papers that fit no subtopic.", "stems": []})["stems"].append(stem)
            state.update(partition={"subtopics": list(by_slug.values())}, phase="final")
    else:
        state["proposals"].append(data)
        state["next"] += 1
        if state["next"] == len(state["chunks"]):
            if len(state["proposals"]) == 1:
                state.update(partition=data, phase="final")
            else:
                state["phase"] = "merge"
    _save(r, state)


def _finish(r, state):
    problems = validate_partition(state["partition"], state["stems"], category=state["category"], **r.PARTITION_LIMITS)
    if _corpus(r._ready_notes({state["category"]})) != state["input_digest"]:
        problems.append("Ready-note membership or content hashes changed during planning; start a new plan")
    key = f"runs/synthesis/{state['category']}/subtopics.json"
    live, _ = _live(r, key)
    # A retry after publication but before checkpoint completion must recognize its own output.
    already_published = bool(live and live.get("checkpoint") == state["checkpoint"] and
                             _hash(live.get("subtopics")) == _hash(state["partition"]["subtopics"]))
    if not already_published and _hash(live) != state["live_hash"]:
        problems.append("The live manifest changed during planning; preserve that edit and start a new plan")
    if problems:
        _fail(r, state, problems, status="invalid" if state.get("new") == [] else "failed")
        return
    years = sorted(m["year"] for m in state["metas"].values() if m["year"].isdigit())
    manifest = {"category": state["category"], "created": (state.get("base") or {}).get("created", r._today()),
                "updated": r._today(), "note_count": len(state["stems"]), "year_range": [years[0], years[-1]] if years else ["", ""],
                "model": r.MODEL_ID, "reasoning": r.REASONING, "problems": [], **state["partition"],
                "input_digest": state["input_digest"], "config_digest": state["config_digest"],
                "checkpoint": state["checkpoint"], "run_id": state.get("run_id")}
    manifest["sha256"] = _hash(manifest["subtopics"])
    if not already_published:
        condition = {"IfMatch": state["live_etag"]} if state["live_etag"] else {"IfNoneMatch": "*"}
        try:
            r.aws.s3.put_object(Bucket=r.BUCKET, Key=key, Body=json.dumps(manifest, ensure_ascii=False, indent=1).encode(),
                                ContentType="application/json", **condition)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code not in ("PreconditionFailed", "ConditionalRequestConflict", "412", "409"):
                raise
            _fail(r, state, ["The live manifest changed during publication; the concurrent edit was preserved"])
            return
    state.update(status="updated" if "new" in state else "planned", phase="done", problems=[])
    _save(r, state)


def plan(event, *, runtime):
    r = runtime
    category = str(event.get("category") or "")
    if not r.CATEGORY_RE.fullmatch(category):
        raise ValueError("category must be a lowercase slug")
    state = _load(r, category, event["resume"]) if event.get("resume") else _initial(r, event, category)
    replayed = False
    if state.get("pending"):
        receipt = r._get_json(state["pending"]["key"])
        if not receipt or receipt.get("status") != "completed":
            receipt = {**(receipt or {}), "status": "completed", "usage_known": False,
                       "problems": ["Previous model attempt was interrupted; response and billed usage are unknown"]}
            r._put_json(state["pending"]["key"], receipt)
        _consume(r, state, receipt)
        replayed = True
    # Completed usage remains accountable even when a subsequent input/configuration check fails.
    if state["config_digest"] != _config(r) or state["input_digest"] != _corpus(r._ready_notes({category})):
        _fail(r, state, ["Planning configuration or ready-note input changed; start a new plan"])
        return _response(state)
    if state["status"] != "partial":
        _save(r, state)  # Repair a catalogue write interrupted after its durable checkpoint.
        return _response(state)
    if replayed and state["phase"] != "final":
        return _response(state)
    if state["phase"] == "final":
        _finish(r, state)
        return _response(state)
    remaining = r._time_left_ms() / 1000
    if remaining < SAVE_MARGIN_SECONDS + MIN_CALL_SECONDS:
        state["deferrals"] += 1
        if state["deferrals"] >= MAX_ATTEMPTS:
            _fail(r, state, ["Insufficient invocation time to start a model call after two deferrals"])
        else:
            _save(r, state)
        return _response(state)
    system, prompt, mapping = _request(r, state)
    if len(prompt) > r.MAX_INPUT_CHARS:
        _fail(r, state, [f"category listing is {len(prompt)} chars, over {r.MAX_INPUT_CHARS}"])
        return _response(state)
    unit = f"{state['phase']}-{state['next']}"
    attempt = state["attempts"].get(unit, 0) + 1
    state["attempts"][unit] = attempt
    key = f"{_prefix(category, state['checkpoint'])}/attempt-{unit}-{attempt}.json"
    state["pending"] = {"unit": unit, "attempt": attempt, "key": key}
    receipt = {"status": "started", "unit": unit, "attempt": attempt, "input_chars": len(prompt),
               "prompt_sha256": _hash([system, prompt]), "model": r.MODEL_ID, "reasoning": r.REASONING,
               "started": r._now().isoformat(), "usage_known": False}
    r._put_json(key, receipt)
    _save(r, state)
    started = r._now()
    try:
        result = r._generate_json(system, prompt, max_tokens=OUTPUT_TOKENS, strict=True,
                                  timeout_seconds=int(min(CALL_TIMEOUT_SECONDS, remaining - SAVE_MARGIN_SECONDS)))
    except Exception as exc:
        result = {"text": "", "data": None, "problem": f"{type(exc).__name__}: {str(exc)[:1000]}",
                  "parse_problem": None, "stop_reason": None, "usage": {},
                  "seconds": round((r._now() - started).total_seconds(), 1)}
    else:
        receipt["usage_known"] = bool(result.get("usage"))
    data, problems = _decode(r, state, result, mapping)
    receipt.update(status="completed", result=result, decoded=data, problems=problems,
                   text_sha256=hashlib.sha256(result["text"].encode()).hexdigest(), text_chars=len(result["text"]))
    r._put_json(key, receipt)
    _consume(r, state, receipt)
    if state["status"] == "partial" and state["phase"] == "final":
        _finish(r, state)
    return _response(state, result)


def status(event, *, runtime):
    r = runtime
    categories = r._categories(event)
    items = r._ready_notes()
    if categories is None:
        categories = sorted({i["category"] for i in items if i.get("category")})
    rows = []
    for category in categories:
        item = r._item(f"category#{category}")
        current = _corpus([x for x in items if x.get("category") == category])
        value = item.get("plan_status") or "missing"
        if event.get("run_id") and item.get("plan_run_id") != event["run_id"]:
            value = "stale"
        if value in ("planned", "updated", "empty") and (item.get("plan_input_digest") != current or item.get("plan_config_digest") != _config(r)):
            value = "stale"
        if value in ("planned", "updated"):
            manifest = r._get_json(f"runs/synthesis/{category}/subtopics.json")
            stems = [x["work_id"] for x in items if x.get("category") == category and x.get("source_note_sha256")]
            if (not manifest or manifest.get("checkpoint") != item.get("plan_checkpoint") or
                    validate_partition(manifest, stems, category=category, **r.PARTITION_LIMITS)):
                value = "invalid"
        rows.append({"category": category, "status": value, "checkpoint": item.get("plan_checkpoint"),
                     "problems": (item.get("plan_problems") or [])[:10]})
    counts = {s: sum(x["status"] == s for x in rows) for s in sorted({x["status"] for x in rows})}
    ready = all(x["status"] in ("planned", "updated", "empty") for x in rows)
    result = "planned" if ready else ("partial" if all(x["status"] in ("planned", "updated", "empty", "partial") for x in rows) else "failed")
    return {"status": result, "ready": ready, "categories": rows, "counts": counts}


def diagnostics(event, *, runtime):
    r = runtime
    category = str(event.get("category") or "")
    if not r.CATEGORY_RE.fullmatch(category):
        raise ValueError("category must be a lowercase slug")
    state = _load(r, category, event.get("checkpoint"))
    unit = str(event.get("unit") or "")
    attempt = event.get("attempt", 1)
    start, limit = event.get("start", 0), event.get("max_chars", 4000)
    if not re.fullmatch(r"(?:propose|assign|merge)-\d+", unit) or type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
        raise ValueError("unit and attempt must identify a planning call")
    if type(start) is not int or start < 0 or type(limit) is not int or not 1 <= limit <= 8000:
        raise ValueError("start must be nonnegative and max_chars must be 1 to 8000")
    key = f"{_prefix(category, state['checkpoint'])}/attempt-{unit}-{attempt}.json"
    receipt = r._get_json(key)
    if not receipt:
        raise ValueError("attempt receipt not found")
    result = receipt.get("result") or {}
    text = result.get("text") or ""
    return {"key": key, "status": receipt["status"], "model": receipt.get("model"), "reasoning": receipt.get("reasoning"),
            "stop_reason": result.get("stop_reason"), "parse_problem": result.get("parse_problem"),
            "request_id": result.get("request_id"), "usage": result.get("usage"), "usage_known": receipt.get("usage_known"),
            "seconds": result.get("seconds"), "input_chars": receipt.get("input_chars"),
            "sha256": receipt.get("text_sha256"), "text": text[start:start + limit], "total_chars": len(text),
            "has_more": start + limit < len(text), "problems": receipt.get("problems", [])[:10]}
