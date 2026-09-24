from __future__ import annotations

import json

import pytest

from byeori import synthesis_lambda as lam
from byeori import synthesis_planner as planner
from byeori.synthesis_pages import parse_plan_json
from test_synthesis_lambda import (NOTE_TEMPLATE, PARTITION, STEMS, partition_answer,
                                  small_partitions, world)


def _resume(result, **extra):
    return lam.plan_subtopics({"category": "asd-ndd", "resume": result["checkpoint"], **extra})


def _ledger(world):
    return [x for x in world.table.values() if x["id_kind"] == "category_plan"]


def _add(world, stem):
    world.s3.put_object("b", f"wiki/sources/{stem}.md", NOTE_TEMPLATE.format(
        title="New result", author="Author", year=2026, category="asd-ndd", summary="Summary.", gene="SCN2A").encode())
    world.notes.append({"work_id": stem, "source_note_key": f"wiki/sources/{stem}.md",
                        "source_note_sha256": f"hash-{stem}", "category": "asd-ndd"})


@pytest.mark.parametrize("text", [
    '{"x":1,"x":2}', '{"x":1} {"y":2}', 'Before {"x":1}', '```json\n{"x":1}\n```', '{"x":',
])
def test_planning_json_rejects_ambiguous_or_incomplete_values(text):
    assert parse_plan_json(text)[1]


def test_one_call_per_invocation_even_when_time_remains(world, small_partitions, monkeypatch):
    calls = []
    original = lam._generate
    def generate(system, prompt, **kwargs):
        calls.append(kwargs)
        return original(system, prompt, **kwargs)
    monkeypatch.setattr(lam, "_generate", generate)
    monkeypatch.setattr(lam, "_time_left_ms", lambda: 900_000)
    world.responses += ["not JSON", partition_answer(PARTITION)]
    first = lam.plan_subtopics({"category": "asd-ndd", "replan": True})
    assert first["status"] == "partial" and first["calls"] == 1 and len(world.responses) == 1
    assert calls[0]["timeout_seconds"] == 600 and calls[0]["max_tokens"] == 6000
    second = _resume(first, replan=True)
    assert second["status"] == "planned" and second["checkpoint"] == first["checkpoint"]
    assert second["calls"] == 1 and second["plan_calls"] == 2
    assert _resume(second)["calls"] == 0
    assert _ledger(world)[0]["calls"] == 2


def test_low_remaining_time_defers_without_spinning_forever(world, small_partitions, monkeypatch):
    monkeypatch.setattr(lam, "_time_left_ms", lambda: 89_000)
    first = lam.plan_subtopics({"category": "asd-ndd"})
    assert first["status"] == "partial" and first["calls"] == 0
    second = _resume(first)
    assert second["status"] == "failed" and second["calls"] == 0
    assert _resume(second)["status"] == "failed" and _ledger(world)[0]["calls"] == 0


def test_receipt_replay_recovers_initial_transport_retry_without_model_or_double_usage(world, small_partitions, monkeypatch):
    world.responses.append(partition_answer(PARTITION))
    original = planner._consume
    def crash(*args):
        raise RuntimeError("crash after durable response before checkpoint advancement")
    monkeypatch.setattr(planner, "_consume", crash)
    event = {"category": "asd-ndd", "run_id": "same-execution", "replan": True}
    with pytest.raises(RuntimeError, match="durable response"):
        lam.plan_subtopics(event)
    monkeypatch.setattr(planner, "_consume", original)
    recovered = lam.plan_subtopics(event)
    assert recovered["status"] == "planned" and recovered["calls"] == 0 and recovered["plan_calls"] == 1
    again = lam.plan_subtopics(event)
    assert again["checkpoint"] == recovered["checkpoint"] and again["calls"] == 0
    assert len(_ledger(world)) == 1 and _ledger(world)[0]["input_tokens"] == 10


def test_interrupted_call_is_unknown_usage_and_consumes_persisted_retry_budget(world, small_partitions, monkeypatch):
    original = lam._generate
    def die(*args, **kwargs):
        raise SystemExit("hard process stop")
    monkeypatch.setattr(lam, "_generate", die)
    event = {"category": "asd-ndd", "run_id": "interrupted", "replan": True}
    with pytest.raises(SystemExit):
        lam.plan_subtopics(event)
    monkeypatch.setattr(lam, "_generate", original)
    interrupted = lam.plan_subtopics(event)
    assert interrupted["status"] == "partial" and interrupted["calls"] == 0
    assert interrupted["unknown_usage_attempts"] == 1 and interrupted["plan_calls"] == 1
    world.responses.append("still not JSON")
    failed = _resume(interrupted)
    assert failed["status"] == "failed" and failed["plan_calls"] == 2
    assert _resume(failed)["calls"] == 0
    assert _ledger(world)[0]["unknown_usage_attempts"] == 1


def test_two_failures_preserve_live_plan_and_block_gate_even_with_old_manifest(world, small_partitions):
    world.responses.append(partition_answer(PARTITION))
    good = lam.plan_subtopics({"category": "asd-ndd"})
    live = world.s3.objects["runs/synthesis/asd-ndd/subtopics.json"]
    world.responses += ["broken", "broken again"]
    first = lam.plan_subtopics({"category": "asd-ndd", "replan": True})
    failed = _resume(first)
    assert failed["status"] == "failed" and good["checkpoint"] != failed["checkpoint"]
    assert world.s3.objects["runs/synthesis/asd-ndd/subtopics.json"] == live
    gate = lam.category_plan_status({"categories": ["asd-ndd"]})
    assert gate["status"] == "failed" and gate["ready"] is False


@pytest.mark.parametrize("change", ["note", "config"])
def test_resume_refuses_changed_source_or_configuration(world, small_partitions, monkeypatch, change):
    world.responses.append("bad")
    first = lam.plan_subtopics({"category": "asd-ndd"})
    if change == "note":
        world.notes[0]["source_note_sha256"] = "changed"
    else:
        monkeypatch.setattr(lam, "REASONING", "different")
    second = _resume(first)
    assert second["status"] == "failed" and second["calls"] == 0
    assert _ledger(world)[0]["calls"] == 1


@pytest.mark.parametrize("defect", ["duplicate", "missing", "unknown", "null-title", "non-string-id"])
def test_compact_partition_rejects_identity_and_schema_defects(world, small_partitions, defect):
    answer = json.loads(partition_answer(PARTITION))
    if defect == "duplicate":
        answer["subtopics"][0]["ids"][1] = "p0000"
    elif defect == "missing":
        answer["subtopics"][0]["ids"].pop()
    elif defect == "unknown":
        answer["subtopics"][0]["ids"][0] = "p9999"
    elif defect == "null-title":
        answer["subtopics"][0]["title"] = None
    else:
        answer["subtopics"][0]["ids"][0] = 1
    world.responses += [json.dumps(answer), json.dumps(answer)]
    failed = _resume(lam.plan_subtopics({"category": "asd-ndd"}))
    assert failed["status"] == "failed" and "runs/synthesis/asd-ndd/subtopics.json" not in world.s3.objects


@pytest.mark.parametrize("answer", ["bad JSON", '{"assignments":{}}',
    '{"assignments":{"p9999":"cohort-studies"}}', '{"assignments":{"p0000":"unknown"}}',
    '{"assignments":{"p0000":[]}}', '{"assignments":{"p0000":"cohort-studies","p0000":"asd-ndd-other"}}'])
def test_assignment_failure_never_becomes_other_or_overwrites_live(world, small_partitions, answer):
    world.s3.put_object("b", "runs/synthesis/asd-ndd/subtopics.json", json.dumps({**PARTITION, "note_count": 6}).encode())
    _add(world, "new-paper")
    live = world.s3.objects["runs/synthesis/asd-ndd/subtopics.json"]
    world.responses += [answer, answer]
    failed = _resume(lam.plan_subtopics({"category": "asd-ndd"}))
    assert failed["status"] == "failed" and world.s3.objects["runs/synthesis/asd-ndd/subtopics.json"] == live


def test_valid_explicit_other_assignment_is_allowed(world, small_partitions):
    world.s3.put_object("b", "runs/synthesis/asd-ndd/subtopics.json", json.dumps({**PARTITION, "note_count": 6}).encode())
    _add(world, "new-paper")
    world.responses.append('{"assignments":{"p0000":"asd-ndd-other"}}')
    result = lam.plan_subtopics({"category": "asd-ndd"})
    assert result["status"] == "updated"
    assert world.s3.json("runs/synthesis/asd-ndd/subtopics.json")["subtopics"][-1]["stems"] == ["new-paper"]


def test_merge_retries_parse_error_and_disambiguates_same_slug_across_chunks(world, monkeypatch):
    monkeypatch.setattr(lam, "PARTITION_SPLIT", 3)
    monkeypatch.setattr(lam, "PARTITION_LIMITS", {"min_subtopics": 1, "max_subtopics": 12, "min_notes": 1})
    parts = [STEMS[i::2] for i in range(2)]
    for i, part in enumerate(parts):
        world.responses.append(partition_answer({"subtopics": [{"slug": "same-slug", "title": f"Different {i}",
            "scope": f"Mechanism {i}.", "stems": part}]}, part))
    world.responses += ["bad merge", json.dumps({"merge": {"g0000": "first", "g0001": "second"},
        "subtopics": [{"slug": s, "title": s.title(), "scope": "Scope."} for s in ("first", "second")]})]
    result = lam.plan_subtopics({"category": "asd-ndd"})
    for _ in range(3):
        assert result["status"] == "partial" and result["calls"] == 1
        result = _resume(result)
    assert result["status"] == "planned" and result["plan_calls"] == 4
    assert [x["stems"] for x in world.s3.json("runs/synthesis/asd-ndd/subtopics.json")["subtopics"]] == parts


def test_801_long_stems_are_partitioned_with_short_ids_and_exactly_restored(world, monkeypatch):
    world.notes.clear()
    stems = [f"paper-{i:04d}-" + "a-long-source-filename-" * 8 for i in range(801)]
    for stem in stems:
        _add(world, stem)
    monkeypatch.setattr(lam, "PARTITION_LIMITS", {"min_subtopics": 1, "max_subtopics": 12, "min_notes": 1})
    parts = [stems[i::4] for i in range(4)]
    for part in parts:
        world.responses.append(partition_answer({"subtopics": [{"slug": "mechanism", "title": "Mechanism",
            "scope": "Shared mechanism.", "stems": part}]}, part))
    world.responses.append(json.dumps({"merge": {f"g{i:04d}": "mechanism" for i in range(4)},
        "subtopics": [{"slug": "mechanism", "title": "Mechanism", "scope": "Shared mechanism."}]}))
    prompts = []
    original = lam._generate
    def generate(system, prompt, **kwargs):
        prompts.append(prompt)
        return original(system, prompt, **kwargs)
    monkeypatch.setattr(lam, "_generate", generate)
    result = lam.plan_subtopics({"category": "asd-ndd"})
    for _ in range(4):
        assert result["status"] == "partial" and result["calls"] == 1
        result = _resume(result)
    assert result["status"] == "planned" and result["plan_calls"] == 5
    assigned = world.s3.json("runs/synthesis/asd-ndd/subtopics.json")["subtopics"][0]["stems"]
    assert sorted(assigned) == stems and len(set(assigned)) == 801
    assert all(len(p) <= 250 for p in parts)
    assert not any(stem in prompt for stem in stems for prompt in prompts)


def test_diagnostics_preserve_parse_and_stop_causes_and_return_only_bounded_excerpt(world, small_partitions, monkeypatch):
    text = '{"subtopics":' + " " * 9000
    def generate(*args, **kwargs):
        return {"text": text, "usage": {"inputTokens": 100, "outputTokens": 6000}, "stop_reason": "max_tokens",
                "seconds": 590, "request_id": "bedrock-request"}
    monkeypatch.setattr(lam, "_generate", generate)
    result = lam.plan_subtopics({"category": "asd-ndd"})
    event = {"category": "asd-ndd", "checkpoint": result["checkpoint"], "unit": "propose-0", "attempt": 1, "max_chars": 8000}
    diagnostic = lam.plan_diagnostics_read(event)
    assert len(diagnostic["text"]) == 8000 and diagnostic["has_more"]
    assert diagnostic["stop_reason"] == "max_tokens" and diagnostic["parse_problem"]
    assert diagnostic["request_id"] == "bedrock-request" and diagnostic["seconds"] == 590
    assert not any(k in diagnostic for k in ("decoded", "mapping", "metas", "partition"))
    with pytest.raises(ValueError, match="8000"):
        lam.plan_diagnostics_read({**event, "max_chars": 8001})
    with pytest.raises(ValueError):
        lam.plan_diagnostics_read({**event, "checkpoint": "../other"})


def test_truncated_but_parseable_assignment_is_rejected(world, small_partitions, monkeypatch):
    world.s3.put_object("b", "runs/synthesis/asd-ndd/subtopics.json", json.dumps({**PARTITION, "note_count": 6}).encode())
    _add(world, "new-paper")
    def generate(*args, **kwargs):
        return {"text": '{"assignments":{"p0000":"cohort-studies"}}', "usage": {"outputTokens": 1},
                "stop_reason": "max_tokens", "seconds": 1}
    monkeypatch.setattr(lam, "_generate", generate)
    first = lam.plan_subtopics({"category": "asd-ndd"})
    assert first["status"] == "partial" and "max_tokens" in first["problems"][0]
    assert _resume(first)["status"] == "failed"


def test_gate_checks_current_inputs_and_live_manifest(world, small_partitions):
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    assert lam.category_plan_status({"categories": ["asd-ndd"]})["ready"]
    key = "runs/synthesis/asd-ndd/subtopics.json"
    live = world.s3.objects.pop(key)
    gate = lam.category_plan_status({"categories": ["asd-ndd"]})
    assert not gate["ready"] and gate["categories"][0]["status"] == "invalid"
    world.s3.objects[key] = live
    world.notes[0]["source_note_sha256"] = "changed"
    assert lam.category_plan_status({"categories": ["asd-ndd"]})["categories"][0]["status"] == "stale"


def test_empty_category_has_explicit_checkpoint_and_ready_status(world):
    result = lam.plan_subtopics({"category": "absent"})
    assert result["status"] == "empty" and result["checkpoint"] and result["calls"] == 0
    assert lam.category_plan_status({"categories": ["absent"]})["ready"]


def test_pending_completed_usage_is_recorded_before_stale_input_rejection(world, small_partitions, monkeypatch):
    world.responses.append(partition_answer(PARTITION))
    original = planner._consume
    monkeypatch.setattr(planner, "_consume", lambda *args: (_ for _ in ()).throw(RuntimeError("crash")))
    event = {"category": "asd-ndd", "run_id": "usage-stale"}
    with pytest.raises(RuntimeError):
        lam.plan_subtopics(event)
    checkpoint = world.table["category#asd-ndd"]["plan_checkpoint"]
    world.notes[0]["source_note_sha256"] = "changed"
    monkeypatch.setattr(planner, "_consume", original)
    result = lam.plan_subtopics(event)  # Initial SFN transport retry has no resume token yet.
    assert result["status"] == "failed" and result["plan_calls"] == 1
    assert _ledger(world)[0]["input_tokens"] == 10
    assert lam.plan_subtopics({"category": "asd-ndd", "resume": checkpoint})["plan_calls"] == 1


def test_gate_does_not_adopt_success_from_an_earlier_run(world, small_partitions):
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd", "run_id": "old-run"})
    assert lam.category_plan_status({"categories": ["asd-ndd"], "run_id": "old-run"})["ready"]
    assert not lam.category_plan_status({"categories": ["asd-ndd"], "run_id": "new-run"})["ready"]


@pytest.mark.parametrize("preexisting", [True, False])
def test_conditional_publication_preserves_edit_after_final_read(world, small_partitions, monkeypatch, preexisting):
    key = "runs/synthesis/asd-ndd/subtopics.json"
    if preexisting:
        world.s3.put_object("b", key, json.dumps({**PARTITION, "note_count": 6}).encode())
    world.responses.append(partition_answer(PARTITION))
    original = world.s3.put_object
    edit = json.dumps({"user_edit": "must survive"}).encode()
    def concurrent(Bucket, Key, Body, **kwargs):
        if Key == key:
            world.s3.objects[key] = edit
        return original(Bucket, Key, Body, **kwargs)
    monkeypatch.setattr(world.s3, "put_object", concurrent)
    result = lam.plan_subtopics({"category": "asd-ndd", "replan": True})
    assert result["status"] == "failed" and world.s3.objects[key] == edit
    assert result["plan_calls"] == 1
