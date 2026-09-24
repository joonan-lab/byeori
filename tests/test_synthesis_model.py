"""Synthesis runs on its own model, falls back when that model's output is filtered, and can be tried
on another model without touching the wiki or the catalogue."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from byeori import synthesis_lambda as lam

PRIMARY, FALLBACK = "global.anthropic.claude-opus-5-5", "global.anthropic.claude-opus-5"


class Bedrock:
    def __init__(self, filtered=(PRIMARY,)):
        self.requests, self.filtered = [], set(filtered)

    def converse(self, **request):
        self.requests.append(request)
        if request["modelId"] in self.filtered:
            return {"output": {"message": {"content": []}}, "stopReason": "content_filtered",
                    "usage": {"inputTokens": 30000, "outputTokens": 6}}
        return {"output": {"message": {"content": [{"text": "## Scope\nwritten"}]}}, "stopReason": "end_turn",
                "usage": {"inputTokens": 30000, "outputTokens": 9000}}


class S3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType=None, **conditions):
        self.objects[Key] = Body


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(lam, "MODEL_ID", PRIMARY)
    monkeypatch.setattr(lam, "FALLBACK_MODEL_ID", FALLBACK)
    monkeypatch.setattr(lam, "REASONING", "xhigh")
    monkeypatch.setattr(lam, "TRIAL", None)


def test_a_filtered_synthesis_call_is_written_again_by_the_fallback(models, monkeypatch):
    bedrock = Bedrock()
    monkeypatch.setattr(lam, "aws", SimpleNamespace(bedrock=bedrock, s3=None, table=None))
    result = lam._generate("system", "prompt")

    assert [r["modelId"] for r in bedrock.requests] == [PRIMARY, FALLBACK]
    assert all(r["additionalModelRequestFields"]["output_config"]["effort"] == "xhigh" for r in bedrock.requests)
    assert result["text"] == "## Scope\nwritten" and result["stop_reason"] == "end_turn"
    assert result["model_id"] == FALLBACK
    assert result["filtered_attempt"] == {"model_id": PRIMARY, "usage": {"inputTokens": 30000, "outputTokens": 6},
                                          "seconds": result["filtered_attempt"]["seconds"]}


def test_an_unfiltered_call_names_the_primary_and_a_trial_never_falls_back(models, monkeypatch):
    bedrock = Bedrock(filtered=())
    monkeypatch.setattr(lam, "aws", SimpleNamespace(bedrock=bedrock, s3=None, table=None))
    assert lam._generate("s", "p")["model_id"] == PRIMARY and "filtered_attempt" not in lam._generate("s", "p")

    bedrock = Bedrock()
    monkeypatch.setattr(lam, "aws", SimpleNamespace(bedrock=bedrock, s3=None, table=None))
    monkeypatch.setattr(lam, "TRIAL", {"run": "t", "prefix": "runs/model-trials/t/claude-opus-5-5-xhigh/"})
    result = lam._generate("s", "p")
    assert [r["modelId"] for r in bedrock.requests] == [PRIMARY] and result["stop_reason"] == "content_filtered"


def test_usage_keeps_the_fallback_and_the_declined_tokens_apart(models):
    usage = lam.Usage()
    usage.add({"usage": {"inputTokens": 100, "outputTokens": 10}, "seconds": 1.0, "model_id": PRIMARY})
    assert "fallback_calls" not in usage.fields(), "a run without a fallback reports what it always did"
    usage.add({"usage": {"inputTokens": 200, "outputTokens": 20}, "seconds": 2.0, "model_id": FALLBACK,
               "filtered_attempt": {"model_id": PRIMARY, "usage": {"inputTokens": 200, "outputTokens": 5}, "seconds": 1.0}})
    fields = usage.fields()
    assert fields["input_tokens"] == 300 and fields["output_tokens"] == 30 and fields["calls"] == 2
    assert fields["fallback_calls"] == 1 and fields["fallback_input_tokens"] == 200 and fields["fallback_output_tokens"] == 20
    assert fields["filtered_calls"] == 1 and fields["filtered_input_tokens"] == 200 and fields["filtered_output_tokens"] == 5


def test_a_trial_page_goes_under_runs_and_leaves_the_wiki_and_catalogue_alone(models, monkeypatch):
    s3 = S3()
    monkeypatch.setattr(lam, "aws", SimpleNamespace(bedrock=None, s3=s3, table=None))
    monkeypatch.setattr(lam, "TRIAL", {"run": "t", "prefix": "runs/model-trials/t/claude-opus-5-5-xhigh/"})
    monkeypatch.setattr(lam, "publish_page", lambda *a, **k: pytest.fail("a trial must not publish"))
    lam._record("subtopic#cat/slug", "subtopic", {"synthesis_status": "ready"})   # table is None: must not be touched
    usage = lam.Usage()
    usage.add({"usage": {"inputTokens": 1, "outputTokens": 1}, "seconds": 1.0, "model_id": PRIMARY})
    out = lam._publish(kind="subtopic", work_id="subtopic#cat/slug", ident="cat/slug",
                       page_key="wiki/overviews/cat/slug.md", failed_key="wiki/overviews/cat/failed/slug.md",
                       page_text="---\nx: 1\n---\n\n## Scope\nwritten", model_text="## Scope\nwritten",
                       result={"problems": [], "usage": usage, "total": usage}, members={}, note_count=0, created=None)
    assert out["status"] == "ready" and out["key"] == "runs/model-trials/t/claude-opus-5-5-xhigh/wiki/overviews/cat/slug.md"
    assert sorted(s3.objects) == ["runs/model-trials/t/claude-opus-5-5-xhigh/runs/synthesis/model-text/subtopic/cat/slug.md",
                                  "runs/model-trials/t/claude-opus-5-5-xhigh/wiki/overviews/cat/slug.md"]


def test_a_page_written_partly_by_the_fallback_says_so_in_its_frontmatter(models, monkeypatch):
    s3 = S3()
    monkeypatch.setattr(lam, "aws", SimpleNamespace(bedrock=None, s3=s3, table=SimpleNamespace(update_item=lambda **r: None)))
    monkeypatch.setattr(lam, "publish_page", lambda s3_, bucket, key, text, **k: (
        s3.objects.__setitem__(key, text), {"sha256": "x", "etag": "e", "errors": [], "connections": [], "catalogs": []})[1])
    usage = lam.Usage()
    usage.add({"usage": {"inputTokens": 1, "outputTokens": 1}, "seconds": 1.0, "model_id": PRIMARY})
    lam._publish(kind="subtopic", work_id="subtopic#cat/slug", ident="cat/slug", page_key="wiki/overviews/cat/slug.md",
                 failed_key="wiki/overviews/cat/failed/slug.md", page_text="---\nx: 1\n---\n\n## Scope\nw",
                 model_text="## Scope\nw", result={"problems": [], "usage": usage, "total": usage, "fallback_calls": 2},
                 members={}, note_count=0, created=None)
    page = s3.objects["wiki/overviews/cat/slug.md"]
    assert f'ingest_fallback_model_id: "{FALLBACK}"' in page and "ingest_fallback_calls: 2" in page


def test_partials_carry_the_model_that_wrote_them_so_the_page_can_count_them(models, monkeypatch):
    class HeadS3(S3):
        def put_object(self, Bucket, Key, Body, ContentType=None, Metadata=None, **conditions):
            self.objects[Key] = (Body, Metadata or {})

        def head_object(self, Bucket, Key):
            return {"Metadata": self.objects[Key][1]}

    s3 = HeadS3()
    monkeypatch.setattr(lam, "aws", SimpleNamespace(bedrock=None, s3=s3, table=None))
    lam._put_text("runs/synthesis/subtopics/partials/a/L0-000.md", "x", model_id=FALLBACK)
    lam._put_text("runs/synthesis/subtopics/partials/a/L0-001.md", "y", model_id=PRIMARY)
    lam._put_text("runs/synthesis/subtopics/partials/a/L0-002.md", "z")
    assert lam._fallback_partials(["runs/synthesis/subtopics/partials/a/L0-000.md",
                                   "runs/synthesis/subtopics/partials/a/L0-001.md",
                                   "runs/synthesis/subtopics/partials/a/L0-002.md"]) == 1


def test_each_invocation_starts_from_the_configured_model_and_no_trial(monkeypatch):
    monkeypatch.setattr(lam, "CONFIGURED_MODEL_ID", PRIMARY)
    monkeypatch.setattr(lam, "CONFIGURED_REASONING", "xhigh")
    monkeypatch.setattr(lam, "MODEL_ID", "left-over")
    monkeypatch.setattr(lam, "REASONING", "low")
    monkeypatch.setattr(lam, "TRIAL", {"run": "old"})
    seen = {}
    monkeypatch.setitem(lam.ACTIONS, "reference", lambda event: seen.update(model=lam.MODEL_ID, effort=lam.REASONING, trial=lam.TRIAL))
    lam.handler({"action": "reference"}, None)
    assert seen == {"model": PRIMARY, "effort": "xhigh", "trial": None}


def test_a_trial_names_a_valid_run_model_and_effort(monkeypatch):
    monkeypatch.setattr(lam, "TRIAL", None)
    trial = lam._start_trial({"trial_run": "synth-trial-1", "model_id": FALLBACK, "effort": "xhigh"})
    assert trial["prefix"] == "runs/model-trials/synth-trial-1/claude-opus-5-xhigh/" and lam.MODEL_ID == FALLBACK
    for bad in ({"trial_run": "../x"}, {"trial_run": "ok", "model_id": "gpt-5"}, {"trial_run": "ok", "effort": "huge"},
                {"trial_run": "ok", "mode": "update"}):
        with pytest.raises(ValueError):
            lam._start_trial(bad)


def test_the_synthesis_function_has_its_own_model_effort_and_fallback():
    from pathlib import Path
    root = Path(__file__).parents[1]
    template, deploy = (root / "infra/template.yaml").read_text(), (root / "scripts/deploy.sh").read_text()
    for line in ("SYNTHESIS_MODEL_ID: !Ref SynthesisModelId", "SYNTHESIS_FALLBACK_MODEL_ID: !Ref SynthesisFallbackModelId",
                 "SYNTHESIS_REASONING: !Ref SynthesisReasoning"):
        assert line in template
    # Synthesis runs on Opus 5 at high since the user's decision of 2026-09-23 (Opus 5.5's filtering); one value switches it.
    for override in ("SynthesisModelId=global.anthropic.claude-opus-5 ", "SynthesisFallbackModelId=global.anthropic.claude-opus-5 ",
                     "SynthesisReasoning=high "):
        assert override in deploy


def test_a_synthesis_run_is_priced_by_the_model_that_spent_each_token():
    from byeori.synthesis import split_synthesis_usd
    result = {"input_tokens": 300_000, "output_tokens": 30_000, "fallback_input_tokens": 100_000, "fallback_output_tokens": 10_000,
              "fallback_model_id": FALLBACK, "filtered_input_tokens": 100_000, "filtered_output_tokens": 1_000}
    # 5.5: 200k in, 20k out = 0.8 + 0.4; Opus 5: 100k in, 10k out = 0.5 + 0.25; declined 5.5: 0.4 + 0.02.
    assert split_synthesis_usd(PRIMARY, result) == 2.37
    assert split_synthesis_usd(PRIMARY, {"input_tokens": 100_000, "output_tokens": 10_000}) == 0.6


def test_a_trial_can_ask_for_the_fallback_so_the_production_path_is_seen_without_the_wiki(monkeypatch):
    monkeypatch.setattr(lam, "TRIAL", None)
    monkeypatch.setattr(lam, "CONFIGURED_MODEL_ID", PRIMARY)
    monkeypatch.setattr(lam, "FALLBACK_MODEL_ID", FALLBACK)
    trial = lam._start_trial({"trial_run": "synth-fallback-check", "effort": "high", "fallback": True})
    assert trial["fallback"] is True and trial["prefix"] == "runs/model-trials/synth-fallback-check/claude-opus-5-5-high-fallback/"
    bedrock = Bedrock()
    monkeypatch.setattr(lam, "aws", SimpleNamespace(bedrock=bedrock, s3=None, table=None))
    result = lam._generate("s", "p")
    assert [r["modelId"] for r in bedrock.requests] == [PRIMARY, FALLBACK] and result["model_id"] == FALLBACK
    assert lam._start_trial({"trial_run": "plain"})["fallback"] is False
