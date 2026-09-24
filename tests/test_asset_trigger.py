"""byeori.asset_trigger: which stored object means a new paper needs its figures cut."""
from __future__ import annotations

import pytest

from byeori import asset_trigger
from byeori.asset_trigger import stem_of


# ---------------------------------------------------------------------------------------------
# Which object announces a paper
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key, stem", [
    ("papers/ahanger-2026-lamina-associated-domains/clean.md", "ahanger-2026-lamina-associated-domains"),
    ("sources/W2741809807.md", "W2741809807"),
])
def test_both_ingest_routes_announce_their_paper(key, stem):
    """An uploaded original is extracted to clean.md; an OpenAlex one to sources/{work_id}.md."""
    assert stem_of(key) == stem


@pytest.mark.parametrize("key", [
    "wiki/sources/W2741809807.md",          # an evidence note is about a paper, not a paper
    "wiki/questions/how-do-lads-resolve.md",
    "papers/some-paper/original.pdf",       # the PDF alone is not yet a paper here
    "papers/some-paper/grobid.tei.xml",
    "papers/some-paper/meta.json",
    "papers/some-paper/assets/assets.md",   # what this worker itself writes
    "index/documents.json",
    "jobs/20260922T101500Z/0.json",
    "",
])
def test_nothing_else_starts_a_task(key):
    assert stem_of(key) is None


def test_a_nested_key_is_not_a_stem():
    """"sources/a/b.md" would name a paper "a/b", which no other key in the bucket agrees with."""
    assert stem_of("sources/nested/paper.md") is None


# ---------------------------------------------------------------------------------------------
# What the handler does with one
# ---------------------------------------------------------------------------------------------

def event(key: str) -> dict:
    return {"detail-type": "Object Created", "detail": {"object": {"key": key}}}


class Fake:
    """Stands in for S3 and ECS, recording what was asked of them."""

    def __init__(self, *, stored: set[str] = frozenset()):
        self.stored, self.started = set(stored), []

    def head_object(self, Bucket, Key):  # noqa: N803 - boto3's own spelling
        if Key not in self.stored:
            raise RuntimeError("NoSuchKey")
        return {}

    def run_task(self, **kwargs):
        env = kwargs["overrides"]["containerOverrides"][0]["environment"]
        self.started.append({name: value for name, value in
                             ((e["name"], e["value"]) for e in env)})
        self.last = kwargs
        return {"tasks": [{"taskArn": "arn:aws:ecs:ap-northeast-2:1:task/byeori/abc123"}]}


@pytest.fixture
def wired(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(asset_trigger, "s3", fake)
    monkeypatch.setattr(asset_trigger, "ecs", fake)
    monkeypatch.setattr(asset_trigger, "BUCKET", "bucket")
    monkeypatch.setattr(asset_trigger, "CLUSTER", "byeori-extract")
    monkeypatch.setattr(asset_trigger, "TASK_DEFINITION", "arn:aws:ecs:...:task-definition/assets:3")
    monkeypatch.setattr(asset_trigger, "SUBNETS", ["subnet-a", "subnet-b"])
    monkeypatch.setattr(asset_trigger, "SECURITY_GROUPS", ["sg-1"])
    return fake


def test_a_new_paper_starts_one_task_for_itself(wired):
    result = asset_trigger.handler(event("papers/kim-2026-a-paper/clean.md"))

    assert wired.started == [{"STEMS": "kim-2026-a-paper"}]
    assert result["started"] == "kim-2026-a-paper"
    assert wired.last["count"] == 1 and wired.last["launchType"] == "FARGATE"


def test_a_paper_that_already_has_its_crops_starts_nothing(wired):
    """Extraction runs again after a fix; the figures do not need cutting twice."""
    wired.stored.add("papers/kim-2026-a-paper/assets/assets.md")

    result = asset_trigger.handler(event("papers/kim-2026-a-paper/clean.md"))

    assert wired.started == [] and "skipped" in result


def test_an_unrelated_object_starts_nothing(wired):
    result = asset_trigger.handler(event("wiki/overviews/chromatin.md"))

    assert wired.started == [] and result["skipped"] == "not an extraction"


def test_a_half_configured_task_is_an_error_not_a_silent_skip(wired, monkeypatch):
    """A missing subnet would otherwise look exactly like a paper with no figures."""
    monkeypatch.setattr(asset_trigger, "SUBNETS", [])

    with pytest.raises(RuntimeError, match="not configured"):
        asset_trigger.handler(event("papers/kim-2026-a-paper/clean.md"))


def test_a_refused_task_is_raised_so_the_event_is_retried(wired, monkeypatch):
    monkeypatch.setattr(wired, "run_task",
                        lambda **kw: {"failures": [{"reason": "RESOURCE:MEMORY"}], "tasks": []})

    with pytest.raises(RuntimeError, match="RESOURCE:MEMORY"):
        asset_trigger.handler(event("papers/kim-2026-a-paper/clean.md"))
