import re

import pytest

from byeori import synthesis_lambda as lam
from byeori.wiki_connections import BACKLINK_START, PageConflictError, publish_page
from test_synthesis_lambda import STEMS, _planned, concept_text, world


def test_synthesis_refresh_preserves_live_backlinks_and_publishes_real_note_links(world):
    _planned(world)
    world.responses.append(concept_text(STEMS))
    generated = lam.page({"kind": "concept", "slug": "scn2a"})
    key = generated["key"]
    publish_page(world.s3, "b", "wiki/questions/why-channel.md", "# Why channel?\n[[concepts/scn2a]]", create_only=True)
    before_note = world.s3.text(f"wiki/sources/{STEMS[0]}.md")
    assert "[[concepts/scn2a|SCN2A]]" in before_note
    refreshed = lam.page({"kind": "concept", "slug": "scn2a", "mode": "refresh"})
    text = world.s3.text(key)
    assert refreshed["status"] == "ready"
    assert text.count("[[questions/why-channel|Why channel?]]") == 1
    assert text.count(BACKLINK_START) == 1
    assert text.count("## Related concepts") == 1 and text.count("## Notes") == 1
    assert re.findall(r"^## .*", world.s3.text(f"wiki/sources/{STEMS[0]}.md"), re.M) == re.findall(r"^## .*", before_note, re.M)
    assert "[[concepts/scn2a|SCN2A]]" in world.s3.text("wiki/indexes/concepts.md")


def test_synthesis_update_reads_current_canonical_body_not_archived_model_copy(world, monkeypatch):
    _planned(world)
    world.responses.append(concept_text(STEMS))
    generated = lam.page({"kind": "concept", "slug": "scn2a"})
    key = generated["key"]
    current = world.s3.text(key).replace("## Definition\n", "## Definition\nCurrent canonical addition.\n", 1)
    world.s3.put_object(Bucket="b", Key=key, Body=current.encode())
    seen = []
    def generate(system, prompt, **kwargs):
        seen.append(prompt)
        return {"text": concept_text(STEMS), "usage": {}, "stop_reason": "end_turn", "seconds": 0}
    monkeypatch.setattr(lam, "_generate", generate)
    result = lam.page({"kind": "concept", "slug": "scn2a", "mode": "update", "established": "One new result."})
    assert result["status"] == "ready"
    assert len(seen) == 1 and "Current canonical addition." in seen[0]
    assert "## Notes" not in seen[0], "Code-owned note navigation is rebuilt separately"


def test_synthesis_model_output_does_not_overwrite_a_concurrent_published_revision(world, monkeypatch):
    _planned(world)
    world.responses.append(concept_text(STEMS))
    generated = lam.page({"kind": "concept", "slug": "scn2a"})
    key = generated["key"]
    calls = []
    def generate(system, prompt, **kwargs):
        calls.append(prompt)
        current = world.s3.text(key) + "\nConcurrent scientific addition.\n"
        world.s3.put_object(Bucket="b", Key=key, Body=current.encode())
        return {"text": concept_text(STEMS), "usage": {}, "stop_reason": "end_turn", "seconds": 0}
    monkeypatch.setattr(lam, "_generate", generate)
    with pytest.raises(PageConflictError):
        lam.page({"kind": "concept", "slug": "scn2a", "mode": "update", "established": "One new result."})
    assert len(calls) == 1
    assert "Concurrent scientific addition." in world.s3.text(key)
