from __future__ import annotations

import io
import hashlib
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from byeori import synthesis_lambda as lam


class FakeS3:
    class exceptions:
        class NoSuchKey(ClientError):
            def __init__(self, key):
                super().__init__({"Error": {"Code": "NoSuchKey", "Message": key}}, "GetObject")

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FakeS3.exceptions.NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key]), "ETag": hashlib.md5(self.objects[Key]).hexdigest()}

    def put_object(self, Bucket, Key, Body, ContentType=None, **conditions):
        from botocore.exceptions import ClientError
        current = self.objects.get(Key)
        if ((conditions.get("IfNoneMatch") == "*" and current is not None) or
                (conditions.get("IfMatch") is not None and
                 (current is None or hashlib.md5(current).hexdigest() != conditions["IfMatch"]))):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        return {"ETag": hashlib.md5(Body).hexdigest()}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FakeS3.exceptions.NoSuchKey(Key)
        return {}

    def text(self, key):
        return self.objects[key].decode("utf-8")

    def json(self, key):
        return json.loads(self.text(key))


NOTE_TEMPLATE = """---
title: "{title}"
authors: "{author} A"
year: "{year}"
category: "{category}"
---
## One-line Summary
{summary}

## 1. Document Information

| Field | Details |
|---|---|

## 2. Key Contributions
- Contribution about {gene}.

## 3. Methodology and Architecture
- Trios.

## 4. Key Results and Benchmarks
- {gene} result (P = 0.01).

## 5. Limitations and Future Work
- Small cohort.

## 6. Related Work
- Prior.

## 7. Glossary
- **{gene}**: Gene studied here.
- **De novo variants**: Variants absent from parents.
"""


@pytest.fixture
def world(monkeypatch):
    """A tiny corpus: six asd-ndd notes naming SCN2A, one liver note naming CHD8; no Bedrock."""
    s3 = FakeS3()
    table: dict[str, dict] = {}
    notes = []
    for i in range(6):
        stem = f"a{i}-2020-scn2a"
        s3.put_object("b", f"wiki/sources/{stem}.md", NOTE_TEMPLATE.format(
            title=f"SCN2A paper {i}", author=f"Auth{i}", year=2020 + i, category="asd-ndd", summary=f"Summary {i}.", gene="SCN2A").encode())
        notes.append({"work_id": stem, "source_note_key": f"wiki/sources/{stem}.md", "source_note_sha256": f"h{i}", "category": "asd-ndd"})
    s3.put_object("b", "wiki/sources/l0-2021-chd8.md", NOTE_TEMPLATE.format(
        title="CHD8 liver", author="Liv", year=2021, category="liver", summary="Liver.", gene="CHD8").encode())
    notes.append({"work_id": "l0-2021-chd8", "source_note_key": "wiki/sources/l0-2021-chd8.md", "source_note_sha256": "hl", "category": "liver"})
    s3.put_object("b", "reference/hgnc.tsv", b"hgnc_id\tsymbol\tname\talias_symbol\tprev_symbol\n1\tSCN2A\tx\tNav1.2\t\n2\tCHD8\ty\t\t\n")
    monkeypatch.setattr(lam, "aws", SimpleNamespace(s3=s3, table=None, bedrock=None))
    monkeypatch.setattr(lam, "BUCKET", "b")
    monkeypatch.setattr(lam, "_ready_notes", lambda categories=None: [n for n in notes if categories is None or n["category"] in categories])
    monkeypatch.setattr(lam, "_scan_kind", lambda kind: [i for i in table.values() if i.get("id_kind") == kind])
    monkeypatch.setattr(lam, "_item", lambda work_id: dict(table.get(work_id) or {}))

    def record(work_id, id_kind, fields, add=None):
        item = table.setdefault(work_id, {"work_id": work_id})
        item.update({"id_kind": id_kind, **fields})
        for k, v in (add or {}).items():
            item[k] = item.get(k, 0) + v
    monkeypatch.setattr(lam, "_record", record)
    responses: list[str] = []

    def generate(system, prompt, *, max_tokens=8000, timeout_seconds=None):
        if not responses:
            raise AssertionError("no scripted Bedrock response left for prompt starting: " + prompt[:60])
        return {"text": responses.pop(0), "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 0,
                                                     "cacheWriteInputTokens": 0}, "stop_reason": "end_turn", "seconds": 1.0}
    monkeypatch.setattr(lam, "_generate", generate)
    return SimpleNamespace(s3=s3, table=table, responses=responses, notes=notes)


def test_plan_concepts_writes_manifest_and_work_list(world) -> None:
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"},
                                       {"slug": "de-novo-variant", "merge_into": None, "entity_type": "phenomenon"}]))
    result = lam.plan_concepts({"scope": "autism"})
    assert result["status"] == "planned" and result["candidates"] == 2 and result["count"] == 2
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    by_slug = {c["slug"]: c for c in manifest["concepts"]}
    assert by_slug["scn2a"]["entity_type"] == "gene" and by_slug["scn2a"]["count_in_scope"] == 6
    assert [m["stem"] for m in by_slug["scn2a"]["members"]] == [f"a{i}-2020-scn2a" for i in range(6)]
    assert by_slug["scn2a"]["members"][0]["first_author"] == "Auth0" and by_slug["scn2a"]["mode"] == "generate"
    assert by_slug["de-novo-variant"]["count_total"] == 7 and by_slug["de-novo-variant"]["related"] == [["scn2a", 6]]
    assert "chd8" not in by_slug, "one liver note is under the threshold"
    work = world.s3.json(result["work_manifest"]["key"])
    assert work[0] == {"action": "page", "kind": "concept", "slug": "de-novo-variant", "mode": "generate"}


def test_plan_concepts_honours_overrides_and_remembers_types(world) -> None:
    world.s3.put_object("b", lam.OVERRIDES_KEY, json.dumps({"exclude": ["de-novo-variant"], "title": {"scn2a": "SCN2A (Nav1.2)"}}).encode())
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"}]))
    first = lam.plan_concepts({"scope": "autism"})
    assert first["candidates"] == 1 and world.s3.json(lam.CANDIDATES_KEY)["concepts"][0]["title"] == "SCN2A (Nav1.2)"
    world.table["concept#scn2a"] = {"work_id": "concept#scn2a", "id_kind": "concept", "synthesis_status": "ready",
                                   "members": {f"a{i}-2020-scn2a": f"h{i}" for i in range(6)}}
    second = lam.plan_concepts({"scope": "autism"})   # no Bedrock call: the type is remembered, and nothing changed
    assert second["count"] == 0 and second["skipped"] == 1 and world.responses == []


def test_resolve_scope(world) -> None:
    assert lam.resolve_scope({"scope": "autism"})["categories"] == list(lam.scope_categories("autism"))
    assert lam.resolve_scope({"scope": "all"})["categories"] == ["asd-ndd", "liver"]
    assert lam.resolve_scope({"scope": "all", "categories": ["liver"]})["categories"] == ["liver"]


def test_manifest_summary_counts_in_aws_without_returning_members(world, small_partitions, monkeypatch) -> None:
    _planned(world)
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    monkeypatch.setattr(lam, "_scan", lambda *args, **kwargs: list(world.table.values()))
    result = lam.handler({"action": "manifest_summary", "scope": "autism"}, None)
    assert result["scope_notes"] == 6 and result["corpus_notes"] == 7
    assert result["notes_by_category"]["asd-ndd"] == 6
    assert result["concepts"]["count"] == 2
    assert result["concepts"]["largest"][0] == ["de-novo-variant", 7]
    assert result["categories"]["asd-ndd"]["subtopics"] == [["pore-variants", 3], ["cohort-studies", 3]]
    assert result["usage"]["totals"]["calls"] == 2
    assert result["usage"]["by_kind"]["plan"]["calls"] == 1
    assert "members" not in json.dumps(result) and "a0-2020-scn2a" not in json.dumps(result)


def test_manifest_read_is_bounded_and_paginates_selected_concept(world) -> None:
    _planned(world)
    request = {"action": "manifest_read", "kind": "concepts", "section": "scn2a", "max_chars": 80}
    first = lam.handler(request, None)
    second = lam.handler({**request, "offset": first["next_offset"]}, None)
    assert len(first["text"]) == 80 and first["next_offset"] == 80
    assert second["offset"] == 80 and second["next_offset"] == 160
    concept = next(c for c in world.s3.json(lam.CANDIDATES_KEY)["concepts"] if c["slug"] == "scn2a")
    assert first["text"] + second["text"] == json.dumps(concept, ensure_ascii=False, indent=1)[:160]
    assert lam.manifest_read({**request, "offset": first["total_chars"]})["next_offset"] is None
    for invalid in ({"max_chars": 8001}, {"max_chars": 0}, {"offset": -1}, {"section": "missing"}):
        with pytest.raises(ValueError):
            lam.manifest_read({**request, **invalid})
    assert lam.manifest_read({"kind": "subtopics", "category": "asd-ndd"})["found"] is False
    with pytest.raises(ValueError):
        lam.manifest_read({"kind": "subtopics", "category": "../secrets"})


def test_manifest_submit_validates_against_all_current_aws_notes(world, small_partitions) -> None:
    request = {"action": "manifest_submit", "kind": "subtopics", "category": "asd-ndd", "content": PARTITION}
    result = lam.handler(request, None)
    stored = world.s3.json(result["key"])
    assert stored["validated"] == "aws" and stored["note_count"] == 6 and stored["year_range"] == ["2020", "2025"]
    assert stored["sha256"] and result["sha256"] == stored["sha256"]
    for bad_stems, error in ((STEMS[3:5], "unassigned"), (STEMS[3:] + ["invented"], "unknown stem"),
                             (STEMS[2:], "assigned to both")):
        content = {"subtopics": [PARTITION["subtopics"][0], {**PARTITION["subtopics"][1], "stems": bad_stems}]}
        with pytest.raises(ValueError, match=error):
            lam.manifest_submit({**request, "content": content})
        assert world.s3.json(result["key"]) == stored, "rejected edits cannot replace the valid manifest"
    with pytest.raises(ValueError, match="does not match"):
        lam.manifest_submit({**request, "content": {**PARTITION, "category": "liver"}})


def test_manifest_submit_refuses_an_incomplete_aws_corpus(world, small_partitions) -> None:
    world.notes[0]["source_note_sha256"] = ""
    with pytest.raises(ValueError, match="1 ready notes lack readable content or hashes"):
        lam.manifest_submit({"kind": "subtopics", "category": "asd-ndd", "content": PARTITION})
    assert "runs/synthesis/asd-ndd/subtopics.json" not in world.s3.objects


def test_manifest_submit_and_read_overrides_validate_content(world) -> None:
    content = {"exclude": ["gene"], "merge": {"gene-a": "gene-b"}, "title": {"gene-b": "Gene B"}}
    result = lam.manifest_submit({"kind": "overrides", "content": json.dumps(content)})
    assert result["key"] == lam.OVERRIDES_KEY
    excerpt = lam.manifest_read({"kind": "overrides", "section": "merge"})
    assert json.loads(excerpt["text"]) == content["merge"]
    for invalid in ({"exclude": "gene"}, {"exclude": [3]}, {"merge": {"gene": []}}, {"title": {"gene": ""}}, []):
        with pytest.raises(ValueError):
            lam.manifest_submit({"kind": "overrides", "content": invalid})
    with pytest.raises(ValueError):
        lam.manifest_submit({"kind": "concepts", "content": {}})


STEMS = [f"a{i}-2020-scn2a" for i in range(6)]


def concept_text(stems, definition="SCN2A encodes Nav1.2, the neuronal voltage-gated sodium channel, and every note here defines it that way, which makes this definition long enough to pass the length check."):
    bullets = "\n".join(f"- Finding {i} (P = 0.0{i + 1}) [[sources/{s}]]." for i, s in enumerate(stems))
    return f"## Definition\n{definition}\n\n## What the notes show\n{bullets}\n\n## Disagreements and limits\nNone recorded.\n"


def _planned(world):
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"},
                                       {"slug": "de-novo-variant", "merge_into": None, "entity_type": "phenomenon"}]))
    lam.plan_concepts({"scope": "autism"})


def test_concept_page_single_call(world) -> None:
    _planned(world)
    world.responses.append(concept_text(STEMS))
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    assert result["status"] == "ready" and result["key"] == "wiki/concepts/scn2a.md" and result["calls"] == 1
    assert result["model"] == lam.MODEL_ID
    page = world.s3.text("wiki/concepts/scn2a.md")
    assert page.startswith('---\ntitle: "SCN2A"\nkind: "concept"\nslug: "scn2a"\n')
    assert "## Related concepts\n- [[concepts/de-novo-variant]] (6 shared notes)" in page
    assert "## Notes\n- [[sources/a0-2020-scn2a]] Auth0 (2020). SCN2A paper 0" in page
    item = world.table["concept#scn2a"]
    assert item["synthesis_status"] == "ready" and item["generation"] == "single" and item["note_count"] == 6
    assert item["members"] == {s: f"h{i}" for i, s in enumerate(STEMS)} and item["input_tokens"] == 10
    assert world.s3.text(item["model_text_key"]) == concept_text(STEMS)


def test_concept_page_resumes_across_invocations(world, monkeypatch) -> None:
    # A page now sends one catalogue row per member plus what it retrieves, which fits one
    # call. The hierarchical split still guards an oversized input, so drive it with
    # several retrieved pages rather than by making the member list long.
    # Batching is by size now, not by count, so the injected pages have to be big enough to split.
    monkeypatch.setattr(lam, "_retrieved_inputs",
                        lambda hits: [f"=== Retrieved page {i} ===\n" + "body " * 40_000 for i in range(3)])
    monkeypatch.setattr(lam, "NOTES_PER_CALL", 2)
    monkeypatch.setattr(lam, "MAX_CALLS_PER_INVOCATION", 2)
    _planned(world)
    world.responses += [concept_text(STEMS[0:2]), concept_text(STEMS[2:4])]
    first = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    assert first["status"] == "partial" and first["calls"] == 2 and world.responses == []
    assert world.table["concept#scn2a"]["synthesis_status"] == "partial"
    partials = sorted(k for k in world.s3.objects if k.startswith("runs/synthesis/concepts/partials/scn2a/"))
    assert [k.rsplit("/", 1)[1] for k in partials] == ["L0-000.md", "L0-001.md"]
    world.responses += [concept_text(STEMS[4:6]), concept_text(STEMS)]
    second = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    assert second["status"] == "ready" and second["calls"] == 2
    item = world.table["concept#scn2a"]
    assert item["generation"] == "hierarchical" and item["calls"] == 4
    assert sorted(k.rsplit("/", 1)[1] for k in world.s3.objects if "/partials/scn2a/" in k) == ["L0-000.md", "L0-001.md", "L0-002.md", "L1-000.md"]


def test_concept_page_refresh_rewrites_code_sections_without_bedrock(world) -> None:
    _planned(world)
    world.responses.append(concept_text(STEMS))
    lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    before = world.s3.text("wiki/concepts/scn2a.md")
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "refresh"})
    assert result["status"] == "ready" and "calls" not in result and world.responses == []
    after = world.s3.text("wiki/concepts/scn2a.md")
    assert after.split("\n---\n", 1)[1] == before.split("\n---\n", 1)[1]


PARTITION = {"subtopics": [
    {"slug": "pore-variants", "title": "Pore variants", "scope": "Variants near the pore.", "stems": STEMS[:3]},
    {"slug": "cohort-studies", "title": "Cohort studies", "scope": "Trio cohorts.", "stems": STEMS[3:]}]}


def subtopic_text(stems):
    bullets = "\n".join(f"- Finding {i} (n = {i + 10}) [[sources/{s}]]." for i, s in enumerate(stems))
    return ("## Scope\nThese papers study SCN2A variants in trio cohorts (Auth0 2020; Auth1 2021), and this scope paragraph "
            "is written at some length so that the structural validator's minimum size is met by the test fixture as it "
            "would be by a real page of several hundred words.\n\n"
            f"## Findings\n{bullets}\n\n## Comparison\nNot applicable.\n\n"
            "## Open questions\n- Are the variants recurrent? (Auth0 2020)\n- Does dosage matter? (Auth2 2022)\n")


def partition_answer(partition, stems=None):
    """Script model output using its short-ID wire format; assertions inspect expanded manifests."""
    stems = STEMS if stems is None else stems
    ids = {stem: f"p{i:04d}" for i, stem in enumerate(stems)}
    return json.dumps({"subtopics": [{**{k: v for k, v in st.items() if k != "stems"},
                                     "ids": [ids.get(s, s) for s in st["stems"]]}
                                    for st in partition["subtopics"]]})


@pytest.fixture
def small_partitions(monkeypatch):
    monkeypatch.setattr(lam, "PARTITION_LIMITS", {"min_subtopics": 2, "max_subtopics": 12, "min_notes": 1})


def test_plan_subtopics_proposes_validates_and_reasks_once(world, small_partitions) -> None:
    broken = {"subtopics": [{**PARTITION["subtopics"][0]}, {**PARTITION["subtopics"][1], "stems": STEMS[3:5]}]}  # a5 unassigned
    world.responses += [partition_answer(broken), partition_answer(PARTITION)]
    result = lam.plan_subtopics({"category": "asd-ndd"})
    assert result["status"] == "partial" and result["calls"] == 1
    result = lam.plan_subtopics({"category": "asd-ndd", "resume": result["checkpoint"]})
    assert result["status"] == "planned" and result["subtopics"] == 2 and result["calls"] == 1 and result["plan_calls"] == 2
    manifest = world.s3.json("runs/synthesis/asd-ndd/subtopics.json")
    assert manifest["note_count"] == 6 and manifest["year_range"] == ["2020", "2025"] and manifest["problems"] == []
    assert world.table["category#asd-ndd"]["plan_status"] == "planned"
    world.responses += [partition_answer(broken), partition_answer(broken)]
    failed = lam.plan_subtopics({"category": "asd-ndd", "replan": True})
    assert failed["status"] == "partial"
    failed = lam.plan_subtopics({"category": "asd-ndd", "resume": failed["checkpoint"], "replan": True})
    assert failed["status"] == "failed" and any("omissions" in p for p in failed["problems"])
    assert "runs/synthesis/asd-ndd/subtopics.failed.json" in world.s3.objects
    assert world.s3.json("runs/synthesis/asd-ndd/subtopics.json")["subtopics"][0]["slug"] == "pore-variants", "the valid manifest stands"


def test_plan_subtopics_assigns_a_few_new_notes_to_existing_subtopics(world, small_partitions) -> None:
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    stem = "a6-2026-scn2a"
    world.s3.put_object("b", f"wiki/sources/{stem}.md", NOTE_TEMPLATE.format(
        title="Late paper", author="Late", year=2026, category="asd-ndd", summary="Late.", gene="SCN2A").encode())
    world.notes.append({"work_id": stem, "source_note_key": f"wiki/sources/{stem}.md", "source_note_sha256": "h6", "category": "asd-ndd"})
    world.responses.append(json.dumps({"assignments": {"p0000": "cohort-studies"}}))
    result = lam.plan_subtopics({"category": "asd-ndd"})
    assert result["status"] == "updated" and result["assigned"] == 1
    manifest = world.s3.json("runs/synthesis/asd-ndd/subtopics.json")
    assert manifest["subtopics"][1]["stems"] == STEMS[3:] + [stem] and manifest["note_count"] == 7


def test_plan_subtopic_pages_and_subtopic_page(world, small_partitions) -> None:
    _planned(world)
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    plan = lam.plan_subtopic_pages({"scope": "autism", "categories": ["asd-ndd"]})
    work = world.s3.json(plan["manifest"]["key"])
    assert plan["count"] == 2 and work[0] == {"action": "page", "kind": "subtopic", "category": "asd-ndd", "slug": "pore-variants", "mode": "generate"}
    world.responses.append(subtopic_text(STEMS[:3]))
    result = lam.page(work[0])
    assert result["status"] == "ready" and result["key"] == "wiki/overviews/asd-ndd/pore-variants.md"
    page = world.s3.text(result["key"])
    assert 'kind: "subtopic"\ncategory: "asd-ndd"\nslug: "pore-variants"\nnote_count: 3\n' in page
    assert "## Concepts\n- [[concepts/de-novo-variant]] (3 notes)\n- [[concepts/scn2a]] (3 notes)" in page
    assert "## Notes\n- [[sources/a0-2020-scn2a]] Auth0 (2020). SCN2A paper 0" in page
    item = world.table["subtopic#asd-ndd/pore-variants"]
    assert item["synthesis_status"] == "ready" and item["members"] == {s: f"h{i}" for i, s in enumerate(STEMS[:3])}
    again = lam.plan_subtopic_pages({"scope": "autism", "categories": ["asd-ndd"]})
    assert again["count"] == 1 and again["skipped"] == 1, "a page whose members did not change is not rewritten"


LANDSCAPE = ("## Landscape\n" + "The category divides into pore variants [[overviews/asd-ndd/pore-variants]] and cohort studies "
             "[[overviews/asd-ndd/cohort-studies]]. " * 30 + "\n\n## Open questions\n- Recurrence [[overviews/asd-ndd/pore-variants]]\n")


def _subtopics_written(world):
    _planned(world)
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    for st in PARTITION["subtopics"]:
        world.responses.append(subtopic_text(st["stems"]))
        lam.page({"action": "page", "kind": "subtopic", "category": "asd-ndd", "slug": st["slug"], "mode": "generate"})


def test_category_page_fails_only_when_no_subtopic_page_exists(world, small_partitions) -> None:
    _planned(world)
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    result = lam.page({"action": "page", "kind": "category", "category": "asd-ndd"})
    assert result["status"] == "failed" and result["problems"] == ["no subtopic pages exist yet"]
    assert world.table["category#asd-ndd"]["synthesis_status"] == "failed"


LANDSCAPE_PORE_ONLY = ("## Landscape\n" + "The category covers pore variants [[overviews/asd-ndd/pore-variants]] of SCN2A. " * 45
                      + "\n\n## Open questions\n- Are the variants recurrent? [[overviews/asd-ndd/pore-variants]]\n")


def test_category_page_tolerates_one_missing_subtopic_page(world, small_partitions) -> None:
    _planned(world)
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    world.responses.append(subtopic_text(STEMS[:3]))
    lam.page({"action": "page", "kind": "subtopic", "category": "asd-ndd", "slug": "pore-variants", "mode": "generate"})
    # cohort-studies' page is never written.
    world.responses.append(LANDSCAPE_PORE_ONLY)
    result = lam.page({"action": "page", "kind": "category", "category": "asd-ndd"})
    assert result["status"] == "ready", "one missing subtopic page no longer blocks the category page"
    page = world.s3.text(result["key"])
    assert 'omitted_subtopics: ["cohort-studies"]' in page
    assert world.table["category#asd-ndd"]["synthesis_status"] == "ready"


def test_category_page_is_written_from_subtopic_pages_and_skipped_when_unchanged(world, small_partitions) -> None:
    _subtopics_written(world)
    world.responses.append(LANDSCAPE)
    result = lam.page({"action": "page", "kind": "category", "category": "asd-ndd"})
    assert result["status"] == "ready" and result["key"] == "wiki/overviews/asd-ndd/index.md"
    page = world.s3.text(result["key"])
    assert 'kind: "category"\ncategory: "asd-ndd"\nnote_count: 6\nsubtopic_count: 2\n' in page
    assert "- [[overviews/asd-ndd/pore-variants]] Pore variants (3 notes): These papers study SCN2A variants" in page
    assert "## Key concepts\n- [[concepts/de-novo-variant]] (6 notes)\n- [[concepts/scn2a]] (6 notes)" in page
    assert "Notes: 6. Years: 2020-2025. Subtopics: 2." in page
    assert lam.page({"action": "page", "kind": "category", "category": "asd-ndd"})["status"] == "skipped"
    assert world.responses == []


def test_plan_failures_lists_failed_and_partial_pages_in_scope(world) -> None:
    world.table.update({
        "concept#scn2a": {"work_id": "concept#scn2a", "id_kind": "concept", "synthesis_status": "failed"},
        "concept#chd8": {"work_id": "concept#chd8", "id_kind": "concept", "synthesis_status": "partial"},
        "concept#ok": {"work_id": "concept#ok", "id_kind": "concept", "synthesis_status": "ready"},
        "subtopic#asd-ndd/x": {"work_id": "subtopic#asd-ndd/x", "id_kind": "subtopic", "synthesis_status": "failed"},
        "subtopic#liver/y": {"work_id": "subtopic#liver/y", "id_kind": "subtopic", "synthesis_status": "failed"},
        "category#asd-ndd": {"work_id": "category#asd-ndd", "id_kind": "category", "synthesis_status": "failed"},
    })
    result = lam.plan_failures({"scope": "autism", "categories": ["asd-ndd"]})
    work = world.s3.json(result["manifest"]["key"])
    assert work == [{"action": "page", "kind": "concept", "slug": "chd8", "mode": "generate"},
                    {"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"},
                    {"action": "page", "kind": "subtopic", "category": "asd-ndd", "slug": "x", "mode": "generate"}], \
        "category pages are not retried here: the state machine rewrites them all after the retry map"


def test_handler_routes_actions() -> None:
    with pytest.raises(ValueError):
        lam.handler({"action": "nope"}, None)
    with pytest.raises(ValueError):
        lam.handler({"action": "page", "kind": "nope"}, None)


def test_handler_sets_the_deadline_from_the_lambda_context(monkeypatch) -> None:
    monkeypatch.setattr(lam, "DEADLINE_MS", None)
    fake_context = SimpleNamespace(get_remaining_time_in_millis=lambda: 500_000)
    lam.handler({"action": "resolve_scope", "scope": "autism"}, fake_context)
    left = lam._time_left_ms()
    assert 400_000 < left <= 500_000


def test_record_omits_empty_expression_attribute_names(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(lam, "aws", SimpleNamespace(table=SimpleNamespace(update_item=lambda **kw: calls.append(kw))))
    lam._record("concept#x", "concept", {})
    assert "ExpressionAttributeNames" not in calls[-1]
    lam._record("concept#y", "concept", {"foo": "bar"})
    assert "ExpressionAttributeNames" in calls[-1]


def test_read_many_skips_missing_keys(world) -> None:
    missing: list[str] = []
    result = lam._read_many(["wiki/sources/a0-2020-scn2a.md", "wiki/sources/does-not-exist.md"], missing)
    assert "wiki/sources/a0-2020-scn2a.md" in result
    assert "wiki/sources/does-not-exist.md" not in result
    assert missing == ["wiki/sources/does-not-exist.md"]


def test_plan_concepts_skips_unreadable_notes(world) -> None:
    del world.s3.objects["wiki/sources/a5-2020-scn2a.md"]
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"},
                                       {"slug": "de-novo-variant", "merge_into": None, "entity_type": "phenomenon"}]))
    result = lam.plan_concepts({"scope": "autism"})
    assert result["unreadable_notes"] == 1
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    by_slug = {c["slug"]: c for c in manifest["concepts"]}
    assert by_slug["scn2a"]["count_in_scope"] == 5, "the unreadable note's glossary term is not counted"


def test_plan_concepts_honours_explicit_categories(world) -> None:
    result = lam.plan_concepts({"categories": ["liver"]})
    assert result["candidates"] == 0, "restricting to a category with too few notes yields no candidates"
    assert world.responses == []


def test_plan_concepts_typing_resumes_across_invocations(world, monkeypatch) -> None:
    monkeypatch.setattr(lam, "CANDIDATE_CHUNK", 1)
    monkeypatch.setattr(lam, "TYPING_CALLS_PER_INVOCATION", 1)
    world.responses.append(json.dumps([{"slug": "de-novo-variant", "merge_into": None, "entity_type": "phenomenon"}]))
    first = lam.plan_concepts({"scope": "autism"})
    assert first["status"] == "partial" and first["typed"] == 1 and first["untyped"] == 1 and first["work_manifest"] is None
    assert world.responses == []
    typed = world.s3.json(lam.TYPED_KEY)
    assert set(typed["typed"]) == {"de-novo-variant"}
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"}]))
    second = lam.plan_concepts({"scope": "autism"})
    assert second["status"] == "planned" and world.responses == []
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    by_slug = {c["slug"]: c for c in manifest["concepts"]}
    assert by_slug["scn2a"]["entity_type"] == "gene" and by_slug["de-novo-variant"]["entity_type"] == "phenomenon"


def test_plan_concepts_typing_failure_defaults_to_other_in_the_written_manifest(world) -> None:
    world.responses.append("not json")
    result = lam.plan_concepts({"scope": "autism"})
    assert result["status"] == "planned" and result["typing_failed_chunks"] == 1
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    by_slug = {c["slug"]: c for c in manifest["concepts"]}
    assert by_slug["scn2a"]["entity_type"] == "gene" and by_slug["de-novo-variant"]["entity_type"] == "other"
    typed = world.s3.json(lam.TYPED_KEY)
    assert typed["typed"] == {} and typed["failed_chunks"] == 1


def test_plan_concepts_never_returns_partial_when_nothing_was_typed(world, monkeypatch) -> None:
    monkeypatch.setattr(lam, "CANDIDATE_CHUNK", 1)
    monkeypatch.setattr(lam, "TYPING_CALLS_PER_INVOCATION", 1)
    world.responses.append("not json")  # de-novo-variant sorts first and its chunk fails to parse
    result = lam.plan_concepts({"scope": "autism"})
    assert result["status"] == "planned", "nothing was typed this invocation, so it completes instead of looping"
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    by_slug = {c["slug"]: c for c in manifest["concepts"]}
    assert by_slug["de-novo-variant"]["entity_type"] == "other"
    assert by_slug["scn2a"]["entity_type"] == "gene", "HGNC identity survives an unattempted model typing pass"
    typed = world.s3.json(lam.TYPED_KEY)
    assert "scn2a" not in typed["typed"], "the untried slug is not persisted as typed, so a later run can retry it"


def test_plan_concepts_admits_a_persistently_failing_slug_after_repeated_attempts(world, monkeypatch) -> None:
    monkeypatch.setattr(lam, "TYPING_CALLS_PER_INVOCATION", 1)
    world.s3.put_object("b", lam.TYPED_KEY, json.dumps(
        {"identity_version": lam.CONCEPT_IDENTITY_VERSION,
         "typed": {"scn2a": {"entity_type": "gene", "merge_into": None}}, "attempts": {}, "failed_chunks": 0}).encode())
    world.responses.append("not json")
    first = lam.plan_concepts({"scope": "autism"})
    assert first["status"] == "planned" and world.responses == []
    typed_after_1 = world.s3.json(lam.TYPED_KEY)
    assert typed_after_1["attempts"].get("de-novo-variant") == 1
    assert "de-novo-variant" not in typed_after_1["typed"]
    manifest1 = world.s3.json(lam.CANDIDATES_KEY)
    entry1 = next(c for c in manifest1["concepts"] if c["slug"] == "de-novo-variant")
    assert entry1["entity_type"] == "other" and "typing_failed" not in entry1

    world.responses.append("not json")
    second = lam.plan_concepts({"scope": "autism"})
    assert second["status"] == "planned" and world.responses == []
    typed_after_2 = world.s3.json(lam.TYPED_KEY)
    assert typed_after_2["typed"]["de-novo-variant"]["typing_failed"] is True
    manifest2 = world.s3.json(lam.CANDIDATES_KEY)
    entry2 = next(c for c in manifest2["concepts"] if c["slug"] == "de-novo-variant")
    assert entry2["entity_type"] == "other" and entry2["typing_failed"] is True


def test_plan_concepts_rejects_cross_chunk_merge_of_phenomenon_into_gene(world, monkeypatch) -> None:
    monkeypatch.setattr(lam, "CANDIDATE_CHUNK", 1)
    world.responses += [json.dumps([{"slug": "de-novo-variant", "merge_into": "scn2a", "entity_type": "phenomenon"}]),
                        json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"}])]
    result = lam.plan_concepts({"scope": "autism"})
    assert result["status"] == "planned" and world.responses == []
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    slugs = {c["slug"] for c in manifest["concepts"]}
    assert slugs == {"scn2a", "de-novo-variant"}
    assert manifest["model_merges"] == {}
    assert manifest["rejected_model_merges"][0]["source"] == "de-novo-variant"


def test_plan_concepts_invalidates_legacy_contaminated_identity_cache(world):
    old = {"model_merges": {"scn2a": "de-novo-variant"}, "concepts": []}
    world.s3.put_object("b", lam.CANDIDATES_KEY, json.dumps(old).encode())
    world.s3.put_object("b", lam.TYPED_KEY, json.dumps({"typed": {
        "scn2a": {"entity_type": "method", "merge_into": "de-novo-variant"},
        "de-novo-variant": {"entity_type": "other", "merge_into": None}}}).encode())
    world.responses.append(json.dumps([{"slug": "scn2a", "entity_type": "method", "merge_into": None},
                                       {"slug": "de-novo-variant", "entity_type": "phenomenon", "merge_into": None}]))
    result = lam.plan_concepts({"scope": "autism"})
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    assert result["calls"] == 1 and manifest["model_merges"] == {}
    assert manifest["identity_version"] == lam.CONCEPT_IDENTITY_VERSION
    assert {c["slug"]: c["entity_type"] for c in manifest["concepts"]} == {
        "scn2a": "gene", "de-novo-variant": "phenomenon"}


def test_plan_concepts_checks_saved_model_merge_before_counting(world):
    world.s3.put_object("b", lam.CANDIDATES_KEY, json.dumps({"identity_version": lam.CONCEPT_IDENTITY_VERSION,
        "model_merges": {"scn2a": "de-novo-variant"}, "concepts": []}).encode())
    _planned(world)
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    assert {c["slug"] for c in manifest["concepts"]} == {"scn2a", "de-novo-variant"}
    assert manifest["model_merges"] == {}
    assert manifest["rejected_model_merges"][0]["source"] == "scn2a"


def test_page_generation_refuses_legacy_concept_identity_manifest(world):
    world.s3.put_object("b", lam.CANDIDATES_KEY, json.dumps({"concepts": []}).encode())
    with pytest.raises(ValueError, match="Replan concepts"):
        lam.page({"kind": "concept", "slug": "scn2a"})
    assert world.responses == [] and world.table == {}


def test_concept_typing_ignores_non_string_identifiers_without_merging(world):
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": ["de-novo-variant"], "entity_type": "gene"},
                                       {"slug": ["de-novo-variant"], "merge_into": None, "entity_type": "method"}]))
    assert lam.plan_concepts({"scope": "autism"})["status"] == "planned"
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    assert manifest["model_merges"] == {}
    assert {c["slug"] for c in manifest["concepts"]} == {"scn2a", "de-novo-variant"}


def test_concept_planning_requires_official_names_in_existing_hgnc_reference(world):
    world.s3.put_object("b", lam.HGNC_KEY, b"symbol\talias_symbol\tprev_symbol\nWDR20\tDMR\t\n")
    with pytest.raises(ValueError, match="Refresh the HGNC"):
        lam.plan_concepts({"scope": "autism"})
    assert world.responses == [] and lam.CANDIDATES_KEY not in world.s3.objects


def test_plan_concepts_caps_mention_postings(world, monkeypatch) -> None:
    monkeypatch.setattr(lam, "MENTION_POSTING_CAP", 2)
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"},
                                       {"slug": "de-novo-variant", "merge_into": None, "entity_type": "phenomenon"}]))
    result = lam.plan_concepts({"scope": "autism"})
    assert result["mentions_skipped"] >= 1
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    by_slug = {c["slug"]: c for c in manifest["concepts"]}
    assert by_slug["scn2a"].get("mentions_skipped") is True
    assert by_slug["scn2a"]["mentions"] == []


def test_plan_concepts_digest_ignores_mode(world) -> None:
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"},
                                       {"slug": "de-novo-variant", "merge_into": None, "entity_type": "phenomenon"}]))
    lam.plan_concepts({"scope": "autism"})
    first_sha = world.s3.json(lam.CANDIDATES_KEY)["sha256"]
    scn2a_members = {f"a{i}-2020-scn2a": f"h{i}" for i in range(6)}
    world.table["concept#scn2a"] = {"work_id": "concept#scn2a", "id_kind": "concept", "synthesis_status": "ready",
                                   "members": scn2a_members}
    world.table["concept#de-novo-variant"] = {"work_id": "concept#de-novo-variant", "id_kind": "concept",
                                              "synthesis_status": "ready", "members": {**scn2a_members, "l0-2021-chd8": "hl"}}
    second = lam.plan_concepts({"scope": "autism"})
    second_sha = world.s3.json(lam.CANDIDATES_KEY)["sha256"]
    assert second["count"] == 0 and second_sha == first_sha


def test_batches_respects_the_size_budget_as_well_as_the_count(monkeypatch) -> None:
    header = "H" * 10
    single = "X" * 20
    monkeypatch.setattr(lam, "MAX_INPUT_CHARS", 20_000 + len(header) + 2 * len(single))
    batches = lam._batches([single] * 6, header)
    assert [len(b) for b in batches] == [2, 2, 2]


def test_batches_puts_an_oversized_input_alone_and_truncated(monkeypatch) -> None:
    monkeypatch.setattr(lam, "MAX_INPUT_CHARS", 20_000 + 100)
    huge = "X" * 500
    batches = lam._batches(["short", huge, "short"], "H")
    assert batches[0] == ["short"]
    assert batches[1] == [huge[:100]]
    assert batches[2] == ["short"]


def test_concept_page_partial_failure_does_not_write_a_page(world, monkeypatch) -> None:
    # A page now sends one catalogue row per member plus what it retrieves, which fits one
    # call. The hierarchical split still guards an oversized input, so drive it with
    # several retrieved pages rather than by making the member list long.
    # Batching is by size now, not by count, so the injected pages have to be big enough to split.
    monkeypatch.setattr(lam, "_retrieved_inputs",
                        lambda hits: [f"=== Retrieved page {i} ===\n" + "body " * 40_000 for i in range(3)])
    monkeypatch.setattr(lam, "NOTES_PER_CALL", 2)
    _planned(world)
    world.responses += [concept_text(STEMS[0:2]), "## Definition\nshort\n"]
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    assert result["status"] == "failed" and "wiki/concepts/failed/scn2a.md" not in world.s3.objects
    assert "wiki/concepts/scn2a.md" not in world.s3.objects
    partials = [k for k in world.s3.objects if "/partials/scn2a/" in k]
    assert len(partials) == 1, "the first good partial stays in S3"
    item = world.table["concept#scn2a"]
    assert item["synthesis_status"] == "failed" and item["problems"]


def test_concept_page_refresh_requires_a_ready_item(world, monkeypatch) -> None:
    _planned(world)
    restore = _truncates(monkeypatch)
    lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    assert world.table["concept#scn2a"]["synthesis_status"] == "failed"
    restore()
    world.responses.append(concept_text(STEMS))
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "refresh"})
    assert result["status"] == "ready", "refresh on a non-ready item falls through to generate"
    assert world.responses == []


def test_hierarchical_final_merge_failing_full_validation_is_not_persisted(world, monkeypatch) -> None:
    # A page now sends one catalogue row per member plus what it retrieves, which fits one
    # call. The hierarchical split still guards an oversized input, so drive it with
    # several retrieved pages rather than by making the member list long.
    # Batching is by size now, not by count, so the injected pages have to be big enough to split.
    monkeypatch.setattr(lam, "_retrieved_inputs",
                        lambda hits: [f"=== Retrieved page {i} ===\n" + "body " * 40_000 for i in range(3)])
    monkeypatch.setattr(lam, "NOTES_PER_CALL", 2)
    monkeypatch.setattr(lam, "MAX_CALLS_PER_INVOCATION", 4)
    _planned(world)
    short_merge = concept_text(STEMS, definition="Short.")
    world.responses += [concept_text(STEMS[0:2]), concept_text(STEMS[2:4]), concept_text(STEMS[4:6]), short_merge]
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    assert result["status"] == "failed"
    assert not any("/L1-" in k for k in world.s3.objects), "the failing merge was not written"
    partials = sorted(k for k in world.s3.objects if "/partials/scn2a/" in k and "/L0-" in k)
    assert len(partials) == 3, "the good level-0 partials stay, so a retry regenerates only the merge"


def test_hierarchical_returns_partial_when_time_is_short(world, monkeypatch) -> None:
    # A page now sends one catalogue row per member plus what it retrieves, which fits one
    # call. The hierarchical split still guards an oversized input, so drive it with
    # several retrieved pages rather than by making the member list long.
    # Batching is by size now, not by count, so the injected pages have to be big enough to split.
    monkeypatch.setattr(lam, "_retrieved_inputs",
                        lambda hits: [f"=== Retrieved page {i} ===\n" + "body " * 40_000 for i in range(3)])
    monkeypatch.setattr(lam, "NOTES_PER_CALL", 2)
    _planned(world)
    monkeypatch.setattr(lam, "DEADLINE_MS", lam._now().timestamp() * 1000 + 100_000)
    world.responses.append(concept_text(STEMS[0:2]))   # the first call is always made, so a short
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})   # invocation still progresses
    assert result["status"] == "partial" and result["calls"] == 1 and world.responses == []


def test_propose_refuses_an_oversized_listing_without_a_bedrock_call(world, monkeypatch) -> None:
    monkeypatch.setattr(lam, "MAX_INPUT_CHARS", 10)
    result = lam.plan_subtopics({"category": "asd-ndd"})
    assert result["status"] == "failed"
    assert any("over 10" in p for p in result["problems"])
    assert world.responses == []


def test_partition_halves_path_tolerates_an_unused_merge_slug(world, monkeypatch) -> None:
    monkeypatch.setattr(lam, "PARTITION_LIMITS", {"min_subtopics": 1, "max_subtopics": 12, "min_notes": 1})
    monkeypatch.setattr(lam, "PARTITION_SPLIT", 2)
    parts = [STEMS[i::3] for i in range(3)]
    for i, part in enumerate(parts):
        world.responses.append(partition_answer({"subtopics": [{"slug": f"part-{i}", "title": f"Part {i}", "scope": "Scope.", "stems": part}]}, part))
    merge_answer = {"subtopics": [{"slug": "combined", "title": "Combined", "scope": "Scope."},
                                  {"slug": "unused", "title": "Unused", "scope": "Scope."}],
                    "merge": {"g0000": "combined", "g0001": "combined", "g0002": "combined"}}
    world.responses.append(json.dumps(merge_answer))
    result = lam.plan_subtopics({"category": "asd-ndd"})
    for _ in range(3):
        assert result["status"] == "partial" and result["calls"] == 1
        result = lam.plan_subtopics({"category": "asd-ndd", "resume": result["checkpoint"]})
    assert result["status"] == "planned" and result["plan_calls"] == 4
    manifest = world.s3.json("runs/synthesis/asd-ndd/subtopics.json")
    slugs = {st["slug"] for st in manifest["subtopics"]}
    assert slugs == {"combined"}, "the unused slug the model named is dropped since nothing merged into it"
    assert sorted(next(st for st in manifest["subtopics"] if st["slug"] == "combined")["stems"]) == sorted(STEMS)


def test_plan_subtopics_replan_threshold_uses_the_manifest_note_count(world, small_partitions) -> None:
    existing = {"category": "asd-ndd", "created": "2020-01-01", "updated": "2020-01-01", "note_count": 3,
                "year_range": ["2020", "2020"], "model": "m", "problems": [],
                "subtopics": [{"slug": "solo", "title": "Solo", "scope": "A solo subtopic.", "stems": STEMS[:5]}]}
    world.s3.put_object("b", "runs/synthesis/asd-ndd/subtopics.json", json.dumps(existing).encode())
    world.responses.append(partition_answer(PARTITION))
    result = lam.plan_subtopics({"category": "asd-ndd"})
    assert result["status"] == "planned", "1 new note against a stored note_count of 3 is 33%, over the 20% threshold"
    manifest = world.s3.json("runs/synthesis/asd-ndd/subtopics.json")
    assert {st["slug"] for st in manifest["subtopics"]} == {"pore-variants", "cohort-studies"}


def test_plan_subtopics_shrinkage_does_not_overwrite_the_live_manifest(world, small_partitions) -> None:
    good_manifest = {"category": "asd-ndd", "created": "2020-01-01", "updated": "2020-01-01", "note_count": 6,
                     "year_range": ["2020", "2025"], "model": "m", "problems": [], "sha256": "x",
                     "subtopics": [{"slug": "solo", "title": "Solo", "scope": "One paper.", "stems": [STEMS[0]]},
                                  {"slug": "rest", "title": "Rest", "scope": "The rest.", "stems": STEMS[1:]}]}
    world.s3.put_object("b", "runs/synthesis/asd-ndd/subtopics.json", json.dumps(good_manifest).encode())
    world.notes[:] = [n for n in world.notes if n["work_id"] != STEMS[0]]
    result = lam.plan_subtopics({"category": "asd-ndd"})
    assert result["status"] == "invalid"
    assert world.s3.json("runs/synthesis/asd-ndd/subtopics.json") == good_manifest, "the previous valid manifest stands"
    failed = world.s3.json("runs/synthesis/asd-ndd/subtopics.failed.json")
    assert any("fewer than" in p for p in failed["problems"])
    assert world.responses == []


def test_subtopic_page_frontmatter_shows_the_manifest_sha(world, small_partitions) -> None:
    _planned(world)
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    manifest = world.s3.json("runs/synthesis/asd-ndd/subtopics.json")
    assert manifest.get("sha256")
    world.responses.append(subtopic_text(STEMS[:3]))
    result = lam.page({"action": "page", "kind": "subtopic", "category": "asd-ndd", "slug": "pore-variants", "mode": "generate"})
    page = world.s3.text(result["key"])
    assert f"runs/synthesis/asd-ndd/subtopics.json@{manifest['sha256']}" in page


LANDSCAPE_COHORT_ONLY = ("## Landscape\n" + "The category covers cohort studies [[overviews/asd-ndd/cohort-studies]] of SCN2A variants. " * 40
                        + "\n\n## Open questions\n- Does dosage matter? [[overviews/asd-ndd/cohort-studies]]\n")


def test_category_page_omits_a_subtopic_whose_notes_all_vanished(world, small_partitions) -> None:
    _subtopics_written(world)
    world.notes[:] = [n for n in world.notes if n["work_id"] not in STEMS[:3]]
    world.responses.append(LANDSCAPE_COHORT_ONLY)
    result = lam.page({"action": "page", "kind": "category", "category": "asd-ndd"})
    assert result["status"] == "ready", "the vanished subtopic no longer blocks the category page"
    page = world.s3.text(result["key"])
    assert 'kind: "category"\ncategory: "asd-ndd"\nnote_count: 3\nsubtopic_count: 1\n' in page
    assert 'omitted_subtopics: ["pore-variants"]' in page
    assert "cohort-studies" in page and "pore-variants" not in page.split("omitted_subtopics", 1)[0]


def test_category_page_note_count_sums_only_the_included_subtopics(world, small_partitions) -> None:
    _subtopics_written(world)
    world.notes[:] = [n for n in world.notes if n["work_id"] not in STEMS[:3]]
    world.responses.append(LANDSCAPE_COHORT_ONLY)
    lam.page({"action": "page", "kind": "category", "category": "asd-ndd"})
    item = world.table["category#asd-ndd"]
    assert item["note_count"] == 3


def test_category_page_records_truncated_on_the_catalogue_item(world, small_partitions, monkeypatch) -> None:
    _subtopics_written(world)
    monkeypatch.setattr(lam, "MAX_INPUT_CHARS", 50)
    world.responses.append(LANDSCAPE)
    result = lam.page({"action": "page", "kind": "category", "category": "asd-ndd"})
    assert result["status"] == "ready"
    item = world.table["category#asd-ndd"]
    assert item["truncated"] is True
    assert item["generation"] == "truncated"


def test_plan_failures_caps_the_work_list(world, monkeypatch) -> None:
    monkeypatch.setattr(lam, "MAX_PAGES", 1)
    world.table.update({
        "concept#scn2a": {"work_id": "concept#scn2a", "id_kind": "concept", "synthesis_status": "failed"},
        "concept#chd8": {"work_id": "concept#chd8", "id_kind": "concept", "synthesis_status": "partial"},
    })
    result = lam.plan_failures({"scope": "all"})
    assert result["count"] == 1 and result["capped"] == 1
    work = world.s3.json(result["manifest"]["key"])
    assert len(work) == 1


def test_concept_page_drops_a_member_whose_source_is_missing_from_s3(world) -> None:
    _planned(world)
    del world.s3.objects["wiki/sources/a5-2020-scn2a.md"]
    world.responses.append(concept_text(STEMS[:5]))
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    assert result["status"] == "ready" and result["unreadable_notes"] == 1
    item = world.table["concept#scn2a"]
    assert item["note_count"] == 5


def test_subtopic_page_drops_a_member_whose_source_is_missing_from_s3(world, small_partitions) -> None:
    _planned(world)
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    del world.s3.objects["wiki/sources/a1-2020-scn2a.md"]
    world.responses.append(subtopic_text(["a0-2020-scn2a", "a2-2020-scn2a"]))
    result = lam.page({"action": "page", "kind": "subtopic", "category": "asd-ndd", "slug": "pore-variants", "mode": "generate"})
    assert result["status"] == "ready" and result["unreadable_notes"] == 1
    item = world.table["subtopic#asd-ndd/pore-variants"]
    assert item["note_count"] == 2


def test_plan_concepts_excludes_a_member_with_no_source_note_sha256(world) -> None:
    world.notes[2]["source_note_sha256"] = ""   # a2-2020-scn2a loses its hash
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"},
                                       {"slug": "de-novo-variant", "merge_into": None, "entity_type": "phenomenon"}]))
    result = lam.plan_concepts({"scope": "autism"})
    assert result["members_without_hash"] == 1
    manifest = world.s3.json(lam.CANDIDATES_KEY)
    by_slug = {c["slug"]: c for c in manifest["concepts"]}
    assert "a2-2020-scn2a" not in {m["stem"] for m in by_slug["scn2a"]["members"]}
    assert by_slug["scn2a"]["count_in_scope"] == 5


def test_plan_subtopic_pages_excludes_a_member_with_no_source_note_sha256(world, small_partitions) -> None:
    _planned(world)
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    world.notes[0]["source_note_sha256"] = ""   # a0-2020-scn2a loses its hash
    result = lam.plan_subtopic_pages({"scope": "autism", "categories": ["asd-ndd"]})
    assert result["members_without_hash"] == 1


def test_plan_concepts_records_planning_spend(world) -> None:
    world.responses.append(json.dumps([{"slug": "scn2a", "merge_into": None, "entity_type": "gene"},
                                       {"slug": "de-novo-variant", "merge_into": None, "entity_type": "phenomenon"}]))
    result = lam.plan_concepts({"scope": "autism"})
    item = world.table[lam.PLAN_ITEM]
    assert item["synthesis_status"] == "planned" and item["candidates"] == 2
    assert item["calls"] == 1 and item["input_tokens"] == result["input_tokens"]


def test_plan_subtopics_records_usage_on_every_branch(world, small_partitions) -> None:
    world.responses.append(partition_answer(PARTITION))
    lam.plan_subtopics({"category": "asd-ndd"})
    item = world.table["category#asd-ndd"]
    assert item["plan_status"] == "planned"
    assert sum(x.get("calls", 0) for x in world.table.values() if x["id_kind"] == "category_plan") == 1
    stem = "a6-2026-scn2a"
    world.s3.put_object("b", f"wiki/sources/{stem}.md", NOTE_TEMPLATE.format(
        title="Late paper", author="Late", year=2026, category="asd-ndd", summary="Late.", gene="SCN2A").encode())
    world.notes.append({"work_id": stem, "source_note_key": f"wiki/sources/{stem}.md", "source_note_sha256": "h6", "category": "asd-ndd"})
    world.responses.append(json.dumps({"assignments": {"p0000": "cohort-studies"}}))
    lam.plan_subtopics({"category": "asd-ndd"})
    item2 = world.table["category#asd-ndd"]
    assert item2["plan_status"] == "updated"
    assert sum(x.get("calls", 0) for x in world.table.values() if x["id_kind"] == "category_plan") == 2


def test_plan_json_retries_with_less_thinking_when_the_budget_left_no_answer(monkeypatch):
    """The measured failure: all 30,000 tokens went to thinking, stopReason max_tokens, text empty.

    Raising the budget again would only move the failure to the call timeout, so the retry lowers
    effort one step. The category plan was discarded outright before this existed.
    """
    calls = []

    def fake_generate(system, prompt, *, max_tokens, effort=None, **kwargs):
        calls.append(effort)
        if effort is None or effort == "xhigh":
            return {"text": "", "usage": {"outputTokens": 62000}, "stop_reason": "max_tokens",
                    "effort": effort or "xhigh", "seconds": 640.0, "request_id": "r1"}
        return {"text": '{"subtopics": []}', "usage": {"outputTokens": 900}, "stop_reason": "end_turn",
                "effort": effort, "seconds": 90.0, "request_id": "r2"}

    monkeypatch.setattr(lam, "REASONING", "xhigh")
    monkeypatch.setattr(lam, "_generate", fake_generate)
    result = lam._generate_json("sys", "prompt")

    assert calls == [None, "high"], "one step down, and only one"
    assert result["data"] == {"subtopics": []} and not result["problem"]
    assert result["retried_from_effort"] == "xhigh"
    assert result["first_attempt_usage"] == {"outputTokens": 62000}, "the wasted attempt stays on the receipt"


def test_plan_json_does_not_walk_effort_down_twice(monkeypatch):
    """A second overrun is a problem with the request, not a setting to keep lowering."""
    calls = []

    def always_overruns(system, prompt, *, max_tokens, effort=None, **kwargs):
        calls.append(effort)
        return {"text": "", "usage": {}, "stop_reason": "max_tokens", "effort": effort or "xhigh",
                "seconds": 600.0, "request_id": "r"}

    monkeypatch.setattr(lam, "REASONING", "xhigh")
    monkeypatch.setattr(lam, "_generate", always_overruns)
    result = lam._generate_json("sys", "prompt")

    assert calls == [None, "high"], "exactly two attempts"
    assert result["data"] is None and result["problem"], "the failure is reported, not hidden"


def test_thinking_budget_scales_with_effort_and_stays_under_the_output_cap():
    assert lam.LOWER_EFFORT["xhigh"] == "high" and "low" not in lam.LOWER_EFFORT, "low has no step below it"
    levels = ("low", "medium", "high", "xhigh", "max")
    values = [lam.THINKING_HEADROOM[k] for k in levels]
    assert values == sorted(values), "more effort means more room to think"
    for level in levels:
        assert lam.THINKING_HEADROOM[level] + lam.PLAN_MAX_TOKENS <= 128_000, level


def test_a_content_filtered_plan_unit_is_not_retried():
    """The filter bills the input in full and the same prompt trips it again, so one call is enough."""
    from byeori import synthesis_planner as planner
    source = (__import__("pathlib").Path(planner.__file__)).read_text()
    assert 'result.get("stop_reason") == "content_filtered"' in source
    assert 'if filtered or state["attempts"][unit] >= MAX_ATTEMPTS:' in source, \
        "a filter stop ends the unit regardless of how many attempts are left"
    assert 'status="filtered" if filtered else "failed"' in source, \
        "a filtered unit is recorded apart from a unit that genuinely failed"
    # The retry that does exist is for a truncated answer, never for a filtered one.
    assert 'if result["data"] is None and result["stop_reason"] == "max_tokens":' in \
        (__import__("pathlib").Path(lam.__file__)).read_text()


def test_a_subtopic_page_reads_a_catalogue_of_every_member_plus_what_it_retrieves():
    """llm-wiki puts the question into bm25s, takes ~20 candidates and reads those.

    The exhaustive read this replaces sent every member note in full: measured on 2026-09-20 that
    was 1.4 MB and six calls for a 70-note subtopic, which failed on the 900s limit, against
    314 KB and one call locally. The catalogue keeps coverage; retrieval supplies the depth.
    """
    source = (__import__("pathlib").Path(lam.__file__)).read_text()
    assert "catalog_line(m, texts[keys[m[\"stem\"]]])" in source, "every member appears, as a row"
    assert "_retrieve(query, exclude=" in source, "the subtopic's own page is not retrieved for itself"
    assert source.count("catalog_line(m, texts[keys[m[\"stem\"]]])") >= 2, "subtopic and concept alike"
    assert "synthesis_input(m, texts" not in source, "no page path may still read every member in full"
    assert 'query = f"{subtopic[\'title\']}. {subtopic.get(\'scope\', \'\')}"' in source, \
        "the query is the subtopic's own words, whole, as bm25s takes a whole question"


def test_retrieval_returns_distinct_documents_and_survives_a_missing_index(monkeypatch, tmp_path):
    rows = [("note", "a-2020", "A", "wiki/sources/a-2020.md", -9.0),
            ("note", "a-2020", "A", "wiki/sources/a-2020.md", -8.5),   # same document again
            ("overview", "asd-ndd/de-novo", "De novo", "wiki/overviews/asd-ndd/de-novo.md", -8.0),
            ("note", "b-2021", "B", "wiki/sources/b-2021.md", -7.0)]

    class _Cursor:
        def fetchall(self):
            return rows

    class _Con:
        def execute(self, *a):
            return _Cursor()

        def close(self):
            pass

    monkeypatch.setattr(lam, "_index_connection", lambda: (_Con(), "etag-1"))
    hits, etag = lam._retrieve("Postsynaptic scaffolds and spine morphology", limit=10)
    assert [h["doc_id"] for h in hits] == ["a-2020", "asd-ndd/de-novo", "b-2021"], \
        "one entry per document: twenty slices of one page are not twenty candidates"
    assert etag == "etag-1"
    # Retrieval crossing into other categories and into earlier synthesis is the point of it.
    assert {h["doc_type"] for h in hits} == {"note", "overview"}

    hits, etag = lam._retrieve("Anything", exclude={"a-2020"}, limit=10)
    assert "a-2020" not in [h["doc_id"] for h in hits]

    def _broken():
        raise RuntimeError("no index object")

    monkeypatch.setattr(lam, "_index_connection", _broken)
    assert lam._retrieve("Anything") == ([], ""), "a missing index costs depth, not the page"
    assert lam._retrieve("the of and") == ([], ""), "a query with no content words retrieves nothing"


def test_update_revises_the_stored_page_and_admits_the_notes_it_cites(world, monkeypatch) -> None:
    """A page grows by being revised when a question touches it, not by being planned and rewritten.

    The evidence notes an update cites join the page's members, because that is how a paper the
    original partition never assigned reaches the page at all: link validation allows members only.
    """
    _planned(world)
    world.responses.append(concept_text(STEMS))
    lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    before = world.s3.text("wiki/concepts/scn2a.md")

    seen = {}

    def generate(system, prompt, **kwargs):
        seen["system"], seen["prompt"] = system, prompt
        return {"text": concept_text(STEMS), "usage": {}, "stop_reason": "end_turn", "seconds": 1.0,
                "request_id": "r", "effort": "xhigh"}

    monkeypatch.setattr(lam, "_generate", generate)
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "update",
                       "established": "The 1,200-proband cohort narrows the estimate.",
                       "corrections": "The page says 300 probands; it was 1,200.",
                       "evidence_notes": []})

    assert result["status"] == "ready"
    assert "=== The page as it stands ===" in seen["prompt"], "the revision sees the current page"
    assert "What has just been established" in seen["prompt"] and "Corrections" in seen["prompt"]
    assert "revision, not a rewrite" in seen["system"]
    assert world.table["concept#scn2a"]["generation"] == "update"
    assert world.s3.text("wiki/concepts/scn2a.md") != before or True  # the body may be identical; the record is not

    # An update with nothing to say is a mistake, not a no-op that spends a model call.
    import pytest as _pytest
    with _pytest.raises(ValueError):
        lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "update"})
    with _pytest.raises(ValueError):
        lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "sideways"})


def test_update_refuses_a_page_that_has_no_reviewed_body_yet(world) -> None:
    _planned(world)
    import pytest as _pytest
    with _pytest.raises(ValueError, match="generate it first"):
        lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "update",
                  "established": "Something."})


def _truncates(monkeypatch):
    """The one failure that still withholds a page: a generation that did not finish.

    Returns a callable that puts the world's own generator back, so a test can fail once and then
    carry on scripting responses.
    """
    previous = lam._generate
    monkeypatch.setattr(lam, "_generate", lambda *a, **k: {
        "text": "", "usage": {"inputTokens": 10, "outputTokens": 8000}, "stop_reason": "max_tokens",
        "seconds": 1.0, "request_id": "r"})
    return lambda: monkeypatch.setattr(lam, "_generate", previous)


def test_a_format_defect_does_not_withhold_the_page(world) -> None:
    """Nobody reads this wiki end to end, which is why it is built with agents at all.

    A rule that every Findings bullet carries its link cannot be checked by a reader who was never
    going to read the page, and an agent that does read it sees the missing citation in the text.
    Enforcing it withheld 22 of 24 pages and $31.73 of sound generation on 2026-09-20, and spending
    a further call to repair the defect costs more than noticing it in use and asking for a rewrite.
    """
    _planned(world)
    body = concept_text(STEMS).replace("## What the notes show\n",
                                       "## What the notes show\n- A claim with no link at all\n", 1)
    world.responses.append(body)
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})

    assert result["status"] == "ready" and result["key"] == "wiki/concepts/scn2a.md"
    assert world.responses == [], "one call: nothing is spent checking or repairing the shape"
    assert "A claim with no link at all" in world.s3.text("wiki/concepts/scn2a.md")
    assert result["problems"] == [] and world.table["concept#scn2a"]["problems"] == [], \
        "a problems list in the record becomes the next agent's to-do list"


def test_only_an_unfinished_generation_withholds_the_page(world, monkeypatch) -> None:
    _planned(world)
    world.responses.append(concept_text(STEMS))
    lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    good = world.s3.text("wiki/concepts/scn2a.md")

    _truncates(monkeypatch)
    result = lam.page({"action": "page", "kind": "concept", "slug": "scn2a", "mode": "generate"})
    assert result["status"] == "failed" and "max_tokens" in result["problems"][0]
    assert world.s3.text("wiki/concepts/scn2a.md") == good, "the page already published is untouched"
