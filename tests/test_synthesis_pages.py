from __future__ import annotations

import pytest

from byeori import promote
from byeori.synthesis_manifest import parse_frontmatter
from byeori.synthesis_pages import (CATEGORY_MIN_CHARS, CATEGORY_SECTIONS, CONCEPT_SECTIONS, SUBTOPIC_SECTIONS,
                                          _first_sentence, category_page, concept_page, parse_json, subtopic_page,
                                          validate_links, validate_structure)

CONCEPT_TEXT = """## Definition
SCN2A encodes Nav1.2, the neuronal voltage-gated sodium channel. Two notes define it this way.

## What the notes show
- Three de novo SCN2A variants clustered near the pore (P = 0.004) [[sources/a-2020-x]].
- Loss-of-function variants associated with autism, gain-of-function with epilepsy [[sources/b-2021-y]] [[sources/a-2020-x]].

## Disagreements and limits
None recorded.
"""
MEMBERS = [{"stem": "a-2020-x", "title": "X", "first_author": "A", "year": "2020", "category": "asd-ndd", "sha256": "1"},
           {"stem": "b-2021-y", "title": "Y", "first_author": "B", "year": "2021", "category": "asd-models", "sha256": "2"}]


def test_validate_structure_accepts_the_expected_sections_in_order() -> None:
    assert validate_structure(CONCEPT_TEXT, CONCEPT_SECTIONS, min_chars=100) == []
    swapped = CONCEPT_TEXT.replace("## Definition", "## Scope")
    assert any("expected" in p for p in validate_structure(swapped, CONCEPT_SECTIONS, min_chars=100))
    assert any("start with" in p for p in validate_structure("intro\n" + CONCEPT_TEXT, CONCEPT_SECTIONS, min_chars=100))
    assert any("scratchpad" in p for p in validate_structure(CONCEPT_TEXT + "\nWait, I need to check.\n", CONCEPT_SECTIONS, min_chars=100))
    assert any("frontmatter" in p for p in validate_structure("---\ntitle: x\n---\n" + CONCEPT_TEXT, CONCEPT_SECTIONS, min_chars=100))
    assert any("too short" in p for p in validate_structure("## Definition\nx\n\n## What the notes show\n\n## Disagreements and limits\n", CONCEPT_SECTIONS))


def test_validate_links_checks_bullets_and_link_targets() -> None:
    allowed = {"a-2020-x", "b-2021-y"}
    assert validate_links(CONCEPT_TEXT, allowed_stems=allowed) == []
    no_link = CONCEPT_TEXT.replace(" [[sources/b-2021-y]] [[sources/a-2020-x]]", "")
    assert any("bullet without" in p for p in validate_links(no_link, allowed_stems=allowed))
    foreign = CONCEPT_TEXT.replace("b-2021-y", "z-1999-q")
    assert validate_links(foreign, allowed_stems=allowed) == ["link to a note outside the members: z-1999-q"]
    concept_link = CONCEPT_TEXT + "\nSee [[concepts/chd8]].\n"
    assert any("kind the model may not write" in p for p in validate_links(concept_link, allowed_stems=allowed))
    landscape = "## Landscape\nAs [[overviews/asd-ndd/de-novo]] shows.\n\n## Open questions\n- Why? [[overviews/asd-ndd/de-novo]]\n"
    assert validate_links(landscape, allowed_stems=set(), allowed_pages={"asd-ndd/de-novo"}, linked_sections=set()) == []
    assert validate_links(landscape, allowed_stems=set(), allowed_pages=set(), linked_sections=set()) == \
        ["link to a page outside this category: asd-ndd/de-novo", "link to a page outside this category: asd-ndd/de-novo"]


def test_parse_json_strips_fences() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == ({"a": 1}, None)
    assert parse_json("not json")[0] is None and "did not return JSON" in parse_json("not json")[1]


def test_concept_page_assembly() -> None:
    concept = {"slug": "scn2a", "title": "SCN2A", "aliases": ["Nav1.2"], "entity_type": "gene"}
    page = concept_page(CONCEPT_TEXT, concept=concept, members=MEMBERS,
                        mentions=[{"stem": "c-2022-z", "title": "Z", "first_author": "C", "year": "2022"}],
                        related=[("chd8", 2)], model_id="global.anthropic.claude-opus-5", reasoning="xhigh",
                        generation="single", manifest_ref="runs/synthesis/concepts/candidates.json@abc", created=None, today="2026-09-21")
    fields, body = parse_frontmatter(page)
    assert fields["kind"] == "concept" and fields["slug"] == "scn2a" and fields["aliases"] == ["Nav1.2"]
    assert fields["categories"] == ["asd-models", "asd-ndd"] and fields["note_count"] == 2
    assert fields["source_notes"] == [{"stem": "a-2020-x", "sha256": "1"}, {"stem": "b-2021-y", "sha256": "2"}]
    assert fields["ingest_model"] == "opus" and fields["ingest_model_version"] == "5" and fields["ingest_reasoning"] == "xhigh"
    assert fields["created"] == "2026-09-21" and fields["updated"] == "2026-09-21"
    assert body.startswith("## Definition")
    assert "## Related concepts\n- [[concepts/chd8]] (2 shared notes)" in body
    assert "## Notes\n- [[sources/a-2020-x]] A (2020). X\n- [[sources/b-2021-y]] B (2021). Y" in body
    assert "Also mentioned in the body of:\n- [[sources/c-2022-z]] C (2022). Z" in body
    assert [h for h in body.splitlines() if h.startswith("## ")] == \
        ["## Definition", "## What the notes show", "## Disagreements and limits", "## Related concepts", "## Notes"]


def test_subtopic_and_category_page_assembly() -> None:
    text = ("## Scope\nTwo papers on de novo variants (A 2020; B 2021).\n\n## Findings\n- Finding one [[sources/a-2020-x]].\n\n"
            "## Comparison\nNot applicable.\n\n## Open questions\n- Is it recurrent? (A 2020)\n")
    page = subtopic_page(text, category="asd-ndd", subtopic={"slug": "de-novo-variants", "title": "De novo variants", "scope": "s"},
                         members=MEMBERS, concepts=[("scn2a", 2)], model_id="global.anthropic.claude-opus-5", reasoning="xhigh",
                         generation="hierarchical", manifest_ref="runs/synthesis/asd-ndd/subtopics.json", created="2026-09-20", today="2026-09-21")
    fields, body = parse_frontmatter(page)
    assert fields["kind"] == "subtopic" and fields["category"] == "asd-ndd" and fields["created"] == "2026-09-20"
    assert "## Concepts\n- [[concepts/scn2a]] (2 notes)" in body and "## Notes\n- [[sources/a-2020-x]]" in body
    assert [h for h in body.splitlines() if h.startswith("## ")] == \
        ["## Scope", "## Findings", "## Comparison", "## Open questions", "## Concepts", "## Notes"]
    landscape = "## Landscape\nThe field rests on [[overviews/asd-ndd/de-novo-variants]].\n\n## Open questions\n- Recurrence [[overviews/asd-ndd/de-novo-variants]]\n"
    cat = category_page(landscape, category="asd-ndd",
                        subtopics=[{"slug": "de-novo-variants", "title": "De novo variants", "scope": "Two papers. More.", "note_count": 2, "sha256": "p1"}],
                        key_concepts=[("scn2a", 5)], coverage={"note_count": 2, "year_min": "2020", "year_max": "2021", "generated_at": "2026-09-21T00:00:00+00:00"},
                        model_id="global.anthropic.claude-opus-5", reasoning="xhigh", manifest_ref="runs/synthesis/asd-ndd/subtopics.json",
                        created=None, today="2026-09-21")
    fields, body = parse_frontmatter(cat)
    assert fields["kind"] == "category" and fields["source_pages"] == [{"slug": "de-novo-variants", "sha256": "p1"}]
    assert fields["generation"] == "single"
    assert [h for h in body.splitlines() if h.startswith("## ")] == ["## Landscape", "## Subtopics", "## Key concepts", "## Open questions", "## Coverage"]
    assert "- [[overviews/asd-ndd/de-novo-variants]] De novo variants (2 notes): Two papers." in body
    assert "- [[concepts/scn2a]] (5 notes)" in body and "Notes: 2" in body and "Years: 2020-2021" in body


def test_validate_links_accepts_an_aliased_link_but_wants_the_link_last() -> None:
    allowed = {"a-2020-x"}
    aliased = "## What the notes show\n- Finding one [[sources/a-2020-x|A 2020]].\n"
    assert validate_links(aliased, allowed_stems=allowed) == []
    trailing_note = "## What the notes show\n- Finding one [[sources/a-2020-x]] (Table 2).\n"
    assert any("bullet without" in p for p in validate_links(trailing_note, allowed_stems=allowed))


def test_forbidden_pattern_does_not_flag_ordinary_prose() -> None:
    caught = validate_structure(CONCEPT_TEXT + "\nWait, I need to write the wiki page.\n", CONCEPT_SECTIONS, min_chars=100)
    assert any("scratchpad" in p for p in caught)
    caught = validate_structure(CONCEPT_TEXT + "\nLet me collect key numbers.\n", CONCEPT_SECTIONS, min_chars=100)
    assert any("scratchpad" in p for p in caught)
    clean = validate_structure(CONCEPT_TEXT + "\nWait times in the clinic averaged 9 months.\n", CONCEPT_SECTIONS, min_chars=100)
    assert not any("scratchpad" in p for p in clean)


def test_first_sentence_does_not_split_at_an_abbreviation() -> None:
    assert _first_sentence("Papers on rare variants, e.g. de novo SNVs, in large cohorts. More follows.") == \
        "Papers on rare variants, e.g. de novo SNVs, in large cohorts."
    assert _first_sentence("Work following Sanders et al. 2015 on trios. Next.") == \
        "Work following Sanders et al. 2015 on trios."


def test_validate_links_accepts_wrapped_and_nested_bullets() -> None:
    allowed = {"a-2020-x", "b-2021-y"}
    wrapped = "## Findings\n- Finding one continues onto\n  a second line [[sources/a-2020-x]].\n"
    assert validate_links(wrapped, allowed_stems=allowed) == []
    header = ("## Findings\n- Variant classes:\n  - Missense [[sources/a-2020-x]].\n"
             "  - Truncating [[sources/b-2021-y]].\n")
    assert validate_links(header, allowed_stems=allowed) == []
    numbered = "## Findings\n1. Finding without a link.\n"
    assert any("bullet without" in p for p in validate_links(numbered, allowed_stems=allowed))


def test_validate_links_accepts_level_3_subsection_headings() -> None:
    allowed = {"a-2020-x", "b-2021-y"}
    notes_show = ("## What the notes show\n### Missense variants\n- Finding one [[sources/a-2020-x]].\n"
                 "### Truncating variants\n- Finding two [[sources/b-2021-y]].\n")
    assert validate_links(notes_show, allowed_stems=allowed) == []
    findings = ("## Findings\n### Cohort A\n- Finding one [[sources/a-2020-x]].\n"
               "### Cohort B\n- Finding two [[sources/b-2021-y]].\n")
    assert validate_links(findings, allowed_stems=allowed) == []


def test_validate_links_still_requires_a_link_under_a_level_3_heading() -> None:
    allowed = {"a-2020-x"}
    text = "## Findings\n### Cohort A\n- Finding without a link.\n"
    assert any("bullet without" in p for p in validate_links(text, allowed_stems=allowed))


def test_validate_links_reports_malformed_links() -> None:
    assert validate_links("See [[Sources/z-1999-q]] for detail.", allowed_stems=set()) == \
        ["malformed link: [[Sources/z-1999-q]]"]
    assert validate_links("See [[scn2a]] for detail.", allowed_stems=set()) == ["malformed link: [[scn2a]]"]


def test_parse_json_finds_json_amid_prose() -> None:
    assert parse_json('Here is the JSON:\n```json\n{"a": 1}\n```\nLet me know.') == ({"a": 1}, None)
    assert parse_json('{"a": 1}\nThat is the partition.') == ({"a": 1}, None)
    assert parse_json("not json")[0] is None and "did not return JSON" in parse_json("not json")[1]


def test_validate_structure_gates_the_category_prompts_length() -> None:
    short_category = "## Landscape\n" + ("Short. " * 60) + "\n\n## Open questions\n- Q?\n"
    assert len(short_category) < CATEGORY_MIN_CHARS
    assert any("too short" in p for p in validate_structure(short_category, CATEGORY_SECTIONS))


def test_category_page_rejects_model_text_without_a_landscape_section() -> None:
    with pytest.raises(ValueError, match=r"model text lacks ## Landscape"):
        category_page("## Open questions\nq\n", category="asd-ndd", subtopics=[], key_concepts=[],
                      coverage={"note_count": 0, "year_min": "", "year_max": "", "generated_at": "x"},
                      model_id="global.anthropic.claude-opus-5", reasoning="xhigh",
                      manifest_ref="m", created=None, today="2026-09-21")


def test_validate_links_checks_the_comparison_table_for_note_links() -> None:
    allowed = {"a-2020-x"}
    good = "## Comparison\n| Study | Result |\n| --- | --- |\n| A 2020 | x [[sources/a-2020-x]] |\n"
    assert validate_links(good, allowed_stems=allowed) == []
    not_applicable = "## Comparison\nNot applicable.\n"
    assert validate_links(not_applicable, allowed_stems=allowed) == []
    bad = "## Comparison\n| Study | Result |\n| --- | --- |\n| A 2020 | x |\n"
    assert any("Comparison: table rows" in p for p in validate_links(bad, allowed_stems=allowed))


def test_forbidden_pattern_only_flags_a_real_placeholder() -> None:
    clean = validate_structure(CONCEPT_TEXT + "\nThe [insert size] was 350 bp.\n", CONCEPT_SECTIONS, min_chars=100)
    assert not any("placeholder" in p for p in clean)
    caught = validate_structure(CONCEPT_TEXT + "\n[insert citation here]\n", CONCEPT_SECTIONS, min_chars=100)
    assert any("placeholder" in p for p in caught)


def test_concept_page_requires_sha256_on_every_member() -> None:
    concept = {"slug": "scn2a", "title": "SCN2A"}
    bad_members = [{"stem": "a-2020-x", "title": "X", "first_author": "A", "year": "2020", "category": "asd-ndd"}]
    with pytest.raises(ValueError, match="member a-2020-x has no sha256"):
        concept_page(CONCEPT_TEXT, concept=concept, members=bad_members, mentions=[], related=[],
                    model_id="global.anthropic.claude-opus-5", reasoning="xhigh", generation="single",
                    manifest_ref="m", created=None, today="2026-09-21")


def test_concept_page_round_trips_through_the_search_indexs_parser() -> None:
    concept = {"slug": "scn2a", "title": "SCN2A", "aliases": ["Nav1.2"], "entity_type": "gene"}
    page = concept_page(CONCEPT_TEXT, concept=concept, members=MEMBERS, mentions=[], related=[],
                        model_id="global.anthropic.claude-opus-5", reasoning="xhigh", generation="single",
                        manifest_ref="runs/synthesis/concepts/candidates.json@abc", created=None, today="2026-09-21")
    lines, _body = promote.split_page(page)
    fields = promote.parse_frontmatter(lines)
    assert fields["source_notes"] == [{"stem": "a-2020-x", "sha256": "1"}, {"stem": "b-2021-y", "sha256": "2"}]
    assert fields["note_count"] == 2
