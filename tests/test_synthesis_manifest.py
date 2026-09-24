from __future__ import annotations

import pytest

from byeori.synthesis_manifest import (SCOPES, first_author, member_digest, note_metadata, scope_categories,
                                             stale_mode, synthesis_input, validate_partition)

NOTE = """---
title: "Rate of de novo mutations"
authors: "Kong A, Frigge ML, Masson G"
year: "2012"
category: "germline-mutation"
---
## One-line Summary
78 trios gave 1.20e-8 per base per generation.

## 1. Document Information

| Field | Details |
|---|---|
| Title | Rate |

## 2. Key Contributions
- Paternal age adds 2.01 mutations per year.

## 3. Methodology and Architecture
- WGS of 78 trios.

## 4. Key Results and Benchmarks
- 1.20e-8.

## 5. Limitations and Future Work
- Few fathers over 40.

## 6. Related Work
- Builds on Lynch (2010).

## 7. Glossary
- **DNM**: de novo mutation.
"""


def test_scope_categories() -> None:
    assert scope_categories("all") is None
    assert scope_categories("autism") == SCOPES["autism"] and "asd-ndd" in SCOPES["autism"]
    with pytest.raises(ValueError):
        scope_categories("nope")


def test_first_author_handles_initials_and_full_names() -> None:
    assert first_author("Kong A, Frigge ML") == "Kong"
    assert first_author("Bin Fu") == "Fu"
    assert first_author("Jane Q. Public and Someone Else") == "Public"
    assert first_author("") == "Unknown"
    assert first_author("Ahmed Abdelhak et al.") == "Abdelhak"
    assert first_author("Abelson et al.") == "Abelson"
    assert first_author("A Abromeit 1, B Other") == "Abromeit"
    assert first_author("van der Meer J") == "van der Meer"
    assert first_author("Smith Jr") == "Smith"
    assert first_author("Silvia De Rubeis, Xin He") == "De Rubeis"


def test_note_metadata_and_synthesis_input() -> None:
    meta = note_metadata("kong-2012-rate", NOTE)
    assert meta == {"stem": "kong-2012-rate", "title": "Rate of de novo mutations", "first_author": "Kong", "year": "2012",
                    "category": "germline-mutation", "summary": "78 trios gave 1.20e-8 per base per generation."}
    text = synthesis_input(meta, NOTE)
    assert text.startswith("=== Note kong-2012-rate | Kong (2012). Rate of de novo mutations ===\n")
    assert "## 2. Key Contributions" in text and "## 4. Key Results" in text and "## 5. Limitations" in text
    assert "Document Information" not in text and "Related Work" not in text and "Glossary" not in text


def test_stale_mode() -> None:
    current = {f"s{i}": "h" for i in range(20)}
    assert stale_mode(None, current) == "generate"
    assert stale_mode(dict(current), current) == "skip"
    small = dict(current); small["s0"] = "changed"
    assert stale_mode(small, current) == "refresh"
    five = dict(current); five.update({f"x{i}": "h" for i in range(5)})
    assert stale_mode(five, current) == "generate"
    assert stale_mode({"s0": "h", "s1": "h"}, {"s0": "h", "s1": "h", "s2": "h"}) == "generate"  # 1 of 3 is over 20%


def test_parse_frontmatter_drops_the_blank_line_after_the_fence() -> None:
    from byeori.synthesis_manifest import parse_frontmatter
    fields, body = parse_frontmatter('---\ntitle: "T"\nwork_ids: ["W1"]\nnote: "a: b"\n---\n\n## Definition\nx\n')
    assert fields == {"title": "T", "work_ids": ["W1"], "note": "a: b"} and body == "## Definition\nx\n"
    assert parse_frontmatter("no frontmatter") == ({}, "no frontmatter")


def test_member_digest_is_order_independent() -> None:
    assert member_digest({"a": "1", "b": "2"}) == member_digest({"b": "2", "a": "1"})
    assert member_digest({"a": "1"}) != member_digest({"a": "2"})


def test_validate_partition_reports_every_defect() -> None:
    stems = [f"s{i}" for i in range(12)]
    bad = {"subtopics": [{"slug": "Bad Slug", "title": "", "scope": "", "stems": ["s0", "zz"]},
                         {"slug": "a", "title": "A", "scope": "x", "stems": ["s0", "s1", "s2", "s3", "s4"]},
                         {"slug": "a", "title": "A", "scope": "x", "stems": []}]}
    problems = validate_partition(bad, stems, category="asd-ndd")
    assert any("bad slug" in p for p in problems) and any("duplicate subtopic a" in p for p in problems)
    assert any("s0 assigned to both" in p for p in problems) and any("unknown stem zz" in p for p in problems)
    assert any("unassigned" in p for p in problems) and any("2 subtopics, need 4 to 12" in p for p in problems)
    assert validate_partition({"nope": 1}, stems, category="asd-ndd") == ["partition must be an object with a subtopics list"]
    ok = {"subtopics": [{"slug": f"t{j}", "title": "T", "scope": "s", "stems": stems[j * 3:(j + 1) * 3]} for j in range(4)]}
    assert validate_partition(ok, stems, category="asd-ndd", min_notes=3) == []
    assert any("t0: 3 notes, fewer than 5" in p for p in validate_partition(ok, stems, category="asd-ndd"))
    ok["subtopics"].append({"slug": "asd-ndd-other", "title": "Other", "scope": "rest", "stems": []})
    assert validate_partition(ok, stems, category="asd-ndd", min_notes=3) == []  # an empty -other bucket is allowed
    scalar_stems = {"subtopics": [{"slug": "t0", "title": "T", "scope": "s", "stems": 5}]}
    problems = validate_partition(scalar_stems, stems, category="asd-ndd")
    assert any("stems must be a list" in p for p in problems)
    string_stems = {"subtopics": [{"slug": "t0", "title": "T", "scope": "s", "stems": "s0,s1"}]}
    problems = validate_partition(string_stems, stems, category="asd-ndd")
    assert any("stems must be a list" in p for p in problems)
    digit_slug = {"subtopics": [{"slug": "123", "title": "T", "scope": "s", "stems": []}]}
    assert any("bad slug" in p for p in validate_partition(digit_slug, stems, category="asd-ndd"))


def test_validate_partition_rejects_reserved_slugs() -> None:
    stems = [f"s{i}" for i in range(6)]
    reserved = {"subtopics": [{"slug": "index", "title": "T", "scope": "s", "stems": stems[:3]},
                             {"slug": "failed", "title": "T", "scope": "s", "stems": stems[3:]}]}
    problems = validate_partition(reserved, stems, category="asd-ndd", min_notes=3)
    assert "index is a reserved slug: name the subtopic after what its papers study" in problems
    assert "failed is a reserved slug: name the subtopic after what its papers study" in problems
