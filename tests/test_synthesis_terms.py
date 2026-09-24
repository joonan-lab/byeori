from __future__ import annotations

import pytest

from byeori.synthesis_terms import (Candidate, HgncTable, MentionIndex, apply_title_overrides,
                                          co_occurrence, count_candidates, flatten_merges, glossary_entries,
                                          merge_candidates, normalise, note_terms, safe_model_merges, slugify, split_aliases)

NOTE = """---
title: "SCN2A variants in autism"
category: "asd-ndd"
---
## One-line Summary
x

## 4. Key Results and Benchmarks
- Three variants ## P < 0.001 in Table 2.

## 7. Glossary

- **SCN2A**: Gene encoding Nav1.2, the neuronal sodium channel.
- **PRS (polygenic risk score)**: Weighted sum of risk alleles.
- **HbF / F-cells**: Fetal hemoglobin; F-cells carry it.
- **De novo variants**: Variants absent from both parents.
"""
HGNC = "hgnc_id\tsymbol\tname\talias_symbol\tprev_symbol\nHGNC:1\tSCN2A\tsodium channel\tNav1.2|SCN2A1\tSCN2A2\nHGNC:2\tCHD8\tchromodomain\t\t\n"


def test_glossary_entries_reads_bold_terms_and_definitions() -> None:
    entries = glossary_entries(NOTE)
    assert [t for t, _ in entries] == ["SCN2A", "PRS (polygenic risk score)", "HbF / F-cells", "De novo variants"]
    assert entries[0][1].startswith("Gene encoding Nav1.2")


def test_split_aliases() -> None:
    assert split_aliases("HbF / F-cells") == ["HbF", "F-cells"]
    assert split_aliases("PRS (polygenic risk score)") == ["PRS", "polygenic risk score"]
    assert split_aliases("SCN2A") == ["SCN2A"]
    assert split_aliases("mg/kg") == ["mg/kg"]
    assert split_aliases("MAPK/ERK pathway") == ["MAPK/ERK pathway"]
    assert split_aliases("β0/β0 versus β+/β+ genotype") == ["β0/β0"]


def test_normalise_folds_case_plurals_greek_and_punctuation() -> None:
    assert normalise("De novo variants") == "de novo variant"
    assert normalise("Nrxn1α") == "nrxn1a"
    assert normalise("T‑cells") == "t-cell"
    assert normalise("analysis") == "analysis"  # -is is not a plural
    assert normalise("NRXN1Α") == "nrxn1a"  # capital Greek alpha
    assert normalise("θ-burst") == "th-burst"
    assert normalise("SNPs") == "snp"
    assert normalise("CNVs") == "cnv"
    assert normalise("rare CNVs") == "rare cnv"
    assert normalise("de novo SNVs") == "de novo snv"
    assert normalise("µM concentration") == normalise("μM concentration") == "mm concentration"


def test_hgnc_resolves_symbols_aliases_and_previous_symbols() -> None:
    table = HgncTable.from_tsv(HGNC)
    assert table.resolve("SCN2A") == "SCN2A"
    assert table.resolve("scn2a") == "SCN2A"
    assert table.resolve("Nav1.2") == "SCN2A"
    assert table.resolve("SCN2A2") == "SCN2A"
    assert table.resolve("SCN2A gene") == "SCN2A"
    assert table.resolve("polygenic risk score") is None
    assert table.resolve("cat") is None and table.resolve("Impact") is None
    assert table.resolve("pten") is None  # digit-free and not written in caps; merged by hand


def test_hgnc_from_tsv_requires_symbol_column() -> None:
    with pytest.raises(ValueError):
        HgncTable.from_tsv("hgnc_id\tname\nHGNC:1\tsodium channel\n")


def test_hgnc_resolve_ignores_alias_claimed_by_two_genes() -> None:
    tsv = ("hgnc_id\tsymbol\tname\talias_symbol\tprev_symbol\n"
           "HGNC:10\tGENEA\tgene a\tMDS\t\n"
           "HGNC:11\tGENEB\tgene b\tMDS\t\n")
    table = HgncTable.from_tsv(tsv)
    assert table.resolve("MDS") is None


def test_note_terms_folds_each_glossary_line_onto_one_key() -> None:
    terms = note_terms("smith-2020-x", "asd-ndd", NOTE, HgncTable.from_tsv(HGNC))
    assert set(terms.kinds) == {"scn2a", "polygenic-risk-score", "hbf", "f-cell", "de-novo-variant"}
    assert terms.kinds["scn2a"] == "gene"
    assert terms.aliases["polygenic-risk-score"] == {"PRS", "polygenic risk score"}
    assert terms.aliases["hbf"] == {"HbF"} and terms.aliases["f-cell"] == {"F-cells"}
    assert terms.definitions["scn2a"].startswith("Gene encoding")


def test_grouped_distinct_genes_never_become_each_others_aliases():
    hgnc = HgncTable.from_tsv(HGNC + "HGNC:3\tFMR1\tfragile X messenger ribonucleoprotein 1\tFMRP\t\n")
    text = "## 7. Glossary\n- **CHD8 / FMR1**: Genes with targets in the network.\n- **CHD8 (FMR1)**: Different genes.\n- **SCN2A / Nav1.2**: SCN2A encodes this channel.\n"
    terms = note_terms("one", "asd-models", text, hgnc)
    assert terms.kinds == {"chd8": "gene", "fmr1": "gene", "scn2a": "gene"}
    assert terms.aliases["chd8"] == {"CHD8"}
    assert terms.aliases["fmr1"] == {"FMR1"}
    assert terms.aliases["scn2a"] == {"SCN2A", "Nav1.2"}


def test_numbered_gene_alias_remains_canonical_without_definition_repeating_symbol():
    terms = note_terms("one", "asd-models", "## 7. Glossary\n- **Nav1.2**: Neuronal sodium channel.\n",
                       HgncTable.from_tsv(HGNC))
    assert terms.kinds == {"scn2a": "gene"}


def test_expanded_acronyms_do_not_become_gene_aliases_or_each_other():
    hgnc = HgncTable.from_tsv(HGNC + "HGNC:4\tWDR20\tWD repeat domain 20\tDMR\t\n")
    text = ("## 7. Glossary\n- **DMR (differentially methylated region)**: A genomic interval.\n"
            "- **DMR (Development and Maintenance of Relationships)**: A behavioral subdomain.\n"
            "- **DMR**: A contiguous genomic interval with coordinated methylation differences.\n"
            "- **WDR20 (WD repeat domain 20)**: A protein.\n")
    terms = note_terms("one", "asd-models", text, hgnc)
    assert terms.kinds["differentially-methylated-region"] == "term"
    assert terms.kinds["development-and-maintenance-of-relationship"] == "term"
    assert terms.kinds["dmr"] == "term"
    assert terms.aliases["wdr20"] == {"WDR20", "WD repeat domain 20"}


def test_automatic_merges_require_identity_not_shared_topic_or_short_acronym():
    def candidate(slug, title, kind="term", aliases=()):
        return Candidate(slug, title, kind, list(aliases), 1, 1, [slug], [])
    rows = [candidate("chd8", "CHD8", "gene"), candidate("fmr1", "FMR1", "gene"),
            candidate("fxs", "Fragile X syndrome"), candidate("wgb", "whole genome bisulfite sequencing"),
            candidate("dmr", "differentially methylated region", aliases=["DMR"]),
            candidate("relationship", "Development and Maintenance of Relationships", aliases=["DMR"]),
            candidate("prs", "PRS", aliases=["polygenic risk score"]),
            candidate("polygenic-risk-score", "polygenic risk score")]
    accepted, rejected = safe_model_merges(rows, {"chd8": "fmr1", "fxs": "fmr1", "wgb": "dmr",
                                                 "relationship": "dmr", "prs": "polygenic-risk-score"})
    assert accepted == {"prs": "polygenic-risk-score"}
    assert len(rejected) == 4


def test_note_terms_drops_stop_terms() -> None:
    text = """## 7. Glossary

- **Autism spectrum disorder (ASD)**: definition here.
- **Odds ratio**: definition too.
"""
    terms = note_terms("s1", "asd-ndd", text, None)
    assert terms.kinds == {}


def test_slugify() -> None:
    assert slugify("SCN2A") == "scn2a" and slugify("de novo variant") == "de-novo-variant"
    assert slugify("Nav1.2 / channel") == "nav1-2-channel"


def _note(stem, category, terms):
    from byeori.synthesis_terms import NoteTerms
    n = NoteTerms(stem=stem, category=category)
    for slug, kind, title in terms:
        n.kinds[slug] = kind
        n.titles[slug] = title
        n.aliases[slug] = {title}
        n.definitions[slug] = f"{title} defined in {stem}"
    return n


def test_count_candidates_applies_threshold_and_scope() -> None:
    notes = [_note(f"n{i}", "asd-ndd", [("scn2a", "gene", "SCN2A")]) for i in range(5)]
    notes += [_note("m1", "liver", [("scn2a", "gene", "SCN2A"), ("nav1-2", "gene", "Nav1.2")])]
    out = count_candidates(notes, scope={"asd-ndd"}, threshold=5, stop_terms=set(), merge={}, exclude=[])
    by_slug = {c.slug: c for c in out}
    assert set(by_slug) == {"scn2a"}
    assert by_slug["scn2a"].count_total == 6 and by_slug["scn2a"].count_in_scope == 5
    assert by_slug["scn2a"].glossary_stems == ["m1", "n0", "n1", "n2", "n3", "n4"]
    below = count_candidates(notes, scope={"asd-ndd"}, threshold=6, stop_terms=set(), merge={}, exclude=[])
    assert below == []


def test_count_candidates_drops_stop_terms() -> None:
    notes = [_note(f"n{i}", "asd-ndd", [("scn2a", "gene", "SCN2A"), ("gwas", "term", "GWAS")]) for i in range(5)]
    with_stop = count_candidates(notes, scope={"asd-ndd"}, threshold=5, stop_terms={"gwas"}, merge={}, exclude=[])
    assert {c.slug for c in with_stop} == {"scn2a"}
    without_stop = count_candidates(notes, scope={"asd-ndd"}, threshold=5, stop_terms=set(), merge={}, exclude=[])
    assert {c.slug for c in without_stop} == {"scn2a", "gwas"}


def test_count_candidates_applies_merge_and_exclude() -> None:
    notes = [_note(f"n{i}", "asd-ndd", [("scn2a", "gene", "SCN2A")]) for i in range(5)]
    notes += [_note(f"p{i}", "asd-ndd", [("polygenic-risk-score", "term", "polygenic risk score")]) for i in range(3)]
    notes += [_note(f"q{i}", "asd-ndd", [("prs", "term", "PRS")]) for i in range(2)]
    out = count_candidates(notes, scope={"asd-ndd"}, threshold=5, stop_terms=set(),
                           merge={"prs": "polygenic-risk-score"}, exclude=[])
    by_slug = {c.slug: c for c in out}
    assert set(by_slug) == {"scn2a", "polygenic-risk-score"}
    assert by_slug["polygenic-risk-score"].count_in_scope == 5
    assert set(by_slug["polygenic-risk-score"].aliases) == {"PRS", "polygenic risk score"}
    assert by_slug["polygenic-risk-score"].samples[0].endswith("defined in p0")
    without_scn2a = count_candidates(notes, scope={"asd-ndd"}, threshold=5, stop_terms=set(),
                                     merge={"prs": "polygenic-risk-score"}, exclude=["scn2a"])
    assert [c.slug for c in without_scn2a] == ["polygenic-risk-score"]


def test_count_candidates_counts_a_note_once_even_when_two_glossary_lines_merge() -> None:
    from byeori.synthesis_terms import NoteTerms
    note = NoteTerms(stem="only-note", category="asd-ndd")
    for slug, title in (("prs", "PRS"), ("polygenic-risk-score", "polygenic risk score")):
        note.kinds[slug], note.titles[slug], note.aliases[slug], note.definitions[slug] = "term", title, {title}, "d"
    out = count_candidates([note], scope={"asd-ndd"}, threshold=2, stop_terms=set(),
                           merge={"prs": "polygenic-risk-score"}, exclude=[])
    assert out == [], "one note is one note, however many glossary lines name the concept"
    out = count_candidates([note], scope={"asd-ndd"}, threshold=1, stop_terms=set(),
                           merge={"prs": "polygenic-risk-score"}, exclude=[])
    assert out[0].count_total == 1 and out[0].count_in_scope == 1 and out[0].glossary_stems == ["only-note"]
    assert set(out[0].aliases) == {"PRS", "polygenic risk score"}


def test_count_candidates_follows_transitive_merge_to_terminal_slug() -> None:
    notes = [_note("n1", "asd-ndd", [("a", "term", "A")]), _note("n2", "asd-ndd", [("c", "term", "C")])]
    out = count_candidates(notes, scope={"asd-ndd"}, threshold=1, stop_terms=set(),
                           merge={"a": "b", "b": "c"}, exclude=[])
    assert [c.slug for c in out] == ["c"]
    assert out[0].glossary_stems == ["n1", "n2"]


def test_count_candidates_title_is_the_most_frequent_surface_form() -> None:
    notes = [_note("n1", "asd-ndd", [("polygenic-risk-score", "term", "PRS")]),
             _note("n2", "asd-ndd", [("polygenic-risk-score", "term", "PRS")]),
             _note("n3", "asd-ndd", [("polygenic-risk-score", "term", "polygenic risk score")])]
    out = count_candidates(notes, scope={"asd-ndd"}, threshold=1, stop_terms=set(), merge={}, exclude=[])
    assert out[0].title == "PRS"
    reversed_out = count_candidates(list(reversed(notes)), scope={"asd-ndd"}, threshold=1, stop_terms=set(),
                                    merge={}, exclude=[])
    assert reversed_out[0].title == "PRS"


def test_flatten_merges_follows_chains_and_guards_cycles() -> None:
    assert flatten_merges({"a": "b", "b": "c"}) == {"a": "c", "b": "c"}
    assert flatten_merges({"x": "y", "y": "x"}) == {}


def test_merge_candidates_folds_members_and_recounts_scope() -> None:
    a = Candidate(slug="prs", title="PRS", kind="term", aliases=["PRS"], count_total=2, count_in_scope=2,
                  glossary_stems=["q0", "q1"], samples=[])
    b = Candidate(slug="polygenic-risk-score", title="polygenic risk score", kind="term", aliases=["polygenic risk score"],
                  count_total=3, count_in_scope=3, glossary_stems=["p0", "p1", "p2"], samples=[])
    merged = merge_candidates([a, b], {"prs": "polygenic-risk-score"}, in_scope=lambda stem: stem != "p2")
    assert [c.slug for c in merged] == ["polygenic-risk-score"]
    assert merged[0].glossary_stems == ["p0", "p1", "p2", "q0", "q1"] and merged[0].count_in_scope == 4
    assert "PRS" in merged[0].aliases


def test_merge_candidates_follows_transitive_merges_regardless_of_order() -> None:
    a = Candidate(slug="a", title="A", kind="term", aliases=["A"], count_total=1, count_in_scope=1,
                  glossary_stems=["s1"], samples=[])
    b = Candidate(slug="b", title="B", kind="term", aliases=["B"], count_total=1, count_in_scope=1,
                  glossary_stems=["s2"], samples=[])
    c = Candidate(slug="c", title="C", kind="term", aliases=["C"], count_total=1, count_in_scope=1,
                  glossary_stems=["s3"], samples=[])
    merged = merge_candidates([a, b, c], {"b": "c", "a": "b"}, in_scope=lambda stem: True)
    assert [x.slug for x in merged] == ["c"]
    assert merged[0].glossary_stems == ["s1", "s2", "s3"]


def test_mention_index_matches_whole_words_case_sensitively_for_genes() -> None:
    c = Candidate(slug="scn2a", title="SCN2A", kind="gene", aliases=["Nav1.2"], count_total=1, count_in_scope=1,
                  glossary_stems=["a"], samples=[])
    t = Candidate(slug="de-novo-variant", title="de novo variant", kind="term", aliases=["De novo variants"],
                  count_total=1, count_in_scope=1, glossary_stems=["a"], samples=[])
    index = MentionIndex([c, t])
    bodies = {"a": "We sequenced SCN2A in trios.", "b": "Nav1.2 currents fell; scn2a is lowercase here.",
              "c": "The SCN2A1 pseudogene and SCN2A-related epilepsy.", "d": "de novo variants were rare"}
    for stem, body in bodies.items():
        index.add(stem, body)
    assert index.confirm("scn2a", bodies) == ["a", "b", "c"]
    assert index.confirm("de-novo-variant", bodies) == ["d"]
    assert index.confirm("scn2a", {k: v for k, v in bodies.items() if k != "b"}) == ["a", "c"]


def test_mention_index_allows_plural_for_non_gene_terms() -> None:
    c = Candidate(slug="polygenic-risk-score", title="polygenic risk score", kind="term", aliases=[],
                  count_total=1, count_in_scope=1, glossary_stems=["a"], samples=[])
    index = MentionIndex([c])
    bodies = {"a": "polygenic risk scores were computed for each cohort."}
    index.add("a", bodies["a"])
    assert index.confirm("polygenic-risk-score", bodies) == ["a"]


def test_mention_index_matches_short_upper_case_alias_exactly() -> None:
    c = Candidate(slug="nitric-oxide", title="nitric oxide", kind="term", aliases=["NO"],
                  count_total=1, count_in_scope=1, glossary_stems=["x"], samples=[])
    index = MentionIndex([c])
    bodies = {"x": "There was no effect on behaviour.", "y": "Nitric oxide signalling rose.",
              "z": "NO donors were applied."}
    for stem, body in bodies.items():
        index.add(stem, body)
    assert index.confirm("nitric-oxide", bodies) == ["y", "z"]


def test_co_occurrence_ranks_shared_notes() -> None:
    members = {"scn2a": {"a", "b", "c"}, "chd8": {"b", "c", "d"}, "prs": {"c"}, "far": {"z"}}
    assert co_occurrence(members, "scn2a") == [("chd8", 2), ("prs", 1)]


def test_apply_title_overrides() -> None:
    c = Candidate(slug="prs", title="PRS", kind="term", aliases=[], count_total=1, count_in_scope=1, glossary_stems=[], samples=[])
    assert apply_title_overrides([c], {"prs": "Polygenic risk score"})[0].title == "Polygenic risk score"


def test_glossary_to_mentions_end_to_end() -> None:
    note_a_text = """---
title: "SCN2A variants in autism"
category: "asd-ndd"
---
## 7. Glossary

- **SCN2A**: Gene encoding Nav1.2, the neuronal sodium channel.
- **De novo variants**: Variants absent from both parents.
"""
    note_b_text = """---
title: "SNP burden analysis"
category: "asd-ndd"
---
## 7. Glossary

- **SNPs**: Single nucleotide polymorphisms.
- **de novo variant**: A variant that newly arose in the child.
"""
    hgnc = HgncTable.from_tsv(HGNC)
    note_a = note_terms("note-a", "asd-ndd", note_a_text, hgnc)
    note_b = note_terms("note-b", "asd-ndd", note_b_text, hgnc)
    candidates = count_candidates([note_a, note_b], scope=None, threshold=1, stop_terms=set(), merge={}, exclude=[])
    assert {c.slug for c in candidates} == {"scn2a", "de-novo-variant", "snp"}
    index = MentionIndex(candidates)
    bodies = {"note-a": "We studied SCN2A carriers and de novo variants in trios.",
              "note-b": "SNP burden was high; de novo variant counts were similar."}
    for stem, body in bodies.items():
        index.add(stem, body)
    assert index.confirm("de-novo-variant", bodies) == ["note-a", "note-b"]
