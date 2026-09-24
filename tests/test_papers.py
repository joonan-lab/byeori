from __future__ import annotations

from pathlib import Path

from byeori.papers import read_frontmatter, select_papers


def test_read_frontmatter_and_select_papers(tmp_path: Path) -> None:
    (tmp_path / "sources").mkdir()
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "abdi-2023-x.pdf").write_bytes(b"%PDF-1.4 test")
    (tmp_path / "papers" / "zhou-2020-y.pdf").write_bytes(b"%PDF-1.4 test")
    (tmp_path / "sources" / "abdi-2023-x.md").write_text(
        '---\ntitle: "T"\ndoi: "10.1/x"\ncategory: "asd-ndd"\npmid: "1"\nnot_a_field: "z"\n---\n## One-line Summary\nautism\n')
    (tmp_path / "sources" / "zhou-2020-y.md").write_text('---\ntitle: "L"\ncategory: "liver"\n---\n## One-line Summary\nhepatocytes\n')
    (tmp_path / "sources" / "no-pdf-2021-z.md").write_text('---\ntitle: "N"\ncategory: "asd-ndd"\n---\n')
    fields = read_frontmatter(tmp_path / "sources" / "abdi-2023-x.md")
    assert fields["doi"] == "10.1/x" and fields["pmid"] == "1" and "not_a_field" not in fields
    assert [p["stem"] for p in select_papers(tmp_path)] == ["abdi-2023-x"], "autism only, and only with a PDF"
    assert [p["stem"] for p in select_papers(tmp_path, select="all")] == ["abdi-2023-x", "zhou-2020-y"]
    assert [p["stem"] for p in select_papers(tmp_path, stems=["zhou-2020-y"])] == ["zhou-2020-y"]


def test_theses_and_short_reports_are_not_selected(tmp_path: Path) -> None:
    """A thesis extracts past the model's input limit; a short report carries too little."""
    (tmp_path / "sources").mkdir(); (tmp_path / "papers").mkdir()
    for stem in ("tang-2017-thesis", "dennison-2026-report", "abdi-2023-article"):
        (tmp_path / "papers" / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "sources" / "tang-2017-thesis.md").write_text(
        '---\ntitle: "T"\ncategory: "asd-ndd"\ndocument_type: "doctoral-thesis"\n---\n')
    (tmp_path / "sources" / "dennison-2026-report.md").write_text(
        '---\ntitle: "R"\ncategory: "asd-ndd"\ndocument_type: "short-report"\n---\n')
    (tmp_path / "sources" / "abdi-2023-article.md").write_text(
        '---\ntitle: "A"\ncategory: "asd-ndd"\ndocument_type: "article"\n---\n')
    assert [p["stem"] for p in select_papers(tmp_path)] == ["abdi-2023-article"]
    assert [p["stem"] for p in select_papers(tmp_path, stems=["tang-2017-thesis"])] == [], \
        "an explicit stem does not override the document-type rule"


def test_amendment_notices_are_not_selected(tmp_path: Path) -> None:
    """An "Addendum: ..." amends a paper that is already in the corpus; it is not a paper."""
    from byeori.papers import is_amendment_notice
    (tmp_path / "sources").mkdir(); (tmp_path / "papers").mkdir()
    cases = {
        "abramson-2024-addendum-accurate-structure-prediction": ("Addendum: Accurate structure prediction", False),
        "gudmundsson-2021-addendum-mutational-constraint": ("Addendum: The mutational constraint spectrum", False),
        "smith-2020-erratum-something": ("Erratum: Something", False),
        "jones-2021-author-correction-x": ("Author Correction: X", False),
        "abramson-2024-accurate-structure-prediction": ("Accurate structure prediction with AlphaFold 3", True),
        "kim-2023-retraction-of-neurite-outgrowth": ("Retraction of neurite outgrowth by Sema3A", True),
    }
    for stem, (title, keep) in cases.items():
        (tmp_path / "papers" / f"{stem}.pdf").write_bytes(b"%PDF-1.4")
        (tmp_path / "sources" / f"{stem}.md").write_text(f'---\ntitle: "{title}"\ncategory: "asd-ndd"\n---\n')
        assert is_amendment_notice(stem, {"title": title}) is not keep, stem
    kept = {p["stem"] for p in select_papers(tmp_path, select="all")}
    assert kept == {s for s, (_, keep) in cases.items() if keep}, kept
