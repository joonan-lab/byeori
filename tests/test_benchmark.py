from __future__ import annotations

from pathlib import Path

from byeori.benchmark import select_llm_wiki_questions


def test_select_llm_wiki_questions_filters_autism(tmp_path: Path) -> None:
    q = tmp_path / "wiki" / "questions"
    q.mkdir(parents=True)
    (q / "index.md").write_text("# index\n")
    (q / "a.md").write_text("---\ntitle: \"Does CHD8 loss change cortical growth?\"\ntags: [CHD8, autism]\n---\n## Question\nx [[asd-ndd/cotney-2015-x]] and [[overviews/y]]\n")
    (q / "b.md").write_text("---\ntitle: \"Is HIF stable in neurons\"\n---\n## Question\nx [[glia/z]]\n")
    rows = select_llm_wiki_questions(tmp_path)
    assert [r["stem"] for r in rows] == ["a"]
    assert rows[0]["title"].endswith("?") and rows[0]["papers"] == ["cotney-2015-x"] and rows[0]["tags"] == ["CHD8", "autism"]
    assert [r["stem"] for r in select_llm_wiki_questions(tmp_path, select="all")] == ["a", "b"]
    assert select_llm_wiki_questions(tmp_path, select="all")[1]["title"] == "Is HIF stable in neurons?"
