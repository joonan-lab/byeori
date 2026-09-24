"""Evidence packet selection over the shared BM25 index (docs/LAB-QUESTION-WORKFLOW.md section 5)."""
import sqlite3
import time

import pytest
from botocore.exceptions import ClientError
from lab_fakes import MemoryS3, build_index, index_connection, source_note, wiki_with_index

from byeori import evidence_packet as ep
from byeori.evidence_packet import PacketLimits, build_packet, classify, outline, page_body, strip_managed_blocks

INDEX_KEY = "index/wiki-index-v2.sqlite3"
QUESTION = "Was regional inheritance stable in the cohort?"


def opened(s3):
    return index_connection(s3.objects[INDEX_KEY]), s3.etag(INDEX_KEY)


def kinds(packet, key):
    document = next(d for d in packet["documents"] if d["key"] == key)
    return [s["kind"] for s in document["sections"]]


# ---------------------------------------------------------------------------------------------
# Outline and classification
# ---------------------------------------------------------------------------------------------

def test_outline_of_a_note_with_frontmatter_yields_offsets_and_kinds():
    text = source_note()
    sections = outline(text)
    body = page_body(text)
    assert [s.name for s in sections] == ["(opening)", "Methods", "Results", "Limitations", "Related pages"]
    assert [s.kind for s in sections] == ["other", "methods", "results", "limitations", "links"]
    assert [s.order for s in sections] == [0, 1, 2, 3, 4]
    for section in sections:
        assert body[section.start:section.end] == section.text
        assert section.text == section.text.strip() and section.text
    assert "title:" not in body and "Regional inheritance" in body


def test_korean_headings_classify_like_english():
    text = "# 논문\n\n## 방법\n\n120 가족 코호트.\n\n## 결과\n\n유전이 안정적이었다.\n\n## 해석\n\n의미.\n\n## 한계\n\n표본이 작다.\n"
    assert [(s.name, s.kind) for s in outline(text)] == [
        ("방법", "methods"), ("결과", "results"), ("해석", "interpretation"), ("한계", "limitations")]
    assert classify("Key findings", "x") == "results"
    assert classify("Study design and cohort", "x") == "methods"
    assert classify("Discussion", "x") == "interpretation"
    assert classify("Open questions", "x") == "limitations"
    assert classify("Counter-evidence", "x") == "limitations"
    assert classify("Background", "Plain prose about the field.") == "other"


def test_managed_blocks_are_removed_before_outlining():
    text = source_note() + "\n<!-- byeori:catalog:start -->\n### Catalog\n\n- [[overviews/catalogued]]\n<!-- byeori:catalog:end -->\n"
    stripped = strip_managed_blocks(text)
    assert "byeori:" not in stripped and "new-insight" not in stripped and "catalogued" not in stripped
    sections = outline(text)
    assert all("new-insight" not in s.text and "catalogued" not in s.text for s in sections)
    assert "Linked pages" not in [s.name for s in sections]
    assert "Catalog" not in [s.name for s in sections]


def test_unmanaged_linked_pages_heading_is_a_links_section():
    text = "# Page\n\n## Results\n\nA result sentence with numbers (n = 12).\n\n### Linked pages\n\n- [[concepts/a]]\n- [[concepts/b]]\n"
    sections = outline(text)
    assert [(s.name, s.kind) for s in sections] == [("Results", "results"), ("Linked pages", "links")]
    assert "[[concepts/a]]" not in sections[0].text


def test_section_of_mostly_links_is_links_by_ratio():
    assert classify("See also", "- [[concepts/a]]\n- [[overviews/b|Overview B]]\n- [text](https://example.org)\n") == "links"
    assert classify("Results", "- [[concepts/a]]\n") == "links"
    assert classify("See also", "- [[concepts/a]] explains the mechanism in detail across three cohorts.\n") == "other"


# ---------------------------------------------------------------------------------------------
# Selection and budgets
# ---------------------------------------------------------------------------------------------

def test_selection_keeps_a_limitations_section_even_when_it_ranks_last():
    text = ("# Paper\n\n## Results\n\nRegional inheritance was stable (n = 120).\n\n## Methods\n\nA cohort design.\n\n"
            "## Discussion\n\nInterpretation of stability.\n\n## Limitations\n\nSingle site.\n")
    sections = outline(text)
    limits = PacketLimits(max_sections=2)
    chosen = ep.select_sections(QUESTION, sections, "Results", limits)
    assert [s.name for s in chosen] == ["Results", "Limitations"]
    chosen = ep.select_sections(QUESTION, sections, "Results", PacketLimits(max_sections=3))
    assert [s.name for s in chosen] == ["Results", "Limitations", "Methods"]
    chosen = ep.select_sections(QUESTION, sections, "Results", PacketLimits())
    assert [s.name for s in chosen] == ["Results", "Limitations", "Methods"]


def test_select_sections_honours_remaining_budget_and_opening_best():
    sections = outline(source_note())
    assert ep.select_sections(QUESTION, sections, "", PacketLimits(), max_sections=1)[0].name == "(opening)"
    assert ep.select_sections(QUESTION, sections, "Results", PacketLimits(), max_sections=0) == []


def test_byte_budget_drops_the_lowest_ranked_document():
    long_results = "Regional inheritance was stable in this cohort. " * 40
    pages = {"wiki/sources/paper-one.md": source_note("Paper one", stem="paper-one", results=long_results),
             "wiki/sources/paper-two.md": source_note("Paper two", stem="paper-two", results=long_results + " Regional regional.")}
    s3 = wiki_with_index(pages)
    limits = PacketLimits(total_bytes=len(long_results.encode()) + 400)
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=limits)
    assert len(packet["documents"]) == 1
    read_key = packet["documents"][0]["key"]
    dropped = next(k for k in pages if k != read_key)
    assert {"key": dropped, "reason": "byte_budget"} in packet["omitted"]
    assert packet["total_bytes"] <= limits.total_bytes
    assert s3.writes == []


def long_note(stem: str, *, words: str, methods_chars: int = 9000, results_chars: int = 9000) -> str:
    """A note whose methods and results are both longer than any per-section cut."""
    methods = ((words + " cohort design ") * 2000)[:methods_chars]
    results = ((words + " was stable ") * 2000)[:results_chars]
    return (f"---\ntitle: {stem}\ncategory: asd-ndd\nyear: 2020\njournal: Nature\ndoi: 10.1000/{stem}\n---\n\n"
            f"# {stem}\n\nOne line.\n\n## Methods\n\n{methods}\n\n## Results\n\n{results}\n\n"
            f"## Limitations\n\nThe {words} cohort was small.\n")


def test_a_supporting_section_is_cut_shorter_than_a_decisive_one():
    """Methods is context; results and limitations carry what the answer is judged on."""
    s3 = wiki_with_index({"wiki/sources/paper-one.md": long_note("paper-one", words="regional inheritance")})
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    sections = {s["kind"]: s for s in packet["documents"][0]["sections"]}
    assert len(sections["methods"]["text"]) == PacketLimits().context_chars == 2500
    assert sections["methods"]["truncated"] is True
    assert len(sections["results"]["text"]) == PacketLimits().section_chars == 6000
    assert sections["results"]["truncated"] is True
    assert sections["limitations"]["truncated"] is False


def test_one_document_cannot_spend_the_budget_the_documents_behind_it_need():
    """The 2026-09-22 failure: four requested reads filled the packet and the top hits were dropped."""
    pages = {f"wiki/sources/paper-{n}.md": long_note(f"paper-{n}", words="regional inheritance") for n in range(1, 7)}
    s3 = wiki_with_index(pages)
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits(),
                          extra_reads=["wiki/sources/paper-1.md", "wiki/sources/paper-2.md"])

    read_keys = [d["key"] for d in packet["documents"]]
    assert read_keys[:2] == ["wiki/sources/paper-1.md", "wiki/sources/paper-2.md"]   # requested reads still lead
    assert len(read_keys) == len(pages)                                              # and nothing is starved
    assert not [o for o in packet["omitted"] if o["reason"] == "byte_budget"]
    assert packet["total_bytes"] <= PacketLimits().total_bytes
    share = PacketLimits().total_bytes // PacketLimits().max_documents
    for document in packet["documents"]:
        assert sum(len(s["text"].encode()) for s in document["sections"]) <= share


def test_a_document_that_wins_a_slot_always_gets_its_best_section():
    """A share computed for many candidates must not drop a page to nothing."""
    pages = {f"wiki/sources/paper-{n}.md": long_note(f"paper-{n}", words="regional inheritance") for n in range(1, 5)}
    s3 = wiki_with_index(pages)
    limits = PacketLimits(total_bytes=8000, max_documents=4, max_sections=12, section_chars=3000, context_chars=3000)
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=limits)

    assert len(packet["documents"]) >= 2
    for document in packet["documents"]:
        assert document["sections"], document["key"]
    assert packet["total_bytes"] <= limits.total_bytes


def test_a_question_matching_two_pages_uses_the_budget_it_needs_not_one_eighth_of_it():
    pages = {"wiki/sources/paper-one.md": long_note("paper-one", words="regional inheritance"),
             "wiki/sources/paper-two.md": long_note("paper-two", words="regional inheritance")}
    s3 = wiki_with_index(pages)
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())

    assert len(packet["documents"]) == 2
    for document in packet["documents"]:
        assert [s["kind"] for s in document["sections"]] == ["methods", "results", "limitations"]
    assert packet["total_bytes"] > PacketLimits().total_bytes // PacketLimits().max_documents


def test_long_results_section_is_cut_and_marks_truncated_decisive():
    results = ("stable " * 1500).strip()  # 9,000 characters minus the final space
    assert len(results) >= 8990
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note(results=results)})
    packet = build_packet("stable inheritance", index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    document = packet["documents"][0]
    cut = next(s for s in document["sections"] if s["name"] == "Results")
    assert cut["truncated"] is True and len(cut["text"]) == 6000
    assert cut["end"] - cut["start"] == 6000 and cut["full_end"] > cut["end"]
    assert {"key": "wiki/sources/paper-one.md", "section": "Results"} in packet["truncated"]
    assert packet["evidence_state"] == "truncated_decisive"


def test_packet_of_only_link_lists_is_links_only():
    page = "# Hub\n\n## Linked overviews\n\n- [[overviews/regional-inheritance]]\n- [[overviews/cohort-inheritance]]\n"
    s3 = wiki_with_index({"wiki/concepts/hub.md": page})
    packet = build_packet("regional inheritance", index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    assert kinds(packet, "wiki/concepts/hub.md") == ["links"]
    assert packet["evidence_state"] == "links_only"


def test_question_pages_are_excluded_from_search_and_links_only_when_read():
    question_page = "---\ntitle: Regional inheritance question\n---\n\n# Regional inheritance question\n\n## Answer\n\nRegional inheritance was stable.\n"
    s3 = wiki_with_index({"wiki/questions/regional.md": question_page})
    con, etag = opened(s3)
    assert ep.search(con, "regional inheritance", 10) == []
    explicit = ep.search(con, "regional inheritance", 10, doc_type="question")
    assert [h["key"] for h in explicit] == ["wiki/questions/regional.md"]
    packet = build_packet("regional inheritance", index=(con, etag), s3=s3, bucket="b", limits=PacketLimits(),
                          extra_reads=["wiki/questions/regional.md"])
    assert [d["key"] for d in packet["documents"]] == ["wiki/questions/regional.md"]
    assert packet["documents"][0]["doc_type"] == "question"
    assert packet["evidence_state"] == "links_only"


def test_empty_search_is_insufficient():
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    packet = build_packet("zebrafish optogenetics", index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    assert packet["documents"] == [] and packet["evidence_state"] == "insufficient"
    assert packet["queries"][0]["hits"] == []


def test_sufficient_packet_records_versions_queries_and_hits():
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits(),
                          extra_queries=["regional inheritance stability cohort"])
    assert packet["question"] == QUESTION
    assert packet["index_etag"] == s3.etag(INDEX_KEY)
    assert [q["query"] for q in packet["queries"]] == [QUESTION, "regional inheritance stability cohort"]
    hit = packet["queries"][0]["hits"][0]
    assert hit["key"] == "wiki/sources/paper-one.md" and hit["doc_type"] == "note" and hit["title"] == "Paper one"
    assert isinstance(hit["score"], float)
    document = packet["documents"][0]
    assert document["etag"] == s3.etag("wiki/sources/paper-one.md")
    assert document["version_id"] == "v1"
    assert len(document["sha256"]) == 64
    assert kinds(packet, "wiki/sources/paper-one.md") == ["methods", "results", "limitations"]
    assert packet["evidence_state"] == "sufficient"
    assert packet["total_bytes"] == sum(len(s["text"].encode()) for d in packet["documents"] for s in d["sections"])
    assert packet["selection_notes"]
    assert s3.writes == []


def test_supplemental_query_beyond_limit_raises():
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    with pytest.raises(ValueError):
        build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits(searches=1),
                     extra_queries=["one more"])


def test_max_documents_and_max_sections_are_recorded_as_omissions():
    pages = {f"wiki/sources/paper-{i}.md": source_note(f"Paper {i}", stem=f"paper-{i}") for i in range(3)}
    s3 = wiki_with_index(pages)
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits(max_documents=2, max_sections=4))
    assert len(packet["documents"]) == 2
    assert sum(len(d["sections"]) for d in packet["documents"]) == 4
    reasons = {o["reason"] for o in packet["omitted"]}
    assert "max_documents" in reasons or "max_sections" in reasons


def test_unpublishable_hit_key_is_omitted_not_fatal():
    draft = "# Draft\n\n## Results\n\nRegional inheritance in a draft page.\n"
    s3 = wiki_with_index({"wiki/drafts/draft-one.md": draft, "wiki/sources/paper-one.md": source_note()})
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    assert {"key": "wiki/drafts/draft-one.md", "reason": "invalid_key"} in packet["omitted"]
    assert [d["key"] for d in packet["documents"]] == ["wiki/sources/paper-one.md"]
    assert "wiki/drafts/draft-one.md" not in s3.reads


def test_missing_page_is_omitted_not_fatal():
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    del s3.objects["wiki/sources/paper-one.md"]
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    assert packet["documents"] == []
    assert {"key": "wiki/sources/paper-one.md", "reason": "not_found"} in packet["omitted"]
    assert packet["evidence_state"] == "insufficient"


# ---------------------------------------------------------------------------------------------
# Index access, search keys, reads and backlinks
# ---------------------------------------------------------------------------------------------

def test_open_index_downloads_once_and_refreshes_when_etag_changes(tmp_path):
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    con, etag = ep.open_index(s3, "b", INDEX_KEY, tmp_path)
    assert etag == s3.etag(INDEX_KEY)
    assert con.execute("SELECT count(*) FROM docs").fetchone()[0] == 1
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO links VALUES ('a','b','c','d')")
    con.close()
    con, _ = ep.open_index(s3, "b", INDEX_KEY, tmp_path)
    con.close()
    assert s3.reads.count(INDEX_KEY) == 1
    s3._store(INDEX_KEY, build_index({"wiki/sources/paper-one.md": source_note(), "wiki/sources/paper-two.md": source_note("Two", stem="paper-two")}))
    con, etag2 = ep.open_index(s3, "b", INDEX_KEY, tmp_path)
    assert etag2 != etag and s3.reads.count(INDEX_KEY) == 2
    assert con.execute("SELECT count(*) FROM docs").fetchone()[0] == 2
    con.close()


def test_search_adds_keys_with_the_campaign_folder_mapping():
    pages = {"wiki/sources/paper-one.md": source_note(),
             "wiki/concepts/inheritance.md": "# Inheritance\n\n## Summary\n\nRegional inheritance across cohorts.\n",
             "wiki/overviews/asd-ndd/regional.md": "# Regional overview\n\n## Summary\n\nRegional inheritance overview.\n",
             "wiki/asd-ndd/paper-page.md": "# Paper page\n\n## Results\n\nRegional inheritance in a paper page.\n"}
    s3 = wiki_with_index(pages)
    con, _ = opened(s3)
    hits = ep.search(con, "regional inheritance", 10)
    keys = {h["doc_type"]: h["key"] for h in hits}
    assert keys["note"] == "wiki/sources/paper-one.md"
    assert keys["concept"] == "wiki/concepts/inheritance.md"
    assert keys["overview"] == "wiki/overviews/asd-ndd/regional.md"
    assert keys["paper"] == "wiki/asd-ndd/paper-page.md"  # indexed path without the data/ prefix
    assert all(set(h) >= {"key", "title", "doc_type", "score", "section", "doc_id"} for h in hits)


def test_read_excerpt_defaults_to_an_outline_with_empty_text():
    s3 = MemoryS3({"wiki/sources/paper-one.md": source_note()})
    result = ep.read_excerpt(s3, "b", "wiki/sources/paper-one.md")
    assert result["mode"] == "outline" and result["text"] == ""
    assert [s["name"] for s in result["sections"]] == ["(opening)", "Methods", "Results", "Limitations", "Related pages"]
    assert result["sections"][2]["kind"] == "results" and result["sections"][2]["chars"] > 0
    assert result["metadata"]["title"] == "Paper one" and result["metadata"]["doi"] == "10.1000/paper-one"
    assert result["etag"] == s3.etag("wiki/sources/paper-one.md") and result["version_id"] == "v1"
    assert len(result["sha256"]) == 64


def test_read_excerpt_returns_a_section_window_with_next_start():
    s3 = MemoryS3({"wiki/sources/paper-one.md": source_note()})
    first = ep.read_excerpt(s3, "b", "wiki/sources/paper-one.md", section="results", max_chars=10)
    assert first["mode"] == "section" and first["section"] == "Results" and first["kind"] == "results"
    assert first["text"] == "Regional i" and first["start"] == 0 and first["next_start"] == 10 and first["has_more"] is True
    rest = ep.read_excerpt(s3, "b", "wiki/sources/paper-one.md", section="Results", start=10, max_chars=8000)
    assert first["text"] + rest["text"] == "Regional inheritance was stable (n = 120, p = 0.01)."
    assert rest["next_start"] is None and rest["has_more"] is False
    assert rest["total_chars"] == first["total_chars"]


def test_read_excerpt_rejects_oversized_windows_and_bad_keys():
    s3 = MemoryS3({"wiki/sources/paper-one.md": source_note()})
    with pytest.raises(ValueError):
        ep.read_excerpt(s3, "b", "wiki/sources/paper-one.md", section="Results", max_chars=8001)
    with pytest.raises(ValueError):
        ep.read_excerpt(s3, "b", "wiki/sources/paper-one.md", section="Results", max_chars=0)
    with pytest.raises(ValueError):
        ep.read_excerpt(s3, "b", "wiki/sources/paper-one.md", section="Nope")
    with pytest.raises(ValueError):
        ep.read_excerpt(s3, "b", "papers/paper-one/original.pdf")
    with pytest.raises(ValueError):
        ep.read_excerpt(s3, "b", "wiki/../index/wiki-index-v2.sqlite3")
    with pytest.raises(FileNotFoundError):
        ep.read_excerpt(s3, "b", "wiki/sources/missing.md")
    assert s3.writes == []


def test_backlinks_return_citing_documents_from_the_links_table():
    pages = {"wiki/sources/paper-one.md": source_note(),
             "wiki/overviews/existing.md": "# Existing\n\n## Summary\n\nAn overview citing [[sources/paper-one]].\n"}
    s3 = wiki_with_index(pages)
    con, etag = opened(s3)
    result = ep.backlinks((con, etag), "wiki/overviews/existing.md")
    assert result["doc_type"] == "overview" and result["doc_id"] == "existing"
    assert result["backlinks"] == [{"key": "wiki/sources/paper-one.md", "doc_type": "note", "doc_id": "paper-one", "title": "Paper one"}]
    assert result["index_etag"] == etag and result["links_table"] is True
    cited = ep.backlinks(con, "wiki/sources/paper-one.md")
    assert [b["key"] for b in cited["backlinks"]] == ["wiki/overviews/existing.md"]
    assert ep.backlinks(con, "wiki/concepts/nobody.md")["backlinks"] == []


def test_requested_read_with_unusable_key_is_omitted_not_fatal():
    """A model chooses request_lookup keys after reading page text; a bad key must not abort the packet."""
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits(),
                          extra_reads=["wiki/drafts/x.md", {"key": "../papers/x.md"}, {"nokey": 1}, 42])
    reasons = [o for o in packet["omitted"] if o["reason"] == "invalid_key"]
    assert len(reasons) == 4
    assert [d["key"] for d in packet["documents"]] == ["wiki/sources/paper-one.md"]
    assert "wiki/drafts/x.md" not in s3.reads


def test_oversized_page_is_skipped_as_too_large_and_never_read_whole():
    from byeori.evidence_packet import MAX_PAGE_BYTES, PageTooLarge, read_page
    assert MAX_PAGE_BYTES == 512_000  # a note-sized bound; the outline scans below are linear within it
    huge = "# Huge\n\n## Results\n\n" + ("regional inheritance " * 20) + "x" * (MAX_PAGE_BYTES + 10)
    s3 = wiki_with_index({"wiki/sources/huge.md": huge, "wiki/sources/paper-one.md": source_note()})
    with pytest.raises(PageTooLarge):
        read_page(s3, "b", "wiki/sources/huge.md")
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    assert {"key": "wiki/sources/huge.md", "reason": "too_large"} in packet["omitted"]
    assert [d["key"] for d in packet["documents"]] == ["wiki/sources/paper-one.md"]


def test_page_derived_title_and_section_names_are_bounded():
    from byeori.evidence_packet import NAME_MAX_CHARS, TITLE_MAX_CHARS
    long_title = "T" * 5000
    long_heading = "Results " + "h" * 5000
    page = (f"---\ntitle: {long_title}\n---\n\n# {long_title}\n\n## {long_heading}\n\n"
            "Regional inheritance was stable across sites.\n\n## Limitations\n\nSmall cohort.\n")
    s3 = wiki_with_index({"wiki/sources/long.md": page})
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    document = packet["documents"][0]
    assert len(document["title"]) <= TITLE_MAX_CHARS
    assert all(len(section["name"]) <= NAME_MAX_CHARS for section in document["sections"])
    assert all(len(entry.get("section", "")) <= NAME_MAX_CHARS for entry in packet["omitted"] + packet["truncated"])


# ---------------------------------------------------------------------------------------------
# Adversarial pages, keys and queries (review F)
# ---------------------------------------------------------------------------------------------

def elapsed(function, *args):
    started = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - started


def test_outline_and_strip_are_linear_on_adversarial_text():
    """A page under MAX_PAGE_BYTES must never stall the answer worker past its Lambda timeout."""
    from byeori.wiki_connections import BACKLINK_START
    heading_of_spaces = "# T\n\n## " + " " * 500_000 + "x\n\nRegional inheritance.\n"
    newline_section = "# T\n\n## Results\n\na" + "\n" * 200_000 + "b\n"
    unmatched_markers = "# T\n\n## Results\n\n" + BACKLINK_START * 10_000 + "\nRegional inheritance.\n"
    brackets = "# T\n\n## Results\n\n" + "[" * 200_000 + "\n"
    for text in (heading_of_spaces, newline_section, unmatched_markers, brackets):
        assert len(text.encode()) <= ep.MAX_PAGE_BYTES
        sections, seconds = elapsed(outline, text)
        assert seconds < 1.0, f"outline took {seconds:.1f}s"
        assert sections
    assert [s.name for s in outline(heading_of_spaces)] == ["x"]
    assert outline(newline_section)[0].text == "a" + "\n" * 200_000 + "b"
    stripped, seconds = elapsed(strip_managed_blocks, unmatched_markers)
    assert seconds < 1.0 and stripped == unmatched_markers
    assert BACKLINK_START in outline(unmatched_markers)[0].text


def test_strip_managed_blocks_matches_the_regex_semantics():
    """The marker loop removes exactly what the campaign's non-greedy block patterns remove."""
    from byeori.wiki_connections import BACKLINK_BLOCK, BACKLINK_END, BACKLINK_START, CATALOG_BLOCK, CATALOG_END, CATALOG_START
    samples = [
        "a" + BACKLINK_START + "x" + BACKLINK_END + "b" + BACKLINK_START + "y" + BACKLINK_END + "c",
        "a" + BACKLINK_START + "x" + BACKLINK_START + "y" + BACKLINK_END + "b" + BACKLINK_END + "c",
        "a" + BACKLINK_END + "b" + BACKLINK_START + "c",
        BACKLINK_START + CATALOG_START + "x" + BACKLINK_END + CATALOG_END + "tail",
        "a" + CATALOG_START + "\n- [[overviews/x]]\n" + CATALOG_END + "\n" + BACKLINK_START + "\n" + BACKLINK_END,
        "no markers at all",
        "",
    ]
    for text in samples:
        expected = CATALOG_BLOCK.sub("", BACKLINK_BLOCK.sub("", text))
        assert strip_managed_blocks(text) == expected


def test_page_key_rejects_keys_longer_than_512_bytes():
    exact = "wiki/sources/" + "a" * (512 - len("wiki/sources/") - len(".md")) + ".md"
    assert len(exact.encode("utf-8")) == 512 and ep.page_key(exact) == exact
    long_key = "wiki/sources/" + "a" * 600 + ".md"
    with pytest.raises(ValueError):
        ep.page_key(long_key)
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits(), extra_reads=[long_key])
    assert [o["reason"] for o in packet["omitted"]] == ["invalid_key"]
    assert [d["key"] for d in packet["documents"]] == ["wiki/sources/paper-one.md"]
    assert all(not read.startswith("wiki/sources/aaaa") for read in s3.reads)


class DeniedS3(MemoryS3):
    """A bucket whose read of one key fails with an S3 error other than a missing object."""

    def __init__(self, objects, denied: str, code: str = "AccessDenied"):
        super().__init__(objects)
        self.denied, self.code = denied, code

    def get_object(self, *, Bucket, Key, **kwargs):
        if Key == self.denied:
            raise ClientError({"Error": {"Code": self.code}}, "GetObject")
        return super().get_object(Bucket=Bucket, Key=Key, **kwargs)


def test_unreadable_page_is_omitted_as_read_error_not_fatal():
    pages = {"wiki/sources/paper-one.md": source_note(),
             "wiki/sources/paper-two.md": source_note("Paper two", stem="paper-two")}
    s3 = DeniedS3(pages, denied="wiki/sources/paper-two.md")
    s3._store(INDEX_KEY, build_index(pages))
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    assert {"key": "wiki/sources/paper-two.md", "reason": "read_error:AccessDenied"} in packet["omitted"]
    assert [d["key"] for d in packet["documents"]] == ["wiki/sources/paper-one.md"]
    assert packet["evidence_state"] == "sufficient"
    with pytest.raises(ClientError):
        ep.read_page(s3, "b", "wiki/sources/paper-two.md")  # a direct client read still surfaces the error
    with pytest.raises(FileNotFoundError):
        ep.read_page(s3, "b", "wiki/sources/missing.md")


def test_invalid_utf8_page_is_omitted_as_not_utf8():
    pages = {"wiki/sources/paper-one.md": source_note()}
    s3 = wiki_with_index(pages)
    s3._store("wiki/sources/paper-one.md", b"# Bad\n\n## Results\n\n\xff\xfe regional inheritance\n")
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    assert packet["omitted"] == [{"key": "wiki/sources/paper-one.md", "reason": "not_utf8"}]
    assert packet["evidence_state"] == "insufficient"


def test_search_strips_control_characters_and_rejects_an_empty_remainder():
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    con, _ = opened(s3)
    assert [h["key"] for h in ep.search(con, "regional\x00 inheritance", 10)] == ["wiki/sources/paper-one.md"]
    assert [h["key"] for h in ep.search(con, "regional\n\tinheritance", 10)] == ["wiki/sources/paper-one.md"]
    assert [h["key"] for h in ep.search(con, "\x01regional\x7f \x1binheritance\r", 10)] == ["wiki/sources/paper-one.md"]
    for query in ("\x00\x00", "\x00 \x7f", "", "   ", 42):
        with pytest.raises(ValueError):
            ep.search(con, query, 10)


def test_nul_only_question_completes_as_insufficient_instead_of_raising():
    s3 = wiki_with_index({"wiki/sources/paper-one.md": source_note()})
    packet = build_packet("\x00\x00\x00", index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    assert packet["queries"] == [{"query": "\x00\x00\x00", "hits": [], "error": "invalid_query"}]
    assert packet["documents"] == [] and packet["evidence_state"] == "insufficient"
    assert packet["question"] == "\x00\x00\x00"
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits(), extra_queries=["\x00"])
    assert [q["query"] for q in packet["queries"]] == [QUESTION, "\x00"]
    assert packet["queries"][0]["hits"] and "error" not in packet["queries"][0]
    assert packet["queries"][1] == {"query": "\x00", "hits": [], "error": "invalid_query"}
    assert packet["evidence_state"] == "sufficient"
    with pytest.raises(ValueError):
        build_packet("   ", index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())


def synthesis_page(sections: dict[str, str]) -> str:
    """An overview whose headings are its own prose, as every synthesis page's are."""
    body = "\n\n".join(f"## {name}\n\n{text}" for name, text in sections.items())
    return "---\ntitle: How the axis is constructed\ncategory: liver\n---\n\n# How the axis is constructed\n\n" + body + "\n"


def test_every_section_of_a_synthesis_page_is_evidence():
    """A note's headings are fixed, so its kinds are read off them; a synthesis names its sections
    in its own words and the kind patterns call them all `other`. On 2026-09-24 a liver question got
    2 of an overview's 8 sections and had to reconstruct the rest indirectly."""
    stable = "Regional inheritance was stable in the cohort. "
    page = synthesis_page({
        "1. Five constructions of the axis": stable * 20,
        "2. Cell-resolved imaging changes the conclusion": stable * 20,
        "3. How much is zonated is not comparable": stable * 20,
        "4. Which models exist here, and which do not": stable * 20,
    })
    s3 = wiki_with_index({"wiki/overviews/axis.md": page,
                          "wiki/sources/paper-one.md": source_note("Paper one", stem="paper-one")})
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    overview = next(d for d in packet["documents"] if d["key"] == "wiki/overviews/axis.md")
    assert len(overview["sections"]) == 4, overview["sections"]
    # Three of the four are `other`: the kind patterns have nothing to match in their headings.
    assert sum(1 for s in overview["sections"] if s["kind"] == "other") == 3
    # A note is unchanged: it still contributes its best section plus the kind picks, not everything.
    note = next(d for d in packet["documents"] if d["key"] == "wiki/sources/paper-one.md")
    assert len(note["sections"]) <= 4


def test_a_synthesis_section_is_not_cut_to_the_background_length():
    """Its prose is the cross-paper conclusion, not background that could crowd out results."""
    long_text = "Regional inheritance was stable in the cohort. " * 400
    s3 = wiki_with_index({"wiki/overviews/axis.md": synthesis_page({"1. Five constructions": long_text})})
    packet = build_packet(QUESTION, index=opened(s3), s3=s3, bucket="b", limits=PacketLimits())
    section = packet["documents"][0]["sections"][0]
    assert section["kind"] == "other"
    assert len(section["text"]) == PacketLimits().section_chars == 6000
