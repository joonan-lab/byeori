"""byeori.lab_pages: answered questions as Markdown, outside the index, reachable by link."""
from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from byeori import lab_pages
from byeori.lab_store import ANSWER_PAGE_PREFIX, ConditionFailed, PageWriter
from lab_fakes import MemoryS3

NOW = datetime(2026, 9, 22, 9, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[1]
QUESTION = "코호트에서 지역 유전 안정성 결과는 무엇인가요?"
ANSWER = {
    "answer": "지역 유전 안정성은 코호트에서 유지되었습니다 (n = 120, p = 0.01).",
    "citations": [{"key": "wiki/sources/paper-one.md", "section": "Results"},
                  {"key": "wiki/sources/paper-one.md", "section": "Limitations"},
                  {"key": "wiki/overviews/regional.md", "section": None}],
    "limitations": ["단일 기관 코호트 120가족."],
    "unresolved_items": ["다른 조상 배경에서의 재현은 확인되지 않았습니다."],
    "evidence_state": "sufficient",
    "status": "completed",
    "model_id": "global.anthropic.claude-opus-5",
    "policy_revision": "2026-09-21-v1",
    "member_id": "m1",
    "question": QUESTION,
}


def world(**kwargs) -> tuple[MemoryS3, PageWriter]:
    s3 = MemoryS3(**kwargs)
    return s3, PageWriter(s3, "bucket")


def publish(writer, answer=None, *, question=QUESTION, job_id="j1", period="2026-09"):
    return lab_pages.publish_answer(writer, answer or ANSWER, question=question, job_id=job_id,
                                    period=period, now=NOW)


# ---------------------------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------------------------

def test_an_answered_question_becomes_one_markdown_page_under_the_unindexed_prefix():
    s3, writer = world()

    report = publish(writer)

    key = "wiki/lab-questions/2026-09/j1.md"
    assert report["page"] == key and report["errors"] == []
    assert key.startswith(ANSWER_PAGE_PREFIX)
    body = s3.objects[key].decode("utf-8")
    assert body.startswith("---\n") and 'title: "코호트에서 지역 유전 안정성 결과는 무엇인가요?"' in body
    assert "indexed: false" in body and 'category: "lab-questions"' in body
    assert 'job_id: "j1"' in body and 'evidence_state: "sufficient"' in body
    assert ANSWER["answer"] in body
    assert "- [[sources/paper-one]] — Results, Limitations" in body
    assert "- [[overviews/regional]]" in body
    assert "단일 기관 코호트 120가족." in body
    assert "다른 조상 배경에서의 재현은 확인되지 않았습니다." in body
    assert "runs/lab-questions/j1/" in body


def test_the_page_never_carries_who_asked():
    """A page the whole lab can open must not say which member asked (design section 7)."""
    s3, writer = world()

    publish(writer)

    body = s3.objects["wiki/lab-questions/2026-09/j1.md"].decode("utf-8")
    assert "m1" not in body and "member" not in body.casefold()


def test_an_answer_with_no_body_or_citations_still_gets_a_page_that_says_so():
    """The four 2026-09-22 answers cut at the output limit had exactly this shape."""
    s3, writer = world()

    report = publish(writer, {"answer": "", "citations": [], "limitations": ["출력 한도에서 잘렸습니다."],
                              "unresolved_items": [], "evidence_state": "insufficient", "status": "partial"})

    assert report["page"] and report["hubs"] == [] and report["errors"] == []
    body = s3.objects["wiki/lab-questions/2026-09/j1.md"].decode("utf-8")
    assert "_이 질문에는 답변 본문이 저장되지 않았습니다._" in body and "- _인용 없음_" in body
    assert "출력 한도에서 잘렸습니다." in body


# ---------------------------------------------------------------------------------------------
# The hubs: how an indexed page reaches a question
# ---------------------------------------------------------------------------------------------

def test_every_cited_page_gets_a_hub_and_later_answers_are_added_to_it():
    s3, writer = world()

    first = publish(writer, job_id="j1")
    second = publish(writer, {**ANSWER, "citations": [{"key": "wiki/sources/paper-one.md", "section": "Results"}]},
                     question="두 번째 질문", job_id="j2")

    assert [hub["outcome"] for hub in first["hubs"]] == ["created", "created"]
    assert [hub["outcome"] for hub in second["hubs"]] == ["appended"]
    hub = s3.objects["wiki/lab-questions/by-page/sources/paper-one.md"].decode("utf-8")
    assert hub.index("[[lab-questions/2026-09/j2|두 번째 질문]]") < hub.index("[[lab-questions/2026-09/j1")
    assert "# [[sources/paper-one]] 을 근거로 답한 질문" in hub and "indexed: false" in hub
    # A synthesis page is reachable the same way as a source note.
    assert "wiki/lab-questions/by-page/overviews/regional.md" in s3.objects


def test_the_same_answer_published_twice_does_not_repeat_its_hub_entry():
    s3, writer = world()

    publish(writer)
    again = publish(writer)

    assert [hub["outcome"] for hub in again["hubs"]] == ["unchanged", "unchanged"]
    hub = s3.objects["wiki/lab-questions/by-page/sources/paper-one.md"].decode("utf-8")
    assert hub.count("[[lab-questions/2026-09/j1") == 1


def test_a_hub_two_answers_reach_at_once_keeps_both_entries():
    """S3 refuses the second conditional write; the retry reads the newer hub and appends to it."""
    s3, writer = world(conflict_keys={"wiki/lab-questions/by-page/sources/paper-one.md"})

    report = publish(writer, {**ANSWER, "citations": [{"key": "wiki/sources/paper-one.md"}]})

    assert [hub["outcome"] for hub in report["hubs"]] == ["created"] and report["errors"] == []
    assert "[[lab-questions/2026-09/j1" in s3.objects["wiki/lab-questions/by-page/sources/paper-one.md"].decode("utf-8")


def test_a_hub_that_will_not_settle_is_reported_and_the_page_still_stands():
    s3, writer = world()
    calls = {"n": 0}
    original = writer.put_markdown

    def always_conflict(key, text, **kwargs):
        if key.startswith(lab_pages.HUB_PREFIX):
            calls["n"] += 1
            raise ConditionFailed(f"{key} keeps moving")
        return original(key, text, **kwargs)

    writer.put_markdown = always_conflict
    report = publish(writer, {**ANSWER, "citations": [{"key": "wiki/sources/paper-one.md"}]})

    assert report["page"] == "wiki/lab-questions/2026-09/j1.md"
    assert [hub["outcome"] for hub in report["hubs"]] == ["conflict"]
    assert calls["n"] == lab_pages.HUB_ATTEMPTS


def test_a_question_page_never_becomes_a_hub_of_its_own():
    assert lab_pages.link_of("wiki/lab-questions/2026-09/j1.md") is None
    assert lab_pages.link_of("wiki/lab-questions/by-page/sources/paper-one.md") is None
    assert lab_pages.link_of("wiki/sources/paper-one.md") == "sources/paper-one"
    assert lab_pages.link_of("wiki/concepts/chromatin.md") == "concepts/chromatin"
    for other in ("papers/paper-one/original.pdf", "index/wiki-index-v2.sqlite3", "wiki/sources/a/b/c.md",
                  "runs/lab-questions/j1/answer.json", "wiki/top.md", None, 7):
        assert lab_pages.link_of(other) is None, other


def test_the_standing_line_an_indexed_page_carries_points_at_its_own_hub():
    line = lab_pages.page_link_line("sources/paper-one")
    assert line == "- 이 페이지를 근거로 답한 랩 질문: [[lab-questions/by-page/sources/paper-one]]"
    assert lab_pages.hub_key("sources/paper-one") == "wiki/lab-questions/by-page/sources/paper-one.md"


# ---------------------------------------------------------------------------------------------
# The boundary: this module can only reach its own prefix, and never fails an answer
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "wiki/sources/paper-one.md", "wiki/overviews/regional.md", "wiki/questions/x.md",
    "papers/paper-one/original.pdf", "index/wiki-index-v2.sqlite3", "runs/lab-questions/j1/answer.json",
    "wiki/lab-questions/../sources/paper-one.md", "wiki/lab-questions/x.json", "wiki/lab-questionsX/a.md",
])
def test_the_page_writer_refuses_every_key_outside_its_one_prefix(key):
    s3, writer = world()
    with pytest.raises(ValueError):
        writer.put_markdown(key, "# never written\n")
    with pytest.raises(ValueError):
        writer.get_markdown(key)
    assert s3.writes == []


def test_a_page_that_cannot_be_written_is_reported_and_never_raised():
    """The member already has the answer when this runs; a page must not take it away."""
    s3, writer = world()

    def refuse(*_args, **_kwargs):
        raise RuntimeError("bucket unreachable")

    writer.put_markdown = refuse
    report = publish(writer)

    assert report["page"] is None and report["hubs"] == []
    assert report["errors"] == [{"key": "wiki/lab-questions/2026-09/j1.md", "error": "RuntimeError",
                                 "message": "bucket unreachable"}]


def test_the_module_writes_through_the_page_writer_and_names_no_other_key():
    """A static read of the module: no literal key outside the prefix, and no S3 client call."""
    source = (ROOT / "src/byeori/lab_pages.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"put_object", "delete_object", "copy_object"}:
            raise AssertionError(f"lab_pages calls S3 directly through {node.attr}")
    # ``"wiki/"`` alone is the prefix this module strips to build a link; anything longer is a key.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        value = node.value
        if value in {"wiki/", "papers/", "index/"}:
            continue
        if value.startswith(("wiki/", "papers/", "index/")):
            assert value.startswith(ANSWER_PAGE_PREFIX), value
