"""Publisher and archive injection into the research engine (docs/LAB-QUESTION-WORKFLOW.md section 7, P4).

The approved research worker reuses ``question_agent.run_answer`` with an injected publisher so one
scope check covers every wiki write, and keeps the final answer in an operational receipt instead of
``wiki/questions/``. These tests pin the injection points and confirm that the defaults still take the
campaign path through ``wiki_connections.publish_page``.
"""
from __future__ import annotations

import copy
import itertools

import pytest

from byeori import question_agent
from byeori.question_agent import WikiTools, question_key_for, run_answer
from byeori.wiki_connections import publish_page
from lab_fakes import MemoryS3, source_note

CALL_IDS = itertools.count()
MODEL = "global.anthropic.claude-opus-5"
NOTE, OVERVIEW, CONCEPT = "wiki/sources/paper-one.md", "wiki/overviews/existing.md", "wiki/concepts/new-insight.md"
OVERVIEW_TEXT = "# Existing synthesis\n\nEarlier claim.\n\nUnrelated conclusion stays exactly as written.\n"
NEW_CLAIM = "The regional result narrows the earlier claim [[sources/paper-one]]."
NEW_PAGE = "A reusable distinction\n\n[[sources/paper-one]] [[overviews/existing]]\n"
ANSWER = "영역과 단일 위치는 다른 측정 단위입니다. [[sources/paper-one]]"


def turn(*calls, answer=None):
    content = [{"text": answer}] if answer is not None else []
    for name, args in calls:
        content.append({"toolUse": {"toolUseId": f"call-{next(CALL_IDS)}", "name": name, "input": args}})
    return {"output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use" if calls else "end_turn", "usage": {"inputTokens": 100, "outputTokens": 50}}


class Model:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def converse(self, client, request):
        self.requests.append(copy.deepcopy(request))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response, 0, 1


class RecordingPublisher:
    """Delegates to the real publisher and records every call it received."""

    def __init__(self, delegate=publish_page):
        self.delegate = delegate
        self.calls: list[dict] = []

    def __call__(self, s3, bucket, key, text, *, check_remaining=None, **options):
        self.calls.append({"key": key, "text": text, "options": dict(options), "check_remaining": check_remaining})
        return self.delegate(s3, bucket, key, text, check_remaining=check_remaining, **options)


def search(event):
    return {"index_etag": "current-index", "results": [
        {"doc_type": "note", "doc_id": "paper-one", "title": "Original paper",
         "section": "Results", "score": 3, "path": "data/sources/paper-one.md"}]}


def research_turns():
    return [
        turn(("search_wiki", {"query": "Which studies compare regional and site-level inheritance?"})),
        turn(("read_page", {"key": NOTE}), ("read_page", {"key": OVERVIEW})),
        turn(("edit_page", {"key": OVERVIEW, "old_text": "Earlier claim.", "new_text": NEW_CLAIM}),
             ("write_page", {"key": CONCEPT, "markdown": NEW_PAGE})),
        turn(("refresh_links", {"key": CONCEPT})),
        turn(answer=ANSWER),
    ]


def run(cloud, responses, title="영역과 단일 위치의 결과가 왜 다른가", **options):
    model = Model(responses)
    result = run_answer({"title": title}, s3=cloud, bucket="bucket", model_client=object(), model_id=MODEL,
                        reasoning="default", converse=model.converse, search=search, **options)
    return result, model


def wiki_keys(cloud):
    return [key for key, _ in cloud.writes if key.startswith("wiki/")]


def test_default_path_publishes_through_publish_page_and_archives_the_question_page(monkeypatch):
    recorder = RecordingPublisher()
    monkeypatch.setattr(question_agent, "publish_page", recorder)
    cloud = MemoryS3({NOTE: source_note(), OVERVIEW: OVERVIEW_TEXT})

    result, _ = run(cloud, research_turns())

    assert result["status"] == "answer_ready", result
    slug, question_key = question_key_for(result["title"])
    assert (result["slug"], result["question_key"]) == (slug, question_key)
    assert [call["key"] for call in recorder.calls] == [OVERVIEW, CONCEPT, CONCEPT, question_key]
    assert recorder.calls[0]["options"].keys() == {"expected_etag"}
    assert recorder.calls[1]["options"] == {"create_only": True}
    assert recorder.calls[2]["options"].keys() == {"expected_etag"}
    assert recorder.calls[3]["options"]["create_only"] is True and recorder.calls[3]["options"]["expected_etag"] is None
    assert all(callable(call["check_remaining"]) for call in recorder.calls)
    assert ANSWER in cloud.text(question_key)
    assert NEW_CLAIM in cloud.text(OVERVIEW) and cloud.text(CONCEPT).startswith(NEW_PAGE)


def test_wiki_tools_default_publisher_is_publish_page_and_an_injected_one_replaces_it():
    cloud = MemoryS3({OVERVIEW: OVERVIEW_TEXT})
    assert WikiTools(cloud, "bucket", search, "auto").publisher is publish_page

    recorder = RecordingPublisher()
    tools = WikiTools(cloud, "bucket", search, "auto", publisher=recorder)
    assert tools.publisher is recorder
    tools.call("read_page", {"key": OVERVIEW})
    saved = tools.call("edit_page", {"key": OVERVIEW, "old_text": "Earlier claim.", "new_text": NEW_CLAIM})
    assert [call["key"] for call in recorder.calls] == [OVERVIEW]
    assert recorder.calls[0]["options"] == {"expected_etag": cloud.etag(OVERVIEW)} or \
        recorder.calls[0]["options"]["expected_etag"] is not None
    assert saved["key"] == OVERVIEW and tools.writes[OVERVIEW] is saved


def test_injected_publisher_receives_every_edit_creation_refresh_and_the_final_question_page():
    recorder = RecordingPublisher()
    cloud = MemoryS3({NOTE: source_note(), OVERVIEW: OVERVIEW_TEXT})

    result, _ = run(cloud, research_turns(), publisher=recorder)

    assert result["status"] == "answer_ready", result
    assert [call["key"] for call in recorder.calls] == [OVERVIEW, CONCEPT, CONCEPT, result["question_key"]]
    assert result["question_key"].startswith("wiki/questions/")
    assert recorder.calls[-1]["text"].startswith("---\n") and ANSWER in recorder.calls[-1]["text"]
    assert {page["key"] for page in result["pages_written"]} == {OVERVIEW, CONCEPT}


def test_injected_publisher_refusal_surfaces_as_a_tool_error_and_the_run_continues():
    def refusing(s3, bucket, key, text, *, check_remaining=None, **options):
        if key == CONCEPT:
            raise ValueError("outside the approved research scope")
        return publish_page(s3, bucket, key, text, check_remaining=check_remaining, **options)

    cloud = MemoryS3({NOTE: source_note(), OVERVIEW: OVERVIEW_TEXT})
    result, model = run(cloud, [
        turn(("read_page", {"key": OVERVIEW})),
        turn(("edit_page", {"key": OVERVIEW, "old_text": "Earlier claim.", "new_text": NEW_CLAIM}),
             ("write_page", {"key": CONCEPT, "markdown": NEW_PAGE})),
        turn(answer=ANSWER),
    ], publisher=refusing, publish_question=False)

    tool_results = [block for block in model.requests[2]["messages"][-1]["content"] if "toolResult" in block]
    assert [block["toolResult"]["status"] for block in tool_results] == ["success", "error"]
    assert "outside the approved research scope" in tool_results[1]["toolResult"]["content"][0]["json"]["error"]
    assert result["status"] == "answer_partial" and result["answer"] == ANSWER
    assert [page["key"] for page in result["pages_written"]] == [OVERVIEW]
    assert [error["key"] for error in result["page_errors"]] == [CONCEPT]
    assert CONCEPT not in cloud.objects


def test_publish_question_false_writes_no_question_page_and_still_reports_ready():
    recorder = RecordingPublisher()
    cloud = MemoryS3({NOTE: source_note(), OVERVIEW: OVERVIEW_TEXT})

    result, _ = run(cloud, research_turns(), publisher=recorder, publish_question=False)

    assert result["status"] == "answer_ready", result
    assert result["answer"] == ANSWER
    assert result["question_key"] is None and result["question_sha256"] is None
    assert not any(key.startswith("wiki/questions/") for key in cloud.objects)
    assert not any(key.startswith("wiki/questions/") for key in cloud.reads)   # the version pre-read is skipped too
    assert [call["key"] for call in recorder.calls] == [OVERVIEW, CONCEPT, CONCEPT]
    assert {page["key"] for page in result["pages_written"]} == {OVERVIEW, CONCEPT}
    assert result["connections"] and all(entry["key"] != result["question_key"] for entry in result["connections"])
    trace = cloud.json(result["trace_key"])
    assert trace["result"]["status"] == "answer_ready" and trace["result"]["question_key"] is None


def test_publish_question_false_keeps_a_missing_answer_from_reading_as_ready():
    cloud = MemoryS3()
    result, _ = run(cloud, [RuntimeError("Model unavailable"), RuntimeError("Model unavailable")],
                    publish_question=False)

    assert result["status"] == "answer_failed" and result["answer"] == ""
    assert any("No final answer was produced" in problem for problem in result["problems"])
    assert not any(key.startswith("wiki/") for key in cloud.objects)


def test_publish_question_false_with_a_page_error_is_partial_not_ready():
    cloud = MemoryS3({OVERVIEW: OVERVIEW_TEXT})
    result, _ = run(cloud, [
        turn(("edit_page", {"key": OVERVIEW, "old_text": "Earlier claim.", "new_text": NEW_CLAIM})),  # not read first
        turn(answer=ANSWER),
    ], publish_question=False)

    assert result["status"] == "answer_partial" and result["answer"] == ANSWER
    assert [error["key"] for error in result["page_errors"]] == [OVERVIEW]
    assert cloud.text(OVERVIEW) == OVERVIEW_TEXT


def test_archive_receives_the_final_result_record_before_return():
    archived = []
    cloud = MemoryS3({NOTE: source_note(), OVERVIEW: OVERVIEW_TEXT})

    result, _ = run(cloud, research_turns(), publish_question=False, archive=archived.append)

    assert len(archived) == 1 and archived[0] is result
    assert archived[0]["status"] == "answer_ready" and archived[0]["answer"] == ANSWER
    assert archived[0]["trace_key"].startswith("runs/agents/")
    assert cloud.json(result["trace_key"])["result"]["answer"] == ANSWER


def test_archive_failure_is_recorded_as_a_problem_and_downgrades_ready_to_partial():
    def failing(record):
        raise OSError("receipt store unavailable")

    cloud = MemoryS3({NOTE: source_note(), OVERVIEW: OVERVIEW_TEXT})
    result, _ = run(cloud, research_turns(), publish_question=False, archive=failing)

    assert result["status"] == "answer_partial"
    assert any("Answer archive failed: receipt store unavailable" in problem for problem in result["problems"])
    assert result["answer"] == ANSWER and {page["key"] for page in result["pages_written"]} == {OVERVIEW, CONCEPT}


@pytest.mark.parametrize("title", ["RFWD2와 UBE3A의 차이는?", "What explains the difference?"])
def test_question_key_for_matches_the_engine_slug(title):
    cloud = MemoryS3()
    result, _ = run(cloud, [turn(answer="An answer.")], title=title)
    slug, key = question_key_for(title)
    assert result["slug"] == slug and result["question_key"] == key == f"wiki/questions/{slug}.md"
    assert key in cloud.objects


def test_new_parameters_are_keyword_only_with_defaults():
    import inspect

    parameters = inspect.signature(run_answer).parameters
    for name, default in (("publisher", None), ("publish_question", True), ("archive", None)):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default == default
    assert list(parameters)[:10] == ["event", "s3", "bucket", "model_client", "model_id", "reasoning", "converse",
                                     "search", "remaining_ms", "publisher"]
    tools = inspect.signature(WikiTools.__init__).parameters
    assert list(tools) == ["self", "s3", "bucket", "search", "reread", "check_remaining", "publisher"]
    assert tools["publisher"].default is None
