"""Exercise the AWS research loop through its real wiki reader and publisher."""
import copy
import itertools
import json
from pathlib import Path

import pytest

from byeori.question_agent import WikiTools, run_answer
from byeori.wiki_connections import PageConflictError
from test_wiki_connections import S3, source_note


CALL_IDS = itertools.count()


class AgentS3(S3):
    """Keep publication CAS checks while accepting independent run trace objects."""
    def put_object(self, Bucket, Key, Body, ContentType, **conditions):
        if Key.startswith("runs/"):
            self.objects[Key] = Body
            self.writes.append((Key, conditions))
            return {"ETag": self.etag(Key)}
        return super().put_object(Bucket, Key, Body, ContentType, **conditions)


def turn(*calls, answer=None, reasoning=False):
    content = [{"reasoningContent": {"reasoningText": {"text": "PRIVATE_REASONING", "signature": "test"}}}] if reasoning else []
    if answer is not None:
        content.append({"text": answer})
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


def run(cloud, responses, *, search=None, title="What explains the difference?", **event):
    model = Model(responses)
    result = run_answer({"title": title, **event}, s3=cloud, bucket="bucket", model_client=object(),
                        model_id="global.anthropic.claude-opus-5", reasoning="default",
                        converse=model.converse, search=search or (lambda event: {"results": []}))
    return result, model


def test_research_loop_searches_again_reads_edits_creates_and_connects_without_local_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    note = "wiki/sources/paper-one.md"
    overview = "wiki/overviews/existing.md"
    concept = "wiki/concepts/new-insight.md"
    original = "# Existing synthesis\n\nEarlier claim.\n\nUnrelated conclusion stays exactly as written.\n"
    cloud = AgentS3({note: source_note(), overview: original})
    searches = []

    def search(event):
        searches.append(event)
        return {"index_etag": "current-index", "results": [
            {"doc_type": "note", "doc_id": "paper-one", "title": "Original paper",
             "section": "Results", "score": 3, "path": "data/sources/paper-one.md"}]}

    new_claim = "The regional result narrows the earlier claim [[sources/paper-one]]."
    new_page = "A reusable distinction\n\n| Scale | Meaning |\n|---|---|\n| Region | Stable |\n\n[[sources/paper-one]] [[overviews/existing]]\n"
    answer = "영역과 단일 위치는 다른 측정 단위입니다. [[sources/paper-one]]"
    result, model = run(cloud, [
        turn(("search_wiki", {"query": "Which studies compare regional and site-level inheritance?"})),
        turn(("read_page", {"key": note}), ("read_page", {"key": overview})),
        turn(("search_wiki", {"query": "What experimental design explains the different units?"})),
        turn(("edit_page", {"key": overview, "old_text": "Earlier claim.", "new_text": new_claim}),
             ("write_page", {"key": concept, "markdown": new_page})),
        turn(answer=answer, reasoning=True),
    ], search=search, title="영역과 단일 위치의 결과가 왜 다른가")

    assert result["status"] == "answer_ready", result
    assert result["answer"] == answer and result["index_etag"] == "current-index"
    assert len(searches) == 2 and all(query["doc_type"] is None for query in searches)
    assert {page["key"] for page in result["pages_written"]} == {overview, concept}
    assert new_claim in cloud.objects[overview].decode()
    assert "Unrelated conclusion stays exactly as written." in cloud.objects[overview].decode()
    assert cloud.objects[concept].decode().startswith(new_page)
    assert "[[concepts/new-insight|" in cloud.objects[note].decode()
    assert "[[overviews/existing|" in cloud.objects[note].decode()
    assert "[[concepts/new-insight|" in cloud.objects["wiki/indexes/concepts.md"].decode()
    assert "[[indexes/concepts|Concepts]]" in cloud.objects["wiki/index.md"].decode()
    assert answer in cloud.objects[result["question_key"]].decode()
    trace = json.loads(cloud.objects[result["trace_key"]])
    assert trace["result"]["answer"] == answer
    assert [tool["name"] for step in trace["steps"] for tool in step["tools"]] == [
        "search_wiki", "read_page", "read_page", "search_wiki", "edit_page", "write_page"]
    assert "PRIVATE_REASONING" not in cloud.objects[result["trace_key"]].decode()
    assert model.requests[1]["messages"][-1]["content"][0]["toolResult"]["status"] == "success"
    assert not list(tmp_path.iterdir())


def test_original_continuation_enriches_source_note_for_future_questions():
    note = "wiki/sources/paper-one.md"
    original = "A" * 40000 + "The later result distinguishes regional stability from site-level changes."
    cloud = AgentS3({note: source_note(), "papers/paper-one/clean.md": original})
    old = "## 4. Key Results and Benchmarks\nOriginal scientific text."
    new = "## 4. Key Results and Benchmarks\nThe later result distinguishes regional stability from site-level changes."
    result, model = run(cloud, [
        turn(("read_page", {"key": note})),
        turn(("read_original", {"stem": "paper-one", "max_chars": 40000})),
        turn(("read_original", {"stem": "paper-one", "start": 40000, "max_chars": 40000})),
        turn(("edit_page", {"key": note, "old_text": old, "new_text": new})),
        turn(answer="The later original text resolves the distinction. [[sources/paper-one]]"),
    ])

    assert result["status"] == "answer_ready", result
    assert [read["start"] for read in result["reread"]] == [0, 40000]
    assert sum(read["used_chars"] for read in result["reread"]) == len(original)
    first_chunk = model.requests[2]["messages"][-1]["content"][0]["toolResult"]["content"][0]["json"]
    final_chunk = model.requests[3]["messages"][-1]["content"][0]["toolResult"]["content"][0]["json"]
    assert first_chunk["text"] == original[:40000] and first_chunk["next_start"] == 40000
    assert final_chunk["text"] == original[40000:] and final_chunk["has_more"] is False
    saved = cloud.objects[note].decode()
    assert new in saved and 'doi: "10.1/example"' in saved
    assert "## 5. Limitations and Future Work\nOriginal scientific text." in saved
    assert cloud.objects["papers/paper-one/clean.md"].decode() == original
    next_session = WikiTools(cloud, "bucket", lambda event: {"results": []}, "auto")
    assert new in next_session.call("read_page", {"key": note})["text"]


@pytest.mark.parametrize("stem", ["paper-one", "W123456789"])
def test_stored_original_supports_workshop_extraction_layout(stem):
    key = f"sources/{stem}.md"
    cloud = AgentS3({key: "Stored original extraction."})
    runtime = WikiTools(cloud, "bucket", lambda event: {"results": []}, "auto")
    result = runtime.call("read_original", {"stem": stem})

    assert result["key"] == key and result["text"] == "Stored original extraction."
    assert result["has_more"] is False and result["next_start"] is None


def test_original_rereading_can_be_disabled_without_fetching_the_original():
    cloud = AgentS3({"papers/paper-one/clean.md": "Stored original."})
    runtime = WikiTools(cloud, "bucket", lambda event: {"results": []}, "never")
    with pytest.raises(ValueError, match="disabled original rereading"):
        runtime.call("read_original", {"stem": "paper-one"})
    assert runtime.originals == [] and cloud.writes == []


def test_conflicting_scientific_edit_needs_fresh_read_and_preserves_other_authors_text():
    key = "wiki/overviews/existing.md"
    cloud = AgentS3({key: "# Topic\nEarlier claim.\nUntouched text.\n"})
    runtime = WikiTools(cloud, "bucket", lambda event: {"results": []}, "auto")
    runtime.call("read_page", {"key": key})
    cloud.objects[key] = b"# Topic\nEarlier claim.\nUntouched text.\nConcurrent scientific addition.\n"
    edit = {"key": key, "old_text": "Earlier claim.", "new_text": "Narrowed claim."}
    with pytest.raises(PageConflictError):
        runtime.call("edit_page", edit)
    with pytest.raises((PageConflictError, ValueError)):
        runtime.call("edit_page", edit)
    assert b"Earlier claim." in cloud.objects[key] and cloud.writes == []

    runtime.call("read_page", {"key": key})
    runtime.call("edit_page", edit)
    text = cloud.objects[key].decode()
    assert "Narrowed claim." in text and "Concurrent scientific addition." in text
    assert "Untouched text." in text and runtime.failures == {}
    with pytest.raises(ValueError, match="Read the page"):
        runtime.call("edit_page", {"key": key, "old_text": "Narrowed claim.", "new_text": "Another claim."})


def test_edit_requires_the_actual_passage_to_have_been_read():
    key = "wiki/overviews/existing.md"
    cloud = AgentS3({key: "Opening.\n" + "X" * 100 + "\nUnread conclusion."})
    runtime = WikiTools(cloud, "bucket", lambda event: {"results": []}, "auto")
    runtime.call("read_page", {"key": key, "max_chars": 8})
    with pytest.raises(ValueError, match="Read the passage"):
        runtime.call("edit_page", {"key": key, "old_text": "Unread conclusion.", "new_text": "Replacement."})
    assert cloud.writes == []


@pytest.mark.parametrize("markdown", [
    "A short reusable distinction.\n",
    "# Flexible heading\n\nAn opening.\n\n### Comparison\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
])
def test_new_synthesis_accepts_freeform_headings_and_length(markdown):
    cloud = AgentS3()
    runtime = WikiTools(cloud, "bucket", lambda event: {"results": []}, "auto")
    result = runtime.call("write_page", {"key": "wiki/concepts/insight.md", "markdown": markdown})
    assert result["errors"] == []
    assert cloud.objects[result["key"]].decode() == markdown


def test_failed_page_write_preserves_successful_page_and_returns_the_actual_answer():
    cloud = AgentS3()
    failed = "wiki/concepts/blocked.md"
    saved = "wiki/overviews/saved.md"
    cloud.denied.add(failed)
    answer = "A substantive answer, with the remaining save failure reported."
    result, _ = run(cloud, [
        turn(("write_page", {"key": saved, "markdown": "A reusable finding.\n"}),
             ("write_page", {"key": failed, "markdown": "Another finding.\n"})),
        turn(answer=answer),
    ])

    assert result["status"] == "answer_partial" and result["answer"] == answer
    assert [page["key"] for page in result["pages_written"]] == [saved]
    assert [error["key"] for error in result["page_errors"]] == [failed]
    assert saved in cloud.objects and failed not in cloud.objects
    assert answer in cloud.objects[result["question_key"]].decode()
    trace = json.loads(cloud.objects[result["trace_key"]])
    assert [tool["status"] for tool in trace["steps"][0]["tools"]] == ["success", "error"]


def test_model_failure_after_publication_retains_the_wiki_and_checkpoint():
    key = "wiki/concepts/saved.md"
    cloud = AgentS3()
    result, _ = run(cloud, [
        turn(("write_page", {"key": key, "markdown": "A reusable conclusion.\n"})),
        RuntimeError("Model unavailable"),
    ])

    assert result["status"] == "answer_partial" and result["question_key"] is None
    assert cloud.objects[key] == b"A reusable conclusion.\n"
    assert [page["key"] for page in result["pages_written"]] == [key]
    assert any("Model unavailable" in problem for problem in result["problems"])
    trace = json.loads(cloud.objects[result["trace_key"]])
    assert trace["steps"][0]["tools"][0]["status"] == "success"
    assert trace["result"]["status"] == "answer_partial"


def test_edit_validation_failure_remains_visible_after_a_final_answer():
    key = "wiki/overviews/existing.md"
    original = "# Topic\nOriginal scientific conclusion.\n"
    cloud = AgentS3({key: original})
    answer = "The available evidence is limited; the attempted page edit did not save."
    result, _ = run(cloud, [
        turn(("edit_page", {"key": key, "old_text": "Original scientific conclusion.",
                            "new_text": "Updated conclusion."})),
        turn(answer=answer),
    ])

    assert result["status"] == "answer_partial" and result["answer"] == answer
    assert result["pages_written"] == []
    assert any(error["key"] == key and "Read the page" in error["error"] for error in result["page_errors"])
    assert cloud.objects[key].decode() == original
    assert answer in cloud.objects[result["question_key"]].decode()


def test_korean_questions_with_same_english_tokens_keep_distinct_pages_and_titles():
    cloud = AgentS3()
    titles = ["RFWD2와 UBE3A의 차이는?", "RFWD2와 UBE3A의 공통점은?"]
    answers = ["두 유전자의 차이에 대한 답변.", "두 유전자의 공통점에 대한 답변."]
    results = [run(cloud, [turn(answer=answer)], title=title)[0] for title, answer in zip(titles, answers)]

    assert results[0]["question_key"] != results[1]["question_key"]
    for result, title, answer in zip(results, titles, answers):
        assert result["status"] == "answer_ready" and result["title"] == title
        page = cloud.objects[result["question_key"]].decode()
        assert f"title: {json.dumps(title, ensure_ascii=False)}" in page
        assert answer in page


def test_question_publication_conflict_retains_final_answer_and_trace():
    cloud = AgentS3()
    concurrent_answer = b"# Concurrent answer\nAnother completed answer must remain.\n"

    def concurrent(store, key):
        if key.startswith("wiki/questions/"):
            store.before_put = None
            store.objects[key] = concurrent_answer

    cloud.before_put = concurrent
    answer = "The final answer remains available even if its question page changed."
    result, _ = run(cloud, [turn(answer=answer)])
    key = f"wiki/questions/{result['slug']}.md"

    assert result["status"] == "answer_partial" and result["answer"] == answer
    assert result["question_key"] is None and result["question_sha256"] is None
    assert any(error["key"] == key for error in result["page_errors"])
    assert cloud.objects[key] == concurrent_answer
    trace = json.loads(cloud.objects[result["trace_key"]])
    assert trace["steps"][-1]["text"] == answer
    assert trace["result"]["answer"] == answer and trace["result"]["status"] == "answer_partial"


def test_repeated_expensive_inputs_leave_a_text_only_final_call_and_preserve_saved_pages():
    note = "wiki/sources/paper-one.md"
    concept = "wiki/concepts/regional-stability.md"
    cloud = AgentS3({note: source_note()})
    research = [
        turn(("read_page", {"key": note})),
        turn(("write_page", {"key": concept, "markdown": "Regional stability is a distinct unit. [[sources/paper-one]]\n"})),
        turn(("read_page", {"key": concept})),
    ]
    for response in research:
        response["usage"] = {"inputTokens": 60000, "cacheWriteInputTokens": 40000,
                             "cacheReadInputTokens": 20000, "outputTokens": 2000}
    answer = "Regional and single-site stability describe different units. [[sources/paper-one]]"
    result, model = run(cloud, [*research, turn(answer=answer, reasoning=True)], budget_usd=2.5)

    assert len(model.requests) == 4
    assert all("toolConfig" in request for request in model.requests[:-1])
    final = model.requests[-1]
    assert "toolConfig" not in final
    assert all(set(block) == {"text"} for message in final["messages"] for block in message["content"])
    assert final["inferenceConfig"]["maxTokens"] >= 1024
    context = json.loads(final["messages"][0]["content"][0]["text"])
    assert {read["result"]["key"] for read in context["evidence_already_read"]} == {note, concept}
    assert [page["key"] for page in context["saved_pages"]] == [concept]
    assert result["answer"] == answer and result["question_key"]
    assert result["completion_limited"] is True
    assert result["status"] == "answer_partial"
    assert result["estimated_usd"] < result["budget_usd"]
    assert result["page_errors"] == [] and result["connection_errors"] == []
    assert result["usage"]["cacheWriteInputTokens"] == 120000
    assert [page["key"] for page in result["pages_written"]] == [concept]
    saved_question = cloud.objects[result["question_key"]].decode()
    assert answer in saved_question and "[[concepts/regional-stability]]" in saved_question
    assert "[[questions/" in cloud.objects[concept].decode()
    trace_text = cloud.objects[result["trace_key"]].decode()
    assert json.loads(trace_text)["steps"][-1]["phase"] == "final_answer"
    assert "PRIVATE_REASONING" not in trace_text


class AbruptTermination(BaseException):
    """Model a Lambda termination after the per-turn checkpoint was committed."""


def interrupted_checkpoint(cloud, *, title="Which result changes the existing explanation?", failed_edit=None):
    note = "wiki/sources/paper-one.md"
    concept = "wiki/concepts/already-saved.md"
    response = turn(
        ("search_wiki", {"query": "Which experiment distinguishes the two explanations?"}),
        ("read_page", {"key": note}),
        ("read_original", {"stem": "paper-one", "max_chars": 40000}),
        ("write_page", {"key": concept, "markdown": "An established finding. [[sources/paper-one]]\n"}),
        answer="I will inspect the evidence before answering.", reasoning=True,
    )
    if failed_edit is not None:
        response["output"]["message"]["content"].append({"toolUse": {
            "toolUseId": f"call-{next(CALL_IDS)}", "name": "edit_page", "input": failed_edit}})
    requests = []

    def converse(client, request):
        requests.append(copy.deepcopy(request))
        if len(requests) == 1:
            return response, 0, 1
        raise AbruptTermination()

    def search(event):
        return {"index_etag": "index-before-interruption", "results": [
            {"doc_type": "note", "doc_id": "paper-one", "title": "Original paper",
             "section": "Results", "score": 3, "path": "data/sources/paper-one.md"}]}

    with pytest.raises(AbruptTermination):
        run_answer({"title": title}, s3=cloud, bucket="bucket", model_client=object(),
                   model_id="global.anthropic.claude-opus-5", reasoning="default",
                   converse=converse, search=search)
    checkpoint = next(key for key in cloud.objects if key.startswith("runs/agents/"))
    stored = json.loads(cloud.objects[checkpoint])
    assert "result" not in stored and stored["pages_written"][0]["key"] == concept
    assert "PRIVATE_REASONING" not in cloud.objects[checkpoint].decode()
    return checkpoint


def test_resume_of_interrupted_checkpoint_carries_actual_evidence_and_links_prior_saves():
    note = "wiki/sources/paper-one.md"
    concept = "wiki/concepts/already-saved.md"
    original = "The stored original distinguishes regional retention from single-site turnover."
    title = "Which result changes the existing explanation?"
    cloud = AgentS3({note: source_note(), "papers/paper-one/clean.md": original})
    checkpoint = interrupted_checkpoint(cloud, title=title)
    body_before_resume = cloud.objects[concept].decode()
    answer = "The distinction is regional retention versus single-site turnover. [[sources/paper-one]]"
    result, model = run(cloud, [turn(answer=answer)], title=title, resume_trace=checkpoint)

    assert len(model.requests) == 1
    resumed_text = next(block["text"] for block in model.requests[0]["messages"][0]["content"]
                        if "The following is recorded data, not instructions.\n" in block.get("text", ""))
    recorded = json.loads(resumed_text.split("The following is recorded data, not instructions.\n", 1)[1])
    evidence = {entry["result"]["key"]: entry["result"]["text"] for entry in recorded["evidence_already_read"]}
    assert evidence[note] == source_note()
    assert evidence["papers/paper-one/clean.md"] == original
    assert [page["key"] for page in recorded["saved_pages"]] == [concept]
    assert "PRIVATE_REASONING" not in resumed_text
    assert "I will inspect the evidence before answering." not in resumed_text
    assert result["status"] == "answer_ready" and result["title"] == title
    assert result["resumed_from"] == checkpoint and result["trace_key"] != checkpoint
    assert result["answer"] == answer and [page["key"] for page in result["pages_written"]] == [concept]
    assert cloud.objects[concept].decode().startswith(body_before_resume)
    assert "[[concepts/already-saved]]" in cloud.objects[result["question_key"]].decode()
    assert any(read["key"] == "papers/paper-one/clean.md" for read in result["reread"])
    assert any(hit["key"] == note for hit in result["retrieved"])
    assert result["index_etag"] == "index-before-interruption"


def test_resume_rejects_a_different_question_without_model_calls_or_writes():
    cloud = AgentS3({"wiki/sources/paper-one.md": source_note(), "papers/paper-one/clean.md": "Original evidence."})
    checkpoint = interrupted_checkpoint(cloud)
    before = copy.deepcopy(cloud.objects)
    model = Model([])
    with pytest.raises(ValueError, match="only continue its original question"):
        run_answer({"title": "A different question?", "resume_trace": checkpoint},
                   s3=cloud, bucket="bucket", model_client=object(),
                   model_id="global.anthropic.claude-opus-5", reasoning="default",
                   converse=model.converse, search=lambda event: {"results": []})
    assert model.requests == [] and cloud.objects == before


def test_incomplete_edit_arguments_can_be_retried_without_a_stale_failure():
    key = "wiki/overviews/existing.md"
    old, new = "Earlier claim.", "A narrowed claim supported by the experiment."
    cloud = AgentS3({key: "# Existing synthesis\n\n" + old + "\n\nUnrelated text remains.\n"})
    incomplete = turn(("edit_page", {"key": key, "old_text": old}))
    incomplete["stopReason"] = "max_tokens"
    result, model = run(cloud, [
        turn(("read_page", {"key": key})),
        incomplete,
        turn(("edit_page", {"key": key, "old_text": old, "new_text": new})),
        turn(answer="The experiment narrows the earlier explanation. [[overviews/existing]]"),
    ])

    error = model.requests[2]["messages"][-1]["content"][0]["toolResult"]
    assert error["status"] == "error"
    assert "new_text" in error["content"][0]["json"]["error"]
    assert result["status"] == "answer_ready" and result["page_errors"] == []
    assert result["question_key"] and [page["key"] for page in result["pages_written"]] == [key]
    saved = cloud.objects[key].decode()
    assert new in saved and old not in saved and "Unrelated text remains." in saved
    trace = json.loads(cloud.objects[result["trace_key"]])
    assert trace["steps"][1]["tools"][0]["status"] == "error"
    assert trace["steps"][2]["tools"][0]["status"] == "success"


@pytest.mark.parametrize("repair", [False, True])
def test_resume_preserves_an_unresolved_checkpoint_edit_error_until_repaired(repair):
    concept = "wiki/concepts/already-saved.md"
    old, new = "An established finding.", "A clarified finding."
    cloud = AgentS3({"wiki/sources/paper-one.md": source_note(), "papers/paper-one/clean.md": "Original evidence."})
    title = "Which result changes the existing explanation?"
    checkpoint = interrupted_checkpoint(cloud, title=title, failed_edit={"key": concept, "old_text": old})
    responses = [
        turn(("read_page", {"key": concept})),
        turn(("edit_page", {"key": concept, "old_text": old, "new_text": new})),
    ] if repair else []
    result, _ = run(cloud, [*responses, turn(answer="The recorded evidence supports a distinction. [[sources/paper-one]]")],
                    title=title, resume_trace=checkpoint)

    assert result["question_key"] and result["answer"]
    if repair:
        assert result["status"] == "answer_ready" and result["page_errors"] == []
        assert new in cloud.objects[concept].decode()
    else:
        assert result["status"] == "answer_partial"
        assert [error["key"] for error in result["page_errors"]] == [concept]
        assert "new_text" in result["page_errors"][0]["error"]
        assert old in cloud.objects[concept].decode() and new not in cloud.objects[concept].decode()


def test_execution_time_expiry_retains_saved_work_and_a_resumable_checkpoint():
    cloud = AgentS3()
    concept = "wiki/concepts/saved-before-deadline.md"
    remaining = {"ms": 900000}
    requests = []

    def converse(client, request):
        requests.append(copy.deepcopy(request))
        remaining["ms"] = 24000
        return turn(("write_page", {"key": concept, "markdown": "An established reusable finding.\n"})), 0, 1

    result = run_answer({"title": "What does the evidence establish?"}, s3=cloud, bucket="bucket",
                        model_client=object(), model_id="global.anthropic.claude-opus-5", reasoning="default",
                        converse=converse, search=lambda event: {"results": []}, remaining_ms=lambda: remaining["ms"])

    assert len(requests) == 1
    assert result["status"] == "answer_partial" and result["completion_limited"] is True
    assert result["answer"] == "" and result["question_key"] is None
    assert [page["key"] for page in result["pages_written"]] == [concept]
    assert cloud.objects[concept] == b"An established reusable finding.\n"
    assert any("saved work is resumable" in problem for problem in result["problems"])
    assert json.loads(cloud.objects[result["trace_key"]])["result"]["pages_written"] == result["pages_written"]


def test_the_deployed_budget_sets_what_a_question_may_spend_when_the_caller_names_none(monkeypatch):
    """The campaign never names a budget, so raising it has to be a deployment setting: all twelve
    questions of the 2026-09-23 student-synthesis run stopped at the old default of 5."""
    import importlib

    from byeori import question_agent

    monkeypatch.setenv("QUESTION_BUDGET_USD", "12")
    reloaded = importlib.reload(question_agent)
    try:
        assert reloaded.DEFAULT_BUDGET_USD == 12.0
    finally:
        monkeypatch.delenv("QUESTION_BUDGET_USD")
        importlib.reload(question_agent)


def test_the_deployment_carries_the_budget_to_the_question_lambda():
    from pathlib import Path

    template = (Path(__file__).parents[1] / "infra/template.yaml").read_text()
    deploy = (Path(__file__).parents[1] / "scripts/deploy.sh").read_text()
    assert "QUESTION_BUDGET_USD: !Ref QuestionBudgetUsd" in template
    assert "QuestionBudgetUsd:" in template and "MaxValue: 20" in template
    # The value the stack is deployed with, so a reader of deploy.sh sees what a question may spend.
    assert "QuestionBudgetUsd=" in deploy


def test_the_question_page_is_written_to_a_fixed_shape(cloud_catalog, tmp_path):
    """llm-wiki's 465 question pages hold a median of 4,079 characters with no length rule; five
    fixed sections do it. Byeori had no shape and its campaign answers averaged 13,756."""
    from byeori import question_agent

    sections = [line for line in question_agent.ANSWER_SHAPE.splitlines() if line.startswith("## ")]
    assert sections == ["## Question", "## Sharper follow-up", "## What the knowledge base holds",
                        "## Tentative answer from the knowledge base", "## Related Pages"]
    # Both ways into the final answer carry it: the tool loop's own last turn, and the separate
    # text-only call the budget forces. A shape only one of them sees is the shape of half the runs.
    source = (Path(__file__).parents[1] / "src/byeori/question_agent.py").read_text()
    assert source.count("ANSWER_SHAPE}") + source.count("ANSWER_SHAPE})") >= 2


def test_the_result_says_whether_the_time_went_on_research_or_on_writing_the_answer(tmp_path, monkeypatch):
    """A run that takes eight minutes is a different problem depending on which half it was."""
    monkeypatch.chdir(tmp_path)
    note = "wiki/sources/paper-one.md"
    cloud = AgentS3({note: source_note()})
    responses = [turn(("read_page", {"key": note})),
                 turn(answer="Regional stability differs. [[sources/paper-one]]")]
    result, _model = run(cloud, responses)
    assert result["research_seconds"] <= result["seconds"]
    # The loop answered in its own last turn, so nothing was spent on a separate answer call.
    assert result["answer_seconds"] == 0.0


def test_the_agent_can_keep_a_search_inside_one_field(tmp_path, monkeypatch):
    """Reading a field's whole catalog cost more than it saved: liver.md is 145,559 characters, and
    the run that was told to read it first connected 13 orphan notes for $3.40 against 21 for $2.57
    without it (2026-09-23). A category-filtered search asks the index the same question for free."""
    monkeypatch.chdir(tmp_path)
    note = "wiki/sources/paper-one.md"
    cloud = AgentS3({note: source_note()})
    asked = []

    def search(event):
        asked.append(event)
        return {"index_etag": "current-index", "results": [
            {"doc_type": "note", "doc_id": "paper-one", "title": "Original paper",
             "section": "Results", "score": 1.0}]}

    responses = [turn(("search_wiki", {"query": "zonation", "category": "liver"})),
                 turn(answer="Zonation differs. [[sources/paper-one]]")]
    run(cloud, responses, search=search)
    assert asked[0]["category"] == "liver"
    # The tool has to advertise it, or the model never sends it.
    from byeori.question_agent import TOOLS
    spec = next(t["toolSpec"] for t in TOOLS if t["toolSpec"]["name"] == "search_wiki")
    assert "category" in spec["inputSchema"]["json"]["properties"]
    assert "category" not in spec["inputSchema"]["json"]["required"]
