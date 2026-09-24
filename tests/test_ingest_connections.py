"""Exercise Lambda publication and timeout adapters without constructing AWS clients."""
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import botocore.config
import botocore.exceptions
import pytest

from byeori import question_agent
from byeori.wiki_connections import BACKLINK_START, PageConflictError, publish_page
from test_wiki_connections import S3, source_note


MODEL = "global.anthropic.claude-opus-5"


class ReadTrackingS3(S3):
    def __init__(self, objects=None):
        super().__init__(objects)
        self.reads = []

    def get_object(self, Bucket, Key):
        self.reads.append(Key)
        return super().get_object(Bucket, Key)


def load_lambda(cloud=None):
    source = Path(__file__).parents[1] / "src/byeori/ingest_lambda.py"
    tree = ast.parse(source.read_text())
    # Every helper the selected functions call has to be named here, or it is simply absent from
    # the namespace and the call fails with NameError.
    functions = {"_read_published", "_source_note", "_synthesize", "_answer_question",
                 "_document_information", "_llm_wiki_frontmatter", "_model_family", "_yaml_scalar",
                 "_source_collection", "_strip_reporting_summary", "_model_slug", "_note_models"}
    constants = {"SOURCE_NOTE_SECTIONS", "SOURCE_NOTE_SYSTEM", "TOPIC_SECTIONS", "TOPIC_SYSTEM",
                 "MAX_TOPIC_NOTES", "WORK_ID_PATTERN", "INGEST_AGENT", "INGEST_AGENT_VERSION",
                 "WRITE_NOW", "REPORTING_SUMMARY_MARKERS", "REPORTING_SUMMARY_MIN_OFFSET",
                 "TRIAL_RUN_PATTERN"}
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in functions
                or isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in constants]
    namespace = {"re": re, "json": json, "hashlib": hashlib, "datetime": datetime,
                 "timezone": timezone, "Decimal": Decimal, "botocore": botocore}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    updates = []
    namespace.update(s3=cloud, table=SimpleNamespace(update_item=lambda **request: updates.append(request)),
                     BUCKET_NAME="bucket", DRAFT_MAX_INPUT_CHARS=200000, INGEST_HARNESS="aws-bedrock",
                     INGEST_REASONING="default", _model_id=lambda event: MODEL,
                     NOTE_MODEL_ID=MODEL, NOTE_FALLBACK_MODEL_ID="", _table_updates=updates)
    return namespace


def generation(text):
    return {"text": text, "problems": [], "usage": {"inputTokens": 100, "outputTokens": 50},
            "generated_at": "2026-09-20T00:00:00+00:00", "seconds": 1.0, "stop_reason": "end_turn"}


def generated_source_body():
    return ("## One-line Summary\nOriginal scientific summary.\n\n"
            "## 2. Key Contributions\nOriginal verified contribution.\n\n"
            "## 3. Methodology and Architecture\nOriginal experimental design.\n\n"
            "## 4. Key Results and Benchmarks\nNewly read result from the original.\n\n"
            "## 5. Limitations and Future Work\nOriginal limitation stays.\n\n"
            "## 6. Related Work\nOriginal scientific relationship.\n\n"
            "## 7. Glossary\nOriginal glossary stays.\n")


def source_environment(cloud):
    namespace = load_lambda(cloud)
    item = {"ingest_status": "fulltext_ready", "source_key": "papers/paper-one/clean.md",
            "text_extractor": "grobid", "text_extracted_date": "2026-09-19"}
    meta = {"title": "Original paper", "doi": "10.1/example", "authors": "A Researcher",
            "year": 2026, "category": "methylation", "pdf_sha256": "original-pdf-hash"}
    namespace["_stem_item"] = lambda stem: (item, meta)
    return namespace


def test_source_regeneration_reads_current_note_before_model_and_preserves_links_and_metadata():
    key = "wiki/sources/paper-one.md"
    cloud = ReadTrackingS3({key: source_note(), "papers/paper-one/clean.md": "Canonical extraction evidence."})
    publish_page(cloud, "bucket", "wiki/concepts/context.md", "# Existing context\n[[sources/paper-one]]")
    previous, etag = cloud.objects[key].decode(), cloud.etag(key)
    cloud.reads.clear()
    namespace = source_environment(cloud)

    def generate(system, prompt, sections, first, model_id):
        assert cloud.reads[:2] == [key, "papers/paper-one/clean.md"]
        assert previous in prompt and "Canonical extraction evidence." in prompt
        assert "preserve its verified evidence" in prompt and model_id == MODEL
        return generation(generated_source_body())

    namespace["_generate"] = generate
    result = namespace["_source_note"]({"stem": "paper-one"})
    saved = cloud.objects[key].decode()

    assert result["status"] == "source_ready" and result["publication"]["errors"] == []
    assert "Newly read result from the original." in saved and "Original limitation stays." in saved
    assert saved.count("[[concepts/context|Existing context]]") == 1 and saved.count(BACKLINK_START) == 1
    assert 'doi: "10.1/example"' in saved and "## 1. Document Information" in saved
    assert "[[sources/paper-one|Original paper]]" in cloud.objects["wiki/indexes/sources.md"].decode()
    assert [conditions for target, conditions in cloud.writes if target == key][-1] == {"IfMatch": etag}
    assert result["source_note_sha256"] == hashlib.sha256(cloud.objects[key]).hexdigest()
    update = namespace["_table_updates"][-1]["ExpressionAttributeValues"]
    assert update[":digest"] == result["source_note_sha256"] and update[":st"] == "source_ready"


def test_first_source_generation_creates_note_and_catalog_conditionally():
    key = "wiki/sources/paper-one.md"
    cloud = ReadTrackingS3({"papers/paper-one/clean.md": "Canonical extraction evidence."})
    namespace = source_environment(cloud)
    namespace["_generate"] = lambda *args: generation(generated_source_body())
    result = namespace["_source_note"]({"stem": "paper-one"})

    assert result["publication"]["replaced"] is False
    assert (key, {"IfNoneMatch": "*"}) in cloud.writes
    assert "[[sources/paper-one|Original paper]]" in cloud.objects["wiki/indexes/sources.md"].decode()
    assert "[[indexes/sources|Sources]]" in cloud.objects["wiki/index.md"].decode()


def test_source_regeneration_does_not_overwrite_a_scientific_edit_made_during_generation():
    key = "wiki/sources/paper-one.md"
    cloud = ReadTrackingS3({key: source_note(), "papers/paper-one/clean.md": "Canonical extraction."})
    namespace = source_environment(cloud)
    concurrent = source_note().replace("Original scientific text.", "Concurrent original-grounded correction.").encode()

    def generate(*args):
        cloud.objects[key] = concurrent
        return generation(generated_source_body())

    namespace["_generate"] = generate
    with pytest.raises(PageConflictError):
        namespace["_source_note"]({"stem": "paper-one"})

    assert cloud.objects[key] == concurrent and cloud.writes == []
    assert namespace["_table_updates"] == [], "A failed CAS must not advertise the rejected body as ready"


def test_a_model_trial_writes_only_under_runs_and_leaves_the_wiki_and_catalog_alone():
    key = "wiki/sources/paper-one.md"
    cloud = ReadTrackingS3({key: source_note(), "papers/paper-one/clean.md": "Canonical extraction evidence."})
    before = dict(cloud.objects)
    namespace = source_environment(cloud)

    def generate(system, prompt, sections, first, model_id):
        # A trial compares models on the same input, so the note already in the wiki stays out of it.
        assert "Existing source note" not in prompt and "Canonical extraction evidence." in prompt
        return generation(generated_source_body())

    namespace["_generate"] = generate
    result = namespace["_source_note"]({"stem": "paper-one", "trial_run": "opus-5-5-vs-5-20260923"})
    trial_key = "runs/model-trials/opus-5-5-vs-5-20260923/claude-opus-5/paper-one.md"

    assert result["key"] == trial_key and result["status"] == "source_ready"
    assert [target for target, _ in cloud.writes] == [trial_key]
    assert all(cloud.objects[k] == v for k, v in before.items())
    assert namespace["_table_updates"] == []
    assert "Newly read result from the original." in cloud.objects[trial_key].decode()
    assert result["usage"] == {"inputTokens": 100, "outputTokens": 50}


def filtered(output_tokens=6):
    return {"text": "", "problems": ["stop reason content_filtered"], "stop_reason": "content_filtered",
            "usage": {"inputTokens": 30000, "outputTokens": output_tokens},
            "generated_at": "2026-09-23T00:00:00+00:00", "seconds": 3.0}


def test_a_note_the_note_model_declines_is_written_by_the_fallback_and_the_decline_is_kept():
    key = "wiki/sources/paper-one.md"
    cloud = ReadTrackingS3({"papers/paper-one/clean.md": "Canonical extraction evidence."})
    namespace = source_environment(cloud)
    namespace.update(NOTE_MODEL_ID="global.anthropic.claude-opus-5-5", NOTE_FALLBACK_MODEL_ID=MODEL)
    calls = []

    def generate(system, prompt, sections, first, model_id):
        calls.append(model_id)
        return filtered() if model_id.endswith("opus-5-5") else generation(generated_source_body())

    namespace["_generate"] = generate
    result = namespace["_source_note"]({"stem": "paper-one"})
    saved = cloud.objects[key].decode()

    assert calls == ["global.anthropic.claude-opus-5-5", MODEL]
    assert result["status"] == "source_ready" and result["model_id"] == MODEL
    assert f'ingest_model_id: "{MODEL}"' in saved and 'ingest_model_version: "5"' in saved
    assert result["filtered_attempt"] == {"model_id": "global.anthropic.claude-opus-5-5", "stop_reason": "content_filtered",
                                          "usage": {"inputTokens": 30000, "outputTokens": 6}, "seconds": 3.0}
    update = namespace["_table_updates"][-1]
    values = update["ExpressionAttributeValues"]
    assert values[":model"] == MODEL and values[":fmodel"] == "global.anthropic.claude-opus-5-5"
    assert values[":finp"] == 30000 and values[":fout"] == 6
    assert "REMOVE" not in update["UpdateExpression"]


def test_the_note_model_writes_alone_when_it_is_not_declined_and_old_decline_fields_go():
    cloud = ReadTrackingS3({"papers/paper-one/clean.md": "Canonical extraction evidence."})
    namespace = source_environment(cloud)
    namespace.update(NOTE_MODEL_ID="global.anthropic.claude-opus-5-5", NOTE_FALLBACK_MODEL_ID=MODEL)
    calls = []
    namespace["_generate"] = lambda *args: calls.append(args[4]) or generation(generated_source_body())
    result = namespace["_source_note"]({"stem": "paper-one"})

    assert calls == ["global.anthropic.claude-opus-5-5"] and result["model_id"] == "global.anthropic.claude-opus-5-5"
    assert result["filtered_attempt"] is None
    assert "REMOVE source_note_filtered_model" in namespace["_table_updates"][-1]["UpdateExpression"]


def test_a_note_read_from_ocr_text_is_told_so_and_asked_to_flag_what_looks_misread():
    cloud = ReadTrackingS3({"papers/paper-one/clean.md": "TY now appears clear that schizophrenia ..."})
    namespace = source_environment(cloud)
    item, meta = namespace["_stem_item"]("paper-one")
    item.update(text_preprocess="ocr", text_extractor="grobid-0.8.2+ocrmypdf")
    prompts = []
    namespace["_generate"] = lambda system, prompt, *rest: prompts.append(prompt) or generation(generated_source_body())
    namespace["_source_note"]({"stem": "paper-one"})
    assert "Extracted full text (GROBID, from OCR of a scanned PDF;" in prompts[0]
    assert "flag" in prompts[0] and "do not correct" in prompts[0].lower()


def test_a_born_digital_extraction_keeps_the_plain_label():
    cloud = ReadTrackingS3({"papers/paper-one/clean.md": "Canonical extraction evidence."})
    namespace = source_environment(cloud)
    prompts = []
    namespace["_generate"] = lambda system, prompt, *rest: prompts.append(prompt) or generation(generated_source_body())
    namespace["_source_note"]({"stem": "paper-one"})
    assert "Extracted full text (GROBID):" in prompts[0] and "OCR" not in prompts[0]


class FailedNoteS3(ReadTrackingS3):
    """A failed note is written to wiki/sources/failed/ without a condition, as it always has been."""

    def put_object(self, Bucket, Key, Body, ContentType, **conditions):
        if Key.startswith("wiki/sources/failed/") and not conditions:
            self.objects[Key] = Body
            self.writes.append((Key, conditions))
            return {"ETag": self.etag(Key)}
        return super().put_object(Bucket, Key, Body, ContentType, **conditions)


def test_a_model_the_caller_names_or_a_trial_is_never_swapped_for_the_fallback():
    for event in ({"stem": "paper-one", "model_id": "global.anthropic.claude-opus-5-5"},
                  {"stem": "paper-one", "trial_run": "refusal-check"}):
        cloud = FailedNoteS3({"papers/paper-one/clean.md": "Canonical extraction evidence."})
        namespace = source_environment(cloud)
        namespace.update(NOTE_MODEL_ID="global.anthropic.claude-opus-5-5", NOTE_FALLBACK_MODEL_ID=MODEL,
                         _model_id=lambda event: event["model_id"])
        calls = []
        namespace["_generate"] = lambda *args: calls.append(args[4]) or filtered()
        result = namespace["_source_note"](event)
        assert calls == ["global.anthropic.claude-opus-5-5"] and result["status"] == "source_failed"


def test_a_model_trial_name_cannot_steer_the_write_out_of_its_folder():
    cloud = ReadTrackingS3({"papers/paper-one/clean.md": "Canonical extraction evidence."})
    namespace = source_environment(cloud)
    namespace["_generate"] = lambda *args: generation(generated_source_body())
    for bad in ("../wiki", "Opus", "a/b", ""):
        with pytest.raises(ValueError):
            namespace["_source_note"]({"stem": "paper-one", "trial_run": bad})
    assert cloud.writes == []


def synthesis_environment(cloud):
    namespace = load_lambda(cloud)
    namespace["_reviewed_note"] = lambda work_id: (
        {"reviewed_sha256": "reviewed-hash"}, cloud.objects[f"wiki/sources/{work_id}.md"].decode(),
        f"wiki/sources/{work_id}.md")
    return namespace


def test_existing_synthesis_is_read_before_generation_and_published_with_reciprocal_links():
    key = "wiki/overviews/comparison.md"
    original = '---\ntitle: "Comparison"\ncreated: "2026-01-01"\n---\n# Comparison\nExisting argument.\nUnrelated material stays.\n'
    cloud = ReadTrackingS3({key: original, "wiki/sources/W1.md": source_note()})
    publish_page(cloud, "bucket", "wiki/concepts/context.md", "# Existing context\n[[overviews/comparison]]")
    previous, etag = cloud.objects[key].decode(), cloud.etag(key)
    namespace = synthesis_environment(cloud)

    def generate(system, prompt, sections, first, model_id):
        assert previous in prompt and "Existing overview: preserve unrelated material" in prompt
        return generation("## Scope\nA comparison.\n\n## Synthesis\nRevised argument [[sources/W1]].\nUnrelated material stays.\n")

    namespace["_generate"] = generate
    result = namespace["_synthesize"]({"topic": "comparison", "title": "Comparison", "work_ids": ["W1"]})
    saved = cloud.objects[key].decode()

    assert result["status"] == "model_topic" and result["publication"]["errors"] == []
    assert "Revised argument [[sources/W1]]." in saved and "Unrelated material stays." in saved
    assert 'created: "2026-01-01"' in saved
    assert saved.count("[[concepts/context|Existing context]]") == 1
    assert "[[overviews/comparison|Comparison]]" in cloud.objects["wiki/sources/W1.md"].decode()
    assert "[[overviews/comparison|Comparison]]" in cloud.objects["wiki/indexes/overviews.md"].decode()
    assert [conditions for target, conditions in cloud.writes if target == key][-1] == {"IfMatch": etag}
    assert result["topic_sha256"] == hashlib.sha256(cloud.objects[key]).hexdigest()


def test_synthesis_conflict_keeps_the_newer_scientific_body():
    key = "wiki/overviews/comparison.md"
    cloud = ReadTrackingS3({key: "# Comparison\nExisting science.\n", "wiki/sources/W1.md": source_note()})
    namespace = synthesis_environment(cloud)
    concurrent = b"# Comparison\nAnother researcher's correction.\n"

    def generate(*args):
        cloud.objects[key] = concurrent
        return generation("## Scope\nA comparison.\n## Synthesis\nMy new argument [[sources/W1]].\n")

    namespace["_generate"] = generate
    with pytest.raises(PageConflictError):
        namespace["_synthesize"]({"topic": "comparison", "title": "Comparison", "work_ids": ["W1"]})
    assert cloud.objects[key] == concurrent and cloud.writes == []


def test_question_sdk_timeout_is_recomputed_for_every_model_call(monkeypatch):
    namespace = load_lambda()
    clients, backoffs = [], []
    latest = {"milliseconds": 900000}
    context = SimpleNamespace(get_remaining_time_in_millis=lambda: latest["milliseconds"])

    def client(service, *, config):
        bounded = SimpleNamespace(service=service, config=config)
        clients.append(bounded)
        return bounded

    def backoff(bounded, request, *, budget):
        backoffs.append((bounded, request, budget))
        return {"response": request["turn"]}, 0, 1

    expected = {"answer": "An unchanged worker answer", "pages_written": [], "connections": []}

    def run(event, **kwargs):
        assert kwargs["remaining_ms"] is context.get_remaining_time_in_millis
        assert kwargs["s3"].service == "s3"
        for index, milliseconds in enumerate([900000, 285000, 270000, 260000, 30000]):
            latest["milliseconds"] = milliseconds
            response = kwargs["converse"](kwargs["model_client"], {"turn": index, "toolConfig": {"tools": []}})
            assert response == ({"response": index}, 0, 1)
        return expected

    monkeypatch.setattr(question_agent, "run_answer", run)
    namespace.update(boto3=SimpleNamespace(client=client), bedrock=object(), _wiki_search=object(),
                     _converse_with_backoff=backoff)
    result = namespace["_answer_question"]({"title": "Which evidence settles this?"}, context)

    assert result is expected
    model_clients = [bounded for bounded in clients if bounded.service == "bedrock-runtime"]
    assert [bounded.config.read_timeout for bounded in model_clients] == [240, 240, 240, 240, 10]
    assert all(bounded.config.connect_timeout == 5 for bounded in model_clients)
    assert all(bounded.config.retries == {"total_max_attempts": 1} for bounded in model_clients)
    assert [budget for _, _, budget in backoffs] == [30, 25, 10, 0, 0]
    assert [bounded for bounded, _, _ in backoffs] == model_clients
    assert clients[0].service == "s3" and clients[0].config.read_timeout == 5
    assert clients[0].config.connect_timeout == 3 and clients[0].config.retries == {"total_max_attempts": 2}


def test_question_model_timeout_is_bounded_without_a_lambda_context(monkeypatch):
    namespace = load_lambda()
    clients, budgets = [], []

    def client(service, *, config):
        result = SimpleNamespace(service=service, config=config)
        clients.append(result)
        return result

    def run(event, **kwargs):
        assert kwargs["remaining_ms"] is None
        kwargs["converse"](kwargs["model_client"], {"turn": 0, "toolConfig": {"tools": []}})
        return {"answer": "Available evidence."}

    monkeypatch.setattr(question_agent, "run_answer", run)
    namespace.update(boto3=SimpleNamespace(client=client), bedrock=object(), _wiki_search=object(),
                     _converse_with_backoff=lambda client, request, *, budget: budgets.append(budget))
    namespace["_answer_question"]({"title": "Which evidence settles this?"})

    assert clients[-1].service == "bedrock-runtime" and clients[-1].config.read_timeout == 240
    assert clients[-1].config.retries == {"total_max_attempts": 1} and budgets == [30]


@pytest.mark.parametrize("remaining_ms, expected_timeout", [(600000, 560), (90000, 50), (30000, 5), (None, 860)])
def test_final_answer_wait_reserves_publication_time_without_retries(monkeypatch, remaining_ms, expected_timeout):
    namespace = load_lambda()
    clients, budgets = [], []
    context = (SimpleNamespace(get_remaining_time_in_millis=lambda: remaining_ms)
               if remaining_ms is not None else None)
    request = {"messages": [{"role": "user", "content": [{"text": "Recorded research evidence."}]}]}
    expected = {"answer": "The researcher's final answer.", "question_key": "wiki/questions/result.md"}

    def client(service, *, config):
        result = SimpleNamespace(service=service, config=config)
        clients.append(result)
        return result

    def backoff(client, actual_request, *, budget):
        assert actual_request is request
        budgets.append(budget)
        if remaining_ms == 600000:
            # A final response arriving after five minutes used to be cut off at 240s.
            assert client.config.read_timeout >= 300
            assert remaining_ms / 1000 - client.config.read_timeout - client.config.connect_timeout >= 35
        return expected, 0, 1

    def run(event, **kwargs):
        result, _, attempts = kwargs["converse"](kwargs["model_client"], request)
        assert attempts == 1
        return result

    monkeypatch.setattr(question_agent, "run_answer", run)
    namespace.update(boto3=SimpleNamespace(client=client), bedrock=object(), _wiki_search=object(),
                     _converse_with_backoff=backoff)
    assert namespace["_answer_question"]({"title": "Which evidence settles this?"}, context) is expected
    model_clients = [client for client in clients if client.service == "bedrock-runtime"]
    assert len(model_clients) == 1
    assert model_clients[0].config.read_timeout == expected_timeout
    assert model_clients[0].config.connect_timeout == 5
    assert model_clients[0].config.retries == {"total_max_attempts": 1}
    assert budgets == [0]
