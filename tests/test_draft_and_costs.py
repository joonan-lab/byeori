from __future__ import annotations

import ast
import json
from pathlib import Path

from byeori import costs


TEMPLATE = Path(__file__).parents[1] / "infra" / "template.yaml"
LAMBDA_SOURCE = TEMPLATE.parents[1] / "src/byeori/ingest_lambda.py"


def _lambda_names(*names: str):
    tree = ast.parse(LAMBDA_SOURCE.read_text())
    wanted = [
        node for node in tree.body
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names)
        or (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, (ast.Import, ast.ImportFrom)) and not any("boto3" in a.name for a in node.names))
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "index.py", "exec"), namespace)
    return namespace


GOOD_NOTE = "\n".join([
    "## Citation", "A, B, C et al. (2021). Title. Nature Genetics. DOI 10.1000/x.", "",
    "## Methods", "Exome sequencing of 1,000 trios. " * 20, "",
    "## Results", "Twelve genes reached exome-wide significance (Table 1). " * 20, "",
    "## Limitations", "Authors: small cohort. Reviewer note: no replication cohort described in the text.", "",
    "## Evidence boundary", "Numbers in Table 2 were not captured by the extraction and must be checked in the PDF.", "",
    "## Related pages", "- de novo variants in autism", "- exome sequencing cohorts",
])


def test_lambda_draft_validator_accepts_complete_note_and_rejects_defects():
    ns = _lambda_names("DRAFT_SECTIONS", "DRAFT_FORBIDDEN", "_validate_sections", "_validate_draft")
    validate = ns["_validate_draft"]
    assert validate(GOOD_NOTE) == []
    assert any("missing section ## Results" in p for p in validate(GOOD_NOTE.replace("## Results", "## Findings")))
    assert any("placeholder" in p for p in validate(GOOD_NOTE + "\n[TODO: add table]"))
    assert any("start with" in p for p in validate("Here is the note.\n" + GOOD_NOTE))
    assert any("frontmatter" in p for p in validate("---\ntitle: x\n---\n" + GOOD_NOTE))
    assert any("too short" in p for p in validate("\n".join(s for s in ns["DRAFT_SECTIONS"])))


def test_a_note_about_ai_is_not_mistaken_for_the_model_speaking_of_itself():
    """Two notes on AI papers failed on 2026-09-23 because the check matched the bare words."""
    from byeori import promote, synthesis_pages
    ns = _lambda_names("DRAFT_SECTIONS", "DRAFT_FORBIDDEN", "_validate_sections", "_validate_draft")
    about_ai = ("Participants judged whether the opponent was an AI or a human. The authors cite a "
                "discussion of GPT-4 as an AI chatbot for medicine.")
    assert ns["_validate_draft"](GOOD_NOTE.replace("Authors: small cohort.", about_ai)) == []
    for speaking in ("As an AI, I cannot read the table.", "As an AI language model I lack the figure."):
        assert any("placeholder" in p for p in ns["_validate_draft"](GOOD_NOTE + "\n" + speaking))
        assert promote.PLACEHOLDER.search(speaking) and synthesis_pages.FORBIDDEN.search(speaking)
    assert not promote.PLACEHOLDER.search(about_ai) and not synthesis_pages.FORBIDDEN.search(about_ai)


def test_lambda_draft_frontmatter_is_code_written_with_provenance():
    ns = _lambda_names("_yaml_scalar", "_draft_frontmatter")
    ns["BUCKET_NAME"] = "bucket"  # the module reads it from the environment; inject it for the test
    work = {"doi": "https://doi.org/10.1000/x", "display_name": 'Title with "quotes"', "publication_year": 2021,
            "primary_location": {"source": {"display_name": "Nature Genetics"}},
            "authorships": [{"author": {"display_name": "A B"}}]}
    item = {"pdf_key": "papers/W1.pdf", "pdf_sha256": "abc", "source_key": "sources/W1.md", "grobid_sha256": "def"}
    text = ns["_draft_frontmatter"]("W1", work, item, "global.anthropic.claude-sonnet-5",
                                    {"inputTokens": 10, "outputTokens": 5}, "2026-09-17T00:00:00+00:00",
                                    "model_draft", [])
    assert text.startswith("---\n") and text.rstrip().endswith("---")
    assert 'title: "Title with \\"quotes\\""' in text
    assert "pdf_sha256: \"abc\"" in text and "ingest_model: \"global.anthropic.claude-sonnet-5\"" in text
    assert "ingest_harness: aws-lambda-bedrock" in text and "review_status: unreviewed" in text


def test_lambda_draft_requires_ingested_source_and_anthropic_model():
    source = LAMBDA_SOURCE.read_text()
    assert 'if item.get("ingest_status") not in {"fulltext_ready", "model_draft", "draft_failed"}' in source
    assert '"anthropic" not in model_id' in source
    assert all(action in TEMPLATE.read_text() for action in ("bedrock:InvokeModel", "s3:GetObject", "dynamodb:GetItem"))


def test_cost_estimate_and_ledger(tmp_path):
    usage = {"inputTokens": 12_000, "outputTokens": 3_000, "cacheReadInputTokens": 2_000, "cacheWriteInputTokens": 0}
    sonnet = costs.estimate_draft_usd("global.anthropic.claude-sonnet-5", usage)
    assert sonnet == round((12_000 * 2.0 + 2_000 * 0.2 + 3_000 * 10.0) / 1_000_000, 5)
    assert costs.estimate_draft_usd("global.anthropic.claude-opus-5", usage) > sonnet
    assert costs.estimate_draft_usd("some.other.model", usage) is None
    costs.record(tmp_path, {"step": "ingest", "work_id": "W1", "seconds": 4.0, "estimated_usd": 0.02})
    costs.record(tmp_path, {"step": "draft", "work_id": "W1", "seconds": 30.5, "estimated_usd": sonnet,
                            "input_tokens": 12_000, "output_tokens": 3_000})
    summary = costs.summarize(tmp_path)
    assert summary["entries"] == 2
    assert summary["by_step"]["draft"]["input_tokens"] == 12_000
    assert summary["estimated_usd"] == round(0.02 + sonnet, 4)
    assert summary["seconds"] == 34.5
    lines = (tmp_path / "cost-ledger.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["recorded_at"]


def test_converse_cost_charges_uncached_cache_read_and_cache_write_independently():
    usage = {"inputTokens": 10_000, "outputTokens": 2_000,
             "cacheReadInputTokens": 60_000, "cacheWriteInputTokens": 4_000}
    # 0.05 uncached + 0.05 output + 0.03 read + 0.025 write.
    assert costs.estimate_draft_usd("global.anthropic.claude-opus-5", usage) == 0.155
    assert costs.estimate_draft_usd("global.anthropic.claude-opus-5", {"inputTokens": 10_000}) == 0.05


def test_opus_5_5_is_priced_as_itself_not_as_the_opus_5_its_id_contains():
    usage = {"inputTokens": 10_000, "outputTokens": 2_000,
             "cacheReadInputTokens": 60_000, "cacheWriteInputTokens": 4_000}
    # 0.04 uncached + 0.04 output + 0.012 read + 0.02 write.
    assert costs.estimate_draft_usd("global.anthropic.claude-opus-5-5", usage) == 0.112
    assert costs.price_table("global.anthropic.claude-opus-5")["input"] == 5.0


def test_lambda_download_decodes_gzip_by_magic_bytes_or_header(tmp_path):
    import gzip
    ns = _lambda_names("_finalize_download")
    raw = tmp_path / "a.xml.raw"
    raw.write_bytes(gzip.compress(b"<TEI>ok</TEI>"))
    digest, size = ns["_finalize_download"](raw, tmp_path / "a.xml", "")
    assert (tmp_path / "a.xml").read_bytes() == b"<TEI>ok</TEI>" and size == 13
    raw2 = tmp_path / "b.xml.raw"
    raw2.write_bytes(b"<TEI>plain</TEI>")
    digest2, size2 = ns["_finalize_download"](raw2, tmp_path / "b.xml", "identity")
    assert (tmp_path / "b.xml").read_bytes() == b"<TEI>plain</TEI>" and not raw2.exists()
    assert digest != digest2


def test_lambda_topic_validator_and_routing():
    ns = _lambda_names("DRAFT_FORBIDDEN", "TOPIC_SECTIONS", "_validate_sections")
    wrong = "\n".join(["## Summary", "Text. " * 200, "", "## Key findings", "- finding " * 60])
    assert any("start with ## Scope" in p for p in ns["_validate_sections"](wrong, ns["TOPIC_SECTIONS"], "## Scope"))
    source = LAMBDA_SOURCE.read_text()
    assert 'if action == "synthesize":' in source
    assert 'if item.get("review_status") != "reviewed" or not item.get("reviewed_key"):' in source
    assert "does not match the hash recorded at promotion" in source


def test_lambda_llm_wiki_frontmatter_and_document_information():
    ns = _lambda_names("_model_family", "_llm_wiki_frontmatter", "_document_information", "_source_collection", "INGEST_HARNESS",
                       "INGEST_AGENT", "INGEST_AGENT_VERSION", "INGEST_REASONING", "EFFORT_LEVELS")
    ns["BUCKET_NAME"] = "bucket"
    assert ns["_model_family"]("global.anthropic.claude-opus-5") == ("opus", "5")
    assert ns["_model_family"]("global.anthropic.claude-haiku-4-5-20251001-v1:0") == ("haiku", "4.5")
    meta = {"title": "T", "authors": "A B, C D", "year": "2023", "doi": "10.1/x", "category": "asd-ndd",
            "journal": "Genome Medicine", "pmid": "123", "pdf_sha256": "abc", "source_collection": "publisher-pdf"}
    item = {"text_extractor": "grobid-0.8.2", "text_extracted_date": "2026-09-17"}
    fm = ns["_llm_wiki_frontmatter"]("abdi-2023-x", item, meta, "global.anthropic.claude-opus-5", [("source", "abdi-2023-x.md")])
    for line in ('title: "T"', 'category: "asd-ndd"', 'ingest_model: "opus"', 'ingest_model_version: "5"',
                 'ingest_agent: "byeori-ingest"', 'text_extractor: "grobid-0.8.2"', 'pdf_sha256: "abc"', 'pmid: "123"',
                 'source: "abdi-2023-x.md"', 'pdf_filename: "abdi-2023-x.pdf"', 'ingest_harness: "aws-bedrock"',
                 'ingest_model_id: "global.anthropic.claude-opus-5"', 'ingest_reasoning: "default"'):
        assert line in fm, line
    table = ns["_document_information"](meta)
    assert table.startswith("## 1. Document Information") and "| DOI | 10.1/x |" in table and "| PMID | 123 |" in table
    source = LAMBDA_SOURCE.read_text()
    assert 'if action == "source_note":' in source
    assert "SOURCE_NOTE_SECTIONS" in source and '"## 7. Glossary"' in source


def test_lambda_generate_enables_thinking_only_when_a_tier_is_set():
    source = LAMBDA_SOURCE.read_text()
    assert 'EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")' in source
    assert '{"thinking": {"type": "adaptive"}, "output_config": {"effort": INGEST_REASONING}}' in source
    assert 'IngestReasoning:' in TEMPLATE.read_text() and 'INGEST_REASONING: !Ref IngestReasoning' in TEMPLATE.read_text()


def test_lambda_bedrock_client_waits_and_never_retries():
    source = LAMBDA_SOURCE.read_text()
    assert 'boto3.client("bedrock-runtime", config=botocore.config.Config(read_timeout=880, connect_timeout=10, retries={"max_attempts": 1}))' in source
    assert "      Timeout: 900\n" in TEMPLATE.read_text()


def test_lambda_answer_question_action_goes_to_the_research_agent():
    source = LAMBDA_SOURCE.read_text()
    assert 'if action == "answer_question":\n        return _answer_question(event, _context)' in source
    assert "return run_answer(event, s3=agent_s3" in source


def test_lambda_validator_counts_headings_at_line_start_only():
    """Papers write "##" inline as a significance marker; that must not count as a section."""
    ns = _lambda_names("DRAFT_SECTIONS", "DRAFT_FORBIDDEN", "_validate_sections", "_validate_draft")
    note = GOOD_NOTE.replace("Twelve genes reached exome-wide significance (Table 1). ",
                             "Twelve genes reached significance (Fig 4C; # P < 0.001 and ## P < 0.0001). ")
    assert ns["_validate_draft"](note) == [], "inline ## must not be counted as a heading"
    # The real failure seen on 2026-09-17: the model glued the first heading to a preamble sentence.
    glued = "Table 1 numbers not captured.## Citation" + GOOD_NOTE.split("## Citation", 1)[1]
    assert any("must start with" in p for p in ns["_validate_draft"](glued))


def test_lambda_builds_the_index_from_s3_without_a_local_copy():
    source = LAMBDA_SOURCE.read_text()
    assert 'if action == "build_index":' in source and "def _build_wiki_index(event):" in source
    assert "s3:ListBucket" in TEMPLATE.read_text(), "listing the bucket is needed to find the pages"
    ns = _lambda_names("_index_documents", "_split_sections")
    page = ('---\ntitle: "T"\nyear: "2023"\njournal: "Nature"\ndoi: "10.1/x"\n---\n\n# T\n\n'
            "## Summary\nbody text\n\n## Results\n- a finding\n")
    (meta, sections, _links), = ns["_index_documents"]([("wiki/asd-ndd/abdi-2023-x.md", page)])
    assert meta[0] == "paper" and meta[1] == "abdi-2023-x" and meta[2] == "T" and meta[8] == "asd-ndd"
    assert meta[4] == "2023" and meta[5] == "Nature" and meta[6] == "10.1/x"
    assert meta[9] == "wiki/asd-ndd/abdi-2023-x.md", "the row carries the S3 key the reader fetches"
    assert [name for name, _ in sections] == ["Summary", "Results"], "the # title is not a section"
    (meta2, _, _links2), = ns["_index_documents"]([("wiki/sources/abdi-2023-x.md", page)])
    assert meta2[0] == "note" and meta2[3] == "data/sources/abdi-2023-x.md"
    # Indexing and retrieval must split a page identically, or a hit loses the text behind it.
    assert ns["_split_sections"](page.split("---\n", 2)[2]) == [("Summary", "body text"), ("Results", "- a finding")]


def test_lambda_reads_originals_inside_aws_under_bounded_excerpts():
    from byeori.question_agent import WikiTools
    from test_question_agent import AgentS3
    original = "A" * 40000 + "Evidence after the first chunk."
    cloud = AgentS3({"papers/test-paper/clean.md": original})
    reader = WikiTools(cloud, "bucket", lambda event: {}, "auto")
    first = reader.call("read_original", {"stem": "test-paper"})
    second = reader.call("read_original", {"stem": "test-paper", "start": first["next_start"]})
    assert first["has_more"] and len(first["text"]) == 40000
    assert second["text"] == "Evidence after the first chunk." and not second["has_more"]
    assert reader.originals[1]["start"] == 40000


def test_thinking_headroom_covers_reasoning_before_the_answer_is_written():
    """Adaptive thinking and the answer share one budget; a flat allowance truncated both Lambdas."""
    ns = _lambda_names("THINKING_HEADROOM", "DRAFT_MAX_OUTPUT_TOKENS")
    headroom = ns["THINKING_HEADROOM"]
    assert set(headroom) == {"low", "medium", "high", "xhigh", "max"}, "one entry per effort level"
    assert sorted(headroom.values()) == list(headroom.values()) or True
    assert [headroom[k] for k in ("low", "medium", "high", "xhigh", "max")] == sorted(headroom.values()), \
        "more effort means more thinking, so more room"
    # The measured failure spent 30,000 tokens thinking and wrote nothing; xhigh must clear that.
    assert headroom["xhigh"] + ns["DRAFT_MAX_OUTPUT_TOKENS"] > 30_000
    # Opus 5 caps one response at 128,000 output tokens. Its 1M figure is the context window.
    for level, value in headroom.items():
        assert value + ns["DRAFT_MAX_OUTPUT_TOKENS"] <= 128_000, level
    source = LAMBDA_SOURCE.read_text()
    assert "THINKING_HEADROOM.get(INGEST_REASONING, 24_000)" in source
    assert "+ (24000 if thinking else 0)" not in source, "the flat allowance must be gone"


def test_content_filtered_question_is_recorded_rather_than_dropped():
    """Filtered questions are kept, not written off: the lab researches several of these areas.

    Four of the 464 benchmark questions stopped with content_filtered. Bedrock's own classifier
    does it - no Guardrail is configured - and it bills: the six stopped calls on 2026-09-20 cost
    $0.8911, input tokens in full plus the output produced before the stop. So each one is kept as
    a skipped answer with its trace and what it cost, and the run continues.
    """
    from test_question_agent import AgentS3, run, turn
    cloud = AgentS3()
    response = turn(answer="")
    response["stopReason"] = "content_filtered"
    result, _ = run(cloud, [response])
    assert result["status"] == "answer_skipped"
    assert result["skipped_reason"] == "content_filtered"
    assert result["question_key"] is None
    assert not any(key.startswith("wiki/") for key in cloud.objects)
    assert result["trace_key"] in cloud.objects
    assert result["usage"]["inputTokens"] > 0

    benchmark = (TEMPLATE.parents[1] / "src/byeori/benchmark.py").read_text()
    assert '"skipped": len(skipped)' in benchmark, "skips are counted apart from failures"
    assert '"failed": len(run["rows"]) - len(ok) - len(skipped)' in benchmark, \
        "a skip must not inflate the failure count"


def test_throttling_is_waited_out_but_a_timeout_is_never_retried():
    """A throttled call is rejected before inference, so a retry is free. A timeout is not.

    botocore's own retries stay off on both Lambdas because re-issuing a long Bedrock call would
    run the model, and bill it, a second time. Only pre-inference rejections are retried here.
    """
    import botocore.exceptions

    ns = _lambda_names("THROTTLE_CODES", "THROTTLE_MAX_WAIT_SECONDS", "_converse_with_backoff")
    ns["random"] = __import__("random")
    slept: list[float] = []
    ns["time"] = type("T", (), {"sleep": staticmethod(lambda s: slept.append(s))})
    ns["botocore"] = botocore

    def error(code):
        return botocore.exceptions.ClientError({"Error": {"Code": code}}, "Converse")

    class _Client:
        def __init__(self, failures, code="ThrottlingException"):
            self.left, self.code = failures, code

        def converse(self, **kwargs):
            if self.left:
                self.left -= 1
                raise error(self.code)
            return {"ok": True}

    response, waited, attempts = ns["_converse_with_backoff"](_Client(2), {})
    assert response == {"ok": True} and attempts == 3, "it retried twice and then succeeded"
    assert waited > 0 and len(slept) == 2, "it backed off between attempts"
    assert slept[1] > slept[0], "the wait grows"

    # A read timeout means the model may still be generating; retrying would bill it twice.
    class _TimedOut:
        def converse(self, **kwargs):
            raise botocore.exceptions.ReadTimeoutError(endpoint_url="https://bedrock")

    try:
        ns["_converse_with_backoff"](_TimedOut(), {})
    except botocore.exceptions.ReadTimeoutError:
        pass
    else:
        raise AssertionError("a timeout must propagate, not retry")

    # An unbounded throttle must give up rather than sit inside the Lambda's 900 s limit.
    slept.clear()
    try:
        ns["_converse_with_backoff"](_Client(999), {}, budget=20)
    except botocore.exceptions.ClientError:
        pass
    else:
        raise AssertionError("the wait budget must be enforced")
    assert sum(slept) <= ns["THROTTLE_MAX_WAIT_SECONDS"]

    for source in (LAMBDA_SOURCE.read_text(),
                   (TEMPLATE.parents[1] / "src/byeori/synthesis_lambda.py").read_text()):
        assert "_converse_with_backoff(" in source, "both Lambdas go through the backoff"
        assert 'retries={"max_attempts": 0}' in source or 'retries={"max_attempts": 1}' in source, \
            "botocore's blanket retry stays off"


def test_every_request_is_logged_in_aws_not_on_the_caller_s_machine():
    """Byeori is reached through MCP from whatever machine a lab member is at.

    A log kept on one of those machines records that person's share and nothing once 25 people are
    using it, so the request and what came back are written where every caller's traffic lands.
    """
    source = LAMBDA_SOURCE.read_text()
    assert "def _log_request(event, result, started, error=None):" in source
    assert 'key = f"runs/requests/{now:%Y-%m-%d}/' in source, "one object per call, grouped by day"
    assert '"caller": caller' in source and "get_caller_identity" in source, "who asked"
    assert '"request": ' in source, "the question itself, not a summary of it"
    assert 'except Exception as exc:  # noqa: BLE001\n        print(f"_log_request: {exc}")' in source, \
        "logging must never fail the call it is logging"
    # The handler wraps the real dispatch so a raised error is logged too.
    assert "def handler(event, _context):\n    started =" in source and "def _handle(event, _context):" in source
    assert "_log_request(event, None, started, error=exc)" in source


def test_the_four_core_rules_are_in_byeori_not_in_a_caller_s_prompt():
    """They say where an answer may come from and what to do when the wiki falls short.

    A caller can forget them; the Lambda that writes the page cannot, so they live in the system
    prompts Byeori sends, on both the answering and the page-writing sides.
    """
    from byeori.question_agent import SYSTEM as answering  # what answer_question sends
    assert "No web search" in answering, "1"
    assert "only sources of truth" in answering, "2"
    assert "full text" in answering and "does not have to read that text again" in answering, \
        "3 — Byeori reads the text extracted at ingest, and the wiki has to end up carrying it"
    assert "ask for the paper" in answering and "Do not improvise" in answering, "4"

    from byeori import synthesis_pages as sp
    for system in (sp.CONCEPT_SYSTEM, sp.SUBTOPIC_SYSTEM, sp.UPDATE_SYSTEM):
        assert sp.CORE_RULES in system, "a page is written under the same rules an answer is"


def test_the_note_names_the_intake_the_upload_recorded():
    """Frontmatter said `llm-wiki` for 446 papers that came from the lab's shared folder, because
    the fallback asserted a provenance nothing had stated."""
    ns = _lambda_names("_source_collection", "_model_family", "_llm_wiki_frontmatter", "INGEST_HARNESS",
                       "INGEST_AGENT", "INGEST_AGENT_VERSION", "INGEST_REASONING", "EFFORT_LEVELS")
    ns["BUCKET_NAME"] = "bucket"
    collection = ns["_source_collection"]
    # An explicit field wins over everything.
    assert collection({"source_collection": "publisher-pdf", "source": "to-s3"}) == "publisher-pdf"
    # Otherwise the intake the uploader recorded, which is what was missing.
    assert collection({"source": "to-s3"}) == "to-s3"
    # And only with neither does it fall back to what every upload used to be.
    assert collection({}) == "llm-wiki"

    frontmatter = ns["_llm_wiki_frontmatter"](
        "agarwal-2026-x", {}, {"source": "to-s3"}, "global.anthropic.claude-opus-5", [])
    assert 'source_collection: "to-s3"' in frontmatter
    assert 'source_collection: "llm-wiki"' not in frontmatter


# ---------------------------------------------------------------------------------------------
# A prompt that ends in the paper's own text reads as a document to continue (2026-09-22)
# ---------------------------------------------------------------------------------------------

def test_the_reporting_checklist_is_cut_from_the_extraction():
    ns = _lambda_names("REPORTING_SUMMARY_MARKERS", "REPORTING_SUMMARY_MIN_OFFSET", "_strip_reporting_summary")
    strip = ns["_strip_reporting_summary"]
    paper = "# A paper\n\n" + ("Results follow. " * 300)
    checklist = "\n\nnature portfolio reporting summary\n\nField-specific reporting\n\nLife sciences\n"
    assert strip(paper + checklist) == paper.rstrip()
    assert strip(paper) == paper          # nothing to cut leaves the text alone


def test_a_checklist_phrase_near_the_start_is_not_treated_as_the_appendix():
    """A paper that mentions reporting standards in its own opening keeps all of its text."""
    ns = _lambda_names("REPORTING_SUMMARY_MARKERS", "REPORTING_SUMMARY_MIN_OFFSET", "_strip_reporting_summary")
    strip = ns["_strip_reporting_summary"]
    early = "# A paper about the reporting summary as a research object\n\nBody text here.\n"
    assert strip(early) == early


def test_the_instruction_is_repeated_after_the_extracted_text():
    ns = _lambda_names("WRITE_NOW")
    closer = ns["WRITE_NOW"].format(first="## One-line Summary")
    assert "not a document to continue" in closer
    assert "## One-line Summary" in closer
    assert "reporting checklist" in closer


def test_notes_have_their_own_model_and_fallback_while_answers_keep_the_draft_model():
    template = TEMPLATE.read_text()
    deploy = (TEMPLATE.parents[1] / "scripts/deploy.sh").read_text()
    assert "NOTE_MODEL_ID: !Ref NoteModelId" in template and "NOTE_FALLBACK_MODEL_ID: !Ref NoteFallbackModelId" in template
    # Every model is back on Opus 5 since the user's decision of 2026-09-23; the switch stays one value.
    assert "NoteModelId=global.anthropic.claude-opus-5 " in deploy + " "
    assert "NoteFallbackModelId=global.anthropic.claude-opus-5 " in deploy + " "
    assert "DraftModelId=global.anthropic.claude-opus-5 " in deploy + " "
    source = LAMBDA_SOURCE.read_text()
    assert 'NOTE_MODEL_ID = os.environ.get("NOTE_MODEL_ID") or DRAFT_MODEL_ID' in source
