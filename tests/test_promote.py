from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from byeori.aws_store import AwsStore
from byeori.catalog import Catalog
from byeori.cli import build_parser, command_promote_draft
from byeori.config import Settings
from byeori.mcp_server import wiki_upload_key
from byeori.promote import parse_frontmatter, promote_draft_in_worker as promote_draft, split_page


BODY = "\n".join([
    "## Citation", "A, B, C et al. (2021). Title. Nature Genetics. DOI 10.1000/x.", "",
    "## Methods", "Exome sequencing of 1,000 trios. " * 20, "",
    "## Results", "Twelve genes reached exome-wide significance (Table 1). " * 20, "",
    "## Limitations", "Authors: small cohort. Reviewer note: no replication cohort described in the text.", "",
    "## Evidence boundary", "Numbers in Table 2 were not captured by the extraction and must be checked in the PDF.", "",
    "## Related pages", "- de novo variants in autism", "- exome sequencing cohorts", "",
])


def frontmatter(**overrides: Any) -> str:
    fields: dict[str, Any] = {
        "work_id": '"W1"', "doi": '"https://doi.org/10.1000/x"', "title": '"Title with \\"quotes\\""',
        "authors": '["A B", "C D"]', "publication_year": "2021", "journal": '"Nature Genetics"',
        "pdf_path": '"s3://bucket/papers/W1.pdf"', "pdf_sha256": '"pdfhash"', "source_format": "pdf",
        "text_extractor": "openalex-grobid", "source_key": '"sources/W1.md"', "grobid_sha256": '"xmlhash"',
        "ingest_harness": "aws-lambda-bedrock", "ingest_model": '"global.anthropic.claude-sonnet-5"',
        "ingest_input_tokens": "10", "ingest_output_tokens": "5", "drafted_at": '"2026-09-17T00:00:00+00:00"',
        "draft_status": "model_draft", "draft_problems": "[]", "review_status": "unreviewed",
    }
    fields.update(overrides)
    return "---\n" + "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n---\n"


def settings_for(root: Path, *, aws: bool = True) -> Settings:
    settings = Settings(root=root, data_dir=root / "data", state_dir=root / "state", openalex_api_key=None,
                        aws_region="us-east-1", aws_bucket="bucket" if aws else None,
                        aws_table="table" if aws else None, aws_ingest_function="function" if aws else None)
    settings.ensure_directories()
    return settings


class FakeStore:
    def __init__(self, item: dict[str, Any], s3_text: str | None = None) -> None:
        self.item = item
        self.s3_text = s3_text
        self.uploads: list[tuple[str, str, bool]] = []
        self.marked: list[tuple[str, dict[str, Any], str]] = []

    def get_item(self, work_id: str) -> dict[str, Any]:
        assert work_id == "W1"
        return dict(self.item)

    def get_text(self, key: str) -> str:
        assert key == "wiki/drafts/W1.md" and self.s3_text is not None
        return self.s3_text

    def put_text(self, key: str, text: str, *, create_only=False):
        if self.uploads:
            raise FileExistsError(key)
        self.uploads.append((key, text, create_only))
        return {"bucket": "bucket", "key": key}

    def mark_reviewed(self, work_id: str, fields: dict[str, Any], *, expected_draft_sha256: str) -> dict[str, Any]:
        self.marked.append((work_id, fields, expected_draft_sha256))
        return {"table": "table", "work_id": work_id, "fields": sorted(fields)}


def item_for(draft_sha256: str, **overrides: Any) -> dict[str, Any]:
    item = {"work_id": "W1", "ingest_status": "model_draft", "review_status": "unreviewed",
            "draft_sha256": draft_sha256, "draft_key": "wiki/drafts/W1.md",
            "pdf_sha256": "pdfhash", "grobid_sha256": "xmlhash"}
    item.update(overrides)
    return item


def test_split_and_parse_frontmatter_reads_lambda_format() -> None:
    lines, body = split_page(frontmatter() + BODY)
    fields = parse_frontmatter(lines)
    assert fields["title"] == 'Title with "quotes"' and fields["authors"] == ["A B", "C D"]
    assert fields["publication_year"] == 2021 and fields["draft_problems"] == []
    assert fields["source_format"] == "pdf" and fields["review_status"] == "unreviewed"
    assert body.startswith("## Citation")


def store_for(text=None, **overrides):
    text = text if text is not None else frontmatter() + BODY
    return FakeStore(item_for(hashlib.sha256(text.encode()).hexdigest(), **overrides), text)


def test_promote_publishes_directly_without_local_copy(tmp_path):
    settings = settings_for(tmp_path)
    store = store_for()
    result = promote_draft(settings, "W1", reviewer="Reviewer", method="person", note="Checked against PDF", store=store)
    key, text, create_only = store.uploads[0]
    fields = parse_frontmatter(split_page(text)[0])
    assert split_page(text)[1] == BODY
    assert key == "wiki/sources/W1.md" and create_only
    assert fields["review_status"] == "reviewed" and fields["reviewed_by"] == "Reviewer"
    assert fields["draft_edited_in_review"] is False
    assert store.marked[0][2] == store.item["draft_sha256"]
    assert result["reviewed_path"] == "s3://bucket/wiki/sources/W1.md"
    assert not settings.data_dir.exists()
    with pytest.raises(FileExistsError):
        promote_draft(settings, "W1", reviewer="Reviewer", store=store)


def test_promote_requires_explicit_reviewed_text_for_edits(tmp_path):
    settings = settings_for(tmp_path)
    store = store_for()
    text = frontmatter() + BODY.replace("Twelve genes", "Eleven genes")
    with pytest.raises(ValueError, match="pass --edited"):
        promote_draft(settings, "W1", reviewer="r", reviewed_text=text, store=store)
    with pytest.raises(ValueError, match="explicitly supplied"):
        promote_draft(settings, "W1", reviewer="r", edited=True, store=store)
    result = promote_draft(settings, "W1", reviewer="r", reviewed_text=text, edited=True, store=store)
    assert result["draft_edited_in_review"] is True
    assert store.marked[0][2] == store.item["draft_sha256"]
    assert not settings.data_dir.exists()


@pytest.mark.parametrize("text,changes,match", [
    (frontmatter(draft_status="draft_failed") + BODY, {}, "only model_draft"),
    (frontmatter() + BODY + "[TODO: check]", {}, "placeholder"),
    (None, {"ingest_status":"fulltext_ready"}, "ingest_status"),
    (None, {"review_status":"reviewed"}, "already"),
    (None, {"pdf_sha256":"other"}, "pdf_sha256"),
])
def test_promote_rejects_invalid_or_stale_evidence(tmp_path, text, changes, match):
    store = store_for(text, **changes)
    with pytest.raises(ValueError, match=match):
        promote_draft(settings_for(tmp_path), "W1", reviewer="r", store=store)
    assert not store.uploads and not store.marked


def test_promote_dry_run_reads_s3_but_writes_nothing(tmp_path):
    settings = settings_for(tmp_path)
    store = store_for()
    result = promote_draft(settings, "W1", reviewer="r", dry_run=True, store=store)
    assert result["dry_run"] and result["checks"][0].startswith("read draft from s3://")
    assert not settings.data_dir.exists() and not store.uploads and not store.marked


def test_promote_without_aws_fails_explicitly(tmp_path):
    with pytest.raises(RuntimeError, match="requires"):
        promote_draft(settings_for(tmp_path, aws=False), "W1", reviewer="r")


def test_cli_promote_resolves_aws_candidate_and_records_ledger(tmp_path, cloud_catalog, capsys):
    settings = settings_for(tmp_path)
    with Catalog(settings) as catalog:
        catalog.save_candidate({"work_id":"W1", "doi":"10.1000/x", "title":"Title"})
    text = frontmatter() + BODY
    cloud_catalog.items["W1"].update(item_for(hashlib.sha256(text.encode()).hexdigest()))
    cloud_catalog.objects["wiki/drafts/W1.md"] = text.encode()
    args = build_parser().parse_args(["promote-draft", "10.1000/x", "--reviewer", "Reviewer", "--method", "person"])
    assert command_promote_draft(settings, args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["work_id"] == "W1" and output["ledger"]["step"] == "promote"
    assert (settings.state_dir / "cost-ledger.jsonl").exists()
    assert not settings.data_dir.exists()


class FakeTable:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def update_item(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if self.fail:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException", "Message": "x"}}, "UpdateItem")


class FakeDynamo:
    def __init__(self, table: FakeTable) -> None:
        self.table = table

    def Table(self, name: str) -> FakeTable:  # noqa: N802 - boto3 naming
        assert name == "table"
        return self.table


class FakeSession:
    def __init__(self, table: FakeTable) -> None:
        self.table = table

    def resource(self, service: str) -> FakeDynamo:
        assert service == "dynamodb"
        return FakeDynamo(self.table)


def test_store_mark_reviewed_is_conditional_on_draft_hash(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    table = FakeTable()
    store = AwsStore(settings, session=FakeSession(table))  # type: ignore[arg-type]
    store.mark_reviewed("W1", {"review_status": "reviewed", "reviewed_by": "r"}, expected_draft_sha256="abc")
    call = table.calls[0]
    assert call["Key"] == {"work_id": "W1"}
    assert call["ConditionExpression"] == "draft_sha256 = :expected_draft AND ingest_status = :model_draft"
    assert call["ExpressionAttributeValues"][":expected_draft"] == "abc"
    assert set(call["ExpressionAttributeNames"].values()) == {"review_status", "reviewed_by"}
    failing = AwsStore(settings, session=FakeSession(FakeTable(fail=True)))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no longer holds"):
        failing.mark_reviewed("W1", {"review_status": "reviewed"}, expected_draft_sha256="abc")


def test_wiki_upload_key_keeps_extractions_safe(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, aws=False)
    assert wiki_upload_key("sources/W1.md") == "wiki/sources/W1.md"
    assert wiki_upload_key("wiki/papers/W1.md") == "wiki/papers/W1.md"
    with pytest.raises(ValueError):
        wiki_upload_key("candidates/W1.md")
    with pytest.raises(ValueError):
        wiki_upload_key("../pyproject.toml")


def test_upload_is_limited_to_the_shared_layer(tmp_path: Path) -> None:
    """logs/, agenda/, and state/ stay in this repository; only sources/ and wiki/ reach S3."""
    settings = settings_for(tmp_path, aws=False)
    for outside in ("../logs/2026-09-18-claude-macair.md", "../agenda/README.md", "../state/cost-ledger.jsonl",
                    "candidates/W1.json"):
        with pytest.raises(ValueError):
            wiki_upload_key(outside)
    assert wiki_upload_key("sources/W1.md") == "wiki/sources/W1.md"
