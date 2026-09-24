"""byeori.paper_upload: the per-PDF upload code the lab's script already ran, now behind
the installer's `upload-pdf` subcommand too (spec 2026-09-24, task 9b)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from byeori import paper_upload, papers
from byeori.config import Settings

STEM = "smith-2024-cortical-organoids"
PDF_KEY, META_KEY = f"papers/{STEM}/original.pdf", f"papers/{STEM}/meta.json"


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / f"{STEM}.pdf"
    path.write_bytes(b"%PDF-1.4 smith organoids")
    return path


@pytest.fixture
def cloud(cloud_catalog, monkeypatch):
    for attribute in ("s3", "table"):
        monkeypatch.delattr(papers._thread_local, attribute, raising=False)
    monkeypatch.setenv("AWS_KIRO_WIKI_BUCKET", "bucket")
    monkeypatch.setenv("AWS_KIRO_WIKI_TABLE", "table")
    cloud_catalog.metadata = {}

    def head_object(Bucket, Key):  # noqa: N803 - boto3's own spelling
        if Key not in cloud_catalog.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"Metadata": dict(cloud_catalog.metadata.get(Key, {}))}

    def upload_file(filename, bucket, key, ExtraArgs=None):  # noqa: N803
        cloud_catalog.objects[key] = Path(filename).read_bytes()
        cloud_catalog.metadata[key] = dict((ExtraArgs or {}).get("Metadata", {}))

    cloud_catalog.head_object = head_object
    cloud_catalog.upload_file = upload_file
    return cloud_catalog


def test_a_new_pdf_is_uploaded_with_meta_json_and_pdf_uploaded_status(cloud, pdf):
    result = paper_upload.upload_one(Settings.from_env(), pdf, source="upload-pdf")
    assert result["state"] == "uploaded"
    assert cloud.objects[PDF_KEY] == pdf.read_bytes()
    assert cloud.metadata[PDF_KEY]["sha256"] == result["pdf_sha256"]
    meta = json.loads(cloud.objects[META_KEY])
    assert meta["source"] == "upload-pdf" and meta["pdf_sha256"] == result["pdf_sha256"]
    item = cloud.items[STEM]
    assert item["ingest_status"] == "pdf_uploaded"
    assert item["pdf_sha256"] == result["pdf_sha256"]


def test_the_same_bytes_again_are_already_present_and_keep_existing_keys(cloud, pdf):
    first = paper_upload.upload_one(Settings.from_env(), pdf, source="upload-pdf")
    assert first["state"] == "uploaded"
    before_meta = cloud.objects[META_KEY]
    before_item = dict(cloud.items[STEM])

    second = paper_upload.upload_one(Settings.from_env(), pdf, source="upload-pdf")

    assert second["state"] == "already_present"
    assert cloud.objects[META_KEY] == before_meta
    assert cloud.items[STEM] == before_item


def test_different_bytes_under_the_same_stem_are_a_conflict_and_write_nothing(cloud, pdf, tmp_path):
    paper_upload.upload_one(Settings.from_env(), pdf, source="upload-pdf")
    before_meta = cloud.objects[META_KEY]
    before_item = dict(cloud.items[STEM])
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4 a completely different file")

    result = paper_upload.upload_one(Settings.from_env(), other, stem=STEM, source="upload-pdf")

    assert result["state"] == "conflict_existing_original"
    assert cloud.objects[META_KEY] == before_meta
    assert cloud.items[STEM] == before_item


def test_a_stem_that_is_not_a_lowercase_stem_raises_value_error(cloud, pdf):
    with pytest.raises(ValueError):
        paper_upload.upload_one(Settings.from_env(), pdf, stem="Not_A_Lowercase_Stem", source="upload-pdf")


def test_a_file_name_that_is_not_a_stem_needs_stem(cloud, tmp_path):
    odd = tmp_path / "My Paper.pdf"
    odd.write_bytes(b"%PDF-1.4 odd name")
    with pytest.raises(ValueError, match="--stem"):
        paper_upload.upload_one(Settings.from_env(), odd, source="upload-pdf")
    assert cloud.objects == {} or not any(key.startswith("papers/") for key in cloud.objects)
    result = paper_upload.upload_one(Settings.from_env(), odd, stem="doe-2024-odd-name", source="upload-pdf")
    assert result["state"] == "uploaded"


def test_meta_json_does_not_record_the_uploaders_local_path(cloud, pdf):
    paper_upload.upload_one(Settings.from_env(), pdf, source="upload-pdf")
    meta = json.loads(cloud.objects[META_KEY])
    assert "local_pdf" not in meta
    assert str(pdf.parent) not in json.dumps(meta)
